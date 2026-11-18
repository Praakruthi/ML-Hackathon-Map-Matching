"""Expand one PathSnap prediction payload with true-distance candidates.

Existing candidate probabilities are retained byte-for-byte.  Only newly
added candidates are scored with the frozen model artifact; no route files are
read and no model is trained or selected here.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decoder.candidate_generation import TrueDistanceCandidateIndex
from src.decoder.integration import load_prediction_payload, load_trip_inputs_from_zip


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--model-zip", type=Path, required=True)
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--extra-candidates", type=int, default=5)
    parser.add_argument("--max-distance-m", type=float, default=250.0)
    parser.add_argument("--max-detour-factor", type=float, default=8.0,
                        help="Allow at most GPS displacement * factor + 100 m")
    parser.add_argument("--detour-slack-m", type=float, default=100.0)
    args = parser.parse_args()

    loaded = load_trip_inputs_from_zip(
        args.dataset_zip, args.predictions, split="student_test", trip_id=args.trip_id
    )
    model = _load_model(args.model_zip)
    index = TrueDistanceCandidateIndex(
        loaded.road_network.nodes, tuple(loaded.road_network.arcs.values())
    )
    track = loaded.observation_data
    payload = json.loads(json.dumps(loaded.predictions))
    previous_endpoints: set[int] = set()
    previous_positions = None
    previous_point = None
    added = []
    for observation in payload["observations"]:
        observation_id = int(observation.get("observation_id", 0))
        point = (track[observation_id]["longitude"], track[observation_id]["latitude"])
        existing = {int(candidate["arc_id"]) for candidate in observation["candidates"]}
        ranked = index.rank(
            point,
            pool_size=args.pool_size,
            max_distance_m=args.max_distance_m,
        )
        # Expansion is attempted only when the current lattice has no directed
        # continuation from the states that survived the preceding step.
        current_positions = {
            int(candidate["arc_id"]): loaded.road_network.candidate_position(
                int(candidate["arc_id"]),
                loaded.candidate_positions[observation_id][int(candidate["arc_id"])]
                .distance_from_start_m,
            )
            for candidate in observation["candidates"]
        }
        if previous_positions is None:
            feasible_current = set(current_positions)
        else:
            feasible_current = {
                arc_id for arc_id, position in current_positions.items()
                if any(
                    loaded.road_network.path_between_positions(start, position) is not None
                    for start in previous_positions.values()
                )
            }
        additions = []
        if previous_positions is not None and not feasible_current:
            assert previous_point is not None
            gps_distance = _gps_distance(previous_point, point)
            maximum_route = gps_distance * args.max_detour_factor + args.detour_slack_m
            for item in ranked:
                if item[0] in existing:
                    continue
                arc_id, distance, fraction = item
                candidate_position = loaded.road_network.candidate_position(
                    arc_id, loaded.road_network.arc(arc_id).length_m * fraction
                )
                if any(
                    (path := loaded.road_network.path_between_positions(start, candidate_position))
                    is not None and path.distance_m <= maximum_route
                    for start in previous_positions.values()
                ):
                    additions.append(item)
                if len(additions) >= args.extra_candidates:
                    break
        if additions:
            ranked_distances = {arc_id: distance for arc_id, distance, _ in ranked}
            all_distances = [ranked_distances.get(int(c["arc_id"]), args.max_distance_m)
                             for c in observation["candidates"]]
            all_distances.extend(distance for _, distance, _ in additions)
            minimum = min(all_distances)
            heading, speed = _motion(track, observation_id, len(track))
            current_endpoints = {
                endpoint
                for candidate in observation["candidates"]
                for endpoint in (
                    loaded.road_network.arc(int(candidate["arc_id"])).source,
                    loaded.road_network.arc(int(candidate["arc_id"])).target,
                )
            }
            current_endpoints.update(
                endpoint
                for arc_id, _, _ in additions
                for endpoint in (
                    loaded.road_network.arc(arc_id).source,
                    loaded.road_network.arc(arc_id).target,
                )
            )
            for arc_id, distance, fraction in additions:
                arc = loaded.road_network.arc(arc_id)
                feature_row = [[
                    distance,
                    _heading_difference(heading, loaded.road_network, arc_id),
                    speed,
                    distance - minimum,
                    int(bool(previous_endpoints) and bool(
                        {arc.source, arc.target} & previous_endpoints
                    )),
                ]]
                probability = _positive_probability(model, feature_row)
                observation["candidates"].append(
                    {"arc_id": arc_id, "ml_probability": probability}
                )
                added.append({"observation_id": observation_id, "arc_id": arc_id,
                              "distance_m": distance, "probability": probability})
            previous_endpoints = current_endpoints
        else:
            previous_endpoints = {
                endpoint
                for candidate in observation["candidates"]
                for endpoint in (
                    loaded.road_network.arc(int(candidate["arc_id"])).source,
                    loaded.road_network.arc(int(candidate["arc_id"])).target,
                )
            }

        for arc_id, _, fraction in additions:
            current_positions[arc_id] = loaded.road_network.candidate_position(
                arc_id, loaded.road_network.arc(arc_id).length_m * fraction
            )
        if additions:
            feasible_current.update(item[0] for item in additions)
        previous_positions = {
            arc_id: current_positions[arc_id] for arc_id in feasible_current
        }
        previous_point = point

    payload["candidate_generation"] = {
        "method": "true_segment_distance_expansion",
        "pool_size": args.pool_size,
        "extra_candidates": args.extra_candidates,
        "max_distance_m": args.max_distance_m,
        "added_candidates": added,
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"trip_id": args.trip_id, "added": len(added), "output": str(args.output)}, indent=2))
    return 0


def _load_model(model_zip: Path):
    with ZipFile(model_zip) as archive:
        artifact = joblib.load(BytesIO(archive.read("pathsnap_best_model.joblib")))
    return artifact["model"] if isinstance(artifact, dict) else artifact


def _positive_probability(model, row) -> float:
    probabilities = model.predict_proba(pd.DataFrame(row, columns=[
        "distance_m", "heading_diff_deg", "speed_mps",
        "relative_distance_m", "connectivity",
    ]))
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"frozen model has no positive class 1: {classes}")
    probability = float(probabilities[0][classes.index(1)])
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("model returned an invalid positive-class probability")
    return probability


def _motion(track, index: int, count: int) -> tuple[float, float]:
    left, right = max(0, index - 1), min(count - 1, index + 1)
    first, second = track[left], track[right]
    dx, dy = _metric_delta(first, second)
    displacement = math.hypot(dx, dy)
    dt = float(second["timestamp"] - first["timestamp"])
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360 if displacement else float("nan"), displacement / dt if dt > 0 else float("nan")


def _metric_delta(first, second):
    scale = 6_371_000.0 * math.cos(math.radians(first["latitude"]))
    return (scale * math.radians(second["longitude"] - first["longitude"]),
            6_371_000.0 * math.radians(second["latitude"] - first["latitude"]))


def _gps_distance(first, second):
    dx, dy = _metric_delta(
        {"longitude": first[0], "latitude": first[1]},
        {"longitude": second[0], "latitude": second[1]},
    )
    return math.hypot(dx, dy)


def _heading_difference(heading, network, arc_id):
    if not math.isfinite(heading):
        return float("nan")
    arc = network.arc(arc_id)
    start, end = network.nodes[arc.source], network.nodes[arc.target]
    dx, dy = _metric_delta({"longitude": start[0], "latitude": start[1]},
                           {"longitude": end[0], "latitude": end[1]})
    road_heading = (math.degrees(math.atan2(dx, dy)) + 360) % 360
    return abs((heading - road_heading + 180) % 360 - 180)


if __name__ == "__main__":
    raise SystemExit(main())
