"""Evaluate one supplied PathSnap trip using the completed Member 3 pipeline.

This adapter expects Member 2 predictions and Member 1 candidate positions as
JSON. It does not train models or generate candidate positions.

Example:
    python scripts/evaluate_map_matching.py \
        --dataset-root "C:/data/Student folder" --split train \
        --trip-id 00000001 --predictions predictions.json \
        --candidate-positions candidate_positions.json --output-dir results

Candidate-position JSON format:
    {"0": {"2641": {"distance_from_start_m": 12.5}}}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decoder.evaluation import (
    aggregate_evaluations,
    evaluate_trip,
    export_aggregate_results_csv,
    export_trip_results_csv,
    load_reference_route,
)
from src.decoder.transition import build_road_network


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Dataset split, such as train or student_test")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-positions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trip_dir = args.dataset_root / args.split / args.trip_id
    nodes = _read_coordinates(trip_dir / f"{args.trip_id}.nodes")
    arcs = _read_integer_pairs(trip_dir / f"{args.trip_id}.arcs")
    network = build_road_network(nodes, arcs, trip_id=args.trip_id)
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    positions_payload = json.loads(args.candidate_positions.read_text(encoding="utf-8"))
    positions = _read_candidate_positions(positions_payload, predictions, network)
    observations = _read_track_metadata(trip_dir / f"{args.trip_id}.track", predictions)

    # Ground truth is loaded only after the prediction inputs and candidate
    # metadata have been prepared. It is passed only to evaluation functions.
    reference = load_reference_route(trip_dir / f"{args.trip_id}.route", network)
    evaluation = evaluate_trip(predictions, network, observations, positions, reference)
    aggregates = aggregate_evaluations((evaluation,))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    export_trip_results_csv((evaluation,), args.output_dir / "per_trip_results.csv")
    export_aggregate_results_csv(aggregates, args.output_dir / "aggregate_results.csv")
    print(f"Wrote evaluation results to {args.output_dir}")
    return 0


def _read_coordinates(path: Path):
    return [tuple(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_integer_pairs(path: Path):
    return [tuple(map(int, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_track_metadata(path: Path, predictions):
    track = [tuple(map(float, line.split())) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = {}
    for index, observation in enumerate(predictions["observations"]):
        if index >= len(track):
            raise ValueError(f"track file has no row for prediction observation {index}")
        longitude, latitude, timestamp = track[index]
        metadata[int(observation.get("observation_id", index))] = {
            "longitude": longitude,
            "latitude": latitude,
            "timestamp": timestamp,
        }
    return metadata


def _read_candidate_positions(payload, predictions, network):
    positions = {}
    for index, observation in enumerate(predictions["observations"]):
        observation_id = int(observation.get("observation_id", index))
        raw_positions = payload.get(str(observation_id), payload.get(observation_id, {}))
        positions[observation_id] = {}
        for candidate in observation["candidates"]:
            arc_id = int(candidate["arc_id"])
            raw = raw_positions.get(str(arc_id), raw_positions.get(arc_id))
            if not isinstance(raw, dict) or "distance_from_start_m" not in raw:
                raise ValueError(
                    f"missing distance_from_start_m for observation {observation_id}, arc {arc_id}"
                )
            positions[observation_id][arc_id] = network.candidate_position(
                arc_id, float(raw["distance_from_start_m"])
            )
    return positions


if __name__ == "__main__":
    raise SystemExit(main())
