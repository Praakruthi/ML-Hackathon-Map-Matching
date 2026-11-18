"""Run the supplied single-model predictions through Member 3's pipeline.

This runner consumes one prediction artifact, either the JSON file or the ZIP
containing ``pathsnap_hmm_input.json``. It does not load or retrain the model.
It can process one trip or every trip represented in the prediction artifact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile
from collections.abc import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decoder.hmm import HMMConfig, match_trip
from src.decoder.integration import load_prediction_payload, load_trip_inputs_from_zip
from src.decoder.route_reconstruction import reconstruct_route


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-zip", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", default="student_test")
    parser.add_argument("--trip-id", help="One trip; omit to process every predicted trip")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    trip_ids = [args.trip_id] if args.trip_id else _prediction_trip_ids(args.predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for trip_id in trip_ids:
        summaries.append(_run_trip(args.dataset_zip, args.predictions, args.split, trip_id, args.output_dir))
    (args.output_dir / "integration_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2))
    return 0 if all(item["status"] == "ok" for item in summaries) else 1


def _run_trip(dataset_zip, prediction_source, split, trip_id, output_dir):
    started = time.perf_counter()
    try:
        inputs = load_trip_inputs_from_zip(
            dataset_zip, prediction_source, split=split, trip_id=trip_id
        )
        decode_started = time.perf_counter()
        hmm_result = match_trip(
            inputs.predictions,
            inputs.road_network,
            inputs.observation_data,
            inputs.candidate_positions,
            HMMConfig(emission_mode="ml"),
        )
        hmm_seconds = time.perf_counter() - decode_started
        reconstruction_started = time.perf_counter()
        route = reconstruct_route(
            hmm_result, inputs.road_network, inputs.candidate_positions
        )
        reconstruction_seconds = time.perf_counter() - reconstruction_started
        summary = {
            "trip_id": trip_id,
            "gps_observations": inputs.observation_count,
            "candidate_records": inputs.candidate_count,
            "average_candidates_per_observation": inputs.candidate_count / inputs.observation_count,
            "selected_classifier": inputs.selected_model or "not recorded in payload",
            "hmm_emission_mode": "ml",
            "viterbi_selected_arcs": list(hmm_result.arc_sequence),
            "reconstructed_route_arcs": list(route.route_arc_sequence),
            "reconstructed_route_distance_m": route.total_distance_m,
            "hmm_runtime_s": hmm_seconds,
            "reconstruction_runtime_s": reconstruction_seconds,
            "total_runtime_s": time.perf_counter() - started,
            "status": "ok",
        }
        (output_dir / f"{trip_id}_predicted_route.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"Trip ID: {trip_id} | GPS observations: {inputs.observation_count} | "
            f"candidates: {inputs.candidate_count} | model: {summary['selected_classifier']} | "
            f"route arcs: {route.reconstructed_arc_count} | status: ok"
        )
        return summary
    except Exception as error:
        summary = {
            "trip_id": trip_id,
            "status": "failed",
            "failure_reason": f"{type(error).__name__}: {error}",
            "total_runtime_s": time.perf_counter() - started,
        }
        print(f"Trip ID: {trip_id} | status: failed | reason: {summary['failure_reason']}")
        return summary


def _prediction_trip_ids(source: Path) -> list[str]:
    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            payload = json.loads(archive.read("pathsnap_hmm_input.json").decode("utf-8"))
    else:
        payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = [payload]
    return [str(item["trip_id"]) for item in payload]


if __name__ == "__main__":
    raise SystemExit(main())
