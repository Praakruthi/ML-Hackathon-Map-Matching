"""Unit tests for trip discovery and source-file parsers."""

from pathlib import Path

import pytest

from backend.config import PROJECT_ROOT
from backend.services.trip_service import (
	InvalidTripDataError,
	MissingTripFilesError,
	TripNotFoundError,
	TripService,
	parse_arcs,
	parse_nodes,
	parse_track,
)


def write_trip(
	dataset_root: Path,
	category: str,
	trip_id: str,
	*,
	track: str = "8.5 49.8 1.0\n8.6 49.9 2.0\n",
	nodes: str = "8.4 49.7\n8.5 49.8\n",
	arcs: str = "0 1\n1 0\n",
) -> Path:
	trip_path = dataset_root / category / trip_id
	trip_path.mkdir(parents=True)
	(trip_path / f"{trip_id}.track").write_text(track, encoding="utf-8")
	(trip_path / f"{trip_id}.nodes").write_text(nodes, encoding="utf-8")
	(trip_path / f"{trip_id}.arcs").write_text(arcs, encoding="utf-8")
	return trip_path


def test_discovery_finds_dynamic_training_and_test_trips(tmp_path: Path) -> None:
	write_trip(tmp_path, "train", "train-zeta")
	write_trip(tmp_path, "train", "train-alpha")
	write_trip(tmp_path, "student_test", "test-new-id")
	(tmp_path / "train" / "incomplete").mkdir()

	service = TripService(tmp_path)
	training = service.discover_trips("train")
	test = service.discover_trips("student_test")

	assert [trip["trip_id"] for trip in training] == [
		"incomplete",
		"train-alpha",
		"train-zeta",
	]
	assert training[0]["available"] is False
	assert [trip["trip_id"] for trip in test] == ["test-new-id"]
	assert test[0]["dataset_category"] == "student_test"


def test_parsers_read_sample_whitespace_records(tmp_path: Path) -> None:
	track_path = tmp_path / "sample.track"
	nodes_path = tmp_path / "sample.nodes"
	arcs_path = tmp_path / "sample.arcs"
	track_path.write_text("8.595203 49.844053 1.000000\n", encoding="utf-8")
	nodes_path.write_text("8.581679 49.857880\n8.584637 49.858295\n", encoding="utf-8")
	arcs_path.write_text("0 1\n1 0\n", encoding="utf-8")

	track = parse_track(track_path)
	nodes = parse_nodes(nodes_path)
	arcs = parse_arcs(arcs_path, node_count=2)

	assert track.rows == [(8.595203, 49.844053, 1.0)]
	assert nodes.rows == [(8.581679, 49.85788), (8.584637, 49.858295)]
	assert arcs.rows == [(0, 1), (1, 0)]


def test_parsers_reject_wrong_columns_nonfinite_values_and_bad_indices(
	tmp_path: Path,
) -> None:
	track_path = tmp_path / "bad.track"
	nodes_path = tmp_path / "bad.nodes"
	arcs_path = tmp_path / "bad.arcs"
	track_path.write_text("8.5 49.8\n", encoding="utf-8")
	nodes_path.write_text("nan 49.8\n", encoding="utf-8")
	arcs_path.write_text("0 2\n", encoding="utf-8")

	with pytest.raises(InvalidTripDataError, match="3 columns"):
		parse_track(track_path)
	with pytest.raises(InvalidTripDataError, match="non-finite"):
		parse_nodes(nodes_path)
	with pytest.raises(InvalidTripDataError, match="valid range"):
		parse_arcs(arcs_path, node_count=2)


def test_empty_file_is_rejected(tmp_path: Path) -> None:
	empty_track = tmp_path / "empty.track"
	empty_track.write_text(" \n\t\n", encoding="utf-8")

	with pytest.raises(InvalidTripDataError, match="empty"):
		parse_track(empty_track)


def test_missing_trip_and_required_file_are_reported(tmp_path: Path) -> None:
	service = TripService(tmp_path)
	with pytest.raises(TripNotFoundError):
		service.load_trip("not-here", "train")

	trip_path = write_trip(tmp_path, "train", "incomplete")
	(trip_path / "incomplete.arcs").unlink()
	with pytest.raises(MissingTripFilesError, match="arcs"):
		service.load_trip("incomplete", "train")


def test_missing_dataset_category_lists_as_empty(tmp_path: Path) -> None:
	assert TripService(tmp_path).discover_trips("student_test") == []


def test_relative_dataset_path_is_resolved_from_project_root() -> None:
	assert TripService("data/dataset").dataset_root == PROJECT_ROOT / "data" / "dataset"