"""Tests for the backend API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app
from backend.services.trip_service import TripService


def write_trip(
    dataset_root: Path,
    category: str,
    trip_id: str,
    *,
    arcs: str = "0 1\n1 0\n",
) -> Path:
    trip_path = dataset_root / category / trip_id
    trip_path.mkdir(parents=True)
    (trip_path / f"{trip_id}.track").write_text(
        "8.5 49.8 1.0\n8.6 49.9 2.0\n",
        encoding="utf-8",
    )
    (trip_path / f"{trip_id}.nodes").write_text(
        "8.4 49.7\n8.5 49.8\n",
        encoding="utf-8",
    )
    (trip_path / f"{trip_id}.arcs").write_text(arcs, encoding="utf-8")
    return trip_path


@pytest.fixture
def api_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    dataset_root = tmp_path / "dataset"
    write_trip(dataset_root, "train", "training-trip-new")
    write_trip(dataset_root, "student_test", "test-trip-new")
    monkeypatch.setattr(
        "backend.api.trip_service",
        TripService(dataset_root),
    )
    return TestClient(app)


def test_health_returns_application_status() -> None:
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "application": settings.app_name,
        "status": "ok",
        "api_version": settings.api_version,
    }


def test_get_trips_returns_both_dynamic_categories(api_client: TestClient) -> None:
    response = api_client.get("/trips")

    assert response.status_code == 200
    payload = response.json()
    assert [trip["trip_id"] for trip in payload["training_trips"]] == [
        "training-trip-new"
    ]
    assert payload["training_trips"][0]["dataset_category"] == "train"
    assert [trip["trip_id"] for trip in payload["test_trips"]] == ["test-trip-new"]
    assert payload["test_trips"][0]["files"] == {
        "track": True,
        "nodes": True,
        "arcs": True,
    }


def test_get_trip_returns_counts_and_coordinates(api_client: TestClient) -> None:
    response = api_client.get(
        "/trips/training-trip-new",
        params={"dataset_category": "train"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["gps_observation_count"] == 2
    assert payload["road_node_count"] == 2
    assert payload["road_arc_count"] == 2
    assert payload["gps_observations"][0] == {
        "coordinates": [8.5, 49.8],
        "third_column": 1.0,
    }
    assert payload["road_network"] is None
    assert "route" not in payload


def test_get_trip_includes_only_requested_network_sample(
    api_client: TestClient,
) -> None:
    response = api_client.get(
        "/trips/training-trip-new",
        params={
            "dataset_category": "train",
            "include_network": "true",
            "network_limit": 1,
        },
    )

    assert response.status_code == 200
    network = response.json()["road_network"]
    assert len(network["arcs"]) == 1
    assert {node["node_index"] for node in network["nodes"]} == {0, 1}


def test_get_trip_returns_not_found_for_unknown_id(api_client: TestClient) -> None:
    response = api_client.get(
        "/trips/not-present",
        params={"dataset_category": "train"},
    )

    assert response.status_code == 404


def test_get_trip_returns_unprocessable_for_malformed_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    trip_path = write_trip(dataset_root, "train", "bad-trip", arcs="0 2\n")
    (trip_path / "bad-trip.track").write_text("8.5 49.8\n", encoding="utf-8")
    monkeypatch.setattr("backend.api.trip_service", TripService(dataset_root))

    response = TestClient(app).get(
        "/trips/bad-trip",
        params={"dataset_category": "train"},
    )

    assert response.status_code == 422


def test_get_trip_returns_not_found_when_required_file_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    trip_path = write_trip(dataset_root, "train", "incomplete-trip")
    (trip_path / "incomplete-trip.arcs").unlink()
    monkeypatch.setattr("backend.api.trip_service", TripService(dataset_root))

    response = TestClient(app).get(
        "/trips/incomplete-trip",
        params={"dataset_category": "train"},
    )

    assert response.status_code == 404