"""Tests for the backend API."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.config import settings
from backend.main import app, run_backend
from backend.services.trip_service import TripService
from backend.services.model_service import ModelService
from backend.services.performance_service import PerformanceService


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


def test_module_status_reports_pipeline_not_ready(api_client: TestClient) -> None:
    response = api_client.get("/modules/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["pipeline_ready"] is False
    assert {module["name"] for module in payload["modules"]} == {
        "Member 1 - Data Engineering",
        "Member 2 - Machine Learning",
        "Member 3 - HMM and Route Reconstruction",
    }
    assert all(not module["ready"] for module in payload["modules"])


def test_models_endpoint_lists_artifacts_without_loading_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"not deserialized")
    monkeypatch.setattr("backend.api.model_service", ModelService(tmp_path))

    response = TestClient(app).get("/models")

    assert response.status_code == 200
    assert response.json()["models"][0]["filename"] == "model.joblib"


def test_performance_endpoint_returns_empty_without_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "backend.api.performance_service",
        PerformanceService(tmp_path / "missing.json"),
    )

    response = TestClient(app).get("/models/performance")

    assert response.status_code == 200
    assert response.json()["status"] == "empty"
    assert response.json()["models"] == []


def test_prediction_validates_trip_before_reporting_pipeline_unavailable(
    api_client: TestClient,
) -> None:
    invalid_trip = api_client.post(
        "/predict/does-not-exist",
        json={"dataset_category": "train"},
    )
    unavailable = api_client.post(
        "/predict/training-trip-new",
        json={"dataset_category": "train"},
    )

    assert invalid_trip.status_code == 404
    assert invalid_trip.json()["error"]["code"] == "trip_not_found"
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "prediction_unavailable"
    assert unavailable.json()["error"]["context"]["modules"]


def test_prediction_rejects_invalid_dataset_category(api_client: TestClient) -> None:
    response = api_client.post(
        "/predict/training-trip-new",
        json={"dataset_category": "student"},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


def test_backend_launcher_uses_configured_bind_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_run(app_path: str, **kwargs: object) -> None:
        captured["app_path"] = app_path
        captured.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    run_backend()

    assert captured == {
        "app_path": "backend.main:app",
        "host": settings.backend_host,
        "port": settings.backend_port,
        "reload": True,
    }