"""Tests for Member 4 adapters, artifact services, and orchestration."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.integrations.base import ModuleUnavailableError
from backend.integrations.member1_adapter import Member1Adapter
from backend.integrations.member2_adapter import Member2Adapter
from backend.integrations.member3_adapter import Member3Adapter
from backend.integrations.route_serializer import (
	RouteSerializationError,
	serialize_route,
)
from backend.services.model_service import ModelService
from backend.services.performance_service import (
	InvalidPerformanceArtifactError,
	PerformanceService,
)
from backend.services.prediction_service import (
	PredictionService,
	PredictionUnavailableError,
)


class FakeMember1:
	def load_trip(self, trip_id: str, dataset_category: str):
		return {
			"gps_points": [(10.0, 20.0), (11.0, 21.0)],
			"road_graph": {
				"nodes": [(10.0, 20.0), (11.0, 21.0)],
				"arcs": [(0, 1)],
			},
		}

	def generate_candidates(self, gps_points, road_graph):
		return {"candidates": [0]}

	def extract_features(self, candidates):
		return {"features": [[1.0]]}


class FakeMember2:
	def load_best_model(self, model_id=None):
		return {"model_id": model_id or "test-model"}

	def predict_candidates(self, model, features):
		return {"probabilities": [1.0]}


class FakeMember3:
	def reconstruct_route(self, gps_points, candidates, probabilities, road_graph):
		return {"arc_ids": [0], "distance_m": 100.0}

	def validate_route(self, predicted_arc_ids, road_graph):
		return {"valid": True, "connected": True}


def test_default_adapters_are_unavailable_with_clear_readiness() -> None:
	for adapter in (Member1Adapter(), Member2Adapter(), Member3Adapter()):
		assert adapter.readiness().ready is False
		with pytest.raises(ModuleUnavailableError):
			adapter.implementation()


def test_prediction_orchestration_uses_injected_adapter_contracts() -> None:
	service = PredictionService(
		Member1Adapter(FakeMember1()),
		Member2Adapter(FakeMember2()),
		Member3Adapter(FakeMember3()),
	)

	result = service.predict_trip("trip-1", "train", "test-model")

	assert result["status"] == "completed"
	assert result["route"]["arc_ids"] == [0]
	assert result["route"]["coordinates"] == [(10.0, 20.0), (11.0, 21.0)]
	assert result["route"]["valid"] is True
	assert result["route"]["connected"] is True
	assert result["processing_time_s"] >= 0


def test_prediction_service_does_not_fabricate_results_when_unavailable() -> None:
	with pytest.raises(PredictionUnavailableError) as error:
		PredictionService().predict_trip("trip-1", "train")
	assert len(error.value.modules) == 3


def test_route_serializer_checks_connectivity_and_node_references() -> None:
	serialized = serialize_route(
		[0, 1],
		[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
		[(0, 1), (1, 2)],
	)
	assert serialized["connected"] is True
	assert serialized["coordinates"] == [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
	with pytest.raises(RouteSerializationError):
		serialize_route([3], [(0.0, 0.0)], [(0, 1)])


def test_model_inventory_does_not_deserialize_artifacts(tmp_path: Path) -> None:
	(tmp_path / "candidate.joblib").write_bytes(b"not a model")
	(tmp_path / "README.md").write_text("no model", encoding="utf-8")
	result = ModelService(tmp_path).list_models()
	assert [model["filename"] for model in result["models"]] == ["candidate.joblib"]


def test_performance_artifact_returns_real_metrics(tmp_path: Path) -> None:
	artifact = tmp_path / "performance.json"
	artifact.write_text(
		json.dumps(
			{
				"dataset_category": "train",
				"models": [{"model_name": "baseline", "f1_score": 0.75}],
			}
		),
		encoding="utf-8",
	)
	response = PerformanceService(artifact).get_performance()
	assert response["status"] == "available"
	assert response["models"][0]["f1_score"] == 0.75


def test_student_test_metrics_are_blocked_without_explicit_permission(
	tmp_path: Path,
	monkeypatch: pytest.MonkeyPatch,
) -> None:
	artifact = tmp_path / "test-performance.json"
	artifact.write_text(
		json.dumps(
			{
				"dataset_category": "student_test",
				"student_test_evaluation_permitted": True,
				"models": [{"model_name": "candidate", "accuracy": 0.9}],
			}
		),
		encoding="utf-8",
	)
	monkeypatch.setattr(
		"backend.services.performance_service.settings",
		SimpleNamespace(allow_student_test_evaluation=False),
	)
	response = PerformanceService(artifact).get_performance()
	assert response["status"] == "blocked"
	assert response["models"] == []


def test_invalid_performance_artifact_is_rejected(tmp_path: Path) -> None:
	artifact = tmp_path / "invalid.json"
	artifact.write_text("{not valid json", encoding="utf-8")
	with pytest.raises(InvalidPerformanceArtifactError):
		PerformanceService(artifact).get_performance()