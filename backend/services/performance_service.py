"""Read evaluation metrics only from validated, provenance-bearing artifacts."""

import json
import logging

from pydantic import ValidationError

from backend.config import settings
from backend.schemas.performance import PerformanceArtifact

logger = logging.getLogger(__name__)


class InvalidPerformanceArtifactError(Exception):
	"""Raised when a saved performance artifact is malformed or untrusted."""


class PerformanceService:
	MAX_ARTIFACT_BYTES = 5_000_000

	def __init__(self, results_path=None) -> None:
		self.results_path = results_path or settings.performance_results_path

	def get_performance(self) -> dict[str, object]:
		if not self.results_path.is_file():
			return {
				"status": "empty",
				"dataset_category": None,
				"models": [],
				"message": "No evaluation results are available yet.",
			}
		try:
			if self.results_path.stat().st_size > self.MAX_ARTIFACT_BYTES:
				raise InvalidPerformanceArtifactError(
					"Evaluation artifact exceeds the maximum supported size."
				)
			artifact = PerformanceArtifact.model_validate_json(
				self.results_path.read_text(encoding="utf-8")
			)
		except (OSError, ValueError, ValidationError) as error:
			logger.exception("Could not read the evaluation results artifact")
			raise InvalidPerformanceArtifactError(
				"Evaluation results are invalid or unreadable."
			) from error

		if artifact.dataset_category == "student_test" and not (
			settings.allow_student_test_evaluation
			and artifact.student_test_evaluation_permitted
		):
			return {
				"status": "blocked",
				"dataset_category": "student_test",
				"models": [],
				"message": (
					"Student-test evaluation results are hidden unless project rules "
					"explicitly permit their use."
				),
			}
		return {
			"status": "available",
			"dataset_category": artifact.dataset_category,
			"models": [model.model_dump() for model in artifact.models],
			"message": "Evaluation results loaded from the saved artifact.",
		}


performance_service = PerformanceService()