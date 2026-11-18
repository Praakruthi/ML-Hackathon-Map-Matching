"""Non-destructive inventory of model artifacts."""

import logging
from datetime import datetime, timezone

from backend.config import settings

logger = logging.getLogger(__name__)


class ModelInventoryError(Exception):
	"""Raised when the model artifact directory cannot be inspected."""


class ModelService:
	ARTIFACT_SUFFIXES = {".joblib", ".pkl", ".pickle"}

	def __init__(self, model_directory=None) -> None:
		self.model_directory = model_directory or settings.model_directory

	def list_models(self) -> dict[str, object]:
		if not self.model_directory.exists():
			return {"models": [], "message": "No model artifacts are available."}
		try:
			artifacts = sorted(
				path
				for path in self.model_directory.iterdir()
				if path.is_file() and path.suffix.lower() in self.ARTIFACT_SUFFIXES
			)
			models = [
				{
					"model_id": path.name,
					"filename": path.name,
					"artifact_type": path.suffix.lower().lstrip("."),
					"size_bytes": path.stat().st_size,
					"modified_at": datetime.fromtimestamp(
						path.stat().st_mtime,
						tz=timezone.utc,
					).isoformat(),
				}
				for path in artifacts
			]
		except OSError as error:
			logger.exception("Could not inspect model artifact directory")
			raise ModelInventoryError("Could not inspect model artifacts.") from error
		return {
			"models": models,
			"message": (
				f"{len(models)} model artifact(s) found; artifacts are listed but not loaded."
				if models
				else "No model artifacts are available."
			),
		}


model_service = ModelService()