"""Backend and frontend connection settings."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _configured_path(variable: str, default: Path) -> Path:
	path = Path(os.getenv(variable, str(default))).expanduser()
	return path if path.is_absolute() else PROJECT_ROOT / path


@dataclass(frozen=True)
class Settings:
	app_name: str = os.getenv("APP_NAME", "RouteMatch AI")
	api_version: str = os.getenv("API_VERSION", "1.0.0")
	backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
	backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
	backend_api_url: str = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
	dataset_root: Path = _configured_path(
		"DATASET_ROOT", PROJECT_ROOT / "data" / "dataset"
	)
	model_directory: Path = _configured_path("MODEL_DIRECTORY", PROJECT_ROOT / "models")
	performance_results_path: Path = _configured_path(
		"PERFORMANCE_RESULTS_PATH", PROJECT_ROOT / "outputs" / "performance.json"
	)
	allow_student_test_evaluation: bool = (
		os.getenv("ALLOW_STUDENT_TEST_EVALUATION", "false").strip().lower()
		in {"1", "true", "yes"}
	)
	log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
	cors_origins: tuple[str, ...] = tuple(
		origin.strip()
		for origin in os.getenv(
			"CORS_ORIGINS",
			"http://localhost:8501,http://127.0.0.1:8501",
		).split(",")
		if origin.strip()
	)

settings = Settings()
