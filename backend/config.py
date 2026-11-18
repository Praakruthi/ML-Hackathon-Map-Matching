"""Backend and frontend connection settings."""

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
	app_name: str = os.getenv("APP_NAME", "RouteMatch AI")
	api_version: str = os.getenv("API_VERSION", "1.0.0")
	backend_host: str = os.getenv("BACKEND_HOST", "127.0.0.1")
	backend_port: int = int(os.getenv("BACKEND_PORT", "8000"))
	backend_api_url: str = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")
	dataset_root: Path = Path(
		os.getenv("DATASET_ROOT", str(PROJECT_ROOT / "data" / "dataset"))
	).expanduser()
	cors_origins: tuple[str, ...] = tuple(
		origin.strip()
		for origin in os.getenv(
			"CORS_ORIGINS",
			"http://localhost:8501,http://127.0.0.1:8501",
		).split(",")
		if origin.strip()
	)


settings = Settings()
