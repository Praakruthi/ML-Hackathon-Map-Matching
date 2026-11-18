"""Backend application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router
from backend.config import settings

app = FastAPI(title=settings.app_name, version=settings.api_version)

app.add_middleware(
	CORSMiddleware,
	allow_origins=list(settings.cors_origins),
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(router)
