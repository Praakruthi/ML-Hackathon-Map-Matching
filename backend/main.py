"""Backend application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request

from backend.api_errors import APIError
from backend.api import router
from backend.config import settings

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO))
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version=settings.api_version)

app.add_middleware(
	CORSMiddleware,
	allow_origins=list(settings.cors_origins),
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)

app.include_router(router)


def run_backend() -> None:
	"""Start Uvicorn using the configured backend host and port."""
	import uvicorn

	uvicorn.run(
		"backend.main:app",
		host=settings.backend_host,
		port=settings.backend_port,
		reload=True,
	)


@app.exception_handler(APIError)
async def handle_api_error(request: Request, error: APIError) -> JSONResponse:
	logger.warning("API request failed: %s %s: %s", request.method, request.url.path, error.code)
	return JSONResponse(
		status_code=error.status_code,
		content={
			"error": {
				"code": error.code,
				"message": error.message,
				"context": error.context,
			}
		},
	)


@app.exception_handler(RequestValidationError)
async def handle_request_validation_error(
	request: Request,
	error: RequestValidationError,
) -> JSONResponse:
	return JSONResponse(
		status_code=422,
		content={
			"error": {
				"code": "request_validation_error",
				"message": "Request parameters or body failed validation.",
				"context": {"issues": error.errors()},
			}
		},
	)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(
	request: Request,
	error: StarletteHTTPException,
) -> JSONResponse:
	return JSONResponse(
		status_code=error.status_code,
		content={
			"error": {
				"code": "http_error",
				"message": str(error.detail),
				"context": None,
			}
		},
		headers=error.headers,
	)


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
	logger.exception("Unhandled request failure for %s %s", request.method, request.url.path)
	return JSONResponse(
		status_code=500,
		content={
			"error": {
				"code": "internal_error",
				"message": "An unexpected backend error occurred.",
				"context": None,
			}
		},
	)


if __name__ == "__main__":
	run_backend()
