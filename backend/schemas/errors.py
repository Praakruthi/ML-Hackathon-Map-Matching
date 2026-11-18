"""Consistent API error response models."""

from typing import Any

from pydantic import BaseModel


class ErrorDetail(BaseModel):
	code: str
	message: str
	context: dict[str, Any] | None = None


class ErrorResponse(BaseModel):
	error: ErrorDetail