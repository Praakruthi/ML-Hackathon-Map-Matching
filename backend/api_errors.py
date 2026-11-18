"""Typed errors translated into consistent HTTP responses."""

from typing import Any


class APIError(Exception):
	def __init__(
		self,
		status_code: int,
		code: str,
		message: str,
		context: dict[str, Any] | None = None,
	) -> None:
		self.status_code = status_code
		self.code = code
		self.message = message
		self.context = context
		super().__init__(message)