"""Requests-based client for the RouteMatch AI backend API."""

from typing import Any

import requests

from backend.config import settings


class APIClientError(Exception):
	"""Raised when a backend request fails or returns invalid JSON."""


class APIClient:
	def __init__(self, base_url: str | None = None, timeout: float = 10) -> None:
		self.base_url = (base_url or settings.backend_api_url).rstrip("/")
		self.timeout = timeout

	def get_health(self) -> dict[str, Any]:
		return self._get("/health")

	def get_trips(self) -> dict[str, Any]:
		return self._get("/trips")

	def get_trip(
		self,
		trip_id: str,
		dataset_category: str,
		*,
		include_network: bool = False,
		network_limit: int = 1000,
	) -> dict[str, Any]:
		return self._get(
			f"/trips/{trip_id}",
			params={
				"dataset_category": dataset_category,
				"include_network": include_network,
				"network_limit": network_limit,
			},
		)

	def _get(
		self,
		path: str,
		*,
		params: dict[str, str | int | bool] | None = None,
	) -> dict[str, Any]:
		try:
			response = requests.get(
				f"{self.base_url}{path}",
				params=params,
				timeout=self.timeout,
			)
		except requests.RequestException as error:
			raise APIClientError("Could not connect to the backend API.") from error

		try:
			payload = response.json()
		except ValueError as error:
			raise APIClientError("The backend returned an invalid response.") from error

		if not response.ok:
			detail = payload.get("detail", "Backend request failed.")
			raise APIClientError(str(detail))
		if not isinstance(payload, dict):
			raise APIClientError("The backend returned an unexpected response.")
		return payload


api_client = APIClient()