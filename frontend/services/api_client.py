"""Requests-based client for the RouteMatch AI backend API."""

from typing import Any

import requests

from backend.config import settings


class APIClientError(Exception):
	"""Raised when a backend request fails or returns invalid JSON."""

	def __init__(
		self,
		message: str,
		*,
		status_code: int | None = None,
		payload: dict[str, Any] | None = None,
	) -> None:
		self.status_code = status_code
		self.payload = payload or {}
		super().__init__(message)


class APIClient:
	def __init__(self, base_url: str | None = None, timeout: float = 10) -> None:
		self.base_url = (base_url or settings.backend_api_url).rstrip("/")
		self.timeout = timeout

	def get_health(self) -> dict[str, Any]:
		return self._request("GET", "/health")

	def get_trips(self) -> dict[str, Any]:
		return self._request("GET", "/trips")

	def get_trip(
		self,
		trip_id: str,
		dataset_category: str,
		*,
		include_network: bool = False,
		network_limit: int = 1000,
	) -> dict[str, Any]:
		return self._request(
			"GET",
			f"/trips/{trip_id}",
			params={
				"dataset_category": dataset_category,
				"include_network": include_network,
				"network_limit": network_limit,
			},
		)

	def get_module_status(self) -> dict[str, Any]:
		return self._request("GET", "/modules/status")

	def get_models(self) -> dict[str, Any]:
		return self._request("GET", "/models")

	def get_performance(self) -> dict[str, Any]:
		return self._request("GET", "/models/performance")

	def predict_trip(
		self,
		trip_id: str,
		dataset_category: str,
		model_id: str | None = None,
	) -> dict[str, Any]:
		return self._request(
			"POST",
			f"/predict/{trip_id}",
			json={"dataset_category": dataset_category, "model_id": model_id},
		)

	def _request(
		self,
		method: str,
		path: str,
		*,
		params: dict[str, str | int | bool] | None = None,
		json: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		try:
			response = requests.request(
				method,
				f"{self.base_url}{path}",
				params=params,
				json=json,
				timeout=self.timeout,
			)
		except requests.RequestException as error:
			raise APIClientError("Could not connect to the backend API.") from error

		try:
			payload = response.json()
		except ValueError as error:
			raise APIClientError("The backend returned an invalid response.") from error
		if not isinstance(payload, dict):
			raise APIClientError("The backend returned an unexpected response.")

		if not response.ok:
			api_error = payload.get("error", {})
			detail = api_error.get("message", payload.get("detail", "Backend request failed."))
			raise APIClientError(
				str(detail),
				status_code=response.status_code,
				payload=payload,
			)
		return payload


api_client = APIClient()