"""HTTP routes for the RouteMatch AI backend."""

import logging

from fastapi import APIRouter
from fastapi import Query
from starlette import status

from backend.api_errors import APIError
from backend.config import settings
from backend.schemas.trips import (
	DatasetCategory,
	TripDetailResponse,
	TripListResponse,
)
from backend.schemas.models import ModelsResponse
from backend.schemas.modules import ModulesStatusResponse
from backend.schemas.performance import PerformanceResponse
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.trip_service import (
	DatasetAccessError,
	InvalidTripDataError,
	MissingTripFilesError,
	TripNotFoundError,
	trip_service,
)
from backend.services.model_service import ModelInventoryError, model_service
from backend.services.performance_service import (
	InvalidPerformanceArtifactError,
	performance_service,
)
from backend.services.prediction_service import (
	PredictionUnavailableError,
	prediction_service,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
def get_health() -> dict[str, str]:
	"""Report application identity and backend availability."""
	return {
		"application": settings.app_name,
		"status": "ok",
		"api_version": settings.api_version,
	}


@router.get("/trips", response_model=TripListResponse)
def get_trips() -> dict[str, list[dict[str, object]]]:
	"""List trip folders and required-file availability without parsing data."""
	try:
		return {
			"training_trips": trip_service.discover_trips("train"),
			"test_trips": trip_service.discover_trips("student_test"),
		}
	except DatasetAccessError as error:
		logger.exception("Trip discovery failed")
		raise APIError(
			status.HTTP_503_SERVICE_UNAVAILABLE,
			"dataset_unavailable",
			str(error),
		) from error


@router.get("/trips/{trip_id}", response_model=TripDetailResponse)
def get_trip(
	trip_id: str,
	dataset_category: DatasetCategory = Query(...),
	include_network: bool = False,
	network_limit: int = Query(default=1000, ge=1, le=5000),
) -> dict[str, object]:
	"""Return trajectory data and optionally a capped road-network subset."""
	try:
		return trip_service.load_trip(
			trip_id,
			dataset_category,
			include_network=include_network,
			network_limit=network_limit,
		)
	except TripNotFoundError as error:
		raise APIError(status.HTTP_404_NOT_FOUND, "trip_not_found", str(error)) from error
	except MissingTripFilesError as error:
		raise APIError(status.HTTP_404_NOT_FOUND, "trip_files_missing", str(error)) from error
	except InvalidTripDataError as error:
		raise APIError(
			status.HTTP_422_UNPROCESSABLE_CONTENT,
			"invalid_trip_data",
			str(error),
		) from error
	except DatasetAccessError as error:
		logger.exception("Trip loading failed")
		raise APIError(
			status.HTTP_503_SERVICE_UNAVAILABLE,
			"dataset_unavailable",
			str(error),
		) from error


@router.get("/modules/status", response_model=ModulesStatusResponse)
def get_module_status() -> dict[str, object]:
	return prediction_service.module_registry.get_status()


@router.get("/models", response_model=ModelsResponse)
def get_models() -> dict[str, object]:
	try:
		return model_service.list_models()
	except ModelInventoryError as error:
		raise APIError(
			status.HTTP_503_SERVICE_UNAVAILABLE,
			"model_inventory_unavailable",
			str(error),
		) from error


@router.get("/models/performance", response_model=PerformanceResponse)
def get_model_performance() -> dict[str, object]:
	try:
		return performance_service.get_performance()
	except InvalidPerformanceArtifactError as error:
		raise APIError(
			status.HTTP_422_UNPROCESSABLE_CONTENT,
			"invalid_performance_artifact",
			str(error),
		) from error


@router.post("/predict/{trip_id}", response_model=PredictionResponse)
def predict_trip(trip_id: str, request: PredictionRequest) -> dict[str, object]:
	try:
		metadata = trip_service.get_trip_metadata(trip_id, request.dataset_category)
		files = metadata["files"]
		if not all(files.values()):
			missing = [name for name, available in files.items() if not available]
			raise MissingTripFilesError(
				f"Trip {trip_id!r} is missing required files: {', '.join(missing)}."
			)
		return prediction_service.predict_trip(
			trip_id,
			request.dataset_category,
			request.model_id,
		)
	except TripNotFoundError as error:
		raise APIError(status.HTTP_404_NOT_FOUND, "trip_not_found", str(error)) from error
	except MissingTripFilesError as error:
		raise APIError(status.HTTP_404_NOT_FOUND, "trip_files_missing", str(error)) from error
	except PredictionUnavailableError as error:
		raise APIError(
			status.HTTP_503_SERVICE_UNAVAILABLE,
			"prediction_unavailable",
			str(error),
			{"modules": error.modules},
		) from error
	except (ValueError, KeyError) as error:
		logger.exception("Prediction pipeline failed for trip %s", trip_id)
		raise APIError(
			status.HTTP_500_INTERNAL_SERVER_ERROR,
			"prediction_failed",
			"The prediction pipeline failed while processing this trip.",
		) from error
