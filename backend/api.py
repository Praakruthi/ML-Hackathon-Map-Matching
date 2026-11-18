"""HTTP routes for the RouteMatch AI backend."""

from fastapi import APIRouter
from fastapi import HTTPException, Query
from starlette import status

from backend.config import settings
from backend.schemas.trips import (
	DatasetCategory,
	TripDetailResponse,
	TripListResponse,
)
from backend.services.trip_service import (
	DatasetAccessError,
	InvalidTripDataError,
	MissingTripFilesError,
	TripNotFoundError,
	trip_service,
)

router = APIRouter()


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
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail=str(error),
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
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=str(error),
		) from error
	except MissingTripFilesError as error:
		raise HTTPException(
			status_code=status.HTTP_404_NOT_FOUND,
			detail=str(error),
		) from error
	except InvalidTripDataError as error:
		raise HTTPException(
			status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
			detail=str(error),
		) from error
	except DatasetAccessError as error:
		raise HTTPException(
			status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
			detail=str(error),
		) from error
