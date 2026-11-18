"""Dataset discovery and validated parsers for GPS trips and road networks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Literal, TypeVar

from backend.config import PROJECT_ROOT, settings

DatasetCategory = Literal["train", "student_test"]
Row = TypeVar("Row")


class TripServiceError(Exception):
	"""Base error for dataset access and validation failures."""


class TripNotFoundError(TripServiceError):
	"""Raised when a trip does not exist in the requested category."""


class MissingTripFilesError(TripServiceError):
	"""Raised when a trip is missing one or more required input files."""


class InvalidTripDataError(TripServiceError):
	"""Raised when a trip file is empty, malformed, or inconsistent."""


class DatasetAccessError(TripServiceError):
	"""Raised when the dataset cannot be read from disk."""


@dataclass(frozen=True)
class ParsedFile(Generic[Row]):
	row_count: int
	rows: list[Row]
	row_indices: list[int]


def _parse_whitespace_file(
	path: Path,
	column_count: int,
	parse_row: Callable[[list[str], int], Row],
	*,
	limit: int | None = None,
	selected_indices: set[int] | None = None,
	validate_row: Callable[[Row, int], None] | None = None,
) -> ParsedFile[Row]:
	rows: list[Row] = []
	row_indices: list[int] = []
	row_count = 0
	try:
		with path.open("r", encoding="utf-8") as source:
			for line_number, line in enumerate(source, start=1):
				columns = line.split()
				if not columns:
					continue
				if len(columns) != column_count:
					raise InvalidTripDataError(
						f"{path.name}:{line_number} must contain exactly "
						f"{column_count} columns."
					)
				try:
					row = parse_row(columns, line_number)
				except InvalidTripDataError:
					raise
				except (TypeError, ValueError, OverflowError) as error:
					raise InvalidTripDataError(
						f"{path.name}:{line_number} contains an invalid value."
					) from error
				if validate_row is not None:
					validate_row(row, line_number)
				if selected_indices is not None:
					if row_count in selected_indices:
						rows.append(row)
						row_indices.append(row_count)
				elif limit is None or len(rows) < limit:
					rows.append(row)
					row_indices.append(row_count)
				row_count += 1
	except OSError as error:
		raise DatasetAccessError(f"Could not read {path.name}.") from error

	if row_count == 0:
		raise InvalidTripDataError(f"{path.name} is empty.")
	return ParsedFile(row_count=row_count, rows=rows, row_indices=row_indices)


def _parse_finite_floats(columns: list[str], line_number: int) -> tuple[float, ...]:
	values = tuple(float(value) for value in columns)
	if not all(math.isfinite(value) for value in values):
		raise InvalidTripDataError(
			f"Line {line_number} contains a non-finite numeric value."
		)
	return values


def parse_track(path: Path) -> ParsedFile[tuple[float, float, float]]:
	"""Parse longitude, latitude, and an uninterpreted third track value."""
	return _parse_whitespace_file(
		path,
		3,
		lambda columns, line: _parse_finite_floats(columns, line),
	)


def parse_nodes(
	path: Path,
	*,
	limit: int | None = None,
	selected_indices: set[int] | None = None,
) -> ParsedFile[tuple[float, float]]:
	"""Parse finite node coordinates, optionally retaining selected rows only."""
	return _parse_whitespace_file(
		path,
		2,
		lambda columns, line: _parse_finite_floats(columns, line),
		limit=limit,
		selected_indices=selected_indices,
	)


def parse_arcs(
	path: Path,
	node_count: int,
	*,
	limit: int | None = None,
) -> ParsedFile[tuple[int, int]]:
	"""Parse zero-based node-index pairs and validate every endpoint."""
	def parse_row(columns: list[str], line_number: int) -> tuple[int, int]:
		try:
			return int(columns[0]), int(columns[1])
		except ValueError as error:
			raise InvalidTripDataError(
				f"{path.name}:{line_number} must contain integer node indices."
			) from error

	def validate_row(arc: tuple[int, int], line_number: int) -> None:
		if any(node_index < 0 or node_index >= node_count for node_index in arc):
			raise InvalidTripDataError(
				f"{path.name}:{line_number} references a node outside "
				f"the valid range 0..{node_count - 1}."
			)

	return _parse_whitespace_file(
		path,
		2,
		parse_row,
		limit=limit,
		validate_row=validate_row,
	)


class TripService:
	"""Discover trips and load only the data requested for one trip."""

	REQUIRED_SUFFIXES = ("track", "nodes", "arcs")

	def __init__(self, dataset_root: Path | str | None = None) -> None:
		root = Path(dataset_root or settings.dataset_root).expanduser()
		self.dataset_root = root if root.is_absolute() else PROJECT_ROOT / root

	def discover_trips(self, category: DatasetCategory) -> list[dict[str, object]]:
		category_path = self.dataset_root / category
		if not category_path.exists():
			return []
		if not category_path.is_dir():
			raise DatasetAccessError(f"{category_path} is not a directory.")
		try:
			trip_directories = sorted(
				(path for path in category_path.iterdir() if path.is_dir()),
				key=lambda path: path.name,
			)
		except OSError as error:
			raise DatasetAccessError(f"Could not list {category_path}.") from error

		return [self._metadata(path, category) for path in trip_directories]

	def get_trip_metadata(
		self,
		trip_id: str,
		category: DatasetCategory,
	) -> dict[str, object]:
		if not trip_id or Path(trip_id).name != trip_id or trip_id in {".", ".."}:
			raise TripNotFoundError(f"Trip {trip_id!r} was not found.")
		category_path = self.dataset_root / category
		trip_path = category_path / trip_id
		if not trip_path.is_dir():
			raise TripNotFoundError(
				f"Trip {trip_id!r} was not found in {category}."
			)
		return self._metadata(trip_path, category)

	def load_trip(
		self,
		trip_id: str,
		category: DatasetCategory,
		*,
		include_network: bool = False,
		network_limit: int = 1000,
	) -> dict[str, object]:
		metadata = self.get_trip_metadata(trip_id, category)
		files = metadata["files"]
		assert isinstance(files, dict)
		missing = [name for name, available in files.items() if not available]
		if missing:
			raise MissingTripFilesError(
				f"Trip {trip_id!r} is missing required files: {', '.join(missing)}."
			)

		trip_path = self.dataset_root / category / trip_id
		track_file = parse_track(trip_path / f"{trip_id}.track")
		node_file = parse_nodes(trip_path / f"{trip_id}.nodes", limit=0)
		arc_file = parse_arcs(
			trip_path / f"{trip_id}.arcs",
			node_file.row_count,
			limit=network_limit if include_network else 0,
		)

		road_network: dict[str, object] | None = None
		if include_network:
			referenced_nodes = {
				node_index
				for arc in arc_file.rows
				for node_index in arc
			}
			network_nodes = parse_nodes(
				trip_path / f"{trip_id}.nodes",
				selected_indices=referenced_nodes,
			)
			node_rows = [
				{"node_index": index, "coordinates": list(coordinates)}
				for index, coordinates in zip(
					network_nodes.row_indices,
					network_nodes.rows,
				)
			]
			node_rows_by_index = {
				row["node_index"]: row for row in node_rows
			}
			arc_rows = [
				{
					"arc_index": index,
					"source_node": arc[0],
					"target_node": arc[1],
				}
				for index, arc in zip(arc_file.row_indices, arc_file.rows)
			]
			road_network = {
				"nodes": list(node_rows_by_index.values()),
				"arcs": arc_rows,
				"truncated": (
					arc_file.row_count > network_limit
					or len(referenced_nodes) < node_file.row_count
				),
				"limit": network_limit,
			}

		return {
			**metadata,
			"gps_observation_count": track_file.row_count,
			"road_node_count": node_file.row_count,
			"road_arc_count": arc_file.row_count,
			"gps_observations": [
				{
					"coordinates": [longitude, latitude],
					"third_column": third_column,
				}
				for longitude, latitude, third_column in track_file.rows
			],
			"road_network": road_network,
		}

	def _metadata(self, trip_path: Path, category: DatasetCategory) -> dict[str, object]:
		files = {
			suffix: (trip_path / f"{trip_path.name}.{suffix}").is_file()
			for suffix in self.REQUIRED_SUFFIXES
		}
		return {
			"trip_id": trip_path.name,
			"dataset_category": category,
			"files": files,
			"available": all(files.values()),
		}


trip_service = TripService()