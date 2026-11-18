"""Folium visualization for GPS observations and bounded road networks."""

from math import ceil
from typing import Any

import folium

MAX_GPS_MARKERS = 200
MAX_GPS_POLYLINE_POINTS = 10_000


def create_trip_map(
	gps_observations: list[dict[str, Any]],
	road_network: dict[str, Any] | None = None,
	predicted_route: dict[str, Any] | None = None,
) -> folium.Map:
	"""Build a map from [longitude, latitude] coordinates."""
	if not gps_observations:
		raise ValueError("A map requires at least one GPS observation.")

	coordinates = [observation["coordinates"] for observation in gps_observations]
	map_coordinates = [
		[float(latitude), float(longitude)] for longitude, latitude in coordinates
	]
	center = [
		sum(point[0] for point in map_coordinates) / len(map_coordinates),
		sum(point[1] for point in map_coordinates) / len(map_coordinates),
	]
	trip_map = folium.Map(location=center, zoom_start=14, control_scale=True)
	gps_layer = folium.FeatureGroup(name="Raw GPS trajectory")

	polyline_stride = max(1, ceil(len(map_coordinates) / MAX_GPS_POLYLINE_POINTS))
	polyline_points = map_coordinates[::polyline_stride]
	if polyline_points[-1] != map_coordinates[-1]:
		polyline_points.append(map_coordinates[-1])
	folium.PolyLine(
		polyline_points,
		color="#1677a6",
		weight=4,
		opacity=0.85,
		tooltip="GPS trajectory",
	).add_to(gps_layer)

	marker_stride = max(1, ceil(len(map_coordinates) / MAX_GPS_MARKERS))
	marker_indices = set(range(0, len(map_coordinates), marker_stride))
	marker_indices.update({0, len(map_coordinates) - 1})
	for index in sorted(marker_indices):
		color = "#16805d" if index == 0 else "#c74440" if index == len(map_coordinates) - 1 else "#f2a541"
		folium.CircleMarker(
			location=map_coordinates[index],
			radius=5 if index in {0, len(map_coordinates) - 1} else 3,
			color=color,
			fill=True,
			fill_opacity=0.9,
			tooltip="Start" if index == 0 else "End" if index == len(map_coordinates) - 1 else f"GPS observation {index + 1}",
		).add_to(gps_layer)
	gps_layer.add_to(trip_map)

	if road_network:
		nodes = {
			int(node["node_index"]): node["coordinates"]
			for node in road_network.get("nodes", [])
		}
		road_layer = folium.FeatureGroup(name="Road network sample")
		for arc in road_network.get("arcs", []):
			source = nodes.get(int(arc["source_node"]))
			target = nodes.get(int(arc["target_node"]))
			if source is None or target is None:
				continue
			folium.PolyLine(
				[
					[float(source[1]), float(source[0])],
					[float(target[1]), float(target[0])],
				],
				color="#59636e",
				weight=2,
				opacity=0.55,
			).add_to(road_layer)
		road_layer.add_to(trip_map)

	if predicted_route and predicted_route.get("coordinates"):
		route_points = [
			[float(latitude), float(longitude)]
			for longitude, latitude in predicted_route["coordinates"]
		]
		folium.PolyLine(
			route_points,
			color="#16805d",
			weight=6,
			opacity=0.9,
			tooltip="Predicted road route",
		).add_to(folium.FeatureGroup(name="Predicted road route").add_to(trip_map))

	folium.LayerControl(collapsed=True).add_to(trip_map)

	return trip_map