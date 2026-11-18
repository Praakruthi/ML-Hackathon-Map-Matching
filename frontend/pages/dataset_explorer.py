"""Streamlit page for browsing available GPS trips."""

from typing import Any

import streamlit as st
from streamlit_folium import st_folium

from frontend.components.map_visualization import create_trip_map
from frontend.services.api_client import APIClientError, api_client


def render_dataset_explorer() -> None:
	st.title("Dataset Explorer")
	st.write("Browse GPS observations and road-network data for an available trip.")

	try:
		with st.spinner("Loading available trips..."):
			trip_listing = api_client.get_trips()
	except APIClientError as error:
		st.error(f"Could not load trips: {error}")
		return

	category_label = st.radio(
		"Dataset",
		["Training", "Student test"],
		horizontal=True,
	)
	category = "train" if category_label == "Training" else "student_test"
	trips: list[dict[str, Any]] = trip_listing[
		"training_trips" if category == "train" else "test_trips"
	]
	available_trips = [trip for trip in trips if trip["available"]]
	missing_file_count = len(trips) - len(available_trips)
	if missing_file_count:
		st.info(f"{missing_file_count} trip folder(s) are missing required input files.")
	if not available_trips:
		st.warning("No trips with all required files are available in this dataset.")
		return

	trip_id = st.selectbox(
		"Trip",
		[trip["trip_id"] for trip in available_trips],
	)
	include_network = st.checkbox(
		"Include road network sample",
		value=False,
		help="Loads at most 1,000 arcs and only their endpoint coordinates.",
	)

	try:
		with st.spinner(f"Loading trip {trip_id}..."):
			trip = api_client.get_trip(
				trip_id,
				category,
				include_network=include_network,
				network_limit=1000,
			)
	except APIClientError as error:
		st.error(f"Could not load trip {trip_id}: {error}")
		return

	st.caption(
		f"Trip **{trip['trip_id']}** | Dataset: **{category_label}**"
	)
	metric_columns = st.columns(3)
	metric_columns[0].metric("GPS observations", trip["gps_observation_count"])
	metric_columns[1].metric("Road nodes", trip["road_node_count"])
	metric_columns[2].metric("Road arcs", trip["road_arc_count"])

	observations = trip["gps_observations"]
	st.subheader("GPS trajectory")
	st.dataframe(
		[
			{
				"longitude": point["coordinates"][0],
				"latitude": point["coordinates"][1],
				"third_column (unverified)": point["third_column"],
			}
			for point in observations
		],
		width="stretch",
		height=240,
	)
	st.caption(
		"Coordinates are ordered as longitude, latitude. The third track column "
		"is preserved; its meaning is unverified."
	)

	network = trip.get("road_network") if include_network else None
	if network and network["truncated"]:
		st.caption(
			f"Road network visualization is limited to a sample of up to "
			f"{network['limit']} arcs."
		)
	st_folium(
		create_trip_map(observations, network),
		width=None,
		height=600,
		key=f"trip-map-{category}-{trip_id}-{include_network}",
	)