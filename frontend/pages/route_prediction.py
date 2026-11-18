"""Prediction interface that reports unavailable integrations without fake output."""

from typing import Any

import streamlit as st
from streamlit_folium import st_folium

from frontend.components.map_visualization import create_trip_map
from frontend.services.api_client import APIClientError, api_client


def render_route_prediction() -> None:
	st.title("Route Prediction")
	st.write("Inspect a trip and request a route reconstruction from the connected pipeline.")

	try:
		listing = api_client.get_trips()
		module_status = api_client.get_module_status()
	except APIClientError as error:
		st.error(f"Could not load prediction inputs: {error}")
		return

	if not module_status["pipeline_ready"]:
		st.info(
			"Prediction is not yet available. Connect the Member 1 preprocessing, "
			"Member 2 model, and Member 3 route-reconstruction adapters first. "
			"No prediction results are generated in this state."
		)

	dataset_label = st.radio("Dataset", ["Training", "Student test"], horizontal=True)
	category = "train" if dataset_label == "Training" else "student_test"
	trips: list[dict[str, Any]] = listing[
		"training_trips" if category == "train" else "test_trips"
	]
	available_trips = [trip for trip in trips if trip["available"]]
	if not available_trips:
		st.warning("No complete trips are available in this dataset.")
		return

	trip_id = st.selectbox("Trip", [trip["trip_id"] for trip in available_trips])
	try:
		with st.spinner(f"Loading raw GPS observations for {trip_id}..."):
			trip = api_client.get_trip(trip_id, category)
	except APIClientError as error:
		st.error(f"Could not load trip: {error}")
		return

	st.caption(
		f"{trip['gps_observation_count']} GPS observations · "
		f"{trip['road_node_count']} road nodes · {trip['road_arc_count']} road arcs"
	)
	st.subheader("Raw GPS trajectory")
	st_folium(
		create_trip_map(trip["gps_observations"]),
		width=None,
		height=500,
		key=f"prediction-input-{category}-{trip_id}",
	)

	if st.button("Initiate map matching", type="primary"):
		try:
			with st.spinner("Checking integrations and processing the trip..."):
				result = api_client.predict_trip(trip_id, category)
		except APIClientError as error:
			st.error(f"Prediction unavailable: {error}")
			context = error.payload.get("error", {}).get("context") or {}
			for module in context.get("modules", []):
				st.caption(f"{module['name']}: {module['message']}")
		else:
			st.session_state["route_prediction_result"] = result

	result = st.session_state.get("route_prediction_result")
	if not result or result.get("trip_id") != trip_id or result.get("dataset_category") != category:
		return

	route = result["route"]
	st.subheader("Reconstructed route")
	columns = st.columns(4)
	columns[0].metric("Route arcs", route["arc_count"])
	columns[1].metric("Route nodes", route["node_count"])
	columns[2].metric("Connected", "Yes" if route["connected"] else "No")
	columns[3].metric("Processing time", f"{result['processing_time_s']:.3f} s")
	st.write(f"Route validity: **{'Valid' if route['valid'] else 'Invalid'}**")
	if route.get("distance_m") is not None:
		st.write(f"Route distance: **{route['distance_m']:.1f} m**")
	st.dataframe(
		{"Predicted arc ID": route["arc_ids"]},
		width="stretch",
		height=220,
	)
	st_folium(
		create_trip_map(trip["gps_observations"], predicted_route=route),
		width=None,
		height=550,
		key=f"prediction-result-{category}-{trip_id}",
	)