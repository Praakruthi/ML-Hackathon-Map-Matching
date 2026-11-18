"""Project dashboard with live dataset and module readiness summaries."""

import streamlit as st

from frontend.services.api_client import APIClientError, api_client


def render_dashboard() -> None:
	st.title("RouteMatch AI")
	st.write(
		"GPS trajectory map matching links noisy location observations to a road "
		"network, then reconstructs a connected route."
	)
	st.caption(
		"The application foundation is ready. Prediction remains unavailable "
		"until the member adapters are connected."
	)

	try:
		with st.spinner("Loading project status..."):
			trips = api_client.get_trips()
			modules = api_client.get_module_status()
	except APIClientError as error:
		st.error(f"Could not load dashboard status: {error}")
		return

	training_count = sum(trip["available"] for trip in trips["training_trips"])
	test_count = sum(trip["available"] for trip in trips["test_trips"])
	st.subheader("Dataset")
	metrics = st.columns(2)
	metrics[0].metric("Available training trips", training_count)
	metrics[1].metric("Available student-test trips", test_count)

	st.subheader("Member modules")
	for module in modules["modules"]:
		label = "Ready" if module["ready"] else "Not integrated"
		with st.container(border=True):
			st.markdown(f"**{module['name']}**  ·  {label}")
			st.caption(module["message"])
	if not modules["pipeline_ready"]:
		st.info("The prediction pipeline is unavailable until all three adapters are connected.")