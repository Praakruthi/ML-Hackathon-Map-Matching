"""Streamlit frontend entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from frontend.pages.dashboard import render_dashboard
from frontend.pages.dataset_explorer import render_dataset_explorer
from frontend.pages.model_performance import render_model_performance
from frontend.pages.route_prediction import render_route_prediction
from frontend.services.api_client import APIClientError, api_client

st.set_page_config(page_title="RouteMatch AI", layout="wide")

st.navigation(
	[
		st.Page(render_dashboard, title="Dashboard", default=True),
		st.Page(render_dataset_explorer, title="Dataset Explorer"),
		st.Page(render_route_prediction, title="Route Prediction"),
		st.Page(render_model_performance, title="Model Performance"),
	],
	position="hidden",
)

with st.sidebar:
	st.title("RouteMatch AI")
	page_name = st.radio(
		"Navigation",
		["Dashboard", "Dataset Explorer", "Route Prediction", "Model Performance"],
	)
	st.divider()
	st.subheader("Backend connection")
	try:
		health = api_client.get_health()
	except APIClientError:
		st.warning("Backend unavailable")
	else:
		if health.get("status") == "ok":
			st.success("Backend connected")
		else:
			st.warning("Backend status unavailable")

{
	"Dashboard": render_dashboard,
	"Dataset Explorer": render_dataset_explorer,
	"Route Prediction": render_route_prediction,
	"Model Performance": render_model_performance,
}[page_name]()