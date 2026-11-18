"""Streamlit frontend entry point."""

import requests
import streamlit as st

from frontend.pages.dataset_explorer import render_dataset_explorer
from frontend.services.api_client import APIClientError, api_client

st.set_page_config(page_title="RouteMatch AI", layout="wide")

with st.sidebar:
	st.title("RouteMatch AI")
	page = st.radio("Navigation", ["Overview", "Dataset Explorer"])
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

if page == "Dataset Explorer":
	render_dataset_explorer()
else:
	st.title("GPS Trajectory Map Matching")
	st.write(
		"Explore GPS trajectories and road-network data as RouteMatch AI "
		"integrates the team modules."
	)
	st.caption("Dataset browsing is available from the Dataset Explorer page.")