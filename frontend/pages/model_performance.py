"""Model comparison dashboard backed only by saved evaluation artifacts."""

import streamlit as st
import plotly.express as px

from frontend.services.api_client import APIClientError, api_client


def render_model_performance() -> None:
	st.title("Model Performance")
	st.write("Evaluation metrics are loaded from saved, provenance-checked results.")

	try:
		with st.spinner("Loading model and evaluation information..."):
			models = api_client.get_models()
			performance = api_client.get_performance()
	except APIClientError as error:
		st.error(f"Could not load model information: {error}")
		return

	if models["models"]:
		st.subheader("Model artifacts")
		st.dataframe(models["models"], width="stretch", hide_index=True)
	else:
		st.info(models["message"])

	st.subheader("Evaluation results")
	if performance["status"] == "blocked":
		st.warning(performance["message"])
		return
	if performance["status"] != "available" or not performance["models"]:
		st.info("No actual evaluation results are available yet. Metrics will appear here after an approved evaluation artifact is saved.")
		return

	rows = performance["models"]
	st.caption(f"Evaluation dataset: {performance['dataset_category']}")
	comparison = [
		{
			"Model": row["model_name"],
			"Accuracy": row.get("accuracy"),
			"Precision": row.get("precision"),
			"Recall": row.get("recall"),
			"F1-score": row.get("f1_score"),
		}
		for row in rows
	]
	st.dataframe(comparison, width="stretch", hide_index=True)
	chart_rows = [
		{"Model": row["model_name"], "Metric": name, "Score": row[key]}
		for row in rows
		for name, key in (
			("Accuracy", "accuracy"),
			("Precision", "precision"),
			("Recall", "recall"),
			("F1-score", "f1_score"),
		)
		if row.get(key) is not None
	]
	if chart_rows:
		chart = px.bar(
			chart_rows,
			x="Model",
			y="Score",
			color="Metric",
			barmode="group",
			range_y=[0, 1],
		)
		st.plotly_chart(chart, width="stretch")

	for row in rows:
		with st.expander(f"Details: {row['model_name']}"):
			if row.get("confusion_matrix"):
				st.write("Confusion matrix")
				confusion_chart = px.imshow(
					row["confusion_matrix"],
					text_auto=True,
					color_continuous_scale="Blues",
					labels={"x": "Predicted", "y": "Actual", "color": "Count"},
				)
				st.plotly_chart(confusion_chart, width="stretch")
			if row.get("cross_validation"):
				st.write("Cross-validation")
				st.json(row["cross_validation"])
			if row.get("hyperparameter_results"):
				st.write("Hyperparameter optimization")
				st.dataframe(row["hyperparameter_results"], width="stretch")
			if row.get("route_metrics"):
				st.write("Route reconstruction metrics")
				st.json(row["route_metrics"])