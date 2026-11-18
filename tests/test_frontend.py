"""Frontend component tests for the map layer contract."""

from typing import Any

from frontend.services.api_client import APIClient
from frontend.components.map_visualization import create_trip_map
from frontend.pages import dashboard, dataset_explorer, model_performance, route_prediction


def test_streamlit_page_renderers_import() -> None:
	assert callable(dashboard.render_dashboard)
	assert callable(dataset_explorer.render_dataset_explorer)
	assert callable(model_performance.render_model_performance)
	assert callable(route_prediction.render_route_prediction)


def test_api_client_health_and_trip_methods_use_shared_request(
	monkeypatch,
) -> None:
	calls: list[tuple[str, str]] = []

	class Response:
		ok = True

		@staticmethod
		def json() -> dict[str, Any]:
			return {"status": "ok"}

	def fake_request(method: str, url: str, **kwargs: Any) -> Response:
		calls.append((method, url))
		return Response()

	monkeypatch.setattr(
		"frontend.services.api_client.requests.request",
		fake_request,
	)
	client = APIClient("http://backend")

	assert client.get_health() == {"status": "ok"}
	assert client.get_trips() == {"status": "ok"}
	assert calls == [("GET", "http://backend/health"), ("GET", "http://backend/trips")]


def test_trip_map_separates_raw_network_and_predicted_layers() -> None:
	trip_map = create_trip_map(
		[
			{"coordinates": [1.0, 2.0], "third_column": 1.0},
			{"coordinates": [3.0, 4.0], "third_column": 2.0},
		],
		road_network={
			"nodes": [
				{"node_index": 0, "coordinates": [1.0, 2.0]},
				{"node_index": 1, "coordinates": [3.0, 4.0]},
			],
			"arcs": [{"arc_index": 0, "source_node": 0, "target_node": 1}],
		},
		predicted_route={"coordinates": [[1.0, 2.0], [3.0, 4.0]]},
	)

	rendered = trip_map.get_root().render()
	assert "Raw GPS trajectory" in rendered
	assert "Road network sample" in rendered
	assert "Predicted road route" in rendered
	assert "[2.0, 1.0]" in rendered