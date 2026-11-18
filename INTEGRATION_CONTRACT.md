# Integration Contract

This document defines the Member 4 adapter boundary. The adapter signatures and payloads below are standardized proposals for team confirmation; member implementations should be connected by injecting implementations into the adapters, not by coupling API routes to member internals.

## Readiness

`GET /modules/status` distinguishes a detected Python package from a connected adapter. A package being importable does not mean its implementation satisfies this contract. The current `src/decoder/` implementation is detected for Member 3 but is not connected because the standardized Member 1 and Member 2 inputs have not been integrated. Member 1 and Member 2 adapters are also not connected.

Prediction remains unavailable until all three adapters report ready. The backend returns HTTP 503 with code `prediction_unavailable` and the readiness details. It does not fabricate a route.

## Member 1 - Data Engineering

Expected standardized interface:

```python
load_trip(trip_id: str, dataset_category: str) -> Mapping[str, Any]
generate_candidates(gps_points, road_graph) -> Mapping[str, Any]
extract_features(candidates) -> Mapping[str, Any]
```

`load_trip` returns `gps_points` as ordered `(longitude, latitude)` pairs and `road_graph` with `nodes` and `arcs`. Candidate and feature mappings retain observation order and original zero-based `.arcs` row IDs. The API's raw-trip service is not a replacement for Member 1 candidate generation or feature engineering.

## Member 2 - Machine Learning

Expected standardized interface:

```python
load_best_model(model_id: str | None = None) -> Any
predict_candidates(model: Any, features: Mapping[str, Any]) -> Mapping[str, Any]
```

Prediction output must preserve the observation IDs and candidate arc IDs and provide a probability/score for each candidate. Model feature ordering and probability meaning must be documented by the team.

## Member 3 - HMM and Route Reconstruction

Expected standardized interface:

```python
reconstruct_route(gps_points, candidates, probabilities, road_graph) -> Mapping[str, Any]
validate_route(predicted_arc_ids, road_graph) -> Mapping[str, Any]
```

The route mapping contains ordered `arc_ids`; it may include `distance_m`. Validation returns `valid` and `connected` booleans. The Member 4 route serializer resolves arc IDs against the same trip's road graph and returns geographic coordinates in `(longitude, latitude)` order.

The existing `src/decoder` functions have richer typed inputs (including `RoadNetwork`, observation metadata, and candidate positions), so the Member 3 adapter remains unconnected until a mapping to this contract is confirmed.

## Shared Conventions

- Raw coordinates and backend route coordinates use `(longitude, latitude)`.
- Folium rendering converts coordinates to `[latitude, longitude]`.
- Arc IDs are original zero-based row indices in the trip's `.arcs` file; they are trip-scoped.
- GPS observations retain their source order.
- `.route` is reference data, not prediction input. The prediction endpoint and adapters never read it.
- Student-test reference routes must not be used for prediction. Evaluation results from `student_test` are hidden unless both `ALLOW_STUDENT_TEST_EVALUATION=true` is explicitly configured and the artifact declares `student_test_evaluation_permitted: true`.
- Model feature ordering must be documented and versioned with model artifacts.

## Performance Artifact

The performance endpoint reads only `outputs/performance.json`; it does not calculate metrics. The file must contain a `dataset_category` (`train` or `student_test`) and a `models` list. Each model may provide actual `accuracy`, `precision`, `recall`, `f1_score`, `confusion_matrix`, `cross_validation`, `hyperparameter_results`, and `route_metrics`. Unsupported, malformed, or unprovenanced data is not rendered. No artifact currently exists, so the performance page shows an empty state.

## API Endpoints

- `GET /health`
- `GET /trips`
- `GET /trips/{trip_id}?dataset_category=train`
- `POST /predict/{trip_id}` with `{"dataset_category": "train", "model_id": null}`
- `GET /models`
- `GET /models/performance`
- `GET /modules/status`

Errors use `{"error": {"code": "...", "message": "...", "context": null}}`; unavailable predictions include module readiness in `context`.
