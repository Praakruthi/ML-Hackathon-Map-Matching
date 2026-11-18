# Integration Contract

All interfaces in this document are **PROPOSED** and subject to team confirmation. This document describes intended boundaries only; no implementation is included yet.

## Member 1 - Data Engineering

Proposed interfaces:

- `load_trip(trip_id)` - Load one trip and its GPS observations.
- `generate_candidates(gps_points, road_graph)` - Generate candidate road arcs for the observations.
- `extract_features(candidates)` - Produce model input features for candidate arcs.

## Member 2 - Machine Learning

Proposed interfaces:

- `load_best_model(model_path)` - Load the selected model artifact.
- `predict_candidates(features)` - Return a probability or score for each candidate.

## Member 3 - HMM and Route Reconstruction

Proposed interfaces:

- `reconstruct_route(gps_points, candidates, probabilities, road_graph)` - Select an ordered route from candidate predictions.
- `validate_route(predicted_arc_ids, road_graph)` - Check that the predicted route is valid against the road graph.

## Member 4 - Backend and Frontend

- Integrate the team modules through FastAPI.
- Provide structured prediction responses.
- Display results through Streamlit and Folium.

## Common Conventions

- Coordinates are represented as `(longitude, latitude)`.
- Arc IDs refer to original zero-based row indices in `.arcs`.
- GPS observations retain their original ordering.
- Predicted routes are represented as ordered arc IDs.
- Model feature ordering must be documented and agreed upon by the data engineering and machine learning members.

All names, signatures, data shapes, and conventions above remain **PROPOSED** until confirmed by the team.
