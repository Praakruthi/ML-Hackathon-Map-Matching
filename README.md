# RouteMatch AI - GPS Trajectory Map Matching and Route Reconstruction

## Project Overview

RouteMatch AI is a four-member hackathon project for matching ordered GPS observations to a road network and reconstructing a connected route. The repository currently includes the FastAPI and Streamlit application foundation, dataset discovery, trip visualization, module adapter boundaries, and artifact-backed model/performance views. Prediction remains unavailable until the Member 1, 2, and 3 adapters are connected.

## Problem Domain

GPS trajectory map matching associates noisy, ordered GPS observations with the road network they most likely represent. The planned work will explore candidate road segments, model-based scoring, and route reconstruction while keeping the original observation order and road-arc identifiers consistent.

## Technology Stack

- Python
- FastAPI and Uvicorn for the backend API
- Streamlit, Folium, and Streamlit-Folium for the interactive map interface
- NumPy, pandas, scikit-learn, SciPy, NetworkX, and joblib for data and modeling workflows
- Matplotlib and Plotly for visualization
- pytest for testing

## Team Responsibilities

- **Member 1 - Data Engineering:** Organize input data, load trips, generate road candidates, and extract features.
- **Member 2 - Machine Learning:** Prepare model workflows, load the selected model, and score candidate roads.
- **Member 3 - HMM and Route Reconstruction:** Develop route reconstruction and route validation components.
- **Member 4 - Backend and Frontend:** Integrate the modules through FastAPI and present prediction results using Streamlit and Folium.

These responsibilities are an initial proposal and can be adjusted by the team.

## Project Structure

```text
ML-Hackathon-Map-Matching/
├── README.md
├── requirements.txt
├── .gitignore
├── INTEGRATION_CONTRACT.md
├── data/
│   └── README.md
├── notebooks/
│   ├── member1/.gitkeep
│   ├── member2/.gitkeep
│   ├── member3/.gitkeep
│   └── member4/.gitkeep
├── src/
│   ├── member1/__init__.py
│   ├── member2/__init__.py
│   └── member3/__init__.py
├── backend/
│   ├── main.py, api.py, config.py, api_errors.py
│   ├── services/ (trip, module, model, performance, prediction)
│   ├── integrations/ (member adapters, shared contracts, route serializer)
│   └── schemas/ (typed API contracts)
├── frontend/
│   ├── app.py
│   ├── pages/ (dashboard, dataset explorer, route prediction, performance)
│   ├── components/ (Folium map visualization)
│   └── services/ (backend API client)
├── models/
│   └── README.md
├── outputs/
│   └── README.md
└── tests/
	└── .gitkeep
```

## Installation (Windows CMD)

From the repository root, create and activate a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Local Execution (Windows CMD)

In the first terminal, start FastAPI:

```bash
set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
python -m backend.main
```

In a second terminal, start Streamlit:

```bash
set "BACKEND_API_URL=http://127.0.0.1:8000"
python -m streamlit run frontend/app.py
```

The backend must be running before the frontend can show a connected status. Swagger documentation is available at `http://127.0.0.1:8000/docs`.

## Two-Laptop Execution

On **Laptop A (backend)**, run `ipconfig` and note the active adapter's IPv4 address. Allow inbound TCP port 8000 on the private network in Windows Firewall. From the repository root, start the backend:

```bash
set "BACKEND_HOST=0.0.0.0"
set "BACKEND_PORT=8000"
set "CORS_ORIGINS=http://localhost:8501,http://192.168.1.25:8501"
python -m backend.main
```

Replace `192.168.1.25` with Laptop B's IPv4 address in `CORS_ORIGINS` if browser-origin requests are used. On **Laptop B (frontend)**, install the same project dependencies, set the backend laptop's address, allow inbound TCP port 8501 if accessing the UI from another device, and start Streamlit:

```bash
set "BACKEND_API_URL=http://192.168.1.20:8000"
python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port 8501
```

Replace `192.168.1.20` with Laptop A's IPv4 address. Open `http://<Laptop-B-IPv4>:8501` in a browser. Keep both laptops on a network that permits device-to-device traffic. `BACKEND_HOST`, `BACKEND_PORT`, `BACKEND_API_URL`, `CORS_ORIGINS`, `DATASET_ROOT`, `MODEL_DIRECTORY`, and `PERFORMANCE_RESULTS_PATH` are configurable through environment variables.

## Prediction And Evaluation Status

The prediction screen does not generate placeholder routes: it reports which adapters are missing until all three standardized integrations are connected. Model artifacts are inventoried without loading/deserializing them. Model-performance values are read only from a validated `outputs/performance.json` artifact; no metrics are currently present. Student-test reference routes are never used by prediction and test metrics remain blocked unless project rules and artifact provenance explicitly permit them.

Adapter payloads and evaluation artifact fields are documented in [INTEGRATION_CONTRACT.md](INTEGRATION_CONTRACT.md).