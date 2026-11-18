# ML-Hackathon-Map-Matching

## Project Overview

This repository is the starting structure for a four-member machine learning hackathon project focused on GPS trajectory map matching. The project will organize data preparation, candidate generation, model experimentation, route reconstruction, and a user-facing demonstration. The initial repository contains configuration and placeholders only; no algorithms or application workflows have been implemented.

## Problem Domain

GPS trajectory map matching associates noisy, ordered GPS observations with the road network they most likely represent. The planned work will explore candidate road segments, model-based scoring, and route reconstruction while keeping the original observation order and road-arc identifiers consistent.

## Technology Stack

- Python
- FastAPI and Uvicorn for the proposed backend API
- Streamlit, Folium, and Streamlit-Folium for the proposed interactive map interface
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
│   ├── main.py
│   ├── api.py
│   ├── config.py
│   ├── services/.gitkeep
│   ├── integrations/.gitkeep
│   └── schemas/.gitkeep
├── frontend/
│   ├── app.py
│   ├── pages/.gitkeep
│   ├── components/.gitkeep
│   └── services/.gitkeep
├── models/
│   └── README.md
├── outputs/
│   └── README.md
└── tests/
	└── .gitkeep
```

## Installation

Installation instructions are a placeholder until the team confirms its development environment. The initial dependencies are listed in `requirements.txt`.

```bash
python -m venv .venv
```

## Execution

Execution instructions are a placeholder. The backend and frontend entry-point files are intentionally empty except for module docstrings, so there is no runnable application yet.