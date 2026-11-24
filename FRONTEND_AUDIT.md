# Frontend Audit — Medical Inference Application

## 1. Existing Frontend (`apps/frontend/app.py`)
- Streamlit-based UI (`apps/frontend/app.py`, `client.py`, `visualizations.py`) requiring a separate Streamlit server process (`port 8501`).
- Provides basic single-image prediction upload, model card display, and static PIL overlays.

## 2. Gaps Addressed by Standalone Inference Control Center GUI (`src/medical_imaging_inference/web/index.html`)
- Served directly by FastAPI at `/` and `/inference-center` with zero extra build tooling or separate frontend process required.
- Implements all **17 navigation sections** (`A. Dashboard` through `Q. Settings`) in a responsive dark/light medical imaging theme.
- **Interactive HTML5 Canvas Medical Image Viewer:** Zoom, Pan, Window/Level (brightness/contrast), Side-by-Side Original vs Prediction, Segmentation Mask overlay with Opacity slider, Detection Bounding Box overlay with Confidence Threshold slider, and Classification probability charts.
- Preserves existing Streamlit frontend untouched for backward compatibility.
