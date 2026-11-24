# Medical Inference Application — Architecture Audit

**Repository:** `scientific7/Medical_Inference_Application` (Layer 4)  
**Audit Date:** 2026-11-24  
**Baseline Test Run:** `43 passed, 13 skipped in 1.78s` (Python `3.10.22`)

## 1. Ecosystem Separation of Responsibilities

| Repository | Layer | Owned Responsibilities | Layer 4 Boundary |
| :--- | :---: | :--- | :--- |
| `Medical_Dataset_Pipeline` | Layer 1 | Ingestion, validation, EDA, splits, dataset releases | Read-only provenance & sample metadata |
| `medical_imaging_model` | Layer 2 | Model architectures, training, Optuna HPO, MLflow runs | Consumes approved exported artifacts via Layer 3 |
| `MLOps-and-model-governance` | Layer 3 | Model registry, versions, approval workflows, risk, release manifests | Source of truth for approved models; receives runtime telemetry |
| `Medical_Inference_Application` | Layer 4 | **Model selection, image input/validation, preprocessing, real PyTorch/TF & AWS compute inference, classification/segmentation/detection visualization, async jobs, history, reports, runtime monitoring & alerts** | **Owns standalone Inference Control Center GUI & backend** |
| `Medical_Imaging_Platform` | Layer 5 | Unified cross-repository UI & BFF coordination | Integrates with Layer 4 APIs in integrated mode |

## 2. Existing Architecture Inventory
- **Backend:** FastAPI (`src/medical_imaging_inference/app.py`, 804 LOC) with upload size middleware (`UploadLimits`), bounded concurrency (`asyncio.Semaphore`), and strict deadline enforcement.
- **Runtime:** `runtime.py` executes real PyTorch TorchScript (`inference_model.ts`) and TensorFlow SavedModel (`serving_model`) with LRU `ModelCache` keyed by `(model_id, version, artifact_digest)`.
- **Contracts:** `preprocessing.py` (`PreprocessingSpec`) and `postprocessing.py` (`classification`, `segmentation`, `detection` with NMS & RLE masks).
- **Persistence:** `audit.py` (SQLAlchemy SQLite/PostgreSQL `prediction_records`) + `batches.py` (in-memory `BatchQueue`).
- **Frontend:** Existing Streamlit UI (`apps/frontend/app.py`); extended with a standalone 17-section FastAPI-served HTML5/Canvas Inference Control Center GUI at `/inference-center` and `/`.
