# Model Integration Audit

## 1. Governance-First Model Discovery (`registry.py` + `security.py`)
- `Medical_Inference_Application` **never creates a duplicate model registry** and **never exposes arbitrary training checkpoints**.
- Models are discovered from:
  1. **Layer 3 MLOps Governance API** (`GET {MLOPS_API_URL}/api/v1/inference/models` and `GET {MLOPS_API_URL}/api/v1/models/{model_id}/approved/{version}`), or
  2. **MLflow Model Registry** (cross-checked against Layer 3 approval), or
  3. **Local verified development catalogue** (`INFERENCE_LOCAL_CATALOGUE`, restricted to `INFERENCE_ENVIRONMENT=development`).
- Every model package directory is SHA-256 verified (`security.verify_directory`) before loading into `runtime.ModelCache`. Symlinks, path escapes, or modified weights are strictly rejected.

## 2. Preprocessing & Postprocessing Fidelity
- `PreprocessingSpec.from_manifest(manifest)` binds image size, color space, scale, resize mode, normalization, and channel order directly to the approved model manifest.
- Real PyTorch TorchScript (`inference_model.ts`) and TensorFlow SavedModel (`serving_model`) execution is supported across `classification`, `segmentation`, and `detection`.
