# Backend Audit — Medical Inference Application

## 1. Existing FastAPI Core (`src/medical_imaging_inference/app.py`)
- **Verified Working (43 passing tests):**
  - Health probes: `GET /health`, `GET /live`, `GET /ready`.
  - Synchronous prediction: `POST /api/v1/predictions/{model_id}/{version}` with upload limits (`20 MB`, `40M px`), concurrency semaphore, and deadline timeout.
  - Batch queue: `POST /api/v1/inference/batch`, `GET /api/v1/inference/batch/{batch_id}`.
  - Model lifecycle & cache: `GET /api/v1/models/active`, `POST /api/v1/models/load`, `POST /api/v1/models/unload`, `GET /api/v1/models/approved`, `GET /api/v1/models`.
  - History & telemetry: `GET /api/v1/inference/history`, `GET /api/v1/system/metrics`, `GET /api/v1/system/status`.

## 2. Gaps Addressed by Control Center Backend (`control_center_db.py` + `control_center_api.py`)
1. **Persistent Job Store:** Replace ephemeral-only batch tracking with relational `inference_jobs` and `inference_job_events` supporting `pending`, `queued`, `starting`, `running`, `completed`, `failed`, `cancelled`, plus `cancel` and `retry`.
2. **Upload & Validation Store:** `POST /api/v1/uploads` validates PNG/JPEG/TIFF/WEBP/DICOM/NIfTI headers, strips PHI identifiers from logs, persists metadata + SHA-256, and serves preview bytes for the interactive viewer.
3. **Report Generator:** Generates PDF (`%PDF-1.4`), HTML, and JSON inference reports explicitly labeling all outputs as AI model predictions.
4. **Alerts & Compute Management:** Persists operational alerts with acknowledgement/resolution and integrates with AWS `ap-south-1` (`medtriage-admin`) with graceful local mock fallback.
