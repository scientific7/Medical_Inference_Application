# Inference Feature Matrix

Classification key: **Implemented & Tested**, **Implemented (Untested)**, **Partially Implemented**, **Missing**, **Blocked (External)**.

| # | Feature Area | Baseline Status | Control Center Extension | Backend API | GUI Section |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | **Dashboard (Live KPIs & Charts)** | Partially Implemented (`/api/v1/system/metrics`) | Full persisted KPIs, latency/volume/task charts, activity | `GET /api/v1/dashboard` | `A. Dashboard` |
| 2 | **Model Catalogue (Governance-Sourced)** | Implemented & Tested (`/api/v1/models`) | Search/filter by task/modality/arch, approval & card drawer | `GET /api/v1/catalogue` | `B. Model Catalogue` |
| 3 | **Inference Workspace (Single/Batch/Async)** | Partially Implemented (`/api/v1/predictions/...`) | Persistent async job queue, cancel, retry, real TorchScript | `POST /api/v1/inference`, `/cancel`, `/retry` | `C. Inference Workspace` |
| 4 | **Image Input & Validation (PHI Guard)** | Partially Implemented (`PNG/JPEG/TIFF/WEBP`) | Upload store + DICOM/NIfTI header inspection + PHI redaction | `POST/GET /api/v1/uploads` | `D. Image Input` |
| 5 | **Classification Module** | Implemented & Tested (`postprocessing.py`) | Probability bar charts, top-k, uncalibrated score disclaimer | `POST /api/v1/inference` | `E. Classification` |
| 6 | **Segmentation Module** | Implemented & Tested (`postprocessing.py`) | Canvas mask overlay, opacity slider, class area stats | `POST /api/v1/inference` | `F. Segmentation` |
| 7 | **Detection Module** | Implemented & Tested (`postprocessing.py`) | Bounding boxes canvas, confidence slider, class table | `POST /api/v1/inference` | `G. Detection` |
| 8 | **Interactive Medical Image Viewer** | Partially Implemented (Streamlit static PIL) | HTML5 Canvas viewer: zoom, pan, window/level, side-by-side, overlays | `GET /api/v1/uploads/{id}/preview` | `J. Results & Visualization` |
| 9 | **Inference Job Management** | Partially Implemented (in-memory `BatchQueue`) | Persistent SQLite/PG `inference_jobs` + event timeline | `GET /api/v1/inference`, `/api/v1/inference/{job_id}` | `H. Inference Jobs` |
| 10 | **Inference History** | Implemented & Tested (`prediction_records`) | Searchable/filterable unified history + immutable provenance | `GET /api/v1/history` | `I. Inference History` |
| 11 | **Report Generation (PDF/HTML/JSON)** | Missing | Real evidence-backed reports with AI-prediction disclaimer | `GET /api/v1/reports/{job_id}` | `K. Reports` |
| 12 | **Performance Analytics** | Partially Implemented (`/api/v1/system/metrics`) | Per-model, per-task, per-profile latency & error breakdown | `GET /api/v1/analytics` | `L. Performance Analytics` |
| 13 | **AWS Compute Management** | Partially Implemented (Terraform + S3) | Live `ap-south-1` AWS Batch/EC2/GPU quota check + mock fallback | `GET /api/v1/compute`, `/api/v1/workers` | `M. AWS Compute` |
| 14 | **Monitoring & Alerts** | Partially Implemented (`telemetry.py`) | Persistent alert lifecycle (ack/assign/resolve) + Layer 3 forward | `GET/POST/PATCH /api/v1/alerts`, `/api/v1/monitoring` | `N. Monitoring & Alerts` |
| 15 | **Integration Center** | Missing | Live probes to Layers 1, 2, 3, 5 + AWS health | `GET /api/v1/integrations` | `O. Integration Center` |
| 16 | **Administration & RBAC** | Partially Implemented (`security.py`) | User/role table, audit & retention settings | `GET/POST /api/v1/admin/users` | `P. Administration` |
| 17 | **Settings** | Missing | Masked env config, AWS mode, service URLs | `GET/PUT /api/v1/settings` | `Q. Settings` |
