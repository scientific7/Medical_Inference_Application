# Missing Functionalities Identified During P0 Audit

1. **Standalone Unified Web GUI (17 Sections):** Only a minimal separate-process Streamlit app existed; no FastAPI-served Control Center GUI with interactive Canvas medical image viewer.
2. **Persistent Inference Job Queue (`inference_jobs`):** Existing `BatchQueue` was in-memory only and lost job state on restart; lacked `cancel`, `retry`, and job event timeline.
3. **Upload & Multi-Format Validation Store:** No `/api/v1/uploads` endpoint supporting preview bytes, DICOM/NIfTI metadata validation, and PHI tag redaction.
4. **Inference Report Generation:** No PDF/HTML/JSON report endpoint attaching model version, preprocessing contract, predictions, and AI-prediction disclaimers.
5. **AWS Compute Management & Alerts UI/API:** No `/api/v1/compute`, `/api/v1/workers`, or persistent `/api/v1/alerts` with acknowledge/resolve workflow and Layer 3 MLOps forwarding.
6. **Integration Center & Administration:** No cross-repository connectivity dashboard (Layers 1, 2, 3, 5 + AWS) or RBAC/settings management endpoints.
7. **Live PyTorch Runtime in Local `.venv`:** PyTorch (`torch==2.5.1`) was not installed in `.venv`, leaving real TorchScript inference tests skipped until installed.
