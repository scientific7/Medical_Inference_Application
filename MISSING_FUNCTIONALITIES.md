# Missing Functionalities — Medical_Inference_Application (Updated Traceability)

All original P0 gaps plus cross-repository enforcement gaps have been tracked and updated with execution evidence (`94d2843` + local updates).

| Req ID | Original Requirement | Previous Status | Current Status | Implementation & Test Evidence | Remaining Limitation |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `INFER-001` | Standalone 17-Section Control Center GUI & Interactive Canvas Medical Image Viewer | MISSING | COMPLETED AND VERIFIED | `src/medical_imaging_inference/web/index.html` + live browser verification on `:8002/inference-center` | None |
| `INFER-002` | Persistent Async Inference Job Queue (`inference_jobs`), Upload Store & PHI Redaction | MISSING | COMPLETED AND VERIFIED | `control_center_db.py` + `control_center_api.py` (`52 passed, 0 failed`) | None |
| `INFER-003` | Real PyTorch 2.5.1 TorchScript Classification, Segmentation & Detection + Reports (PDF/HTML/JSON) | PARTIAL (PyTorch missing in `.venv`) | COMPLETED AND VERIFIED | Installed `torch==2.5.1`; `tests/test_control_center.py` + live browser & 5-repo E2E verified real TorchScript forward passes | TensorFlow/Postgres optional tests (`6 skipped`) |
| `INFER-004` | Layer 3 MLOps Release Suspension Enforcement (`_check_governance_suspension`) | MISSING | COMPLETED AND VERIFIED | Added `_check_governance_suspension` in `control_center_api.py` querying Layer 3 `/api/v1/inference/eligibility/{id}/{v}` | None |
| `INFER-005` | Segmentation PNG Mask Export, Structured Detection JSON Export & Batch Image Upload (`POST /api/v1/uploads/batch`, `GET /api/v1/inference/{id}/export`) | MISSING | COMPLETED AND VERIFIED | Implemented in `control_center_api.py` and verified in `tests/test_control_center.py` (real `\x89PNG` mask bytes + 2-file batch upload) | None |
| `INFER-006` | AWS GPU Inference (`g5.xlarge`) | BLOCKED | BLOCKED (AWS Quota + SSO) | AWS Batch CPU inference previously verified (`31a9f5e5-2b59-4941-9da2-d06247dd3f02`, `SUCCEEDED`); EC2 G/VT quota `0.0 vCPUs` & SSO expired | Requires AWS GPU quota increase + `aws sso login --profile medtriage-admin` |
