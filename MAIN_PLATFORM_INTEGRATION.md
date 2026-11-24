# Main Platform Integration Contract (`scientific7/medical-imaging-platform-`)

## 1. Standalone Mode vs. Integrated Mode
- **Standalone Mode:** `Medical_Inference_Application` runs independently (`uvicorn medical_imaging_inference.app:app --port 8002`) with its own Control Center GUI at `/inference-center` and `/`.
- **Integrated Mode:** `Medical_Imaging_Platform` (Layer 5 BFF on `:8080`) coordinates Layer 4 via documented REST contracts without duplicating the inference engine or model loader.

## 2. Stable REST Contracts Exposed for Layer 5 & Layer 3
| Endpoint | Method | Consumer | Purpose |
| :--- | :---: | :--- | :--- |
| `/health`, `/live`, `/ready` | `GET` | Layer 5, K8s/ALB | Health & readiness probes |
| `/api/v1/catalogue` & `/api/v1/models` | `GET` | Layer 5 | Approved governance-verified models |
| `/api/v1/predictions/{model_id}/{version}` | `POST` | Layer 5 | Synchronous single-image inference |
| `/api/v1/inference` & `/api/v1/inference/{job_id}` | `POST/GET` | Layer 5 | Persistent async inference jobs |
| `/api/v1/history` & `/api/v1/results` | `GET` | Layer 5 | Searchable immutable prediction history |
| `/api/v1/reports/{job_id}?format=pdf\|html\|json` | `GET` | Layer 5 | Inference report downloads |
| `/api/v1/system/metrics` & `/api/v1/monitoring` | `GET` | Layer 3, Layer 5 | Runtime latency, errors, worker health |
| `/api/v1/models/load` & `/api/v1/models/unload` | `POST` | Layer 3 | Authorized model activation / rollback |
