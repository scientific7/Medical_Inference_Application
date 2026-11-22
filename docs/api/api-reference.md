# API reference

The FastAPI application serves interactive OpenAPI documentation at `/docs`.
All routes except health and readiness require an OIDC bearer token.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health`, `/ready` | Process health and a governance reachability check |
| GET | `/api/v1/models` | Browse by task, framework, architecture, version, or approval |
| GET | `/api/v1/models/{model_id}/versions/{version}` | Exact version metadata |
| GET | `/api/v1/models/active` | Loaded versions in this process |
| POST | `/api/v1/models/load`, `/api/v1/models/unload` | Administrator cache operations with JSON model ID and version |
| POST | `/api/v1/predictions/{model_id}/{version}` | Run inference with the multipart field `image` |
| POST | `/api/v1/inference/batch` | Queue multipart fields `model_id`, `version`, and repeated `images` |
| GET | `/api/v1/inference/batch/{batch_id}` | Job state and individual request IDs |
| GET | `/api/v1/inference/history` | Paginated summaries filtered by task, model, state, and date |
| GET | `/api/v1/inference/history/{prediction_id}` | Detailed prediction metadata |
| GET | `/api/v1/evaluations` | Evaluation summaries enabled for display |
| GET | `/api/v1/system/metrics` | Requester-scoped counts and latency, with system-wide access for administrators |
| GET | `/api/v1/system/status` | Administrator dependency status and limits |

Prediction responses include a request ID, model identity, device, task,
latency, and output metadata. Segmentation returns row-major `mask_runs`
as pairs of class ID and run length in original-image coordinates.
Detection returns `boxes_xyxy` in pixels relative to the uploaded image.
Classification scores are labeled as uncalibrated.

The API uses FastAPI's `detail` error envelope. Handled prediction failures
include an `X-Request-ID` response header. Oversized requests return 413,
unsupported media return 415, and invalid images return 422. Registry
and runtime availability failures return 503.
