# Environment configuration

| Variable | Purpose |
| --- | --- |
| `INFERENCE_ENVIRONMENT` | Use `production` for deployed services, which require PostgreSQL. |
| `OIDC_ISSUER`, `OIDC_JWKS_URI`, `OIDC_AUDIENCE` | Verify signed API access tokens. Use HTTPS endpoints in production. |
| `OIDC_ROLES_CLAIM` | Role claim name, default `roles`. |
| `MLOPS_API_URL`, `MLOPS_SERVICE_TOKEN` | Governance approval origin and scoped service credential. |
| `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_TOKEN` | MLflow registry origin and optional scoped credential. |
| `INFERENCE_LOCAL_CATALOGUE` | Development JSON catalogue path. |
| `INFERENCE_S3_ARTIFACT_ROOT`, `INFERENCE_S3_CACHE_ROOT` | S3 model prefix and private local cache directory. |
| `INFERENCE_ARTIFACT_ROOT` | Private package mount for local approved artifacts. |
| `INFERENCE_DATABASE_URL` | PostgreSQL URL using the psycopg driver. |
| `INFERENCE_DATABASE_PATH` | Development SQLite file path. |
| `INFERENCE_DEVICE` | `cpu` or `gpu`. Defaults to CPU. |
| `INFERENCE_DISPLAY_TEST_METRICS` | Set to `true` only when approved test metrics may be displayed. |
| `INFERENCE_API_URL` | Backend origin used by Streamlit. |

Keep credentials in the deployment secret store or a restricted local
environment file. The image build context only includes application files
and packaging metadata. See [Docker deployment](docker-deployment.md) for
Compose-specific settings.
