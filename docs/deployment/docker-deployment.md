# Docker deployment

Copy `.env.example` to `.env` and fill in OIDC and MLOps settings. Set
`FRAMEWORK_EXTRAS=,pytorch` or `FRAMEWORK_EXTRAS=,tensorflow` in the Compose
environment to include the selected framework. The default build supports API
and UI smoke checks with no model runtime. Set `INFERENCE_ARTIFACT_HOST_PATH`
to a private package mount readable by container UID 10001, or configure the S3
artifact root. Governance paths for local packages must reference the same
container-visible location.

To validate the example configuration directly, run
`INFERENCE_ENV_FILE=.env.example docker compose --env-file .env.example config -q`.

Run `docker compose up --build`. Open the frontend on loopback port 8501
and the API docs on loopback port 8000. The Compose stack uses embedded SQLite
for local history. Configure the PostgreSQL URL for an external database and
run `docker compose run --rm backend python -m medical_imaging_inference.audit`
with migration credentials before serving traffic. MLflow and governance
are external services.

For production, build the backend and frontend targets separately. Pin the
published image digests in the deployment configuration. Terminate HTTPS at
the ingress and route traffic to the frontend over a private connection.
Pass service credentials to the backend through deployment-managed
configuration. Run a single backend process while the cache and batch queue
are process-local.
