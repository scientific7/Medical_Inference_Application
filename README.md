# Medical Imaging Inference

For the connected local research GUI and four-repository workflow, see
[research platform setup](docs/integration/research-platform.md).
Start all three GUIs with `python3 scripts/start_research_platform.py`.
The shared Python contracts package is maintained in `packages/contracts`;
consumers pin its Git commit. Successful predictions retain the experiment,
dataset receipt and manifest digests in prediction history.

Python 3.10 API for exact approved model versions from the MLOps registry. It validates upload size, type and readability, verifies package and weight digests, applies package preprocessing, supports classification/segmentation/detection output adapters, and records output metadata plus an input checksum without retaining image bytes or patient identifiers.

Shared interfaces are owned by the [platform contracts repository](https://github.com/scientific7/medical-imaging-platform-contracts).

## Local run

Create a Python 3.10 environment and install `-e '.[test]'`. Set `OIDC_ISSUER`, `OIDC_JWKS_URI`, `OIDC_AUDIENCE`, `MLOPS_API_URL`, `MLOPS_SERVICE_TOKEN`, and `INFERENCE_ARTIFACT_ROOT` to the shared private model package mount, then run `uvicorn medical_imaging_inference.app:app`. The MLOps service token must be scoped to approved-model discovery.

The API is documented at `/docs`; `/health` is process health and `/ready` checks model-registry reachability. `POST /api/v1/predictions/{model_id}/{version}` accepts a supported image upload and returns a structured, research-only prediction.

The runtime accepts numerically checked TorchScript (`inference_model.ts`) or TensorFlow SavedModel (`serving_model/`) packages. It rejects arbitrary Python checkpoints. Use the S3 and PostgreSQL adapters for production deployments. The local development profile uses the private package mount and SQLite.

## Streamlit workstation

Install the frontend extra with `pip install -e '.[frontend]'`. Start the API
and run `streamlit run apps/frontend/app.py`. The dashboard uses live API data
for model discovery, loaded versions, request history, and usage metrics.
The inference page previews an upload and displays task-specific predictions.
See the [frontend guide](apps/frontend/README.md).

## Docker

Build and run the local stack with Docker Compose. The stack includes FastAPI
and Streamlit with local SQLite history by default. See the
[Docker deployment guide](docs/deployment/docker-deployment.md).

## Documentation

- [System architecture](docs/architecture/system-architecture.md)
- [Local platform integration audit](docs/architecture/local-platform-integration-audit.md)
- [Model and dataset integration](docs/INTEGRATION.md)
- [API reference](docs/api/api-reference.md)
- [AWS deployment and rollback](docs/deployment/aws-deployment.md)
- [Medical data privacy](docs/security/medical-data-privacy.md)
- [Operations and monitoring](docs/operations/monitoring.md)

## Verification

Install `.[test,frontend,postgres,aws]` and run `pytest -q`. Set
`INFERENCE_TEST_POSTGRES_URL` to an isolated test database to include
PostgreSQL integration. Framework loader and AWS deployment checks require
the relevant environment and approved test packages.

This is a research workstation. Read the privacy and deployment guides
before using sensitive uploads. Evaluation comparisons require verified
dataset and protocol identifiers. Batch state is process-local.
