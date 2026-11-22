# Inference integration

## Ownership

The inference service owns input validation, prediction requests/results and inference metadata. It does not import model training modules or load arbitrary Python checkpoints. The MLOps service remains authoritative for model approval. Packages are read from a private shared mount in development or an approved S3 digest prefix in production.

See [approved model flow](approved_model_flow.mmd) for the editable integration diagram.

## Runtime configuration

Use Python 3.10 and install `.[test,pytorch]`, `.[test,tensorflow]`, or both framework extras. Configure:

- `OIDC_ISSUER`, `OIDC_JWKS_URI`, `OIDC_AUDIENCE`, optional `OIDC_ROLES_CLAIM`.
- `MLOPS_API_URL`, `MLOPS_SERVICE_TOKEN` for scoped approved-model lookup.
- `INFERENCE_ARTIFACT_ROOT`, mounted read-only to the same package root used by MLOps.
- `INFERENCE_DATABASE_PATH` for local prediction metadata.

Required user roles are `inference_user` or `inference_admin`; the service identity needs `inference_reader` at MLOps. SSO is configuration-driven; no identity-provider endpoint or credentials are checked into the repository.

## API

- `GET /health`: process health.
- `GET /ready`: checks MLOps registry reachability.
- `POST /api/v1/predictions/{model_id}/{version}`: multipart image; resolves exactly that approved version and returns predictions.
- `GET /api/v1/predictions/{prediction_id}`: fetches a requester's own prediction; `inference_admin` can access all rows.

Uploads are limited to 20 MiB and 40 megapixels and accepted types depend on the model input profile. The API verifies the registry artifact digest and weight digest, applies package resize/RGB/scale settings, then executes checked TorchScript or TensorFlow SavedModel exports. Prediction metadata stores input SHA-256 but not the image bytes or patient identifiers. Output includes a research-only disclaimer.

## Current limits

Live AWS, MLflow, governance, and cross-framework deployment validation
requires configured service identities and approved packages. Retention
automation and calibration remain pending. TorchScript and SavedModel
exports are required, and architecture-specific parity remains unverified
across the full upstream catalogue.

## Validated preprocessing profiles

The legacy classification and segmentation manifest emitted by training contains
`image_size` in height–width order, `color: RGB`, and `scale: "[0,1]"`.
The inference adapter maps it to the observed training-reader behavior of
bilinear stretch resizing, raw image orientation, and CHW for PyTorch or HWC
for TensorFlow. This profile accepts single-frame JPEG and PNG uploads.

Other packages must provide an explicit `preprocessing` object with
`version: "1"`, `image_size`, `color`, `scale`, `resize`,
`aspect_ratio: stretch`, `orientation: raw`, `channel_order`, and
`formats`. An optional `mean` and `std` pair is applied after scaling
and in the declared color order. TIFF and WebP require explicit opt-in.
Uploads must match their extension, declared media type, and decoded format.
Unsupported profiles fail before framework execution.

## Runtime lifecycle

The process caches two loaded exports by model ID, version, and verified package
digest. Each prediction still resolves current approval with MLOps. The
`/api/v1/models/active` route lists cached versions and integrity status.
Administrators can preload or evict an exact version with
`POST /api/v1/models/load` and `POST /api/v1/models/unload`.
Eviction releases the cache reference while existing inference retains its
model reference. Subsequent requests can reload an approved model.
`INFERENCE_DEVICE` accepts `cpu` (default) or `gpu`; an unavailable requested
GPU causes a load error. Cache state belongs to the serving process.

## Registry discovery

Set `MLFLOW_TRACKING_URI` to the MLflow registry HTTP origin and optionally
`MLFLOW_TRACKING_TOKEN` to its bearer credential. The catalogue reads the
paginated MLflow model-version search endpoint. Use model-version tags
`governance.model_id` and `governance.version` when MLflow identifiers differ
from governance identifiers. Otherwise the MLflow name and numeric version
are used for exact lookup. Current sibling tracking code logs runs. A release
operator must publish model-version entries and their identifier mapping for
them to appear in the catalogue.

For development without MLflow, set `INFERENCE_LOCAL_CATALOGUE` to a JSON array
of objects containing `model_id` and `version`. Both discovery profiles use the
same independent exact-version governance check. An absent approved record
appears as ineligible. A connection or service authentication failure returns
HTTP 503. Private artifact paths and MLflow sources are omitted from responses.

The authenticated `GET /api/v1/models` route supports task, framework,
architecture, version, and approval filters. Append `/{model_id}/versions`
for a model's versions or `/{model_id}/versions/{version}` for exact metadata.

## Batch inference

Submit multipart fields `model_id`, `version`, and repeated `images` to
`POST /api/v1/inference/batch`. A 202 response contains the batch ID. Poll
`GET /api/v1/inference/batch/{batch_id}` for status and individual request IDs.
Jobs can finish as completed, partial, or failed. Requesters and administrators
can retrieve batch status.

The current queue supports eight files and 40 MiB per job, four outstanding
jobs, and two concurrent inference operations per process. Upload bytes are
held in memory for the job. Batch status is process-local and bounded to 64
entries. Prediction metadata remains in the database. Restarting the process
discards in-flight work and batch status.

## Evaluation and operations

The training package supplies an evaluation summary derived from its test
report. The API exposes approved-model summaries at `/api/v1/evaluations`
only when operators explicitly set `INFERENCE_DISPLAY_TEST_METRICS=true`.
It returns numeric metrics and provenance fields. The workstation shows
side-by-side comparisons only when task, receipt digest, test manifest
digest, split, and evaluation protocol digest all match. Current training
packages lack the protocol digest, so they appear as individual reports.
Confusion matrices and example visualizations require additional publication
metadata and a display policy.

The administrator-only `/api/v1/system/status` route provides live database
and registry reachability, loaded model status, deployment version, and
configured upload and concurrency limits.

## Output profiles

Packages can declare a version 1 `postprocessing` object in the verified
manifest. Classification supports `classification_mode` values
`multiclass`, `binary` (one positive-class logit), and `multilabel`
(one logit per class). Set `top_k` and `threshold` to control ranked and
selected labels. Scores are uncalibrated. For binary segmentation,
`threshold` sets the foreground cutoff in score space.

Detection accepts decoded `boxes`, `labels`, and `scores`. Set
`detection_format` to `xyxy_pixels`, `xyxy_normalized`, or
`yxyx_normalized` to identify the coordinate convention. Optional
`threshold` and `iou_threshold` enable score filtering and class-wise
overlap suppression. Exports must supply decoded outputs in this mapping.
Raw architecture heads require a compatible export or a dedicated decoder.
