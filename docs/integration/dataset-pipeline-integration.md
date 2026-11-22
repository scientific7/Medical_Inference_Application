# Dataset pipeline integration

The training repository consumes verified dataset release receipts from the
dataset pipeline. The authenticated `GET /api/v1/published-releases` endpoint
publishes release IDs, dataset IDs, tasks, receipt URIs, and receipt digests.
The inference catalogue carries the release ID and receipt digest from the
approved model manifest and ordinary inference works without a live dataset
pipeline.

Set `INFERENCE_DATASET_API_URL` to the backend origin and configure
`INFERENCE_DATASET_API_TOKEN` if the catalogue requires a bearer token.
The model details endpoint surfaces the dataset ID only when the published
release matches the approved release ID, task, and receipt digest. Source
URIs stay private. An unavailable pipeline is reported in the model details
while the approved provenance remains visible.

The current upstream endpoint returns at most 250 sorted releases and has
no pagination or exact published-receipt lookup. A release outside that
window is reported as absent from the catalogue response.
