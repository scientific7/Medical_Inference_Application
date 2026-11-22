# Inference flow

After HTTP request-length and access checks, the backend resolves the
requested model version with governance, obtains a verified export,
validates and transforms images, executes the model, and maps results
back to the uploaded image. It stores metadata and returns predictions.
Batch submission queues bounded work and polls per-item request IDs.

[Inference sequence](../diagrams/inference-sequence.mmd)
