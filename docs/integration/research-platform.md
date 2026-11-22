# Connected research platform

The four repositories have distinct ownership: Dataset Pipeline publishes
datasets; Model Engineering trains and exports; MLOps registers and reviews;
Inference runs an approved version. Shared schemas are packaged in this
repository's `packages/contracts`, pinned by commit in consumers. The old local
`Medical_Inference_Application` directory was a contracts-only checkout; this
inference checkout is the source pushed to GitHub `Medical_Inference_Application`.

## Local GUI and APIs

Install each service's documented extras into its `.venv310` (Dataset Pipeline
may use `.venv-impl`). Install inference `.[test,frontend,pytorch]` and MLOps
`.[test,dashboard]`. Then, from the inference repository:

```bash
python3 scripts/start_research_platform.py --check
python3 scripts/start_research_platform.py
```

Dataset GUI/API: ports 8501/8000. Governance GUI/API: 8502/8081. Inference
GUI/API: 8503/8082. All bind to 127.0.0.1. No institutional SSO is required in
this explicit local mode. `X-Research-Actor` selects a local development actor;
use a different actor for independent review. This is an educational identity
simulation, not identity assurance. Normal deployed API entry points retain
OIDC authentication. Ctrl-C stops the launched processes. Cloud jobs run only
when explicitly submitted through the dataset API or CLI.

## Handoffs

1. Discover a dataset using `GET /api/v1/published-releases`; pin its URI and
   receipt digest. Use the model `open_verified_release()` reader, which
   authenticates it and materializes checksum-verified assets.
2. Train using the frozen partitions. Save receipt and manifest digests with
   experiment ID, Git commit, preprocessing and class mapping. For inference,
   export TorchScript or TensorFlow SavedModel and include it before registering.
3. Submit the package with `POST /api/v1/models/register` and an idempotency key.
   The package must be beneath the shared artifact root. Request review, then
   use a different actor to approve. Changed package bytes invalidate approval.
4. Inference queries the governance API for that exact approved model version,
   checks its package digest, and stores prediction output and dataset/experiment
   lineage. An archived version stops serving on subsequent requests.

## Reproducible acceptance test

```bash
../medical_imaging_model/.venv310/bin/python packages/contracts/scripts/test_platform.py
```

The interpreter needs both ML frameworks plus the services' test dependencies.
Arguments allow alternative checkout names. Synthetic classification traverses
all four implementations for both frameworks. Segmentation and detection verify
publisher/reader compatibility; negatives remain in the detection inventory.
ASGI transport and synthetic authentication are test-specific; business logic,
training, exports, registry transitions and inference execution are real.

This local test does not certify clinical performance, historical dataset
review, production SSO, hosted GUI availability or AWS scale. Live institutional
OIDC, DNS/TLS and resource sizing are deployment configuration. Research use
alone does not supply those values.
