# Connected research platform

For the audited local target architecture, verified gaps, and phased acceptance
criteria, see the [local platform integration audit](../architecture/local-platform-integration-audit.md).

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

Inference uses Layer 3's approved model catalogue when neither MLflow nor a
local catalogue file is configured. The loopback launcher opens the inference
workstation with its local research identity. Model Explorer can retrieve
cards from verified approved packages and show their supported image formats.

## Live workstation acceptance

`tests/test_loopback_workstation.py` starts separate governance, inference,
and Streamlit processes with temporary SQLite databases and package storage.
It drives Chrome through upload, visualization, mask download, and history.
Install Playwright in the test environment and have Chrome installed.
Set `INFERENCE_BROWSER_PYTHON` to a Python 3.10 interpreter with inference,
Streamlit, and PyTorch installed. Set `INFERENCE_BROWSER_GOVERNANCE_PYTHON`
to the governance interpreter if its dependencies are in a separate
environment. The pytest interpreter also needs PyTorch and the Model
Engineering package on its import path.

```bash
INFERENCE_BROWSER_PYTHON="$PWD/.venv310/bin/python" \
INFERENCE_BROWSER_GOVERNANCE_PYTHON="$PWD/../MLOps-and-model-governance/.venv310/bin/python" \
python -m pytest -q tests/test_loopback_workstation.py
```

The detection fixture uses Model Engineering's actual YOLO-grid exporter
with deterministic synthetic weights. The producer checks eager-versus-reloaded
TorchScript parity, and the workstation verifies governance approval and
original-image coordinates. The Faster R-CNN fixture also uses the Model Engineering exporter. It
preserves native image sizes and validates two uploads with different
detection counts against the eager model. The runtime requires Torchvision
in addition to PyTorch. RetinaNet and DETR exports remain outside this
verified handoff.

### CUDA acceptance

Run `bash scripts/check_cuda_acceptance.sh` on a Linux host with an NVIDIA
CUDA device. Set `INFERENCE_BROWSER_PYTHON` to an environment with CUDA-enabled
PyTorch and Torchvision, inference and test dependencies, and Playwright.
Install Chrome and provision the companion Model Engineering and governance
checkouts. Set `INFERENCE_BROWSER_GOVERNANCE_PYTHON` if governance uses a
separate interpreter. The preflight requires GPU availability, CUDA NMS and
Chrome before it runs the TorchScript device tests and governed browser flow.
The browser test compares saved Faster R-CNN output with eager CUDA output
and checks that history reports `cuda:0`.

The manual GitHub Actions job targets a trusted self-hosted Linux x64 runner
labelled `cuda-inference`. Configure repository variables
`INFERENCE_CUDA_PYTHON` and `INFERENCE_CUDA_COMPANION_ROOT` with absolute paths.
The companion root must contain compatible, pinned checkouts of
`medical_imaging_model` and `MLOps-and-model-governance`.
Set `INFERENCE_CUDA_GOVERNANCE_PYTHON` for a separate governance environment.
The workflow must be available on the repository's default branch before
GitHub can dispatch it against `integration/research-platform`.
