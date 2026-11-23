# CUDA acceptance runner

The optional CUDA workflow is a manually dispatched acceptance check. The workflow
file must be committed and pushed to `integration/research-platform` before GitHub
can list or run it. It does not run on pull requests and only runs on that branch.
This keeps CUDA hardware and its self-hosted runner out of untrusted PR execution.

## Runner requirements

Register a GitHub Actions self-hosted runner for this repository with all labels:

- `self-hosted`
- `linux`
- `x64`
- `cuda-inference`

The host must have an NVIDIA GPU and a driver compatible with the installed CUDA
PyTorch wheel, Chrome, and enough disk space for Python environments and synthetic
test artifacts. Keep the runner dedicated to this repository or otherwise isolate
its filesystem and credentials. The workflow does not need AWS credentials or
medical data.

Prepare a Python 3.10 inference environment that can import this repository's
test/application dependencies, PyTorch 2.5.1, Torchvision 0.20.1, and
`playwright.sync_api`. Install the Chrome browser expected by Playwright's
`channel="chrome"` selection. The acceptance script checks Python and framework
versions, CUDA availability, and Torchvision NMS before running tests.

Provide a local companion root containing these sibling checkouts:

```text
<companion-root>/medical_imaging_model/
<companion-root>/MLOps-and-model-governance/
```

The loopback test uses synthetic model artifacts and temporary local databases.
The companion checkouts must include the supported integration interfaces and be
kept at compatible revisions. Do not place credentials or private patient data in
these checkouts.

## Repository Actions variables

In the repository's **Settings → Secrets and variables → Actions → Variables**, set:

| Variable | Value |
| --- | --- |
| `INFERENCE_CUDA_COMPANION_ROOT` | Absolute path to the parent directory containing both companion checkouts. |
| `INFERENCE_CUDA_PYTHON` | Absolute path to the Python 3.10 interpreter with inference, CUDA PyTorch, Torchvision, and Playwright dependencies. |
| `INFERENCE_CUDA_GOVERNANCE_PYTHON` | Optional Python 3.10 interpreter with the governance service dependencies; omit it to reuse the inference interpreter. |

These values are paths on the self-hosted runner, not paths on a GitHub-hosted
machine. The workflow checks that both companion directories exist before linking
them into the checkout.

## Run the acceptance workflow

After the runner is online and variables are configured, open **Actions → CUDA
acceptance → Run workflow**, select `integration/research-platform`, and start it.
The workflow checks CUDA and Torchvision NMS, runs the CPU/CUDA runtime checks,
approved synthetic detection-package execution, and the loopback browser workflow.
It fails instead of reporting a pass if CUDA is unavailable or versions do not
match.

The workflow has not been run until GitHub reports a completed run. MPS testing on a
developer workstation is separate evidence and does not establish CUDA support.
