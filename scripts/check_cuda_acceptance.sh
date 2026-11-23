#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${INFERENCE_BROWSER_PYTHON:?Set the CUDA-enabled inference Python interpreter}"
export INFERENCE_BROWSER_PYTHON
export INFERENCE_BROWSER_DEVICE=cuda
export PYTHONPATH="$PWD/src:$PWD/packages/contracts/src:$PWD/../medical_imaging_model/src${PYTHONPATH:+:$PYTHONPATH}"

"$INFERENCE_BROWSER_PYTHON" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(f"CUDA acceptance requires Python 3.10, got {sys.version.split()[0]}.")

import torch
import torchvision
from playwright.sync_api import sync_playwright

if torch.__version__.split("+")[0] != "2.5.1":
    raise SystemExit(f"Expected PyTorch 2.5.1, got {torch.__version__}.")
if torchvision.__version__.split("+")[0] != "0.20.1":
    raise SystemExit(f"Expected Torchvision 0.20.1, got {torchvision.__version__}.")
if not torch.cuda.is_available():
    raise SystemExit("CUDA acceptance requires an available CUDA device.")
boxes = torch.tensor([[0., 0., 1., 1.]], device="cuda")
torchvision.ops.nms(boxes, torch.ones(1, device="cuda"), 0.5).cpu()
with sync_playwright() as driver:
    driver.chromium.launch(channel="chrome").close()
print("CUDA and Torchvision NMS ready:", torch.cuda.get_device_name())
PY

if [[ -n "${INFERENCE_BROWSER_GOVERNANCE_PYTHON:-}" ]]; then
  "$INFERENCE_BROWSER_GOVERNANCE_PYTHON" - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit("The governance test interpreter must be Python 3.10.")
PY
fi

"$INFERENCE_BROWSER_PYTHON" -m pytest -q \
  'tests/test_runtime.py::test_torch_export_runs_on_requested_device[cuda]' \
  'tests/test_runtime.py::test_approved_detection_export_runs_on_requested_device[cuda]' \
  tests/test_loopback_workstation.py
