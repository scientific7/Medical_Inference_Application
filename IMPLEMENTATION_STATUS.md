# Implementation status

**Repository** medical-imaging-inference-platform, Layer 4
**Current phase** P4 CUDA runner acceptance; container CI follow-up
**Current branch** `integration/research-platform`

## Current milestone — Detection package CI coverage and CUDA runner gate

1. **Original requirements.** Run the PyTorch runtime and approved detection
   package tests on CUDA, include the browser-based approved-model flow, and keep
   container builds working with the pinned contracts dependency.
2. **Implementation status.** The approved synthetic detection-package test now
   covers CPU, CUDA, and MPS. The PyTorch CI matrix runs the runtime suite on CPU.
   The CUDA workflow is pushed to this branch but is not listed by GitHub until it
   is present on the repository's default branch. The Docker image now installs its
   Git dependency in builder stages only. No CUDA runner is configured.
3. **Modified files.** `tests/test_runtime.py` checks approved package integrity,
   loading, and detections on requested devices. `.github/workflows/ci.yml` adds the
   runtime suite to the PyTorch job. `.github/workflows/cuda-acceptance.yml` and
   `scripts/check_cuda_acceptance.sh` provide manual CUDA/browser acceptance.
   `Dockerfile` isolates Git in dependency-build stages. `docs/deployment/cuda-runner.md`
   documents setup. This report records remaining external dependencies.
4. **Testing.** Full local suite: 46 passed, 4 skipped, zero failures. Targeted
   runtime suite: 7 passed, 1 skipped. MPS executed; CUDA skipped. Script syntax,
   workflow YAML parsing, documentation links, and `git diff --check` passed.
   The push at `117b065` triggered CI: Terraform, unit, PyTorch/TensorFlow framework,
   lint, frontend, integration, and docs jobs passed. Docker initially failed because
   the slim image lacked Git for the pinned direct Git dependency. Follow-up commit
   `33a6a13` moved Git into dependency-builder stages. Both push and pull-request CI
   runs for that commit passed, including container build, startup, and health checks.
   GitHub currently lists only CI, with no CUDA runner or repository Actions variables.
5. **Functional evidence.** Approved synthetic detection output, package digest
   verification, and device placement passed on CPU and MPS. CUDA is unverified.
6. **Integration.** PyTorch CI ran the CPU coverage successfully. The manual CUDA
   workflow requires a runner labelled `self-hosted`, `linux`,
   `x64`, `cuda-inference`; repository variables `INFERENCE_CUDA_COMPANION_ROOT`
   and `INFERENCE_CUDA_PYTHON`; and optionally
   `INFERENCE_CUDA_GOVERNANCE_PYTHON`. See
   [CUDA runner setup](docs/deployment/cuda-runner.md).
7. **Known issues.** Runner registration, runner dependencies, companion checkouts,
   and Actions variables require repository-owner setup. CUDA remains unverified.
8. **Acceptance criteria.** Local CPU/MPS behavior, PyTorch/TensorFlow CI, Docker
   build and health checks are verified. CUDA and browser integration on CUDA remain
   unverified pending default-branch workflow setup and runner registration.
9. **Next milestone.** Provision/register the documented runner and variables, then
   manually dispatch **CUDA acceptance** on `integration/research-platform`.

### Browser and producer verification update

The loopback test now selects CUDA explicitly when requested, compares Faster
R-CNN predictions to eager output on that device, and verifies that history
reports `cuda:0`. The CPU run found that the companion package writer had
stopped invoking the Faster R-CNN exporter. Restored the writer and training
entry-point wiring in Model Engineering's `packaging.py` and `experiment.py`.
Updated its detection test and packaging guide.

The corrected CPU browser test passed against live loopback services. All
three Model Engineering detection tests passed. The inference suite passed
46 tests with three optional skips. The CUDA preflight stopped with device
unavailable. No CUDA workflow was dispatched. The recipe and strict preflight
are verified locally, and CUDA execution remains blocked by runner availability.

## Previous milestone — Approved detection runtime on available accelerators

1. **Original requirements.** Verify the existing TorchScript device handling on
   available devices, and exercise an approved detection package through package
   verification, model loading, and inference on CUDA and Apple MPS where available.
2. **Implementation status.** Completed for Apple MPS. CUDA coverage is implemented
   but skipped because this host has no CUDA device. The repository's own `.venv310`
   does not include PyTorch; the Model Engineering Python 3.10 environment was used
   to execute the runtime suite with PyTorch 2.5.1.
3. **Modified files.** `tests/test_runtime.py` creates a synthetic approved
   detection package with verified file and package digests, loads it through
   `ModelCache`, runs inference on available accelerators, and checks output
   device placement and values. `IMPLEMENTATION_STATUS.md` records evidence.
4. **Testing.** `PYTHONPATH=src:packages/contracts/src
   ../medical_imaging_model/.venv310/bin/python -m pytest -q tests/test_runtime.py`
   reported 6 passed, 1 skipped. The full suite with the same interpreter and
   `PYTHONPATH` reported 45 passed, 4 skipped, zero failures. CUDA was skipped;
   MPS detection inference ran. One upstream Starlette deprecation warning remains.
5. **Functional evidence.** A synthetic detector was packaged, its checksums
   validated as an approved artifact, loaded by the inference runtime on MPS, and
   returned expected normalized boxes, labels, and scores on the MPS device.
   Existing CPU and unavailable-device checks also passed.
6. **Integration.** This covers the Layer 4 approved-package verification and
   PyTorch runtime path. It uses synthetic data and requires no governance service.
7. **Known issues.** CUDA execution remains unverified until a CUDA-capable runner
   is available. The inference repository's local environment lacks PyTorch; the
   compatible sibling environment supplied it for this test run.
8. **Acceptance criteria.** Approved artifact verification, MPS detection output
   values and device placement are verified. CUDA is unverified, not passed.
9. **Next milestone.** Run the same suite on a CUDA-capable CI runner and add that
   runner as an optional CI job if the project has access to one.

## Previous milestone — Bounded inference deadlines

1. **Original requirements.** Bound slot wait and inference execution, test timeout
   recovery and audit behavior, and verify CUDA if hardware is available.
2. **Implementation status.** Completed single-image and shared batch-item
   deadline handling. CUDA execution is blocked on this host by unavailable hardware.
3. **Modified files.** `app.py` isolates worker state and holds the slot until
   completion. `tests/test_deadlines.py` covers the lifecycle. The CI workflow
   runs the new test. `.env.example` and the environment guide document the
   deadline. This report records the milestone.
4. **Testing.** The full suite ran with local framework dependencies and isolated
   PostgreSQL, reporting 46 passed, one opt-in browser skip, and zero failures.
   Ruff, local documentation links, and diff checks passed. The PyTorch CUDA
   availability probe returned false.
5. **Functional evidence.** A model blocked inside its worker returned HTTP 504.
   A second request timed out waiting for the same slot without invoking the
   model. Releasing the worker restored capacity. Both timeouts had one
   audit record and no output, and the next request completed successfully.
6. **Integration.** The exact-version prediction endpoint and batch items use
   the same deadline path. Governance and framework interfaces are unchanged.
7. **Known issues.** Native calls are allowed to finish and hold capacity while
   running. Multipart parsing and audit persistence sit outside the deadline.
8. **Acceptance criteria.** Verified timeout response, request IDs, slot retention,
   queue timeout, audit consistency, and recovery. CUDA execution is unverified.
9. **Next milestone.** Run the existing TorchScript device test on a CUDA runner,
   then add approved detection coverage for that runtime.

## Previous milestone — Device compatibility and operational metrics

1. **Original requirements.** Verify available execution devices, expose model-load
   duration, and show inference latency percentiles in the dashboard.
2. **Implementation status.** Completed device selection, scoped percentiles, and
   in-memory load timing. Verified PyTorch CPU and Apple MPS execution. CUDA
   execution remains unverified because this host has no CUDA device.
3. **Modified files.** `runtime.py` adds explicit devices and cache load timing.
   `audit.py` calculates nearest-rank latency percentiles. Streamlit's
   `apps/frontend/app.py` displays the percentiles and existing loaded-model
   tables show timing. `test_runtime.py`, `test_history.py`, `test_frontend.py`,
   and `test_postgres.py` verify these paths. The monitoring and environment
   guides describe the settings. This report records the milestone.
4. **Testing.** `bash /tmp/miip-postgres-check.sh` ran `python -m pytest -q`
   with local framework dependencies and an isolated PostgreSQL database.
   Results were 45 passed, one opt-in browser test skipped, and zero failed.
   Ruff lint and formatting, documentation links, and diff checks passed.
5. **Functional evidence.** A real TorchScript linear model produced identical
   output on CPU and MPS, and the test checked the output tensor's device.
   An unavailable CUDA request was rejected. Cache reuse preserved the
   original load duration. A mixed-outcome SQLite fixture returned scoped
   P50/P95/P99 values of 20/30/30 ms. PostgreSQL and the Streamlit P95
   display passed their assertions.
6. **Integration.** Existing metrics and active-model endpoints carry additive
   fields. `gpu` keeps its framework-specific behavior. Explicit `cuda` and
   `mps` select PyTorch devices. Torchvision scripted detection is refused on MPS.
7. **Known issues.** MPS evidence covers the tested classification operators.
   CUDA execution, broad accelerator operator parity, and persisted load-time
   trends remain unverified or unimplemented.
8. **Acceptance criteria.** Verified CPU and MPS classification device placement,
   unavailable CUDA handling, load-time capture and reuse, scoped percentiles in
   SQLite and PostgreSQL, and percentile rendering. CUDA execution is unverified.
9. **Next milestone.** Add bounded inference deadlines with lifecycle tests,
   then verify model loading and execution on a CUDA runner when available.

## Previous milestone — Faster R-CNN export

1. **Original requirements.** Verify a second detector architecture against
   its eager model. Demonstrate governance approval and browser inference
   for images producing different numbers of detections.
2. **Implementation status.** Completed for the CPU Torchvision Faster R-CNN
   path. GPU parity, RetinaNet, and DETR remain unverified.
3. **Modified files.** Model Engineering's `packaging.py` now calls the scripted
   exporter and publishes export and parity digests. Inference's
   `preprocessing.py` validates the native-size aspect ratio contract.
   `tests/test_preprocessing.py` verifies tensor shape and scaling.
   `tests/test_loopback_workstation.py` fixes an ambiguous browser selector.
   This report records the verification.
4. **Testing.** Inference `pytest -q` with source paths configured passed
   37 tests with five optional skips. The opt-in Chrome test passed.
   Model Engineering `pytest -q tests/test_detection_pipeline.py` passed
   all three tests. An earlier browser run hit a duplicate text selector.
   The renamed checkout also required source paths for pytest discovery.
5. **Functional evidence.** A governed package produced 20 detections from
   a 32×32 upload and 86 from a 96×32 upload. Stored boxes, labels and scores
   matched the eager detector, and both records matched the approved digest.
6. **Integration.** The package declares native RGB input and the
   Torchvision scripted detection tuple. The embedded detector handles
   its own resizing and returns original-image pixel coordinates.
7. **Known issues.** Browser inputs and model weights are synthetic.
   A separate producer test verifies fitted-checkpoint export. The
   scripted runtime requires compatible PyTorch and Torchvision builds.
8. **Acceptance criteria.** Verified fitted-checkpoint packaging, export reload
   parity, native-size preprocessing, approval, variable detection counts,
   browser visualization, and history. Accelerator parity is unverified.
9. **Next milestone.** Verify model device compatibility and add model-loading
   duration and latency percentiles to the operational dashboard.

## Previous milestone — Upstream YOLO-grid export

1. **Original requirements.** Export an actual Model Engineering detector,
   compare eager and reloaded outputs, and verify approval, image upload,
   detection visualization, and history through the live workstation.
2. **Implementation status.** Completed for single-image CPU YOLO-grid.
   Faster R-CNN, RetinaNet, and DETR exports remain unsupported.
3. **Modified files.** In inference, `src/medical_imaging_inference/preprocessing.py`
   implements training-equivalent resizing, `tests/test_preprocessing.py` checks
   exact pixel parity, `tests/test_loopback_workstation.py` replaces the mock
   detector with the upstream export, and `docs/integration/research-platform.md`
   documents the handoff. In Model Engineering, `packaging.py` produces and
   verifies the export, `experiment.py` packages the fitted checkpoint,
   `tests/test_detection_pipeline.py` exercises training-to-package, and
   `docs/checkpoints-and-packaging.md` describes the artifact.
4. **Testing.** Inference `pytest -q` passed 36 with five optional skips.
   The opt-in Chrome test passed. `pytest -q tests/test_preprocessing.py`
   passed 11 with PyTorch. Model Engineering
   `pytest -q tests/test_detection_pipeline.py` passed three.
   Ruff, documentation links, and both repositories' diff checks passed.
5. **Functional evidence.** Reloaded TorchScript matched eager boxes,
   class IDs, and scores on three deterministic tensors. Real Layer 3 HTTP
   approval and a Chrome upload displayed class lesion. The history record
   retained the approved digest and the 4×6 image box `[1, 1.5, 3, 4.5]`.
6. **Integration.** The approved Model Engineering package declares
   `torch_bilinear` and normalized xyxy coordinates. Layer 4 applies
   PyTorch interpolation with `align_corners=False`, then maps the output
   to the uploaded image size.
7. **Known issues.** Browser weights are deterministic synthetic weights.
   The producer test trains a separate synthetic checkpoint. CPU fixed-shape
   batch-one export is the scope of the parity checks.
8. **Acceptance criteria.** Verified fitted-checkpoint packaging, eager versus
   reloaded parity, preprocessing parity, governance approval, live
   visualization, coordinate mapping, and history. Other architecture and
   accelerator exports remain unverified.
9. **Next milestone.** Add source-versus-export parity and a governed handoff
   for a second detector architecture, including variable detection counts.

## Previous milestone — Live workstation and task acceptance

1. **Original requirements.** Run independent loopback services. Demonstrate
   model approval, discovery, upload, task execution, visualization, and history.
   Verify segmentation class identity, mask export, and detection coordinates.
2. **Implementation status.** Completed for compatible synthetic TorchScript exports
   on CPU for all three tasks. Upstream production-model detection handoff is
   blocked by the unsupported Model Engineering export. Live TensorFlow browser
   coverage and advanced result comparisons are outside the verified scope.
3. **Modified files.** `apps/frontend/visualizations.py` decodes and exports class
   IDs as 16-bit PNG. `apps/frontend/app.py` exposes the mask download.
   `tests/test_loopback_workstation.py` starts isolated services and exercises Chrome.
   `docs/integration/research-platform.md` documents reproduction and the upstream
   dependency. `IMPLEMENTATION_STATUS.md` records the evidence.
4. **Testing.** The opt-in `python -m pytest -q tests/test_loopback_workstation.py`
   passed once with the final changes. The base suite passed 36 with four
   optional skips and zero failures. An initial browser-suite attempt failed
   because the selected interpreter lacked Uvicorn. The test now accepts
   separate governance and inference interpreters. Ruff, local doc links and
   `git diff --check` passed.
5. **Functional evidence.** Governance registered and separately approved three
   packages. Chrome uploaded a 4×6 synthetic PNG for each approved task.
   Classification displayed the predicted class. Segmentation rendered the
   overlay and downloaded a 4×6 PNG containing class ID 1. Detection rendered
   the annotated image and persisted the scaled box `[0, 0, 2, 3]`. All three
   records were reachable through history and matched their approved digests.
6. **Integration.** Real Layer 3 HTTP registration, review, approval, and discovery
   were exercised against temporary SQLite databases and a shared local package
   mount. Model Engineering's package writer was reused with synthetic exports.
7. **Known issues.** The fixture verifies a decoded TorchScript detector and
   does not establish compatibility with unsupported upstream detector heads.
   The mask PNG supports IDs up to 65,535. These tests use loopback research
   identities and CPU runtimes.
8. **Acceptance criteria.** Verified live services, approval, task selection,
   upload, inference, classification display, mask visualization and download,
   detection visualization and coordinate conversion, and history lineage.
   Upstream detection export remains unverified.
9. **Next milestone.** Establish an upstream detection export with
   source-versus-export parity, then run it through the approved browser flow.
   Extend runtime and operational telemetry tests for CPU and available accelerators.

## Previous milestone — Audit and local classification

## 1. Original requirements

Audit the repository and trace requirements. Discover only approved models
for ordinary inference, verify the selected package and its inputs, and
demonstrate local classification through saved prediction history. Preserve
model identity, preprocessing and application provenance, and support
filtering by model version.

## 2. Implementation status

| Status | Scope |
| --- | --- |
| Completed | Repository audit review, local classification API lifecycle for PyTorch and TensorFlow, prediction provenance fields, model-version history filtering |
| Partially completed | P0 dependency and architecture compatibility checks, GUI workflow verification, complete prediction lineage |
| Not implemented | Configurable retention, Apple MPS execution, inference execution deadlines, standalone artifact retrieval and historical visualization |
| Blocked | No local implementation blocker identified. Accelerator coverage requires suitable hardware. |

## 3. Modified files

| File | Purpose |
| --- | --- |
| `src/medical_imaging_inference/app.py` | Records validated input metadata and runtime provenance and forwards history version filters. |
| `src/medical_imaging_inference/audit.py` | Applies the exact version filter in history queries. Reuses the existing lineage column. |
| `apps/frontend/app.py` | Adds a version filter to prediction history. |
| `tests/test_local_lifecycle.py` | Asserts persisted provenance and matching and nonmatching version filters for both frameworks. |
| `docs/api/api-reference.md` | Documents the history filter. |
| `docs/architecture/inference-traceability.md` | Updates audit evidence and links this report. |
| `IMPLEMENTATION_STATUS.md` | Records milestone scope, tests, and acceptance evidence. |

Other worktree changes predated this milestone. The audit is in
[requirement traceability](docs/architecture/inference-traceability.md) and the
[local integration audit](docs/architecture/local-platform-integration-audit.md).

## 4. Testing

- `.venv310/bin/python -m pytest -q` reported 36 passed, zero failed and three skipped. The skips were PostgreSQL and the two optional framework tests.
- `python -m pytest -q tests/test_local_lifecycle.py`, with the sibling Model Engineering Python 3.10 packages on `PYTHONPATH`, reported two passed and zero failed. Tests used both actual framework exports with mocked governance HTTP and SQLite persistence.
- The documented cross-repository command, `../medical_imaging_model/.venv310/bin/python packages/contracts/scripts/test_platform.py`, passed all seven tests. It uses real synthetic training, governance transitions and inference behind ASGI test transports.
- Ruff, documentation link checking, and `git diff --check` passed.

## 5. Functional evidence

The API discovered approved version 1, returned its verified model card,
ran real TorchScript and SavedModel predictions on a synthetic 3×4 PNG,
returned scores summing to one and stored the prediction with its exact
artifact digest. History returned version 1, persisted the input dimensions,
MIME type and preprocessing version, and returned no rows for version 2.
A subsequent request for the deprecated version received HTTP 404.

## 6. Integration

Local discovery and approval consume the Layer 3 catalogue and exact-version
HTTP contracts. Production OIDC remains distinct from the loopback research
identity. The lifecycle fixtures generate model exports using PyTorch and
TensorFlow. They exercise the HTTP response contract with mocked governance.

## 7. Known issues

Approval timestamp is recorded when supplied. An immutable approval event
identifier is not established by the tested projected contract. Upload bytes
remain ephemeral and original-image replay from history is unavailable.
Model compatibility checks still need coverage for unsupported runtime
versions and accelerator operations. History GUI rendering of the new filter
has not been tested through browser interaction.

## 8. Acceptance criteria

| Criterion | Evidence |
| --- | --- |
| 1–3 Approved discovery, checksum, and load | Verified by both framework tests. |
| 4–6 Upload, validation, preprocessing | Verified by synthetic lifecycle and input validation tests. |
| 7–8 Classification execution and scores | Verified with real local CPU framework exports. |
| 9 Visualized results | Unverified end to end. Component tests cover the workstation separately. |
| 10–12 Persistence, version trace and history | Verified by both framework lifecycle tests. |
| 13 API tests | Verified for the executed suite. |
| 14 End-to-end tests | Verified for HTTP-to-runtime and history with a mocked governance response, and through the cross-repository ASGI suite. Live browser workflow remains unverified. |
| 15 Documentation matches actual functionality | Local link check and documented scope. Full documentation review remains open. |

## 9. Next milestone

Demonstrate the live loopback workstation workflow with separately running
governance and inference services, then cover segmentation and detection
exports through the same package approval, upload, and visualization path.
