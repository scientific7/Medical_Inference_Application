"""Opt-in browser smoke test with isolated databases and real loopback services."""

import copy
import os
import socket
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import httpx
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def serve(stack, command, env, path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log = stack.enter_context(path.open("w+"))
    proc = subprocess.Popen([*command, str(port)], cwd=ROOT, env=env, stdout=log, stderr=log)

    def stop():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    stack.callback(stop)
    return proc, f"http://127.0.0.1:{port}"


def wait_ready(proc, origin, health):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            if httpx.get(origin + health, timeout=1).is_success:
                return
        except httpx.RequestError:
            pass
        time.sleep(0.2)
    pytest.fail(f"Service at {origin} did not become ready")


@pytest.mark.skipif(
    not os.getenv("INFERENCE_BROWSER_PYTHON"), reason="Loopback test interpreter not configured"
)
def test_approved_tasks_in_live_workstation(tmp_path):
    torch = pytest.importorskip("torch")
    playwright = pytest.importorskip("playwright.sync_api")
    from medical_imaging_models.models.detection import build_torch_faster_rcnn
    from medical_imaging_models.models.yolo import build_torch_yolo_grid
    from medical_imaging_models.packaging import write_model_package

    class Classification(torch.nn.Module):
        def forward(self, image):
            return image.mean(dim=(2, 3))[:, :2]

    class Segmentation(torch.nn.Module):
        def forward(self, image):
            return image[:, :1] - 0.5

    detector = build_torch_yolo_grid(1, grid_size=1, width=8).eval()
    with torch.no_grad():
        for parameter in detector.parameters():
            parameter.zero_()

    torch.manual_seed(8)
    faster = build_torch_faster_rcnn(2, min_size=32, max_size=96).eval()
    target = os.getenv("INFERENCE_BROWSER_DEVICE", "cpu")
    assert target in {"cpu", "cuda"}, "Loopback device must be cpu or cuda"
    if target == "cuda":
        assert torch.cuda.is_available(), "A CUDA runner is required for CUDA loopback acceptance"
    eager = copy.deepcopy(faster).to(target) if target == "cuda" else faster
    upload_cases = {}
    for width in (32, 96):
        path = tmp_path / f"faster-{width}.png"
        Image.new("RGB", (width, 32), (255, 0, 0)).save(path)
        pixels = torch.zeros(3, 32, width)
        pixels[0] = 1
        with torch.inference_mode():
            upload_cases[path] = {key: value.cpu() for key, value in eager([pixels.to(target)])[0].items()}
    assert len({len(value["boxes"]) for value in upload_cases.values()}) == 2

    workspace = ROOT.parent
    artifacts = tmp_path / "packages"
    python = os.environ["INFERENCE_BROWSER_PYTHON"]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            map(
                str,
                (
                    ROOT / "src",
                    ROOT / "packages/contracts/src",
                    workspace / "MLOps-and-model-governance/src",
                    workspace / "medical_imaging_model/src",
                    *os.getenv("PYTHONPATH", "").split(os.pathsep),
                ),
            )
        ),
        "INFERENCE_ENVIRONMENT": "development",
        "ENVIRONMENT": "development",
        "MLOPS_POLICY_DIR": str(workspace / "MLOps-and-model-governance/config/policies"),
        "MLOPS_DATABASE_PATH": str(tmp_path / "registry.sqlite3"),
        "INFERENCE_DATABASE_PATH": str(tmp_path / "predictions.sqlite3"),
        "MLOPS_ARTIFACT_ROOT": str(artifacts),
        "INFERENCE_ARTIFACT_ROOT": str(artifacts),
        "INFERENCE_RESEARCH_UI": "1",
        "MLOPS_SERVICE_TOKEN": "local-test",
        "INFERENCE_DEVICE": target,
        "MLFLOW_TRACKING_URI": "",
        "INFERENCE_LOCAL_CATALOGUE": "",
        "INFERENCE_DATABASE_URL": "",
        "INFERENCE_S3_ARTIFACT_ROOT": "",
        "INFERENCE_DATASET_API_URL": "",
    }
    image = tmp_path / "synthetic.png"
    Image.new("RGB", (4, 6), (255, 0, 0)).save(image)
    example = torch.zeros(1, 3, 2, 2)
    with ExitStack() as stack:
        registry, registry_url = serve(
            stack,
            [
                os.getenv("INFERENCE_BROWSER_GOVERNANCE_PYTHON") or python,
                "scripts/research_api.py",
                "mlops",
                "--port",
            ],
            env,
            tmp_path / "governance.log",
        )
        wait_ready(registry, registry_url, "/health")
        env["MLOPS_API_URL"] = registry_url
        api, api_url = serve(
            stack, [python, "scripts/research_api.py", "inference", "--port"], env, tmp_path / "inference.log"
        )
        wait_ready(api, api_url, "/ready")
        env["INFERENCE_API_URL"] = api_url
        ui, ui_url = serve(
            stack,
            [
                python,
                "-m",
                "streamlit",
                "run",
                "apps/frontend/app.py",
                "--server.address=127.0.0.1",
                "--server.headless=true",
                "--server.port",
            ],
            env,
            tmp_path / "streamlit.log",
        )
        wait_ready(ui, ui_url, "/_stcore/health")
        approvals = {}
        for model_id, task, module in (
            ("classification", "classification", Classification()),
            ("segmentation", "segmentation", Segmentation()),
            ("detection", "detection", detector),
            ("faster-rcnn", "detection", faster),
        ):
            export = tmp_path / f"{task}.ts"
            spec = {
                "image_size": [32, 32] if model_id == "faster-rcnn" else [2, 2],
                "color": "RGB",
                "scale": "[0,1]",
            }
            package = artifacts / model_id
            package.mkdir(parents=True)
            if task == "detection":
                target = tmp_path / f"{model_id}-weights.pt"
                torch.save(module.state_dict(), target)
            else:
                torch.jit.trace(module, example).save(str(export))
                target = package / "inference_model.ts"
                target.write_bytes(export.read_bytes())
            write_model_package(
                package,
                {
                    "experiment_id": task,
                    "dataset_release_id": "synthetic-release",
                    "dataset_receipt_digest": "a" * 64,
                    "manifest_digests": {name: "b" * 64 for name in ("train", "validation", "test")},
                    "framework": "pytorch",
                    "architecture": {"detection": "yolo_grid", "faster-rcnn": "faster_rcnn_resnet50_fpn"}.get(
                        model_id, "synthetic"
                    ),
                    "task": task,
                    "preprocessing": spec,
                    "class_mapping": {"red": 0, "lesion": 1},
                    "evaluation_summary": {
                        "classification": {"accuracy": 1.0, "macro_f1": 1.0},
                        "segmentation": {"dice": 1.0, "iou": 1.0},
                        "detection": {"map": 1.0},
                    }[task],
                    "config_sha256": "c" * 64,
                    "git_commit": "synthetic-browser-fixture",
                },
                target,
                detector=module if task == "detection" else None,
            )
            submit = {"X-Research-Actor": "trainer", "Idempotency-Key": f"candidate-{model_id}"}
            response = httpx.post(
                registry_url + "/api/v1/models/register",
                headers=submit,
                json={
                    "model_id": model_id,
                    "version": "1",
                    "artifact_path": model_id,
                    "experiment_id": task,
                    "dataset_release_id": "synthetic-release",
                },
            )
            assert response.status_code == 201, response.text
            route = f"{registry_url}/api/v1/models/{model_id}/versions/1"
            response = httpx.post(
                route + "/request-review", headers=submit, json={"reason": "Synthetic test"}
            )
            assert response.status_code == 200, response.text
            response = httpx.post(route + "/approve", headers={"X-Research-Actor": "reviewer"})
            assert response.status_code == 200, response.text
            approvals[model_id] = response.json()["artifact_digest"]

        with playwright.sync_playwright() as browser_driver:
            browser = browser_driver.chromium.launch(channel="chrome")
            page = browser.new_page()
            for task in ("classification", "segmentation", "detection"):
                page.goto(ui_url)
                page.get_by_text("Inference", exact=True).click()
                page.get_by_role("combobox").first.click()
                page.get_by_role("option", name=task.title(), exact=True).click()
                page.get_by_text(task, exact=True).wait_for()
                page.locator('input[type="file"]').set_input_files(image)
                page.get_by_role("button", name="Run inference").click()
                page.get_by_text("Export prediction metadata", exact=True).wait_for()
                if task == "classification":
                    page.get_by_role("heading", name="red").wait_for()
                elif task == "segmentation":
                    page.get_by_text("Overlay", exact=True).wait_for()
                    with page.expect_download() as downloaded:
                        page.get_by_text("Download class mask", exact=True).click()
                    with Image.open(downloaded.value.path()) as mask:
                        assert mask.size == (4, 6)
                        assert mask.getextrema() == (1, 1)
                else:
                    page.get_by_text("Detections", exact=True).wait_for()
            for path in upload_cases:
                page.goto(ui_url)
                page.get_by_text("Inference", exact=True).click()
                page.get_by_role("combobox").first.click()
                page.get_by_role("option", name="Detection", exact=True).click()
                page.get_by_text("detection", exact=True).wait_for()
                page.get_by_role("combobox").nth(1).click()
                page.get_by_role("option", name="faster-rcnn", exact=True).click()
                page.get_by_test_id("stMainBlockContainer").get_by_text("faster-rcnn", exact=True).wait_for()
                page.locator('input[type="file"]').set_input_files(path)
                page.get_by_text(path.name, exact=True).wait_for()
                page.get_by_role("button", name="Run inference").click()
                page.get_by_text("Detections", exact=True).wait_for()
            page.get_by_text("History", exact=True).click()
            page.get_by_text("Request details", exact=True).wait_for()
            browser.close()

        rows = httpx.get(api_url + "/api/v1/inference/history").json()["items"]
        assert len(rows) == 5
        for row in rows:
            assert row["status"] == "completed"
            assert row["device"] == ("cuda:0" if target == "cuda" else "cpu")
            assert row["artifact_sha256"] == approvals[row["model_id"]]
            detail = httpx.get(api_url + f"/api/v1/inference/history/{row['prediction_id']}").json()
            if row["task"] == "segmentation":
                assert detail["output"]["mask_runs"] == [[1, 24]]
            if row["model_id"] == "detection":
                assert detail["output"]["boxes_xyxy"] == [[1.0, 1.5, 3.0, 4.5]]
                assert detail["output"]["labels"] == ["lesion"]
            if row["model_id"] == "faster-rcnn":
                width = detail["lineage"]["input_dimensions"]["width"]
                expected = upload_cases[tmp_path / f"faster-{width}.png"]
                torch.testing.assert_close(torch.tensor(detail["output"]["boxes_xyxy"]), expected["boxes"])
                torch.testing.assert_close(torch.tensor(detail["output"]["scores"]), expected["scores"])
                assert detail["output"]["class_ids"] == expected["labels"].tolist()
