from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from medical_imaging_inference import runtime
from medical_imaging_inference.app import app
from medical_imaging_inference.security import authorize, file_digest


def test_cache_reuses_verified_version_and_keys_by_digest(monkeypatch):
    record = {"state": "approved", "model_id": "lesion", "version": "1", "artifact_digest": "a"}
    loads = []

    def verify(metadata):
        loads.append(metadata["artifact_digest"])
        return Path("."), {}

    monkeypatch.setattr(runtime, "verified_package", verify)
    monkeypatch.setattr(runtime, "load_export", lambda *_: SimpleNamespace(device="cpu"))
    cache = runtime.ModelCache(capacity=1)
    loaded = cache.get(record)
    assert cache.get(record) is loaded
    timing = cache.active()[0]["load_ms"]
    assert timing >= 0
    assert cache.get(record) is loaded
    assert cache.active()[0]["load_ms"] == timing
    assert loads == ["a"]
    with pytest.raises(ValueError, match="not approved"):
        cache.get({**record, "state": "candidate_submitted"})
    assert cache.get({**record, "artifact_digest": "b"}) is not loaded
    assert loads == ["a", "b"]
    assert cache.unload("lesion", "1")
    assert not cache.unload("lesion", "1")


def test_deployment_routes_require_admin():
    app.dependency_overrides[authorize] = lambda: {"subject": "researcher", "roles": ["inference_user"]}
    try:
        client = TestClient(app)
        selection = {"model_id": "lesion", "version": "1"}
        assert client.get("/api/v1/models/active").status_code == 200
        assert client.post("/api/v1/models/load", json=selection).status_code == 403
        assert client.post("/api/v1/models/unload", json=selection).status_code == 403
    finally:
        app.dependency_overrides.pop(authorize)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_torch_export_runs_on_requested_device(device, tmp_path, monkeypatch):
    import numpy as np

    torch = pytest.importorskip("torch")
    monkeypatch.setenv("INFERENCE_DEVICE", device)
    manifest = {
        "framework": "pytorch",
        "task": "classification",
        "preprocessing": {"image_size": [2, 2], "color": "RGB", "scale": "[0,1]"},
    }
    if (device == "cuda" and not torch.cuda.is_available()) or (
        device == "mps" and not torch.backends.mps.is_available()
    ):
        with pytest.raises(ValueError, match="execution is unavailable"):
            runtime.load_export(tmp_path, manifest)
        return

    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(12, 2, bias=False))
    with torch.no_grad():
        model[1].weight.fill_(1)
    torch.jit.trace(model, torch.zeros(1, 3, 2, 2)).save(str(tmp_path / "inference_model.ts"))
    loaded = runtime.load_export(tmp_path, manifest)
    output = loaded.run(np.ones((3, 2, 2), dtype=np.float32))
    assert output.device.type == device
    torch.testing.assert_close(output.cpu(), torch.tensor([[12.0, 12.0]]))


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_approved_detection_export_runs_on_requested_device(device, tmp_path, monkeypatch):
    import hashlib
    import json

    import numpy as np

    torch = pytest.importorskip("torch")
    available = device == "cpu" or (
        torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    )
    if not available:
        pytest.skip(f"{device} is unavailable on this runner")

    package = tmp_path / "packages" / "detector"
    package.mkdir(parents=True)

    class SyntheticDetector(torch.nn.Module):
        def forward(self, image):
            batch = image.shape[0]
            zero = image[:, 0, 0, 0].reshape(batch, 1) * 0
            boxes = torch.cat((zero + 0.1, zero + 0.2, zero + 0.8, zero + 0.9), dim=1).reshape(batch, 1, 4)
            labels = zero.to(torch.int64) + 1
            scores = zero + 0.75
            return {"boxes": boxes, "labels": labels, "scores": scores}

    export = package / "inference_model.ts"
    example = torch.zeros(1, 3, 8, 8)
    torch.jit.trace(SyntheticDetector(), example, strict=False).save(str(export))
    manifest = {
        "task": "detection",
        "framework": "pytorch",
        "architecture": "synthetic_detector",
        "runtime_output": "batched_detection_dict",
        "class_mapping": {"lesion": 1},
        "postprocessing": {"version": "1", "detection_format": "xyxy_normalized"},
        "preprocessing": {
            "version": "1",
            "image_size": [8, 8],
            "color": "RGB",
            "scale": "[0,1]",
            "resize": "torch_bilinear",
            "aspect_ratio": "stretch",
            "orientation": "raw",
            "channel_order": "CHW",
            "formats": ["PNG"],
        },
        "weights": {"path": "inference_model.ts", "sha256": file_digest(export)},
    }
    manifest_path = package / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    filenames = sorted(path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file())
    package_digest = hashlib.sha256(
        "".join(f"{name}:{file_digest(package / name)}\n" for name in filenames).encode()
    ).hexdigest()
    record = {
        "state": "approved",
        "model_id": "detector",
        "version": "1",
        "artifact_path": str(package),
        "artifact_digest": package_digest,
    }
    monkeypatch.setenv("INFERENCE_ARTIFACT_ROOT", str(package.parent))
    monkeypatch.setenv("INFERENCE_DEVICE", device)
    loaded = runtime.ModelCache().get(record)
    output = loaded.run(np.ones((3, 8, 8), dtype=np.float32))
    assert loaded.device == ("cuda:0" if device == "cuda" else device)
    assert output["boxes"].device.type == device
    assert output["labels"].device.type == device
    assert output["scores"].device.type == device
    torch.testing.assert_close(output["boxes"].cpu(), torch.tensor([[0.1, 0.2, 0.8, 0.9]]))
