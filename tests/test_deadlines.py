import asyncio
import io
from threading import Event

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from medical_imaging_inference import app as application
from medical_imaging_inference.preprocessing import PreprocessingSpec
from medical_imaging_inference.runtime import LoadedModel
from medical_imaging_inference.security import authorize


def test_deadline_retains_slot_and_audits_once(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "deadline.sqlite"))
    monkeypatch.setenv("INFERENCE_DEADLINE_SECONDS", "0.35")
    slots = asyncio.Semaphore(1)
    monkeypatch.setattr(application, "inference_slots", slots)
    monkeypatch.setitem(
        application.app.dependency_overrides,
        authorize,
        lambda: {"subject": "test", "roles": ["inference_user"]},
    )
    entered, release = Event(), Event()
    calls = []

    def run(_):
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return np.array([[0.0, 1.0]])

    async def approved(model_id, version):
        return {"state": "approved", "model_id": model_id, "version": version, "artifact_digest": "test"}

    manifest = {
        "framework": "pytorch",
        "task": "classification",
        "class_mapping": {"a": 0, "b": 1},
        "preprocessing": {"image_size": [2, 2], "color": "RGB", "scale": "[0,1]"},
    }
    model = LoadedModel(manifest, PreprocessingSpec.from_manifest(manifest), "cpu", run)
    monkeypatch.setattr(application, "_approved_record", approved)
    monkeypatch.setattr(application.models, "get", lambda _: model)
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    files = {"image": ("test.png", image.getvalue(), "image/png")}
    path = "/api/v1/predictions/test/1"
    with TestClient(application.app) as client:
        try:
            first = client.post(path, files=files)
            assert first.status_code == 504
            assert entered.is_set()
            assert slots.locked()
            second = client.post(path, files=files)
            assert second.status_code == 504
            assert len(calls) == 1
        finally:
            release.set()

        async def drain():
            await asyncio.wait_for(slots.acquire(), 3)
            slots.release()

        client.portal.call(drain)
        monkeypatch.setenv("INFERENCE_DEADLINE_SECONDS", "3")
        assert client.post(path, files=files).status_code == 200
        rows = client.get("/api/v1/inference/history").json()["items"]
        assert len(rows) == 3
        assert sum(row["status"] == "completed" for row in rows) == 1
        for response in (first, second):
            detail = client.get("/api/v1/inference/history/" + response.headers["x-request-id"]).json()
            assert detail["error_code"] == "http_504"
            assert detail["output"] is None
