import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from medical_imaging_inference import app as application
from medical_imaging_inference.preprocessing import PreprocessingSpec
from medical_imaging_inference.runtime import LoadedModel
from medical_imaging_inference.security import authorize


def test_batch_records_partial_failure_and_scopes_status(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "batch.sqlite3"))
    user = {"subject": "owner", "roles": ["inference_user"]}
    monkeypatch.setitem(application.app.dependency_overrides, authorize, lambda: user)

    async def approved(model_id, version):
        return {"model_id": model_id, "version": version, "state": "approved", "artifact_digest": "test"}

    manifest = {
        "task": "classification",
        "framework": "pytorch",
        "class_mapping": {"one": 0, "two": 1},
        "preprocessing": {"image_size": [2, 2], "color": "RGB", "scale": "[0,1]"},
    }
    model = LoadedModel(
        manifest, PreprocessingSpec.from_manifest(manifest), "cpu", lambda _: np.array([[0.0, 1.0]])
    )
    monkeypatch.setattr(application, "_approved_record", approved)
    monkeypatch.setattr(application.models, "get", lambda _: model)
    image = io.BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    with TestClient(application.app) as client:
        response = client.post(
            "/api/v1/inference/batch",
            data={"model_id": "test", "version": "1"},
            files=[
                ("images", ("good.png", image.getvalue(), "image/png")),
                ("images", ("bad.png", b"bad", "image/png")),
            ],
        )
        assert response.status_code == 202
        path = f"/api/v1/inference/batch/{response.json()['batch_id']}"
        job = client.get(path).json()
        assert job["status"] == "partial"
        assert [item["status"] for item in job["items"]] == ["completed", "failed"]
        prediction = client.get(f"/api/v1/inference/history/{job['items'][0]['prediction_id']}").json()
        assert prediction["output"]["predicted_class"] == "two"
        assert client.get("/api/v1/system/metrics").json()["failed"] == 1
        user["subject"] = "another-user"
        assert client.get(path).status_code == 404
