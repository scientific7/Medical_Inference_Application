import hashlib
import io
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medical_imaging_inference import app as application
from medical_imaging_inference.runtime import ModelCache
from medical_imaging_inference.security import authorize, file_digest


@pytest.mark.parametrize("framework", ["pytorch", "tensorflow"])
def test_approved_package_upload_and_history(framework, tmp_path, monkeypatch):
    package = tmp_path / "approved" / "example"
    package.mkdir(parents=True)
    if framework == "pytorch":
        torch = pytest.importorskip("torch")

        class Classifier(torch.nn.Module):
            def forward(self, image):
                return image.mean(dim=(2, 3))[:, :2]

        weight = package / "inference_model.ts"
        torch.jit.trace(Classifier(), torch.zeros(1, 3, 2, 2)).save(str(weight))
    else:
        tf = pytest.importorskip("tensorflow")

        class Classifier(tf.Module):
            @tf.function(input_signature=[tf.TensorSpec([None, 2, 2, 3], tf.float32)])
            def predict(self, image):
                return tf.reduce_mean(image, axis=(1, 2))[:, :2]

        model = Classifier()
        tf.saved_model.save(
            model, str(package / "serving_model"), signatures={"serving_default": model.predict}
        )
        weight = package / "serving_model" / "saved_model.pb"

    manifest = {
        "task": "classification",
        "framework": framework,
        "architecture": "synthetic",
        "class_mapping": {"red": 0, "green": 1},
        "preprocessing": {"image_size": [2, 2], "color": "RGB", "scale": "[0,1]"},
        "weights": {"path": weight.relative_to(package).as_posix(), "sha256": file_digest(weight)},
    }
    manifest_path = package / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    (package / "model_card.md").write_text("# Example\n\n## Intended use\n\nResearch.\n")
    names = sorted(path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file())
    digest = hashlib.sha256(
        "".join(f"{name}:{file_digest(package / name)}\n" for name in names).encode()
    ).hexdigest()
    record = {
        **{
            key: manifest[key]
            for key in ("task", "framework", "architecture", "class_mapping", "preprocessing")
        },
        "model_id": "example",
        "version": "1",
        "status": "approved",
        "artifact_uri": str(package),
        "artifact_sha256": digest,
        "model_manifest_sha256": file_digest(manifest_path),
    }

    def respond(request):
        if request.url.path == "/api/v1/inference/models":
            return httpx.Response(200, json={"items": [record]})
        assert request.url.path == "/api/v1/models/example/approved/1"
        return httpx.Response(200, json=record) if record["status"] == "approved" else httpx.Response(404)

    transport = httpx.MockTransport(respond)
    real_client = httpx.AsyncClient
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setenv("MLOPS_SERVICE_TOKEN", "test-token")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("INFERENCE_LOCAL_CATALOGUE", raising=False)
    monkeypatch.setenv("INFERENCE_ARTIFACT_ROOT", str(package.parent))
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "history.sqlite3"))
    monkeypatch.setattr(application, "models", ModelCache())
    monkeypatch.setitem(
        application.app.dependency_overrides,
        authorize,
        lambda: {"subject": "test", "roles": ["inference_user"]},
    )
    image = io.BytesIO()
    Image.new("RGB", (3, 4), (255, 0, 0)).save(image, format="PNG")
    files = {"image": ("image.png", image.getvalue(), "image/png")}
    with TestClient(application.app) as client:
        entries = client.get("/api/v1/models?approved=true").json()["items"]
        assert [item["version"] for item in entries] == ["1"]
        assert "artifact_uri" not in entries[0]
        card = client.get("/api/v1/models/example/versions/1/card").json()
        assert card["formats"] == ["JPEG", "PNG"]
        assert "Research." in card["card"]
        response = client.post("/api/v1/predictions/example/1", files=files)
        assert response.status_code == 200, response.text
        prediction = response.json()
        assert prediction["output"]["predicted_class"] == "red"
        assert np.isclose(sum(prediction["output"]["scores"].values()), 1)
        history = client.get(f"/api/v1/inference/history/{prediction['prediction_id']}").json()
        assert history["artifact_sha256"] == digest
        assert history["model_version"] == "1"
        assert history["lineage"]["input_format"] == "image/png"
        assert history["lineage"]["input_dimensions"] == {"width": 3, "height": 4}
        assert history["lineage"]["preprocessing_version"] == "1"
        assert history["lineage"]["application_version"] == application.app.version
        assert client.get("/api/v1/inference/history?model_version=2").json()["items"] == []
        assert len(client.get("/api/v1/inference/history?model_version=1").json()["items"]) == 1
        record["status"] = "deprecated"
        assert client.post("/api/v1/predictions/example/1", files=files).status_code == 404
