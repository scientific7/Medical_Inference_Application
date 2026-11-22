from fastapi.testclient import TestClient

from medical_imaging_inference.app import MAX_IMAGE_BYTES, app
from medical_imaging_inference.security import authorize

PATH = "/api/v1/predictions/model/1"


def test_oversize_and_mismatched_bodies_are_rejected_before_parsing():
    with TestClient(app) as client:
        assert client.post(PATH, headers={"Content-Length": str(MAX_IMAGE_BYTES * 2)}).status_code == 413
        assert client.post(PATH, content=b"x", headers={"Content-Length": "2"}).status_code == 400
        assert client.post(PATH, content=iter([b"x"])).status_code == 411


def test_viewer_cannot_submit_predictions(monkeypatch):
    monkeypatch.setitem(
        app.dependency_overrides, authorize, lambda: {"subject": "viewer", "roles": ["inference_viewer"]}
    )
    with TestClient(app) as client:
        response = client.post(PATH, files={"image": ("test.png", b"x", "image/png")})
        assert response.status_code == 403
