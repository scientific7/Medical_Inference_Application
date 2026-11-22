import httpx
from fastapi.testclient import TestClient

from medical_imaging_inference import app as application
from medical_imaging_inference.security import authorize


def test_model_details_match_published_receipt_and_survive_outage(monkeypatch):
    model = {
        "model_id": "lesion",
        "version": "1",
        "task": "classification",
        "dataset_release_id": "published/release",
        "dataset_receipt_digest": "approved-digest",
    }

    async def catalogue():
        return [model]

    monkeypatch.setattr(application, "catalogue", catalogue)
    monkeypatch.setenv("INFERENCE_DATASET_API_URL", "https://dataset.example")
    monkeypatch.setenv("INFERENCE_DATASET_API_TOKEN", "test-only-token")
    monkeypatch.setitem(
        application.app.dependency_overrides, authorize, lambda: {"roles": ["inference_viewer"]}
    )
    record = {
        "release_id": model["dataset_release_id"],
        "dataset_id": "dataset",
        "task": model["task"],
        "release_receipt_digest": model["dataset_receipt_digest"],
        "release_status": "published",
        "release_receipt_uri": "s3://private/receipt",
    }
    unavailable = False

    def respond(request):
        assert request.url.path == "/api/v1/published-releases"
        assert request.headers["Authorization"] == "Bearer test-only-token"
        return httpx.Response(503) if unavailable else httpx.Response(200, json={"items": [record]})

    original_client = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **options: original_client(transport=transport, **options)
    )
    with TestClient(application.app) as client:
        path = "/api/v1/models/lesion/versions/1"
        detail = client.get(path).json()
        assert detail["dataset_publication"]["dataset_id"] == "dataset"
        assert "release_receipt_uri" not in detail["dataset_publication"]
        record["release_receipt_digest"] = "different-digest"
        assert client.get(path).json()["dataset_publication"] == {"status": "provenance_mismatch"}
        unavailable = True
        detail = client.get(path).json()
        assert detail["dataset_publication"] == {"status": "unavailable"}
        assert detail["dataset_receipt_digest"] == "approved-digest"
