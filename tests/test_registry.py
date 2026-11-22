import httpx
from fastapi.testclient import TestClient

from medical_imaging_inference import registry
from medical_imaging_inference.app import app
from medical_imaging_inference.security import authorize


def test_catalogue_uses_governance_and_fails_closed(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setenv("MLOPS_API_URL", "https://governance.example")
    monkeypatch.setenv("MLOPS_SERVICE_TOKEN", "mock-token")
    offline = False

    def respond(request):
        if request.url.host == "mlflow.example":
            if offline:
                return httpx.Response(503)
            if request.url.params.get("page_token") == "next":
                return httpx.Response(200, json={"model_versions": [{"name": "candidate", "version": "1"}]})
            return httpx.Response(
                200,
                json={
                    "model_versions": [
                        {
                            "name": "registry-name",
                            "version": "3",
                            "run_id": "training-run",
                            "tags": [
                                {"key": "governance.model_id", "value": "lesion"},
                                {"key": "governance.version", "value": "v1"},
                            ],
                        }
                    ],
                    "next_page_token": "next",
                },
            )
        assert request.headers["Authorization"] == "Bearer mock-token"
        if request.url.path.endswith("/lesion/approved/v1"):
            return httpx.Response(
                200,
                json={
                    "model_id": "lesion",
                    "version": "v1",
                    "state": "approved",
                    "artifact_path": "/private/location",
                    "metadata": {"task": "classification"},
                },
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        registry.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    monkeypatch.setitem(
        app.dependency_overrides, authorize, lambda: {"subject": "test", "roles": ["inference_user"]}
    )
    with TestClient(app) as client:
        all_versions = client.get("/api/v1/models").json()["items"]
        assert [(row["model_id"], row["approved"]) for row in all_versions] == [
            ("lesion", True),
            ("candidate", False),
        ]
        detail = client.get("/api/v1/models/lesion/versions/v1").json()
        assert detail["training_run_id"] == "training-run"
        assert "artifact_path" not in detail
        filtered = client.get("/api/v1/models?approved=true&task=classification").json()["items"]
        assert filtered == [{key: value for key, value in detail.items() if key != "dataset_publication"}]
        offline = True
        assert client.get("/api/v1/models").status_code == 503
