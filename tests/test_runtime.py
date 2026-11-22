from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medical_imaging_inference import runtime
from medical_imaging_inference.app import app
from medical_imaging_inference.security import authorize


def test_cache_reuses_verified_version_and_keys_by_digest(monkeypatch):
    record = {"state": "approved", "model_id": "lesion", "version": "1", "artifact_digest": "a"}
    loads = []

    def verify(metadata):
        loads.append(metadata["artifact_digest"])
        return Path("."), {}

    monkeypatch.setattr(runtime, "verified_package", verify)
    monkeypatch.setattr(runtime, "load_export", lambda *_: object())
    cache = runtime.ModelCache(capacity=1)
    loaded = cache.get(record)
    assert cache.get(record) is loaded
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
