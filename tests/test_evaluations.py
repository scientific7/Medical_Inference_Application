from fastapi.testclient import TestClient

from medical_imaging_inference.app import app
from medical_imaging_inference.evaluations import comparison_key, evaluation_record
from medical_imaging_inference.security import authorize


def test_test_metrics_require_opt_in_and_full_comparison_lineage(monkeypatch):
    identity = {"model_id": "lesion", "version": "1"}
    manifest = {
        "task": "classification",
        "dataset_receipt_digest": "receipt",
        "manifest_digests": {"test": "split"},
        "evaluation_summary": {"auc": 0.9, "internal_path": "/private/report", "bad": float("nan")},
    }
    monkeypatch.delenv("INFERENCE_DISPLAY_TEST_METRICS", raising=False)
    assert evaluation_record(identity, manifest) is None
    monkeypatch.setenv("INFERENCE_DISPLAY_TEST_METRICS", "true")
    report = evaluation_record(identity, manifest)
    assert report["metrics"] == {"auc": 0.9}
    assert comparison_key(report) is None
    verified = evaluation_record(identity, {**manifest, "evaluation_protocol_digest": "protocol"})
    assert comparison_key(verified)
    assert comparison_key({**verified, "test_manifest_digest": "different"}) != comparison_key(verified)


def test_system_status_is_admin_only(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "history.sqlite3"))
    for key in ("MLFLOW_TRACKING_URI", "INFERENCE_LOCAL_CATALOGUE", "MLOPS_SERVICE_TOKEN"):
        monkeypatch.delenv(key, raising=False)
    roles = ["inference_user"]
    monkeypatch.setitem(app.dependency_overrides, authorize, lambda: {"subject": "test", "roles": roles})
    with TestClient(app) as client:
        assert client.get("/api/v1/system/status").status_code == 403
        roles.append("inference_admin")
        status = client.get("/api/v1/system/status").json()
        assert status["database"] == "reachable"
        assert status["governance"] == "unavailable"
        assert status["catalogue_status"] == "not configured"
