from fastapi.testclient import TestClient

from medical_imaging_inference import audit
from medical_imaging_inference.app import app
from medical_imaging_inference.security import authorize


def test_history_scope_metrics_and_failed_upload(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "history.sqlite3"))
    alice = {"subject": "alice", "roles": ["inference_user"]}
    monkeypatch.setitem(app.dependency_overrides, authorize, lambda: alice)
    audit.record_prediction(
        {
            "prediction_id": "private",
            "model_id": "nodule",
            "model_version": "1",
            "status": "completed",
            "created_at": "2026-01-01T00:00:00+00:00",
            "latency_ms": 4,
            "task": "detection",
            "output": {"count": 0},
        },
        b"example",
        "bob",
    )

    with TestClient(app) as client:
        response = client.post("/api/v1/predictions/nodule/1", files={"image": ("a.txt", b"x", "text/plain")})
        assert response.status_code == 415
        request_id = response.headers["x-request-id"]
        items = client.get("/api/v1/inference/history?status=failed&model_id=nodule").json()["items"]
        assert [item["prediction_id"] for item in items] == [request_id]
        assert items[0]["error_code"] == "http_415"
        assert client.get("/api/v1/inference/history/private").status_code == 404
        assert client.get("/api/v1/system/metrics").json()["total"] == 1
        alice["roles"] = ["inference_admin"]
        metrics = client.get("/api/v1/system/metrics").json()
        assert (metrics["total"], metrics["successful"], metrics["failed"]) == (2, 1, 1)
        assert metrics["average_latency_ms"] == 4
        detail = client.get("/api/v1/inference/history/private").json()
        assert detail["output"] == {"count": 0}
        assert "input_sha256" not in detail and "actor" not in detail


def test_existing_sqlite_records_remain_readable(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "legacy.sqlite3"
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(path))
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE prediction_records (
            prediction_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, model_version TEXT NOT NULL,
            artifact_sha256 TEXT NOT NULL, input_sha256 TEXT NOT NULL, output_json TEXT NOT NULL,
            latency_ms REAL NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL, actor TEXT NOT NULL)""")
        db.execute("""INSERT INTO prediction_records VALUES
            ('legacy', 'model', '1', 'digest', 'input', '{}', 3, '2026-01-01T00:00:00+00:00',
             'completed', 'owner')""")
    result = audit.get_prediction("legacy", "owner")
    assert result["output"] == {}
    assert result["task"] is None
    assert audit.metrics("owner", admin=False)["tasks"] == [{"task": "unknown", "count": 1}]
