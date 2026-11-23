"""Opt-in integration check against an isolated PostgreSQL test database."""

import os
import uuid

import pytest

from medical_imaging_inference import audit


def test_postgresql_history_contract(monkeypatch):
    url = os.getenv("INFERENCE_TEST_POSTGRES_URL")
    if not url:
        pytest.skip("PostgreSQL integration URL is not configured")
    monkeypatch.setenv("INFERENCE_DATABASE_URL", url)
    audit.migrate()
    audit.migrate()
    actor = uuid.uuid4().hex
    prediction_id = uuid.uuid4().hex
    try:
        audit.record_prediction(
            {
                "prediction_id": prediction_id,
                "model_id": "synthetic",
                "model_version": "1",
                "status": "completed",
                "created_at": "2026-01-01T00:00:00+00:00",
                "latency_ms": 5.0,
                "task": "classification",
                "output": {"predicted_class": "example"},
            },
            b"synthetic",
            actor,
        )
        assert audit.get_prediction(prediction_id, "different") is None
        assert audit.get_prediction(prediction_id, actor)["output"] == {"predicted_class": "example"}
        assert audit.history(actor, admin=False, limit=10, offset=0)[0]["prediction_id"] == prediction_id
        metrics = audit.metrics(actor, admin=False)
        assert metrics["average_latency_ms"] == 5
        assert metrics["latency_percentiles_ms"] == {"p50": 5, "p95": 5, "p99": 5}
    finally:
        with audit.connect() as db:
            db.execute("DELETE FROM prediction_records WHERE prediction_id=?", (prediction_id,))
