from pathlib import Path

import httpx
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_dashboard_overview_uses_backend_records(monkeypatch):
    calls = []

    def request(method, url, **kwargs):
        path = httpx.URL(url).path
        calls.append(path)
        payloads = {
            "/health": {"status": "ok"},
            "/api/v1/system/metrics": {
                "total": 4,
                "successful": 3,
                "failed": 1,
                "average_latency_ms": 12,
                "daily_requests": [],
                "tasks": [],
            },
            "/api/v1/inference/history": {"items": []},
            "/api/v1/models": {"items": [{"model_id": "test", "version": "1", "approved": True}]},
            "/api/v1/models/active": {"items": []},
        }
        assert kwargs["headers"]["Authorization"] == "Bearer mock-token"
        return httpx.Response(200, json=payloads[path])

    monkeypatch.setattr(httpx, "request", request)
    frontend = Path(__file__).resolve().parents[1] / "apps" / "frontend"
    monkeypatch.syspath_prepend(str(frontend))
    app = AppTest.from_file(str(frontend / "app.py")).run()
    app.text_input[0].set_value("mock-token").run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Requests"] == "4"
    assert "/api/v1/models" in calls
