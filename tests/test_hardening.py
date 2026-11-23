import hashlib
import io
import json
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from medical_imaging_contracts.models import PredictionResult
from PIL import Image

from medical_imaging_inference import app as application
from medical_imaging_inference import registry, runtime
from medical_imaging_inference.preprocessing import PreprocessingSpec
from medical_imaging_inference.runtime import LoadedModel, ModelCache
from medical_imaging_inference.security import authorize, verify_directory


def _make_png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (6, 4), (10, 120, 220)).save(buf, format="PNG")
    return buf.getvalue()


def _build_verified_package(root, *, model_id="lesion", version="1.0.0", weights_bytes=b"weights"):
    package = root / "pkg"
    package.mkdir(parents=True)
    weights = package / "weights" / "model.bin"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(weights_bytes)
    weights_sha = hashlib.sha256(weights_bytes).hexdigest()
    manifest = {
        "model_id": model_id,
        "version": version,
        "task": "classification",
        "framework": "pytorch",
        "class_mapping": {"benign": 0, "malignant": 1},
        "preprocessing": {"image_size": [4, 4], "color": "RGB", "scale": "[0,1]"},
        "weights": {"path": "weights/model.bin", "sha256": weights_sha},
    }
    (package / "model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    filenames = sorted(item.relative_to(package).as_posix() for item in package.rglob("*") if item.is_file())
    pkg_digest = hashlib.sha256(
        "".join(
            f"{name}:{hashlib.sha256((package / name).read_bytes()).hexdigest()}\n" for name in filenames
        ).encode()
    ).hexdigest()
    record = {
        "model_id": model_id,
        "version": version,
        "state": "approved",
        "artifact_path": str(package),
        "artifact_digest": pkg_digest,
    }
    return package, record, manifest


def test_mlops_approved_discovery_filters_non_approved_and_strips_paths(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("INFERENCE_LOCAL_CATALOGUE", raising=False)
    monkeypatch.setenv("MLOPS_API_URL", "https://governance.example")
    monkeypatch.setenv("MLOPS_SERVICE_TOKEN", "svc-token")

    def respond(request):
        assert request.url.path == "/api/v1/inference/models"
        assert request.headers["Authorization"] == "Bearer svc-token"
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "model_id": "lesion",
                        "version": "1.0.0",
                        "state": "approved",
                        "artifact_path": "/secret/path",
                        "artifact_digest": "a" * 64,
                        "metadata": {"task": "classification", "framework": "pytorch"},
                    },
                    {
                        "model_id": "rejected-model",
                        "version": "0.1.0",
                        "state": "rejected",
                        "artifact_path": "/secret/rejected",
                        "artifact_digest": "b" * 64,
                    },
                ]
            },
        )

    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(
        registry.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs)
    )
    monkeypatch.setitem(
        application.app.dependency_overrides,
        authorize,
        lambda: {"subject": "researcher", "roles": ["inference_user"]},
    )
    with TestClient(application.app) as client:
        items = client.get("/api/v1/models").json()["items"]
        assert len(items) == 1
        assert items[0]["model_id"] == "lesion"
        assert items[0]["approved"] is True
        assert "artifact_path" not in items[0]
        approved = client.get("/api/v1/models/approved?task=classification").json()["items"]
        assert approved == items


def test_verify_directory_rejects_unapproved_tampered_and_mismatched_packages(tmp_path):
    package, record, _ = _build_verified_package(tmp_path)
    manifest = verify_directory(package, record)
    assert manifest["model_id"] == "lesion"

    with pytest.raises(ValueError, match="explicit approval"):
        verify_directory(package, {**record, "state": "candidate_submitted"})

    with pytest.raises(ValueError, match="Invalid approved package digest"):
        verify_directory(package, {**record, "artifact_digest": "not-a-sha256"})

    with pytest.raises(ValueError, match="Manifest model_id"):
        verify_directory(package, {**record, "model_id": "other-model"})

    with pytest.raises(ValueError, match="Manifest version"):
        verify_directory(package, {**record, "version": "9.9.9"})

    # Corrupt weights -> weight checksum mismatch
    (package / "weights" / "model.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_directory(package, record)

    # Restore weights, add extra unexpected file -> package digest mismatch
    (package / "weights" / "model.bin").write_bytes(b"weights")
    (package / "rogue.txt").write_text("tamper", encoding="utf-8")
    with pytest.raises(ValueError, match="package digest differs"):
        verify_directory(package, record)
    (package / "rogue.txt").unlink()

    # Non-object JSON manifest
    (package / "model_manifest.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        verify_directory(package, record)


def test_load_export_validates_task_framework_and_class_mapping(tmp_path):
    with pytest.raises(ValueError, match="Unsupported model task"):
        runtime.load_export(tmp_path, {"task": "unknown", "framework": "pytorch", "class_mapping": {}})
    with pytest.raises(ValueError, match="Unsupported model framework"):
        runtime.load_export(tmp_path, {"task": "classification", "framework": "onnx", "class_mapping": {}})
    with pytest.raises(ValueError, match="class_mapping"):
        runtime.load_export(tmp_path, {"task": "classification", "framework": "pytorch"})


def test_prediction_matches_contract_v1_and_records_timeout_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "hardening.sqlite3"))
    monkeypatch.setitem(
        application.app.dependency_overrides,
        authorize,
        lambda: {"subject": "clinician", "roles": ["inference_user", "inference_admin"]},
    )
    digest = "c" * 64

    async def approved(model_id, version):
        return {
            "model_id": model_id,
            "version": version,
            "state": "approved",
            "artifact_digest": digest,
            "approved_at": "2026-01-02T03:04:05+00:00",
            "approval_reference": "apr-42",
            "metadata": {"experiment_id": "exp-9"},
        }

    manifest = {
        "task": "classification",
        "framework": "pytorch",
        "class_mapping": {"benign": 0, "malignant": 1},
        "preprocessing": {"image_size": [4, 4], "color": "RGB", "scale": "[0,1]"},
        "dataset_release_id": "rel-1",
        "dataset_receipt_digest": "d" * 64,
        "manifest_digests": {"train": "1" * 64, "validation": "2" * 64, "test": "3" * 64},
    }
    fast_model = LoadedModel(
        manifest, PreprocessingSpec.from_manifest(manifest), "cpu", lambda _: np.array([[0.2, 2.1]])
    )
    monkeypatch.setattr(application, "_approved_record", approved)
    monkeypatch.setattr(application.models, "get", lambda _: fast_model)

    with TestClient(application.app) as client:
        png = _make_png()
        resp = client.post(
            "/api/v1/predictions/lesion/1.0.0",
            files={"image": ("scan.png", png, "image/png")},
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        # Must validate strictly against PredictionResult v1.0 (extra="forbid")
        validated = PredictionResult.model_validate(payload)
        assert validated.artifact_sha256 == digest
        assert validated.lineage["experiment_id"] == "exp-9"
        assert validated.lineage["dataset_release_id"] == "rel-1"
        assert validated.lineage["approved_at"] == "2026-01-02T03:04:05+00:00"
        assert validated.lineage["approval_reference"] == "apr-42"
        assert "device" not in payload

        # Stored audit record retains device while omitting input_sha256 and actor
        stored = client.get(f"/api/v1/predictions/{validated.prediction_id}").json()
        assert stored["device"] == "cpu"
        assert "input_sha256" not in stored and "actor" not in stored

        # Empty upload rejected with 422
        empty = client.post(
            "/api/v1/predictions/lesion/1.0.0",
            files={"image": ("empty.png", b"", "image/png")},
        )
        assert empty.status_code == 422

        # Slow inference triggers 504 timeout and logs failed audit record
        def slow_run(_pixels):
            time.sleep(0.25)
            return np.array([[0.2, 2.1]])

        slow_model = LoadedModel(manifest, PreprocessingSpec.from_manifest(manifest), "cpu", slow_run)
        monkeypatch.setattr(application.models, "get", lambda _: slow_model)
        monkeypatch.setenv("INFERENCE_TIMEOUT_SECONDS", "0.05")
        timed_out = client.post(
            "/api/v1/predictions/lesion/1.0.0",
            files={"image": ("scan.png", png, "image/png")},
        )
        assert timed_out.status_code == 504
        assert "x-request-id" in timed_out.headers

        # Metrics include error_rate, p95_latency_ms, models, and cache
        metrics = client.get("/api/v1/system/metrics").json()
        assert metrics["total"] == 3
        assert metrics["successful"] == 1
        assert metrics["failed"] == 2
        assert metrics["error_rate"] == pytest.approx(2 / 3, abs=1e-3)
        assert metrics["p95_latency_ms"] is not None
        assert metrics["models"][0]["model_id"] == "lesion"
        assert "capacity" in metrics["cache"]


def test_model_cache_tracks_hits_misses_and_evictions(monkeypatch):
    monkeypatch.setattr(runtime, "verified_package", lambda record: (record["artifact_digest"], {}))
    monkeypatch.setattr(runtime, "load_export", lambda *_: object())
    cache = ModelCache(capacity=1)
    r1 = {"state": "approved", "model_id": "m", "version": "1", "artifact_digest": "1" * 64}
    r2 = {"state": "approved", "model_id": "m", "version": "2", "artifact_digest": "2" * 64}
    cache.get(r1)
    cache.get(r1)
    cache.get(r2)
    assert cache.stats() == {"capacity": 1, "loaded": 1, "hits": 1, "misses": 2, "evictions": 1}


def test_dev_auth_mode_loopback_and_production_guard(monkeypatch):
    monkeypatch.setenv("INFERENCE_AUTH_MODE", "dev")
    monkeypatch.setenv("INFERENCE_ENVIRONMENT", "development")
    with TestClient(application.app) as client:
        res = client.get(
            "/api/v1/models/active",
            headers={"X-Dev-User": "dev-alice", "X-Dev-Roles": "inference_viewer"},
        )
        assert res.status_code == 200
        # Viewer cannot invoke admin route
        assert (
            client.post(
                "/api/v1/models/unload",
                json={"model_id": "m", "version": "1"},
                headers={"X-Dev-Roles": "inference_viewer"},
            ).status_code
            == 403
        )
        # Production must reject dev auth mode (fail-closed)
        monkeypatch.setenv("INFERENCE_ENVIRONMENT", "production")
        assert client.get("/api/v1/models/active").status_code == 503


def test_health_and_readiness_checks_mlops_and_db(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "ready.sqlite3"))
    monkeypatch.setenv("MLOPS_API_URL", "https://governance.example")
    monkeypatch.setenv("MLOPS_SERVICE_TOKEN", "svc-token")
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"status": "ok"}))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))

    with TestClient(application.app) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok"
        assert health["service"] == "medical-imaging-inference"

        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["database"] == "reachable"

        monkeypatch.setattr(application, "_database_reachable", lambda: False)
        assert client.get("/ready").status_code == 503
