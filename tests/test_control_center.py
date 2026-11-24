"""End-to-end tests executing REAL PyTorch TorchScript classification, segmentation, and detection inference."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from medical_imaging_inference.app import app

RESEARCHER = {"X-Dev-User": "clinician-1", "X-Dev-Roles": "inference_user,inference_viewer"}
ADMIN = {"X-Dev-User": "inference-admin-1", "X-Dev-Roles": "inference_admin,inference_user,inference_viewer"}
VIEWER = {"X-Dev-User": "viewer-1", "X-Dev-Roles": "inference_viewer"}


def _load_builder():
    script = Path(__file__).resolve().parents[1] / "scripts" / "prepare_verified_torchscript_models.py"
    spec = importlib.util.spec_from_file_location("prep_ts", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def client(tmp_path, monkeypatch):
    art_root = tmp_path / "artifacts"
    builder = _load_builder()
    cat_path = builder.build_all(art_root)

    monkeypatch.setenv("INFERENCE_ARTIFACT_ROOT", str(art_root))
    monkeypatch.setenv("INFERENCE_LOCAL_CATALOGUE", str(cat_path))
    monkeypatch.setenv("INFERENCE_DATABASE_PATH", str(tmp_path / "inf_cc.sqlite3"))
    monkeypatch.setenv("INFERENCE_UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("INFERENCE_AUTH_MODE", "dev")
    monkeypatch.setenv("INFERENCE_ENVIRONMENT", "development")
    monkeypatch.setenv("INFERENCE_AWS_MODE", "mock")
    monkeypatch.setenv("INFERENCE_DEVICE", "cpu")
    return TestClient(app)


def test_real_torchscript_classification_segmentation_and_detection_workflow(client: TestClient):
    # 1. GUI & Dashboard bootstrap
    gui = client.get("/inference-center")
    assert gui.status_code == 200
    assert "Medical Imaging Inference Center" in gui.text
    assert "Q. Settings" in gui.text

    cat = client.get("/api/v1/catalogue", headers=RESEARCHER)
    assert cat.status_code == 200
    assert cat.json()["total"] == 3

    # 2. Create real anonymized medical test image upload
    up = client.post("/api/v1/uploads/sample?modality=Dermoscopy", headers=RESEARCHER)
    assert up.status_code == 201
    upload_id = up.json()["upload_id"]
    assert up.json()["width"] == 256 and up.json()["height"] == 256
    assert up.json()["phi_redacted"] is True

    # Preview bytes work
    prev = client.get(f"/api/v1/uploads/{upload_id}/preview", headers=RESEARCHER)
    assert prev.status_code == 200
    assert prev.headers["content-type"] == "image/png"

    # 3. REAL PyTorch TorchScript Classification
    cls_job = client.post(
        "/api/v1/inference",
        json={"model_id": "isic-lesion-classifier", "model_version": "v1.0.0", "upload_id": upload_id},
        headers=RESEARCHER,
    )
    assert cls_job.status_code == 201
    cls_data = cls_job.json()
    assert cls_data["status"] == "completed"
    assert cls_data["device"] == "cpu"
    assert cls_data["result"]["output"]["predicted_class"] in {"benign", "malignant", "indeterminate"}
    assert len(cls_data["result"]["output"]["top_k"]) == 3

    # 4. REAL PyTorch TorchScript Segmentation + PNG Mask Export
    seg_job = client.post(
        "/api/v1/inference",
        json={"model_id": "lesion-unet-segmenter", "model_version": "v1.0.0", "upload_id": upload_id},
        headers=RESEARCHER,
    )
    assert seg_job.status_code == 201
    seg_data = seg_job.json()
    assert seg_data["status"] == "completed"
    assert seg_data["result"]["output"]["mask_shape"] == [256, 256]
    assert len(seg_data["result"]["output"]["mask_runs"]) >= 1
    mask_png = client.get(f"/api/v1/inference/{seg_data['job_id']}/export?format=png", headers=VIEWER)
    assert mask_png.status_code == 200
    assert mask_png.content.startswith(b"\x89PNG")

    # 5. REAL PyTorch TorchScript Detection + Structured JSON Export
    det_job = client.post(
        "/api/v1/inference",
        json={"model_id": "thoracic-nodule-detector", "model_version": "v1.0.0", "upload_id": upload_id},
        headers=RESEARCHER,
    )
    assert det_job.status_code == 201
    det_data = det_job.json()
    assert det_data["status"] == "completed"
    assert len(det_data["result"]["output"]["boxes_xyxy"]) >= 1
    det_json = client.get(f"/api/v1/inference/{det_data['job_id']}/export?format=json", headers=VIEWER)
    assert det_json.status_code == 200
    assert len(det_json.json()["output"]["boxes_xyxy"]) >= 1

    # 5b. Batch Image Upload (2 PNG files)
    batch_up = client.post(
        "/api/v1/uploads/batch",
        files=[
            ("files", ("a.png", prev.content, "image/png")),
            ("files", ("b.png", prev.content, "image/png")),
        ],
        headers=RESEARCHER,
    )
    assert batch_up.status_code == 201
    assert batch_up.json()["total"] == 2

    # 6. Reports (PDF, HTML, JSON) include AI-prediction disclaimer
    job_id = cls_data["job_id"]
    for fmt in ("json", "html", "pdf"):
        rep = client.get(f"/api/v1/reports/{job_id}?format={fmt}", headers=VIEWER)
        assert rep.status_code == 200
        if fmt == "pdf":
            assert rep.content.startswith(b"%PDF-1.4")
        elif fmt == "json":
            assert "NOT autonomous clinical diagnoses" in rep.json()["disclaimer"]

    # 7. Retry job & analytics
    retried = client.post(f"/api/v1/inference/{job_id}/retry", headers=RESEARCHER)
    assert retried.status_code == 201
    assert retried.json()["status"] == "completed"

    analytics = client.get("/api/v1/analytics", headers=VIEWER)
    assert analytics.status_code == 200
    assert analytics.json()["completed_jobs"] == 4


def test_gpu_quota_guardrail_alerts_rbac_and_integrations(client: TestClient):
    up = client.post("/api/v1/uploads/sample", headers=RESEARCHER).json()

    # GPU compute profile guarded (409)
    gpu = client.post(
        "/api/v1/inference",
        json={
            "model_id": "isic-lesion-classifier",
            "model_version": "v1.0.0",
            "upload_id": up["upload_id"],
            "compute_profile": "aws-gpu-g5-xlarge",
        },
        headers=RESEARCHER,
    )
    assert gpu.status_code == 409

    # Viewer cannot submit inference or mutate settings (403)
    assert (
        client.post(
            "/api/v1/inference",
            json={"model_id": "isic-lesion-classifier", "model_version": "v1.0.0", "upload_id": up["upload_id"]},
            headers=VIEWER,
        ).status_code
        == 403
    )
    assert client.put("/api/v1/settings", json={"aws_mode": "mock"}, headers=VIEWER).status_code == 403
    assert client.put("/api/v1/settings", json={"aws_mode": "mock"}, headers=ADMIN).status_code == 200

    # Alerts & Integrations
    al = client.post("/api/v1/alerts", json={"title": "Latency spike", "severity": "warning"}, headers=RESEARCHER)
    assert al.status_code == 201
    aid = al.json()["alert_id"]
    res = client.patch(f"/api/v1/alerts/{aid}", json={"status": "resolved"}, headers=RESEARCHER)
    assert res.status_code == 200
    assert res.json()["status"] == "resolved"

    integ = client.get("/api/v1/integrations", headers=VIEWER)
    assert integ.status_code == 200
    assert len(integ.json()["services"]) == 4
