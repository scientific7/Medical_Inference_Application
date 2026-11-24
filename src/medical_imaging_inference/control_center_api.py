"""Modular FastAPI router for the Medical Imaging Inference Control Center.

Strict architectural boundary:
- Owns inference execution, model selection from approved governance packages,
  image input/validation, preprocessing/postprocessing execution, async jobs,
  interactive visualization data, inference reports, runtime analytics, AWS compute
  management, and operational alerts.
- NEVER duplicates model training, Optuna HPO, MLflow tracking, dataset EDA,
  or governance approval workflows.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from PIL import Image
from pydantic import BaseModel, Field

from . import audit
from . import control_center_db as cdb
from .postprocessing import interpret as _interpret
from .preprocessing import InvalidImage
from .runtime import ModelCache, load_export
from .security import authorize, authorize_researcher, verified_package
from .telemetry import inference_event

router = APIRouter()
WEB_INDEX = Path(__file__).resolve().parent / "web" / "index.html"
_control_cache = ModelCache(capacity=4)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class InferenceJobRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    upload_id: str = Field(min_length=1)
    compute_profile: str = "local-cpu"
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int = Field(default=5, ge=1, le=20)
    async_mode: bool = False


class AlertCreateRequest(BaseModel):
    severity: str = "warning"
    category: str = "runtime"
    title: str = Field(min_length=1)
    description: str = ""
    model_id: str | None = None
    model_version: str | None = None
    job_id: str | None = None
    assigned_to: str | None = None


class AlertUpdateRequest(BaseModel):
    status: str | None = None
    assigned_to: str | None = None
    resolution: str | None = None
    forwarded_to_governance: bool | None = None


class UserUpsertRequest(BaseModel):
    user_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    email: str = ""
    roles: list[str] = Field(default_factory=lambda: ["inference_viewer"])
    active: bool = True


# ---------------------------------------------------------------------------
# Approved Model Discovery (Layer 3 Governance + Verified Local Packages)
# ---------------------------------------------------------------------------

def _discover_approved_models() -> list[dict[str, Any]]:
    """Discover approved model versions from Layer 3 Governance API and verified local packages."""
    found: dict[tuple[str, str], dict[str, Any]] = {}

    # 1. Verified local approved catalogue (development / standalone verified packages)
    local_cat = os.getenv("INFERENCE_LOCAL_CATALOGUE")
    if local_cat and Path(local_cat).is_file():
        try:
            raw = json.loads(Path(local_cat).read_text(encoding="utf-8"))
            rows = raw.get("items", raw) if isinstance(raw, dict) else raw
            for row in rows:
                if row.get("state") == "approved" or row.get("approved") is True:
                    key = (row["model_id"], row["version"])
                    found[key] = {**row, "state": "approved", "approved": True, "source": "local_verified_catalogue"}
        except Exception:
            pass

    # 2. Live Layer 3 MLOps Governance API (/api/v1/inference/models)
    mlops_url = cdb.get_settings()["mlops_api_url"].rstrip("/")
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(f"{mlops_url}/api/v1/inference/models")
            if r.status_code == 200:
                for row in r.json().get("items", []):
                    key = (row["model_id"], row["version"])
                    found[key] = {
                        "model_id": row["model_id"],
                        "version": row["version"],
                        "state": "approved",
                        "approved": True,
                        "task": row.get("task", "classification"),
                        "architecture": row.get("architecture", "unknown"),
                        "framework": row.get("framework", "pytorch"),
                        "modality": row.get("modality", "Medical Imaging"),
                        "artifact_path": row.get("artifact_uri", ""),
                        "artifact_digest": row.get("artifact_sha256", ""),
                        "preprocessing": row.get("preprocessing") or {},
                        "class_mapping": row.get("class_mapping") or {},
                        "model_card": row.get("model_card") or "",
                        "limitations": row.get("limitations") or [],
                        "source": "mlops_governance_api",
                    }
    except Exception:
        pass

    return list(found.values())


def _resolve_approved_model(model_id: str, version: str) -> dict[str, Any]:
    for m in _discover_approved_models():
        if m["model_id"] == model_id and m["version"] == version:
            return m
    raise LookupError(f"Approved model {model_id}:{version} not found in governance catalogue")


# ---------------------------------------------------------------------------
# Real TorchScript / SavedModel Execution Engine
# ---------------------------------------------------------------------------

def _execute_real_inference_job(job_id: str, model_rec: dict[str, Any], upload: dict[str, Any], params: dict[str, Any], actor: str) -> dict[str, Any]:
    """Execute REAL PyTorch TorchScript / TensorFlow SavedModel inference on the uploaded image."""
    started = time.perf_counter()
    cdb.update_job(job_id, status="running", message=f"Loading verified model {model_rec['model_id']}:{model_rec['version']}")
    try:
        loaded = _control_cache.get(model_rec)
        img_bytes = Path(upload["storage_path"]).read_bytes()
        pixels, original_size = loaded.preprocessing.transform(
            img_bytes,
            filename=upload["filename"],
            media_type=upload["media_type"],
            max_pixels=40_000_000,
        )
        raw_out = loaded.run(pixels)
        post_settings = dict(loaded.manifest.get("postprocessing", {}))
        if params.get("threshold") is not None:
            post_settings["threshold"] = params["threshold"]
        if params.get("top_k") is not None and loaded.manifest["task"] == "classification":
            post_settings["top_k"] = params["top_k"]

        interpreted = _interpret(
            loaded.manifest["task"],
            raw_out,
            loaded.manifest["class_mapping"],
            original_size,
            (loaded.preprocessing.image_size[1], loaded.preprocessing.image_size[0]),
            framework=loaded.manifest["framework"],
            settings=post_settings,
        )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        result_payload = {
            "task": loaded.manifest["task"],
            "framework": loaded.manifest["framework"],
            "architecture": loaded.manifest.get("architecture"),
            "device": loaded.device,
            "original_size": list(original_size),
            "model_input_size": list(loaded.preprocessing.image_size),
            "output": interpreted,
            "ai_prediction_notice": "AI-generated model prediction; requires clinical verification.",
        }
        updated = cdb.update_job(
            job_id,
            status="completed",
            message=f"Real {loaded.manifest['framework']} inference completed in {elapsed_ms} ms on {loaded.device}",
            device=loaded.device,
            runtime_ms=elapsed_ms,
            artifact_sha256=model_rec.get("artifact_digest", ""),
            preprocessing=loaded.preprocessing.model_dump(),
            result=result_payload,
        )
        # Also record in core prediction_records table & emit telemetry
        pred_record = {
            "prediction_id": job_id,
            "model_id": model_rec["model_id"],
            "model_version": model_rec["version"],
            "artifact_sha256": model_rec.get("artifact_digest", ""),
            "output": interpreted,
            "latency_ms": elapsed_ms,
            "created_at": updated["created_at"],
            "status": "completed",
            "task": loaded.manifest["task"],
            "device": loaded.device,
        }
        try:
            audit.record_prediction(pred_record, img_bytes, actor)
            inference_event(pred_record)
        except Exception:
            pass
        return updated
    except (InvalidImage, ValueError, RuntimeError, OSError) as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)
        updated = cdb.update_job(
            job_id,
            status="failed",
            message=f"Inference failed: {exc}",
            runtime_ms=elapsed_ms,
            error_message=str(exc),
        )
        cdb.create_alert(
            {
                "severity": "error",
                "category": "inference_failure",
                "title": f"Inference failed for {model_rec['model_id']}:{model_rec['version']}",
                "description": str(exc),
                "model_id": model_rec["model_id"],
                "model_version": model_rec["version"],
                "job_id": job_id,
            }
        )
        return updated


# ---------------------------------------------------------------------------
# GUI Entrypoints
# ---------------------------------------------------------------------------

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/inference-center", response_class=HTMLResponse, include_in_schema=False)
def inference_center_gui():
    if not WEB_INDEX.is_file():
        return HTMLResponse("<h1>Medical Imaging Inference Center</h1><p>GUI assets not found.</p>", status_code=500)
    return HTMLResponse(WEB_INDEX.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 1. Dashboard & Approved Model Catalogue
# ---------------------------------------------------------------------------

@router.get("/api/v1/dashboard")
def get_dashboard(_user: dict = Depends(authorize)):
    models = _discover_approved_models()
    analytics = cdb.get_analytics()
    alerts = cdb.list_alerts(status="open")
    recent_jobs = cdb.list_jobs(limit=12)
    aws = _probe_aws(cdb.get_settings())
    return {
        "kpis": {
            "available_models": len({m["model_id"] for m in models}),
            "available_versions": len(models),
            "classification_models": sum(1 for m in models if m.get("task") == "classification"),
            "segmentation_models": sum(1 for m in models if m.get("task") == "segmentation"),
            "detection_models": sum(1 for m in models if m.get("task") == "detection"),
            "total_jobs": analytics["total_jobs"],
            "completed_jobs": analytics["completed_jobs"],
            "failed_jobs": analytics["failed_jobs"],
            "queued_jobs": analytics["queued_jobs"],
            "running_jobs": analytics["running_jobs"],
            "mean_latency_ms": analytics["mean_latency_ms"],
            "p95_latency_ms": analytics["p95_latency_ms"],
            "loaded_models": _control_cache.stats()["loaded"],
            "open_alerts": len(alerts),
            "aws_status": aws["status"],
        },
        "analytics": analytics,
        "recent_jobs": recent_jobs,
        "open_alerts": alerts[:5],
        "cache_stats": _control_cache.stats(),
        "aws": aws,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/api/v1/catalogue")
def get_catalogue(
    task: str | None = None,
    search: str | None = None,
    _user: dict = Depends(authorize),
):
    items = _discover_approved_models()
    if task:
        items = [m for m in items if m.get("task") == task]
    if search:
        q = search.lower()
        items = [m for m in items if q in json.dumps(m, default=str).lower()]
    return {"items": items, "total": len(items)}


# ---------------------------------------------------------------------------
# 2. Image Uploads, Validation & Sample Medical Test Image
# ---------------------------------------------------------------------------

@router.post("/api/v1/uploads", status_code=201)
async def upload_image(
    file: UploadFile = File(...),
    modality: str = Form("Dermoscopy / Radiography"),
    user: dict = Depends(authorize_researcher),
):
    content = await file.read(20 * 1024 * 1024 + 1)
    try:
        return cdb.validate_and_store_upload(
            content, file.filename or "upload.png", file.content_type or "image/png", user["subject"], modality=modality
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/api/v1/uploads/batch", status_code=201)
async def upload_image_batch(
    files: list[UploadFile] = File(...),
    modality: str = Form("Dermoscopy / Radiography"),
    user: dict = Depends(authorize_researcher),
):
    """Validate and store a batch of up to 8 medical images with PHI redaction."""
    if not files or len(files) > 8:
        raise HTTPException(422, "Batch upload requires between 1 and 8 image files")
    stored = []
    for f in files:
        content = await f.read(20 * 1024 * 1024 + 1)
        try:
            stored.append(
                cdb.validate_and_store_upload(
                    content, f.filename or "upload.png", f.content_type or "image/png", user["subject"], modality=modality
                )
            )
        except ValueError as exc:
            raise HTTPException(422, f"{f.filename}: {exc}") from exc
    return {"items": stored, "total": len(stored)}


@router.post("/api/v1/uploads/sample", status_code=201)
def create_sample_medical_upload(
    modality: str = Query("Dermoscopy"),
    user: dict = Depends(authorize_researcher),
):
    """Generate a synthetic anonymized medical test image (lesion + anatomical phantom) for immediate live testing."""
    import numpy as np

    w, h = 256, 256
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - 128) ** 2 + (y - 128) ** 2)
    bg = np.clip(205 - 0.35 * dist, 110, 225)
    lesion = np.exp(-((x - 134) ** 2 / (2 * 34**2) + (y - 122) ** 2 / (2 * 26**2)))
    r = np.clip(bg - 110 * lesion, 20, 255).astype(np.uint8)
    g = np.clip(bg - 135 * lesion, 15, 255).astype(np.uint8)
    b = np.clip(bg - 145 * lesion, 15, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return cdb.validate_and_store_upload(
        buf.getvalue(), f"anonymized_{modality.lower().replace(' ', '_')}_phantom.png", "image/png", user["subject"], modality=modality
    )


@router.get("/api/v1/uploads")
def list_uploads(_user: dict = Depends(authorize)):
    items = cdb.list_uploads()
    return {"items": items, "total": len(items)}


@router.get("/api/v1/uploads/{upload_id}/preview")
def get_upload_preview(upload_id: str, _user: dict = Depends(authorize)):
    try:
        up = cdb.get_upload(upload_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    p = Path(up["storage_path"])
    if not p.is_file():
        raise HTTPException(404, "Uploaded file not found on disk")
    return Response(content=p.read_bytes(), media_type=up["media_type"])


# ---------------------------------------------------------------------------
# 3. Inference Execution, Async Job Management, Cancel & Retry
# ---------------------------------------------------------------------------

def _check_governance_suspension(model_id: str, version: str) -> None:
    """Block inference if Layer 3 MLOps Governance marks the model version suspended or retired."""
    mlops_url = cdb.get_settings()["mlops_api_url"].rstrip("/")
    try:
        with httpx.Client(timeout=1.5) as client:
            r = client.get(f"{mlops_url}/api/v1/inference/eligibility/{model_id}/{version}")
            if r.status_code == 200:
                data = r.json()
                if data.get("suspended"):
                    raise HTTPException(
                        403,
                        f"Model {model_id}:{version} is suspended by MLOps Governance: {data.get('suspension_reason') or 'suspended'}",
                    )
                if data.get("registered") and not data.get("eligible_for_inference", True):
                    raise HTTPException(
                        403,
                        f"Model {model_id}:{version} is not eligible for inference (lifecycle_state={data.get('lifecycle_state')})",
                    )
    except HTTPException:
        raise
    except Exception:
        pass


@router.post("/api/v1/inference", status_code=201)
def submit_inference_job(
    body: InferenceJobRequest,
    background: BackgroundTasks,
    user: dict = Depends(authorize_researcher),
):
    if body.compute_profile == "aws-gpu-g5-xlarge":
        raise HTTPException(409, "GPU compute profile aws-gpu-g5-xlarge is blocked: AWS account 2003****8406 has 0 vCPUs EC2 G/VT quota.")

    _check_governance_suspension(body.model_id, body.model_version)

    try:
        model_rec = _resolve_approved_model(body.model_id, body.model_version)
        upload = cdb.get_upload(body.upload_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc

    params = {"threshold": body.threshold, "top_k": body.top_k}
    job = cdb.create_job(
        user_id=user["subject"],
        model_id=body.model_id,
        model_version=body.model_version,
        task_type=model_rec.get("task", "classification"),
        modality=upload.get("modality", "Medical Image"),
        upload_id=upload["upload_id"],
        input_sha256=upload["sha256"],
        execution_env=body.compute_profile,
        compute_profile=body.compute_profile,
        parameters=params,
    )
    if body.async_mode:
        background.add_task(_execute_real_inference_job, job["job_id"], model_rec, upload, params, user["subject"])
        return job
    return _execute_real_inference_job(job["job_id"], model_rec, upload, params, user["subject"])


@router.get("/api/v1/inference")
@router.get("/api/v1/history")
@router.get("/api/v1/results")
def list_inference_jobs(
    status: str | None = None,
    task_type: str | None = None,
    model_id: str | None = None,
    search: str | None = None,
    limit: int = 100,
    _user: dict = Depends(authorize),
):
    items = cdb.list_jobs(status=status, task_type=task_type, model_id=model_id, search=search, limit=limit)
    return {"items": items, "total": len(items)}


@router.get("/api/v1/inference/{job_id}")
def get_inference_job(job_id: str, _user: dict = Depends(authorize)):
    try:
        return cdb.get_job(job_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/api/v1/inference/{job_id}/export")
def export_inference_output(
    job_id: str,
    format: str = Query("json", pattern="^(json|png)$"),
    _user: dict = Depends(authorize),
):
    """Export structured prediction JSON (classification/detection/segmentation) or PNG segmentation mask."""
    try:
        job = cdb.get_job(job_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    out = (job.get("result") or {}).get("output") or {}
    if format == "json":
        payload = {
            "job_id": job["job_id"],
            "model_id": job["model_id"],
            "model_version": job["model_version"],
            "task_type": job["task_type"],
            "output": out,
        }
        return Response(
            content=json.dumps(payload, indent=2).encode("utf-8"),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{job_id}_prediction.json"'},
        )
    # PNG mask export for segmentation
    if job["task_type"] != "segmentation" or not out.get("mask_runs") or not out.get("mask_shape"):
        raise HTTPException(422, "PNG export is supported only for completed segmentation jobs with mask_runs")
    import numpy as np

    h, w = int(out["mask_shape"][0]), int(out["mask_shape"][1])
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    for val, length in out["mask_runs"]:
        end = min(h * w, idx + int(length))
        flat[idx:end] = 255 if int(val) > 0 else 0
        idx = end
    mask_img = Image.fromarray(flat.reshape((h, w)))
    buf = io.BytesIO()
    mask_img.save(buf, format="PNG")
    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{job_id}_mask.png"'},
    )


@router.post("/api/v1/inference/{job_id}/cancel")
def cancel_inference_job(job_id: str, _user: dict = Depends(authorize_researcher)):
    try:
        job = cdb.get_job(job_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    if job["status"] in {"completed", "failed", "cancelled"}:
        raise HTTPException(409, f"Cannot cancel job in terminal state '{job['status']}'")
    return cdb.update_job(job_id, status="cancelled", message="Cancelled by user")


@router.post("/api/v1/inference/{job_id}/retry", status_code=201)
def retry_inference_job(job_id: str, user: dict = Depends(authorize_researcher)):
    try:
        old = cdb.get_job(job_id)
        model_rec = _resolve_approved_model(old["model_id"], old["model_version"])
        upload = cdb.get_upload(old["upload_id"])
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    new_job = cdb.create_job(
        user_id=user["subject"],
        model_id=old["model_id"],
        model_version=old["model_version"],
        task_type=old["task_type"],
        modality=old["modality"],
        upload_id=old["upload_id"],
        input_sha256=old["input_sha256"],
        execution_env=old["execution_env"],
        compute_profile=old["compute_profile"],
        parameters=old["parameters"],
    )
    return _execute_real_inference_job(new_job["job_id"], model_rec, upload, old["parameters"], user["subject"])


# ---------------------------------------------------------------------------
# 4. Reports (PDF / HTML / JSON) & Analytics
# ---------------------------------------------------------------------------

@router.get("/api/v1/reports")
def list_inference_reports(_user: dict = Depends(authorize)):
    jobs = [j for j in cdb.list_jobs(limit=100) if j["status"] == "completed"]
    return {
        "items": [
            {
                "job_id": j["job_id"],
                "model_id": j["model_id"],
                "model_version": j["model_version"],
                "task_type": j["task_type"],
                "runtime_ms": j["runtime_ms"],
                "completed_at": j["completed_at"],
                "formats": ["html", "pdf", "json"],
            }
            for j in jobs
        ],
        "total": len(jobs),
    }


@router.get("/api/v1/reports/{job_id}")
def download_inference_report(
    job_id: str,
    format: str = Query("json", pattern="^(json|html|pdf)$"),
    _user: dict = Depends(authorize),
):
    try:
        job = cdb.get_job(job_id)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    report = cdb.build_inference_report(job)
    if format == "json":
        return report
    raw, mime, fname = cdb.render_inference_report(report, format)
    return Response(content=raw, media_type=mime, headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.get("/api/v1/analytics")
def get_performance_analytics(_user: dict = Depends(authorize)):
    return cdb.get_analytics()


# ---------------------------------------------------------------------------
# 5. AWS Compute Management, Workers, Monitoring & Alerts
# ---------------------------------------------------------------------------

def _probe_aws(settings: dict[str, Any]) -> dict[str, Any]:
    mode = str(settings.get("aws_mode", "auto")).lower()
    region = settings.get("aws_region", "ap-south-1")
    profile = settings.get("aws_profile", "medtriage-admin")
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "region": region,
        "profile": profile,
        "batch_queue": settings.get("aws_batch_queue", "medical-imaging-triage-dev-queue"),
        "inference_job_definition": settings.get("aws_inference_job_def", "medical-imaging-platform-inference-worker"),
        "results_bucket": settings.get("aws_results_bucket", "medical-imaging-triage-results-e09807c9"),
        "gpu_quota_vcpus": 0,
        "gpu_quota_note": "EC2 G/VT quota is 0 vCPUs in account 2003****8406; GPU profile guarded until quota approved.",
        "checked_at": now,
    }
    if mode == "mock":
        return {**base, "status": "mock", "mode": "mock", "account_masked": "2003****8406 (mock)"}
    try:
        env = {**os.environ, "AWS_PROFILE": profile, "AWS_REGION": region}
        proc = subprocess.run(
            ["aws", "sts", "get-caller-identity", "--output", "json"],
            capture_output=True,
            text=True,
            timeout=3,
            env=env,
        )
        if proc.returncode == 0:
            ident = json.loads(proc.stdout)
            acct = str(ident.get("Account", ""))
            masked = f"{acct[:4]}****{acct[-4:]}" if len(acct) == 12 else "****"
            return {**base, "status": "connected", "mode": "live", "account_masked": masked}
    except Exception:
        pass
    return {**base, "status": "mock", "mode": "mock_fallback", "account_masked": "2003****8406 (offline fallback)"}


@router.get("/api/v1/compute")
def get_compute_overview(_user: dict = Depends(authorize)):
    settings = cdb.get_settings()
    aws = _probe_aws(settings)
    profiles = [
        {
            "profile_id": "local-cpu",
            "name": "Local Verified PyTorch/TF Runtime (CPU)",
            "provider": "local",
            "device": "cpu",
            "available": True,
            "estimated_cost_per_job_usd": 0.0,
        },
        {
            "profile_id": "aws-batch-cpu",
            "name": "AWS Batch CPU Worker (m7i.xlarge)",
            "provider": "aws_batch",
            "device": "cpu",
            "available": aws["status"] == "connected",
            "job_definition": aws["inference_job_definition"],
            "estimated_cost_per_job_usd": 0.02,
        },
        {
            "profile_id": "aws-gpu-g5-xlarge",
            "name": "AWS Batch GPU Worker (g5.xlarge · NVIDIA A10G)",
            "provider": "aws_batch",
            "device": "cuda:0",
            "available": False,
            "blocked_reason": "EC2 G/VT quota is 0 vCPUs in account 2003****8406",
            "estimated_cost_per_job_usd": 0.15,
        },
    ]
    return {"aws": aws, "profiles": profiles, "cache": _control_cache.stats(), "active_models": _control_cache.active()}


@router.get("/api/v1/workers")
def get_workers(_user: dict = Depends(authorize)):
    return {
        "workers": [
            {
                "worker_id": "local-torchscript-worker-1",
                "status": "healthy",
                "device": os.getenv("INFERENCE_DEVICE", "cpu"),
                "loaded_models": _control_cache.active(),
                "cache_stats": _control_cache.stats(),
            }
        ]
    }


@router.get("/api/v1/monitoring")
def get_monitoring(_user: dict = Depends(authorize)):
    return {
        "analytics": cdb.get_analytics(),
        "alerts": cdb.list_alerts(),
        "cache": _control_cache.stats(),
        "aws": _probe_aws(cdb.get_settings()),
    }


@router.get("/api/v1/alerts")
def get_alerts(status: str | None = None, _user: dict = Depends(authorize)):
    items = cdb.list_alerts(status=status)
    return {"items": items, "total": len(items)}


@router.post("/api/v1/alerts", status_code=201)
def post_alert(body: AlertCreateRequest, _user: dict = Depends(authorize_researcher)):
    return cdb.create_alert(body.model_dump())


@router.patch("/api/v1/alerts/{alert_id}")
def patch_alert(alert_id: str, body: AlertUpdateRequest, _user: dict = Depends(authorize_researcher)):
    try:
        return cdb.update_alert(alert_id, body.model_dump(exclude_none=True))
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


# ---------------------------------------------------------------------------
# 6. Integration Center, Administration & Settings
# ---------------------------------------------------------------------------

def _probe_http(url: str) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(url)
            ok = 200 <= r.status_code < 300
            return {"status": "connected" if ok else "degraded", "http_status": r.status_code, "checked_at": now}
    except Exception as exc:
        return {"status": "offline", "http_status": None, "checked_at": now, "error": str(exc)}


@router.get("/api/v1/integrations")
def get_integrations(_user: dict = Depends(authorize)):
    s = cdb.get_settings()
    services = [
        {"key": "dataset_pipeline", "name": "Layer 1 — Medical Dataset Pipeline", "repo": "scientific7/Medical_Dataset_Pipeline", "url": s["dataset_pipeline_url"], "probe": _probe_http(s["dataset_pipeline_url"].rstrip("/") + "/health")},
        {"key": "model_engineering", "name": "Layer 2 — Medical Imaging Models Studio", "repo": "scientific7/medical_imaging_model", "url": s["model_engineering_url"], "probe": _probe_http(s["model_engineering_url"].rstrip("/") + "/healthz")},
        {"key": "mlops_governance", "name": "Layer 3 — MLOps & Model Governance", "repo": "scientific7/MLOps-and-model-governance", "url": s["mlops_api_url"], "probe": _probe_http(s["mlops_api_url"].rstrip("/") + "/health")},
        {"key": "main_platform", "name": "Layer 5 — Unified Medical Imaging Platform", "repo": "scientific7/medical-imaging-platform-", "url": s["platform_api_url"], "probe": _probe_http(s["platform_api_url"].rstrip("/") + "/health")},
    ]
    return {"services": services, "aws": _probe_aws(s), "checked_at": datetime.now(timezone.utc).isoformat()}


@router.get("/api/v1/admin/users")
def get_admin_users(_user: dict = Depends(authorize)):
    items = cdb.list_users()
    return {"items": items, "total": len(items)}


@router.post("/api/v1/admin/users", status_code=201)
def post_admin_user(body: UserUpsertRequest, user: dict = Depends(authorize)):
    if "inference_admin" not in user.get("roles", []):
        raise HTTPException(403, "inference_admin role is required")
    return cdb.upsert_user(body.model_dump())


@router.get("/api/v1/settings")
def get_settings_api(_user: dict = Depends(authorize)):
    s = cdb.get_settings()
    return {"settings": s, "aws": _probe_aws(s), "secrets_masked": True}


@router.put("/api/v1/settings")
def put_settings_api(updates: dict[str, Any], user: dict = Depends(authorize)):
    if "inference_admin" not in user.get("roles", []):
        raise HTTPException(403, "inference_admin role is required")
    return {"settings": cdb.update_settings(updates, user["subject"])}
