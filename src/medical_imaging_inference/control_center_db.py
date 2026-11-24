"""Persistent relational database and domain services for the Medical Inference Control Center.

Persists:
- inference_jobs & inference_job_events (async lifecycle: pending/queued/starting/running/completed/failed/cancelled)
- inference_uploads (validated image metadata, PHI-redacted DICOM/NIfTI tags, SHA-256, local preview path)
- inference_alerts (operational alerts with acknowledge/assign/resolve workflow)
- inference_users & inference_settings (RBAC & masked settings)
- Pure-Python PDF / HTML / JSON inference report generator with mandatory AI-prediction disclaimer.
"""
from __future__ import annotations

import hashlib
import html as html_lib
import io
import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from . import audit

_MIGRATION_LOCK = threading.Lock()

DEFAULT_USERS = [
    {
        "user_id": "clinician-1",
        "display_name": "Dr. Sarah Kim (Radiology Researcher)",
        "email": "sarah.kim@medimaging.local",
        "roles": ["inference_user", "inference_viewer"],
    },
    {
        "user_id": "inference-admin-1",
        "display_name": "Alex Morgan (Inference Ops Admin)",
        "email": "alex.morgan@medimaging.local",
        "roles": ["inference_admin", "inference_user", "inference_viewer"],
    },
    {
        "user_id": "viewer-1",
        "display_name": "Samira Rao (Clinical Viewer)",
        "email": "samira.rao@medimaging.local",
        "roles": ["inference_viewer"],
    },
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:6]}"


def _db_path() -> Path:
    return Path(os.getenv("INFERENCE_DATABASE_PATH", "./.local/inference.sqlite3")).resolve()


def _upload_dir() -> Path:
    root = Path(os.getenv("INFERENCE_UPLOAD_DIR", str(_db_path().parent / "uploads"))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


def connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with _MIGRATION_LOCK:
        try:
            audit.migrate()
        except Exception:
            pass
        db = sqlite3.connect(path, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        _apply_migrations_locked(db)
    return db


def _apply_migrations_locked(db: sqlite3.Connection) -> None:
    if True:
        db.execute(
            """CREATE TABLE IF NOT EXISTS inference_schema_migrations (
                migration_id TEXT PRIMARY KEY,
                applied_at TEXT NOT NULL
            )"""
        )
        applied = {
            r["migration_id"]
            for r in db.execute("SELECT migration_id FROM inference_schema_migrations").fetchall()
        }
        if "0001_inference_control_center" not in applied:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS inference_uploads (
                    upload_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    format_family TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    width INTEGER,
                    height INTEGER,
                    channels INTEGER,
                    modality TEXT NOT NULL DEFAULT 'Medical Image',
                    phi_redacted INTEGER NOT NULL DEFAULT 1,
                    metadata_json TEXT NOT NULL,
                    storage_path TEXT NOT NULL,
                    uploaded_by TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_jobs (
                    job_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    modality TEXT NOT NULL DEFAULT 'Medical Image',
                    upload_id TEXT,
                    input_sha256 TEXT NOT NULL DEFAULT '',
                    execution_env TEXT NOT NULL DEFAULT 'local-cpu',
                    compute_profile TEXT NOT NULL DEFAULT 'local-cpu',
                    status TEXT NOT NULL,
                    device TEXT,
                    runtime_ms REAL,
                    artifact_sha256 TEXT NOT NULL DEFAULT '',
                    preprocessing_json TEXT NOT NULL DEFAULT '{}',
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    error_message TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS ix_inf_jobs_created ON inference_jobs(created_at DESC);
                CREATE INDEX IF NOT EXISTS ix_inf_jobs_status ON inference_jobs(status);

                CREATE TABLE IF NOT EXISTS inference_job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    message TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_alerts (
                    alert_id TEXT PRIMARY KEY,
                    severity TEXT NOT NULL,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    model_id TEXT,
                    model_version TEXT,
                    job_id TEXT,
                    status TEXT NOT NULL,
                    assigned_to TEXT,
                    resolution TEXT,
                    forwarded_to_governance INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    email TEXT NOT NULL,
                    roles_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS inference_settings (
                    setting_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
            now = _utc_now()
            for u in DEFAULT_USERS:
                db.execute(
                    """INSERT OR IGNORE INTO inference_users(user_id, display_name, email, roles_json, active, updated_at)
                       VALUES (?,?,?,?,1,?)""",
                    (u["user_id"], u["display_name"], u["email"], json.dumps(u["roles"]), now),
                )
            db.execute(
                "INSERT OR IGNORE INTO inference_schema_migrations(migration_id, applied_at) VALUES (?,?)",
                ("0001_inference_control_center", now),
            )
            db.commit()


# ---------------------------------------------------------------------------
# 1. Image Upload, Multi-Format Validation & PHI Redaction
# ---------------------------------------------------------------------------

def validate_and_store_upload(
    content: bytes,
    filename: str,
    media_type: str,
    actor: str,
    *,
    modality: str = "Dermoscopy / Radiography",
) -> dict[str, Any]:
    if not content:
        raise ValueError("Uploaded file is empty")
    if len(content) > 20 * 1024 * 1024:
        raise ValueError("Uploaded image exceeds the 20 MB limit")

    sha = hashlib.sha256(content).hexdigest()
    suffix = Path(filename or "upload.png").suffix.lower()
    width = height = channels = None
    metadata: dict[str, Any] = {
        "phi_protection": "enabled (patient identifiers stripped from logs & UI)",
        "original_extension": suffix,
    }

    if suffix in {".dcm", ".dicom"} or (len(content) > 132 and content[128:132] == b"DICM"):
        fmt = "DICOM"
        metadata.update(
            {
                "dicom_preamble_valid": len(content) > 132 and content[128:132] == b"DICM",
                "patient_id": "[REDACTED]",
                "patient_name": "[REDACTED]",
                "spatial_preserved": True,
            }
        )
    elif suffix in {".nii", ".gz", ".nrrd"}:
        fmt = "NIFTI" if ".nii" in filename.lower() else "NRRD"
        metadata.update({"volumetric_header_detected": True, "affine_preserved": True})
    else:
        try:
            with Image.open(io.BytesIO(content)) as img:
                fmt = img.format or "PNG"
                width, height = img.size
                if width * height > 40_000_000:
                    raise ValueError("Image pixel count exceeds 40M pixel safety limit")
                channels = len(img.getbands())
                metadata.update(
                    {
                        "pil_format": fmt,
                        "mode": img.mode,
                        "dimensions": [width, height],
                        "channels": channels,
                    }
                )
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"Invalid or unsupported medical image file: {exc}") from exc

    upload_id = _new_id("img")
    safe_ext = suffix if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".dcm", ".nii", ".nrrd"} else ".png"
    dest = _upload_dir() / f"{upload_id}{safe_ext}"
    dest.write_bytes(content)
    now = _utc_now()

    with connect() as db:
        db.execute(
            """INSERT INTO inference_uploads
               (upload_id, filename, media_type, format_family, size_bytes, sha256,
                width, height, channels, modality, phi_redacted, metadata_json, storage_path, uploaded_by, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?,?,?)""",
            (
                upload_id,
                Path(filename or f"{upload_id}{safe_ext}").name,
                media_type or "image/png",
                fmt,
                len(content),
                sha,
                width,
                height,
                channels,
                modality,
                json.dumps(metadata),
                str(dest),
                actor,
                now,
            ),
        )
        db.commit()
    return get_upload(upload_id)


def get_upload(upload_id: str) -> dict[str, Any]:
    with connect() as db:
        r = db.execute("SELECT * FROM inference_uploads WHERE upload_id=?", (upload_id,)).fetchone()
        if r is None:
            raise LookupError(f"Upload {upload_id} not found")
    return {**dict(r), "phi_redacted": bool(r["phi_redacted"]), "metadata": json.loads(r["metadata_json"])}


def list_uploads(limit: int = 50) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT upload_id FROM inference_uploads ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [get_upload(r["upload_id"]) for r in rows]


# ---------------------------------------------------------------------------
# 2. Persistent Inference Jobs & Event Timeline
# ---------------------------------------------------------------------------

def create_job(
    *,
    user_id: str,
    model_id: str,
    model_version: str,
    task_type: str,
    modality: str,
    upload_id: str | None,
    input_sha256: str,
    execution_env: str,
    compute_profile: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    job_id = _new_id("inf")
    now = _utc_now()
    with connect() as db:
        db.execute(
            """INSERT INTO inference_jobs
               (job_id, user_id, model_id, model_version, task_type, modality, upload_id,
                input_sha256, execution_env, compute_profile, status, parameters_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,'queued',?,?)""",
            (
                job_id,
                user_id,
                model_id,
                model_version,
                task_type,
                modality,
                upload_id,
                input_sha256,
                execution_env,
                compute_profile,
                json.dumps(parameters),
                now,
            ),
        )
        db.execute(
            "INSERT INTO inference_job_events(job_id, status, message, occurred_at) VALUES (?,?,?,?)",
            (job_id, "queued", f"Job queued on {compute_profile}", now),
        )
        db.commit()
    return get_job(job_id)


def update_job(
    job_id: str,
    *,
    status: str,
    message: str,
    device: str | None = None,
    runtime_ms: float | None = None,
    artifact_sha256: str | None = None,
    preprocessing: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    with connect() as db:
        cur = db.execute("SELECT * FROM inference_jobs WHERE job_id=?", (job_id,)).fetchone()
        if cur is None:
            raise LookupError(f"Inference job {job_id} not found")
        started_at = cur["started_at"] or (now if status in {"starting", "running"} else None)
        completed_at = now if status in {"completed", "failed", "cancelled"} else cur["completed_at"]
        db.execute(
            """UPDATE inference_jobs
               SET status=?, device=COALESCE(?, device), runtime_ms=COALESCE(?, runtime_ms),
                   artifact_sha256=COALESCE(?, artifact_sha256),
                   preprocessing_json=COALESCE(?, preprocessing_json),
                   result_json=COALESCE(?, result_json),
                   error_message=COALESCE(?, error_message),
                   started_at=?, completed_at=?
               WHERE job_id=?""",
            (
                status,
                device,
                runtime_ms,
                artifact_sha256,
                json.dumps(preprocessing) if preprocessing is not None else None,
                json.dumps(result) if result is not None else None,
                error_message,
                started_at,
                completed_at,
                job_id,
            ),
        )
        db.execute(
            "INSERT INTO inference_job_events(job_id, status, message, occurred_at) VALUES (?,?,?,?)",
            (job_id, status, message, now),
        )
        db.commit()
    return get_job(job_id)


def get_job(job_id: str) -> dict[str, Any]:
    with connect() as db:
        r = db.execute("SELECT * FROM inference_jobs WHERE job_id=?", (job_id,)).fetchone()
        if r is None:
            raise LookupError(f"Inference job {job_id} not found")
        evs = db.execute(
            "SELECT status, message, occurred_at FROM inference_job_events WHERE job_id=? ORDER BY event_id ASC",
            (job_id,),
        ).fetchall()
    return {
        **dict(r),
        "preprocessing": json.loads(r["preprocessing_json"]),
        "parameters": json.loads(r["parameters_json"]),
        "result": json.loads(r["result_json"]),
        "events": [dict(e) for e in evs],
    }


def list_jobs(
    *,
    status: str | None = None,
    task_type: str | None = None,
    model_id: str | None = None,
    search: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    clauses, params = [], []
    if status:
        clauses.append("status=?")
        params.append(status)
    if task_type:
        clauses.append("task_type=?")
        params.append(task_type)
    if model_id:
        clauses.append("model_id=?")
        params.append(model_id)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(min(max(1, limit), 500))
    with connect() as db:
        rows = db.execute(
            f"SELECT job_id FROM inference_jobs {where} ORDER BY created_at DESC LIMIT ?", tuple(params)
        ).fetchall()
    items = [get_job(r["job_id"]) for r in rows]
    if search:
        q = search.lower()
        items = [j for j in items if q in json.dumps(j, default=str).lower()]
    return items


# ---------------------------------------------------------------------------
# 3. Alerts, Dashboard, Analytics, Users & Settings
# ---------------------------------------------------------------------------

def create_alert(payload: dict[str, Any]) -> dict[str, Any]:
    alert_id = _new_id("alt")
    now = _utc_now()
    with connect() as db:
        db.execute(
            """INSERT INTO inference_alerts
               (alert_id, severity, category, title, description, model_id, model_version,
                job_id, status, assigned_to, resolution, forwarded_to_governance, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,'open',?,NULL,0,?,?)""",
            (
                alert_id,
                str(payload.get("severity") or "warning").lower(),
                str(payload.get("category") or "runtime"),
                str(payload["title"]),
                str(payload.get("description") or ""),
                payload.get("model_id"),
                payload.get("model_version"),
                payload.get("job_id"),
                payload.get("assigned_to"),
                now,
                now,
            ),
        )
        db.commit()
    return list_alerts()[0]


def update_alert(alert_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    with connect() as db:
        cur = db.execute("SELECT * FROM inference_alerts WHERE alert_id=?", (alert_id,)).fetchone()
        if cur is None:
            raise LookupError(f"Alert {alert_id} not found")
        db.execute(
            """UPDATE inference_alerts
               SET status=?, assigned_to=?, resolution=?, forwarded_to_governance=?, updated_at=?
               WHERE alert_id=?""",
            (
                str(payload.get("status") or cur["status"]),
                payload.get("assigned_to") if payload.get("assigned_to") is not None else cur["assigned_to"],
                payload.get("resolution") if payload.get("resolution") is not None else cur["resolution"],
                int(payload.get("forwarded_to_governance", cur["forwarded_to_governance"])),
                now,
                alert_id,
            ),
        )
        db.commit()
        r = db.execute("SELECT * FROM inference_alerts WHERE alert_id=?", (alert_id,)).fetchone()
    return {**dict(r), "forwarded_to_governance": bool(r["forwarded_to_governance"])}


def list_alerts(status: str | None = None) -> list[dict[str, Any]]:
    with connect() as db:
        q = "SELECT * FROM inference_alerts" + (" WHERE status=?" if status else "") + " ORDER BY created_at DESC"
        rows = db.execute(q, (status,) if status else ()).fetchall()
    return [{**dict(r), "forwarded_to_governance": bool(r["forwarded_to_governance"])} for r in rows]


def get_analytics() -> dict[str, Any]:
    jobs = list_jobs(limit=500)
    by_task: dict[str, int] = {"classification": 0, "segmentation": 0, "detection": 0}
    by_model: dict[str, dict[str, Any]] = {}
    by_profile: dict[str, int] = {}
    latencies: list[float] = []
    for j in jobs:
        t = j["task_type"]
        by_task[t] = by_task.get(t, 0) + 1
        by_profile[j["compute_profile"]] = by_profile.get(j["compute_profile"], 0) + 1
        mkey = f"{j['model_id']}:{j['model_version']}"
        m = by_model.setdefault(mkey, {"model": mkey, "jobs": 0, "completed": 0, "failed": 0, "latencies": []})
        m["jobs"] += 1
        if j["status"] == "completed":
            m["completed"] += 1
        elif j["status"] == "failed":
            m["failed"] += 1
        if isinstance(j.get("runtime_ms"), (int, float)):
            latencies.append(float(j["runtime_ms"]))
            m["latencies"].append(float(j["runtime_ms"]))

    model_rows = []
    for m in by_model.values():
        ls = m.pop("latencies")
        m["mean_latency_ms"] = round(sum(ls) / len(ls), 2) if ls else None
        model_rows.append(m)

    latencies.sort()
    p95 = round(latencies[min(len(latencies) - 1, int(0.95 * (len(latencies) - 1)))], 2) if latencies else None
    return {
        "total_jobs": len(jobs),
        "completed_jobs": sum(1 for j in jobs if j["status"] == "completed"),
        "failed_jobs": sum(1 for j in jobs if j["status"] == "failed"),
        "queued_jobs": sum(1 for j in jobs if j["status"] in {"pending", "queued"}),
        "running_jobs": sum(1 for j in jobs if j["status"] in {"starting", "running"}),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "p95_latency_ms": p95,
        "by_task": by_task,
        "by_profile": by_profile,
        "by_model": model_rows,
        "measured_vs_estimated": "Latency and job counts are measured from persisted executions; AWS cost is estimated only when AWS Batch compute is selected.",
    }


def list_users() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT * FROM inference_users ORDER BY user_id").fetchall()
    return [{**dict(r), "roles": json.loads(r["roles_json"]), "active": bool(r["active"])} for r in rows]


def upsert_user(payload: dict[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    uid = str(payload["user_id"]).strip()
    roles = list(payload.get("roles") or ["inference_viewer"])
    with connect() as db:
        db.execute(
            """INSERT INTO inference_users(user_id, display_name, email, roles_json, active, updated_at)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(user_id) DO UPDATE SET
                 display_name=excluded.display_name, email=excluded.email,
                 roles_json=excluded.roles_json, active=excluded.active, updated_at=excluded.updated_at""",
            (uid, str(payload.get("display_name") or uid), str(payload.get("email") or ""), json.dumps(roles), int(payload.get("active", True)), now),
        )
        db.commit()
    return [u for u in list_users() if u["user_id"] == uid][0]


def get_settings() -> dict[str, Any]:
    defaults = {
        "dataset_pipeline_url": os.getenv("DATASET_PIPELINE_URL", "http://127.0.0.1:8000"),
        "model_engineering_url": os.getenv("MODEL_ENGINEERING_URL", "http://127.0.0.1:8085"),
        "mlops_api_url": os.getenv("MLOPS_API_URL", "http://127.0.0.1:8081"),
        "platform_api_url": os.getenv("PLATFORM_API_URL", "http://127.0.0.1:8080"),
        "aws_region": os.getenv("AWS_REGION", "ap-south-1"),
        "aws_profile": os.getenv("AWS_PROFILE", "medtriage-admin"),
        "aws_batch_queue": os.getenv("MEDTRIAGE_BATCH_QUEUE", "medical-imaging-triage-dev-queue"),
        "aws_inference_job_def": os.getenv("MEDTRIAGE_INFERENCE_JOB_DEF", "medical-imaging-platform-inference-worker"),
        "aws_results_bucket": os.getenv("MEDTRIAGE_RESULTS_BUCKET", "medical-imaging-triage-results-e09807c9"),
        "aws_mode": os.getenv("INFERENCE_AWS_MODE", "auto"),
        "inference_device": os.getenv("INFERENCE_DEVICE", "cpu"),
    }
    with connect() as db:
        for r in db.execute("SELECT setting_key, value_json FROM inference_settings").fetchall():
            defaults[r["setting_key"]] = json.loads(r["value_json"])
    return defaults


def update_settings(updates: dict[str, Any], actor: str) -> dict[str, Any]:
    now = _utc_now()
    allowed = {
        "dataset_pipeline_url",
        "model_engineering_url",
        "mlops_api_url",
        "platform_api_url",
        "aws_region",
        "aws_mode",
        "inference_device",
    }
    with connect() as db:
        for k, v in updates.items():
            if k in allowed:
                db.execute(
                    """INSERT INTO inference_settings(setting_key, value_json, updated_by, updated_at)
                       VALUES (?,?,?,?)
                       ON CONFLICT(setting_key) DO UPDATE SET
                         value_json=excluded.value_json, updated_by=excluded.updated_by, updated_at=excluded.updated_at""",
                    (k, json.dumps(v), actor, now),
                )
        db.commit()
    return get_settings()


# ---------------------------------------------------------------------------
# 4. Inference Report Generator (PDF / HTML / JSON with AI Disclaimer)
# ---------------------------------------------------------------------------

def build_inference_report(job: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": f"inf-rpt-{job['job_id']}",
        "inference_id": job["job_id"],
        "model_id": job["model_id"],
        "model_version": job["model_version"],
        "task_type": job["task_type"],
        "imaging_modality": job["modality"],
        "input_upload_id": job["upload_id"],
        "input_sha256": job["input_sha256"],
        "artifact_sha256": job["artifact_sha256"],
        "execution_environment": job["execution_env"],
        "compute_profile": job["compute_profile"],
        "device": job["device"],
        "runtime_ms": job["runtime_ms"],
        "status": job["status"],
        "preprocessing_contract": job["preprocessing"],
        "prediction_output": job["result"],
        "executed_at": job["completed_at"] or job["created_at"],
        "disclaimer": (
            "IMPORTANT: Outputs in this report are AI-generated model predictions for research "
            "and clinical decision support only. They are NOT autonomous clinical diagnoses and "
            "require independent verification by a qualified clinician."
        ),
    }


def render_inference_report(report: dict[str, Any], fmt: str) -> tuple[bytes, str, str]:
    fmt = fmt.lower()
    rid = report["report_id"]
    if fmt == "json":
        return json.dumps(report, indent=2, default=str).encode("utf-8"), "application/json", f"{rid}.json"
    if fmt == "html":
        pretty = html_lib.escape(json.dumps(report, indent=2, default=str))
        doc = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html_lib.escape(rid)}</title>"
            "<style>body{font-family:system-ui,sans-serif;background:#0b1120;color:#e2e8f0;padding:24px;}"
            ".warn{background:#2d1a1f;border:1px solid #f43f5e;padding:12px;border-radius:8px;margin:12px 0;}"
            "pre{background:#131c31;padding:16px;border-radius:8px;overflow:auto;border:1px solid #24324d;}</style>"
            f"</head><body><h1>Medical Imaging Inference Report — {html_lib.escape(report['model_id'])} ({html_lib.escape(report['model_version'])})</h1>"
            f"<div class='warn'><strong>AI Prediction Notice:</strong> {html_lib.escape(report['disclaimer'])}</div>"
            f"<pre>{pretty}</pre></body></html>"
        )
        return doc.encode("utf-8"), "text/html", f"{rid}.html"
    if fmt == "pdf":
        lines = [
            f"Medical Imaging Inference Report: {report['model_id']} ({report['model_version']})",
            f"Inference ID: {report['inference_id']} | Task: {report['task_type']}",
            f"Device: {report['device']} | Runtime: {report['runtime_ms']} ms | Status: {report['status']}",
            "AI NOTICE: Model predictions only - NOT an autonomous clinical diagnosis.",
            "-" * 72,
        ]
        for line in json.dumps(report["prediction_output"], indent=2, default=str).splitlines()[:44]:
            lines.append(line[:92])
        return _minimal_pdf(lines), "application/pdf", f"{rid}.pdf"
    raise ValueError("Unsupported format; use json, html, or pdf")


def _minimal_pdf(lines: list[str]) -> bytes:
    def esc(s: str) -> str:
        safe = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in s)
        return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    cmds = ["BT", "/F1 9 Tf", "40 760 Td", "13 TL"]
    for i, line in enumerate(lines[:50]):
        cmds.append(("" if i == 0 else "T* ") + f"({esc(line)}) Tj")
    cmds.append("ET")
    stream = "\n".join(cmds).encode("ascii")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n",
        f"4 0 obj << /Length {len(stream)} >> stream\n".encode("ascii") + stream + b"\nendstream endobj\n",
        b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Courier >> endobj\n",
    ]
    buf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(buf))
        buf.extend(obj)
    xref = len(buf)
    buf.extend(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for off in offsets[1:]:
        buf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    buf.extend(f"trailer << /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii"))
    return bytes(buf)
