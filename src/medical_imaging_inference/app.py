"""REST inference for the exact model versions approved by the MLOps service."""

from __future__ import annotations

import asyncio
import io
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import AwareDatetime
from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import Headers

from . import audit
from .batches import BatchQueue
from .datasets import published_release
from .http_limits import UploadLimits
from .postprocessing import interpret as _interpret
from .preprocessing import InvalidImage
from .registry import ModelSelection, RegistryUnavailable, approval, catalogue
from .runtime import ModelCache
from .security import authorize, authorize_researcher
from .storage import artifact_store_status
from .telemetry import inference_event

app = FastAPI(title="Medical Imaging Inference", version="1.0.0")
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
models = ModelCache()
batches = BatchQueue()
MAX_BATCH_IMAGES = 8
MAX_BATCH_BYTES = 40 * 1024 * 1024
app.add_middleware(UploadLimits, image_bytes=MAX_IMAGE_BYTES, batch_bytes=MAX_BATCH_BYTES)
INFERENCE_CONCURRENCY = 2
inference_slots = asyncio.Semaphore(INFERENCE_CONCURRENCY)


async def _approved_record(model_id: str, version: str) -> dict:
    try:
        record = await approval(model_id, version)
    except RegistryUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    if record is None:
        raise HTTPException(404, "Requested model version is not approved")
    return record


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
async def readiness():
    import httpx

    base = os.getenv("MLOPS_API_URL", "http://127.0.0.1:8081").rstrip("/")
    token = os.getenv("MLOPS_SERVICE_TOKEN", "")
    if not token:
        raise HTTPException(503, "Model registry identity is not configured")
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base}/health")
        response.raise_for_status()
    except Exception as exc:
        raise HTTPException(503, "MLOps registry is unavailable") from exc
    return {"status": "ready", "mlops_api_configured": True}


@app.post("/api/v1/predictions/{model_id}/{version}")
async def predict(
    model_id: str, version: str, image: UploadFile = File(...), user: dict = Depends(authorize_researcher)
):
    started = time.perf_counter()
    prediction = {
        "schema_version": "1.0",
        "prediction_id": str(uuid.uuid4()),
        "model_id": model_id,
        "model_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        async with inference_slots:
            content = await _run_prediction(prediction, model_id, version, image)
    except HTTPException as exc:
        prediction.update(
            status="failed",
            error_code=f"http_{exc.status_code}",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        inference_event(prediction)
        await run_in_threadpool(audit.record_prediction, prediction, None, user["subject"])
        exc.headers = {**(exc.headers or {}), "X-Request-ID": prediction["prediction_id"]}
        raise
    prediction.update(status="completed", latency_ms=(time.perf_counter() - started) * 1000)
    await run_in_threadpool(audit.record_prediction, prediction, content, user["subject"])
    inference_event(prediction)
    return prediction


async def _run_prediction(prediction: dict, model_id: str, version: str, image: UploadFile) -> bytes:
    content = await image.read(MAX_IMAGE_BYTES + 1)
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the upload size limit")
    if image.content_type not in {"image/jpeg", "image/png", "image/tiff", "image/webp"}:
        raise HTTPException(415, "Unsupported image media type")
    record = await _approved_record(model_id, version)
    if (
        record.get("state") != "approved"
        or record.get("model_id") != model_id
        or record.get("version") != version
    ):
        raise HTTPException(409, "MLOps returned a different or unapproved model version")
    try:
        model = await run_in_threadpool(models.get, record)
        manifest = model.manifest
        spec = model.preprocessing
        prediction.update(
            task=manifest["task"], device=model.device, artifact_sha256=record["artifact_digest"],
            lineage={key: manifest[key] for key in (
                "experiment_id", "dataset_release_id", "dataset_receipt_digest",
                "manifest_digests", "git_commit", "config_digest"
            ) if key in manifest},
        )
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        raise HTTPException(503, "Approved package failed integrity or input contract validation") from exc
    try:
        pixels, original_size = await run_in_threadpool(
            spec.transform,
            content,
            filename=image.filename or "",
            media_type=image.content_type or "",
            max_pixels=MAX_IMAGE_PIXELS,
        )
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    try:
        height, width = spec.image_size
        raw = await run_in_threadpool(model.run, pixels)
        output = await run_in_threadpool(
            _interpret,
            manifest["task"],
            raw,
            manifest["class_mapping"],
            original_size,
            (width, height),
            framework=manifest["framework"],
            settings=manifest.get("postprocessing"),
        )
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        raise HTTPException(503, "Approved model could not execute inference") from exc
    original_width, original_height = original_size
    prediction.update(
        output=output,
        input_dimensions={"width": original_width, "height": original_height},
        disclaimer="Model-generated prediction for research; not a diagnosis.",
    )
    return content


async def _process_batch(batch_id, selection, payloads, user):
    batches.update(batch_id, status="running")
    failures = 0
    try:
        for index, (filename, media_type, content) in enumerate(payloads):
            image = UploadFile(
                io.BytesIO(content), filename=filename, headers=Headers({"content-type": media_type})
            )
            try:
                prediction = await predict(selection.model_id, selection.version, image, user)
                item = {"index": index, "status": "completed", "prediction_id": prediction["prediction_id"]}
            except HTTPException as exc:
                failures += 1
                item = {
                    "index": index,
                    "status": "failed",
                    "error_code": f"http_{exc.status_code}",
                    "prediction_id": (exc.headers or {}).get("X-Request-ID"),
                }
            finally:
                await image.close()
            batches.update(batch_id, item=item)
    except Exception:
        batches.update(batch_id, status="failed")
        raise
    else:
        status = "failed" if failures == len(payloads) else "partial" if failures else "completed"
        batches.update(batch_id, status=status)


@app.post("/api/v1/inference/batch", status_code=202)
async def submit_batch(
    background: BackgroundTasks,
    model_id: str = Form(...),
    version: str = Form(...),
    images: list[UploadFile] = File(...),
    user: dict = Depends(authorize_researcher),
):
    try:
        selection = ModelSelection(model_id=model_id, version=version)
    except ValueError as exc:
        raise HTTPException(422, "Invalid model identity") from exc
    if not 1 <= len(images) <= MAX_BATCH_IMAGES:
        raise HTTPException(422, f"Submit between 1 and {MAX_BATCH_IMAGES} images")
    await _approved_record(model_id, version)
    payloads = []
    remaining = MAX_BATCH_BYTES
    for image in images:
        limit = min(remaining, MAX_IMAGE_BYTES)
        data = await image.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, "Batch or image exceeds the upload size limit")
        remaining -= len(data)
        payloads.append((image.filename or "", image.content_type or "", data))
    batch_id = batches.reserve(user["subject"], len(payloads))
    if batch_id is None:
        raise HTTPException(429, "Batch queue is full")
    background.add_task(_process_batch, batch_id, selection, payloads, user)
    return {"batch_id": batch_id, "status": "queued", "total": len(payloads)}


@app.get("/api/v1/inference/batch/{batch_id}")
def batch_status(batch_id: str, user: dict = Depends(authorize)):
    job = batches.read(batch_id, user["subject"], admin="inference_admin" in user["roles"])
    if job is None:
        raise HTTPException(404, "Batch not found")
    return job


@app.get("/api/v1/inference/history")
def inference_history(
    user: dict = Depends(authorize),
    limit: int = Query(50, ge=1, le=250),
    offset: int = Query(0, ge=0),
    task: str | None = None,
    model_id: str | None = None,
    status: str | None = None,
    start: AwareDatetime | None = None,
    end: AwareDatetime | None = None,
):
    return {
        "items": audit.history(
            user["subject"],
            admin="inference_admin" in user["roles"],
            limit=limit,
            offset=offset,
            task=task,
            model_id=model_id,
            status=status,
            start=start,
            end=end,
        )
    }


@app.get("/api/v1/system/metrics")
def system_metrics(user: dict = Depends(authorize)):
    return audit.metrics(user["subject"], admin="inference_admin" in user["roles"])


@app.get("/api/v1/inference/history/{prediction_id}")
@app.get("/api/v1/predictions/{prediction_id}")
def prediction_history(prediction_id: str, user: dict = Depends(authorize)):
    row = audit.get_prediction(prediction_id, user["subject"], admin="inference_admin" in user["roles"])
    if row is None:
        raise HTTPException(404, "Prediction not found")
    return row


def authorize_admin(user: dict = Depends(authorize)) -> dict:
    if "inference_admin" not in user["roles"]:
        raise HTTPException(403, "Inference administrator role is required")
    return user


@app.get("/api/v1/models/active")
def active_models(_user: dict = Depends(authorize)):
    return {"items": models.active()}


@app.post("/api/v1/models/load")
async def load_model(selection: ModelSelection, _user: dict = Depends(authorize_admin)):
    record = await _approved_record(selection.model_id, selection.version)
    if (record.get("state"), record.get("model_id"), record.get("version")) != (
        "approved",
        selection.model_id,
        selection.version,
    ):
        raise HTTPException(409, "MLOps returned a different or unapproved model version")
    try:
        model = await run_in_threadpool(models.get, record)
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        raise HTTPException(503, "Approved model could not be loaded") from exc
    return {"status": "loaded", **selection.model_dump(), "device": model.device, "integrity": "verified"}


@app.post("/api/v1/models/unload")
def unload_model(selection: ModelSelection, _user: dict = Depends(authorize_admin)):
    if not models.unload(selection.model_id, selection.version):
        raise HTTPException(404, "Model version is not loaded")
    return {"status": "unloaded", **selection.model_dump()}


async def _catalogue() -> list[dict]:
    try:
        return await catalogue()
    except RegistryUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/api/v1/models")
async def list_models(
    task: str | None = None,
    framework: str | None = None,
    architecture: str | None = None,
    version: str | None = None,
    approved: bool | None = None,
    _user: dict = Depends(authorize),
):
    filters = dict(
        task=task, framework=framework, architecture=architecture, version=version, approved=approved
    )
    return {
        "items": [
            row
            for row in await _catalogue()
            if all(value is None or row.get(field) == value for field, value in filters.items())
        ]
    }


@app.get("/api/v1/models/{model_id}")
@app.get("/api/v1/models/{model_id}/versions")
async def model_versions(model_id: str, _user: dict = Depends(authorize)):
    versions = [row for row in await _catalogue() if row["model_id"] == model_id]
    if not versions:
        raise HTTPException(404, "Model not found in catalogue")
    return {"model_id": model_id, "items": versions}


@app.get("/api/v1/models/{model_id}/versions/{version}")
async def model_details(model_id: str, version: str, _user: dict = Depends(authorize)):
    row = await _model_version(model_id, version, _user)
    return {**row, "dataset_publication": await published_release(row)}


async def _model_version(model_id: str, version: str, user: dict):
    versions = await model_versions(model_id, user)
    for row in versions["items"]:
        if row["version"] == version:
            return row
    raise HTTPException(404, "Version not found in catalogue")


@app.get("/api/v1/evaluations")
async def list_evaluations(_user: dict = Depends(authorize)):
    return {"items": [row["evaluation"] for row in await _catalogue() if "evaluation" in row]}


@app.get("/api/v1/evaluations/{model_id}/{version}")
async def get_evaluation(model_id: str, version: str, _user: dict = Depends(authorize)):
    row = await _model_version(model_id, version, _user)
    if "evaluation" not in row:
        raise HTTPException(404, "No evaluation report is available for display")
    return row["evaluation"]


@app.get("/api/v1/system/status")
async def system_status(_user: dict = Depends(authorize_admin)):
    def database_status():
        try:
            with audit.connect() as db:
                db.execute("SELECT prediction_id FROM prediction_records LIMIT 1").fetchone()
            return "reachable"
        except (OSError, ValueError, ImportError, SQLAlchemyError):
            return "unavailable"

    try:
        await readiness()
        governance = "reachable"
    except HTTPException:
        governance = "unavailable"
    catalogue_source = (
        "mlflow"
        if os.getenv("MLFLOW_TRACKING_URI")
        else "local"
        if os.getenv("INFERENCE_LOCAL_CATALOGUE")
        else None
    )
    if catalogue_source:
        try:
            await catalogue()
            catalogue_status = "reachable"
        except RegistryUnavailable:
            catalogue_status = "unavailable"
    else:
        catalogue_status = "not configured"
    artifact_store, s3_status = await run_in_threadpool(artifact_store_status)
    return {
        "backend": "reachable",
        "database": await run_in_threadpool(database_status),
        "governance": governance,
        "catalogue_source": catalogue_source,
        "catalogue_status": catalogue_status,
        "artifact_store": artifact_store,
        "s3": s3_status,
        "environment": os.getenv("INFERENCE_ENVIRONMENT", "development"),
        "deployment_version": app.version,
        "loaded_models": models.active(),
        "upload_limits": {
            "image_bytes": MAX_IMAGE_BYTES,
            "image_pixels": MAX_IMAGE_PIXELS,
            "batch_images": MAX_BATCH_IMAGES,
            "batch_bytes": MAX_BATCH_BYTES,
        },
        "inference_concurrency": INFERENCE_CONCURRENCY,
    }
