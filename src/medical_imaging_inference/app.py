"""REST inference for the exact model versions approved by the MLOps service."""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import os
import time
import uuid
from datetime import datetime, timezone

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from medical_imaging_contracts.models import InferenceRequest
from pydantic import AwareDatetime
from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import Headers

from . import audit
from .batches import BatchQueue
from .datasets import published_release
from .http_limits import UploadLimits
from .postprocessing import interpret as _interpret
from .preprocessing import InvalidImage
from .registry import ModelSelection, RegistryUnavailable, approval, catalogue, catalogue_source_name
from .runtime import ModelCache
from .security import authorize, authorize_researcher, verified_package
from .storage import artifact_store_status
from .telemetry import inference_event


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(1, value)


def _inference_timeout() -> float:
    raw = os.getenv("INFERENCE_TIMEOUT_SECONDS")
    if not raw:
        return 30.0
    try:
        value = float(raw)
    except ValueError:
        return 30.0
    return max(0.05, value)


app = FastAPI(title="Medical Imaging Inference", version="1.0.0")
_cors_origins = [
    origin.strip() for origin in os.getenv("INFERENCE_CORS_ORIGINS", "").split(",") if origin.strip()
]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
models = ModelCache(capacity=_positive_int("INFERENCE_MODEL_CACHE_SIZE", 2))
batches = BatchQueue()
MAX_BATCH_IMAGES = 8
MAX_BATCH_BYTES = 40 * 1024 * 1024
app.add_middleware(UploadLimits, image_bytes=MAX_IMAGE_BYTES, batch_bytes=MAX_BATCH_BYTES)
INFERENCE_CONCURRENCY = _positive_int("INFERENCE_CONCURRENCY", 2)
inference_slots = asyncio.Semaphore(INFERENCE_CONCURRENCY)


def _configured_deadline_seconds() -> float:
    raw = os.getenv("INFERENCE_DEADLINE_SECONDS") or os.getenv("INFERENCE_TIMEOUT_SECONDS") or "30"
    try:
        value = float(raw)
    except ValueError as exc:
        raise HTTPException(503, "INFERENCE_DEADLINE_SECONDS must be numeric") from exc
    if not math.isfinite(value) or value <= 0 or value > 300:
        raise HTTPException(503, "INFERENCE_DEADLINE_SECONDS must be greater than zero and at most 300")
    return value


def _database_reachable() -> bool:
    try:
        with audit.connect() as db:
            db.execute("SELECT prediction_id FROM prediction_records LIMIT 1").fetchone()
        return True
    except (OSError, ValueError, ImportError, SQLAlchemyError):
        return False


_running_predictions: set[asyncio.Task] = set()


def _deadline_check(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise asyncio.TimeoutError


def _prediction_finished(task: asyncio.Task) -> None:
    _running_predictions.discard(task)
    if not task.cancelled():
        task.exception()


async def _bounded_prediction(
    prediction: dict,
    model_id: str,
    version: str,
    image: UploadFile,
    started: float,
    timeout_seconds: float,
    runtime_context: dict[str, str | None],
) -> bytes:
    deadline = started + timeout_seconds
    content = await image.read(MAX_IMAGE_BYTES + 1)
    filename, media_type = image.filename or "", image.content_type or ""
    slots = inference_slots
    try:
        await asyncio.wait_for(slots.acquire(), timeout=deadline - time.perf_counter())
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "Inference deadline exceeded while waiting for an execution slot") from exc

    work = dict(prediction)

    async def execute():
        try:
            await _run_prediction(work, model_id, version, content, filename, media_type, deadline, runtime_context)
        finally:
            slots.release()

    worker = asyncio.create_task(execute())
    _running_predictions.add(worker)
    worker.add_done_callback(_prediction_finished)
    try:
        await asyncio.wait_for(asyncio.shield(worker), timeout=deadline - time.perf_counter())
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "Inference deadline exceeded during model execution") from exc
    except HTTPException:
        prediction.update(work)
        raise
    prediction.update(work)
    return content


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
    return {"status": "ok", "service": "medical-imaging-inference", "version": app.version}


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
    if not await run_in_threadpool(_database_reachable):
        raise HTTPException(503, "Inference database is unavailable")
    artifact_store, _ = await run_in_threadpool(artifact_store_status)
    return {
        "status": "ready",
        "mlops_api_configured": True,
        "database": "reachable",
        "artifact_store": artifact_store,
    }


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
    runtime_context: dict[str, str | None] = {"device": None}
    try:
        deadline_seconds = _configured_deadline_seconds()
        content = await _bounded_prediction(
            prediction, model_id, version, image, started, deadline_seconds, runtime_context
        )
    except HTTPException as exc:
        prediction.update(
            status="failed",
            error_code=f"http_{exc.status_code}",
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        audit_payload = (
            {**prediction, "device": runtime_context["device"]} if runtime_context["device"] else prediction
        )
        inference_event(audit_payload)
        await run_in_threadpool(audit.record_prediction, audit_payload, None, user["subject"])
        exc.headers = {**(exc.headers or {}), "X-Request-ID": prediction["prediction_id"]}
        raise
    prediction.update(status="completed", latency_ms=(time.perf_counter() - started) * 1000)
    audit_payload = (
        {**prediction, "device": runtime_context["device"]} if runtime_context["device"] else prediction
    )
    await run_in_threadpool(audit.record_prediction, audit_payload, content, user["subject"])
    inference_event(audit_payload)
    return prediction


async def _run_prediction(
    prediction: dict,
    model_id: str,
    version: str,
    content: bytes,
    filename: str,
    media_type: str,
    deadline: float,
    runtime_context: dict[str, str | None],
) -> None:
    _deadline_check(deadline)
    if not content:
        raise HTTPException(422, "Uploaded image is empty")
    if len(content) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image exceeds the upload size limit")
    if media_type not in {"image/jpeg", "image/png", "image/tiff", "image/webp"}:
        raise HTTPException(415, "Unsupported image media type")
    InferenceRequest(
        request_id=prediction["prediction_id"],
        model_id=model_id,
        model_version=version,
        image_sha256=hashlib.sha256(content).hexdigest(),
        content_type=media_type,
    )
    record = await _approved_record(model_id, version)
    _deadline_check(deadline)
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
        runtime_context["device"] = model.device
        record_metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
        lineage = {}
        for key in (
            "experiment_id",
            "dataset_release_id",
            "dataset_receipt_digest",
            "manifest_digests",
            "git_commit",
            "config_digest",
        ):
            if key in manifest:
                lineage[key] = manifest[key]
            elif key in record_metadata:
                lineage[key] = record_metadata[key]
        if record.get("approved_at"):
            lineage["approved_at"] = record["approved_at"]
        if record.get("approval_reference"):
            lineage["approval_reference"] = record["approval_reference"]
        prediction.update(
            task=manifest["task"],
            artifact_sha256=record["artifact_digest"],
            lineage=lineage,
        )
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        raise HTTPException(503, "Approved package failed integrity or input contract validation") from exc
    _deadline_check(deadline)
    try:
        pixels, original_size = await run_in_threadpool(
            spec.transform,
            content,
            filename=filename,
            media_type=media_type,
            max_pixels=MAX_IMAGE_PIXELS,
        )
    except InvalidImage as exc:
        raise HTTPException(422, str(exc)) from exc
    _deadline_check(deadline)
    try:
        width, height = original_size if spec.resize == "native" else spec.image_size[::-1]

        def execute_and_decode():
            return _interpret(
                manifest["task"],
                model.run(pixels),
                manifest["class_mapping"],
                original_size,
                (width, height),
                framework=manifest["framework"],
                settings=manifest.get("postprocessing"),
            )

        output = await run_in_threadpool(execute_and_decode)
        _deadline_check(deadline)
    except asyncio.TimeoutError as exc:
        raise HTTPException(504, "Approved model inference timed out") from exc
    except (ValueError, OSError, KeyError, RuntimeError, ImportError) as exc:
        raise HTTPException(503, "Approved model could not execute inference") from exc
    original_width, original_height = original_size
    prediction.update(
        output=output,
        input_dimensions={"width": original_width, "height": original_height},
        disclaimer="Model-generated research prediction; not a diagnosis.",
    )
    prediction["lineage"].update(
        input_format=media_type,
        input_dimensions=prediction["input_dimensions"],
        preprocessing_version=spec.version,
        application_version=app.version,
    )
    if isinstance(record.get("approved_at"), str):
        prediction["lineage"]["approved_at"] = record["approved_at"]


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
    model_version: str | None = None,
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
            model_version=model_version,
            status=status,
            start=start,
            end=end,
        )
    }


@app.get("/api/v1/system/metrics")
def system_metrics(user: dict = Depends(authorize)):
    return {
        **audit.metrics(user["subject"], admin="inference_admin" in user["roles"]),
        "cache": models.stats(),
    }


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


@app.get("/api/v1/models/approved")
async def list_approved_models(
    task: str | None = None,
    framework: str | None = None,
    architecture: str | None = None,
    version: str | None = None,
    _user: dict = Depends(authorize),
):
    filters = dict(
        task=task, framework=framework, architecture=architecture, version=version, approved=True
    )
    return {
        "items": [
            row
            for row in await _catalogue()
            if all(value is None or row.get(field) == value for field, value in filters.items())
        ]
    }


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


@app.get("/api/v1/models/{model_id}/versions/{version}/card")
async def model_card(model_id: str, version: str, _user: dict = Depends(authorize)):
    record = await _approved_record(model_id, version)
    try:
        return await run_in_threadpool(_card_summary, record)
    except (ValueError, OSError, TypeError) as exc:
        raise HTTPException(503, "Approved model card is unavailable") from exc


def _card_summary(record):
    import re

    from .preprocessing import PreprocessingSpec

    package, manifest = verified_package(record)
    with (package / "model_card.md").open(encoding="utf-8") as card:
        text = card.read(65537)
    if not text.strip() or len(text) > 65536:
        raise ValueError("Invalid model card")
    if os.getenv("INFERENCE_DISPLAY_TEST_METRICS") != "true":
        text = re.sub(r"(?ms)^## Evaluation summary[ \t]*$.*?(?=^## |\Z)", "", text)
    spec = PreprocessingSpec.from_manifest(manifest)
    return {"card": text, "formats": spec.formats, "preprocessing": spec.model_dump()}


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
    database = "reachable" if await run_in_threadpool(_database_reachable) else "unavailable"
    try:
        await readiness()
        governance = "reachable"
    except HTTPException:
        governance = "unavailable"
    catalogue_source = catalogue_source_name()
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
        "database": database,
        "governance": governance,
        "catalogue_source": catalogue_source,
        "catalogue_status": catalogue_status,
        "artifact_store": artifact_store,
        "s3": s3_status,
        "environment": os.getenv("INFERENCE_ENVIRONMENT", "development"),
        "deployment_version": app.version,
        "loaded_models": models.active(),
        "model_cache": models.stats(),
        "upload_limits": {
            "image_bytes": MAX_IMAGE_BYTES,
            "image_pixels": MAX_IMAGE_PIXELS,
            "batch_images": MAX_BATCH_IMAGES,
            "batch_bytes": MAX_BATCH_BYTES,
        },
        "inference_concurrency": INFERENCE_CONCURRENCY,
        "inference_timeout_seconds": _inference_timeout(),
    }
