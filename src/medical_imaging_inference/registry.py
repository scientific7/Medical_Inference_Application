"""Catalogue discovery and independent exact-version governance checks."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from .evaluations import evaluation_record


class ModelSelection(BaseModel):
    model_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=128)
    version: str = Field(pattern=r"^[A-Za-z0-9_.-]+$", max_length=128)


class RegistryUnavailable(Exception):
    """A configured catalogue or approval source could not be read."""


async def approval(model_id: str, version: str) -> dict | None:
    base = os.getenv("MLOPS_API_URL", "http://127.0.0.1:8081").rstrip("/")
    token = os.getenv("MLOPS_SERVICE_TOKEN")
    if not token:
        raise RegistryUnavailable("Governance service identity is not configured")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(
                f"{base}/api/v1/models/{quote(model_id, safe='')}/approved/{quote(version, safe='')}",
                headers={"Authorization": f"Bearer {token}"},
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            record = response.json()
            if not isinstance(record, dict):
                raise ValueError("Expected a governance record")
            if (record.get("state"), record.get("model_id"), record.get("version")) != (
                "approved",
                model_id,
                version,
            ):
                raise ValueError("Governance identity does not match")
    except (httpx.HTTPError, ValueError) as exc:
        raise RegistryUnavailable("Could not verify model approval with governance") from exc
    return record


async def mlflow_candidates(base: str) -> list[dict]:
    token = os.getenv("MLFLOW_TRACKING_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    params: dict[str, str | int] = {"max_results": 100}
    candidates = {}
    seen_tokens = set()
    try:
        async with httpx.AsyncClient(timeout=10, headers=headers) as client:
            while True:
                response = await client.get(
                    f"{base.rstrip('/')}/api/2.0/mlflow/model-versions/search",
                    params=params,
                )
                response.raise_for_status()
                page = response.json()
                for row in page.get("model_versions", []):
                    tags = {tag["key"]: tag["value"] for tag in row.get("tags", [])}
                    # Governance versions may be semantic while MLflow uses counters.
                    model_id = tags.get("governance.model_id", row["name"])
                    version = tags.get("governance.version", str(row["version"]))
                    candidates.setdefault(
                        (model_id, version),
                        {
                            "model_id": model_id,
                            "version": version,
                            "mlflow_name": row["name"],
                            "mlflow_version": str(row["version"]),
                            "training_run_id": row.get("run_id"),
                            "task": tags.get("task"),
                            "framework": tags.get("framework"),
                            "architecture": tags.get("architecture"),
                            "created_at_ms": row.get("creation_timestamp"),
                            "registry_status": row.get("status"),
                        },
                    )
                next_token = page.get("next_page_token")
                if not next_token:
                    return list(candidates.values())
                if next_token in seen_tokens:
                    raise ValueError("Repeated catalogue page token")
                seen_tokens.add(next_token)
                params["page_token"] = next_token
    except (httpx.HTTPError, ValueError, TypeError, AttributeError, KeyError) as exc:
        raise RegistryUnavailable("MLflow catalogue is unavailable") from exc


async def catalogue() -> list[dict]:
    base = os.getenv("MLFLOW_TRACKING_URI")
    local = os.getenv("INFERENCE_LOCAL_CATALOGUE")
    if base:
        candidates = await mlflow_candidates(base)
    elif local and os.getenv("INFERENCE_ENVIRONMENT", "development") == "development":
        try:
            raw = await asyncio.to_thread(Path(local).read_text, encoding="utf-8")
            candidates = [ModelSelection.model_validate(row).model_dump() for row in json.loads(raw)]
        except (OSError, ValueError, TypeError) as exc:
            raise RegistryUnavailable("Local development catalogue is unavailable") from exc
    else:
        raise RegistryUnavailable("Model catalogue is not configured")

    # Use bounded fan-out and fail visibly if approval cannot be established.
    entries = []
    for offset in range(0, len(candidates), 8):
        entries.extend(
            await asyncio.gather(*(_with_approval(row) for row in candidates[offset : offset + 8]))
        )
    return entries


async def _with_approval(candidate: dict) -> dict:
    record = await approval(candidate["model_id"], candidate["version"])
    result = {**candidate, "approved": record is not None}
    if record:
        metadata = record.get("metadata", {})
        for field in (
            "task",
            "architecture",
            "framework",
            "experiment_id",
            "dataset_release_id",
            "dataset_receipt_digest",
            "git_commit",
            "config_digest",
            "preprocessing",
            "class_mapping",
        ):
            if field in metadata:
                result[field] = metadata[field]
        evaluation = evaluation_record(candidate, metadata)
        if evaluation:
            result["evaluation"] = evaluation
        result.update(
            created_at=record.get("created_at"),
            approved_at=record.get("approved_at"),
            artifact_digest=record.get("artifact_digest"),
        )
    return result
