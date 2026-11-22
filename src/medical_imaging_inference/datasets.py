"""Read optional published-release metadata without touching dataset assets."""

import os

import httpx


async def published_release(model: dict) -> dict:
    origin = os.getenv("INFERENCE_DATASET_API_URL")
    if not origin:
        return {"status": "not_configured"}
    release_id = model.get("dataset_release_id")
    digest = model.get("dataset_receipt_digest")
    if not release_id or not digest:
        return {"status": "missing_provenance"}
    token = os.getenv("INFERENCE_DATASET_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        async with httpx.AsyncClient(timeout=3, headers=headers) as client:
            response = await client.get(
                f"{origin.rstrip('/')}/api/v1/published-releases", params={"limit": 250}
            )
            response.raise_for_status()
            releases = response.json()["items"]
        for row in releases:
            if row["release_id"] != release_id:
                continue
            if (
                row.get("release_receipt_digest") != digest
                or row.get("release_status") != "published"
                or row.get("task") != model.get("task")
            ):
                return {"status": "provenance_mismatch"}
            return {
                "status": "published",
                "release_id": release_id,
                "dataset_id": row["dataset_id"],
                "task": row["task"],
                "receipt_digest": digest,
            }
    except (httpx.HTTPError, ValueError, TypeError, KeyError, AttributeError):
        return {"status": "unavailable"}
    return {"status": "not_in_catalogue"}
