"""Project approved training evidence into API-safe evaluation records."""

import math
import os


def numeric_metrics(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, metric in value.items():
        if isinstance(metric, dict):
            nested = numeric_metrics(metric)
            if nested:
                result[key] = nested
        elif type(metric) in (float, int) and math.isfinite(metric):
            result[key] = metric
    return result


def evaluation_record(candidate: dict, metadata: dict) -> dict | None:
    # The inspected package writer derives evaluation_summary from test reports.
    # Operators explicitly enable display for this deployment's approved models.
    if os.getenv("INFERENCE_DISPLAY_TEST_METRICS") != "true":
        return None
    values = numeric_metrics(metadata.get("evaluation_summary"))
    if not values:
        return None
    return {
        "model_id": candidate["model_id"],
        "version": candidate["version"],
        "task": metadata.get("task"),
        "architecture": metadata.get("architecture"),
        "dataset_release_id": metadata.get("dataset_release_id"),
        "dataset_receipt_digest": metadata.get("dataset_receipt_digest"),
        "test_manifest_digest": (metadata.get("manifest_digests") or {}).get("test"),
        "protocol_digest": metadata.get("evaluation_protocol_digest"),
        "split": "test",
        "metrics": values,
    }


def comparison_key(report: dict) -> tuple | None:
    key = tuple(
        report.get(field)
        for field in ("task", "dataset_receipt_digest", "test_manifest_digest", "split", "protocol_digest")
    )
    return key if all(key) else None
