"""Operational events for the container log stream."""

import json
import logging

logger = logging.getLogger("uvicorn.error")


def inference_event(prediction: dict) -> None:
    fields = (
        "prediction_id",
        "model_id",
        "model_version",
        "task",
        "device",
        "created_at",
        "latency_ms",
        "status",
        "error_code",
    )
    values = {field: prediction[field] for field in fields if field in prediction}
    logger.info(json.dumps({"event": "inference", **values}, separators=(",", ":")))
