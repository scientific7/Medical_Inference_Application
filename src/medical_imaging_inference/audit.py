"""Operational history with requester-scoped reads and metadata-only writes."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, make_url

CREATE_TABLE = """CREATE TABLE IF NOT EXISTS prediction_records (
    prediction_id TEXT PRIMARY KEY, model_id TEXT NOT NULL, model_version TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL, input_sha256 TEXT NOT NULL, output_json TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL, created_at TEXT NOT NULL, status TEXT NOT NULL,
    actor TEXT NOT NULL, task TEXT, device TEXT, error_code TEXT, lineage_json TEXT)"""

INDEXES = (
    "CREATE INDEX IF NOT EXISTS ix_predictions_actor_time ON prediction_records(actor, created_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_predictions_time ON prediction_records(created_at DESC)",
)


def database_url() -> URL:
    configured = os.getenv("INFERENCE_DATABASE_URL")
    if configured:
        url = make_url(configured)
        if url.drivername != "postgresql+psycopg":
            raise ValueError("INFERENCE_DATABASE_URL requires the postgresql+psycopg driver")
        return url
    if os.getenv("INFERENCE_ENVIRONMENT") == "production":
        raise ValueError("Production requires INFERENCE_DATABASE_URL")
    path = Path(os.getenv("INFERENCE_DATABASE_PATH", "./.local/inference.sqlite3")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return URL.create("sqlite", database=str(path))


@lru_cache(maxsize=8)
def engine(url: URL):
    return create_engine(url, pool_pre_ping=True, hide_parameters=True)


def migrate():
    """Initialize the inference schema using the configured migration identity."""
    url = database_url()
    with engine(url).begin() as db:
        if url.get_backend_name() == "sqlite":
            db.exec_driver_sql("BEGIN IMMEDIATE")
        db.execute(text(CREATE_TABLE))
        if url.get_backend_name() == "sqlite":
            columns = {
                row["name"] for row in db.exec_driver_sql("PRAGMA table_info(prediction_records)").mappings()
            }
            for column in ("task", "device", "error_code", "lineage_json"):
                if column not in columns:
                    db.exec_driver_sql(f"ALTER TABLE prediction_records ADD COLUMN {column} TEXT")
        else:
            db.execute(text("ALTER TABLE prediction_records ADD COLUMN IF NOT EXISTS lineage_json TEXT"))
        for statement in INDEXES:
            db.execute(text(statement))


class Connection:
    def __init__(self, db):
        self.db = db

    def execute(self, sql, values=()):
        # Both drivers consume named binds. SQL strings are internal constants.
        parts = sql.split("?")
        statement = parts[0] + "".join(f":p{index}{part}" for index, part in enumerate(parts[1:]))
        parameters = {f"p{index}": value for index, value in enumerate(values)}
        result = self.db.execute(text(statement), parameters)
        return result.mappings() if result.returns_rows else result


@contextmanager
def connect():
    url = database_url()
    if url.get_backend_name() == "sqlite":
        migrate()
    with engine(url).begin() as db:
        yield Connection(db)


def record_prediction(prediction: dict[str, Any], input_bytes: bytes | None, actor: str) -> None:
    with connect() as db:
        db.execute(
            """INSERT INTO prediction_records (
            prediction_id, model_id, model_version, artifact_sha256, input_sha256,
            output_json, latency_ms, created_at, status, actor, task, device, error_code, lineage_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                prediction["prediction_id"],
                prediction["model_id"],
                prediction["model_version"],
                prediction.get("artifact_sha256", ""),
                hashlib.sha256(input_bytes).hexdigest() if input_bytes is not None else "",
                json.dumps(prediction.get("output"), sort_keys=True, allow_nan=False),
                prediction["latency_ms"],
                prediction["created_at"],
                prediction["status"],
                actor,
                prediction.get("task"),
                prediction.get("device"),
                prediction.get("error_code"),
                json.dumps(prediction.get("lineage", {}), sort_keys=True, allow_nan=False),
            ),
        )


SUMMARY = """prediction_id, model_id, model_version, artifact_sha256,
             latency_ms, created_at, status, task, device, error_code"""


def get_prediction(prediction_id: str, actor: str, *, admin: bool = False) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute(
            f"SELECT {SUMMARY}, output_json, lineage_json FROM prediction_records WHERE prediction_id=? AND (? OR actor=?)",
            (prediction_id, admin, actor),
        ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["output"] = json.loads(result.pop("output_json"))
    result["lineage"] = json.loads(result.pop("lineage_json") or "{}")
    return result


def history(
    actor: str,
    *,
    admin: bool,
    limit: int,
    offset: int,
    task: str | None = None,
    model_id: str | None = None,
    model_version: str | None = None,
    status: str | None = None,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[dict]:
    clauses = ["(? OR actor=?)"]
    values: list[Any] = [admin, actor]
    for field, value in (
        ("task", task),
        ("model_id", model_id),
        ("model_version", model_version),
        ("status", status),
    ):
        if value is not None:
            clauses.append(f"{field}=?")
            values.append(value)
    for operator, value in ((">=", start), ("<", end)):
        if value is not None:
            clauses.append(f"created_at {operator} ?")
            values.append(value.astimezone(timezone.utc).isoformat())
    with connect() as db:
        rows = db.execute(
            f"""SELECT {SUMMARY} FROM prediction_records WHERE {" AND ".join(clauses)}
                ORDER BY created_at DESC, prediction_id DESC LIMIT ? OFFSET ?""",
            (*values, limit, offset),
        ).fetchall()
    return [dict(row) for row in rows]


def metrics(actor: str, *, admin: bool) -> dict:
    with connect() as db:
        totals = db.execute(
            """SELECT COUNT(*) AS total,
            COUNT(CASE WHEN status='completed' THEN 1 END) AS successful,
            COUNT(CASE WHEN status='failed' THEN 1 END) AS failed,
            AVG(CASE WHEN status='completed' THEN latency_ms END) AS average_latency_ms
            FROM prediction_records WHERE (? OR actor=?)""",
            (admin, actor),
        ).fetchone()
        percentiles = db.execute(
            """WITH ranked AS (
                SELECT latency_ms, ROW_NUMBER() OVER (ORDER BY latency_ms) AS position,
                       COUNT(*) OVER () AS sample_count
                FROM prediction_records WHERE status='completed' AND (? OR actor=?)
            )
            SELECT MIN(CASE WHEN position * 100 >= sample_count * 50 THEN latency_ms END) AS p50,
                   MIN(CASE WHEN position * 100 >= sample_count * 95 THEN latency_ms END) AS p95,
                   MIN(CASE WHEN position * 100 >= sample_count * 99 THEN latency_ms END) AS p99
            FROM ranked""",
            (admin, actor),
        ).fetchone()
        tasks = db.execute(
            """SELECT COALESCE(task, 'unknown') AS task, COUNT(*) AS count
            FROM prediction_records WHERE (? OR actor=?) GROUP BY COALESCE(task, 'unknown')""",
            (admin, actor),
        ).fetchall()
        activity = db.execute(
            """SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS count
            FROM prediction_records WHERE (? OR actor=?) GROUP BY day ORDER BY day DESC LIMIT 30""",
            (admin, actor),
        ).fetchall()
        model_rows = db.execute(
            """SELECT model_id, model_version, COUNT(*) AS count,
            COUNT(CASE WHEN status='failed' THEN 1 END) AS failed
            FROM prediction_records WHERE (? OR actor=?)
            GROUP BY model_id, model_version ORDER BY count DESC, model_id ASC""",
            (admin, actor),
        ).fetchall()
        latencies = [
            float(row["latency_ms"])
            for row in db.execute(
                """SELECT latency_ms FROM prediction_records
                WHERE (? OR actor=?) AND status='completed' ORDER BY latency_ms ASC""",
                (admin, actor),
            ).fetchall()
        ]
    totals_dict = dict(totals)
    total_count = int(totals_dict.get("total") or 0)
    failed_count = int(totals_dict.get("failed") or 0)
    error_rate = round(failed_count / total_count, 4) if total_count > 0 else 0.0
    p95_latency_ms = None
    if latencies:
        index = min(len(latencies) - 1, max(0, int((len(latencies) - 1) * 0.95)))
        p95_latency_ms = latencies[index]
    return {
        **totals_dict,
        "latency_percentiles_ms": dict(percentiles),
        "error_rate": error_rate,
        "p95_latency_ms": p95_latency_ms,
        "tasks": [dict(row) for row in tasks],
        "models": [dict(row) for row in model_rows],
        "daily_requests": [dict(row) for row in reversed(activity)],
    }


if __name__ == "__main__":
    migrate()
