"""Structured (JSON-line) logging for each pipeline stage.

Every stage -- generate, heuristics, critique, decide -- emits one JSON record
carrying the shared `trace_id`, so a full run is greppable end to end. This is
the concrete version of the "AI observability" story: the reflection loop is a
pipeline you can trace, not a black box.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

LOGGER_NAME = "reflect_mcp"


class _JsonFormatter(logging.Formatter):
    """Render each record as a single JSON line with any extra `fields`."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, default=str, sort_keys=True)


def get_logger() -> logging.Logger:
    """Return the shared, idempotently-configured package logger."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        level = os.getenv("REFLECT_LOG_LEVEL", "INFO").upper()
        logger.setLevel(getattr(logging, level, logging.INFO))
        # Do not double-log through the root handler.
        logger.propagate = False
    return logger


def log_stage(stage: str, trace_id: str, **fields: Any) -> None:
    """Emit one structured record for a pipeline stage."""
    logger = get_logger()
    record_fields = {"stage": stage, "trace_id": trace_id, **fields}
    logger.info(stage, extra={"fields": record_fields})


def new_trace_id() -> str:
    """Monotonic-ish, human-sortable trace id -- no external dep."""
    return f"trace-{int(time.time() * 1000):x}-{os.getpid():x}"
