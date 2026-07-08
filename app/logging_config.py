"""Structured JSON logging, separate from the SQLite audit trail.

SQLite is the queryable source of truth for the dashboard; these JSON lines
are for log-aggregation tools (Grafana/Loki, CloudWatch, etc.) that expect
one JSON object per line rather than a database to query.
"""
import json
import logging
import os
from datetime import datetime, timezone

REQUEST_LOGGER_NAME = "autopilot.requests"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
    root.handlers = [handler]


def log_request_event(**fields) -> None:
    logging.getLogger(REQUEST_LOGGER_NAME).info("request_completed", extra={"fields": fields})
