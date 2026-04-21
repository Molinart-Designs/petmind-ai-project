import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from src.core.config import settings


_RESERVED_LOG_RECORD_FIELDS = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": settings.environment,
            "service": settings.project_name,
            "version": settings.app_version,
        }

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_FIELDS and not key.startswith("_")
        }

        if extra_fields:
            payload["extra"] = extra_fields

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class PlainTextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        base_message = (
            f"{timestamp} | {record.levelname:<8} | {record.name} | {record.getMessage()}"
        )

        extra_fields = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_RECORD_FIELDS and not key.startswith("_")
        }

        if extra_fields:
            base_message += f" | extra={json.dumps(extra_fields, default=str, ensure_ascii=False)}"

        if record.exc_info:
            base_message += "\n" + self.formatException(record.exc_info)

        return base_message


def _build_handler() -> logging.Handler:
    handler = logging.StreamHandler(sys.stdout)

    if settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(PlainTextFormatter())

    return handler


def configure_logging() -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(settings.log_level)

    if root_logger.handlers:
        root_logger.handlers.clear()

    root_logger.addHandler(_build_handler())


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)

    if not logging.getLogger().handlers:
        configure_logging()

    logger.setLevel(settings.log_level)
    logger.propagate = True
    return logger