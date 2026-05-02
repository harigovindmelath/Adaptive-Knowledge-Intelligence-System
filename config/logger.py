"""
config/logger.py
----------------
Structured JSON logger for the entire AKIS pipeline.
Every log record is emitted as a JSON line to both stdout and a rolling file.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from config.loader import get as cfg


class _JsonFormatter(logging.Formatter):
    """Formats every log record as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Attach any extra structured fields
        for key, val in record.__dict__.items():
            if key.startswith("_") or key in (
                "msg", "args", "levelname", "levelno", "name", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                continue
            payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured with JSON output.
    Multiple calls with the same name return the same logger (std Python behaviour).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured

    level_name: str = cfg("system", "log_level", "INFO")
    logger.setLevel(getattr(logging, level_name, logging.INFO))

    formatter = _JsonFormatter()

    # — stdout handler
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # — file handler (rolling by day is skipped for CPU simplicity; use plain FileHandler)
    log_dir = Path(cfg("system", "log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_dir / "akis.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.propagate = False
    return logger
