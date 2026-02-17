"""Centralized logging configuration for geryon."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_STREAM_HANDLER_ID = "geryon_stream"
_FILE_HANDLER_ID = "geryon_file"
_FORMAT = "ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s"


def _resolve_level(level: int | None) -> int:
    if level is not None:
        return level
    env_level = os.environ.get("GERYON_LOG_LEVEL", "").strip().upper()
    resolved = getattr(logging, env_level, None) if env_level else None
    if resolved is None:
        return logging.WARNING
    return int(resolved)


def _mark_handler(handler: logging.Handler, handler_id: str) -> None:
    setattr(handler, "_geryon_handler_id", handler_id)


def _get_handler(root: logging.Logger, handler_id: str) -> logging.Handler | None:
    for handler in root.handlers:
        if getattr(handler, "_geryon_handler_id", None) == handler_id:
            return handler
    return None


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``geryon`` namespace."""
    return logging.getLogger(f"geryon.{name}")


def setup_logging(*, level: int | None = None) -> None:
    """Configure the root ``geryon`` logger.

    The log level can be set via the *level* parameter or the
    ``GERYON_LOG_LEVEL`` environment variable (DEBUG, INFO, WARNING, ERROR).
    If ``GERYON_LOG_FILE`` is set, a file handler is attached and logs at
    least INFO-level lifecycle events.
    """
    stream_level = _resolve_level(level)

    root = logging.getLogger("geryon")
    stream_handler = _get_handler(root, _STREAM_HANDLER_ID)
    if stream_handler is None:
        stream_handler = logging.StreamHandler()
        _mark_handler(stream_handler, _STREAM_HANDLER_ID)
        root.addHandler(stream_handler)
    stream_handler.setFormatter(logging.Formatter(_FORMAT))
    stream_handler.setLevel(stream_level)

    file_path_raw = os.environ.get("GERYON_LOG_FILE", "").strip()
    file_level: int | None = None
    file_handler = _get_handler(root, _FILE_HANDLER_ID)
    if file_path_raw:
        file_path = Path(file_path_raw).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            file_handler is None
            or not isinstance(file_handler, logging.FileHandler)
            or Path(file_handler.baseFilename).resolve() != file_path
        ):
            if file_handler is not None:
                root.removeHandler(file_handler)
                file_handler.close()
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            _mark_handler(file_handler, _FILE_HANDLER_ID)
            root.addHandler(file_handler)
        file_level = min(stream_level, logging.INFO)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        file_handler.setLevel(file_level)
    elif file_handler is not None:
        root.removeHandler(file_handler)
        file_handler.close()

    effective_level = stream_level
    if file_level is not None:
        effective_level = min(effective_level, file_level)
    root.setLevel(effective_level)
