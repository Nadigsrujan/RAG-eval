"""
Structured Logger — JSON-formatted structured logging for all pipeline events.

Provides trace-correlated logging so every log line can be linked back
to a specific request trace.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

import structlog


def get_logger(
    name: str,
    level: str = "INFO",
    json_format: bool = True,
) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Logger name (typically __name__).
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        json_format: If True, output JSON lines; otherwise, use colored console output.

    Returns:
        A structlog BoundLogger instance.
    """
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger(name)


def setup_logging(level: str = "INFO", json_format: bool = False) -> None:
    """
    Configure root logging for the application.

    Call this once at startup (e.g., in main.py lifespan).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Suppress noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
