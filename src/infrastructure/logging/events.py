from __future__ import annotations

import logging
from typing import Any

from src.infrastructure.logging.context import get_context

_logger = logging.getLogger(__name__)


def _emit(event_type: str, level: int, message: str, **fields: Any) -> None:
    ctx = get_context()
    payload: dict[str, Any] = {
        "event_type": event_type,
        # * Do not use reserved LogRecord attribute names (e.g., 'message') in extra
    }
    # Merge context then explicit fields (explicit wins)
    payload.update(ctx)
    payload.update(fields)
    _logger.log(level, message, extra=payload)


# Engine lifecycle / control


def log_engine_event(message: str, **fields: Any) -> None:
    _emit("engine_event", logging.INFO, message, **fields)


def log_engine_warning(message: str, **fields: Any) -> None:
    _emit("engine_event", logging.WARNING, message, **fields)


def log_engine_error(message: str, **fields: Any) -> None:
    _emit("engine_event", logging.ERROR, message, **fields)


# Decision and order-related


def log_decision_event(message: str, **fields: Any) -> None:
    _emit("decision_event", logging.INFO, message, **fields)


def log_order_event(message: str, **fields: Any) -> None:
    _emit("order_event", logging.INFO, message, **fields)


# Risk management


def log_risk_event(message: str, **fields: Any) -> None:
    _emit("risk_event", logging.INFO, message, **fields)


# Data provider events


def log_data_event(message: str, **fields: Any) -> None:
    _emit("data_event", logging.INFO, message, **fields)


# Database events


def log_db_event(message: str, **fields: Any) -> None:
    _emit("db_event", logging.INFO, message, **fields)
