from __future__ import annotations

import logging
import logging.config
import os
import re
from typing import Any

from src.utils.logging_context import get_context


class SensitiveDataFilter(logging.Filter):
    """
    Redacts sensitive tokens (API keys, secrets) from log messages.

    Applies best-effort redaction on both plain strings and simple mapping args.
    """

    _REDACT = "***"

    # Matches key=value or key: value pairs (case-insensitive) for common sensitive keys
    _KV_PATTERN = re.compile(
        r"(?i)\b(api[_-]?key|api[_-]?secret|secret|token|password|pass|auth|bearer|session)[^=:\s]*\s*([=:])\s*([^\s,;]+)"
    )
    # Matches JSON-style "key": "value" for the same keys
    _JSON_PATTERN = re.compile(
        r"(?i)\"(api[_-]?key|api[_-]?secret|secret|token|password|pass|auth|bearer|session)\"\s*:\s*\"([^\"]+)\""
    )

    _SENSITIVE_KEYS = {
        "api_key",
        "apikey",
        "api-secret",
        "api_secret",
        "secret",
        "token",
        "password",
        "pass",
        "auth",
        "bearer",
        "session",
    }

    @classmethod
    def _redact_text(cls, text: str) -> str:
        text = cls._KV_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)} {cls._REDACT}", text)
        text = cls._JSON_PATTERN.sub(lambda m: f'"{m.group(1)}": "{cls._REDACT}"', text)
        return text

    @classmethod
    def _redact_mapping(cls, mapping: dict[str, Any]) -> dict[str, Any]:
        redacted: dict[str, Any] = {}
        for k, v in mapping.items():
            if isinstance(k, str) and k.lower() in cls._SENSITIVE_KEYS:
                redacted[k] = cls._REDACT
            elif isinstance(v, str):
                redacted[k] = cls._redact_text(v)
            else:
                redacted[k] = v
        return redacted

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            if isinstance(record.msg, str):
                record.msg = self._redact_text(record.msg)
            # Best-effort redaction of mapping/tuple args
            if isinstance(record.args, dict):
                record.args = self._redact_mapping(record.args)
            elif isinstance(record.args, (list, tuple)):
                sanitized = []
                for a in record.args:  # type: ignore[assignment]
                    if isinstance(a, str):
                        sanitized.append(self._redact_text(a))
                    else:
                        sanitized.append(a)
                record.args = tuple(sanitized)
        except Exception:
            # Never fail logging because of filter errors
            return True
        return True


class NamespacePrefixFilter(logging.Filter):
    """Prefixes logger names with 'atb.' for consistent namespacing."""

    _PREFIX = "atb."

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            if record.name and not record.name.startswith(self._PREFIX):
                record.name = f"{self._PREFIX}{record.name}"
        except Exception:
            return True
        return True


class ContextInjectorFilter(logging.Filter):
    """Injects structured context fields from contextvars into LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            ctx = get_context()
            for key, value in ctx.items():
                # Attach as record attributes so formatters (including json) include these via %(key)s if configured
                if not hasattr(record, key):
                    setattr(record, key, value)
        except Exception:
            return True
        return True


def build_logging_config(level_name: str | None = None, json: bool = False) -> dict[str, Any]:
    level = (level_name or os.getenv("LOG_LEVEL") or "INFO").upper()
    if json:
        formatter = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            # Include standard fields; context fields will be injected by ContextInjectorFilter
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    else:
        formatter = {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "redact": {"()": "src.utils.logging_config.SensitiveDataFilter"},
            "ns": {"()": "src.utils.logging_config.NamespacePrefixFilter"},
            "ctx": {"()": "src.utils.logging_config.ContextInjectorFilter"},
        },
        "formatters": {
            "default": formatter,
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
                "filters": ["redact", "ns", "ctx"],
            }
        },
        "loggers": {
            # Reduce noise from chatty libraries while allowing overrides via env
            "sqlalchemy.engine": {"level": os.getenv("LOG_SQLALCHEMY_LEVEL", "WARNING"), "propagate": True},
            "urllib3": {"level": os.getenv("LOG_URLLIB3_LEVEL", "WARNING"), "propagate": True},
            "binance": {"level": os.getenv("LOG_BINANCE_LEVEL", "WARNING"), "propagate": True},
            "ccxt": {"level": os.getenv("LOG_CCXT_LEVEL", "WARNING"), "propagate": True},
        },
        "root": {"handlers": ["console"], "level": level},
    }


def configure_logging(level_name: str | None = None, use_json: bool | None = None) -> None:
    use_json = bool(int(os.getenv("LOG_JSON", "0"))) if use_json is None else use_json
    config = build_logging_config(level_name, json=use_json)
    logging.config.dictConfig(config)
