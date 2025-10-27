from __future__ import annotations

import json
import logging
import logging.config
import re
import time
from typing import Any

from src.config import get_config
from src.infrastructure.logging.context import get_context


class SensitiveDataFilter(logging.Filter):
    """
    Redacts sensitive tokens (API keys, secrets) from log messages.

    Applies best-effort redaction on both plain strings and simple mapping args.
    """

    _REDACT = "***"

    # Matches key=value or key: value pairs (case-insensitive) for common sensitive keys
    _KV_PATTERN = re.compile(
        r"(?i)\b(api[_-]?key|api[_-]?secret|secret|token|password|pass|auth|bearer|session"
        r"|refresh[_-]?token|access[_-]?token|hmac[_-]?secret|private[_-]?key|client[_-]?secret"
        r"|signing[_-]?key|credential(?:s)?)[^=:\s]*\s*([=:])\s*([^\s,;]+)"
    )
    # Matches JSON-style "key": "value" for the same keys
    _JSON_PATTERN = re.compile(
        r"(?i)\"(api[_-]?key|api[_-]?secret|secret|token|password|pass|auth|bearer|session"
        r"|refresh[_-]?token|access[_-]?token|hmac[_-]?secret|private[_-]?key|client[_-]?secret"
        r"|signing[_-]?key|credential(?:s)?)\"\s*:\s*\"([^\"]+)\""
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
        "refresh_token",
        "refresh-token",
        "access_token",
        "access-token",
        "hmac_secret",
        "hmac-secret",
        "private_key",
        "private-key",
        "client_secret",
        "client-secret",
        "signing_key",
        "signing-key",
        "credential",
        "credentials",
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

    # * Built-in LogRecord fields that should not be overwritten
    _RESERVED_FIELDS = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "getMessage",
        "exc_info",
        "exc_text",
        "stack_info",
        "message",
    }

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            ctx = get_context()
            for key, value in ctx.items():
                # * Only add context fields that don't conflict with built-in LogRecord fields
                if key not in self._RESERVED_FIELDS and not hasattr(record, key):
                    setattr(record, key, value)
        except Exception:
            return True
        return True


class SamplingFilter(logging.Filter):
    """Probabilistically sample DEBUG/INFO logs to reduce noise.

    Controlled via config:
    - LOG_SAMPLING_RATE_DEBUG (default 1.0)
    - LOG_SAMPLING_RATE_INFO (default 1.0)
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            cfg = get_config()
            self.rate_debug = float(cfg.get("LOG_SAMPLING_RATE_DEBUG", "1.0"))
            self.rate_info = float(cfg.get("LOG_SAMPLING_RATE_INFO", "1.0"))
        except Exception:
            self.rate_debug = 1.0
            self.rate_info = 1.0

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            if record.levelno <= logging.DEBUG:
                return (hash((record.name, record.lineno, int(time.time() / 10))) % 1000) < int(
                    1000 * self.rate_debug
                )
            if record.levelno == logging.INFO:
                return (hash((record.name, record.lineno, int(time.time() / 10))) % 1000) < int(
                    1000 * self.rate_info
                )
            return True
        except Exception:
            return True


class MaxMessageLengthFilter(logging.Filter):
    """Truncates overly long messages to a max length with an indicator.

    Controlled via config LOG_MAX_MESSAGE_LEN (default 5000 characters).
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            cfg = get_config()
            self.max_len = int(cfg.get("LOG_MAX_MESSAGE_LEN", "5000"))
        except Exception:
            self.max_len = 5000

    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            if isinstance(record.msg, str) and len(record.msg) > self.max_len:
                record.msg = record.msg[: self.max_len] + "... [truncated]"
        except Exception:
            return True
        return True


class SimpleJsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        # * Safely get the message without overwriting the message field
        try:
            message = record.getMessage()
        except Exception:
            message = str(record.msg) if record.msg else ""

        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": message,
        }

        # * Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # * Add any extra fields from the record (excluding built-in fields)
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry)


def build_logging_config(level_name: str | None = None, json: bool = False) -> dict[str, Any]:
    cfg = get_config()
    level = (level_name or cfg.get("LOG_LEVEL", "INFO")).upper()

    # * Use custom JSON formatter for structured logging
    if json:
        formatter = {
            "()": "src.infrastructure.logging.config.SimpleJsonFormatter",
        }
    else:
        formatter = {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "redact": {"()": "src.infrastructure.logging.config.SensitiveDataFilter"},
            "ns": {"()": "src.infrastructure.logging.config.NamespacePrefixFilter"},
            "ctx": {"()": "src.infrastructure.logging.config.ContextInjectorFilter"},
            "sample": {"()": "src.infrastructure.logging.config.SamplingFilter"},
            "truncate": {"()": "src.infrastructure.logging.config.MaxMessageLengthFilter"},
        },
        "formatters": {
            "default": formatter,
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
                "filters": ["redact", "ns", "ctx", "sample", "truncate"],
            }
        },
        "loggers": {
            # Reduce noise from chatty libraries while allowing overrides via config
            "sqlalchemy.engine": {
                "level": cfg.get("LOG_SQLALCHEMY_LEVEL", "WARNING"),
                "propagate": True,
            },
            "urllib3": {"level": cfg.get("LOG_URLLIB3_LEVEL", "WARNING"), "propagate": True},
            "binance": {"level": cfg.get("LOG_BINANCE_LEVEL", "WARNING"), "propagate": True},
            "ccxt": {"level": cfg.get("LOG_CCXT_LEVEL", "WARNING"), "propagate": True},
        },
        "root": {"handlers": ["console"], "level": level},
    }


def configure_logging(level_name: str | None = None, use_json: bool | None = None) -> None:
    cfg = get_config()
    # Determine JSON default: if not explicitly provided, prefer JSON in production-like envs
    if use_json is None:
        # Explicit config override takes precedence if set
        if cfg.get("LOG_JSON") is not None:
            try:
                use_json = bool(int(cfg.get("LOG_JSON", "0")))
            except Exception:
                use_json = False
        else:
            # Heuristics: Railway or ENV/APP_ENV=production -> JSON by default
            is_railway = any(
                cfg.get(k) is not None
                for k in ("RAILWAY_DEPLOYMENT_ID", "RAILWAY_PROJECT_ID", "RAILWAY_SERVICE_ID")
            )
            env_name = (
                cfg.get("ENV") or cfg.get("APP_ENV") or cfg.get("RAILWAY_ENVIRONMENT_NAME") or ""
            ).lower()
            is_production = env_name == "production"
            use_json = is_railway or is_production
    config = build_logging_config(level_name, json=use_json)
    logging.config.dictConfig(config)
