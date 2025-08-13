from __future__ import annotations
import logging
import logging.config
import os
from typing import Any, Dict


def build_logging_config(level_name: str | None = None, json: bool = False) -> Dict[str, Any]:
    level = (level_name or os.getenv("LOG_LEVEL", "INFO")).upper()
    if json:
        formatter = {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    else:
        formatter = {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        }
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": formatter,
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": level,
            }
        },
        "root": {"handlers": ["console"], "level": level},
    }


def configure_logging(level_name: str | None = None, use_json: bool | None = None) -> None:
    use_json = bool(int(os.getenv("LOG_JSON", "0"))) if use_json is None else use_json
    config = build_logging_config(level_name, json=use_json)
    logging.config.dictConfig(config)
