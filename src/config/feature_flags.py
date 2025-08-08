"""
Feature Flags Helper

Provides a small API to resolve feature flags with clear precedence:
1) FEATURE_<UPPER_SNAKE_KEY> environment variable (emergency override)
2) FEATURE_FLAGS_OVERRIDES environment variable (JSON string with per-env diffs)
3) feature_flags.json in project root (git-tracked defaults)
4) Code-provided default when calling helper

Flags are boolean or string values. Constants must not be handled here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.config.paths import get_project_root


_DEFAULT_FLAGS_FILENAME = "feature_flags.json"


def _load_repo_defaults() -> Dict[str, Any]:
    """Load defaults from the git-tracked feature_flags.json file.

    Returns:
        Dict[str, Any]: Default flags or empty dict if file missing/malformed.
    """
    try:
        flags_path: Path = get_project_root() / _DEFAULT_FLAGS_FILENAME
        if not flags_path.exists():
            return {}
        with flags_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except Exception:
    flags_path: Path = get_project_root() / _DEFAULT_FLAGS_FILENAME
    if not flags_path.exists():
        return {}
    try:
        with flags_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
            return {}
    except (json.JSONDecodeError, FileNotFoundError, PermissionError):
        # Fail-soft: if file is missing, permission denied, or JSON is malformed, ignore and return empty map
        return {}


def _load_env_json(var_name: str) -> Dict[str, Any]:
    """Parse a JSON string from an environment variable into a dict.

    Args:
        var_name: The environment variable name to parse.

    Returns:
        Dict[str, Any]: Parsed dict or empty dict on missing/malformed.
    """
    raw = os.environ.get(var_name)
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _to_upper_snake(key: str) -> str:
    return key.upper().replace("-", "_")


def _parse_bool(value: str) -> Optional[bool]:
    """Parse a string into a boolean if it matches common representations.

    Returns True/False for recognized values, otherwise None.
    """
    v = value.strip().lower()
    if v in {"true", "1", "yes", "on", "enabled"}:
        return True
    if v in {"false", "0", "no", "off", "disabled"}:
        return False
    return None


def _resolve_from_sources(key: str) -> Optional[Union[bool, str]]:
    """Resolve a flag value across sources without applying a default."""
    # 1) Emergency per-flag env var: FEATURE_<UPPER_SNAKE_KEY>
    env_key = f"FEATURE_{_to_upper_snake(key)}"
    if env_key in os.environ:
        raw = os.environ.get(env_key, "")
        parsed_bool = _parse_bool(raw)
        return parsed_bool if parsed_bool is not None else raw

    # 2) FEATURE_FLAGS_OVERRIDES (JSON)
    overrides = _load_env_json("FEATURE_FLAGS_OVERRIDES")
    if key in overrides:
        value = overrides[key]
        if isinstance(value, bool) or isinstance(value, str):
            return value

    # 3) feature_flags.json (repo defaults)
    defaults = _load_repo_defaults()
    if key in defaults:
        value = defaults[key]
        if isinstance(value, bool) or isinstance(value, str):
            return value

    # Not found
    return None


def get_flag(key: str, default: Optional[Union[bool, str]] = None) -> Optional[Union[bool, str]]:
    """Get a feature flag as bool or string.

    Args:
        key: Flag key in lower_snake_case (e.g., 'use_prediction_engine').
        default: Optional default value if not set in any source.

    Returns:
        The resolved bool or string value, or the provided default, or None.
    """
    value = _resolve_from_sources(key)
    if value is None:
        return default
    return value


def is_enabled(key: str, default: bool = False) -> bool:
    """Convenience bool getter with robust string parsing.

    Args:
        key: Flag key in lower_snake_case.
        default: Fallback when flag is not defined.

    Returns:
        bool: Resolved boolean value.
    """
    value = _resolve_from_sources(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        parsed = _parse_bool(value)
        if parsed is not None:
            return parsed
    return default


def resolve_all() -> Dict[str, Union[bool, str]]:
    """Resolve and return a merged view of all flags for diagnostics.

    Precedence is applied per key across the three sources.
    """
    merged: Dict[str, Union[bool, str]] = {}

    # Start with repo defaults
    merged.update({k: v for k, v in _load_repo_defaults().items() if isinstance(v, (bool, str))})

    # Apply overrides
    overrides = _load_env_json("FEATURE_FLAGS_OVERRIDES")
    for k, v in overrides.items():
        if isinstance(v, (bool, str)):
            merged[k] = v

    # Apply per-flag env vars
    # To capture keys from env, iterate over existing keys and any FEATURE_* present
    for env_k, env_v in os.environ.items():
        if env_k.startswith("FEATURE_") and env_k not in {"FEATURE_FLAGS_OVERRIDES"}:
            key_lower = env_k[len("FEATURE_"):].lower()
            key_lower = key_lower.replace("_", " ").replace("-", " ").strip().replace(" ", "_")
            parsed_bool = _parse_bool(env_v)
            merged[key_lower] = parsed_bool if parsed_bool is not None else env_v

    return merged


