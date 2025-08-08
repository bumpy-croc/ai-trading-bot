import os
import json
from contextlib import contextmanager
import pytest

from src.config.feature_flags import is_enabled, get_flag, resolve_all

pytestmark = pytest.mark.unit


@contextmanager
def patched_env(env: dict):
    original = os.environ.copy()
    try:
        os.environ.update(env)
        yield
    finally:
        to_delete = [k for k in os.environ.keys() if k not in original]
        for k in to_delete:
            del os.environ[k]
        for k, v in original.items():
            os.environ[k] = v


def test_precedence_emergency_overrides_json_and_repo_defaults():
    with patched_env({
        "FEATURE_FLAGS_OVERRIDES": json.dumps({"use_prediction_engine": False}),
        "FEATURE_USE_PREDICTION_ENGINE": "true",
    }):
        assert is_enabled("use_prediction_engine", default=False) is True


def test_overrides_dominate_repo_defaults():
    with patched_env({"FEATURE_FLAGS_OVERRIDES": json.dumps({"use_prediction_engine": True})}):
        assert is_enabled("use_prediction_engine", default=False) is True


def test_unset_flag_returns_default():
    with patched_env({"FEATURE_FLAGS_OVERRIDES": json.dumps({})}):
        assert is_enabled("nonexistent_flag", default=True) is True
        assert get_flag("nonexistent_flag", default="alpha") == "alpha"


def test_string_flags_supported_from_env():
    with patched_env({"FEATURE_EXPERIMENT_BUCKET": "beta"}):
        assert get_flag("experiment_bucket") == "beta"


def test_resolve_all_contains_expected_keys():
    with patched_env({"FEATURE_FLAGS_OVERRIDES": json.dumps({"use_prediction_engine": True})}):
        all_flags = resolve_all()
        assert isinstance(all_flags, dict)
        assert "use_prediction_engine" in all_flags