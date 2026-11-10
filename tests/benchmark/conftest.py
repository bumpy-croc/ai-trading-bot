"""Benchmarks train real models and assert on wall-clock behaviour.

They are skipped in a general run (``pytest tests``, CI's unit job, ``atb test all``) because
a full training run takes minutes and its outcome depends on machine load, not on the code
under test. Ask for them explicitly with any of:

- a path under ``tests/benchmark`` (``pytest tests/benchmark/test_model_architectures.py``)
- a marker expression naming ``benchmark`` (``pytest -m benchmark``)
- ``ATB_RUN_BENCHMARKS=1``
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

_BENCHMARK_DIR = Path(__file__).resolve().parent
_NEGATED_MARKER = re.compile(r"\bnot\s+benchmark\b")
_MARKER = re.compile(r"\bbenchmark\b")


def _benchmarks_requested(config: pytest.Config) -> bool:
    if os.environ.get("ATB_RUN_BENCHMARKS") == "1":
        return True

    markexpr = config.getoption("markexpr", default="") or ""
    if _MARKER.search(_NEGATED_MARKER.sub("", markexpr)):
        return True

    for arg in config.args:
        path = Path(str(arg).split("::", 1)[0]).resolve()
        if path == _BENCHMARK_DIR or _BENCHMARK_DIR in path.parents:
            return True
    return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _benchmarks_requested(config):
        return
    skip = pytest.mark.skip(
        reason="benchmark: run with `pytest tests/benchmark`, `-m benchmark` or ATB_RUN_BENCHMARKS=1"
    )
    for item in items:
        if _BENCHMARK_DIR in Path(str(item.fspath)).resolve().parents:
            item.add_marker(skip)
