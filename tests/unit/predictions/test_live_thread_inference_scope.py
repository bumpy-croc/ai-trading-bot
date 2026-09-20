"""Engine-spawned threads must infer under the LIVE deadline (fail-closed invariant).

Context variables do not propagate to new threads, so a bare ``threading.Thread``
in a live process infers with no deadline. ``spawn_live_thread`` scopes the
thread LIVE, a guard flags unscoped inference in a live process, and a static
check keeps bare threads out of the live engine package.
"""

import logging
import re
import threading
from pathlib import Path

import pytest

from src.prediction.inference_context import (
    InferenceContext,
    get_inference_context,
    inference_scope,
    register_live_process,
    spawn_live_thread,
    warn_if_unscoped_in_live_process,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

LIVE_ENGINE_DIR = Path(__file__).resolve().parents[3] / "src" / "engines" / "live"


def _run_in_thread(thread: threading.Thread) -> None:
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()


class TestSpawnLiveThread:
    def test_target_runs_under_live_scope(self):
        seen: list[InferenceContext] = []
        thread = spawn_live_thread(lambda: seen.append(get_inference_context()), name="t-live")

        _run_in_thread(thread)

        assert seen == [InferenceContext.LIVE]

    def test_bare_thread_does_not_inherit_live(self):
        """Documents the fail-open shape the helper exists to prevent."""
        seen: list[InferenceContext] = []
        with inference_scope(InferenceContext.LIVE):
            thread = threading.Thread(target=lambda: seen.append(get_inference_context()))
            _run_in_thread(thread)

        assert seen == [InferenceContext.DETERMINISTIC]

    def test_args_kwargs_name_and_daemon_are_honoured(self):
        got: dict = {}

        def target(a, b=None):
            got["a"], got["b"] = a, b

        thread = spawn_live_thread(target, name="named", args=(1,), kwargs={"b": 2})

        assert thread.name == "named"
        assert thread.daemon is True
        _run_in_thread(thread)
        assert got == {"a": 1, "b": 2}

    def test_scope_is_restored_after_target_raises(self):
        def boom():
            raise RuntimeError("x")

        errors: list[BaseException] = []
        original_hook = threading.excepthook
        threading.excepthook = lambda args: errors.append(args.exc_value)
        try:
            _run_in_thread(spawn_live_thread(boom))
        finally:
            threading.excepthook = original_hook

        assert len(errors) == 1
        assert get_inference_context() is InferenceContext.DETERMINISTIC


class TestUnscopedGuard:
    def test_silent_when_no_live_process(self, caplog):
        with caplog.at_level(logging.ERROR):
            warn_if_unscoped_in_live_process()

        assert caplog.records == []

    def test_flags_unscoped_thread_once_in_live_process(self, caplog):
        register_live_process()

        def infer():
            warn_if_unscoped_in_live_process()
            warn_if_unscoped_in_live_process()

        with caplog.at_level(logging.ERROR):
            _run_in_thread(threading.Thread(target=infer, name="rogue"))

        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "rogue" in errors[0].getMessage()

    @pytest.mark.parametrize("context", list(InferenceContext))
    def test_explicit_scope_is_not_flagged(self, caplog, context):
        """A nested backtest's deliberate DETERMINISTIC scope is legitimate."""
        register_live_process()

        with caplog.at_level(logging.ERROR), inference_scope(context):
            warn_if_unscoped_in_live_process()

        assert caplog.records == []

    def test_engine_flags_unscoped_prediction_in_live_process(self, caplog):
        from unittest.mock import patch

        from src.prediction.config import PredictionConfig
        from src.prediction.engine import PredictionEngine

        with (
            patch("src.prediction.engine.PredictionModelRegistry"),
            patch("src.prediction.engine.FeaturePipeline"),
        ):
            engine = PredictionEngine(PredictionConfig())
        register_live_process()

        with caplog.at_level(logging.ERROR):
            assert engine._get_timeout_seconds() is None

        assert any("no inference scope" in r.getMessage() for r in caplog.records)


class TestNoBareThreadsInLiveEngine:
    def test_live_engine_package_uses_spawn_live_thread(self):
        offenders = []
        for path in LIVE_ENGINE_DIR.rglob("*.py"):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"\bthreading\.Thread\(", line) and not line.lstrip().startswith("#"):
                    offenders.append(f"{path.relative_to(LIVE_ENGINE_DIR)}:{lineno}")

        assert offenders == [], (
            "Engine threads must be created via spawn_live_thread so they infer under "
            f"the LIVE deadline: {offenders}"
        )
