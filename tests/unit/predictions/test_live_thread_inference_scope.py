"""Engine-spawned threads must infer under the LIVE deadline (fail-closed invariant).

Context variables do not propagate to new threads, so a bare ``threading.Thread``
in a live process infers with no deadline. ``create_live_thread`` scopes the
thread LIVE, a guard flags unscoped inference in a live process, and a static
check keeps bare threads out of the live engine package.
"""

import logging
import re
import threading
from pathlib import Path

import pytest

from src.infrastructure.live_threads import create_live_thread
from src.prediction.inference_context import (
    InferenceContext,
    get_inference_context,
    inference_scope,
    is_unscoped_in_live_process,
    register_live_process,
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
        thread = create_live_thread(lambda: seen.append(get_inference_context()), name="t-live")

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

        thread = create_live_thread(target, name="named", args=(1,), kwargs={"b": 2})

        assert thread.name == "named"
        assert thread.daemon is True
        _run_in_thread(thread)
        assert got == {"a": 1, "b": 2}

    def test_target_exception_propagates_from_the_scoped_thread(self):
        def boom():
            raise RuntimeError("x")

        errors: list[BaseException] = []
        original_hook = threading.excepthook
        threading.excepthook = lambda args: errors.append(args.exc_value)
        try:
            _run_in_thread(create_live_thread(boom))
        finally:
            threading.excepthook = original_hook

        assert len(errors) == 1


class TestUnscopedGuard:
    def test_silent_when_no_live_process(self, caplog):
        with caplog.at_level(logging.ERROR):
            assert is_unscoped_in_live_process() is False

        assert caplog.records == []

    def test_flags_unscoped_thread_and_logs_once_per_thread(self, caplog):
        register_live_process()
        results: list[bool] = []

        def infer():
            results.append(is_unscoped_in_live_process())
            results.append(is_unscoped_in_live_process())

        with caplog.at_level(logging.ERROR):
            _run_in_thread(threading.Thread(target=infer, name="rogue-1"))
            _run_in_thread(threading.Thread(target=infer, name="rogue-2"))

        assert results == [True] * 4
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 2  # one per thread, even if the OS reuses a thread ident
        assert "rogue-1" in errors[0] and "rogue-2" in errors[1]

    @pytest.mark.parametrize("context", list(InferenceContext))
    def test_explicit_scope_is_not_flagged(self, caplog, context):
        """A nested backtest's deliberate DETERMINISTIC scope is legitimate."""
        register_live_process()

        with caplog.at_level(logging.ERROR), inference_scope(context):
            assert is_unscoped_in_live_process() is False

        assert caplog.records == []

    def _engine(self):
        from unittest.mock import patch

        from src.prediction.config import PredictionConfig
        from src.prediction.engine import PredictionEngine

        with (
            patch("src.prediction.engine.PredictionModelRegistry"),
            patch("src.prediction.engine.FeaturePipeline"),
        ):
            return PredictionEngine(PredictionConfig())

    def test_engine_applies_live_deadline_to_unscoped_thread_in_live_process(self, caplog):
        engine = self._engine()
        register_live_process()

        with caplog.at_level(logging.ERROR):
            assert engine._get_timeout_seconds() == engine.config.live_inference_timeout

        assert any("no inference scope" in r.getMessage() for r in caplog.records)

    def test_engine_keeps_nested_deterministic_scope_deadline_free(self):
        engine = self._engine()
        register_live_process()

        with inference_scope(InferenceContext.DETERMINISTIC):
            assert engine._get_timeout_seconds() is None

    def test_engine_has_no_deadline_outside_live_process(self):
        assert self._engine()._get_timeout_seconds() is None


_THREAD_PATTERNS = [
    (r"\bthreading\.(Thread|Timer)\s*\(", "threading.Thread/Timer"),
    (r"from threading import[^\n]*\b(Thread|Timer)\b", "from threading import Thread/Timer"),
    (r"\bThreadPoolExecutor\b", "ThreadPoolExecutor"),
    (r"\basyncio\.to_thread\b|\brun_in_executor\b", "asyncio thread offload"),
    (r"class\s+\w+\(\s*(threading\.)?Thread\s*\)", "Thread subclass"),
]


class TestNoBareThreadsInLiveEngine:
    def test_live_engine_package_creates_threads_via_create_live_thread(self):
        assert LIVE_ENGINE_DIR.is_dir()
        offenders = []
        for path in LIVE_ENGINE_DIR.rglob("*.py"):
            text = path.read_text()
            for pattern, label in _THREAD_PATTERNS:
                for match in re.finditer(pattern, text):
                    line = text[: match.start()].count("\n") + 1
                    if label == "Thread subclass" and "inference_scope(" in text:
                        continue  # subclass that scopes its own run()
                    offenders.append(f"{path.relative_to(LIVE_ENGINE_DIR)}:{line} ({label})")

        assert offenders == [], (
            "Engine threads must be created via create_live_thread so they infer under "
            f"the LIVE deadline: {offenders}"
        )

    def test_user_data_processor_runs_under_live_scope(self):
        from unittest.mock import MagicMock

        from src.engines.live.user_data_processor import UserDataProcessor

        seen: list[InferenceContext] = []
        processor = UserDataProcessor(order_tracker=MagicMock())
        processor._handle_event = lambda event: seen.append(get_inference_context())
        processor.enqueue({"e": "x"})
        processor.start()
        for _ in range(50):
            if seen:
                break
            threading.Event().wait(0.05)
        processor.stop()
        processor.join(timeout=5)

        assert seen == [InferenceContext.LIVE]
