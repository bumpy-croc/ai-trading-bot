"""Live-engine escalation of persistent inference timeouts (#927).

The prediction engine tracks consecutive live inference timeouts; the live
engine surfaces the breach as a ``system_events`` row plus operator alert
(honest ``alert_sent`` semantics via ``_record_event``), once per episode.
This is an observability escalation only — it never halts trading.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from src.database.models import EventType
from src.engines.live.trading_engine import LiveTradingEngine
from src.prediction.config import PredictionConfig
from src.prediction.engine import PredictionEngine

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _prediction_engine(
    *, consecutive: int = 0, total: int = 0, threshold: int = 10
) -> PredictionEngine:
    """Real PredictionEngine (mocked internals) with seeded timeout counters."""
    config = PredictionConfig()
    config.inference_timeout_escalation_threshold = threshold
    with (
        patch("src.prediction.engine.PredictionModelRegistry"),
        patch("src.prediction.engine.FeaturePipeline"),
    ):
        engine = PredictionEngine(config)
    engine._consecutive_timeout_count = consecutive
    engine._inference_timeout_count = total
    return engine


def _make_engine(strategy=None) -> LiveTradingEngine:
    """Bare engine with only the collaborators the escalation check touches."""
    engine = LiveTradingEngine.__new__(LiveTradingEngine)
    engine.db_manager = MagicMock()
    engine.trading_session_id = 42
    engine._send_alert = MagicMock(return_value=True)
    engine._inference_escalation_active = False
    engine.strategy = strategy
    return engine


def _component_strategy(prediction_engine) -> SimpleNamespace:
    """ComponentStrategy shape: strategy.signal_generator.prediction_engine."""
    return SimpleNamespace(signal_generator=SimpleNamespace(prediction_engine=prediction_engine))


def _runtime_strategy(prediction_engine) -> SimpleNamespace:
    """StrategyRuntime shape: runtime.strategy.signal_generator.prediction_engine."""
    return SimpleNamespace(strategy=_component_strategy(prediction_engine))


class TestIterPredictionEngines:
    def test_finds_engine_on_component_strategy(self):
        pe = _prediction_engine()
        engine = _make_engine(_component_strategy(pe))
        assert list(engine._iter_prediction_engines()) == [pe]

    def test_finds_engine_through_strategy_runtime(self):
        pe = _prediction_engine()
        engine = _make_engine(_runtime_strategy(pe))
        assert list(engine._iter_prediction_engines()) == [pe]

    def test_no_prediction_engine_yields_nothing(self):
        engine = _make_engine(SimpleNamespace(signal_generator=SimpleNamespace()))
        assert list(engine._iter_prediction_engines()) == []

    def test_none_strategy_yields_nothing(self):
        engine = _make_engine(None)
        assert list(engine._iter_prediction_engines()) == []

    def test_non_prediction_engine_objects_are_ignored(self):
        """A mock (or any unrelated object) on the attribute path must not
        fabricate an escalation — getattr on a MagicMock returns truthy
        MagicMocks for every name, so discovery is isinstance-gated."""
        engine = _make_engine(MagicMock())
        assert list(engine._iter_prediction_engines()) == []


class TestCheckInferenceHealth:
    def test_escalation_emits_critical_paged_event_once(self):
        pe = _prediction_engine(consecutive=10, total=10, threshold=10)
        engine = _make_engine(_component_strategy(pe))

        engine._check_inference_health()
        engine._check_inference_health()  # same episode: no second page

        engine.db_manager.log_event.assert_called_once()
        _, kwargs = engine.db_manager.log_event.call_args
        assert kwargs["event_type"] == EventType.ALERT
        assert kwargs["severity"] == "critical"
        assert kwargs["component"] == "prediction"
        assert kwargs["error_code"] == "INFERENCE_TIMEOUT_ESCALATION"
        # Honest alert_sent semantics: the webhook was delivered.
        assert kwargs["alert_sent"] is True
        engine._send_alert.assert_called_once()

    def test_alert_failure_recorded_honestly(self):
        pe = _prediction_engine(consecutive=10, total=10, threshold=10)
        engine = _make_engine(_component_strategy(pe))
        engine._send_alert = MagicMock(return_value=False)  # webhook down

        engine._check_inference_health()

        _, kwargs = engine.db_manager.log_event.call_args
        assert kwargs["alert_sent"] is False
        assert kwargs["alert_method"] is None

    def test_recovery_logs_and_rearms(self):
        pe = _prediction_engine(consecutive=10, total=10, threshold=10)
        engine = _make_engine(_component_strategy(pe))

        engine._check_inference_health()
        assert engine._inference_escalation_active is True

        # Model recovers: a successful inference reset the streak.
        pe._consecutive_timeout_count = 0
        engine._check_inference_health()
        assert engine._inference_escalation_active is False

        events = [call.kwargs for call in engine.db_manager.log_event.call_args_list]
        assert len(events) == 2
        assert events[1]["error_code"] == "INFERENCE_TIMEOUT_RECOVERED"
        assert events[1]["severity"] == "warning"

        # A fresh episode pages again.
        pe._consecutive_timeout_count = 10
        engine._check_inference_health()
        codes = [call.kwargs["error_code"] for call in engine.db_manager.log_event.call_args_list]
        assert codes.count("INFERENCE_TIMEOUT_ESCALATION") == 2

    def test_no_events_below_threshold(self):
        pe = _prediction_engine(consecutive=3, total=7, threshold=10)
        engine = _make_engine(_component_strategy(pe))

        engine._check_inference_health()

        engine.db_manager.log_event.assert_not_called()
        assert engine._inference_escalation_active is False

    def test_zero_threshold_never_escalates(self):
        pe = _prediction_engine(consecutive=10_000, total=10_000, threshold=0)
        engine = _make_engine(_component_strategy(pe))

        engine._check_inference_health()

        engine.db_manager.log_event.assert_not_called()

    def test_faults_are_isolated(self):
        """A broken strategy tree must never crash the trading loop."""

        class ExplodingStrategy:
            @property
            def signal_generator(self):
                raise RuntimeError("boom")

        engine = _make_engine(ExplodingStrategy())
        engine._check_inference_health()  # must not raise
        engine.db_manager.log_event.assert_not_called()


class TestInferenceTimeoutTotals:
    def test_totals_sum_across_prediction_engines(self):
        pe = _prediction_engine(consecutive=2, total=5)
        engine = _make_engine(_component_strategy(pe))
        assert engine.inference_timeout_totals() == (5, 2)

    def test_totals_default_to_zero_without_prediction_engines(self):
        engine = _make_engine(None)
        assert engine.inference_timeout_totals() == (0, 0)

    def test_totals_fault_isolated(self):
        class ExplodingStrategy:
            @property
            def signal_generator(self):
                raise RuntimeError("boom")

        engine = _make_engine(ExplodingStrategy())
        assert engine.inference_timeout_totals() == (0, 0)


class TestLoopWiring:
    def test_trading_loop_runs_inference_health_check_each_iteration(self):
        from tests.unit.engines.live.test_system_halt_gate import (
            _make_engine as _make_full_engine,
        )
        from tests.unit.engines.live.test_system_halt_gate import (
            _run_one_loop_iteration,
        )

        engine = _make_full_engine()
        engine._check_inference_health = MagicMock()

        _run_one_loop_iteration(engine, entry_probe=None, exit_probe=None)

        engine._check_inference_health.assert_called_once()


class TestStatusLineIncludesTimeouts:
    def test_log_status_includes_inference_timeout_counter(self, caplog):
        import logging

        from src.engines.live.monitoring.account_monitor import LiveAccountMonitor

        state = MagicMock()
        state.current_balance = 1000.0
        state.live_position_tracker.positions = {}
        state.live_position_tracker.position_count = 0
        metrics = MagicMock()
        metrics.win_rate = 0.5
        metrics.total_trades = 4
        state.performance_tracker.get_metrics.return_value = metrics
        state.inference_timeout_totals.return_value = (3, 2)

        monitor = LiveAccountMonitor(state)
        with caplog.at_level(logging.INFO, logger="src.engines.live.monitoring.account_monitor"):
            monitor.log_status("BTCUSDT", 50_000.0)

        status_lines = [r.message for r in caplog.records if "Status" in r.message]
        assert status_lines, "expected a status line"
        assert "Inference timeouts: 3" in status_lines[0]
        assert "2 consecutive" in status_lines[0]
