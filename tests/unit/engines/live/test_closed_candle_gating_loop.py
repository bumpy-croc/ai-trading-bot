"""Trading-loop behavior of closed-candle gating (parity plan P1.0, decision D1).

Flag OFF (default, shipped state): the loop is byte-identical to today — the
strategy signal is evaluated every tick on the tail (forming) bar and entries
may follow every tick. The only addition is Signal.metadata stamping (required
in BOTH modes so the staging/prod A/B can attribute decisions).

Flag ON (``closed_candle_gating`` / FEATURE_CLOSED_CANDLE_GATING): signal
evaluation runs exactly once per newly closed bar, at that bar's index, while
every protective path (stop-loss/trailing/exit checks, PnL/MFE-MAE updates)
stays tick-driven on the forming bar — the PM condition on D1.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_providers.data_provider import DataProvider
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.components import Signal, SignalDirection
from src.strategies.components.strategy import TradingDecision


def _make_df(n: int, start: str = "2026-07-01 00:00:00") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq="1h", name="timestamp")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n)],
            "high": [110.0 + i for i in range(n)],
            "low": [90.0 + i for i in range(n)],
            "close": [105.0 + i for i in range(n)],
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def _make_decision() -> TradingDecision:
    return TradingDecision(
        timestamp=datetime.now(UTC),
        signal=Signal(
            direction=SignalDirection.HOLD,
            strength=0.5,
            confidence=0.5,
            metadata={},
        ),
        position_size=0.0,
        regime=None,
        risk_metrics={},
        execution_time_ms=0.0,
        metadata={},
    )


def _make_engine() -> LiveTradingEngine:
    """A paper-mode engine with the database manager patched out."""
    from src.strategies.ml_basic import create_ml_basic_strategy

    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=create_ml_basic_strategy(),
            data_provider=Mock(spec=DataProvider),
            initial_balance=10000,
            enable_live_trading=False,
        )


def _stub_loop_periphery(engine: LiveTradingEngine, frames: list[pd.DataFrame]) -> None:
    """Stub everything around the decision path so `_trading_loop` runs N steps.

    One frame is consumed per loop step; callers pass max_steps=len(frames).
    """
    engine.is_running = True
    engine._sleep_with_interrupt = Mock()
    engine._ensure_ws_health_monitor_alive = Mock()
    engine._get_latest_data = Mock(side_effect=list(frames))
    engine._is_data_fresh = Mock(return_value=True)
    engine.strategy_manager = None
    engine._prepare_strategy_dataframe = Mock(side_effect=lambda df: df)
    engine._is_context_ready = Mock(return_value=(True, ""))
    engine._runtime_process_decision = Mock(side_effect=lambda *a, **k: _make_decision())
    engine._check_exit_conditions = Mock()
    engine._check_entry_conditions = Mock()
    engine.entry_coordinator.process_legacy_short_entry = Mock()
    engine.live_exit_handler.update_trailing_stops = Mock()
    engine.live_exit_handler.check_partial_operations = Mock()
    engine.live_position_tracker.update_pnl = Mock()
    engine.live_position_tracker.update_mfe_mae = Mock()
    engine._update_performance_metrics = Mock()
    engine._check_max_drawdown = Mock()
    engine._log_periodic_account_state = Mock()
    engine._log_status = Mock()


def _run(engine: LiveTradingEngine, frames: list[pd.DataFrame]) -> None:
    _stub_loop_periphery(engine, frames)
    engine._trading_loop("BTCUSDT", "1h", max_steps=len(frames))


# --------------------------------------------------------------------------- #
# Flag wiring
# --------------------------------------------------------------------------- #


@pytest.mark.fast
class TestFlagWiring:
    def test_repo_default_is_off(self):
        """feature_flags.json ships the flag OFF — staging A/B decides the flip
        (plan §5 default-flip quarantine)."""
        flags = json.loads(
            (Path(__file__).resolve().parents[4] / "feature_flags.json").read_text()
        )
        assert flags["closed_candle_gating"] is False

    def test_engine_defaults_to_disabled(self, monkeypatch):
        monkeypatch.delenv("FEATURE_CLOSED_CANDLE_GATING", raising=False)
        engine = _make_engine()
        assert engine._closed_candle_gate.enabled is False

    def test_env_override_enables(self, monkeypatch):
        monkeypatch.setenv("FEATURE_CLOSED_CANDLE_GATING", "true")
        engine = _make_engine()
        assert engine._closed_candle_gate.enabled is True


# --------------------------------------------------------------------------- #
# Flag OFF — inertness proof (today's behavior, evaluated every tick on tail)
# --------------------------------------------------------------------------- #


@pytest.mark.fast
class TestFlagOffInertness:
    def test_signal_evaluated_every_tick_on_forming_bar(self, monkeypatch):
        monkeypatch.delenv("FEATURE_CLOSED_CANDLE_GATING", raising=False)
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df, df, df])

        assert engine._runtime_process_decision.call_count == 3
        for call in engine._runtime_process_decision.call_args_list:
            assert call.args[1] == 4  # tail (forming) bar index — unchanged

    def test_entries_checked_every_tick_on_forming_bar(self, monkeypatch):
        monkeypatch.delenv("FEATURE_CLOSED_CANDLE_GATING", raising=False)
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df, df, df])

        assert engine._check_entry_conditions.call_count == 3
        for call in engine._check_entry_conditions.call_args_list:
            assert call.args[1] == 4
        assert engine.entry_coordinator.process_legacy_short_entry.call_count == 3

    def test_metadata_stamped_forming_in_off_mode(self, monkeypatch):
        """Both modes stamp Signal.metadata so the A/B can attribute decisions."""
        monkeypatch.delenv("FEATURE_CLOSED_CANDLE_GATING", raising=False)
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df])

        decision = engine._check_entry_conditions.call_args.kwargs["runtime_decision"]
        md = decision.signal.metadata
        assert md["decision_bar_closed"] is False
        assert md["decision_bar_open_time"] == df.index[-1].isoformat()
        assert md["decision_bar_close_time"] == (df.index[-1] + pd.Timedelta(hours=1)).isoformat()


# --------------------------------------------------------------------------- #
# Flag ON — closed-bar cadence for decisions, tick-driven protection
# --------------------------------------------------------------------------- #


@pytest.mark.fast
class TestFlagOnGating:
    @pytest.fixture(autouse=True)
    def _enable(self, monkeypatch):
        monkeypatch.setenv("FEATURE_CLOSED_CANDLE_GATING", "true")

    def test_forming_ticks_do_not_retrigger_evaluation(self):
        """Three ticks of the same frame → exactly one evaluation, on the last
        closed bar (index[-2] without buffer evidence)."""
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df, df, df])

        assert engine._runtime_process_decision.call_count == 1
        assert engine._runtime_process_decision.call_args.args[1] == 3

    def test_decision_inputs_come_from_closed_bar(self):
        """The gated evaluation receives the closed bar's close and open time —
        the exact inputs backtest would use at that index."""
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df])

        args = engine._runtime_process_decision.call_args.args
        assert args[3] == pytest.approx(float(df["close"].iloc[-2]))
        assert args[4] == df.index[-2].to_pydatetime().replace(tzinfo=UTC)

    def test_new_closed_bar_triggers_exactly_one_evaluation(self):
        engine = _make_engine()
        df5 = _make_df(5)
        df6 = _make_df(6)

        _run(engine, [df5, df5, df6, df6])

        assert engine._runtime_process_decision.call_count == 2
        indices = [c.args[1] for c in engine._runtime_process_decision.call_args_list]
        assert indices == [3, 4]  # closed frontier of df5, then of df6

    def test_entries_only_on_evaluation_ticks_at_closed_index(self):
        engine = _make_engine()
        df5 = _make_df(5)
        df6 = _make_df(6)

        _run(engine, [df5, df5, df6])

        assert engine._check_entry_conditions.call_count == 2
        entry_indices = [c.args[1] for c in engine._check_entry_conditions.call_args_list]
        assert entry_indices == [3, 4]
        assert engine.entry_coordinator.process_legacy_short_entry.call_count == 2

    def test_protective_paths_stay_tick_driven_on_forming_bar(self):
        """Exit checks, trailing stops and PnL updates run every tick with the
        forming bar's index and live price — never gated (PM condition)."""
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df, df, df])

        assert engine._check_exit_conditions.call_count == 3
        for call in engine._check_exit_conditions.call_args_list:
            assert call.args[1] == 4  # forming-bar index
            assert call.args[2] == pytest.approx(float(df["close"].iloc[-1]))
        assert engine.live_exit_handler.update_trailing_stops.call_count == 3
        for call in engine.live_exit_handler.update_trailing_stops.call_args_list:
            assert call.args[1] == 4
        assert engine.live_position_tracker.update_pnl.call_count == 3

    def test_exit_checks_reuse_cached_closed_bar_decision_between_closes(self):
        """Between bar closes the tick-driven exit path keeps receiving the
        latest closed-bar decision (strategy-exit continuity)."""
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df, df, df])

        decisions = [
            c.kwargs["runtime_decision"] for c in engine._check_exit_conditions.call_args_list
        ]
        assert decisions[0] is not None
        assert decisions[1] is decisions[0]
        assert decisions[2] is decisions[0]

    def test_backfill_does_not_reevaluate_old_bars(self):
        """A resync/backfill that rewinds the frame must not re-run old bars."""
        engine = _make_engine()
        df6 = _make_df(6)
        df5 = _make_df(5)  # older frame after reconnect resync

        _run(engine, [df6, df5, df5])

        assert engine._runtime_process_decision.call_count == 1
        assert engine._runtime_process_decision.call_args.args[1] == 4  # df6's frontier

    def test_buffer_close_event_promotes_tail_to_decision_bar(self):
        """When the kline buffer saw x=true for the tail, the decision runs on
        the tail itself (no one-bar lag)."""
        engine = _make_engine()
        df = _make_df(5)
        engine._kline_buffer = Mock(last_closed_bar_time=df.index[-1])

        _run(engine, [df])

        assert engine._runtime_process_decision.call_count == 1
        assert engine._runtime_process_decision.call_args.args[1] == 4

    def test_safety_mode_defers_but_does_not_consume_the_bar(self):
        """A transiently unready context skips evaluation without marking the
        bar decided; the next safe tick evaluates it once."""
        engine = _make_engine()
        df = _make_df(5)
        _stub_loop_periphery(engine, [df, df])
        engine._is_context_ready = Mock(side_effect=[(False, "no_pred"), (True, "")])

        engine._trading_loop("BTCUSDT", "1h", max_steps=2)

        assert engine._runtime_process_decision.call_count == 1
        assert engine._check_entry_conditions.call_count == 1

    def test_metadata_stamped_closed_in_on_mode(self):
        engine = _make_engine()
        df = _make_df(5)

        _run(engine, [df])

        decision = engine._check_entry_conditions.call_args.kwargs["runtime_decision"]
        md = decision.signal.metadata
        assert md["decision_bar_closed"] is True
        assert md["decision_bar_open_time"] == df.index[-2].isoformat()
        assert md["decision_bar_close_time"] == df.index[-1].isoformat()

    def test_gated_evaluation_logs_at_info(self, caplog):
        engine = _make_engine()
        df = _make_df(5)

        with caplog.at_level(logging.INFO, logger="src.engines.live.trading_engine"):
            _run(engine, [df])

        assert "decided on closed bar" in caplog.text
