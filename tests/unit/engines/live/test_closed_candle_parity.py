"""Decision parity between backtest and the gated live path (parity plan P1.0/D1).

The whole point of closed-candle gating is that, given the same market data,
the live engine makes the *same decision on the same bar* as the backtest
engine. This module pins that property end to end with a real component
strategy and the real runtime pipeline both engines share
(``StrategyRuntime.process(index, context)`` — see
``BacktestEngine._get_runtime_decision`` and
``LiveStrategyRuntimeCoordinator.runtime_process_decision``).

Structure:

- ``_reference_decisions`` reproduces the backtest driver: iterate closed bars,
  call the runtime at each bar's index.
- The live side runs the real ``_trading_loop`` over frames whose tail is a
  *forming* bar whose close deliberately contradicts the closed bar, so an
  ungated loop must disagree.
- ``test_gated_live_matches_backtest_bar_for_bar`` asserts parity.
- ``test_ungated_live_diverges_from_backtest`` is the control: it shows the
  divergence the flag exists to close, so parity above cannot pass vacuously.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data_providers.data_provider import DataProvider
from src.engines.live.trading_engine import LiveTradingEngine
from src.strategies.components import (
    RuntimeContext,
    Signal,
    SignalDirection,
    SignalGenerator,
    StrategyRuntime,
)
from src.strategies.components.position_sizer import PositionSizer
from src.strategies.components.risk_manager import MarketData, Position, RiskManager
from src.strategies.components.strategy import Strategy

WARMUP = 2


class _CloseDeltaSignalGenerator(SignalGenerator):
    """Deterministic, close-sensitive signal: up-bar buys, down-bar sells.

    Close-sensitivity is the point — the reference price is the flip mechanism
    the forming-bar study identified, so a strategy that keys off the evaluated
    bar's close makes forming-bar contamination observable.
    """

    def __init__(self) -> None:
        super().__init__(name="close_delta")

    def generate_signal(self, df: pd.DataFrame, index: int, regime: Any = None) -> Signal:
        if index < 1:
            return Signal(direction=SignalDirection.HOLD, strength=0.0, confidence=0.0)
        delta = float(df["close"].iloc[index]) - float(df["close"].iloc[index - 1])
        direction = SignalDirection.BUY if delta > 0 else SignalDirection.SELL
        return Signal(
            direction=direction,
            strength=1.0,
            confidence=1.0,
            metadata={"close": float(df["close"].iloc[index])},
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        return 1.0


class _FixedRiskManager(RiskManager):
    def __init__(self) -> None:
        super().__init__(name="fixed_risk")

    def calculate_position_size(
        self, signal: Signal, balance: float, regime: Any = None, **context: Any
    ) -> float:
        return 0.01

    def should_exit(
        self, position: Position, current_data: MarketData, regime: Any = None, **context: Any
    ) -> bool:
        return False

    def get_stop_loss(
        self, entry_price: float, signal: Signal, regime: Any = None, **context: Any
    ) -> float:
        return entry_price * 0.98


class _FixedPositionSizer(PositionSizer):
    def __init__(self) -> None:
        super().__init__(name="fixed_sizer")

    def calculate_size(
        self, signal: Signal, balance: float, risk_amount: float, regime: Any = None
    ) -> float:
        return 0.01


def _make_strategy() -> Strategy:
    strategy = Strategy(
        name="closed_candle_parity_probe",
        signal_generator=_CloseDeltaSignalGenerator(),
        risk_manager=_FixedRiskManager(),
        position_sizer=_FixedPositionSizer(),
        enable_logging=False,
    )
    # Both engines skip indices below the dataset warmup; keep it explicit and
    # identical on both sides rather than inherited from component defaults.
    strategy._warmup_override = WARMUP
    return strategy


def _closed_bars(n: int) -> pd.DataFrame:
    """Closed-bar history with alternating up/down closes."""
    idx = pd.date_range("2026-07-01", periods=n, freq="1h", tz=UTC, name="timestamp")
    closes = [100.0 + (5.0 if i % 2 else -5.0) * (i + 1) for i in range(n)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1.0 for c in closes],
            "low": [c - 1.0 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=idx,
    )


def _forming_frames(history: pd.DataFrame, ticks_per_bar: int = 3) -> list[pd.DataFrame]:
    """Live's view: each closed bar followed by a mutating forming tail bar.

    The forming bar's close is set to contradict the closed bar's direction
    (it always walks the opposite way), so any decision taken on the tail
    disagrees with the backtest decision at the closed bar.
    """
    frames: list[pd.DataFrame] = []
    interval = history.index[1] - history.index[0]
    for closed_pos in range(WARMUP, len(history)):
        closed = history.iloc[: closed_pos + 1]
        closed_close = float(closed["close"].iloc[-1])
        prev_close = float(closed["close"].iloc[-2])
        # Forming close moves opposite to the closed bar's own move.
        forming_close = closed_close - 50.0 if closed_close > prev_close else closed_close + 50.0
        for tick in range(ticks_per_bar):
            forming = pd.DataFrame(
                {
                    "open": [closed_close],
                    "high": [max(closed_close, forming_close) + 1.0],
                    "low": [min(closed_close, forming_close) - 1.0],
                    "close": [forming_close + tick],
                    "volume": [10.0 * (tick + 1)],
                },
                index=pd.DatetimeIndex([closed.index[-1] + interval], name=history.index.name),
            )
            frames.append(pd.concat([closed, forming]))
    return frames


def _reference_decisions(history: pd.DataFrame) -> list[tuple[pd.Timestamp, str, float]]:
    """Backtest-side decisions: the runtime at each closed bar's own index."""
    runtime = StrategyRuntime(_make_strategy())
    dataset = runtime.prepare_data(history)
    warmup = max(0, int(dataset.warmup_period or 0))
    out: list[tuple[pd.Timestamp, str, float]] = []
    for index in range(warmup, len(dataset.data)):
        decision = runtime.process(index, RuntimeContext(balance=10_000.0))
        out.append(
            (
                dataset.data.index[index],
                decision.signal.direction.value,
                float(dataset.data["close"].iloc[index]),
            )
        )
    runtime.finalize()
    return out


def _make_engine() -> LiveTradingEngine:
    with patch("src.engines.live.trading_engine.DatabaseManager"):
        return LiveTradingEngine(
            strategy=_make_strategy(),
            data_provider=Mock(spec=DataProvider),
            initial_balance=10_000,
            enable_live_trading=False,
        )


def _run_live(frames: list[pd.DataFrame]) -> list[tuple[pd.Timestamp, str, float]]:
    """Run the real trading loop over `frames`, recording every evaluation.

    Only the loop's *periphery* is stubbed (I/O, exits, entries, metrics); the
    decision path — gate resolution and the shared runtime — runs for real.
    """
    engine = _make_engine()
    recorded: list[tuple[pd.Timestamp, str, float]] = []
    real_decide = engine._runtime_process_decision

    def _record(df, index, balance, current_price, current_time):
        decision = real_decide(df, index, balance, current_price, current_time)
        if decision is not None:
            recorded.append(
                (df.index[index], decision.signal.direction.value, float(current_price))
            )
        return decision

    engine._runtime_process_decision = _record  # type: ignore[method-assign]
    engine.is_running = True
    engine._sleep_with_interrupt = Mock()
    engine._ensure_ws_health_monitor_alive = Mock()
    engine._get_latest_data = Mock(side_effect=list(frames))
    engine._is_data_fresh = Mock(return_value=True)
    engine.strategy_manager = None
    engine._is_context_ready = Mock(return_value=(True, ""))
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

    engine._trading_loop("BTCUSDT", "1h", max_steps=len(frames))
    return recorded


@pytest.fixture
def gating_on(monkeypatch):
    monkeypatch.setenv("FEATURE_CLOSED_CANDLE_GATING", "true")
    yield


@pytest.fixture
def gating_off(monkeypatch):
    monkeypatch.setenv("FEATURE_CLOSED_CANDLE_GATING", "false")
    yield


@pytest.mark.fast
def test_gated_live_matches_backtest_bar_for_bar(gating_on):
    """Same data in, same decision, same bar — the entire point of D1.

    Live sees each bar form tick by tick (with a deliberately contradictory
    forming tail); backtest sees only closed bars. With gating ON the two
    decision sequences must be identical: same bar timestamps, same signal
    directions, same reference prices.
    """
    history = _closed_bars(12)
    expected = _reference_decisions(history)
    # The live loop cannot decide on the final history bar until a successor
    # bar exists, so the last reference bar has no live counterpart.
    actual = _run_live(_forming_frames(history))

    assert actual, "gated live path produced no decisions"
    assert actual == expected[: len(actual)]
    # Exactly one decision per closed bar — no duplicate evaluations across
    # the three ticks per bar.
    assert len(actual) == len({bar for bar, _, _ in actual})


@pytest.mark.fast
def test_ungated_live_diverges_from_backtest(gating_off):
    """Control: without the gate, live decides on forming-bar inputs.

    The decision lands on the same bar *timestamp* (live's tail is that bar,
    still forming), but with the forming close as its reference price — so the
    decision sequence does not match backtest. If this ever stops diverging,
    the parity test above has stopped measuring anything.
    """
    history = _closed_bars(12)
    expected = _reference_decisions(history)
    actual = _run_live(_forming_frames(history))

    assert actual, "ungated live path produced no decisions"
    assert actual != expected[: len(actual)], (
        "ungated live matched backtest — the forming-bar divergence is gone, "
        "so this control no longer guards the parity test"
    )

    closes = {ts: float(c) for ts, c in history["close"].items()}
    contaminated = [
        (bar, price)
        for bar, _, price in actual
        if bar in closes and price != pytest.approx(closes[bar])
    ]
    assert contaminated, "no decision used a forming-bar reference price"


@pytest.mark.fast
def test_decision_reference_price_is_the_bars_final_close(gating_on):
    """The reference price must be the closed bar's close, not the live tick."""
    history = _closed_bars(8)
    closes = {ts: float(c) for ts, c in history["close"].items()}

    for bar, _, price in _run_live(_forming_frames(history)):
        assert price == pytest.approx(closes[bar])
