"""Wiring tests for closed-trade feedback into statistics-tracking position sizers (#840).

KellyCriterionSizer.record_trade previously had zero engine callers, so any
Kelly-sized strategy ran permanently in cold-start fallback. The fix routes
closed-trade outcomes through the shared seam both engines already use:

    engine close path -> PerformanceTracker.record_trade -> trade listeners
        -> Strategy.on_trade_closed -> position_sizer.record_trade (duck-typed)

These tests cover the strategy-side hook, both engines' listener registration,
backtest/live parity of the resulting sizer state, and the cold-start -> warm
Kelly transition through the engine chain.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.config.constants import DEFAULT_KELLY_FALLBACK_FRACTION
from src.engines.backtest.engine import Backtester
from src.engines.shared.models import BaseTrade, PositionSide
from src.strategies.components import (
    EnhancedRegimeDetector,
    Signal,
    SignalDirection,
    SignalGenerator,
    Strategy,
)
from src.strategies.components.position_sizer import (
    FixedFractionSizer,
    KellyCriterionSizer,
    LeveragedPositionSizer,
)
from src.strategies.components.risk_manager import RiskManager as ComponentRiskManager
from src.strategies.kelly_momentum import create_kelly_momentum_strategy

pytestmark = pytest.mark.unit


ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _make_trade(
    pnl: float,
    pnl_percent: float | None,
    *,
    symbol: str = "ETHUSDT",
) -> BaseTrade:
    """Build a completed trade with the shared BaseTrade model."""
    return BaseTrade(
        symbol=symbol,
        side=PositionSide.LONG,
        entry_price=100.0,
        exit_price=100.0 * (1.0 + (pnl_percent or 0.0)),
        entry_time=ENTRY_TIME,
        exit_time=ENTRY_TIME + timedelta(hours=4),
        size=0.1,
        pnl=pnl,
        pnl_percent=pnl_percent,
        exit_reason="test",
    )


def _make_signal(
    direction: SignalDirection = SignalDirection.BUY,
    strength: float = 0.8,
    confidence: float = 0.9,
) -> Signal:
    return Signal(direction=direction, strength=strength, confidence=confidence, metadata={})


def _static_data_provider(df: pd.DataFrame):
    """Minimal deterministic data provider for engine-level tests."""
    from src.data_providers.data_provider import DataProvider

    class _Provider(DataProvider):
        def __init__(self, frame: pd.DataFrame) -> None:
            super().__init__()
            self._df = frame

        def get_historical_data(self, symbol, timeframe, start=None, end=None):
            return self._df.copy()

        def get_live_data(self, symbol, timeframe, limit=100):
            raise NotImplementedError

        def update_live_data(self, symbol, timeframe):
            raise NotImplementedError

        def get_current_price(self, symbol) -> float:
            return float(self._df["close"].iloc[-1])

    return _Provider(df)


def _ohlcv(closes: list[float]) -> pd.DataFrame:
    index = pd.date_range(datetime(2024, 1, 1, tzinfo=UTC), periods=len(closes), freq="1h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1.0] * len(closes),
        },
        index=index,
    )


class _ScriptedSignalGenerator(SignalGenerator):
    """Signal generator that follows a per-index script of directions."""

    def __init__(self, script: dict[int, str]) -> None:
        super().__init__("scripted_signals")
        self._script = script

    def generate_signal(self, df, index, regime=None):
        self.validate_inputs(df, index)
        action = self._script.get(index, "hold")
        if action == "buy":
            return Signal(SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={})
        if action == "sell":
            return Signal(SignalDirection.SELL, strength=1.0, confidence=1.0, metadata={})
        return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

    def get_confidence(self, df, index):
        return 1.0


class _WideStopRiskManager(ComponentRiskManager):
    """Fixed-fraction risk with stops wide enough to never trigger in tests."""

    def __init__(self, fraction: float = 0.1) -> None:
        super().__init__("wide_stop_risk")
        self._fraction = fraction

    def calculate_position_size(self, signal, balance, regime=None):
        if balance <= 0 or signal.direction == SignalDirection.HOLD:
            return 0.0
        return balance * self._fraction

    def should_exit(self, position, current_data, regime=None) -> bool:
        return False

    def get_stop_loss(self, entry_price, signal, regime=None) -> float:
        if signal.direction == SignalDirection.BUY:
            return entry_price * 0.5
        return entry_price * 1.5


def _scripted_kelly_strategy(script: dict[int, str], sizer: KellyCriterionSizer) -> Strategy:
    """Component strategy with scripted entries/exits and a Kelly sizer."""
    strategy = Strategy(
        name="ScriptedKelly",
        signal_generator=_ScriptedSignalGenerator(script),
        risk_manager=_WideStopRiskManager(),
        position_sizer=sizer,
        regime_detector=EnhancedRegimeDetector(),
    )
    strategy.set_risk_overrides({"stop_loss_pct": 0.5, "take_profit_pct": 0.9})
    return strategy


class TestStrategyOnTradeClosed:
    """Strategy.on_trade_closed forwards outcomes to record_trade sizers."""

    def test_winning_trade_forwarded_to_kelly_sizer(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)
        sizer = strategy.position_sizer
        assert isinstance(sizer, KellyCriterionSizer)

        strategy.on_trade_closed(_make_trade(pnl=40.0, pnl_percent=0.04))

        assert sizer.trade_count == 1
        assert list(sizer._trades) == [(True, pytest.approx(0.04), pytest.approx(0.04))]

    def test_losing_trade_forwarded_with_positive_magnitudes(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)
        sizer = strategy.position_sizer

        strategy.on_trade_closed(_make_trade(pnl=-20.0, pnl_percent=-0.02))

        assert sizer.trade_count == 1
        assert list(sizer._trades) == [(False, pytest.approx(0.02), pytest.approx(0.02))]

    def test_breakeven_trade_not_recorded(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(pnl=0.0, pnl_percent=0.0))

        assert strategy.position_sizer.trade_count == 0

    def test_trade_without_pnl_percent_ignored(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(pnl=10.0, pnl_percent=None))

        assert strategy.position_sizer.trade_count == 0

    def test_non_finite_pnl_percent_ignored(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(pnl=10.0, pnl_percent=float("nan")))
        strategy.on_trade_closed(_make_trade(pnl=10.0, pnl_percent=float("inf")))

        assert strategy.position_sizer.trade_count == 0

    def test_sizer_without_record_trade_is_noop(self):
        strategy = Strategy(
            name="NoFeedback",
            signal_generator=_ScriptedSignalGenerator({}),
            risk_manager=_WideStopRiskManager(),
            position_sizer=FixedFractionSizer(),
            regime_detector=EnhancedRegimeDetector(),
        )

        # Must not raise for sizers that do not track statistics.
        strategy.on_trade_closed(_make_trade(pnl=10.0, pnl_percent=0.01))


class TestLeveragedSizerDelegation:
    """LeveragedPositionSizer forwards record_trade to its base sizer."""

    def test_delegates_to_kelly_base_sizer(self):
        kelly = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        leverage_manager = Mock()
        wrapper = LeveragedPositionSizer(base_sizer=kelly, leverage_manager=leverage_manager)

        wrapper.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.02)

        assert kelly.trade_count == 1

    def test_noop_for_base_sizer_without_record_trade(self):
        wrapper = LeveragedPositionSizer(base_sizer=FixedFractionSizer(), leverage_manager=Mock())

        # Must not raise when the base sizer does not track statistics.
        wrapper.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.02)


class TestBacktestEngineWiring:
    """Backtester routes recorded trades to the strategy's position sizer."""

    def test_recorded_trade_reaches_kelly_sizer(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)
        engine = Backtester(
            strategy=strategy,
            data_provider=_static_data_provider(_ohlcv([100.0] * 10)),
            initial_balance=10_000.0,
            log_to_database=False,
        )

        engine.performance_tracker.record_trade(trade=_make_trade(pnl=40.0, pnl_percent=0.04))

        assert strategy.position_sizer.trade_count == 1

    def test_full_backtest_run_records_closed_trade(self):
        """End-to-end: a scripted buy->sell round trip feeds the Kelly sizer."""
        sizer = KellyCriterionSizer(min_trades=2, lookback_trades=20)
        strategy = _scripted_kelly_strategy({0: "buy", 4: "sell"}, sizer)
        closes = [100.0, 102.0, 104.0, 106.0, 108.0, 108.0, 108.0, 108.0]
        engine = Backtester(
            strategy=strategy,
            data_provider=_static_data_provider(_ohlcv(closes)),
            initial_balance=10_000.0,
            log_to_database=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            enable_partial_operations=False,
        )

        engine.run("ETHUSDT", "1h", datetime(2024, 1, 1))

        assert len(engine.trades) == 1
        assert sizer.trade_count == 1
        win, profit_pct, loss_risk_pct = sizer._trades[0]
        assert win is True
        assert profit_pct == pytest.approx(abs(engine.trades[0].pnl_percent))
        assert loss_risk_pct == pytest.approx(abs(engine.trades[0].pnl_percent))


class TestLiveEngineWiring:
    """LiveTradingEngine routes recorded trades to the strategy's position sizer."""

    @pytest.fixture(autouse=True)
    def mock_database_manager(self, monkeypatch):
        from tests.mocks import MockDatabaseManager

        monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)

    def _make_engine(self, strategy):
        from src.engines.live.trading_engine import LiveTradingEngine

        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0
        return LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

    def test_recorded_trade_reaches_kelly_sizer(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)
        engine = self._make_engine(strategy)

        engine.performance_tracker.record_trade(trade=_make_trade(pnl=40.0, pnl_percent=0.04))

        assert strategy.position_sizer.trade_count == 1

    def test_live_exit_path_records_trade_outcome(self):
        """End-to-end through the real live close path (exit coordinator)."""
        from src.engines.live.trading_engine import Position

        strategy = create_kelly_momentum_strategy(min_trades=5)
        engine = self._make_engine(strategy)

        position = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="test-order",
            original_size=0.1,
            current_size=0.1,
        )
        engine.live_position_tracker.track_recovered_position(position, db_id=None)

        engine._execute_exit(
            position=position,
            reason="take_profit",
            limit_price=None,
            current_price=110.0,
            candle_high=None,
            candle_low=None,
            candle=None,
            skip_live_close=False,
        )

        sizer = strategy.position_sizer
        assert sizer.trade_count == 1
        win, profit_pct, _ = sizer._trades[0]
        assert win is True
        # 10% price move on a 10%-of-balance position -> 1% sized return.
        assert profit_pct == pytest.approx(0.01, rel=1e-6)


class TestBacktestLiveParity:
    """Identical trade sequences must produce identical sizer state in both engines."""

    @pytest.fixture(autouse=True)
    def mock_database_manager(self, monkeypatch):
        from tests.mocks import MockDatabaseManager

        monkeypatch.setattr("src.engines.live.trading_engine.DatabaseManager", MockDatabaseManager)

    def test_same_trade_sequence_same_sizer_state_and_size(self):
        from src.engines.live.trading_engine import LiveTradingEngine

        bt_strategy = create_kelly_momentum_strategy(min_trades=5)
        live_strategy = create_kelly_momentum_strategy(min_trades=5)

        backtester = Backtester(
            strategy=bt_strategy,
            data_provider=_static_data_provider(_ohlcv([100.0] * 10)),
            initial_balance=10_000.0,
            log_to_database=False,
        )
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0
        live_engine = LiveTradingEngine(
            strategy=live_strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
        )

        outcomes = [
            (40.0, 0.04),
            (-20.0, -0.02),
            (60.0, 0.06),
            (30.0, 0.03),
            (-10.0, -0.01),
            (50.0, 0.05),
        ]
        for pnl, pct in outcomes:
            backtester.performance_tracker.record_trade(trade=_make_trade(pnl, pct))
            live_engine.performance_tracker.record_trade(trade=_make_trade(pnl, pct))

        bt_sizer = bt_strategy.position_sizer
        live_sizer = live_strategy.position_sizer
        assert list(bt_sizer._trades) == list(live_sizer._trades)
        assert bt_sizer.has_sufficient_history and live_sizer.has_sufficient_history

        signal = _make_signal()
        bt_size = bt_sizer.calculate_size(signal, balance=10_000.0, risk_amount=5_000.0)
        live_size = live_sizer.calculate_size(signal, balance=10_000.0, risk_amount=5_000.0)
        assert bt_size == pytest.approx(live_size)
        assert bt_size > 0


class TestColdStartToWarmTransition:
    """Sizer output must change once min_trades outcomes flow through the engine."""

    def test_kelly_activates_after_min_trades_via_engine(self):
        strategy = create_kelly_momentum_strategy(min_trades=3)
        engine = Backtester(
            strategy=strategy,
            data_provider=_static_data_provider(_ohlcv([100.0] * 10)),
            initial_balance=10_000.0,
            log_to_database=False,
        )
        sizer = strategy.position_sizer
        signal = _make_signal(strength=1.0, confidence=1.0)

        assert not sizer.has_sufficient_history
        cold_size = sizer.calculate_size(signal, balance=10_000.0, risk_amount=10_000.0)
        # Cold start sizes with fallback_fraction: 0.02 * 10_000 = 200.
        assert cold_size == pytest.approx(sizer.fallback_fraction * 10_000.0)

        # Three strong wins (2:1 reward-to-risk) through the engine seam.
        for pnl, pct in [(40.0, 0.04), (-20.0, -0.02), (40.0, 0.04)]:
            engine.performance_tracker.record_trade(trade=_make_trade(pnl, pct))

        assert sizer.has_sufficient_history
        warm_size = sizer.calculate_size(signal, balance=10_000.0, risk_amount=10_000.0)
        assert warm_size > 0
        assert warm_size != pytest.approx(cold_size)


class TestKellyMomentumDefaults:
    """kelly_momentum must not silently diverge from the Kelly constants."""

    def test_fallback_fraction_matches_shared_constant(self):
        strategy = create_kelly_momentum_strategy()
        sizer = strategy.position_sizer
        assert sizer.fallback_fraction == DEFAULT_KELLY_FALLBACK_FRACTION
        assert strategy.base_position_size == DEFAULT_KELLY_FALLBACK_FRACTION
