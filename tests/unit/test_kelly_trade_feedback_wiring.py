"""Wiring tests for closed-trade feedback into statistics-tracking position sizers (#840).

KellyCriterionSizer.record_trade previously had zero engine callers, so any
Kelly-sized strategy ran permanently in cold-start fallback. The fix routes
realized outcomes through two shared seams both engines already use:

    final close  -> PerformanceTracker.record_trade -> trade listeners
    partial exit -> position tracker on_partial_exit hook (both engines)
        -> Strategy.on_trade_closed -> position_sizer.record_trade (duck-typed)

Outcomes are UNSIZED R-multiples (directional price move), so past sizing
decisions cannot bias Kelly's reward:risk statistics. These tests cover the
strategy-side hook, both engines' wiring (final closes AND partial slices),
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
    price_move: float,
    *,
    side: PositionSide = PositionSide.LONG,
    size: float = 0.1,
    entry_price: float = 100.0,
    exit_price: float | None = None,
    symbol: str = "ETHUSDT",
) -> BaseTrade:
    """Build a completed trade with the shared BaseTrade model.

    ``price_move`` is the UNSIZED directional return (decimal): positive is a
    win for the given side. ``pnl``/``pnl_percent`` carry the engines' sized
    conventions (fraction-of-balance) so the trade mirrors real close records.
    """
    if exit_price is None:
        raw_move = price_move if side == PositionSide.LONG else -price_move
        exit_price = entry_price * (1.0 + raw_move)
    return BaseTrade(
        symbol=symbol,
        side=side,
        entry_price=entry_price,
        exit_price=exit_price,
        entry_time=ENTRY_TIME,
        exit_time=ENTRY_TIME + timedelta(hours=4),
        size=size,
        pnl=price_move * size * 10_000.0,
        pnl_percent=price_move * size,
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


def _scripted_kelly_strategy(
    script: dict[int, str],
    sizer: KellyCriterionSizer,
    partial_operations: dict | None = None,
) -> Strategy:
    """Component strategy with scripted entries/exits and a Kelly sizer."""
    strategy = Strategy(
        name="ScriptedKelly",
        signal_generator=_ScriptedSignalGenerator(script),
        risk_manager=_WideStopRiskManager(),
        position_sizer=sizer,
        regime_detector=EnhancedRegimeDetector(),
    )
    overrides: dict = {"stop_loss_pct": 0.5, "take_profit_pct": 0.9}
    if partial_operations is not None:
        overrides["partial_operations"] = partial_operations
    strategy.set_risk_overrides(overrides)
    return strategy


class TestStrategyOnTradeClosed:
    """Strategy.on_trade_closed forwards outcomes to record_trade sizers."""

    def test_winning_trade_forwarded_to_kelly_sizer(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)
        sizer = strategy.position_sizer
        assert isinstance(sizer, KellyCriterionSizer)

        strategy.on_trade_closed(_make_trade(price_move=0.04))

        assert sizer.trade_count == 1
        assert list(sizer._trades) == [(True, pytest.approx(0.04), pytest.approx(0.04))]

    def test_unsized_move_recorded_not_sized_return(self):
        """Regression: Kelly must see the UNSIZED R-multiple, not the sized
        pnl_percent — otherwise past sizing decisions bias reward:risk."""
        strategy = create_kelly_momentum_strategy(min_trades=5)
        sizer = strategy.position_sizer

        # +4% price move at 10% size: sized pnl_percent is 0.004.
        trade = _make_trade(price_move=0.04, size=0.1)
        assert trade.pnl_percent == pytest.approx(0.004)

        strategy.on_trade_closed(trade)

        _, profit_pct, loss_risk_pct = sizer._trades[0]
        assert profit_pct == pytest.approx(0.04)
        assert loss_risk_pct == pytest.approx(0.04)

    def test_losing_trade_forwarded_with_positive_magnitudes(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)
        sizer = strategy.position_sizer

        strategy.on_trade_closed(_make_trade(price_move=-0.02))

        assert sizer.trade_count == 1
        assert list(sizer._trades) == [(False, pytest.approx(0.02), pytest.approx(0.02))]

    def test_short_win_recorded_with_directional_move(self):
        """Short entry 100 -> exit 90 is a +10% winning move."""
        strategy = create_kelly_momentum_strategy(min_trades=5)
        sizer = strategy.position_sizer

        strategy.on_trade_closed(_make_trade(price_move=0.10, side=PositionSide.SHORT))

        assert list(sizer._trades) == [(True, pytest.approx(0.10), pytest.approx(0.10))]

    def test_breakeven_trade_not_recorded(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(price_move=0.0))

        assert strategy.position_sizer.trade_count == 0

    def test_zero_size_bookkeeping_close_not_recorded(self):
        """The near-zero-size close emitted after partials fully consume a
        position must not double count the final slice."""
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(price_move=0.10, size=0.0))
        strategy.on_trade_closed(_make_trade(price_move=0.10, size=1e-12))

        assert strategy.position_sizer.trade_count == 0

    def test_invalid_entry_price_ignored(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(price_move=0.04, entry_price=0.0, exit_price=104.0))

        assert strategy.position_sizer.trade_count == 0

    def test_non_finite_exit_price_ignored(self):
        strategy = create_kelly_momentum_strategy(min_trades=5)

        strategy.on_trade_closed(_make_trade(price_move=0.04, exit_price=float("nan")))
        strategy.on_trade_closed(_make_trade(price_move=0.04, exit_price=float("inf")))

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
        strategy.on_trade_closed(_make_trade(price_move=0.01))


class TestKellySizerSeam:
    """The legacy KellySizer warms up through the same record_trade seam."""

    def test_kelly_sizer_warms_up_via_on_trade_closed(self):
        from src.strategies.components.position_sizer import KellySizer

        sizer = KellySizer(lookback_period=100)
        strategy = Strategy(
            name="LegacyKelly",
            signal_generator=_ScriptedSignalGenerator({}),
            risk_manager=_WideStopRiskManager(),
            position_sizer=sizer,
            regime_detector=EnhancedRegimeDetector(),
        )

        # 15 wins at +4%, 10 losses at -2% (>= 20 trades updates statistics).
        for _ in range(15):
            strategy.on_trade_closed(_make_trade(price_move=0.04))
        for _ in range(10):
            strategy.on_trade_closed(_make_trade(price_move=-0.02))

        assert len(sizer.trade_history) == 25
        assert sizer.win_rate == pytest.approx(0.6)
        assert sizer.avg_win == pytest.approx(0.04)
        assert sizer.avg_loss == pytest.approx(0.02)

    def test_kelly_sizer_record_trade_gates_invalid_input(self):
        from src.strategies.components.position_sizer import KellySizer

        sizer = KellySizer()

        sizer.record_trade(win=True, profit_pct=0.04, loss_risk_pct=0.0)
        sizer.record_trade(win=True, profit_pct=float("nan"), loss_risk_pct=0.02)

        assert len(sizer.trade_history) == 0


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

        engine.performance_tracker.record_trade(trade=_make_trade(price_move=0.04))

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
        # Unsized directional move, NOT the sized pnl_percent.
        closed = engine.trades[0]
        expected_move = (closed.exit_price - closed.entry_price) / closed.entry_price
        assert profit_pct == pytest.approx(expected_move)
        assert loss_risk_pct == pytest.approx(expected_move)

    def test_full_backtest_run_with_partials_feeds_each_slice(self):
        """Each banked partial-exit slice counts as one Kelly outcome, plus
        the final close of the nonzero remainder."""
        sizer = KellyCriterionSizer(min_trades=2, lookback_trades=20)
        strategy = _scripted_kelly_strategy(
            {0: "buy", 4: "sell"},
            sizer,
            partial_operations={
                "exit_targets": [0.03, 0.06],
                "exit_sizes": [0.25, 0.25],
            },
        )
        closes = [100.0, 102.0, 104.0, 106.0, 108.0, 108.0, 108.0, 108.0]
        engine = Backtester(
            strategy=strategy,
            data_provider=_static_data_provider(_ohlcv(closes)),
            initial_balance=10_000.0,
            log_to_database=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        engine.run("ETHUSDT", "1h", datetime(2024, 1, 1))

        # Two partial slices (3% and 6% targets) + final close of remainder.
        assert sizer.trade_count == 3
        assert all(win for win, _, _ in sizer._trades)
        moves = [profit for _, profit, _ in sizer._trades]
        assert moves == [
            pytest.approx(0.04),  # slice at 104
            pytest.approx(0.06),  # slice at 106
            pytest.approx(0.08),  # remainder closed at 108
        ]


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

        engine.performance_tracker.record_trade(trade=_make_trade(price_move=0.04))

        assert strategy.position_sizer.trade_count == 1

    def test_partial_exit_slice_reaches_kelly_sizer(self):
        """A banked partial-exit slice feeds the sizer through the live tracker hook."""
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

        result = engine.live_position_tracker.apply_partial_exit(
            order_id="test-order",
            delta_fraction=0.025,
            price=105.0,
            target_level=0,
            basis_balance=10_000.0,
        )

        assert result is not None
        sizer = strategy.position_sizer
        assert list(sizer._trades) == [(True, pytest.approx(0.05), pytest.approx(0.05))]

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
        # Unsized R-multiple: a 10% price move records 0.10 regardless of the
        # 10%-of-balance position size (sized return would be 0.01).
        assert profit_pct == pytest.approx(0.10, rel=1e-6)


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

        moves = [0.04, -0.02, 0.06, 0.03, -0.01, 0.05]
        for move in moves:
            backtester.performance_tracker.record_trade(trade=_make_trade(price_move=move))
            live_engine.performance_tracker.record_trade(trade=_make_trade(price_move=move))

        bt_sizer = bt_strategy.position_sizer
        live_sizer = live_strategy.position_sizer
        assert list(bt_sizer._trades) == list(live_sizer._trades)
        assert bt_sizer.has_sufficient_history and live_sizer.has_sufficient_history

        signal = _make_signal()
        bt_size = bt_sizer.calculate_size(signal, balance=10_000.0, risk_amount=5_000.0)
        live_size = live_sizer.calculate_size(signal, balance=10_000.0, risk_amount=5_000.0)
        assert bt_size == pytest.approx(live_size)
        assert bt_size > 0

    def test_partial_slice_sequence_parity_across_engines(self):
        """Same partial slice + final close through each engine's real hooks
        must leave both sizers in identical state."""
        from src.engines.backtest.models import ActiveTrade
        from src.engines.live.trading_engine import LiveTradingEngine, Position

        bt_strategy = create_kelly_momentum_strategy(min_trades=5)
        live_strategy = create_kelly_momentum_strategy(min_trades=5)

        backtester = Backtester(
            strategy=bt_strategy,
            data_provider=_static_data_provider(_ohlcv([100.0] * 10)),
            initial_balance=10_000.0,
            log_to_database=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )
        data_provider = Mock()
        data_provider.get_current_price.return_value = 100.0
        live_engine = LiveTradingEngine(
            strategy=live_strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            enable_live_trading=False,
            log_trades=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        # Same position, same partial slice, through each engine's tracker.
        backtester.position_tracker.open_position(
            ActiveTrade(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_time=ENTRY_TIME,
                size=0.1,
                stop_loss=50.0,
                entry_balance=10_000.0,
            )
        )
        live_position = Position(
            symbol="ETHUSDT",
            side=PositionSide.LONG,
            size=0.1,
            entry_price=100.0,
            entry_time=datetime.now(UTC),
            order_id="parity-order",
            original_size=0.1,
            current_size=0.1,
        )
        live_engine.live_position_tracker.track_recovered_position(live_position, db_id=None)

        backtester.position_tracker.apply_partial_exit(
            exit_fraction=0.025, current_price=105.0, basis_balance=10_000.0
        )
        live_engine.live_position_tracker.apply_partial_exit(
            order_id="parity-order",
            delta_fraction=0.025,
            price=105.0,
            target_level=0,
            basis_balance=10_000.0,
        )

        # Final close of the remainder through the shared tracker seam.
        final_close = _make_trade(price_move=0.10, size=0.075)
        backtester.performance_tracker.record_trade(trade=final_close)
        live_engine.performance_tracker.record_trade(trade=final_close)

        bt_trades = list(bt_strategy.position_sizer._trades)
        live_trades = list(live_strategy.position_sizer._trades)
        assert bt_trades == live_trades
        assert len(bt_trades) == 2  # one slice + one final close


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

        # Two wins and a loss (2:1 reward-to-risk) through the engine seam.
        for move in [0.04, -0.02, 0.04]:
            engine.performance_tracker.record_trade(trade=_make_trade(price_move=move))

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
