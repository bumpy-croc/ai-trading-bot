"""Regression tests for partial-exit accounting units in the backtest engine.

Bug family: the partial-exit pipeline mixed two incompatible units.
Position sizes (``BasePosition.size`` / ``current_size`` / ``original_size``)
are fractions of BALANCE, while the exit handler converted the policy's
fraction-of-original into a fraction-of-current-POSITION and then:

1. passed it to ``PartialExitExecutor`` which treats ``exit_fraction`` as a
   fraction of ``basis_balance`` — booking P&L as if the exited slice were up
   to 100% of the account (up to ~3,900x inflation on small positions);
2. subtracted it from ``current_size`` (balance units), clamping positions to
   zero so final closes booked Trade.pnl = 0.0;
3. allowed scale-ins to revive those zeroed positions, whose next partial
   exits booked with exit-fraction-of-current = 1.0.

The correct conversion is ``fraction_of_original * original_size`` — a
balance-fraction delta consistent with both the P&L basis and the size
decrement (the same conversion ``PartialExitPolicy.apply_partial_exit``
already used).

Also covered here:

4. drawdown must reflect open-position mark-to-market, not just realized
   cash balance;
5. strategy-declared ``partial_operations`` overrides must win over
   ``DEFAULT_PARTIAL_EXIT_TARGETS``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.engines.backtest.engine import Backtester
from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.models import ActiveTrade
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.models import PositionSide
from src.engines.shared.partial_operations_manager import PartialOperationsManager
from src.position_management.partial_manager import PartialExitPolicy

pytestmark = pytest.mark.unit


ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)


def _make_exit_handler(
    policy: PartialExitPolicy,
) -> tuple[ExitHandler, PositionTracker, MagicMock]:
    """Build an ExitHandler with zero-cost execution and a real policy."""
    tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)

    risk_manager = MagicMock()
    risk_manager.params = SimpleNamespace(max_daily_risk=1.0)
    risk_manager.daily_risk_used = 0.0

    execution_engine = MagicMock()
    execution_engine.calculate_scale_in_costs.return_value = (0.0, 0.0)

    handler = ExitHandler(
        execution_engine=execution_engine,
        position_tracker=tracker,
        risk_manager=risk_manager,
        execution_model=ExecutionModel(default_fill_policy()),
        partial_manager=PartialOperationsManager(policy=policy),
    )
    return handler, tracker, risk_manager


def _open_long(
    tracker: PositionTracker,
    *,
    size: float = 0.10,
    entry_price: float = 100.0,
    entry_balance: float = 1000.0,
) -> ActiveTrade:
    trade = ActiveTrade(
        symbol="ETHUSDT",
        side=PositionSide.LONG,
        entry_price=entry_price,
        entry_time=ENTRY_TIME,
        size=size,
        stop_loss=entry_price * 0.5,
        entry_balance=entry_balance,
    )
    tracker.open_position(trade)
    return trade


def _single_candle_df(price: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [price],
            "high": [price],
            "low": [price],
            "close": [price],
            "volume": [1.0],
        },
        index=pd.DatetimeIndex([datetime(2024, 1, 2, tzinfo=UTC)]),
    )


class TestPartialExitUnits:
    """Bug 1: exit fraction must be a balance-fraction slice, not a position fraction."""

    def test_partial_exit_books_pnl_proportional_to_exited_slice(self) -> None:
        """A 25%-of-original partial on a 10%-of-balance position at +5%
        must book 5% x 2.5% x basis = $1.25 — NOT 5% x 25% x basis = $12.50."""
        policy = PartialExitPolicy(exit_targets=[0.05], exit_sizes=[0.25])
        handler, tracker, _ = _make_exit_handler(policy)
        _open_long(tracker, size=0.10, entry_price=100.0, entry_balance=1000.0)

        result = handler.check_partial_operations(
            current_price=105.0,
            df=_single_candle_df(105.0),
            index=0,
            balance=1000.0,
        )

        assert result.realized_pnl == pytest.approx(1.25)
        assert len(result.partial_exits) == 1
        assert result.partial_exits[0]["size"] == pytest.approx(0.025)

    def test_partial_exit_decrements_current_size_in_balance_units(self) -> None:
        """Bug 2: subtracting a position-fraction from balance-fraction state
        clamped a 10% position to zero after one 25% partial."""
        policy = PartialExitPolicy(exit_targets=[0.05], exit_sizes=[0.25])
        handler, tracker, _ = _make_exit_handler(policy)
        trade = _open_long(tracker, size=0.10, entry_price=100.0, entry_balance=1000.0)

        handler.check_partial_operations(
            current_price=105.0,
            df=_single_candle_df(105.0),
            index=0,
            balance=1000.0,
        )

        assert trade.current_size == pytest.approx(0.075)
        assert trade.partial_exits_taken == 1

    def test_final_close_after_partial_books_pnl_on_remaining_size(self) -> None:
        """Bug 2 artifact: zeroed positions produced Trade.pnl = 0.0 on close."""
        policy = PartialExitPolicy(exit_targets=[0.05], exit_sizes=[0.25])
        handler, tracker, _ = _make_exit_handler(policy)
        _open_long(tracker, size=0.10, entry_price=100.0, entry_balance=1000.0)

        handler.check_partial_operations(
            current_price=105.0,
            df=_single_candle_df(105.0),
            index=0,
            balance=1000.0,
        )

        close_result = tracker.close_position(
            exit_price=105.0,
            exit_time=datetime(2024, 1, 3, tzinfo=UTC),
            exit_reason="Signal",
            basis_balance=1000.0,
        )

        # Remaining 7.5%-of-balance slice at +5%: 0.05 * 0.075 * 1000 = $3.75
        assert close_result.pnl_cash == pytest.approx(3.75)
        assert close_result.trade.pnl == pytest.approx(3.75)
        assert close_result.trade.size == pytest.approx(0.075)

    def test_risk_manager_adjustment_uses_balance_fraction_delta(self) -> None:
        """Risk manager exposure is tracked in balance-fraction units; the
        partial-exit adjustment must be passed in the same units."""
        policy = PartialExitPolicy(exit_targets=[0.05], exit_sizes=[0.25])
        handler, tracker, risk_manager = _make_exit_handler(policy)
        _open_long(tracker, size=0.10, entry_price=100.0, entry_balance=1000.0)

        handler.check_partial_operations(
            current_price=105.0,
            df=_single_candle_df(105.0),
            index=0,
            balance=1000.0,
        )

        risk_manager.adjust_position_after_partial_exit.assert_called_once()
        _, delta = risk_manager.adjust_position_after_partial_exit.call_args.args
        assert delta == pytest.approx(0.025)

    def test_sequential_partials_consume_exact_original_fractions(self) -> None:
        """Two partials of 25% + 25% of original must leave exactly half the
        original balance-fraction, with per-slice P&L on each slice."""
        policy = PartialExitPolicy(exit_targets=[0.03, 0.06], exit_sizes=[0.25, 0.25])
        handler, tracker, _ = _make_exit_handler(policy)
        trade = _open_long(tracker, size=0.20, entry_price=100.0, entry_balance=1000.0)

        result = handler.check_partial_operations(
            current_price=110.0,
            df=_single_candle_df(110.0),
            index=0,
            balance=1000.0,
        )

        # Each slice: 25% x 0.20 = 0.05 of balance; pnl 10% x 0.05 x 1000 = $5
        assert len(result.partial_exits) == 2
        assert result.realized_pnl == pytest.approx(10.0)
        assert trade.current_size == pytest.approx(0.10)


class TestScaleInGuard:
    """Bug 3: a fully-closed position must not be revived by a scale-in."""

    def test_fully_closed_position_cannot_scale_in(self) -> None:
        policy = PartialExitPolicy(
            exit_targets=[0.02],
            exit_sizes=[1.0],
            scale_in_thresholds=[0.01],
            scale_in_sizes=[0.5],
            max_scale_ins=1,
        )
        handler, tracker, _ = _make_exit_handler(policy)
        trade = _open_long(tracker, size=0.10, entry_price=100.0, entry_balance=1000.0)

        result = handler.check_partial_operations(
            current_price=105.0,
            df=_single_candle_df(105.0),
            index=0,
            balance=1000.0,
        )

        # Whole position exits at the first target...
        assert trade.current_size == pytest.approx(0.0)
        # ...and the scale-in must NOT revive it.
        assert result.scale_ins == []
        assert trade.scale_ins_taken == 0
        assert trade.current_size == pytest.approx(0.0)


class _ScriptedSignalGenerator:
    """Signal generator that follows a per-index script of directions."""

    def __init__(self, script: dict[int, str]) -> None:
        from src.strategies.components import SignalGenerator

        # Build dynamically to satisfy the ABC without a module-level subclass.
        outer_script = script

        class _Inner(SignalGenerator):
            def __init__(self) -> None:
                super().__init__("scripted_signals")

            def generate_signal(self, df, index, regime=None):
                from src.strategies.components import Signal, SignalDirection

                self.validate_inputs(df, index)
                action = outer_script.get(index, "hold")
                if action == "buy":
                    return Signal(SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={})
                if action == "sell":
                    return Signal(SignalDirection.SELL, strength=1.0, confidence=1.0, metadata={})
                return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

            def get_confidence(self, df, index):
                return 1.0

        self.generator = _Inner()


def _build_scripted_strategy(
    script: dict[int, str],
    *,
    position_fraction: float,
    risk_overrides: dict | None = None,
):
    """Component strategy that enters/exits per script at a fixed fraction."""
    from src.strategies.components import EnhancedRegimeDetector, SignalDirection, Strategy
    from src.strategies.components.position_sizer import PositionSizer
    from src.strategies.components.risk_manager import RiskManager as ComponentRiskManager

    class _FixedFractionRiskManager(ComponentRiskManager):
        def __init__(self) -> None:
            super().__init__("fixed_fraction_risk")

        def calculate_position_size(self, signal, balance, regime=None):
            if balance <= 0 or signal.direction == SignalDirection.HOLD:
                return 0.0
            return balance * position_fraction

        def should_exit(self, position, current_data, regime=None) -> bool:
            return False

        def get_stop_loss(self, entry_price, signal, regime=None) -> float:
            # Wide stop so engine-level SL never triggers in these tests.
            if signal.direction == SignalDirection.BUY:
                return entry_price * 0.5
            return entry_price * 1.5

    class _FractionSizer(PositionSizer):
        def __init__(self) -> None:
            super().__init__("fixed_fraction_sizer")

        def calculate_size(self, signal, balance, risk_amount, regime=None):
            if signal.direction == SignalDirection.HOLD:
                return 0.0
            # Strategy.position_size is a CASH amount; the engine converts
            # it back to a balance fraction at entry.
            return balance * position_fraction

        def get_parameters(self):
            return {"name": self.name, "fraction": position_fraction}

    strategy = Strategy(
        name="ScriptedPartialsStrategy",
        signal_generator=_ScriptedSignalGenerator(script).generator,
        risk_manager=_FixedFractionRiskManager(),
        position_sizer=_FractionSizer(),
        regime_detector=EnhancedRegimeDetector(),
    )
    if risk_overrides:
        strategy.set_risk_overrides(risk_overrides)
    return strategy


class _StaticDataProvider:
    """Minimal deterministic data provider for engine-level tests."""

    def __new__(cls, df: pd.DataFrame):
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


def _flat_ohlcv(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=len(closes), freq="1h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=idx,
    )


class TestEngineLevelPartialAccounting:
    """End-to-end invariants through Backtester.run."""

    def test_balance_invariant_with_partial_exits(self) -> None:
        """final_balance - initial_balance must equal the sum of realized
        partial P&L plus the final trade P&L (zero fees/slippage)."""
        # Entry signal at idx 0 -> fills at idx 1 open (100).
        # +4% at idx 2 -> partial 1 (25% of original).
        # +7% at idx 3 -> partial 2 (25% of original).
        # SELL at idx 4 (110) -> close remaining half.
        closes = [100.0, 100.0, 104.0, 107.0, 110.0, 110.0]
        strategy = _build_scripted_strategy(
            {0: "buy", 4: "sell"},
            position_fraction=0.10,
            risk_overrides={
                # Wide SL/TP so only partials + the scripted SELL exit.
                "stop_loss_pct": 0.5,
                "take_profit_pct": 0.9,
                "partial_operations": {
                    "exit_targets": [0.03, 0.06],
                    "exit_sizes": [0.25, 0.25],
                },
            },
        )
        backtester = Backtester(
            strategy=strategy,
            data_provider=_StaticDataProvider(_flat_ohlcv(closes)),
            initial_balance=1000.0,
            log_to_database=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        results = backtester.run("ETHUSDT", "1h", datetime(2024, 1, 1))

        # Slices: partial1 = 4% x 0.025 x 1000 = $1.00
        #         partial2 = 7% x 0.025 x 1000 = $1.75
        #         final    = 10% x 0.05  x 1000 = $5.00
        assert results["total_trades"] == 1
        trade_pnls = [t.pnl for t in backtester.trades]
        assert trade_pnls == [pytest.approx(5.0)]
        assert results["final_balance"] == pytest.approx(1000.0 + 1.0 + 1.75 + 5.0)
        # A profitable run with partial exits must not report a 0% win rate.
        assert results["win_rate"] == pytest.approx(100.0)

    def test_position_fully_consumed_by_partials_closes_like_live(self) -> None:
        """When partial exits consume 100% of the position, the engine must
        close it (parity with live's 'Partial exits complete' path) instead
        of leaving a zombie position open."""
        closes = [100.0, 100.0, 112.0, 112.0, 112.0]
        strategy = _build_scripted_strategy(
            {0: "buy"},
            position_fraction=0.10,
            risk_overrides={
                # Wide SL/TP so the partials-complete path is what closes it.
                "stop_loss_pct": 0.5,
                "take_profit_pct": 0.9,
                "partial_operations": {
                    "exit_targets": [0.03, 0.06, 0.10],
                    "exit_sizes": [0.25, 0.25, 0.50],
                    # Scale-in thresholds also satisfied at +12% — must NOT
                    # revive the fully-consumed position (bug 3).
                    "scale_in_thresholds": [0.02],
                    "scale_in_sizes": [0.30],
                    "max_scale_ins": 1,
                },
            },
        )
        backtester = Backtester(
            strategy=strategy,
            data_provider=_StaticDataProvider(_flat_ohlcv(closes)),
            initial_balance=1000.0,
            log_to_database=False,
            fee_rate=0.0,
            slippage_rate=0.0,
        )

        results = backtester.run("ETHUSDT", "1h", datetime(2024, 1, 1))

        # All three partials realize 12% on their slice of the 10% position:
        # 0.12 x (0.025 + 0.025 + 0.05) x 1000 = $12.00
        assert results["final_balance"] == pytest.approx(1012.0)
        assert backtester.current_trade is None
        assert results["total_trades"] == 1
        assert "Partial exits complete" in backtester.trades[-1].exit_reason


class TestMarkToMarketDrawdown:
    """Bug 4: drawdown must include open-position adverse mark-to-market."""

    def test_open_position_adverse_excursion_appears_in_max_drawdown(self) -> None:
        # Entry at idx 1 (100), price collapses to 90 (-10% on a 10% position
        # = -1% equity), recovers, exits flat at 100.
        closes = [100.0, 100.0, 90.0, 90.0, 100.0, 100.0]
        strategy = _build_scripted_strategy(
            {0: "buy", 4: "sell"},
            position_fraction=0.10,
        )
        backtester = Backtester(
            strategy=strategy,
            data_provider=_StaticDataProvider(_flat_ohlcv(closes)),
            initial_balance=1000.0,
            log_to_database=False,
            fee_rate=0.0,
            slippage_rate=0.0,
            enable_partial_operations=False,
        )

        results = backtester.run("ETHUSDT", "1h", datetime(2024, 1, 1))

        # Realized balance never dropped, but equity drew down ~1%.
        assert results["final_balance"] == pytest.approx(1000.0)
        assert results["max_drawdown"] == pytest.approx(1.0, rel=0.05)


class TestStrategyPartialConfigHydration:
    """Bug 5: strategy-declared partial_operations must override defaults."""

    def test_strategy_partial_overrides_win_over_defaults(self) -> None:
        strategy = _build_scripted_strategy(
            {},
            position_fraction=0.10,
            risk_overrides={
                "partial_operations": {
                    "exit_targets": [0.05, 0.10, 0.20],
                    "exit_sizes": [0.25, 0.25, 0.50],
                    "scale_in_thresholds": [0.02, 0.04],
                    "scale_in_sizes": [0.30, 0.20],
                    "max_scale_ins": 2,
                }
            },
        )
        backtester = Backtester(
            strategy=strategy,
            data_provider=_StaticDataProvider(_flat_ohlcv([100.0, 100.0])),
            initial_balance=1000.0,
            log_to_database=False,
        )

        assert backtester.partial_manager is not None
        assert backtester.partial_manager.exit_targets == [0.05, 0.10, 0.20]
        assert backtester.partial_manager.exit_sizes == [0.25, 0.25, 0.50]
        assert backtester.partial_manager.scale_in_thresholds == [0.02, 0.04]
        assert backtester.partial_manager.max_scale_ins == 2

    def test_defaults_apply_only_without_strategy_config(self) -> None:
        from src.config.constants import DEFAULT_PARTIAL_EXIT_TARGETS

        strategy = _build_scripted_strategy({}, position_fraction=0.10)
        backtester = Backtester(
            strategy=strategy,
            data_provider=_StaticDataProvider(_flat_ohlcv([100.0, 100.0])),
            initial_balance=1000.0,
            log_to_database=False,
        )

        assert backtester.partial_manager is not None
        assert backtester.partial_manager.exit_targets == DEFAULT_PARTIAL_EXIT_TARGETS
