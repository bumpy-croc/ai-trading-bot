"""Scale-in slices are booked from their own fill price, not the first entry.

Before the cost-basis fix a scale-in slice earned ``exit - first_entry`` instead
of ``exit - its_own_fill``, so hyper_growth backtests (scale-ins at +1.5% / +3%)
overstated winners and understated losers. ``entry_price`` keeps driving every
trigger; ``cost_basis_price`` drives P&L, fees and mark-to-market.
"""

from __future__ import annotations

import random
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
from src.risk.risk_manager import RiskParameters

pytestmark = pytest.mark.unit

ENTRY_TIME = datetime(2024, 1, 1, tzinfo=UTC)
EXIT_TIME = datetime(2024, 1, 3, tzinfo=UTC)
BALANCE = 1000.0


def _open(
    tracker: PositionTracker,
    *,
    side: PositionSide = PositionSide.LONG,
    size: float = 0.10,
    entry_price: float = 100.0,
) -> ActiveTrade:
    trade = ActiveTrade(
        symbol="ETHUSDT",
        side=side,
        entry_price=entry_price,
        entry_time=ENTRY_TIME,
        size=size,
        stop_loss=entry_price * (0.5 if side == PositionSide.LONG else 1.5),
        entry_balance=BALANCE,
    )
    tracker.open_position(trade)
    return trade


def _close(tracker: PositionTracker, price: float):
    return tracker.close_position(
        exit_price=price,
        exit_time=EXIT_TIME,
        exit_reason="test",
        basis_balance=BALANCE,
    )


class TestHandComputedCase:
    """Enter 100, scale in at 103, exit at 104."""

    def test_scale_in_slice_earns_from_its_own_fill(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open(tracker)  # 0.10 x 1000 = $100 notional at 100

        tracker.apply_scale_in(0.03, price=103.0)  # $30 notional at 103
        result = _close(tracker, 104.0)

        first_slice = 100.0 * (104.0 / 100.0 - 1.0)  # $4.00
        scale_in_slice = 30.0 * (104.0 / 103.0 - 1.0)  # +0.97% of $30, not +4%
        assert scale_in_slice == pytest.approx(0.29126, abs=1e-4)
        assert result.pnl_cash == pytest.approx(first_slice + scale_in_slice)
        assert result.trade.pnl == pytest.approx(first_slice + scale_in_slice)
        # The old behaviour booked the slice at +4%: $4.00 + $1.20.
        assert result.pnl_cash != pytest.approx(5.2)
        assert trade.entry_price == 100.0

    def test_short_scale_in_slice_earns_from_its_own_fill(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        _open(tracker, side=PositionSide.SHORT)

        tracker.apply_scale_in(0.03, price=97.0)
        result = _close(tracker, 96.0)

        first_slice = 100.0 * (100.0 - 96.0) / 100.0  # $4.00
        scale_in_slice = 30.0 * (97.0 - 96.0) / 97.0  # $0.3093
        assert result.pnl_cash == pytest.approx(first_slice + scale_in_slice)

    def test_unrealized_pnl_marks_each_slice_from_its_own_fill(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        _open(tracker)
        tracker.apply_scale_in(0.03, price=103.0)

        unrealized = tracker.unrealized_pnl_cash(104.0, fallback_basis=BALANCE)

        assert unrealized == pytest.approx(4.0 + 30.0 * (104.0 / 103.0 - 1.0))

    def test_scale_in_slice_loses_from_its_own_fill_on_a_drop(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        _open(tracker)
        tracker.apply_scale_in(0.03, price=103.0)

        result = _close(tracker, 98.0)

        expected = 100.0 * (98.0 / 100.0 - 1.0) + 30.0 * (98.0 / 103.0 - 1.0)
        assert result.pnl_cash == pytest.approx(expected)


class TestNoScaleInIsUnchanged:
    def test_cost_basis_is_the_entry_price_and_nothing_is_recorded(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open(tracker, entry_price=123.456789)

        assert trade.avg_cost_price is None
        assert trade.cost_basis_price == 123.456789

        result = _close(tracker, 130.0)

        assert "avg_cost_price" not in result.trade.metadata
        assert result.pnl_cash == pytest.approx(0.10 * BALANCE * (130.0 / 123.456789 - 1.0))

    def test_scale_in_records_average_cost_on_the_completed_trade(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open(tracker)
        tracker.apply_scale_in(0.03, price=103.0)

        result = _close(tracker, 104.0)

        assert result.trade.metadata["avg_cost_price"] == pytest.approx(
            0.13 / (0.10 / 100 + 0.03 / 103)
        )
        assert result.trade.entry_price == 100.0
        assert trade.entry_price == 100.0


class TestScaleInCap:
    def test_only_the_slice_actually_added_is_blended(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open(tracker, size=0.10)

        # Cap leaves room for 0.02 of the requested 0.05.
        tracker.apply_scale_in(0.05, price=110.0, max_position_size=0.12)

        assert trade.current_size == pytest.approx(0.12)
        assert trade.cost_basis_price == pytest.approx(0.12 / (0.10 / 100.0 + 0.02 / 110.0))

    def test_nothing_added_leaves_cost_basis_unchanged(self) -> None:
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open(tracker, size=0.10)

        tracker.apply_scale_in(0.05, price=110.0, max_position_size=0.05)

        assert trade.cost_basis_price == 100.0


def _slice_ledger_pnl(side: PositionSide, steps: list[tuple], final_price: float) -> float:
    """Independent model: pro-rata exits across slices, each from its own cost."""
    sign = 1.0 if side == PositionSide.LONG else -1.0
    slices = [[0.10 * BALANCE / 100.0, 100.0]]  # [units, own cost]
    held_fraction = 0.10
    pnl = 0.0
    for kind, price, fraction in steps:
        if kind == "scale_in":
            slices.append([fraction * BALANCE / price, price])
            held_fraction += fraction
        else:
            share = fraction / held_fraction
            for entry in slices:
                units, cost = entry
                pnl += sign * units * share * (price - cost)
                entry[0] = units * (1.0 - share)
            held_fraction -= fraction
    return pnl + sum(sign * units * (final_price - cost) for units, cost in slices)


class TestPnlEqualsSliceLedger:
    @pytest.mark.parametrize("side", [PositionSide.LONG, PositionSide.SHORT])
    @pytest.mark.parametrize("seed", range(20))
    def test_pnl_is_sum_over_slices_of_qty_times_exit_minus_own_cost(
        self, side: PositionSide, seed: int
    ) -> None:
        rng = random.Random(seed)
        tracker = PositionTracker(fee_rate=0.0, slippage_rate=0.0)
        trade = _open(tracker, side=side)

        steps: list[tuple] = []
        realized = 0.0
        for _ in range(rng.randint(1, 6)):
            price = 100.0 * rng.uniform(0.8, 1.25)
            held = float(trade.current_size)
            if rng.random() < 0.5:
                fraction = rng.uniform(0.005, 0.05)
                tracker.apply_scale_in(fraction, price=price)
                steps.append(("scale_in", price, fraction))
            else:
                fraction = held * rng.uniform(0.1, 0.6)
                realized += tracker.apply_partial_exit(fraction, price, BALANCE)
                steps.append(("partial_exit", price, fraction))

        final_price = 100.0 * rng.uniform(0.8, 1.25)
        result = _close(tracker, final_price)

        expected = _slice_ledger_pnl(side, steps, final_price)
        assert realized + result.pnl_cash == pytest.approx(expected, abs=1e-9)
        assert result.trade.pnl == pytest.approx(expected, abs=1e-9)


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


def _scripted_strategy(script: dict[int, str], *, position_fraction: float, overrides: dict):
    from src.strategies.components import (
        EnhancedRegimeDetector,
        Signal,
        SignalDirection,
        SignalGenerator,
        Strategy,
    )
    from src.strategies.components.position_sizer import PositionSizer
    from src.strategies.components.risk_manager import RiskManager as ComponentRiskManager

    class _Signals(SignalGenerator):
        def __init__(self) -> None:
            super().__init__("scripted_signals")

        def generate_signal(self, df, index, regime=None):
            self.validate_inputs(df, index)
            action = script.get(index, "hold")
            if action == "buy":
                return Signal(SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={})
            if action == "sell":
                return Signal(SignalDirection.SELL, strength=1.0, confidence=1.0, metadata={})
            return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

        def get_confidence(self, df, index):
            return 1.0

    class _Risk(ComponentRiskManager):
        def __init__(self) -> None:
            super().__init__("fixed_fraction_risk")

        def calculate_position_size(self, signal, balance, regime=None):
            if balance <= 0 or signal.direction == SignalDirection.HOLD:
                return 0.0
            return balance * position_fraction

        def should_exit(self, position, current_data, regime=None) -> bool:
            return False

        def get_stop_loss(self, entry_price, signal, regime=None) -> float:
            return entry_price * 0.5

    class _Sizer(PositionSizer):
        def __init__(self) -> None:
            super().__init__("fixed_fraction_sizer")

        def calculate_size(self, signal, balance, risk_amount, regime=None):
            if signal.direction == SignalDirection.HOLD:
                return 0.0
            return balance * position_fraction

        def get_parameters(self):
            return {"name": self.name, "fraction": position_fraction}

    strategy = Strategy(
        name="ScaleInCostBasisStrategy",
        signal_generator=_Signals(),
        risk_manager=_Risk(),
        position_sizer=_Sizer(),
        regime_detector=EnhancedRegimeDetector(),
    )
    strategy.set_risk_overrides(overrides)
    return strategy


def _data_provider(df: pd.DataFrame):
    from src.data_providers.data_provider import DataProvider

    class _Provider(DataProvider):
        def __init__(self) -> None:
            super().__init__()

        def get_historical_data(self, symbol, timeframe, start=None, end=None):
            return df.copy()

        def get_live_data(self, symbol, timeframe, limit=100):
            raise NotImplementedError

        def update_live_data(self, symbol, timeframe):
            raise NotImplementedError

        def get_current_price(self, symbol) -> float:
            return float(df["close"].iloc[-1])

    return _Provider()


class TestEngineBalanceInvariantWithFees:
    FEE_RATE = 0.001

    def _run(self) -> tuple[dict, Backtester]:
        # Entry fills at 100 (idx 1). Scale-in +50% of original at 103 (idx 2).
        # +4% partial (50% of original) at 104 (idx 3). SELL fills at 106 (idx 4).
        closes = [100.0, 100.0, 103.0, 104.0, 106.0, 106.0]
        strategy = _scripted_strategy(
            {0: "buy", 4: "sell"},
            position_fraction=0.10,
            overrides={
                "stop_loss_pct": 0.5,
                "take_profit_pct": 0.9,
                "partial_operations": {
                    "exit_targets": [0.04],
                    "exit_sizes": [0.50],
                    "scale_in_thresholds": [0.015],
                    "scale_in_sizes": [0.50],
                    "max_scale_ins": 1,
                },
            },
        )
        backtester = Backtester(
            strategy=strategy,
            data_provider=_data_provider(_flat_ohlcv(closes)),
            initial_balance=BALANCE,
            log_to_database=False,
            fee_rate=self.FEE_RATE,
            slippage_rate=0.0,
            # Headroom above the 0.10 entry so the scale-in is not clamped.
            risk_parameters=RiskParameters(max_position_size=0.30, max_daily_risk=0.50),
        )
        return backtester.run("ETHUSDT", "1h", datetime(2024, 1, 1)), backtester

    def test_trade_pnl_and_fees_use_each_slices_own_cost(self) -> None:
        results, backtester = self._run()

        # The entry fee is deducted before entry_balance is stamped on the trade.
        entry_fee = self.FEE_RATE * 0.10 * BALANCE
        basis = BALANCE - entry_fee
        # Slice A: 0.10 x basis at 100. Slice B: 0.05 x basis at 103.
        notional_a, notional_b = 0.10 * basis, 0.05 * basis
        units_a, units_b = notional_a / 100.0, notional_b / 103.0
        total_units = units_a + units_b
        sold_share = 0.05 / 0.15  # partial exits 0.05 of 0.15 cost notional, pro rata
        partial_pnl = sold_share * (units_a * (104.0 - 100.0) + units_b * (104.0 - 103.0))
        final_pnl = (1 - sold_share) * (units_a * (106.0 - 100.0) + units_b * (106.0 - 103.0))
        fees = (
            entry_fee
            + self.FEE_RATE * notional_b  # scale-in
            + self.FEE_RATE * total_units * sold_share * 104.0  # partial exit notional
            + self.FEE_RATE * total_units * (1 - sold_share) * 106.0  # final exit notional
        )

        assert results["total_trades"] == 1
        assert backtester.trades[0].pnl == pytest.approx(partial_pnl + final_pnl)
        assert results["total_fees"] == pytest.approx(fees)

    def test_final_balance_minus_initial_equals_net_trade_pnl(self) -> None:
        results, backtester = self._run()

        net_trade_pnl = sum(t.pnl for t in backtester.trades) - results["total_fees"]
        assert results["final_balance"] - BALANCE == pytest.approx(net_trade_pnl)


class TestTriggersStayOnEntryPrice:
    def test_partial_exit_target_is_still_measured_from_the_first_fill(self) -> None:
        policy = PartialExitPolicy(
            exit_targets=[0.04],
            exit_sizes=[0.50],
            scale_in_thresholds=[0.015],
            scale_in_sizes=[0.50],
            max_scale_ins=1,
        )
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
            max_position_size=0.30,
        )
        trade = _open(tracker)
        df = pd.DataFrame(
            {"open": [103.0], "high": [103.0], "low": [103.0], "close": [103.0], "volume": [1.0]},
            index=pd.DatetimeIndex([datetime(2024, 1, 2, tzinfo=UTC)]),
        )

        handler.check_partial_operations(103.0, df, 0, balance=BALANCE)
        assert trade.scale_ins_taken == 1
        assert trade.partial_exits_taken == 0

        # +4% from the first fill (104), not from the blended cost (~104.7).
        handler.check_partial_operations(104.0, df, 0, balance=BALANCE)

        assert trade.partial_exits_taken == 1
        assert trade.entry_price == 100.0
