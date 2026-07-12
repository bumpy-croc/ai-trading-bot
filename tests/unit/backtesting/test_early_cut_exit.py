"""Backtest ExitHandler integration for the MFE early-cut policy (#971).

Uses the same hand-computable fixture as
tests/unit/position_management/test_early_cut.py: long entry 100.0 at
2024-01-01 00:00 UTC, 1h bars, threshold 2% within 3h, window MFE 1.9%.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest

from src.engines.backtest.execution.exit_handler import ExitHandler
from src.engines.backtest.execution.position_tracker import PositionTracker
from src.engines.backtest.models import ActiveTrade
from src.engines.shared.execution.execution_model import ExecutionModel
from src.engines.shared.execution.fill_policy import default_fill_policy
from src.engines.shared.models import PositionSide
from src.position_management.early_cut import EarlyCutPolicy

pytestmark = [pytest.mark.unit, pytest.mark.fast]

ENTRY_TIME = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
ENTRY_PRICE = 100.0


def _make_df(highs: list[float], lows: list[float]) -> pd.DataFrame:
    index = pd.date_range(start=ENTRY_TIME.replace(tzinfo=None), periods=len(highs), freq="1h")
    closes = [(h + lo) / 2 for h, lo in zip(highs, lows, strict=False)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


def _make_handler(
    tracker: PositionTracker,
    *,
    early_cut_policy: EarlyCutPolicy | None = None,
) -> ExitHandler:
    return ExitHandler(
        execution_engine=Mock(),
        position_tracker=tracker,
        risk_manager=Mock(),
        execution_model=ExecutionModel(default_fill_policy()),
        early_cut_policy=early_cut_policy,
        use_high_low_for_stops=True,
    )


def _open_long(tracker: PositionTracker, stop_loss: float | None = None) -> None:
    tracker.open_position(
        ActiveTrade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            size=0.1,
            stop_loss=stop_loss,
        )
    )


def _candle(df: pd.DataFrame, i: int) -> pd.Series:
    return df.iloc[i]


class TestBacktestEarlyCut:
    def test_cuts_at_window_end_when_mfe_below_threshold(self) -> None:
        df = _make_df(
            highs=[100.8, 101.5, 101.9, 100.9, 100.9],
            lows=[99.5, 99.8, 100.1, 100.0, 100.0],
        )
        tracker = PositionTracker()
        _open_long(tracker)
        handler = _make_handler(
            tracker,
            early_cut_policy=EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3),
        )

        # Bars before the window elapses: hold.
        for i in (1, 2):
            result = handler.check_exit_conditions(
                runtime_decision=None,
                candle=_candle(df, i),
                current_price=float(df["close"].iloc[i]),
                symbol="TEST",
                df=df,
            )
            assert result.should_exit is False, f"unexpected exit at bar {i}"

        # Bar at entry+3h: window MFE 1.9% < 2% => cut at this bar's close.
        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=_candle(df, 3),
            current_price=float(df["close"].iloc[3]),
            symbol="TEST",
            df=df,
        )
        assert result.should_exit is True
        assert result.is_early_cut is True
        assert result.exit_reason.startswith("Early cut")
        assert result.exit_price == float(df["close"].iloc[3])
        # Observability (#976 review): the window MFE that triggered the cut
        # is surfaced on the result. Hand-computed: 1.9%.
        assert result.early_cut_window_mfe_pct == pytest.approx(0.019)

    def test_no_cut_when_mfe_reached_threshold(self) -> None:
        df = _make_df(
            highs=[100.8, 101.5, 102.1, 100.9],
            lows=[99.5, 99.8, 100.1, 100.0],
        )
        tracker = PositionTracker()
        _open_long(tracker)
        handler = _make_handler(
            tracker,
            early_cut_policy=EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3),
        )

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=_candle(df, 3),
            current_price=float(df["close"].iloc[3]),
            symbol="TEST",
            df=df,
        )
        assert result.should_exit is False

    def test_stop_loss_takes_priority_over_early_cut(self) -> None:
        df = _make_df(
            highs=[100.8, 101.0, 101.0, 100.9],
            lows=[99.5, 99.8, 99.9, 94.0],  # bar 3 crashes through the stop
        )
        tracker = PositionTracker()
        _open_long(tracker, stop_loss=95.0)
        handler = _make_handler(
            tracker,
            early_cut_policy=EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3),
        )

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=_candle(df, 3),
            current_price=float(df["close"].iloc[3]),
            symbol="TEST",
            df=df,
        )
        assert result.should_exit is True
        assert result.exit_reason == "Stop loss"
        assert result.is_stop_loss is True
        assert result.is_early_cut is False

    def test_no_policy_means_no_behavior_change(self) -> None:
        df = _make_df(
            highs=[100.8, 101.0, 101.0, 100.9],
            lows=[99.5, 99.8, 99.9, 100.0],
        )
        tracker = PositionTracker()
        _open_long(tracker)
        handler = _make_handler(tracker, early_cut_policy=None)

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=_candle(df, 3),
            current_price=float(df["close"].iloc[3]),
            symbol="TEST",
            df=df,
        )
        assert result.should_exit is False

    def test_policy_without_df_never_cuts(self) -> None:
        """Defensive: a wired policy with no candle history must hold."""
        df = _make_df(
            highs=[100.8, 101.0, 101.0, 100.9],
            lows=[99.5, 99.8, 99.9, 100.0],
        )
        tracker = PositionTracker()
        _open_long(tracker)
        handler = _make_handler(
            tracker,
            early_cut_policy=EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3),
        )

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=_candle(df, 3),
            current_price=float(df["close"].iloc[3]),
            symbol="TEST",
        )
        assert result.should_exit is False


class TestBacktesterWiring:
    def test_backtester_builds_early_cut_policy_from_risk_parameters(self) -> None:
        """The engine must hydrate the policy from RiskParameters.early_cut
        (the ExperimentRunner channel) without any strategy involvement."""
        from src.engines.backtest.engine import Backtester
        from src.risk.risk_manager import RiskParameters

        strategy = Mock()
        strategy.get_risk_overrides = Mock(return_value={})
        backtester = Backtester(
            strategy=strategy,
            data_provider=Mock(),
            risk_parameters=RiskParameters(
                early_cut={"mfe_threshold_pct": 0.015, "evaluation_window_hours": 18}
            ),
            initial_balance=1000.0,
            log_to_database=False,
        )

        assert isinstance(backtester.early_cut_policy, EarlyCutPolicy)
        assert backtester.early_cut_policy.mfe_threshold_pct == 0.015
        assert backtester.exit_handler.early_cut_policy is backtester.early_cut_policy

    def test_backtester_default_has_no_early_cut_policy(self) -> None:
        from src.engines.backtest.engine import Backtester
        from src.risk.risk_manager import RiskParameters

        strategy = Mock()
        strategy.get_risk_overrides = Mock(return_value={})
        backtester = Backtester(
            strategy=strategy,
            data_provider=Mock(),
            risk_parameters=RiskParameters(),
            initial_balance=1000.0,
            log_to_database=False,
        )

        assert backtester.early_cut_policy is None
        assert backtester.exit_handler.early_cut_policy is None


# ---------------------------------------------------------------------------
# End-to-end: full Backtester.run() with a scripted one-entry strategy
# ---------------------------------------------------------------------------


def _build_scripted_strategy():
    """Buy once on bar 1, then hold forever (no signal exits)."""
    from src.strategies.components import (
        EnhancedRegimeDetector,
        PositionSizer,
        Signal,
        SignalDirection,
        SignalGenerator,
        Strategy,
    )
    from src.strategies.components import (
        RiskManager as ComponentRiskManager,
    )

    class _OneEntrySignalGenerator(SignalGenerator):
        def __init__(self):
            super().__init__("one_entry_generator")

        def generate_signal(self, df: pd.DataFrame, index: int, regime=None) -> Signal:
            self.validate_inputs(df, index)
            if index == 1:
                return Signal(SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={})
            return Signal(SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

        def get_confidence(self, df: pd.DataFrame, index: int) -> float:
            self.validate_inputs(df, index)
            return 1.0

    class _FlatRisk(ComponentRiskManager):
        def __init__(self):
            super().__init__("flat_risk")

        def calculate_position_size(self, signal, balance, regime=None, **context):
            if balance <= 0 or signal.direction == SignalDirection.HOLD:
                return 0.0
            return balance * 0.2

        def should_exit(self, position, current_data, regime=None, **context):
            return False

        def get_stop_loss(self, entry_price, signal, regime=None, **context):
            return entry_price * 0.9 if signal.direction == SignalDirection.BUY else entry_price

        def get_parameters(self):
            return {"risk": 0.2}

    class _PassSizer(PositionSizer):
        def __init__(self):
            super().__init__("pass_sizer")

        def calculate_size(self, signal, balance, risk_amount, regime=None, **context):
            self.validate_inputs(balance, risk_amount)
            if signal.direction == SignalDirection.HOLD:
                return 0.0
            return self.apply_bounds_checking(
                risk_amount, balance, min_fraction=0.0, max_fraction=1.0
            )

        def get_parameters(self):
            return {"mode": "pass_through"}

    strategy = Strategy(
        name="OneEntryStrategy",
        signal_generator=_OneEntrySignalGenerator(),
        risk_manager=_FlatRisk(),
        position_sizer=_PassSizer(),
        regime_detector=EnhancedRegimeDetector(),
    )
    # Hold through signal flips (HyperGrowth's live setting) so nothing but
    # the engine-level exits can close the position.
    strategy._extra_metadata = {"ignore_signal_reversal": True}
    return strategy


class _FlatDataProvider:
    """Minimal provider serving a fixed OHLCV frame."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_historical_data(self, symbol, timeframe, start=None, end=None):
        return self._df.copy()

    def get_current_price(self, symbol):
        return float(self._df["close"].iloc[-1])


class TestEndToEndEarlyCut:
    """Full Backtester.run() — hand-computable single-trade scenario.

    30 flat bars at 100.0 (1h). Entry on bar 1. With a 2%-in-3h early cut
    the MFE window (bars strictly inside (entry, entry+3h)) never leaves
    0%, so the trade must be cut 3 hours after entry. Without the policy
    the trade survives to the end of data.
    """

    def _run(self, risk_parameters):
        from src.engines.backtest.engine import Backtester

        index = pd.date_range(start="2024-01-01", periods=30, freq="1h")
        df = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 1000.0,
            },
            index=index,
        )
        backtester = Backtester(
            strategy=_build_scripted_strategy(),
            data_provider=_FlatDataProvider(df),
            risk_parameters=risk_parameters,
            initial_balance=1000.0,
            log_to_database=False,
        )
        results = backtester.run(
            symbol="TESTUSDT",
            timeframe="1h",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 2, 6, tzinfo=UTC),
        )
        return backtester, results

    def test_early_cut_closes_trade_three_hours_after_entry(self) -> None:
        from src.risk.risk_manager import RiskParameters

        backtester, results = self._run(
            RiskParameters(early_cut={"mfe_threshold_pct": 0.02, "evaluation_window_hours": 3})
        )

        assert results["total_trades"] == 1
        trade = backtester.trades[0]
        assert trade.exit_reason.startswith("Early cut")
        held = trade.exit_time - trade.entry_time
        assert held == timedelta(hours=3)
        # Observability (#976 review): exam output carries the window MFE
        # that triggered the cut (flat series => exactly 0.0).
        assert trade.metadata["early_cut_window_mfe_pct"] == 0.0

    def test_without_policy_trade_survives(self) -> None:
        """Control arm: same data, no early_cut config => the position is
        never exited (it is still open at backtest end)."""
        from src.risk.risk_manager import RiskParameters

        backtester, results = self._run(RiskParameters())

        assert results["total_trades"] == 0
        assert backtester.position_tracker.current_trade is not None


class TestRunTimeframeValidation:
    """Review #976 F1/F2: a mis-sized window must fail fast at run() time,
    not silently become a cut-everything (old) or never-cut (new) policy."""

    def _backtester(self, window_hours: float):
        from src.engines.backtest.engine import Backtester
        from src.risk.risk_manager import RiskParameters

        index = pd.date_range(start="2024-01-01", periods=10, freq="1h")
        df = pd.DataFrame(
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0},
            index=index,
        )
        return Backtester(
            strategy=_build_scripted_strategy(),
            data_provider=_FlatDataProvider(df),
            risk_parameters=RiskParameters(
                early_cut={"mfe_threshold_pct": 0.02, "evaluation_window_hours": window_hours}
            ),
            initial_balance=1000.0,
            log_to_database=False,
        )

    def test_window_equal_to_bar_interval_rejected_at_run(self) -> None:
        backtester = self._backtester(window_hours=1)
        with pytest.raises(ValueError, match="evaluation_window_hours"):
            backtester.run(
                symbol="TESTUSDT",
                timeframe="1h",
                start=datetime(2024, 1, 1, tzinfo=UTC),
            )

    def test_non_multiple_window_rejected_at_run(self) -> None:
        backtester = self._backtester(window_hours=1.5)
        with pytest.raises(ValueError, match="whole multiple"):
            backtester.run(
                symbol="TESTUSDT",
                timeframe="1h",
                start=datetime(2024, 1, 1, tzinfo=UTC),
            )
