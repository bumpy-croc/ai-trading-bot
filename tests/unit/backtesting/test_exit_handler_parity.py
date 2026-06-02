"""Parity tests between backtest and live engines for exit semantics.

These tests cover two divergences that previously made backtest results drift
from live behaviour:

1. Time-exit reason: the backtest engine used to hardcode ``"Time limit"`` and
   discard the policy-returned reason. Live preserves the policy reason
   (``"Max holding period"``, ``"Weekend flat"``, etc.) and falls back to
   ``"Time exit"``. Backtest now matches.
2. Margin interest accrual: the live engine deducts borrow interest from
   realised PnL via ``MarginInterestTracker``. Backtest now models the same
   carry cost via the ``annual_margin_interest_rate`` parameter so margin-mode
   strategies don't silently overstate returns.
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
from src.position_management.time_exits import TimeExitPolicy


def _make_handler(
    position_tracker: PositionTracker,
    *,
    time_exit_policy: TimeExitPolicy | None = None,
    annual_margin_interest_rate: float = 0.0,
) -> ExitHandler:
    return ExitHandler(
        execution_engine=Mock(),
        position_tracker=position_tracker,
        risk_manager=Mock(),
        execution_model=ExecutionModel(default_fill_policy()),
        time_exit_policy=time_exit_policy,
        use_high_low_for_stops=True,
        annual_margin_interest_rate=annual_margin_interest_rate,
    )


@pytest.mark.fast
class TestTimeExitReasonParity:
    """Backtest must surface the same exit_reason strings as live."""

    def test_max_holding_period_reason_propagates(self) -> None:
        """Backtest preserves ``"Max holding period"`` from the policy."""
        entry_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        # 4-hour candle 5 hours after entry — past the 4h max-holding window.
        candle_time = entry_time + timedelta(hours=5)

        tracker = PositionTracker()
        tracker.open_position(
            ActiveTrade(
                symbol="TEST",
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_time=entry_time,
                size=0.1,
            )
        )

        handler = _make_handler(
            tracker,
            time_exit_policy=TimeExitPolicy(max_holding_hours=4),
        )

        candle = pd.Series(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            name=candle_time,
        )

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=candle,
            current_price=100.5,
            symbol="TEST",
        )

        assert result.is_time_limit is True
        assert result.exit_reason == "Max holding period"

    def test_weekend_flat_reason_propagates(self) -> None:
        """Backtest preserves ``"Weekend flat"`` from the policy."""
        # 2024-01-06 is a Saturday — weekend_flat triggers.
        candle_time = datetime(2024, 1, 6, 12, 0, tzinfo=UTC)
        entry_time = candle_time - timedelta(hours=1)

        tracker = PositionTracker()
        tracker.open_position(
            ActiveTrade(
                symbol="TEST",
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_time=entry_time,
                size=0.1,
            )
        )

        handler = _make_handler(
            tracker,
            time_exit_policy=TimeExitPolicy(weekend_flat=True),
        )

        candle = pd.Series(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
            name=candle_time,
        )

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=candle,
            current_price=100.0,
            symbol="TEST",
        )

        assert result.is_time_limit is True
        assert result.exit_reason == "Weekend flat"

    def test_default_fallback_matches_live(self, monkeypatch) -> None:
        """When the policy reports a hit but returns no reason, fall back
        to ``"Time exit"`` — matching live's ``time_reason or "Time exit"``.

        Drives the exit handler with a stubbed policy that returns
        ``(True, None)`` and asserts the resolved exit_reason. This is a
        behavioural test (not a source-grep) so future edits to the
        time-exit branch can't regress silently.
        """
        entry_time = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
        candle_time = entry_time + timedelta(hours=5)

        tracker = PositionTracker()
        tracker.open_position(
            ActiveTrade(
                symbol="TEST",
                side=PositionSide.LONG,
                entry_price=100.0,
                entry_time=entry_time,
                size=0.1,
            )
        )

        policy = TimeExitPolicy(max_holding_hours=4)
        # Force the policy to report a hit but with no reason — the exact
        # contract live falls back to ``"Time exit"`` for.
        monkeypatch.setattr(policy, "check_time_exit_conditions", lambda *_a, **_k: (True, None))

        handler = _make_handler(tracker, time_exit_policy=policy)
        candle = pd.Series(
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1000.0},
            name=candle_time,
        )

        result = handler.check_exit_conditions(
            runtime_decision=None,
            candle=candle,
            current_price=100.5,
            symbol="TEST",
        )

        assert result.is_time_limit is True
        assert result.exit_reason == "Time exit"


@pytest.mark.fast
class TestMarginInterestParity:
    """Backtest must charge margin/borrow interest when the rate is set."""

    def _make_position_with_balance(self, entry_time: datetime) -> PositionTracker:
        """Build a tracker with a position whose entry_balance is populated."""
        tracker = PositionTracker()
        trade = ActiveTrade(
            symbol="TEST",
            side=PositionSide.LONG,
            entry_price=100.0,
            entry_time=entry_time,
            size=0.5,
            current_size=0.5,
            original_size=0.5,
            entry_balance=10_000.0,
        )
        tracker.open_position(trade)
        return tracker

    def test_zero_rate_default_charges_no_interest(self) -> None:
        """The default rate (0.0) preserves spot-mode behavior — no carry cost."""
        entry_time = datetime(2024, 1, 1, tzinfo=UTC)
        tracker = self._make_position_with_balance(entry_time)
        handler = _make_handler(tracker, annual_margin_interest_rate=0.0)

        # 24 hours later, no price move — gross PnL is 0, no interest.
        interest = handler._calculate_margin_interest(
            position_notional=5_000.0,
            entry_time=entry_time,
            exit_time=entry_time + timedelta(hours=24),
        )
        assert interest == 0.0

    def test_positive_rate_charges_interest_pro_rata(self) -> None:
        """Interest scales linearly with holding period and notional."""
        entry_time = datetime(2024, 1, 1, tzinfo=UTC)
        tracker = self._make_position_with_balance(entry_time)
        handler = _make_handler(tracker, annual_margin_interest_rate=0.10)

        # 5,000 notional × 10% APR × (24h / 8760h) = ~1.3699 USDT
        interest = handler._calculate_margin_interest(
            position_notional=5_000.0,
            entry_time=entry_time,
            exit_time=entry_time + timedelta(hours=24),
        )
        expected = 5_000.0 * 0.10 * (24.0 / (365.0 * 24.0))
        assert interest == pytest.approx(expected, rel=1e-9)

    def test_negative_holding_returns_zero(self) -> None:
        """Defensive: clock skew or bad inputs must not produce negative interest."""
        entry_time = datetime(2024, 1, 1, tzinfo=UTC)
        tracker = self._make_position_with_balance(entry_time)
        handler = _make_handler(tracker, annual_margin_interest_rate=0.10)

        interest = handler._calculate_margin_interest(
            position_notional=5_000.0,
            entry_time=entry_time,
            exit_time=entry_time - timedelta(hours=1),
        )
        assert interest == 0.0

    def test_init_rejects_negative_rate(self) -> None:
        """Negative or non-finite rates are rejected at construction."""
        tracker = PositionTracker()
        with pytest.raises(ValueError, match="non-negative"):
            _make_handler(tracker, annual_margin_interest_rate=-0.01)

    def test_init_rejects_nan_rate(self) -> None:
        """NaN rate is rejected at construction (defense against bad config)."""
        tracker = PositionTracker()
        with pytest.raises(ValueError, match="non-negative"):
            _make_handler(tracker, annual_margin_interest_rate=float("nan"))


@pytest.mark.fast
class TestBacktesterMarginRateWiring:
    """The Backtester must plumb annual_margin_interest_rate to its ExitHandler."""

    def test_backtester_default_rate_is_zero(self) -> None:
        """Default preserves existing spot-mode behavior — no surprise carry costs."""
        from src.engines.backtest.engine import Backtester

        strategy = Mock()
        strategy.name = "test"
        strategy.get_risk_overrides.return_value = None
        strategy.calculate_indicators = Mock(return_value=None)
        data_provider = Mock()

        bt = Backtester(strategy=strategy, data_provider=data_provider, initial_balance=10_000.0)
        assert bt.annual_margin_interest_rate == 0.0
        assert bt.exit_handler.annual_margin_interest_rate == 0.0

    def test_backtester_propagates_rate_to_exit_handler(self) -> None:
        from src.engines.backtest.engine import Backtester

        strategy = Mock()
        strategy.name = "test"
        strategy.get_risk_overrides.return_value = None
        strategy.calculate_indicators = Mock(return_value=None)
        data_provider = Mock()

        bt = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            initial_balance=10_000.0,
            annual_margin_interest_rate=0.05,
        )
        assert bt.annual_margin_interest_rate == 0.05
        assert bt.exit_handler.annual_margin_interest_rate == 0.05

    def test_backtester_rejects_invalid_rate(self) -> None:
        from src.engines.backtest.engine import Backtester

        strategy = Mock()
        strategy.name = "test"
        data_provider = Mock()

        with pytest.raises(ValueError, match="non-negative"):
            Backtester(
                strategy=strategy,
                data_provider=data_provider,
                initial_balance=10_000.0,
                annual_margin_interest_rate=-0.01,
            )
