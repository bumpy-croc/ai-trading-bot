"""Unit tests for StrategyExitChecker."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.engines.shared.strategy_exit_checker import (
    StrategyExitChecker,
    StrategyExitResult,
)
from src.strategies.components import Signal, SignalDirection


def _make_position(side: str = "long") -> SimpleNamespace:
    """Create a mock position with standard test values."""
    return SimpleNamespace(
        symbol="BTCUSDT",
        side=side,
        entry_price=50000.0,
        entry_balance=10000.0,
        current_size=0.2,
        entry_time=datetime.now(UTC),
    )


def _make_signal(
    direction: SignalDirection = SignalDirection.BUY,
    confidence: float = 0.7,
    strength: float = 0.5,
) -> Signal:
    """Create a Signal with common test defaults."""
    return Signal(
        direction=direction,
        confidence=confidence,
        strength=strength,
        metadata={},
    )


def _make_decision(
    direction: SignalDirection = SignalDirection.BUY,
    metadata: dict | None = None,
) -> SimpleNamespace:
    """Create a mock TradingDecision for testing."""
    return SimpleNamespace(
        signal=_make_signal(direction=direction),
        position_size=1000.0,
        regime=None,
        metadata=metadata,
    )


class TestStrategyExitResult:
    """Test StrategyExitResult dataclass."""

    def test_exit_result_default_hold(self):
        """Test default exit result is hold."""
        result = StrategyExitResult(should_exit=False)

        assert result.should_exit is False
        assert result.exit_reason == "Hold"

    def test_exit_result_with_reason(self):
        """Test exit result with custom reason."""
        result = StrategyExitResult(
            should_exit=True, exit_reason="Stop loss hit"
        )

        assert result.should_exit is True
        assert result.exit_reason == "Stop loss hit"


class TestSignalReversalChecking:
    """Test signal reversal detection in exit checker."""

    def test_check_exit_with_long_position_sell_signal(self):
        """Test long position exits on SELL signal."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=_make_decision(SignalDirection.SELL),
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_check_exit_with_short_position_buy_signal(self):
        """Test short position exits on BUY signal."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("short"),
            current_price=49000.0,
            runtime_decision=_make_decision(SignalDirection.BUY),
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_check_exit_no_reversal_same_direction(self):
        """Test no exit when signal direction matches position side."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=_make_decision(SignalDirection.BUY),
            component_strategy=None,
        )

        assert result.should_exit is False


class TestIgnoreSignalReversal:
    """Test ignore_signal_reversal metadata flag."""

    def test_ignore_signal_reversal_prevents_exit(self):
        """Test ignore_signal_reversal metadata prevents signal reversal exits."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=_make_decision(
                SignalDirection.SELL, metadata={"ignore_signal_reversal": True}
            ),
            component_strategy=None,
        )

        assert result.should_exit is False

    def test_ignore_signal_reversal_false_allows_exit(self):
        """Test ignore_signal_reversal=False allows normal signal reversal exits."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=_make_decision(
                SignalDirection.SELL, metadata={"ignore_signal_reversal": False}
            ),
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_ignore_signal_reversal_with_short_position(self):
        """Test ignore_signal_reversal works for short positions too."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("short"),
            current_price=49000.0,
            runtime_decision=_make_decision(
                SignalDirection.BUY, metadata={"ignore_signal_reversal": True}
            ),
            component_strategy=None,
        )

        assert result.should_exit is False

    def test_empty_metadata_allows_signal_reversal_exit(self):
        """Test empty metadata dict allows normal signal reversal exits."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=_make_decision(SignalDirection.SELL, metadata={}),
            component_strategy=None,
        )

        assert result.should_exit is True

    def test_none_metadata_allows_signal_reversal_exit(self):
        """Test None metadata allows normal signal reversal exits."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=_make_decision(SignalDirection.SELL, metadata=None),
            component_strategy=None,
        )

        assert result.should_exit is True


class TestExitCheckerEdgeCases:
    """Test edge cases and error handling in exit checker."""

    def test_check_exit_with_none_position(self):
        """Test checker handles None position gracefully."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=None,
            current_price=51000.0,
            runtime_decision=_make_decision(SignalDirection.SELL),
            component_strategy=None,
        )

        assert result.should_exit is False

    def test_check_exit_with_none_decision(self):
        """Test checker handles None decision gracefully."""
        checker = StrategyExitChecker()

        result = checker.check_exit(
            position=_make_position("long"),
            current_price=51000.0,
            runtime_decision=None,
            component_strategy=None,
        )

        assert result.should_exit is False

    def test_check_exit_with_position_missing_side(self):
        """Test checker defaults to 'long' when position has no side attribute."""
        checker = StrategyExitChecker()

        # Position without side attribute -- normalize_side(None) returns "long"
        position = SimpleNamespace(
            symbol="BTCUSDT",
            entry_price=50000.0,
            entry_balance=10000.0,
            current_size=0.2,
            entry_time=datetime.now(UTC),
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=_make_decision(SignalDirection.SELL),
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"
