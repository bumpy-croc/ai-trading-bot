"""Unit tests for StrategyExitChecker."""

from datetime import UTC, datetime

import pytest

from src.engines.shared.strategy_exit_checker import (
    StrategyExitChecker,
    StrategyExitResult,
)
from src.strategies.components import Signal, SignalDirection


# Mock runtime decision class
class MockRuntimeDecision:
    """Mock TradingDecision for testing."""

    def __init__(
        self, signal: Signal, position_size: float, regime=None, metadata=None
    ):
        self.signal = signal
        self.position_size = position_size
        self.regime = regime
        self.metadata = metadata


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

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with SELL signal (reversal)
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_check_exit_with_short_position_buy_signal(self):
        """Test short position exits on BUY signal."""
        checker = StrategyExitChecker()

        # Mock short position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "short"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with BUY signal (reversal)
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.BUY,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
        )

        result = checker.check_exit(
            position=position,
            current_price=49000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_check_exit_no_reversal_same_direction(self):
        """Test no exit when signal direction matches position side."""
        checker = StrategyExitChecker()

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with BUY signal (same direction)
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.BUY,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        assert result.should_exit is False


class TestIgnoreSignalReversal:
    """Test ignore_signal_reversal metadata flag."""

    def test_ignore_signal_reversal_prevents_exit(self):
        """Test ignore_signal_reversal metadata prevents signal reversal exits."""
        checker = StrategyExitChecker()

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with SELL signal (reversal) but ignore flag
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
            metadata={"ignore_signal_reversal": True},
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        # Should NOT exit despite signal reversal
        assert result.should_exit is False

    def test_ignore_signal_reversal_false_allows_exit(self):
        """Test ignore_signal_reversal=False allows normal signal reversal exits."""
        checker = StrategyExitChecker()

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with SELL signal and ignore=False
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
            metadata={"ignore_signal_reversal": False},
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        # Should exit when ignore_signal_reversal is False
        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_ignore_signal_reversal_with_short_position(self):
        """Test ignore_signal_reversal works for short positions too."""
        checker = StrategyExitChecker()

        # Mock short position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "short"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with BUY signal (reversal) but ignore flag
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.BUY,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
            metadata={"ignore_signal_reversal": True},
        )

        result = checker.check_exit(
            position=position,
            current_price=49000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        # Should NOT exit despite signal reversal
        assert result.should_exit is False

    def test_empty_metadata_allows_signal_reversal_exit(self):
        """Test empty metadata dict allows normal signal reversal exits."""
        checker = StrategyExitChecker()

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with SELL signal and empty metadata
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
            metadata={},  # Empty dict, no ignore flag
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        # Should exit normally when metadata doesn't have ignore flag
        assert result.should_exit is True

    def test_none_metadata_allows_signal_reversal_exit(self):
        """Test None metadata allows normal signal reversal exits."""
        checker = StrategyExitChecker()

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        # Trading decision with SELL signal and None metadata
        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
            metadata=None,  # No metadata
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        # Should exit normally when metadata is None
        assert result.should_exit is True


class TestExitCheckerEdgeCases:
    """Test edge cases and error handling in exit checker."""

    def test_check_exit_with_none_position(self):
        """Test checker handles None position gracefully."""
        checker = StrategyExitChecker()

        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
        )

        result = checker.check_exit(
            position=None,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        assert result.should_exit is False

    def test_check_exit_with_none_decision(self):
        """Test checker handles None decision gracefully."""
        checker = StrategyExitChecker()

        # Mock long position
        class MockPosition:
            symbol = "BTCUSDT"
            side = "long"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=None,
            component_strategy=None,
        )

        assert result.should_exit is False

    def test_check_exit_with_position_missing_side(self):
        """Test checker defaults to 'long' when position has no side attribute."""
        checker = StrategyExitChecker()

        # Position without side attribute
        class MockPosition:
            symbol = "BTCUSDT"
            entry_price = 50000.0
            entry_balance = 10000.0
            current_size = 0.2
            entry_time = datetime.now(UTC)

        position = MockPosition()

        decision = MockRuntimeDecision(
            signal=Signal(
                direction=SignalDirection.SELL,
                confidence=0.7,
                strength=0.5,
                metadata={},
            ),
            position_size=1000.0,
            regime=None,
        )

        result = checker.check_exit(
            position=position,
            current_price=51000.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        # normalize_side(None) returns "long" as default, so SELL signal triggers exit
        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"
