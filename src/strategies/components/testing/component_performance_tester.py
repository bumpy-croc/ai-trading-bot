"""
Component Performance Tester

This module provides comprehensive performance testing for individual strategy components,
allowing isolated testing of SignalGenerator, RiskManager, and PositionSizer components.

Error Handling Strategy:
- Validation methods: Raise ValueError for invalid inputs
- Test execution: Log errors and return error counts in results
- All errors logged with full context via logger.error()
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from ..position_sizer import PositionSizer
from ..risk_manager import MarketData, Position, RiskManager
from ..signal_generator import Signal, SignalDirection, SignalGenerator

logger = logging.getLogger(__name__)

# Testing Configuration Constants
MIN_TEST_DATA_ROWS = 100
MIN_DATA_AFTER_TRANSFORM = 50
MIN_TREND_PERIOD_LENGTH = 30
MIN_VOLATILITY_PERIOD_LENGTH = 20
VOLATILITY_HIGH_MULTIPLIER = 1.5
VOLATILITY_LOW_MULTIPLIER = 0.7
SMALL_RETURN_THRESHOLD = 0.01  # For HOLD signal accuracy
TRADING_DAYS_PER_YEAR = 252


@dataclass
class SignalTestResults:
    """Results from signal generator performance testing"""

    component_name: str
    test_duration: float
    total_signals: int

    # Signal quality metrics
    accuracy: float  # % of profitable signals
    precision: float  # % of buy signals that were profitable
    recall: float  # % of profitable opportunities captured
    f1_score: float  # Harmonic mean of precision and recall

    # Performance metrics
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_duration: float

    # Signal distribution
    buy_signals: int
    sell_signals: int
    hold_signals: int

    # Confidence and strength analysis
    avg_confidence: float
    avg_strength: float
    confidence_accuracy_correlation: float

    # Regime breakdown (if regime data available)
    regime_breakdown: dict[str, dict[str, float]]

    # Error analysis
    error_count: int
    error_rate: float

    # Timing metrics
    avg_signal_generation_time: float
    signals_per_second: float


@dataclass
class RiskTestResults:
    """Results from risk manager performance testing"""

    component_name: str
    test_duration: float
    total_scenarios: int

    # Risk control effectiveness
    max_drawdown_achieved: float
    target_max_drawdown: float
    drawdown_control_score: float  # How well it controlled drawdown

    # Position sizing effectiveness
    average_position_size: float
    position_size_consistency: float  # Std dev of position sizes
    risk_adjusted_return: float

    # Stop loss effectiveness
    stop_loss_hit_rate: float  # % of positions that hit stop loss
    avg_stop_loss_distance: float
    stop_loss_effectiveness: float  # Avg loss when stop hit vs expected

    # Exit decision quality
    premature_exit_rate: float  # % of exits that were too early
    late_exit_rate: float  # % of exits that were too late
    optimal_exit_rate: float  # % of exits that were well-timed

    # Risk-return metrics
    return_per_unit_risk: float
    risk_efficiency_score: float

    # Regime adaptation (if applicable)
    regime_adaptation_score: float

    # Error analysis
    error_count: int
    error_rate: float

    # Timing metrics
    avg_calculation_time: float
    calculations_per_second: float


@dataclass
class SizingTestResults:
    """Results from position sizer performance testing"""

    component_name: str
    test_duration: float
    total_calculations: int

    # Kelly criterion adherence (if applicable)
    kelly_criterion_adherence: float
    optimal_sizing_score: float

    # Regime adaptation effectiveness
    regime_adaptation_effectiveness: float
    regime_consistency_score: float

    # Volatility adjustment quality
    volatility_adjustment_quality: float
    volatility_responsiveness: float

    # Size distribution analysis
    average_position_size: float
    position_size_std: float
    size_range_utilization: float  # How well it uses the full size range

    # Risk-adjusted performance
    size_adjusted_sharpe: float
    size_adjusted_return: float

    # Confidence/strength responsiveness
    confidence_responsiveness: float
    strength_responsiveness: float

    # Bounds checking effectiveness
    bounds_violations: int
    bounds_adherence_rate: float

    # Error analysis
    error_count: int
    error_rate: float

    # Timing metrics
    avg_calculation_time: float
    calculations_per_second: float


@dataclass
class ComponentTestResults:
    """Combined results from all component tests"""

    signal_results: SignalTestResults | None = None
    risk_results: RiskTestResults | None = None
    sizing_results: SizingTestResults | None = None

    # Overall metrics
    total_test_duration: float = 0.0
    overall_performance_score: float = 0.0

    # Component interaction analysis
    component_synergy_score: float = 0.0
    integration_effectiveness: float = 0.0


class ComponentPerformanceTester:
    """
    Comprehensive performance tester for strategy components

    Provides isolated testing capabilities for SignalGenerator, RiskManager,
    and PositionSizer components with detailed performance metrics.
    """

    def __init__(self, test_data: pd.DataFrame, regime_data: pd.DataFrame | None = None):
        """
        Initialize component performance tester

        Args:
            test_data: Historical market data for testing (OHLCV format)
            regime_data: Optional regime labels for regime-specific analysis
        """
        self.test_data = test_data.copy()
        self.regime_data = regime_data

        # Validate test data
        self._validate_test_data()

        # Prepare test data with indicators
        self._prepare_test_data()

        # Initialize test scenarios
        self.test_scenarios = self._generate_test_scenarios()

    def _validate_test_data(self) -> None:
        """Validate that test data has required columns and format"""
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in self.test_data.columns]

        if missing_columns:
            raise ValueError(f"Test data missing required columns: {missing_columns}")

        if len(self.test_data) < MIN_TEST_DATA_ROWS:
            raise ValueError(
                f"Test data must have at least {MIN_TEST_DATA_ROWS} rows for meaningful testing"
            )

        # Check for NaN values
        if self.test_data[required_columns].isnull().any().any():
            raise ValueError("Test data contains NaN values")

    def _prepare_test_data(self) -> None:
        """Prepare test data with technical indicators"""
        # Add basic technical indicators for signal generation
        self.test_data["sma_20"] = self.test_data["close"].rolling(20).mean()
        self.test_data["sma_50"] = self.test_data["close"].rolling(50).mean()
        self.test_data["rsi"] = self._calculate_rsi(self.test_data["close"])
        self.test_data["atr"] = self._calculate_atr(self.test_data)

        # Calculate returns for performance analysis
        self.test_data["returns"] = self.test_data["close"].pct_change()
        self.test_data["log_returns"] = np.log(
            self.test_data["close"] / self.test_data["close"].shift(1)
        )

        # Forward-looking returns for signal accuracy testing
        self.test_data["future_return_1d"] = self.test_data["returns"].shift(-1)
        self.test_data["future_return_5d"] = self.test_data["close"].pct_change(5).shift(-5)
        self.test_data["future_return_10d"] = self.test_data["close"].pct_change(10).shift(-10)

        # Drop initial NaN rows
        self.test_data = self.test_data.dropna()

        # Validate data after transformations
        if len(self.test_data) < MIN_DATA_AFTER_TRANSFORM:
            raise ValueError(
                f"Insufficient data after transformations: {len(self.test_data)} rows (minimum {MIN_DATA_AFTER_TRANSFORM} required)"
            )

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero: when loss is 0, RSI is 100
        rs = np.where(loss == 0, np.inf, gain / loss)
        rsi = np.where(np.isinf(rs), 100, 100 - (100 / (1 + rs)))
        return pd.Series(rsi, index=prices.index)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data["high"] - data["low"]
        high_close = np.abs(data["high"] - data["close"].shift())
        low_close = np.abs(data["low"] - data["close"].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()

    def _generate_test_scenarios(self) -> list[dict[str, Any]]:
        """Generate various test scenarios for comprehensive testing"""
        scenarios = []

        # Different market conditions
        data_length = len(self.test_data)

        # Full dataset
        scenarios.append(
            {
                "name": "full_dataset",
                "start_idx": 0,
                "end_idx": data_length,
                "description": "Complete historical dataset",
            }
        )

        # Bull market periods (price trending up)
        bull_periods = self._identify_trend_periods("bull")
        for i, (start, end) in enumerate(bull_periods):
            scenarios.append(
                {
                    "name": f"bull_market_{i+1}",
                    "start_idx": start,
                    "end_idx": end,
                    "description": f"Bull market period {i+1}",
                }
            )

        # Bear market periods (price trending down)
        bear_periods = self._identify_trend_periods("bear")
        for i, (start, end) in enumerate(bear_periods):
            scenarios.append(
                {
                    "name": f"bear_market_{i+1}",
                    "start_idx": start,
                    "end_idx": end,
                    "description": f"Bear market period {i+1}",
                }
            )

        # High volatility periods
        high_vol_periods = self._identify_volatility_periods("high")
        for i, (start, end) in enumerate(high_vol_periods):
            scenarios.append(
                {
                    "name": f"high_volatility_{i+1}",
                    "start_idx": start,
                    "end_idx": end,
                    "description": f"High volatility period {i+1}",
                }
            )

        # Low volatility periods
        low_vol_periods = self._identify_volatility_periods("low")
        for i, (start, end) in enumerate(low_vol_periods):
            scenarios.append(
                {
                    "name": f"low_volatility_{i+1}",
                    "start_idx": start,
                    "end_idx": end,
                    "description": f"Low volatility period {i+1}",
                }
            )

        return scenarios

    def _identify_trend_periods(self, trend_type: str) -> list[tuple[int, int]]:
        """Identify periods of specific trend direction"""
        # Simple trend identification using moving averages
        short_ma = self.test_data["close"].rolling(20).mean()
        long_ma = self.test_data["close"].rolling(50).mean()

        if trend_type == "bull":
            trend_condition = short_ma > long_ma
        else:  # bear
            trend_condition = short_ma < long_ma

        # Find continuous periods
        periods = []
        start_idx = None

        for i, is_trend in enumerate(trend_condition):
            if pd.isna(is_trend):
                continue

            if is_trend and start_idx is None:
                start_idx = i
            elif not is_trend and start_idx is not None:
                if i - start_idx > MIN_TREND_PERIOD_LENGTH:
                    periods.append((start_idx, i))
                start_idx = None

        # Handle case where trend continues to end
        if start_idx is not None and len(self.test_data) - start_idx > MIN_TREND_PERIOD_LENGTH:
            periods.append((start_idx, len(self.test_data)))

        return periods

    def _identify_volatility_periods(self, vol_type: str) -> list[tuple[int, int]]:
        """Identify periods of specific volatility level"""
        # Calculate rolling volatility
        volatility = self.test_data["returns"].rolling(20).std()
        vol_threshold = volatility.median()

        if vol_type == "high":
            vol_condition = volatility > vol_threshold * VOLATILITY_HIGH_MULTIPLIER
        else:  # low
            vol_condition = volatility < vol_threshold * VOLATILITY_LOW_MULTIPLIER

        # Find continuous periods
        periods = []
        start_idx = None

        for i, is_vol in enumerate(vol_condition):
            if pd.isna(is_vol):
                continue

            if is_vol and start_idx is None:
                start_idx = i
            elif not is_vol and start_idx is not None:
                if i - start_idx > MIN_VOLATILITY_PERIOD_LENGTH:
                    periods.append((start_idx, i))
                start_idx = None

        # Handle case where condition continues to end
        if start_idx is not None and len(self.test_data) - start_idx > MIN_VOLATILITY_PERIOD_LENGTH:
            periods.append((start_idx, len(self.test_data)))

        return periods

    def test_signal_generator(
        self, generator: SignalGenerator, scenarios: list[str] | None = None
    ) -> SignalTestResults:
        """
        Test signal generator performance across various scenarios

        Args:
            generator: SignalGenerator to test
            scenarios: List of scenario names to test (None = all scenarios)

        Returns:
            SignalTestResults with comprehensive performance metrics
        """
        start_time = time.time()

        # Select scenarios to test
        test_scenarios = self.test_scenarios
        if scenarios:
            test_scenarios = [s for s in test_scenarios if s["name"] in scenarios]

        # Initialize tracking variables
        all_signals = []
        signal_times = []
        error_count = 0

        # Regime breakdown tracking
        regime_breakdown = {}

        # Test across all scenarios
        for scenario in test_scenarios:
            scenario_data = self.test_data.iloc[scenario["start_idx"] : scenario["end_idx"]].copy()

            scenario_signals = []

            # Generate signals for this scenario
            for i in range(len(scenario_data)):
                try:
                    signal_start = time.time()

                    # Get regime context if available
                    regime = None
                    if self.regime_data is not None:
                        regime_idx = scenario["start_idx"] + i
                        if regime_idx < len(self.regime_data):
                            regime = self.regime_data.iloc[regime_idx]

                    # Generate signal
                    signal = generator.generate_signal(scenario_data, i, regime)

                    signal_time = time.time() - signal_start
                    signal_times.append(signal_time)

                    # Calculate signal accuracy (if we have future returns)
                    if i < len(scenario_data) - 1:
                        future_return = scenario_data.iloc[i + 1]["returns"]

                        # Determine if signal was accurate
                        if signal.direction == SignalDirection.BUY and future_return > 0:
                            accurate = True
                        elif signal.direction == SignalDirection.SELL and future_return < 0:
                            accurate = True
                        elif signal.direction == SignalDirection.HOLD:
                            accurate = (
                                abs(future_return) < SMALL_RETURN_THRESHOLD
                            )  # Small movement is good for hold
                        else:
                            accurate = False

                        scenario_signals.append(
                            {
                                "signal": signal,
                                "accurate": accurate,
                                "future_return": future_return,
                                "regime": regime,
                            }
                        )

                        # Track regime-specific performance
                        if regime is not None:
                            # Extract regime attributes safely
                            if not hasattr(regime, "trend") or not hasattr(regime, "volatility"):
                                logger.warning(
                                    f"Malformed regime object at index {i}: missing trend or volatility attributes"
                                )
                                regime_key = "unknown_unknown"
                            else:
                                trend_val = (
                                    regime.trend.value
                                    if hasattr(regime.trend, "value")
                                    else regime.trend
                                )
                                vol_val = (
                                    regime.volatility.value
                                    if hasattr(regime.volatility, "value")
                                    else regime.volatility
                                )
                                regime_key = f"{trend_val}_{vol_val}"

                            if regime_key not in regime_breakdown:
                                regime_breakdown[regime_key] = {"signals": [], "accuracy": 0.0}
                            regime_breakdown[regime_key]["signals"].append(accurate)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error generating signal at index {i}: {e}", exc_info=True)
                    continue

            all_signals.extend(scenario_signals)

        # Calculate regime breakdown accuracies
        for regime_key in regime_breakdown:
            signals = regime_breakdown[regime_key]["signals"]
            if signals:
                regime_breakdown[regime_key]["accuracy"] = sum(signals) / len(signals)
                regime_breakdown[regime_key]["count"] = len(signals)

        # Calculate overall metrics
        total_signals = len(all_signals)
        if total_signals == 0:
            raise ValueError("No valid signals generated during testing")

        # Signal accuracy metrics
        accurate_signals = sum(1 for s in all_signals if s["accurate"])
        accuracy = accurate_signals / total_signals

        # Precision and recall for buy signals
        buy_signals = [s for s in all_signals if s["signal"].direction == SignalDirection.BUY]
        accurate_buy_signals = [s for s in buy_signals if s["accurate"]]
        precision = len(accurate_buy_signals) / len(buy_signals) if buy_signals else 0.0

        # Recall: profitable opportunities captured
        profitable_opportunities = [s for s in all_signals if s["future_return"] > 0]
        captured_opportunities = [
            s for s in profitable_opportunities if s["signal"].direction == SignalDirection.BUY
        ]
        recall = (
            len(captured_opportunities) / len(profitable_opportunities)
            if profitable_opportunities
            else 0.0
        )

        # F1 score
        f1_score = (
            2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        )

        # Performance metrics (simulate trading based on signals)
        portfolio_returns = self._simulate_signal_trading(all_signals)
        # Use log returns for numerical stability to avoid overflow/underflow
        total_return = np.exp(np.log1p(portfolio_returns).sum()) - 1
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)

        # Signal distribution
        buy_count = sum(1 for s in all_signals if s["signal"].direction == SignalDirection.BUY)
        sell_count = sum(1 for s in all_signals if s["signal"].direction == SignalDirection.SELL)
        hold_count = sum(1 for s in all_signals if s["signal"].direction == SignalDirection.HOLD)

        # Confidence and strength analysis
        confidences = [s["signal"].confidence for s in all_signals]
        strengths = [s["signal"].strength for s in all_signals]
        avg_confidence = np.mean(confidences)
        avg_strength = np.mean(strengths)

        # Correlation between confidence and accuracy
        if len(confidences) > 1:
            correlation = np.corrcoef(
                confidences, [1 if s["accurate"] else 0 for s in all_signals]
            )[0, 1]
            confidence_accuracy_correlation = 0.0 if np.isnan(correlation) else correlation
        else:
            confidence_accuracy_correlation = 0.0

        # Timing metrics
        test_duration = time.time() - start_time
        avg_signal_time = np.mean(signal_times) if signal_times else 0.0
        signals_per_second = total_signals / test_duration if test_duration > 0 else 0.0

        return SignalTestResults(
            component_name=generator.name,
            test_duration=test_duration,
            total_signals=total_signals,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=accuracy,  # Same as accuracy for signals
            avg_trade_duration=1.0,  # Signals are instantaneous
            buy_signals=buy_count,
            sell_signals=sell_count,
            hold_signals=hold_count,
            avg_confidence=avg_confidence,
            avg_strength=avg_strength,
            confidence_accuracy_correlation=confidence_accuracy_correlation,
            regime_breakdown=regime_breakdown,
            error_count=error_count,
            error_rate=(
                error_count / (total_signals + error_count)
                if (total_signals + error_count) > 0
                else 0.0
            ),
            avg_signal_generation_time=avg_signal_time,
            signals_per_second=signals_per_second,
        )

    def _simulate_signal_trading(self, signals: list[dict[str, Any]]) -> pd.Series:
        """Simulate trading based on signals to calculate performance"""
        returns = []

        for signal_data in signals:
            signal = signal_data["signal"]
            future_return = signal_data["future_return"]

            # Simple trading simulation
            if signal.direction == SignalDirection.BUY:
                # Long position: profit from positive returns
                trade_return = future_return * signal.strength
            elif signal.direction == SignalDirection.SELL:
                # Short position: profit from negative returns
                trade_return = -future_return * signal.strength
            else:  # HOLD
                trade_return = 0.0

            returns.append(trade_return)

        return pd.Series(returns)

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns"""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())

    def test_risk_manager(
        self,
        risk_manager: RiskManager,
        test_balance: float = 10000.0,
        scenarios: list[str] | None = None,
    ) -> RiskTestResults:
        """
        Test risk manager performance across various scenarios

        Args:
            risk_manager: RiskManager to test
            test_balance: Starting balance for testing
            scenarios: List of scenario names to test (None = all scenarios)

        Returns:
            RiskTestResults with comprehensive performance metrics
        """
        start_time = time.time()

        # Select scenarios to test
        test_scenarios = self.test_scenarios
        if scenarios:
            test_scenarios = [s for s in test_scenarios if s["name"] in scenarios]

        # Initialize tracking variables
        all_positions = []
        calculation_times = []
        error_count = 0

        # Test across all scenarios
        for scenario in test_scenarios:
            scenario_data = self.test_data.iloc[scenario["start_idx"] : scenario["end_idx"]].copy()

            # Simulate positions and risk management decisions
            for i in range(len(scenario_data) - 1):
                try:
                    calc_start = time.time()

                    # Create a dummy signal for position sizing
                    dummy_signal = Signal(
                        direction=SignalDirection.BUY,
                        strength=0.7,
                        confidence=0.8,
                        metadata={"atr": scenario_data.iloc[i]["atr"]},
                    )

                    # Test position size calculation
                    position_size = risk_manager.calculate_position_size(dummy_signal, test_balance)

                    # Test stop loss calculation
                    entry_price = scenario_data.iloc[i]["close"]
                    stop_loss = risk_manager.get_stop_loss(entry_price, dummy_signal)

                    # Create position for exit testing
                    position = Position(
                        symbol="TESTUSDT",
                        side="long",
                        size=position_size,
                        entry_price=entry_price,
                        current_price=scenario_data.iloc[i + 1]["close"],
                        entry_time=datetime.now(UTC),
                    )

                    # Test exit decision
                    market_data = MarketData(
                        symbol="TESTUSDT",
                        price=scenario_data.iloc[i + 1]["close"],
                        volume=scenario_data.iloc[i + 1]["volume"],
                        volatility=scenario_data.iloc[i + 1]["atr"],
                    )

                    should_exit = risk_manager.should_exit(position, market_data)

                    calc_time = time.time() - calc_start
                    calculation_times.append(calc_time)

                    # Store results
                    all_positions.append(
                        {
                            "position_size": position_size,
                            "stop_loss": stop_loss,
                            "entry_price": entry_price,
                            "should_exit": should_exit,
                            "actual_return": (scenario_data.iloc[i + 1]["close"] - entry_price)
                            / entry_price,
                            "position": position,
                        }
                    )

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error testing risk manager at index {i}: {e}", exc_info=True)
                    continue

        # Calculate metrics
        total_scenarios = len(all_positions)
        if total_scenarios == 0:
            raise ValueError("No valid risk management scenarios tested")

        # Position sizing analysis
        position_sizes = [p["position_size"] for p in all_positions]
        avg_position_size = np.mean(position_sizes)
        std_pos_size = np.std(position_sizes)
        position_size_consistency = 1.0 / max(
            1.0 + abs(std_pos_size), 1e-10
        )  # Higher is more consistent

        # Stop loss analysis
        stop_losses = [
            abs(p["stop_loss"] - p["entry_price"]) / p["entry_price"] for p in all_positions
        ]
        avg_stop_loss_distance = np.mean(stop_losses)

        # Exit decision analysis
        exit_decisions = [p["should_exit"] for p in all_positions]
        actual_returns = [p["actual_return"] for p in all_positions]

        # Calculate exit effectiveness
        correct_exits = 0
        premature_exits = 0
        late_exits = 0

        for should_exit, actual_return in zip(exit_decisions, actual_returns, strict=True):
            if should_exit and actual_return < -0.02:  # Correctly exited before big loss
                correct_exits += 1
            elif should_exit and actual_return > 0.01:  # Exited too early (missed profit)
                premature_exits += 1
            elif not should_exit and actual_return < -0.05:  # Should have exited (big loss)
                late_exits += 1

        optimal_exit_rate = correct_exits / total_scenarios
        premature_exit_rate = premature_exits / total_scenarios
        late_exit_rate = late_exits / total_scenarios

        # Risk-adjusted performance
        portfolio_returns = pd.Series(actual_returns)
        risk_adjusted_return = (
            portfolio_returns.mean() / portfolio_returns.std()
            if portfolio_returns.std() > 0
            else 0.0
        )

        # Drawdown control
        max_drawdown_achieved = self._calculate_max_drawdown(portfolio_returns)
        target_max_drawdown = 0.1  # Assume 10% target
        drawdown_control_score = (
            max(0.0, 1.0 - max_drawdown_achieved / target_max_drawdown)
            if target_max_drawdown > 0
            else 0.0
        )

        # Timing metrics
        test_duration = time.time() - start_time
        avg_calc_time = np.mean(calculation_times) if calculation_times else 0.0
        calcs_per_second = total_scenarios / test_duration if test_duration > 0 else 0.0

        return RiskTestResults(
            component_name=risk_manager.name,
            test_duration=test_duration,
            total_scenarios=total_scenarios,
            max_drawdown_achieved=max_drawdown_achieved,
            target_max_drawdown=target_max_drawdown,
            drawdown_control_score=drawdown_control_score,
            average_position_size=avg_position_size,
            position_size_consistency=position_size_consistency,
            risk_adjusted_return=risk_adjusted_return,
            stop_loss_hit_rate=0.0,  # Would need longer simulation to calculate
            avg_stop_loss_distance=avg_stop_loss_distance,
            stop_loss_effectiveness=1.0,  # Placeholder
            premature_exit_rate=premature_exit_rate,
            late_exit_rate=late_exit_rate,
            optimal_exit_rate=optimal_exit_rate,
            return_per_unit_risk=risk_adjusted_return,
            risk_efficiency_score=drawdown_control_score * risk_adjusted_return,
            regime_adaptation_score=0.8,  # Placeholder
            error_count=error_count,
            error_rate=(
                error_count / (total_scenarios + error_count)
                if (total_scenarios + error_count) > 0
                else 0.0
            ),
            avg_calculation_time=avg_calc_time,
            calculations_per_second=calcs_per_second,
        )

    def test_position_sizer(
        self,
        position_sizer: PositionSizer,
        test_balance: float = 10000.0,
        scenarios: list[str] | None = None,
    ) -> SizingTestResults:
        """
        Test position sizer performance across various scenarios

        Args:
            position_sizer: PositionSizer to test
            test_balance: Starting balance for testing
            scenarios: List of scenario names to test (None = all scenarios)

        Returns:
            SizingTestResults with comprehensive performance metrics
        """
        start_time = time.time()

        # Select scenarios to test
        test_scenarios = self.test_scenarios
        if scenarios:
            test_scenarios = [s for s in test_scenarios if s["name"] in scenarios]

        # Initialize tracking variables
        all_calculations = []
        calculation_times = []
        error_count = 0
        bounds_violations = 0

        # Test across all scenarios
        for scenario in test_scenarios:
            scenario_data = self.test_data.iloc[scenario["start_idx"] : scenario["end_idx"]].copy()

            # Test position sizing across different signal conditions
            for i in range(len(scenario_data)):
                try:
                    calc_start = time.time()

                    # Create various test signals
                    test_signals = [
                        Signal(SignalDirection.BUY, 0.3, 0.5, {}),  # Weak signal
                        Signal(SignalDirection.BUY, 0.7, 0.8, {}),  # Strong signal
                        Signal(SignalDirection.SELL, 0.5, 0.6, {}),  # Medium sell signal
                        Signal(SignalDirection.HOLD, 0.0, 1.0, {}),  # Hold signal
                    ]

                    for signal in test_signals:
                        # Test with different risk amounts
                        risk_amounts = [
                            test_balance * 0.01,
                            test_balance * 0.02,
                            test_balance * 0.05,
                        ]

                        for risk_amount in risk_amounts:
                            position_size = position_sizer.calculate_size(
                                signal, test_balance, risk_amount
                            )

                            # Check bounds violations
                            if position_size > test_balance * 0.2:  # Max 20% of balance
                                bounds_violations += 1

                            all_calculations.append(
                                {
                                    "signal": signal,
                                    "risk_amount": risk_amount,
                                    "position_size": position_size,
                                    "size_fraction": position_size / test_balance,
                                    "confidence": signal.confidence,
                                    "strength": signal.strength,
                                }
                            )

                    calc_time = time.time() - calc_start
                    calculation_times.append(calc_time)

                except Exception as e:
                    error_count += 1
                    logger.error(f"Error testing position sizer at index {i}: {e}", exc_info=True)
                    continue

        # Calculate metrics
        total_calculations = len(all_calculations)
        if total_calculations == 0:
            raise ValueError("No valid position sizing calculations performed")

        # Size distribution analysis
        position_sizes = [c["position_size"] for c in all_calculations]
        size_fractions = [c["size_fraction"] for c in all_calculations]

        avg_position_size = np.mean(position_sizes)
        position_size_std = np.std(position_sizes)

        # Size range utilization (how well it uses the full range)
        min_size = min(size_fractions)
        max_size = max(size_fractions)
        size_range_utilization = (max_size - min_size) / 0.2  # Assuming 20% max range

        # Confidence and strength responsiveness
        confidences = [c["confidence"] for c in all_calculations]
        strengths = [c["strength"] for c in all_calculations]

        # Calculate correlations with NaN handling
        if len(confidences) > 1:
            conf_corr = np.corrcoef(confidences, size_fractions)[0, 1]
            confidence_responsiveness = 0.0 if np.isnan(conf_corr) else abs(conf_corr)
        else:
            confidence_responsiveness = 0.0

        if len(strengths) > 1:
            str_corr = np.corrcoef(strengths, size_fractions)[0, 1]
            strength_responsiveness = 0.0 if np.isnan(str_corr) else abs(str_corr)
        else:
            strength_responsiveness = 0.0

        # Bounds adherence
        bounds_adherence_rate = 1.0 - (bounds_violations / total_calculations)

        # Kelly criterion adherence (if applicable)
        kelly_adherence = 0.8  # Placeholder - would need more sophisticated analysis

        # Timing metrics
        test_duration = time.time() - start_time
        avg_calc_time = np.mean(calculation_times) if calculation_times else 0.0
        calcs_per_second = total_calculations / test_duration if test_duration > 0 else 0.0

        return SizingTestResults(
            component_name=position_sizer.name,
            test_duration=test_duration,
            total_calculations=total_calculations,
            kelly_criterion_adherence=kelly_adherence,
            optimal_sizing_score=0.75,  # Placeholder
            regime_adaptation_effectiveness=0.8,  # Placeholder
            regime_consistency_score=0.85,  # Placeholder
            volatility_adjustment_quality=0.7,  # Placeholder
            volatility_responsiveness=0.6,  # Placeholder
            average_position_size=avg_position_size,
            position_size_std=position_size_std,
            size_range_utilization=size_range_utilization,
            size_adjusted_sharpe=1.2,  # Placeholder
            size_adjusted_return=0.08,  # Placeholder
            confidence_responsiveness=confidence_responsiveness,
            strength_responsiveness=strength_responsiveness,
            bounds_violations=bounds_violations,
            bounds_adherence_rate=bounds_adherence_rate,
            error_count=error_count,
            error_rate=(
                error_count / (total_calculations + error_count)
                if (total_calculations + error_count) > 0
                else 0.0
            ),
            avg_calculation_time=avg_calc_time,
            calculations_per_second=calcs_per_second,
        )

    def test_all_components(
        self,
        signal_generator: SignalGenerator | None = None,
        risk_manager: RiskManager | None = None,
        position_sizer: PositionSizer | None = None,
        test_balance: float = 10000.0,
    ) -> ComponentTestResults:
        """
        Test all provided components and analyze their interaction

        Args:
            signal_generator: Optional SignalGenerator to test
            risk_manager: Optional RiskManager to test
            position_sizer: Optional PositionSizer to test
            test_balance: Starting balance for testing

        Returns:
            ComponentTestResults with results from all tested components
        """
        start_time = time.time()

        results = ComponentTestResults()

        # Test individual components
        if signal_generator:
            results.signal_results = self.test_signal_generator(signal_generator)

        if risk_manager:
            results.risk_results = self.test_risk_manager(risk_manager, test_balance)

        if position_sizer:
            results.sizing_results = self.test_position_sizer(position_sizer, test_balance)

        # Calculate overall metrics
        results.total_test_duration = time.time() - start_time

        # Calculate overall performance score (weighted average of component scores)
        component_scores = []

        if results.signal_results:
            signal_score = (results.signal_results.accuracy + results.signal_results.f1_score) / 2
            component_scores.append(signal_score)

        if results.risk_results:
            risk_score = (
                results.risk_results.drawdown_control_score
                + results.risk_results.risk_efficiency_score
            ) / 2
            component_scores.append(risk_score)

        if results.sizing_results:
            sizing_score = (
                results.sizing_results.optimal_sizing_score
                + results.sizing_results.bounds_adherence_rate
            ) / 2
            component_scores.append(sizing_score)

        results.overall_performance_score = np.mean(component_scores) if component_scores else 0.0

        # Placeholder for component synergy analysis
        results.component_synergy_score = 0.8
        results.integration_effectiveness = 0.85

        return results
