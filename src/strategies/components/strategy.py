"""
Composable Strategy Class

This module defines the Strategy class that composes SignalGenerator, RiskManager,
and PositionSizer components to create a unified trading strategy with comprehensive
logging and decision tracking.
"""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .runtime import FeatureGeneratorSpec, StrategyDataset

from .policies import PolicyBundle
from .position_sizer import PositionSizer
from .regime_context import EnhancedRegimeDetector, RegimeContext
from .risk_manager import MarketData, Position, RiskManager
from .signal_generator import Signal, SignalDirection, SignalGenerator


@dataclass
class TradingDecision:
    """
    Complete trading decision with all component outputs

    Attributes:
        timestamp: When the decision was made
        signal: Generated trading signal
        position_size: Calculated position size
        regime: Market regime context
        risk_metrics: Risk-related metrics
        execution_time_ms: Time taken for decision
        metadata: Additional decision metadata
    """

    timestamp: datetime
    signal: Signal
    position_size: float
    regime: RegimeContext | None
    risk_metrics: dict[str, float]
    execution_time_ms: float
    metadata: dict[str, Any]
    policies: PolicyBundle | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "signal": {
                "direction": self.signal.direction.value,
                "strength": self.signal.strength,
                "confidence": self.signal.confidence,
                "metadata": self.signal.metadata,
            },
            "position_size": self.position_size,
            "regime": (
                {
                    "trend": self.regime.trend.value if self.regime else None,
                    "volatility": self.regime.volatility.value if self.regime else None,
                    "confidence": self.regime.confidence if self.regime else None,
                    "duration": self.regime.duration if self.regime else None,
                    "strength": self.regime.strength if self.regime else None,
                }
                if self.regime
                else None
            ),
            "risk_metrics": self.risk_metrics,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata,
            "policies": self.policies.to_dict() if self.policies else None,
        }


class Strategy:
    """
    Composable strategy class that orchestrates components

    This class composes SignalGenerator, RiskManager, and PositionSizer components
    to create a unified trading strategy with comprehensive logging and decision tracking.
    """

    def __init__(
        self,
        name: str,
        signal_generator: SignalGenerator,
        risk_manager: RiskManager,
        position_sizer: PositionSizer,
        regime_detector: EnhancedRegimeDetector | None = None,
        enable_logging: bool = True,
        max_history: int = 1000,
    ):
        """
        Initialize composable strategy

        Args:
            name: Strategy name for identification
            signal_generator: Component for generating trading signals
            risk_manager: Component for risk management and position sizing
            position_sizer: Component for final position size calculation
            regime_detector: Optional regime detection component
            enable_logging: Whether to enable detailed logging
            max_history: Maximum number of decisions to keep in history

        Raises:
            ValueError: If required components are None or invalid
        """
        # Validate required components before initialization
        self._validate_components(name, signal_generator, risk_manager, position_sizer)

        self.name = name
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.regime_detector = regime_detector or EnhancedRegimeDetector()
        self.trading_pair = "BTCUSDT"

        # Logging setup
        self.enable_logging = enable_logging
        self.logger = logging.getLogger(f"Strategy.{name}")
        if enable_logging:
            self.logger.setLevel(logging.INFO)

        # Decision history
        self.decision_history: list[TradingDecision] = []
        self.max_history = max_history

        # Performance metrics
        self.metrics = {
            "total_decisions": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "avg_execution_time_ms": 0.0,
            "avg_signal_confidence": 0.0,
            "avg_position_size": 0.0,
            "last_updated": datetime.now(UTC),
        }

        # Runtime configuration
        self._warmup_override: int | None = None
        self._last_signal: Signal | None = None
        self._additional_risk_context_provider: (
            Callable[[pd.DataFrame, int, Signal], dict[str, Any] | None] | None
        ) = None

        self.logger.info(
            f"Strategy '{name}' initialized with components: "
            f"SignalGen={signal_generator.name}, "
            f"RiskMgr={risk_manager.name}, "
            f"PosSizer={position_sizer.name}"
        )

    @staticmethod
    def _validate_components(
        name: str,
        signal_generator: SignalGenerator | None,
        risk_manager: RiskManager | None,
        position_sizer: PositionSizer | None,
    ) -> None:
        """Validate that required components are properly initialized.

        Args:
            name: Strategy name for error messages.
            signal_generator: Signal generator component to validate.
            risk_manager: Risk manager component to validate.
            position_sizer: Position sizer component to validate.

        Raises:
            ValueError: If any required component is None or invalid.
        """
        if not name or not isinstance(name, str):
            raise ValueError(f"Strategy name must be a non-empty string, got: {name}")

        if signal_generator is None:
            raise ValueError(
                f"Strategy '{name}': signal_generator cannot be None. "
                "Provide a valid SignalGenerator instance."
            )

        if risk_manager is None:
            raise ValueError(
                f"Strategy '{name}': risk_manager cannot be None. "
                "Provide a valid RiskManager instance."
            )

        if position_sizer is None:
            raise ValueError(
                f"Strategy '{name}': position_sizer cannot be None. "
                "Provide a valid PositionSizer instance."
            )

        # Validate components have required interface (duck typing check)
        if not hasattr(signal_generator, "generate_signal"):
            raise ValueError(
                f"Strategy '{name}': signal_generator missing 'generate_signal' method. "
                f"Got type: {type(signal_generator).__name__}"
            )

        if not hasattr(risk_manager, "calculate_position_size"):
            raise ValueError(
                f"Strategy '{name}': risk_manager missing 'calculate_position_size' method. "
                f"Got type: {type(risk_manager).__name__}"
            )

        if not hasattr(position_sizer, "calculate_size"):
            raise ValueError(
                f"Strategy '{name}': position_sizer missing 'calculate_size' method. "
                f"Got type: {type(position_sizer).__name__}"
            )

    @property
    def warmup_period(self) -> int:
        """Return the minimum history required before producing decisions."""

        if self._warmup_override is not None:
            return self._warmup_override

        warmups: list[int] = []
        for component in (
            self.signal_generator,
            self.risk_manager,
            self.position_sizer,
            self.regime_detector,
        ):
            component_warmup = getattr(component, "warmup_period", 0)
            if isinstance(component_warmup, int) and component_warmup >= 0:
                warmups.append(component_warmup)

        return max(warmups or [0])

    def set_warmup_period(self, periods: int) -> None:
        """Override the automatically derived warmup period."""

        if periods < 0:
            raise ValueError("warmup period must be non-negative")
        self._warmup_override = periods

    def get_feature_generators(self) -> Sequence[FeatureGeneratorSpec]:
        """Return feature generators declared by composed components."""

        generators: list[FeatureGeneratorSpec] = []
        for component in (
            self.signal_generator,
            self.risk_manager,
            self.position_sizer,
            self.regime_detector,
        ):
            getter = getattr(component, "get_feature_generators", None)
            if callable(getter):
                component_generators = list(getter())
                if component_generators:
                    generators.extend(component_generators)
        return generators

    def prepare_runtime(self, dataset: StrategyDataset) -> None:
        """Hook for runtime initialisation. Default implementation is a no-op."""

        self.logger.debug(
            "Runtime prepared for %s with %d rows and %d feature sets",
            self.name,
            len(dataset.data),
            len(dataset.feature_caches),
        )

    def finalize_runtime(self) -> None:
        """Hook invoked after runtime execution completes."""

        self.logger.debug("Runtime finalised for %s", self.name)

    def process_candle(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        current_positions: list[Position] | None = None,
    ) -> TradingDecision:
        """
        Process a single candle and make trading decision

        This is the main method that coordinates all components to make a trading decision.

        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame
            balance: Available account balance
            current_positions: List of current positions (optional)

        Returns:
            TradingDecision containing all decision information

        Raises:
            ValueError: If input parameters are invalid
            IndexError: If index is out of bounds
        """
        start_time = time.time()
        timestamp = datetime.now(UTC)

        try:
            # Validate inputs
            self._validate_inputs(df, index, balance)

            # Step 1: Detect market regime
            regime = self._detect_regime(df, index)

            # Step 2: Generate trading signal
            signal = self._generate_signal(df, index, regime)

            # Step 3: Build risk context and calculate risk-based position size
            risk_context = self._build_risk_context(df, index, signal)
            risk_position_size = self._calculate_risk_position_size(
                signal, balance, regime, risk_context
            )

            # Step 4: Apply position sizer adjustments
            final_position_size = self._calculate_final_position_size(
                signal, balance, risk_position_size, regime
            )

            # Step 5: Validate and bound final position size
            validated_position_size = self._validate_position_size(
                final_position_size, signal, balance, regime
            )

            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000

            # Create risk metrics
            risk_metrics = self._calculate_risk_metrics(
                signal, balance, risk_position_size, validated_position_size, regime
            )

            # Create decision metadata
            metadata = self._create_decision_metadata(
                df,
                index,
                balance,
                current_positions,
                regime,
                signal,
                risk_position_size,
                validated_position_size,
            )

            policies = None
            try:
                policy_kwargs = self._prepare_risk_kwargs(
                    self.risk_manager.get_position_policies, risk_context
                )
                policies = self.risk_manager.get_position_policies(
                    signal,
                    balance,
                    regime,
                    **policy_kwargs,
                )
            except (ValueError, KeyError, AttributeError) as policy_error:
                self.logger.debug("Risk policy extraction failed: %s", policy_error)

            # Create trading decision
            decision = TradingDecision(
                timestamp=timestamp,
                signal=signal,
                position_size=validated_position_size,
                regime=regime,
                risk_metrics=risk_metrics,
                execution_time_ms=execution_time_ms,
                metadata=metadata,
                policies=policies,
            )

            # Record decision
            self._record_decision(decision)

            # Log decision
            if self.enable_logging:
                self._log_decision(decision)

            return decision

        except (ValueError, KeyError, IndexError, TypeError) as e:
            # Handle errors gracefully
            execution_time_ms = (time.time() - start_time) * 1000

            self.logger.exception("Error processing candle at index %d", index)

            # Return safe decision
            safe_signal = Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"error": str(e), "component": "strategy"},
            )

            decision = TradingDecision(
                timestamp=timestamp,
                signal=safe_signal,
                position_size=0.0,
                regime=None,
                risk_metrics={"error": True},
                execution_time_ms=execution_time_ms,
                metadata={"error": str(e), "safe_mode": True},
            )

            self._record_decision(decision)
            return decision

    def should_exit_position(
        self,
        position: Position,
        current_data: MarketData,
        regime: RegimeContext | None = None,
    ) -> bool:
        """
        Determine if a position should be exited

        Args:
            position: Current position to evaluate
            current_data: Current market data
            regime: Optional regime context

        Returns:
            True if position should be exited, False otherwise
        """
        try:
            return self.risk_manager.should_exit(position, current_data, regime)
        except (ValueError, KeyError, AttributeError):
            self.logger.exception("Error in exit decision")
            return False  # Conservative default

    def get_stop_loss_price(
        self,
        entry_price: float,
        signal: Signal,
        regime: RegimeContext | None = None,
    ) -> float:
        """
        Get stop loss price for a position

        Args:
            entry_price: Entry price for the position
            signal: Trading signal that triggered the position
            regime: Optional regime context

        Returns:
            Stop loss price level
        """
        try:
            return self.risk_manager.get_stop_loss(entry_price, signal, regime)
        except (ValueError, KeyError, AttributeError):
            self.logger.exception("Error calculating stop loss")
            # Return conservative stop loss
            if signal.direction == SignalDirection.BUY:
                return entry_price * 0.95  # 5% stop loss for long
            if signal.direction == SignalDirection.SELL:
                return entry_price * 1.05  # 5% stop loss for short
            return entry_price

    def get_performance_metrics(self, lookback_decisions: int = 100) -> dict[str, Any]:
        """
        Get strategy performance metrics

        Args:
            lookback_decisions: Number of recent decisions to analyze

        Returns:
            Dictionary of performance metrics
        """
        if not self.decision_history:
            return self.metrics.copy()

        # Get recent decisions
        recent_decisions = self.decision_history[-lookback_decisions:]

        # Calculate metrics
        total_decisions = len(recent_decisions)
        buy_signals = sum(1 for d in recent_decisions if d.signal.direction == SignalDirection.BUY)
        sell_signals = sum(
            1 for d in recent_decisions if d.signal.direction == SignalDirection.SELL
        )
        hold_signals = sum(
            1 for d in recent_decisions if d.signal.direction == SignalDirection.HOLD
        )

        avg_execution_time = sum(d.execution_time_ms for d in recent_decisions) / total_decisions
        avg_confidence = sum(d.signal.confidence for d in recent_decisions) / total_decisions
        avg_position_size = sum(d.position_size for d in recent_decisions) / total_decisions

        # Regime analysis
        regime_distribution = {}
        for decision in recent_decisions:
            if decision.regime:
                regime_key = f"{decision.regime.trend.value}_{decision.regime.volatility.value}"
                regime_distribution[regime_key] = regime_distribution.get(regime_key, 0) + 1

        return {
            "total_decisions": total_decisions,
            "buy_signals": buy_signals,
            "sell_signals": sell_signals,
            "hold_signals": hold_signals,
            "buy_signal_pct": (buy_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            "sell_signal_pct": (sell_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            "hold_signal_pct": (hold_signals / total_decisions) * 100 if total_decisions > 0 else 0,
            "avg_execution_time_ms": avg_execution_time,
            "avg_signal_confidence": avg_confidence,
            "avg_position_size": avg_position_size,
            "regime_distribution": regime_distribution,
            "component_info": {
                "signal_generator": self.signal_generator.get_parameters(),
                "risk_manager": self.risk_manager.get_parameters(),
                "position_sizer": self.position_sizer.get_parameters(),
            },
            "last_updated": datetime.now(UTC).isoformat(),
        }

    def get_recent_decisions(self, count: int = 10) -> list[dict[str, Any]]:
        """
        Get recent trading decisions

        Args:
            count: Number of recent decisions to return

        Returns:
            List of decision dictionaries
        """
        recent = self.decision_history[-count:] if self.decision_history else []
        return [decision.to_dict() for decision in recent]

    def clear_history(self) -> None:
        """Clear decision history and reset metrics"""
        self.decision_history.clear()
        self.metrics = {
            "total_decisions": 0,
            "buy_signals": 0,
            "sell_signals": 0,
            "hold_signals": 0,
            "avg_execution_time_ms": 0.0,
            "avg_signal_confidence": 0.0,
            "avg_position_size": 0.0,
            "last_updated": datetime.now(UTC),
        }
        self.logger.info("Strategy history and metrics cleared")

    def set_additional_risk_context_provider(
        self,
        provider: Callable[[pd.DataFrame, int, Signal], dict[str, Any] | None] | None,
    ) -> None:
        """Register a hook that can enrich the risk context before delegation."""

        self._additional_risk_context_provider = provider

    def get_component_info(self) -> dict[str, dict[str, Any]]:
        """Get information about all components"""
        return {
            "signal_generator": self.signal_generator.get_parameters(),
            "risk_manager": self.risk_manager.get_parameters(),
            "position_sizer": self.position_sizer.get_parameters(),
            "regime_detector": {"type": "EnhancedRegimeDetector"},
        }

    # ------------------------------------------------------------------
    # Legacy compatibility helpers (temporary during migration)
    # ------------------------------------------------------------------

    def set_trading_pair(self, trading_pair: str) -> None:
        """Set default trading pair."""
        self.trading_pair = trading_pair

    def get_trading_pair(self) -> str:
        """Return current trading pair."""
        return self.trading_pair

    def get_parameters(self) -> dict[str, Any]:
        """Expose key configuration parameters for compatibility."""
        params: dict[str, Any] = {
            "name": self.name,
            "trading_pair": self.trading_pair,
            "components": self.get_component_info(),
        }
        for attr in (
            "model_path",
            "sequence_length",
            "use_prediction_engine",
            "model_name",
            "model_type",
            "timeframe",
            "stop_loss_pct",
            "take_profit_pct",
            "risk_per_trade",
            "base_fraction",
            "min_confidence",
        ):
            if hasattr(self, attr):
                params[attr] = getattr(self, attr)
        return params

    def get_risk_overrides(self) -> dict[str, Any] | None:
        """Return configured risk overrides when provided."""
        return getattr(self, "_risk_overrides", None)

    def set_risk_overrides(self, overrides: dict[str, Any] | None) -> None:
        """Set risk overrides for the strategy.

        Args:
            overrides: Dictionary of risk override parameters, or None to clear.
        """
        if overrides is None:
            self._risk_overrides = {}
        else:
            self._risk_overrides = dict(overrides)

    def _validate_inputs(self, df: pd.DataFrame, index: int, balance: float) -> None:
        """Validate input parameters"""
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of bounds for DataFrame of length {len(df)}")

        if balance <= 0:
            raise ValueError(f"Balance must be positive, got {balance}")

        # Check for required columns
        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    def _detect_regime(self, df: pd.DataFrame, index: int) -> RegimeContext | None:
        """Detect market regime"""
        try:
            return self.regime_detector.detect_regime(df, index)
        except Exception as e:
            self.logger.warning("Regime detection failed: %s", e)
            return None

    def _generate_signal(
        self,
        df: pd.DataFrame,
        index: int,
        regime: RegimeContext | None,
    ) -> Signal:
        """Generate trading signal"""
        try:
            return self.signal_generator.generate_signal(df, index, regime)
        except Exception as e:
            self.logger.exception("Signal generation failed")
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={"error": str(e), "component": "signal_generator"},
            )

    def _build_risk_context(
        self,
        df: pd.DataFrame,
        index: int,
        signal: Signal,
    ) -> dict[str, Any]:
        """Construct contextual information for risk management delegation."""

        price = float(df["close"].iloc[index]) if "close" in df.columns else 0.0
        overrides = self.get_risk_overrides() or {}
        context: dict[str, Any] = {
            "df": df,
            "index": index,
            "price": price,
            "indicators": self._collect_indicator_snapshot(df, index, signal),
            "strategy_overrides": overrides,
        }

        if self._additional_risk_context_provider is not None:
            try:
                extra_context = self._additional_risk_context_provider(df, index, signal)
            except (ValueError, KeyError, TypeError) as exc:  # pragma: no cover - defensive logging
                self.logger.debug(
                    "Additional risk context provider failed at index %s: %s",
                    index,
                    exc,
                )
            else:
                if extra_context:
                    context.update(dict(extra_context))

        return context

    def _collect_indicator_snapshot(
        self,
        df: pd.DataFrame,
        index: int,
        signal: Signal,
    ) -> dict[str, Any]:
        """Return a lightweight dictionary of indicator values for the current row."""

        try:
            row = df.iloc[index]
        except (IndexError, KeyError):
            return {}

        snapshot: dict[str, Any] = {}
        for column in df.columns:
            try:
                snapshot[column] = row[column]
            except (KeyError, TypeError):
                continue

        # Propagate signal metadata keys when present for adapters that rely on them.
        if signal.metadata:
            snapshot.update({f"signal_{k}": v for k, v in signal.metadata.items()})

        return snapshot

    def _prepare_risk_kwargs(self, method: Any, context: dict[str, Any]) -> dict[str, Any]:
        """Filter contextual kwargs to those accepted by ``method``."""

        if not context:
            return {}

        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            return dict(context)

        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
        ):
            return dict(context)

        allowed = {name for name in signature.parameters if name not in {"self"}}
        return {key: value for key, value in context.items() if key in allowed}

    def _calculate_risk_position_size(
        self,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None,
        context: dict[str, Any],
    ) -> float:
        """Calculate risk-based position size"""
        try:
            filtered_context = self._prepare_risk_kwargs(
                self.risk_manager.calculate_position_size, context
            )
            return self.risk_manager.calculate_position_size(
                signal,
                balance,
                regime,
                **filtered_context,
            )
        except Exception:
            self.logger.exception("Risk position size calculation failed")
            return 0.0

    def _calculate_final_position_size(
        self,
        signal: Signal,
        balance: float,
        risk_amount: float,
        regime: RegimeContext | None,
    ) -> float:
        """Calculate final position size using position sizer"""
        try:
            return self.position_sizer.calculate_size(signal, balance, risk_amount, regime)
        except Exception:
            self.logger.exception("Final position size calculation failed")
            return risk_amount  # Fallback to risk manager's calculation

    def _validate_position_size(
        self,
        position_size: float,
        signal: Signal,
        balance: float,
        regime: RegimeContext | None,
    ) -> float:
        """Validate and bound position size"""
        if signal.direction == SignalDirection.HOLD:
            return 0.0

        # Apply reasonable bounds
        max_position = balance * 0.25  # Maximum 25% of balance
        min_position = balance * 0.001  # Minimum 0.1% of balance

        # Respect zero position (no trade decision)
        if position_size == 0.0:
            return 0.0

        # Apply bounds only for positive positions
        return max(min_position, min(max_position, position_size))

    def _calculate_risk_metrics(
        self,
        signal: Signal,
        balance: float,
        risk_position_size: float,
        final_position_size: float,
        regime: RegimeContext | None,
    ) -> dict[str, float]:
        """Calculate risk-related metrics"""
        return {
            "risk_position_size": risk_position_size,
            "final_position_size": final_position_size,
            "position_size_ratio": (
                final_position_size / risk_position_size if risk_position_size > 0 else 0
            ),
            "balance_risk_pct": (final_position_size / balance) * 100 if balance > 0 else 0,
            "signal_confidence": signal.confidence,
            "signal_strength": signal.strength,
            "regime_confidence": regime.confidence if regime else 0.0,
        }

    def _create_decision_metadata(
        self,
        df: pd.DataFrame,
        index: int,
        balance: float,
        current_positions: list[Position] | None,
        regime: RegimeContext | None,
        signal: Signal,
        risk_position_size: float,
        final_position_size: float,
    ) -> dict[str, Any]:
        """Create comprehensive decision metadata"""
        metadata = {
            "strategy_name": self.name,
            "index": index,
            "timestamp_data": df.index[index] if hasattr(df.index, "__getitem__") else None,
            "balance": balance,
            "current_positions_count": len(current_positions) if current_positions else 0,
            "components": {
                "signal_generator": self.signal_generator.name,
                "risk_manager": self.risk_manager.name,
                "position_sizer": self.position_sizer.name,
            },
            "market_data": {
                "open": float(df.iloc[index]["open"]),
                "high": float(df.iloc[index]["high"]),
                "low": float(df.iloc[index]["low"]),
                "close": float(df.iloc[index]["close"]),
                "volume": float(df.iloc[index]["volume"]),
            },
            "decision_flow": {
                "risk_position_size": risk_position_size,
                "final_position_size": final_position_size,
                "size_adjustment_ratio": (
                    final_position_size / risk_position_size if risk_position_size > 0 else 0
                ),
            },
        }

        # Runtime engines only allow SELL decisions to enter shorts when strategies
        # explicitly opt in via ``enter_short=True`` metadata. Default to long-only.
        enter_short_flag = signal.metadata.get("enter_short") if signal.metadata else None
        if signal.direction == SignalDirection.SELL:
            metadata["enter_short"] = (
                bool(enter_short_flag) if enter_short_flag is not None else False
            )
        elif enter_short_flag is not None:
            metadata["enter_short"] = bool(enter_short_flag)

        # Add regime information if available
        if regime:
            metadata["regime"] = {
                "trend": regime.trend.value,
                "volatility": regime.volatility.value,
                "confidence": regime.confidence,
                "duration": regime.duration,
                "strength": regime.strength,
            }

        return metadata

    def _record_decision(self, decision: TradingDecision) -> None:
        """Record decision in history and update metrics"""
        self.decision_history.append(decision)

        # Limit history size
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-int(self.max_history * 0.8) :]

        # Update metrics
        self.metrics["total_decisions"] += 1

        if decision.signal.direction == SignalDirection.BUY:
            self.metrics["buy_signals"] += 1
        elif decision.signal.direction == SignalDirection.SELL:
            self.metrics["sell_signals"] += 1
        else:
            self.metrics["hold_signals"] += 1

        # Update running averages
        total = self.metrics["total_decisions"]
        self.metrics["avg_execution_time_ms"] = (
            self.metrics["avg_execution_time_ms"] * (total - 1) + decision.execution_time_ms
        ) / total
        self.metrics["avg_signal_confidence"] = (
            self.metrics["avg_signal_confidence"] * (total - 1) + decision.signal.confidence
        ) / total
        self.metrics["avg_position_size"] = (
            self.metrics["avg_position_size"] * (total - 1) + decision.position_size
        ) / total

        self.metrics["last_updated"] = datetime.now(UTC)

    def _log_decision(self, decision: TradingDecision) -> None:
        """Log trading decision"""
        regime_str = ""
        if decision.regime:
            regime_str = f" | Regime: {decision.regime.trend.value}/{decision.regime.volatility.value} (conf: {decision.regime.confidence:.2f})"

        self.logger.info(
            f"Decision: {decision.signal.direction.value.upper()} "
            f"| Size: {decision.position_size:.2f} "
            f"| Confidence: {decision.signal.confidence:.2f} "
            f"| Strength: {decision.signal.strength:.2f} "
            f"| Time: {decision.execution_time_ms:.1f}ms"
            f"{regime_str}"
        )

    def __str__(self) -> str:
        """String representation of strategy"""
        return (
            f"Strategy(name='{self.name}', "
            f"signal_gen='{self.signal_generator.name}', "
            f"risk_mgr='{self.risk_manager.name}', "
            f"pos_sizer='{self.position_sizer.name}')"
        )

    def __repr__(self) -> str:
        """Detailed representation of strategy"""
        return self.__str__()
