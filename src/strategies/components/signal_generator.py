"""
Signal Generator Components

This module defines the abstract SignalGenerator interface and related data models
for generating trading signals in the component-based strategy architecture.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

from src.infrastructure.circuit_breaker import CircuitBreaker, CircuitBreakerError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .regime_context import RegimeContext
    from .runtime import FeatureGeneratorSpec


class SignalDirection(Enum):
    """Enumeration for signal directions"""

    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """
    Data class representing a trading signal

    Attributes:
        direction: The signal direction (BUY, SELL, HOLD)
        strength: Signal strength from 0.0 to 1.0
        confidence: Confidence in the signal from 0.0 to 1.0
        metadata: Additional signal information and context
    """

    direction: SignalDirection
    strength: float
    confidence: float
    metadata: dict[str, Any]

    def __post_init__(self):
        """Validate signal parameters after initialization"""
        self._validate_signal()

    def _validate_signal(self):
        """Validate signal parameters are within acceptable bounds"""
        if not isinstance(self.direction, SignalDirection):
            raise ValueError(
                f"direction must be a SignalDirection enum, got {type(self.direction)}"
            )

        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be between 0.0 and 1.0, got {self.strength}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(self.metadata)}")


class SignalGenerator(ABC):
    """
    Abstract base class for signal generators

    Signal generators are responsible for analyzing market data and generating
    trading signals with associated confidence scores. They can be regime-aware
    and adapt their behavior based on market conditions.
    """

    def __init__(self, name: str):
        """
        Initialize the signal generator

        Args:
            name: Unique name for this signal generator
        """
        self.name = name

    @abstractmethod
    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """
        Generate a trading signal based on market data

        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context for regime-aware signal generation

        Returns:
            Signal object containing direction, strength, confidence, and metadata

        Raises:
            ValueError: If input parameters are invalid
            IndexError: If index is out of bounds
        """
        pass

    @abstractmethod
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """
        Get confidence score for signal generation at the given index

        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame

        Returns:
            Confidence score between 0.0 and 1.0

        Raises:
            ValueError: If input parameters are invalid
            IndexError: If index is out of bounds
        """
        pass

    def validate_inputs(self, df: pd.DataFrame, index: int) -> None:
        """
        Validate input parameters for signal generation

        Args:
            df: DataFrame to validate
            index: Index to validate

        Raises:
            ValueError: If DataFrame is empty or missing required columns
            IndexError: If index is out of bounds
        """
        if df.empty:
            raise ValueError("DataFrame cannot be empty")

        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of bounds for DataFrame of length {len(df)}")

        required_columns = ["open", "high", "low", "close", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")

    def get_parameters(self) -> dict[str, Any]:
        """
        Get signal generator parameters for logging and serialization

        Returns:
            Dictionary of parameter names and values
        """
        return {"name": self.name, "type": self.__class__.__name__}

    @property
    def warmup_period(self) -> int:
        """Declare the minimum history required by the generator."""

        return 0

    def get_feature_generators(self) -> Sequence["FeatureGeneratorSpec"]:
        """Return feature generators required by this signal generator."""

        return []


class HoldSignalGenerator(SignalGenerator):
    """
    Simple signal generator that always returns HOLD signals

    Useful for testing and as a conservative fallback strategy
    """

    def __init__(self):
        super().__init__("hold_signal_generator")

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """Generate a HOLD signal with neutral strength and high confidence"""
        self.validate_inputs(df, index)

        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=1.0,
            metadata={
                "generator": self.name,
                "index": index,
                "timestamp": df.index[index] if hasattr(df.index, "__getitem__") else None,
                "regime": regime.trend.value if regime else None,
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Always return high confidence for HOLD signals"""
        self.validate_inputs(df, index)
        return 1.0


class RandomSignalGenerator(SignalGenerator):
    """
    Random signal generator for testing purposes

    Generates random signals with configurable probabilities
    """

    def __init__(self, buy_prob: float = 0.3, sell_prob: float = 0.3, seed: int | None = None):
        """
        Initialize random signal generator

        Args:
            buy_prob: Probability of generating BUY signal (0.0 to 1.0)
            sell_prob: Probability of generating SELL signal (0.0 to 1.0)
            seed: Random seed for reproducible results
        """
        super().__init__("random_signal_generator")

        if not 0.0 <= buy_prob <= 1.0:
            raise ValueError(f"buy_prob must be between 0.0 and 1.0, got {buy_prob}")
        if not 0.0 <= sell_prob <= 1.0:
            raise ValueError(f"sell_prob must be between 0.0 and 1.0, got {sell_prob}")
        if buy_prob + sell_prob > 1.0:
            raise ValueError(f"buy_prob + sell_prob cannot exceed 1.0, got {buy_prob + sell_prob}")

        self.buy_prob = buy_prob
        self.sell_prob = sell_prob
        self.hold_prob = 1.0 - buy_prob - sell_prob

        if seed is not None:
            import numpy as np

            np.random.seed(seed)

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """Generate a random signal based on configured probabilities"""
        import numpy as np

        self.validate_inputs(df, index)

        # Generate random signal direction
        rand = np.random.random()
        if rand < self.buy_prob:
            direction = SignalDirection.BUY
        elif rand < self.buy_prob + self.sell_prob:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        # Generate random strength and confidence
        strength = np.random.random() if direction != SignalDirection.HOLD else 0.0
        confidence = np.random.uniform(0.3, 0.9)  # Avoid very low confidence

        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                "generator": self.name,
                "index": index,
                "timestamp": df.index[index] if hasattr(df.index, "__getitem__") else None,
                "regime": regime.trend.value if regime else None,
                "random_seed": rand,
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Return random confidence score"""
        import numpy as np

        self.validate_inputs(df, index)
        return np.random.uniform(0.3, 0.9)

    def get_parameters(self) -> dict[str, Any]:
        """Get random signal generator parameters"""
        params = super().get_parameters()
        params.update(
            {"buy_prob": self.buy_prob, "sell_prob": self.sell_prob, "hold_prob": self.hold_prob}
        )
        return params


class WeightedVotingSignalGenerator(SignalGenerator):
    """
    Weighted voting signal generator

    Combines multiple signal generators using weighted voting to produce
    a consensus signal with aggregated confidence scores.
    """

    def __init__(
        self,
        generators: dict[SignalGenerator, float],
        min_confidence: float = 0.3,
        consensus_threshold: float = 0.6,
    ):
        """
        Initialize weighted voting signal generator

        Args:
            generators: Dictionary mapping signal generators to their weights
            min_confidence: Minimum confidence threshold for individual signals
            consensus_threshold: Threshold for consensus agreement (0.5 = majority)
        """
        super().__init__("weighted_voting_signal_generator")

        if not generators:
            raise ValueError("At least one signal generator must be provided")

        # Validate weights
        total_weight = sum(generators.values())
        if total_weight <= 0:
            raise ValueError("Total weight must be positive")

        # Normalize weights to sum to 1.0
        self.generators = {gen: weight / total_weight for gen, weight in generators.items()}

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be between 0.0 and 1.0, got {min_confidence}")

        if not 0.0 <= consensus_threshold <= 1.0:
            raise ValueError(
                f"consensus_threshold must be between 0.0 and 1.0, got {consensus_threshold}"
            )

        self.min_confidence = min_confidence
        self.consensus_threshold = consensus_threshold

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """Generate consensus signal from weighted voting"""
        self.validate_inputs(df, index)

        # Collect signals from all generators
        signals = []
        for generator, weight in self.generators.items():
            try:
                signal = generator.generate_signal(df, index, regime)
                if signal.confidence >= self.min_confidence:
                    signals.append((signal, weight))
            except Exception as e:
                # Log error but continue with other generators
                logger.warning("Generator %s failed: %s", generator.name, e, exc_info=False)
                continue

        if not signals:
            # No valid signals, return HOLD
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "index": index,
                    "reason": "no_valid_signals",
                    "total_generators": len(self.generators),
                    "valid_signals": 0,
                    "consensus_threshold": self.consensus_threshold,
                },
            )

        # Calculate weighted votes
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_confidence = 0.0
        total_weight = 0.0

        for signal, weight in signals:
            weighted_strength = signal.strength * signal.confidence * weight

            if signal.direction == SignalDirection.BUY:
                buy_score += weighted_strength
            elif signal.direction == SignalDirection.SELL:
                sell_score += weighted_strength
            else:  # HOLD
                hold_score += weighted_strength

            total_confidence += signal.confidence * weight
            total_weight += weight

        # Normalize scores
        total_score = buy_score + sell_score + hold_score
        if total_score > 0:
            buy_score /= total_score
            sell_score /= total_score
            hold_score /= total_score

        # Determine consensus direction
        max_score = max(buy_score, sell_score, hold_score)

        if max_score < self.consensus_threshold:
            # No consensus, return HOLD
            direction = SignalDirection.HOLD
            strength = 0.0
        elif buy_score == max_score:
            direction = SignalDirection.BUY
            strength = buy_score
        elif sell_score == max_score:
            direction = SignalDirection.SELL
            strength = sell_score
        else:
            direction = SignalDirection.HOLD
            strength = 0.0

        # Calculate consensus confidence
        consensus_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

        return Signal(
            direction=direction,
            strength=strength,
            confidence=consensus_confidence,
            metadata={
                "generator": self.name,
                "index": index,
                "buy_score": buy_score,
                "sell_score": sell_score,
                "hold_score": hold_score,
                "consensus_threshold": self.consensus_threshold,
                "valid_signals": len(signals),
                "total_generators": len(self.generators),
            },
        )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get average confidence from all generators"""
        self.validate_inputs(df, index)

        confidences = []
        total_weight = 0.0

        for generator, weight in self.generators.items():
            try:
                confidence = generator.get_confidence(df, index)
                if confidence >= self.min_confidence:
                    confidences.append(confidence * weight)
                    total_weight += weight
            except Exception:
                continue

        if not confidences or total_weight == 0:
            return 0.0

        return sum(confidences) / total_weight

    def get_parameters(self) -> dict[str, Any]:
        """Get weighted voting parameters"""
        params = super().get_parameters()
        params.update(
            {
                "min_confidence": self.min_confidence,
                "consensus_threshold": self.consensus_threshold,
                "generators": {gen.name: weight for gen, weight in self.generators.items()},
                "total_generators": len(self.generators),
            }
        )
        return params


class HierarchicalSignalGenerator(SignalGenerator):
    """
    Hierarchical signal generator

    Uses primary and secondary signal generators with fallback logic.
    Primary generator takes precedence, secondary provides confirmation or fallback.
    """

    def __init__(
        self,
        primary_generator: SignalGenerator,
        secondary_generator: SignalGenerator,
        confirmation_mode: bool = True,
        min_primary_confidence: float = 0.5,
    ):
        """
        Initialize hierarchical signal generator

        Args:
            primary_generator: Primary signal generator
            secondary_generator: Secondary signal generator for confirmation/fallback
            confirmation_mode: If True, secondary must confirm primary; if False, secondary is fallback
            min_primary_confidence: Minimum confidence required from primary generator
        """
        super().__init__("hierarchical_signal_generator")

        if primary_generator is None or secondary_generator is None:
            raise ValueError("Both primary and secondary generators must be provided")

        if not 0.0 <= min_primary_confidence <= 1.0:
            raise ValueError(
                f"min_primary_confidence must be between 0.0 and 1.0, got {min_primary_confidence}"
            )

        self.primary_generator = primary_generator
        self.secondary_generator = secondary_generator
        self.confirmation_mode = confirmation_mode
        self.min_primary_confidence = min_primary_confidence

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """Generate hierarchical signal with primary/secondary logic"""
        self.validate_inputs(df, index)

        # Get primary signal
        try:
            primary_signal = self.primary_generator.generate_signal(df, index, regime)
        except Exception as e:
            # Primary failed, use secondary as fallback
            try:
                secondary_signal = self.secondary_generator.generate_signal(df, index, regime)
                secondary_signal.metadata.update(
                    {
                        "hierarchical_generator": self.name,
                        "primary_failed": True,
                        "primary_error": str(e),
                    }
                )
                return secondary_signal
            except Exception as e2:
                # Both failed, return HOLD
                return Signal(
                    direction=SignalDirection.HOLD,
                    strength=0.0,
                    confidence=0.0,
                    metadata={
                        "generator": self.name,
                        "index": index,
                        "primary_error": str(e),
                        "secondary_error": str(e2),
                        "reason": "both_generators_failed",
                    },
                )

        # Check if primary signal meets confidence threshold
        if primary_signal.confidence < self.min_primary_confidence:
            # Primary confidence too low, use secondary
            try:
                secondary_signal = self.secondary_generator.generate_signal(df, index, regime)
                secondary_signal.metadata.update(
                    {
                        "hierarchical_generator": self.name,
                        "primary_low_confidence": True,
                        "primary_confidence": primary_signal.confidence,
                    }
                )
                return secondary_signal
            except Exception:
                # Secondary failed, return low-confidence primary
                primary_signal.metadata.update(
                    {
                        "hierarchical_generator": self.name,
                        "secondary_failed": True,
                        "low_confidence_primary": True,
                    }
                )
                return primary_signal

        # Primary signal has sufficient confidence
        if not self.confirmation_mode:
            # No confirmation needed, return primary
            primary_signal.metadata.update(
                {"hierarchical_generator": self.name, "mode": "primary_only"}
            )
            return primary_signal

        # Confirmation mode: get secondary signal for confirmation
        try:
            secondary_signal = self.secondary_generator.generate_signal(df, index, regime)
        except Exception:
            # Secondary failed, return primary anyway
            primary_signal.metadata.update(
                {
                    "hierarchical_generator": self.name,
                    "secondary_failed": True,
                    "mode": "primary_only",
                }
            )
            return primary_signal

        # Check for confirmation
        if primary_signal.direction == secondary_signal.direction:
            # Confirmed: combine confidence and strength
            combined_confidence = (primary_signal.confidence + secondary_signal.confidence) / 2
            combined_strength = max(primary_signal.strength, secondary_signal.strength)

            return Signal(
                direction=primary_signal.direction,
                strength=combined_strength,
                confidence=combined_confidence,
                metadata={
                    "generator": self.name,
                    "index": index,
                    "mode": "confirmed",
                    "primary_confidence": primary_signal.confidence,
                    "secondary_confidence": secondary_signal.confidence,
                    "primary_strength": primary_signal.strength,
                    "secondary_strength": secondary_signal.strength,
                },
            )
        else:
            # Not confirmed: return HOLD or weaker signal based on confidence
            if primary_signal.confidence > secondary_signal.confidence * 1.5:
                # Primary much stronger, use it but reduce confidence
                return Signal(
                    direction=primary_signal.direction,
                    strength=primary_signal.strength
                    * 0.7,  # Reduce strength due to lack of confirmation
                    confidence=primary_signal.confidence * 0.8,  # Reduce confidence
                    metadata={
                        "generator": self.name,
                        "index": index,
                        "mode": "unconfirmed_primary",
                        "primary_confidence": primary_signal.confidence,
                        "secondary_confidence": secondary_signal.confidence,
                    },
                )
            else:
                # Conflicting signals, return HOLD
                return Signal(
                    direction=SignalDirection.HOLD,
                    strength=0.0,
                    confidence=(primary_signal.confidence + secondary_signal.confidence)
                    / 4,  # Low confidence
                    metadata={
                        "generator": self.name,
                        "index": index,
                        "mode": "conflicting_signals",
                        "primary_direction": primary_signal.direction.value,
                        "secondary_direction": secondary_signal.direction.value,
                        "primary_confidence": primary_signal.confidence,
                        "secondary_confidence": secondary_signal.confidence,
                    },
                )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence based on primary generator with secondary fallback"""
        self.validate_inputs(df, index)

        try:
            primary_confidence = self.primary_generator.get_confidence(df, index)
            if primary_confidence >= self.min_primary_confidence:
                return primary_confidence
        except Exception:
            pass

        # Primary failed or low confidence, try secondary
        try:
            return self.secondary_generator.get_confidence(df, index)
        except Exception:
            return 0.0

    def get_parameters(self) -> dict[str, Any]:
        """Get hierarchical signal generator parameters"""
        params = super().get_parameters()
        params.update(
            {
                "primary_generator": self.primary_generator.name,
                "secondary_generator": self.secondary_generator.name,
                "confirmation_mode": self.confirmation_mode,
                "min_primary_confidence": self.min_primary_confidence,
            }
        )
        return params


class RegimeAdaptiveSignalGenerator(SignalGenerator):
    """
    Regime-adaptive signal generator

    Selects different signal generators based on detected market regimes,
    adapting strategy to current market conditions.
    """

    def __init__(
        self,
        regime_generators: dict[str, SignalGenerator],
        default_generator: SignalGenerator,
        confidence_adjustment: bool = True,
    ):
        """
        Initialize regime-adaptive signal generator

        Args:
            regime_generators: Dictionary mapping regime keys to signal generators
            default_generator: Default generator for unknown regimes
            confidence_adjustment: Whether to adjust confidence based on regime confidence
        """
        super().__init__("regime_adaptive_signal_generator")

        if not regime_generators:
            raise ValueError("At least one regime generator must be provided")

        if default_generator is None:
            raise ValueError("Default generator must be provided")

        self.regime_generators = regime_generators
        self.default_generator = default_generator
        self.confidence_adjustment = confidence_adjustment

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """Generate regime-adaptive signal"""
        self.validate_inputs(df, index)

        # Select appropriate generator based on regime
        generator = self._select_generator(regime)

        try:
            signal = generator.generate_signal(df, index, regime)
        except Exception as e:
            # Selected generator failed, try default
            try:
                signal = self.default_generator.generate_signal(df, index, regime)
                signal.metadata.update(
                    {
                        "regime_adaptive_generator": self.name,
                        "selected_generator_failed": True,
                        "selected_generator": generator.name,
                        "error": str(e),
                    }
                )
            except Exception as e2:
                # Default also failed, return HOLD
                return Signal(
                    direction=SignalDirection.HOLD,
                    strength=0.0,
                    confidence=0.0,
                    metadata={
                        "generator": self.name,
                        "index": index,
                        "selected_generator_error": str(e),
                        "default_generator_error": str(e2),
                        "reason": "all_generators_failed",
                    },
                )

        # Apply regime-based confidence adjustment
        if self.confidence_adjustment and regime is not None:
            regime_confidence_mult = self._get_regime_confidence_multiplier(regime)
            signal.confidence *= regime_confidence_mult
            signal.confidence = max(0.0, min(1.0, signal.confidence))  # Clamp to [0, 1]

        # Add regime metadata
        signal.metadata.update(
            {
                "regime_adaptive_generator": self.name,
                "selected_generator": generator.name,
                "regime_key": self._get_regime_key(regime),
                "regime_confidence": regime.confidence if regime else None,
            }
        )

        return signal

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence from selected generator"""
        self.validate_inputs(df, index)

        # Without regime context, use default generator
        try:
            return self.default_generator.get_confidence(df, index)
        except Exception:
            return 0.0

    def _select_generator(self, regime: Optional["RegimeContext"]) -> SignalGenerator:
        """Select appropriate generator based on regime"""
        if regime is None:
            return self.default_generator

        regime_key = self._get_regime_key(regime)
        return self.regime_generators.get(regime_key, self.default_generator)

    def _get_regime_key(self, regime: Optional["RegimeContext"]) -> str:
        """Get regime key for generator selection"""
        if regime is None:
            return "unknown"

        # Build regime key from trend and volatility
        trend_key = "unknown"
        if hasattr(regime, "trend"):
            if regime.trend.value == "trend_up":
                trend_key = "bull"
            elif regime.trend.value == "trend_down":
                trend_key = "bear"
            else:
                trend_key = "range"

        vol_key = "unknown"
        if hasattr(regime, "volatility"):
            vol_key = "low_vol" if regime.volatility.value == "low_vol" else "high_vol"

        return f"{trend_key}_{vol_key}"

    def _get_regime_confidence_multiplier(self, regime: "RegimeContext") -> float:
        """Get confidence multiplier based on regime confidence"""
        if not hasattr(regime, "confidence"):
            return 1.0

        # Scale confidence multiplier based on regime confidence
        # High regime confidence = higher signal confidence
        # Low regime confidence = lower signal confidence
        regime_conf = regime.confidence

        if regime_conf > 0.8:
            return 1.1  # Boost confidence in high-confidence regimes
        elif regime_conf > 0.6:
            return 1.0  # Normal confidence
        elif regime_conf > 0.4:
            return 0.9  # Slight reduction
        else:
            return 0.7  # Significant reduction in low-confidence regimes

    def get_parameters(self) -> dict[str, Any]:
        """Get regime-adaptive signal generator parameters"""
        params = super().get_parameters()
        params.update(
            {
                "regime_generators": {key: gen.name for key, gen in self.regime_generators.items()},
                "default_generator": self.default_generator.name,
                "confidence_adjustment": self.confidence_adjustment,
                "total_regime_generators": len(self.regime_generators),
            }
        )
        return params


class CircuitBreakerSignalGenerator(SignalGenerator):
    """
    Signal generator wrapper with circuit breaker protection.

    Protects against repeated failures in signal generation by implementing
    the circuit breaker pattern. After N consecutive failures, the circuit
    opens and returns safe fallback signals until recovery timeout.

    This prevents cascading failures and ensures the trading loop remains
    responsive even when a signal generator is experiencing persistent issues.
    """

    def __init__(
        self,
        wrapped_generator: SignalGenerator,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        fallback_generator: SignalGenerator | None = None,
    ):
        """Initialize circuit breaker wrapper.

        Args:
            wrapped_generator: Signal generator to protect.
            failure_threshold: Failures before opening circuit (default: 5).
            recovery_timeout: Seconds before testing recovery (default: 60s).
            fallback_generator: Optional fallback when circuit is open
                               (defaults to HoldSignalGenerator).
        """
        super().__init__(f"circuit_breaker_{wrapped_generator.name}")

        if wrapped_generator is None:
            raise ValueError("wrapped_generator cannot be None")

        self.wrapped_generator = wrapped_generator
        self.fallback_generator = fallback_generator or HoldSignalGenerator()

        self.circuit_breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=Exception,  # Catch all exceptions
            name=f"signal_gen_{wrapped_generator.name}",
        )

    def generate_signal(
        self, df: pd.DataFrame, index: int, regime: Optional["RegimeContext"] = None
    ) -> Signal:
        """Generate signal with circuit breaker protection.

        Returns:
            Signal from wrapped generator if circuit is closed,
            or fallback signal if circuit is open.
        """
        self.validate_inputs(df, index)

        try:
            # Try to call wrapped generator through circuit breaker
            return self.circuit_breaker.call(
                self.wrapped_generator.generate_signal, df, index, regime
            )
        except CircuitBreakerError as e:
            # Circuit is open, use fallback
            logger.warning(
                "Circuit breaker open for %s, using fallback: %s",
                self.wrapped_generator.name,
                str(e),
            )
            fallback_signal = self.fallback_generator.generate_signal(df, index, regime)
            fallback_signal.metadata.update(
                {
                    "circuit_breaker": "open",
                    "wrapped_generator": self.wrapped_generator.name,
                    "circuit_stats": self.circuit_breaker.get_stats(),
                }
            )
            return fallback_signal
        except Exception as e:
            # Unexpected error in fallback path
            logger.error(
                "Error in circuit breaker signal generation for %s: %s",
                self.wrapped_generator.name,
                e,
                exc_info=True,
            )
            # Return safe HOLD signal
            return Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={
                    "generator": self.name,
                    "index": index,
                    "error": str(e),
                    "reason": "circuit_breaker_fallback_failed",
                },
            )

    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Get confidence with circuit breaker protection."""
        self.validate_inputs(df, index)

        try:
            return self.circuit_breaker.call(self.wrapped_generator.get_confidence, df, index)
        except CircuitBreakerError:
            # Circuit is open, use fallback confidence
            return self.fallback_generator.get_confidence(df, index)
        except Exception:
            return 0.0

    def reset_circuit(self) -> None:
        """Manually reset the circuit breaker to CLOSED state."""
        self.circuit_breaker.reset()

    def get_circuit_stats(self) -> dict[str, Any]:
        """Get current circuit breaker statistics."""
        return self.circuit_breaker.get_stats()

    def get_parameters(self) -> dict[str, Any]:
        """Get circuit breaker signal generator parameters."""
        params = super().get_parameters()
        params.update(
            {
                "wrapped_generator": self.wrapped_generator.name,
                "fallback_generator": self.fallback_generator.name,
                "circuit_breaker_stats": self.circuit_breaker.get_stats(),
            }
        )
        return params

    @property
    def warmup_period(self) -> int:
        """Return warmup period from wrapped generator."""
        return self.wrapped_generator.warmup_period

    def get_feature_generators(self) -> Sequence["FeatureGeneratorSpec"]:
        """Return feature generators from wrapped generator."""
        return self.wrapped_generator.get_feature_generators()
