"""
Unit tests for SignalGenerator components
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.components.signal_generator import (
    HierarchicalSignalGenerator,
    HoldSignalGenerator,
    RandomSignalGenerator,
    RegimeAdaptiveSignalGenerator,
    Signal,
    SignalDirection,
    SignalGenerator,
    WeightedVotingSignalGenerator,
)


class TestSignal:
    """Test Signal dataclass"""

    def test_signal_creation_valid(self):
        """Test creating a valid signal"""
        signal = Signal(
            direction=SignalDirection.BUY, strength=0.8, confidence=0.9, metadata={"test": "data"}
        )

        assert signal.direction == SignalDirection.BUY
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert signal.metadata == {"test": "data"}

    def test_signal_validation_direction(self):
        """Test signal direction validation"""
        with pytest.raises(ValueError, match="direction must be a SignalDirection enum"):
            Signal(direction="invalid", strength=0.5, confidence=0.5, metadata={})

    def test_signal_validation_strength_bounds(self):
        """Test signal strength bounds validation"""
        # Test negative strength
        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            Signal(direction=SignalDirection.BUY, strength=-0.1, confidence=0.5, metadata={})

        # Test strength > 1.0
        with pytest.raises(ValueError, match="strength must be between 0.0 and 1.0"):
            Signal(direction=SignalDirection.BUY, strength=1.1, confidence=0.5, metadata={})

    def test_signal_validation_confidence_bounds(self):
        """Test signal confidence bounds validation"""
        # Test negative confidence
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            Signal(direction=SignalDirection.BUY, strength=0.5, confidence=-0.1, metadata={})

        # Test confidence > 1.0
        with pytest.raises(ValueError, match="confidence must be between 0.0 and 1.0"):
            Signal(direction=SignalDirection.BUY, strength=0.5, confidence=1.1, metadata={})

    def test_signal_validation_metadata_type(self):
        """Test signal metadata type validation"""
        with pytest.raises(ValueError, match="metadata must be a dictionary"):
            Signal(direction=SignalDirection.BUY, strength=0.5, confidence=0.5, metadata="invalid")


class TestSignalDirection:
    """Test SignalDirection enum"""

    def test_signal_direction_values(self):
        """Test signal direction enum values"""
        assert SignalDirection.BUY.value == "buy"
        assert SignalDirection.SELL.value == "sell"
        assert SignalDirection.HOLD.value == "hold"


class MockSignalGenerator(SignalGenerator):
    """Mock signal generator for testing abstract base class"""

    def generate_signal(self, df, index, regime=None):
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=1.0,
            metadata={"generator": self.name},
        )

    def get_confidence(self, df, index):
        return 1.0


class TestSignalGenerator:
    """Test SignalGenerator abstract base class"""

    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")
        data = {
            "open": np.random.uniform(100, 110, 100),
            "high": np.random.uniform(110, 120, 100),
            "low": np.random.uniform(90, 100, 100),
            "close": np.random.uniform(100, 110, 100),
            "volume": np.random.uniform(1000, 10000, 100),
        }
        return pd.DataFrame(data, index=dates)

    def test_signal_generator_initialization(self):
        """Test signal generator initialization"""
        generator = MockSignalGenerator("test_generator")
        assert generator.name == "test_generator"

    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs"""
        generator = MockSignalGenerator("test")
        df = self.create_test_dataframe()

        # Should not raise any exception
        generator.validate_inputs(df, 50)

    def test_validate_inputs_empty_dataframe(self):
        """Test input validation with empty DataFrame"""
        generator = MockSignalGenerator("test")
        df = pd.DataFrame()

        with pytest.raises(ValueError, match="DataFrame cannot be empty"):
            generator.validate_inputs(df, 0)

    def test_validate_inputs_index_out_of_bounds(self):
        """Test input validation with index out of bounds"""
        generator = MockSignalGenerator("test")
        df = self.create_test_dataframe()

        # Test negative index
        with pytest.raises(IndexError, match="Index -1 is out of bounds"):
            generator.validate_inputs(df, -1)

        # Test index >= length
        with pytest.raises(IndexError, match="Index 100 is out of bounds"):
            generator.validate_inputs(df, 100)

    def test_validate_inputs_missing_columns(self):
        """Test input validation with missing required columns"""
        generator = MockSignalGenerator("test")
        df = pd.DataFrame({"price": [100, 101, 102]})

        with pytest.raises(ValueError, match="DataFrame missing required columns"):
            generator.validate_inputs(df, 0)

    def test_get_parameters(self):
        """Test get_parameters method"""
        generator = MockSignalGenerator("test_gen")
        params = generator.get_parameters()

        assert params["name"] == "test_gen"
        assert params["type"] == "MockSignalGenerator"


class TestHoldSignalGenerator:
    """Test HoldSignalGenerator implementation"""

    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=10, freq="1H")
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
        return pd.DataFrame(data, index=dates)

    def test_hold_signal_generator_initialization(self):
        """Test HoldSignalGenerator initialization"""
        generator = HoldSignalGenerator()
        assert generator.name == "hold_signal_generator"

    def test_generate_signal_always_hold(self):
        """Test that HoldSignalGenerator always generates HOLD signals"""
        generator = HoldSignalGenerator()
        df = self.create_test_dataframe()

        for i in range(len(df)):
            signal = generator.generate_signal(df, i)

            assert signal.direction == SignalDirection.HOLD
            assert signal.strength == 0.0
            assert signal.confidence == 1.0
            assert signal.metadata["generator"] == "hold_signal_generator"
            assert signal.metadata["index"] == i

    def test_get_confidence_always_high(self):
        """Test that HoldSignalGenerator always returns high confidence"""
        generator = HoldSignalGenerator()
        df = self.create_test_dataframe()

        for i in range(len(df)):
            confidence = generator.get_confidence(df, i)
            assert confidence == 1.0

    def test_generate_signal_with_regime(self):
        """Test signal generation with regime context"""
        from src.regime.detector import TrendLabel, VolLabel
        from src.strategies.components.regime_context import RegimeContext

        generator = HoldSignalGenerator()
        df = self.create_test_dataframe()

        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.LOW,
            confidence=0.8,
            duration=10,
            strength=0.7,
        )

        signal = generator.generate_signal(df, 5, regime)

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["regime"] == TrendLabel.TREND_UP.value


class TestRandomSignalGenerator:
    """Test RandomSignalGenerator implementation"""

    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=10, freq="1H")
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
        return pd.DataFrame(data, index=dates)

    def test_random_signal_generator_initialization_default(self):
        """Test RandomSignalGenerator initialization with defaults"""
        generator = RandomSignalGenerator()
        assert generator.name == "random_signal_generator"
        assert generator.buy_prob == 0.3
        assert generator.sell_prob == 0.3
        assert abs(generator.hold_prob - 0.4) < 1e-10

    def test_random_signal_generator_initialization_custom(self):
        """Test RandomSignalGenerator initialization with custom probabilities"""
        generator = RandomSignalGenerator(buy_prob=0.4, sell_prob=0.2)
        assert generator.buy_prob == 0.4
        assert generator.sell_prob == 0.2
        assert abs(generator.hold_prob - 0.4) < 1e-10

    def test_random_signal_generator_validation_buy_prob(self):
        """Test buy_prob validation"""
        with pytest.raises(ValueError, match="buy_prob must be between 0.0 and 1.0"):
            RandomSignalGenerator(buy_prob=-0.1)

        with pytest.raises(ValueError, match="buy_prob must be between 0.0 and 1.0"):
            RandomSignalGenerator(buy_prob=1.1)

    def test_random_signal_generator_validation_sell_prob(self):
        """Test sell_prob validation"""
        with pytest.raises(ValueError, match="sell_prob must be between 0.0 and 1.0"):
            RandomSignalGenerator(sell_prob=-0.1)

        with pytest.raises(ValueError, match="sell_prob must be between 0.0 and 1.0"):
            RandomSignalGenerator(sell_prob=1.1)

    def test_random_signal_generator_validation_prob_sum(self):
        """Test that buy_prob + sell_prob cannot exceed 1.0"""
        with pytest.raises(ValueError, match="buy_prob \\+ sell_prob cannot exceed 1.0"):
            RandomSignalGenerator(buy_prob=0.6, sell_prob=0.6)

    def test_generate_signal_reproducible_with_seed(self):
        """Test that signals are reproducible with seed"""
        df = self.create_test_dataframe()

        # Generate signal with first generator
        generator1 = RandomSignalGenerator(seed=42)
        signal1 = generator1.generate_signal(df, 5)

        # Generate signal with second generator (same seed)
        generator2 = RandomSignalGenerator(seed=42)
        signal2 = generator2.generate_signal(df, 5)

        assert signal1.direction == signal2.direction
        # Note: strength and confidence may vary due to separate random calls
        # but direction should be consistent with same seed

    def test_generate_signal_properties(self):
        """Test signal generation properties"""
        generator = RandomSignalGenerator(seed=42)
        df = self.create_test_dataframe()

        signal = generator.generate_signal(df, 5)

        # Check signal properties
        assert isinstance(signal.direction, SignalDirection)
        assert 0.0 <= signal.strength <= 1.0
        assert 0.3 <= signal.confidence <= 0.9  # Based on implementation
        assert signal.metadata["generator"] == "random_signal_generator"
        assert signal.metadata["index"] == 5

    def test_generate_signal_hold_has_zero_strength(self):
        """Test that HOLD signals have zero strength"""
        # Use probabilities that guarantee HOLD
        generator = RandomSignalGenerator(buy_prob=0.0, sell_prob=0.0, seed=42)
        df = self.create_test_dataframe()

        signal = generator.generate_signal(df, 5)

        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0

    def test_get_confidence_range(self):
        """Test confidence score range"""
        generator = RandomSignalGenerator(seed=42)
        df = self.create_test_dataframe()

        confidence = generator.get_confidence(df, 5)

        assert 0.3 <= confidence <= 0.9

    def test_get_parameters(self):
        """Test get_parameters method"""
        generator = RandomSignalGenerator(buy_prob=0.4, sell_prob=0.2)
        params = generator.get_parameters()

        assert params["name"] == "random_signal_generator"
        assert params["type"] == "RandomSignalGenerator"
        assert params["buy_prob"] == 0.4
        assert params["sell_prob"] == 0.2
        assert abs(params["hold_prob"] - 0.4) < 1e-10


class TestWeightedVotingSignalGenerator:
    """Test WeightedVotingSignalGenerator implementation"""

    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=10, freq="1H")
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
        return pd.DataFrame(data, index=dates)

    def create_mock_generator(self, name, direction, strength=0.8, confidence=0.9):
        """Create mock generator that returns specific signal"""

        class MockGen(SignalGenerator):
            def __init__(self, name, direction, strength, confidence):
                super().__init__(name)
                self.direction = direction
                self.strength = strength
                self.confidence = confidence

            def generate_signal(self, df, index, regime=None):
                return Signal(
                    direction=self.direction,
                    strength=self.strength,
                    confidence=self.confidence,
                    metadata={"generator": self.name},
                )

            def get_confidence(self, df, index):
                return self.confidence

        return MockGen(name, direction, strength, confidence)

    def test_weighted_voting_initialization_valid(self):
        """Test WeightedVotingSignalGenerator initialization with valid inputs"""
        gen1 = self.create_mock_generator("gen1", SignalDirection.BUY)
        gen2 = self.create_mock_generator("gen2", SignalDirection.SELL)

        generators = {gen1: 0.6, gen2: 0.4}
        voting_gen = WeightedVotingSignalGenerator(generators)

        assert voting_gen.name == "weighted_voting_signal_generator"
        assert len(voting_gen.generators) == 2
        # Weights should be normalized
        assert abs(sum(voting_gen.generators.values()) - 1.0) < 1e-10

    def test_weighted_voting_initialization_empty_generators(self):
        """Test initialization with empty generators"""
        with pytest.raises(ValueError, match="At least one signal generator must be provided"):
            WeightedVotingSignalGenerator({})

    def test_weighted_voting_initialization_zero_weights(self):
        """Test initialization with zero total weight"""
        gen1 = self.create_mock_generator("gen1", SignalDirection.BUY)
        generators = {gen1: 0.0}

        with pytest.raises(ValueError, match="Total weight must be positive"):
            WeightedVotingSignalGenerator(generators)

    def test_weighted_voting_consensus_buy(self):
        """Test weighted voting with BUY consensus"""
        gen1 = self.create_mock_generator("gen1", SignalDirection.BUY, 0.8, 0.9)
        gen2 = self.create_mock_generator("gen2", SignalDirection.BUY, 0.7, 0.8)
        gen3 = self.create_mock_generator("gen3", SignalDirection.SELL, 0.5, 0.6)

        generators = {gen1: 0.5, gen2: 0.3, gen3: 0.2}
        voting_gen = WeightedVotingSignalGenerator(generators, consensus_threshold=0.6)

        df = self.create_test_dataframe()
        signal = voting_gen.generate_signal(df, 5)

        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0.6  # Should exceed consensus threshold
        assert 0.0 < signal.confidence <= 1.0

    def test_weighted_voting_no_consensus(self):
        """Test weighted voting with no consensus"""
        gen1 = self.create_mock_generator("gen1", SignalDirection.BUY, 0.5, 0.8)
        gen2 = self.create_mock_generator("gen2", SignalDirection.SELL, 0.5, 0.8)
        gen3 = self.create_mock_generator("gen3", SignalDirection.HOLD, 0.0, 0.9)

        generators = {gen1: 0.33, gen2: 0.33, gen3: 0.34}
        voting_gen = WeightedVotingSignalGenerator(generators, consensus_threshold=0.6)

        df = self.create_test_dataframe()
        signal = voting_gen.generate_signal(df, 5)

        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0

    def test_weighted_voting_low_confidence_filter(self):
        """Test filtering of low confidence signals"""
        gen1 = self.create_mock_generator("gen1", SignalDirection.BUY, 0.8, 0.2)  # Low confidence
        gen2 = self.create_mock_generator("gen2", SignalDirection.SELL, 0.7, 0.8)  # High confidence

        generators = {gen1: 0.5, gen2: 0.5}
        voting_gen = WeightedVotingSignalGenerator(generators, min_confidence=0.3)

        df = self.create_test_dataframe()
        signal = voting_gen.generate_signal(df, 5)

        # Should only use gen2 since gen1 has low confidence
        assert signal.direction == SignalDirection.SELL

    def test_weighted_voting_all_generators_fail(self):
        """Test behavior when all generators fail"""

        class FailingGenerator(SignalGenerator):
            def __init__(self, name):
                super().__init__(name)

            def generate_signal(self, df, index, regime=None):
                raise Exception("Generator failed")

            def get_confidence(self, df, index):
                return 0.5

        gen1 = FailingGenerator("failing1")
        gen2 = FailingGenerator("failing2")

        generators = {gen1: 0.5, gen2: 0.5}
        voting_gen = WeightedVotingSignalGenerator(generators)

        df = self.create_test_dataframe()
        signal = voting_gen.generate_signal(df, 5)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "no_valid_signals"

    def test_get_confidence_average(self):
        """Test confidence calculation as weighted average"""
        gen1 = self.create_mock_generator("gen1", SignalDirection.BUY, 0.8, 0.9)
        gen2 = self.create_mock_generator("gen2", SignalDirection.SELL, 0.7, 0.7)

        generators = {gen1: 0.6, gen2: 0.4}
        voting_gen = WeightedVotingSignalGenerator(generators)

        df = self.create_test_dataframe()
        confidence = voting_gen.get_confidence(df, 5)

        # Should be weighted average: (0.9 * 0.6 + 0.7 * 0.4) / 1.0 = 0.82
        expected_confidence = 0.82
        assert abs(confidence - expected_confidence) < 0.01


class TestHierarchicalSignalGenerator:
    """Test HierarchicalSignalGenerator implementation"""

    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=10, freq="1H")
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
        return pd.DataFrame(data, index=dates)

    def create_mock_generator(self, name, direction, strength=0.8, confidence=0.9):
        """Create mock generator that returns specific signal"""

        class MockGen(SignalGenerator):
            def __init__(self, name, direction, strength, confidence):
                super().__init__(name)
                self.direction = direction
                self.strength = strength
                self.confidence = confidence

            def generate_signal(self, df, index, regime=None):
                return Signal(
                    direction=self.direction,
                    strength=self.strength,
                    confidence=self.confidence,
                    metadata={"generator": self.name},
                )

            def get_confidence(self, df, index):
                return self.confidence

        return MockGen(name, direction, strength, confidence)

    def test_hierarchical_initialization_valid(self):
        """Test HierarchicalSignalGenerator initialization with valid inputs"""
        primary = self.create_mock_generator("primary", SignalDirection.BUY)
        secondary = self.create_mock_generator("secondary", SignalDirection.SELL)

        hierarchical = HierarchicalSignalGenerator(primary, secondary)

        assert hierarchical.name == "hierarchical_signal_generator"
        assert hierarchical.primary_generator == primary
        assert hierarchical.secondary_generator == secondary
        assert hierarchical.confirmation_mode is True
        assert hierarchical.min_primary_confidence == 0.5

    def test_hierarchical_initialization_none_generators(self):
        """Test initialization with None generators"""
        with pytest.raises(
            ValueError, match="Both primary and secondary generators must be provided"
        ):
            HierarchicalSignalGenerator(None, None)

    def test_hierarchical_primary_high_confidence_no_confirmation(self):
        """Test primary signal with high confidence, no confirmation mode"""
        primary = self.create_mock_generator("primary", SignalDirection.BUY, 0.8, 0.9)
        secondary = self.create_mock_generator("secondary", SignalDirection.SELL, 0.7, 0.8)

        hierarchical = HierarchicalSignalGenerator(
            primary, secondary, confirmation_mode=False, min_primary_confidence=0.5
        )

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        assert signal.direction == SignalDirection.BUY
        assert signal.strength == 0.8
        assert signal.confidence == 0.9
        assert signal.metadata["mode"] == "primary_only"

    def test_hierarchical_primary_confirmed(self):
        """Test primary signal confirmed by secondary"""
        primary = self.create_mock_generator("primary", SignalDirection.BUY, 0.8, 0.9)
        secondary = self.create_mock_generator("secondary", SignalDirection.BUY, 0.7, 0.8)

        hierarchical = HierarchicalSignalGenerator(
            primary, secondary, confirmation_mode=True, min_primary_confidence=0.5
        )

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        assert signal.direction == SignalDirection.BUY
        assert signal.strength == 0.8  # Max of primary and secondary
        assert abs(signal.confidence - 0.85) < 0.01  # Average of primary and secondary
        assert signal.metadata["mode"] == "confirmed"

    def test_hierarchical_primary_not_confirmed(self):
        """Test primary signal not confirmed by secondary"""
        primary = self.create_mock_generator("primary", SignalDirection.BUY, 0.8, 0.9)
        secondary = self.create_mock_generator("secondary", SignalDirection.SELL, 0.7, 0.6)

        hierarchical = HierarchicalSignalGenerator(
            primary, secondary, confirmation_mode=True, min_primary_confidence=0.5
        )

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        # Primary much stronger (0.9 > 0.6 * 1.5), so use reduced primary
        assert signal.direction == SignalDirection.BUY
        assert signal.strength == 0.8 * 0.7  # Reduced strength
        assert signal.confidence == 0.9 * 0.8  # Reduced confidence
        assert signal.metadata["mode"] == "unconfirmed_primary"

    def test_hierarchical_conflicting_signals_hold(self):
        """Test conflicting signals result in HOLD"""
        primary = self.create_mock_generator("primary", SignalDirection.BUY, 0.8, 0.7)
        secondary = self.create_mock_generator("secondary", SignalDirection.SELL, 0.7, 0.8)

        hierarchical = HierarchicalSignalGenerator(
            primary, secondary, confirmation_mode=True, min_primary_confidence=0.5
        )

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        # Primary not much stronger than secondary, so conflicting signals = HOLD
        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0.0
        assert signal.metadata["mode"] == "conflicting_signals"

    def test_hierarchical_primary_low_confidence_fallback(self):
        """Test fallback to secondary when primary has low confidence"""
        primary = self.create_mock_generator(
            "primary", SignalDirection.BUY, 0.8, 0.3
        )  # Low confidence
        secondary = self.create_mock_generator("secondary", SignalDirection.SELL, 0.7, 0.8)

        hierarchical = HierarchicalSignalGenerator(
            primary, secondary, confirmation_mode=True, min_primary_confidence=0.5
        )

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["primary_low_confidence"] is True

    def test_hierarchical_primary_fails_fallback(self):
        """Test fallback to secondary when primary fails"""

        class FailingGenerator(SignalGenerator):
            def __init__(self, name):
                super().__init__(name)

            def generate_signal(self, df, index, regime=None):
                raise Exception("Primary failed")

            def get_confidence(self, df, index):
                return 0.5

        primary = FailingGenerator("failing_primary")
        secondary = self.create_mock_generator("secondary", SignalDirection.SELL, 0.7, 0.8)

        hierarchical = HierarchicalSignalGenerator(primary, secondary)

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["primary_failed"] is True

    def test_hierarchical_both_fail(self):
        """Test behavior when both generators fail"""

        class FailingGenerator(SignalGenerator):
            def __init__(self, name):
                super().__init__(name)

            def generate_signal(self, df, index, regime=None):
                raise Exception(f"{name} failed")

            def get_confidence(self, df, index):
                return 0.5

        primary = FailingGenerator("failing_primary")
        secondary = FailingGenerator("failing_secondary")

        hierarchical = HierarchicalSignalGenerator(primary, secondary)

        df = self.create_test_dataframe()
        signal = hierarchical.generate_signal(df, 5)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "both_generators_failed"


class TestRegimeAdaptiveSignalGenerator:
    """Test RegimeAdaptiveSignalGenerator implementation"""

    def create_test_dataframe(self):
        """Create test DataFrame with OHLCV data"""
        dates = pd.date_range("2023-01-01", periods=10, freq="1H")
        data = {
            "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        }
        return pd.DataFrame(data, index=dates)

    def create_mock_generator(self, name, direction, strength=0.8, confidence=0.9):
        """Create mock generator that returns specific signal"""

        class MockGen(SignalGenerator):
            def __init__(self, name, direction, strength, confidence):
                super().__init__(name)
                self.direction = direction
                self.strength = strength
                self.confidence = confidence

            def generate_signal(self, df, index, regime=None):
                return Signal(
                    direction=self.direction,
                    strength=self.strength,
                    confidence=self.confidence,
                    metadata={"generator": self.name},
                )

            def get_confidence(self, df, index):
                return self.confidence

        return MockGen(name, direction, strength, confidence)

    def create_test_regime(self, trend="trend_up", volatility="low_vol", confidence=0.8):
        """Create test regime context"""
        from src.regime.detector import TrendLabel, VolLabel
        from src.strategies.components.regime_context import RegimeContext

        trend_map = {
            "trend_up": TrendLabel.TREND_UP,
            "trend_down": TrendLabel.TREND_DOWN,
            "range": TrendLabel.RANGE,
        }

        vol_map = {"low_vol": VolLabel.LOW, "high_vol": VolLabel.HIGH}

        return RegimeContext(
            trend=trend_map.get(trend, TrendLabel.RANGE),
            volatility=vol_map.get(volatility, VolLabel.LOW),
            confidence=confidence,
            duration=10,
            strength=0.7,
        )

    def test_regime_adaptive_initialization_valid(self):
        """Test RegimeAdaptiveSignalGenerator initialization with valid inputs"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY)
        bear_gen = self.create_mock_generator("bear", SignalDirection.SELL)
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD)

        regime_generators = {"bull_low_vol": bull_gen, "bear_high_vol": bear_gen}

        adaptive = RegimeAdaptiveSignalGenerator(regime_generators, default_gen)

        assert adaptive.name == "regime_adaptive_signal_generator"
        assert len(adaptive.regime_generators) == 2
        assert adaptive.default_generator == default_gen
        assert adaptive.confidence_adjustment is True

    def test_regime_adaptive_initialization_empty_generators(self):
        """Test initialization with empty regime generators"""
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD)

        with pytest.raises(ValueError, match="At least one regime generator must be provided"):
            RegimeAdaptiveSignalGenerator({}, default_gen)

    def test_regime_adaptive_initialization_none_default(self):
        """Test initialization with None default generator"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY)
        regime_generators = {"bull_low_vol": bull_gen}

        with pytest.raises(ValueError, match="Default generator must be provided"):
            RegimeAdaptiveSignalGenerator(regime_generators, None)

    def test_regime_adaptive_bull_low_vol_selection(self):
        """Test generator selection for bull low volatility regime"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY, 0.8, 0.9)
        bear_gen = self.create_mock_generator("bear", SignalDirection.SELL, 0.7, 0.8)
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD, 0.0, 0.5)

        regime_generators = {"bull_low_vol": bull_gen, "bear_high_vol": bear_gen}

        adaptive = RegimeAdaptiveSignalGenerator(regime_generators, default_gen)
        regime = self.create_test_regime("trend_up", "low_vol", 0.8)

        df = self.create_test_dataframe()
        signal = adaptive.generate_signal(df, 5, regime)

        assert signal.direction == SignalDirection.BUY
        assert signal.metadata["selected_generator"] == "bull"
        assert signal.metadata["regime_key"] == "bull_low_vol"

    def test_regime_adaptive_unknown_regime_default(self):
        """Test default generator selection for unknown regime"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY, 0.8, 0.9)
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD, 0.0, 0.8)

        regime_generators = {"bull_low_vol": bull_gen}
        adaptive = RegimeAdaptiveSignalGenerator(regime_generators, default_gen)

        # No regime provided
        df = self.create_test_dataframe()
        signal = adaptive.generate_signal(df, 5, None)

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["selected_generator"] == "default"
        assert signal.metadata["regime_key"] == "unknown"

    def test_regime_adaptive_confidence_adjustment(self):
        """Test confidence adjustment based on regime confidence"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY, 0.8, 0.8)
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD, 0.0, 0.5)

        regime_generators = {"bull_low_vol": bull_gen}
        adaptive = RegimeAdaptiveSignalGenerator(
            regime_generators, default_gen, confidence_adjustment=True
        )

        # High regime confidence should boost signal confidence
        regime = self.create_test_regime("trend_up", "low_vol", 0.9)

        df = self.create_test_dataframe()
        signal = adaptive.generate_signal(df, 5, regime)

        # Original confidence 0.8, regime confidence 0.9 > 0.8, so multiplier = 1.1
        expected_confidence = min(1.0, 0.8 * 1.1)
        assert abs(signal.confidence - expected_confidence) < 0.01

    def test_regime_adaptive_confidence_adjustment_disabled(self):
        """Test no confidence adjustment when disabled"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY, 0.8, 0.8)
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD, 0.0, 0.5)

        regime_generators = {"bull_low_vol": bull_gen}
        adaptive = RegimeAdaptiveSignalGenerator(
            regime_generators, default_gen, confidence_adjustment=False
        )

        regime = self.create_test_regime("trend_up", "low_vol", 0.9)

        df = self.create_test_dataframe()
        signal = adaptive.generate_signal(df, 5, regime)

        # Confidence should remain unchanged
        assert signal.confidence == 0.8

    def test_regime_adaptive_generator_fails_fallback(self):
        """Test fallback to default when selected generator fails"""

        class FailingGenerator(SignalGenerator):
            def __init__(self, name):
                super().__init__(name)

            def generate_signal(self, df, index, regime=None):
                raise Exception("Generator failed")

            def get_confidence(self, df, index):
                return 0.5

        failing_gen = FailingGenerator("failing")
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD, 0.0, 0.8)

        regime_generators = {"bull_low_vol": failing_gen}
        adaptive = RegimeAdaptiveSignalGenerator(regime_generators, default_gen)

        regime = self.create_test_regime("trend_up", "low_vol", 0.8)

        df = self.create_test_dataframe()
        signal = adaptive.generate_signal(df, 5, regime)

        assert signal.direction == SignalDirection.HOLD
        assert signal.metadata["selected_generator_failed"] is True
        assert signal.metadata["selected_generator"] == "failing"

    def test_regime_adaptive_all_generators_fail(self):
        """Test behavior when all generators fail"""

        class FailingGenerator(SignalGenerator):
            def __init__(self, name):
                super().__init__(name)

            def generate_signal(self, df, index, regime=None):
                raise Exception(f"{name} failed")

            def get_confidence(self, df, index):
                return 0.5

        failing_gen = FailingGenerator("failing")
        failing_default = FailingGenerator("failing_default")

        regime_generators = {"bull_low_vol": failing_gen}
        adaptive = RegimeAdaptiveSignalGenerator(regime_generators, failing_default)

        regime = self.create_test_regime("trend_up", "low_vol", 0.8)

        df = self.create_test_dataframe()
        signal = adaptive.generate_signal(df, 5, regime)

        assert signal.direction == SignalDirection.HOLD
        assert signal.confidence == 0.0
        assert signal.metadata["reason"] == "all_generators_failed"

    def test_get_parameters(self):
        """Test get_parameters method"""
        bull_gen = self.create_mock_generator("bull", SignalDirection.BUY)
        bear_gen = self.create_mock_generator("bear", SignalDirection.SELL)
        default_gen = self.create_mock_generator("default", SignalDirection.HOLD)

        regime_generators = {"bull_low_vol": bull_gen, "bear_high_vol": bear_gen}

        adaptive = RegimeAdaptiveSignalGenerator(
            regime_generators, default_gen, confidence_adjustment=False
        )

        params = adaptive.get_parameters()

        assert params["name"] == "regime_adaptive_signal_generator"
        assert params["type"] == "RegimeAdaptiveSignalGenerator"
        assert params["default_generator"] == "default"
        assert params["confidence_adjustment"] is False
        assert params["total_regime_generators"] == 2
        assert "bull_low_vol" in params["regime_generators"]
        assert "bear_high_vol" in params["regime_generators"]
