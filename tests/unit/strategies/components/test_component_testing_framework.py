"""
Tests for Component Testing Framework

This module tests the comprehensive component testing framework including
ComponentPerformanceTester, RegimeTester, PerformanceAttributionAnalyzer,
and TestDatasetGenerator.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.strategies.components.testing import (
    ComponentPerformanceTester,
    RegimeTester,
    PerformanceAttributionAnalyzer,
    TestDatasetGenerator,
    MarketScenario,
)
from src.strategies.components.signal_generator import (
    HoldSignalGenerator,
    RandomSignalGenerator,
    SignalDirection,
)
from src.strategies.components.risk_manager import FixedRiskManager
from src.strategies.components.position_sizer import FixedFractionSizer


@pytest.fixture
def sample_test_data():
    """Create sample test data for testing"""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

    # Generate synthetic price data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 200)  # 0.1% daily return, 2% volatility
    prices = 100 * np.exp(np.cumsum(returns))

    # Create OHLCV data
    data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.001, 200)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, 200))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, 200))),
            "close": prices,
            "volume": np.random.lognormal(15, 0.5, 200),  # Log-normal volume
        },
        index=dates,
    )

    # Ensure OHLC consistency
    data["high"] = np.maximum.reduce([data["open"], data["high"], data["low"], data["close"]])
    data["low"] = np.minimum.reduce([data["open"], data["high"], data["low"], data["close"]])

    return data


@pytest.fixture
def sample_regime_data():
    """Create sample regime data for testing"""
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")

    # Create simple regime pattern
    regime_data = pd.DataFrame(
        {
            "trend": ["trend_up"] * 100 + ["trend_down"] * 100,
            "volatility": ["low_vol"] * 50 + ["high_vol"] * 100 + ["low_vol"] * 50,
            "confidence": np.random.uniform(0.6, 0.9, 200),
            "duration": list(range(1, 101)) + list(range(1, 101)),
            "strength": np.random.uniform(0.5, 0.8, 200),
            "regime_type": ["trend_up_low_vol"] * 50
            + ["trend_up_high_vol"] * 50
            + ["trend_down_high_vol"] * 50
            + ["trend_down_low_vol"] * 50,
        },
        index=dates,
    )

    return regime_data


class TestComponentPerformanceTester:
    """Test ComponentPerformanceTester functionality"""

    def test_initialization(self, sample_test_data):
        """Test ComponentPerformanceTester initialization"""
        tester = ComponentPerformanceTester(sample_test_data)

        assert len(tester.test_data) > 0
        assert "sma_20" in tester.test_data.columns
        assert "returns" in tester.test_data.columns
        assert len(tester.test_scenarios) > 0

    def test_signal_generator_testing(self, sample_test_data):
        """Test signal generator performance testing"""
        tester = ComponentPerformanceTester(sample_test_data)

        # Test with HoldSignalGenerator
        hold_generator = HoldSignalGenerator()
        results = tester.test_signal_generator(hold_generator)

        assert results.component_name == "hold_signal_generator"
        assert results.total_signals > 0
        assert 0 <= results.accuracy <= 1
        assert results.hold_signals == results.total_signals  # All signals should be HOLD
        assert results.error_count >= 0

    def test_random_signal_generator_testing(self, sample_test_data):
        """Test with RandomSignalGenerator"""
        tester = ComponentPerformanceTester(sample_test_data)

        # Test with RandomSignalGenerator
        random_generator = RandomSignalGenerator(buy_prob=0.3, sell_prob=0.3, seed=42)
        results = tester.test_signal_generator(random_generator)

        assert results.component_name == "random_signal_generator"
        assert results.total_signals > 0
        assert results.buy_signals > 0
        assert results.sell_signals > 0
        assert results.hold_signals > 0
        assert (
            results.buy_signals + results.sell_signals + results.hold_signals
            == results.total_signals
        )

    def test_risk_manager_testing(self, sample_test_data):
        """Test risk manager performance testing"""
        tester = ComponentPerformanceTester(sample_test_data)

        # Test with FixedRiskManager
        risk_manager = FixedRiskManager(risk_per_trade=0.02, stop_loss_pct=0.05)
        results = tester.test_risk_manager(risk_manager)

        assert results.component_name == "fixed_risk_manager"
        assert results.total_scenarios > 0
        assert results.average_position_size > 0
        assert 0 <= results.drawdown_control_score <= 1
        assert results.error_count >= 0

    def test_position_sizer_testing(self, sample_test_data):
        """Test position sizer performance testing"""
        tester = ComponentPerformanceTester(sample_test_data)

        # Test with FixedFractionSizer
        position_sizer = FixedFractionSizer(fraction=0.02)
        results = tester.test_position_sizer(position_sizer)

        assert results.component_name == "fixed_fraction_sizer"
        assert results.total_calculations > 0
        assert results.average_position_size > 0
        assert 0 <= results.bounds_adherence_rate <= 1
        assert results.error_count >= 0

    def test_all_components_testing(self, sample_test_data):
        """Test testing all components together"""
        tester = ComponentPerformanceTester(sample_test_data)

        # Create components
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager()
        position_sizer = FixedFractionSizer()

        # Test all components
        results = tester.test_all_components(
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
        )

        assert results.signal_results is not None
        assert results.risk_results is not None
        assert results.sizing_results is not None
        assert results.total_test_duration > 0
        assert 0 <= results.overall_performance_score <= 1


class TestRegimeTester:
    """Test RegimeTester functionality"""

    def test_initialization(self, sample_test_data):
        """Test RegimeTester initialization"""
        tester = RegimeTester(sample_test_data)

        assert len(tester.test_data) > 0
        assert len(tester.regime_data) > 0
        assert len(tester.regime_datasets) > 0
        assert "regime_type" in tester.regime_data.columns

    def test_regime_detection(self, sample_test_data):
        """Test regime detection functionality"""
        tester = RegimeTester(sample_test_data)

        regime_data = tester.regime_data

        assert "trend" in regime_data.columns
        assert "volatility" in regime_data.columns
        assert "confidence" in regime_data.columns
        assert "duration" in regime_data.columns
        assert "strength" in regime_data.columns

        # Check that all values are within expected ranges
        assert regime_data["confidence"].between(0, 1).all()
        assert regime_data["duration"].min() >= 1
        assert regime_data["strength"].between(0, 1).all()

    def test_regime_datasets_creation(self, sample_test_data):
        """Test creation of regime-specific datasets"""
        tester = RegimeTester(sample_test_data)

        regime_datasets = tester.regime_datasets

        assert len(regime_datasets) > 0

        for regime_type, dataset in regime_datasets.items():
            assert len(dataset) > 0
            assert "regime_confidence" in dataset.columns
            assert "regime_duration" in dataset.columns
            assert "regime_strength" in dataset.columns

    def test_regime_statistics(self, sample_test_data):
        """Test regime statistics calculation"""
        tester = RegimeTester(sample_test_data)

        stats = tester.get_regime_statistics()

        assert len(stats) > 0

        for regime_type, regime_stats in stats.items():
            assert "periods" in regime_stats
            assert "coverage" in regime_stats
            assert "avg_confidence" in regime_stats
            assert "avg_duration" in regime_stats
            assert "avg_strength" in regime_stats

            assert regime_stats["periods"] > 0
            assert 0 <= regime_stats["coverage"] <= 1

    def test_regime_transition_analysis(self, sample_test_data):
        """Test regime transition analysis"""
        tester = RegimeTester(sample_test_data)

        transition_analysis = tester.create_regime_transition_analysis()

        assert "total_transitions" in transition_analysis
        assert "transition_frequency" in transition_analysis
        assert "transition_matrix" in transition_analysis
        assert "transitions" in transition_analysis

        assert transition_analysis["total_transitions"] >= 0
        assert 0 <= transition_analysis["transition_frequency"] <= 1


class TestPerformanceAttributionAnalyzer:
    """Test PerformanceAttributionAnalyzer functionality"""

    def test_initialization(self, sample_test_data):
        """Test PerformanceAttributionAnalyzer initialization"""
        analyzer = PerformanceAttributionAnalyzer(sample_test_data)

        assert len(analyzer.test_data) > 0
        assert "returns" in analyzer.test_data.columns
        assert len(analyzer.baseline_metrics) > 0
        assert "total_return" in analyzer.baseline_metrics
        assert "sharpe_ratio" in analyzer.baseline_metrics

    def test_baseline_metrics_calculation(self, sample_test_data):
        """Test baseline metrics calculation"""
        analyzer = PerformanceAttributionAnalyzer(sample_test_data)

        metrics = analyzer.baseline_metrics

        assert "total_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics

        # Check that metrics are reasonable
        assert metrics["volatility"] > 0
        assert 0 <= metrics["max_drawdown"] <= 1


class TestTestDatasetGenerator:
    """Test TestDatasetGenerator functionality"""

    def test_initialization(self):
        """Test TestDatasetGenerator initialization"""
        generator = TestDatasetGenerator()

        assert len(generator.market_scenarios) > 0
        assert generator.synthetic_generator is not None

    def test_scenario_definitions(self):
        """Test market scenario definitions"""
        generator = TestDatasetGenerator()

        scenarios = generator.get_all_scenarios()

        assert len(scenarios) > 0
        assert "strong_bull_low_vol" in scenarios
        assert "strong_bear_low_vol" in scenarios
        assert "tight_range_low_vol" in scenarios

        # Test scenario descriptions
        for scenario_name in scenarios:
            description = generator.get_scenario_description(scenario_name)
            assert len(description) > 0

    def test_synthetic_data_generation(self):
        """Test synthetic data generation"""
        generator = TestDatasetGenerator()

        # Generate data for a specific scenario
        data = generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)

        assert len(data) > 0
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns

        # Check OHLC consistency
        assert (data["high"] >= data["open"]).all()
        assert (data["high"] >= data["close"]).all()
        assert (data["low"] <= data["open"]).all()
        assert (data["low"] <= data["close"]).all()

        # Check that technical indicators were added
        assert "sma_20" in data.columns
        assert "rsi" in data.columns
        assert "returns" in data.columns

    def test_edge_case_generation(self):
        """Test edge case dataset generation"""
        generator = TestDatasetGenerator()

        edge_cases = ["missing_data", "extreme_volatility", "zero_volume", "price_gaps"]

        for case_type in edge_cases:
            data = generator.generate_edge_case_dataset(case_type, duration_days=50, seed=42)

            assert len(data) > 0
            assert "open" in data.columns
            assert "close" in data.columns

            # Specific checks for each edge case
            if case_type == "missing_data":
                # Should have some NaN values
                assert data.isnull().any().any()
            elif case_type == "zero_volume":
                # Should have some zero volume periods
                assert (data["volume"] == 0).any()

    def test_regime_labeled_dataset(self):
        """Test regime-labeled dataset creation"""
        generator = TestDatasetGenerator()

        market_data, regime_labels = generator.create_regime_labeled_dataset(
            "strong_bull_low_vol", seed=42
        )

        assert len(market_data) > 0
        assert len(regime_labels) > 0
        assert len(market_data) == len(regime_labels)

        # Check regime label columns
        assert "trend" in regime_labels.columns
        assert "volatility" in regime_labels.columns
        assert "regime_type" in regime_labels.columns
        assert "confidence" in regime_labels.columns

    def test_comprehensive_test_suite(self):
        """Test comprehensive test suite generation"""
        generator = TestDatasetGenerator()

        test_suite = generator.get_comprehensive_test_suite(seed=42)

        assert len(test_suite) > 0

        # Should include scenario datasets
        scenario_datasets = [k for k in test_suite.keys() if k.startswith("scenario_")]
        assert len(scenario_datasets) > 0

        # Should include edge case datasets
        edge_case_datasets = [k for k in test_suite.keys() if k.startswith("edge_case_")]
        assert len(edge_case_datasets) > 0

        # All datasets should be valid
        for name, dataset in test_suite.items():
            assert len(dataset) > 0
            assert "close" in dataset.columns


class TestSyntheticDataGenerator:
    """Test SyntheticDataGenerator functionality"""

    def test_scenario_data_generation(self):
        """Test synthetic data generation for scenarios"""
        from src.strategies.components.testing.test_datasets import SyntheticDataGenerator

        generator = SyntheticDataGenerator()

        # Create a test scenario
        scenario = MarketScenario(
            name="test_scenario",
            description="Test scenario",
            duration_days=100,
            trend_direction="up",
            volatility_level="low",
            trend_strength=0.7,
            volatility_value=0.20,
            initial_price=100.0,
            final_price=120.0,
        )

        data = generator.generate_scenario_data(scenario, seed=42)

        assert len(data) == 100
        assert "open" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns

        # Check that price generally trends upward (allowing for some volatility)
        initial_price = data["close"].iloc[0]
        final_price = data["close"].iloc[-1]
        assert final_price > initial_price * 1.05  # At least 5% gain

    def test_price_series_generation(self):
        """Test price series generation with different parameters"""
        from src.strategies.components.testing.test_datasets import SyntheticDataGenerator

        generator = SyntheticDataGenerator()

        # Test bull market scenario
        bull_scenario = MarketScenario(
            name="bull_test",
            description="Bull test",
            duration_days=50,
            trend_direction="up",
            volatility_level="low",
            trend_strength=0.8,
            volatility_value=0.15,
            initial_price=100.0,
            final_price=110.0,
        )

        prices = generator._generate_price_series(bull_scenario)

        assert len(prices) == 50
        assert prices[0] > 0  # Positive prices
        assert prices[-1] > prices[0]  # Generally upward trend


def test_integration_all_components(sample_test_data):
    """Integration test using all components together"""
    # Create all testers
    component_tester = ComponentPerformanceTester(sample_test_data)
    # Use synthetic data for regime tester since it needs more data
    dataset_generator = TestDatasetGenerator()
    regime_data = dataset_generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)
    regime_tester = RegimeTester(regime_data)
    attribution_analyzer = PerformanceAttributionAnalyzer(sample_test_data)

    # Create components to test
    signal_generator = RandomSignalGenerator(seed=42)
    risk_manager = FixedRiskManager()
    position_sizer = FixedFractionSizer()

    # Test components individually
    signal_results = component_tester.test_signal_generator(signal_generator)
    risk_results = component_tester.test_risk_manager(risk_manager)
    sizing_results = component_tester.test_position_sizer(position_sizer)

    # Test all components together
    combined_results = component_tester.test_all_components(
        signal_generator=signal_generator, risk_manager=risk_manager, position_sizer=position_sizer
    )

    # Verify all results are valid
    assert signal_results.total_signals > 0
    assert risk_results.total_scenarios > 0
    assert sizing_results.total_calculations > 0
    assert combined_results.overall_performance_score >= 0

    # Test regime-specific functionality
    regime_stats = regime_tester.get_regime_statistics()
    assert len(regime_stats) > 0

    # Test synthetic data generation
    synthetic_data = dataset_generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)
    assert len(synthetic_data) > 0

    print("✅ All component testing framework integration tests passed!")


if __name__ == "__main__":
    # Run a quick integration test
    dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 200)
    prices = 100 * np.exp(np.cumsum(returns))

    test_data = pd.DataFrame(
        {
            "open": prices * (1 + np.random.normal(0, 0.001, 200)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, 200))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, 200))),
            "close": prices,
            "volume": np.random.lognormal(15, 0.5, 200),
        },
        index=dates,
    )

    test_data["high"] = np.maximum.reduce(
        [test_data["open"], test_data["high"], test_data["low"], test_data["close"]]
    )
    test_data["low"] = np.minimum.reduce(
        [test_data["open"], test_data["high"], test_data["low"], test_data["close"]]
    )

    test_integration_all_components(test_data)
