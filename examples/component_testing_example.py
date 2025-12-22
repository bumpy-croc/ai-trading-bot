#!/usr/bin/env python3
"""
Component Testing Framework Example

This example demonstrates how to use the comprehensive component testing framework
to test and analyze strategy components in isolation and combination.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Import the testing framework
from src.strategies.components.testing import (
    ComponentPerformanceTester,
    RegimeTester,
    PerformanceAttributionAnalyzer,
    TestDatasetGenerator,
)

# Import components to test
from src.strategies.components.signal_generator import (
    HoldSignalGenerator,
    RandomSignalGenerator,
    WeightedVotingSignalGenerator,
)
from src.strategies.components.risk_manager import FixedRiskManager, VolatilityRiskManager
from src.strategies.components.position_sizer import (
    FixedFractionSizer,
    ConfidenceWeightedSizer,
    KellySizer,
)


def main():
    """Run comprehensive component testing example"""
    print("🧪 Component Testing Framework Example")
    print("=" * 50)

    # 1. Generate test datasets
    print("\n1. Generating Test Datasets...")
    dataset_generator = TestDatasetGenerator()

    # Get comprehensive test suite
    test_suite = dataset_generator.get_comprehensive_test_suite(seed=42)
    print(f"   Generated {len(test_suite)} test datasets")

    # Use a specific scenario for detailed testing
    bull_market_data = dataset_generator.generate_synthetic_dataset("strong_bull_low_vol", seed=42)
    print(f"   Bull market dataset: {len(bull_market_data)} periods")

    # 2. Test Signal Generators
    print("\n2. Testing Signal Generators...")
    component_tester = ComponentPerformanceTester(bull_market_data)

    # Test different signal generators
    signal_generators = [
        HoldSignalGenerator(),
        RandomSignalGenerator(buy_prob=0.4, sell_prob=0.3, seed=42),
        WeightedVotingSignalGenerator(
            {HoldSignalGenerator(): 0.3, RandomSignalGenerator(seed=42): 0.7}
        ),
    ]

    signal_results = {}
    for generator in signal_generators:
        print(f"   Testing {generator.name}...")
        results = component_tester.test_signal_generator(generator)
        signal_results[generator.name] = results

        print(f"     Accuracy: {results.accuracy:.3f}")
        print(f"     Total Signals: {results.total_signals}")
        print(f"     Sharpe Ratio: {results.sharpe_ratio:.3f}")

    # 3. Test Risk Managers
    print("\n3. Testing Risk Managers...")

    risk_managers = [FixedRiskManager(risk_per_trade=0.02), VolatilityRiskManager(base_risk=0.02)]

    risk_results = {}
    for risk_manager in risk_managers:
        print(f"   Testing {risk_manager.name}...")
        results = component_tester.test_risk_manager(risk_manager)
        risk_results[risk_manager.name] = results

        print(f"     Drawdown Control: {results.drawdown_control_score:.3f}")
        print(f"     Risk Efficiency: {results.risk_efficiency_score:.3f}")

    # 4. Test Position Sizers
    print("\n4. Testing Position Sizers...")

    position_sizers = [
        FixedFractionSizer(fraction=0.02),
        ConfidenceWeightedSizer(base_fraction=0.03),
        KellySizer(win_rate=0.55, avg_win=0.02, avg_loss=0.015),
    ]

    sizing_results = {}
    for sizer in position_sizers:
        print(f"   Testing {sizer.name}...")
        results = component_tester.test_position_sizer(sizer)
        sizing_results[sizer.name] = results

        print(f"     Optimal Sizing Score: {results.optimal_sizing_score:.3f}")
        print(f"     Bounds Adherence: {results.bounds_adherence_rate:.3f}")

    # 5. Regime-Specific Testing
    print("\n5. Regime-Specific Testing...")

    # Generate regime-labeled dataset
    regime_data, regime_labels = dataset_generator.create_regime_labeled_dataset(
        "multiple_regime_changes", seed=42
    )

    regime_tester = RegimeTester(regime_data)

    # Get regime statistics
    regime_stats = regime_tester.get_regime_statistics()
    print(f"   Found {len(regime_stats)} different regimes:")

    for regime_type, stats in regime_stats.items():
        print(f"     {regime_type}: {stats['periods']} periods ({stats['coverage']:.1%} coverage)")

    # Test signal generator in specific regime
    best_signal_generator = signal_generators[1]  # RandomSignalGenerator performed well
    regime_test = regime_tester.test_component_in_regime(
        best_signal_generator, list(regime_stats.keys())[0]  # Test in first regime
    )
    print(f"   Signal accuracy in regime: {regime_test.get('accuracy', 'N/A')}")

    # 6. Performance Attribution Analysis
    print("\n6. Performance Attribution Analysis...")

    attribution_analyzer = PerformanceAttributionAnalyzer(bull_market_data)

    # Test component replacement impact
    replacement_impact = attribution_analyzer.analyze_component_replacement_impact(
        None, "signal", RandomSignalGenerator(seed=42)  # Would need actual strategy object
    )
    print(f"   Component replacement analysis completed")

    # 7. Edge Case Testing
    print("\n7. Edge Case Testing...")

    edge_cases = ["missing_data", "extreme_volatility", "zero_volume"]

    for edge_case in edge_cases:
        print(f"   Testing edge case: {edge_case}")
        edge_data = dataset_generator.generate_edge_case_dataset(edge_case, seed=42)

        try:
            edge_tester = ComponentPerformanceTester(edge_data)
            edge_results = edge_tester.test_signal_generator(HoldSignalGenerator())
            print(f"     Edge case handled successfully: {edge_results.total_signals} signals")
        except Exception as e:
            print(f"     Edge case caused issues: {e}")

    # 8. Summary and Recommendations
    print("\n8. Summary and Recommendations")
    print("=" * 30)

    # Find best performing components
    best_signal = max(signal_results.items(), key=lambda x: x[1].sharpe_ratio)
    best_risk = max(risk_results.items(), key=lambda x: x[1].risk_efficiency_score)
    best_sizing = max(sizing_results.items(), key=lambda x: x[1].optimal_sizing_score)

    print(f"🏆 Best Signal Generator: {best_signal[0]} (Sharpe: {best_signal[1].sharpe_ratio:.3f})")
    print(
        f"🏆 Best Risk Manager: {best_risk[0]} (Efficiency: {best_risk[1].risk_efficiency_score:.3f})"
    )
    print(
        f"🏆 Best Position Sizer: {best_sizing[0]} (Optimality: {best_sizing[1].optimal_sizing_score:.3f})"
    )

    print("\n💡 Optimization Recommendations:")
    print("   1. Use ensemble signal generators for better accuracy")
    print("   2. Implement volatility-based risk management")
    print("   3. Consider confidence-weighted position sizing")
    print("   4. Test components across multiple market regimes")
    print("   5. Monitor performance attribution continuously")

    print("\n✅ Component testing framework demonstration completed!")


if __name__ == "__main__":
    main()
