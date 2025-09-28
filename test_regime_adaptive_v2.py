#!/usr/bin/env python3
"""
Test script to demonstrate the new RegimeAdaptiveV2 implementation
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.strategies.regime_adaptive_v2 import RegimeAdaptiveV2


def create_sample_data(days: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing"""
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), periods=days, freq='1H')
    
    # Create realistic price data with different regimes
    np.random.seed(42)
    
    # Bull market (first 30 days)
    bull_trend = np.linspace(100, 120, 30)
    bull_noise = np.random.normal(0, 1, 30)
    bull_prices = bull_trend + bull_noise
    
    # Bear market (next 30 days)
    bear_trend = np.linspace(120, 100, 30)
    bear_noise = np.random.normal(0, 2, 30)  # Higher volatility
    bear_prices = bear_trend + bear_noise
    
    # Range market (last 40 days)
    range_base = 100
    range_noise = np.random.normal(0, 0.5, 40)
    range_prices = range_base + range_noise
    
    # Combine all prices
    all_prices = np.concatenate([bull_prices, bear_prices, range_prices])
    
    # Create OHLCV data
    data = []
    for i, price in enumerate(all_prices):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        open_price = price * (1 + np.random.normal(0, 0.005))
        close_price = price
        volume = np.random.uniform(1000, 10000)
        
        data.append({
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates[:len(data)])
    return df


def test_basic_functionality():
    """Test basic functionality of RegimeAdaptiveV2"""
    
    print("🧪 Testing RegimeAdaptiveV2 Basic Functionality")
    print("=" * 50)
    
    # Create strategy
    strategy = RegimeAdaptiveV2()
    
    # Create sample data
    df = create_sample_data(100)
    
    # Test indicator calculation
    print("📊 Calculating indicators...")
    df_with_indicators = strategy.calculate_indicators(df)
    print(f"✅ Indicators calculated. DataFrame shape: {df_with_indicators.shape}")
    
    # Test entry conditions
    print("\n🔍 Testing entry conditions...")
    entry_signals = []
    for i in range(50, len(df)):  # Start after sufficient data
        signal = strategy.check_entry_conditions(df, i)
        entry_signals.append(signal)
    
    signal_count = sum(entry_signals)
    print(f"✅ Entry signals: {signal_count}/{len(entry_signals)} ({signal_count/len(entry_signals)*100:.1f}%)")
    
    # Test position sizing
    print("\n💰 Testing position sizing...")
    balance = 10000
    position_sizes = []
    for i in range(50, len(df)):
        size = strategy.calculate_position_size(df, i, balance)
        position_sizes.append(size)
    
    avg_size = np.mean(position_sizes)
    print(f"✅ Average position size: {avg_size:.2f} ({avg_size/balance*100:.1f}% of balance)")
    
    # Test regime analysis
    print("\n📈 Regime Analysis:")
    analysis = strategy.get_regime_analysis()
    print(f"Current regime: {strategy.get_current_regime()}")
    print(f"Current strategy: {strategy.get_current_strategy_name()}")
    print(f"Regime confidence: {strategy.get_regime_confidence():.3f}")
    print(f"Total switches: {analysis.get('total_switches', 0)}")
    
    return strategy, df


def test_configuration():
    """Test configuration options"""
    
    print("\n🔧 Testing Configuration Options")
    print("=" * 50)
    
    # Create strategy
    strategy = RegimeAdaptiveV2()
    
    # Test strategy mapping configuration
    print("📋 Configuring strategy mapping...")
    strategy.configure_strategy_mapping(
        bull_low_vol="momentum_leverage",
        bull_high_vol="ensemble_weighted",
        bear_low_vol="bear",
        bear_high_vol="bear",
        range_low_vol="ml_basic",
        range_high_vol="ml_basic"
    )
    print("✅ Strategy mapping configured")
    
    # Test position sizing configuration
    print("💰 Configuring position sizing...")
    strategy.configure_position_sizing(
        bull_low_vol_multiplier=1.2,  # More aggressive in bull markets
        bull_high_vol_multiplier=0.8,
        bear_low_vol_multiplier=0.4,  # More conservative in bear markets
        bear_high_vol_multiplier=0.2,
        range_low_vol_multiplier=0.6,
        range_high_vol_multiplier=0.3
    )
    print("✅ Position sizing configured")
    
    # Test switching parameters
    print("⚙️ Configuring switching parameters...")
    strategy.configure_switching_parameters(
        min_confidence=0.5,           # Higher confidence threshold
        min_regime_duration=15,      # Longer regime stability requirement
        switch_cooldown=25           # Longer cooldown between switches
    )
    print("✅ Switching parameters configured")
    
    # Test parameters
    params = strategy.get_parameters()
    print(f"\n📊 Current Parameters:")
    print(f"  Current strategy: {params['current_strategy']}")
    print(f"  Current regime: {params['current_regime']}")
    print(f"  Min confidence: {params['strategy_config']['min_confidence']}")
    print(f"  Min regime duration: {params['strategy_config']['min_regime_duration']}")
    print(f"  Switch cooldown: {params['strategy_config']['switch_cooldown']}")


def test_performance_comparison():
    """Compare performance with original implementation"""
    
    print("\n⚡ Performance Comparison")
    print("=" * 50)
    
    import time
    
    # Create sample data
    df = create_sample_data(200)
    
    # Test new implementation
    print("🆕 Testing RegimeAdaptiveV2...")
    strategy_v2 = RegimeAdaptiveV2()
    
    start_time = time.time()
    df_v2 = strategy_v2.calculate_indicators(df)
    v2_time = time.time() - start_time
    
    print(f"✅ RegimeAdaptiveV2: {v2_time:.3f}s")
    
    # Test original implementation (if available)
    try:
        from src.strategies.regime_adaptive import RegimeAdaptive
        
        print("🔄 Testing original RegimeAdaptive...")
        strategy_orig = RegimeAdaptive()
        
        start_time = time.time()
        df_orig = strategy_orig.calculate_indicators(df)
        orig_time = time.time() - start_time
        
        print(f"✅ RegimeAdaptive: {orig_time:.3f}s")
        
        # Performance improvement
        improvement = (orig_time - v2_time) / orig_time * 100
        print(f"🚀 Performance improvement: {improvement:.1f}%")
        
    except ImportError:
        print("⚠️ Original RegimeAdaptive not available for comparison")


def main():
    """Main test function"""
    
    print("🚀 RegimeAdaptiveV2 Test Suite")
    print("=" * 60)
    
    try:
        # Test basic functionality
        strategy, df = test_basic_functionality()
        
        # Test configuration
        test_configuration()
        
        # Test performance
        test_performance_comparison()
        
        print("\n✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print(f"  - Strategy: {strategy.get_current_strategy_name()}")
        print(f"  - Regime: {strategy.get_current_regime()}")
        print(f"  - Confidence: {strategy.get_regime_confidence():.3f}")
        
        # Show regime analysis
        analysis = strategy.get_regime_analysis()
        if analysis:
            print(f"  - Total switches: {analysis.get('total_switches', 0)}")
            print(f"  - Average confidence: {analysis.get('average_confidence', 0):.3f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())