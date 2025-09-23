#!/usr/bin/env python3
"""
Generate synthetic market data for backtesting ensemble strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

def generate_synthetic_crypto_data(
    start_date: str = "2023-01-01",
    end_date: str = "2024-01-01", 
    timeframe: str = "1h",
    initial_price: float = 30000.0,
    symbol: str = "BTCUSDT"
):
    """Generate realistic synthetic cryptocurrency price data"""
    
    # Create date range
    if timeframe == "1h":
        freq = "H"
    elif timeframe == "4h":
        freq = "4H"
    elif timeframe == "1d":
        freq = "D"
    else:
        freq = "H"
    
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    n_periods = len(dates)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate price movements with realistic characteristics
    # Base trend (slight upward bias for crypto)
    trend = np.linspace(0, 0.5, n_periods)  # 50% gain over period
    
    # Add cyclical patterns (market cycles)
    cycle1 = 0.3 * np.sin(2 * np.pi * np.arange(n_periods) / (n_periods * 0.25))  # 4 cycles
    cycle2 = 0.15 * np.sin(2 * np.pi * np.arange(n_periods) / (n_periods * 0.1))   # 10 cycles
    
    # Random walk component (volatility)
    volatility = 0.02  # 2% hourly volatility
    random_walk = np.cumsum(np.random.normal(0, volatility, n_periods))
    
    # Combine components
    log_returns = trend + cycle1 + cycle2 + random_walk
    
    # Convert to prices
    prices = initial_price * np.exp(log_returns)
    
    # Generate OHLCV data
    data = []
    for i, price in enumerate(prices):
        # Add some intraperiod volatility for OHLC
        volatility_factor = np.random.uniform(0.995, 1.005)
        high_factor = np.random.uniform(1.0, 1.02)
        low_factor = np.random.uniform(0.98, 1.0)
        
        if i == 0:
            open_price = price
        else:
            open_price = data[i-1]['close']
        
        close_price = price * volatility_factor
        high_price = max(open_price, close_price) * high_factor
        low_price = min(open_price, close_price) * low_factor
        
        # Volume (correlated with volatility)
        base_volume = 1000
        volatility_multiplier = abs(log_returns[i] - (log_returns[i-1] if i > 0 else 0)) * 50
        volume = base_volume * (1 + volatility_multiplier) * np.random.uniform(0.5, 2.0)
        
        data.append({
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Ensure high >= close >= low and high >= open >= low
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    return df

def save_to_cache(df: pd.DataFrame, symbol: str = "BTCUSDT", timeframe: str = "1h", year: int = 2023):
    """Save data to the cache directory in the expected format"""
    
    import hashlib
    
    cache_dir = "/workspace/cache/market_data"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key using the same method as CachedDataProvider
    request_str = f"{symbol}_{timeframe}_{year}"
    cache_key = hashlib.sha256(request_str.encode()).hexdigest()
    
    # Save with expected filename format (hash-based)
    filename = f"{cache_key}.pkl"
    filepath = os.path.join(cache_dir, filename)
    
    # Save as pickle (format expected by cached data provider)
    with open(filepath, 'wb') as f:
        pickle.dump(df, f)
    
    print(f"Saved {len(df)} records to {filepath}")
    print(f"Cache key: {cache_key} for {symbol}_{timeframe}_{year}")
    return filepath

if __name__ == "__main__":
    # Generate data for 2023 (full year)
    print("Generating synthetic crypto data for backtesting...")
    
    # Generate 1h data for 2023
    df_2023 = generate_synthetic_crypto_data(
        start_date="2023-01-01",
        end_date="2024-01-01",
        timeframe="1h",
        initial_price=16000.0,  # Starting from bear market low
        symbol="BTCUSDT"
    )
    
    print(f"Generated {len(df_2023)} hourly candles for 2023")
    print(f"Price range: ${df_2023['close'].min():.2f} - ${df_2023['close'].max():.2f}")
    print(f"Final price: ${df_2023['close'].iloc[-1]:.2f}")
    
    # Save to cache
    save_to_cache(df_2023, "BTCUSDT", "1h", 2023)
    
    # Also generate some 2024 data
    df_2024 = generate_synthetic_crypto_data(
        start_date="2024-01-01",
        end_date="2024-06-01",
        timeframe="1h",
        initial_price=df_2023['close'].iloc[-1],  # Continue from 2023
        symbol="BTCUSDT"
    )
    
    print(f"Generated {len(df_2024)} hourly candles for 2024")
    save_to_cache(df_2024, "BTCUSDT", "1h", 2024)
    
    print("Synthetic data generation complete!")
    print("\nNow you can run backtests with:")
    print("atb backtest ml_basic --symbol BTCUSDT --timeframe 1h --start 2023-01-01 --end 2023-12-31")