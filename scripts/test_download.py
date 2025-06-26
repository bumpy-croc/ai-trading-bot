#!/usr/bin/env python3
"""
Test script for the download_binance_data module

This script demonstrates different ways to use the download functionality.
"""

from download_binance_data import download_data
import pandas as pd

def test_downloads():
    """Test different download scenarios"""
    
    print("🚀 Testing Binance Data Download Script")
    print("=" * 50)
    
    # Test 1: Download ETH/USDT daily data for last 30 days
    print("\n📊 Test 1: ETH/USDT Daily Data (30 days)")
    try:
        csv_file = download_data(
            symbol='ETH/USDT',
            timeframe='1d',
            start_date='2024-11-25T00:00:00Z',
            end_date='2024-12-25T00:00:00Z',
            output_dir='../data/test'
        )
        
        # Load and show sample data
        df = pd.read_csv(csv_file)
        print(f"✅ Success! Downloaded {len(df)} candles")
        print("Sample data:")
        print(df.head(3))
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    # Test 2: Download BTC/USDT hourly data for last 7 days
    print("\n📊 Test 2: BTC/USDT Hourly Data (7 days)")
    try:
        csv_file = download_data(
            symbol='BTC/USDT',
            timeframe='1h',
            start_date='2024-12-18T00:00:00Z',
            end_date='2024-12-25T00:00:00Z',
            output_dir='../data/test'
        )
        
        # Load and show sample data
        df = pd.read_csv(csv_file)
        print(f"✅ Success! Downloaded {len(df)} candles")
        print("Price range:")
        print(f"High: ${df['high'].max():.2f}")
        print(f"Low: ${df['low'].min():.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Testing complete!")
    print("\nUsage examples:")
    print("python download_binance_data.py BTCUSDT --days 365")
    print("python download_binance_data.py ETHUSDT --timeframe 1h --start 2024-01-01 --end 2024-06-01")
    print("python download_binance_data.py SOLUSDT --timeframe 4h --days 90 --output-dir ../data/sol")

if __name__ == '__main__':
    test_downloads() 