#!/usr/bin/env python3
"""
Binance Data Downloader

A standalone script to download historical OHLCV data from Binance and save it to CSV format.
"""

import ccxt
import pandas as pd
import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

def download_data(symbol, timeframe='1d', start_date=None, end_date=None, output_dir='.'):
    """
    Download OHLCV data from Binance and save as CSV. Returns the CSV file path.
    """
    binance = ccxt.binance()
    symbol = symbol.replace('/', '') if '/' in symbol else symbol
    if not symbol.endswith('USDT'):
        raise ValueError('Only USDT pairs are supported (e.g., ETHUSDT, BTCUSDT)')
    symbol = symbol.replace('USDT', '/USDT')

    since = int(pd.Timestamp(start_date).timestamp() * 1000) if start_date else None
    end = int(pd.Timestamp(end_date).timestamp() * 1000) if end_date else None
    all_ohlcv = []
    limit = 1000
    fetch_since = since
    while True:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=limit)
        if not ohlcv:
            break
        all_ohlcv += ohlcv
        if len(ohlcv) < limit:
            break
        fetch_since = ohlcv[-1][0] + 1
        if end and fetch_since > end:
            break
    if not all_ohlcv:
        raise ValueError('No data fetched for the given parameters.')
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"{symbol.replace('/', '')}_{timeframe}.csv")
    df.to_csv(csv_file)
    return csv_file

def main():
    parser = argparse.ArgumentParser(description='Download Binance OHLCV data and save as CSV')
    parser.add_argument('symbol', help='Trading pair symbol (e.g., ETHUSDT, BTCUSDT)')
    parser.add_argument('--timeframe', default='1d', help='Candle timeframe (default: 1d)')
    parser.add_argument('--start_date', default=None, help='Start date (YYYY-MM-DD or ISO format)')
    parser.add_argument('--end_date', default=None, help='End date (YYYY-MM-DD or ISO format)')
    parser.add_argument('--output_dir', default='.', help='Directory to save CSV')
    args = parser.parse_args()
    try:
        csv_file = download_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir
        )
        print(f"CSV saved to {csv_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)

if __name__ == '__main__':
    main() 