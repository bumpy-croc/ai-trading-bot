#!/usr/bin/env python3
"""
Binance Data Downloader

A standalone script to download historical OHLCV data from Binance and save it to CSV format.
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from src.utils.symbol_factory import SymbolFactory


def download_data(
    symbol: str,
    timeframe: str = "1h",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    output_dir: Optional[os.PathLike] = ".",  # Changed here
    fmt: str = "feather",
) -> Path:
    """Download OHLCV data and save to Feather or CSV.

    Args:
        symbol: Trading pair, e.g. ``'BTCUSDT'``.
        timeframe: Binance/ccxt timeframe string, default ``'1h'``.
        start_date, end_date: ISO date strings.  If omitted, downloads all available.
        end_date: end_date: ISO date strings.  If omitted, downloads all available.
        output_dir: Directory to place file (will be created).
        fmt: 'feather' (default) or 'csv'.

    Returns
    -------
    pathlib.Path
        Path to the saved file.
    """
    """
    Download OHLCV data from Binance and save as CSV. Returns the CSV file path.
    Symbol should be in generic format (e.g., 'BTC-USD', 'ETH-USD') and will be converted using SymbolFactory.
    """
    binance = ccxt.binance()
    symbol = SymbolFactory.to_exchange_symbol(symbol, "binance")
    if not symbol.endswith("USDT"):
        raise ValueError("Only USDT pairs are supported (e.g., ETH-USD, BTC-USD, ETHUSDT, BTCUSDT)")
    symbol = symbol.replace("USDT", "/USDT")

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
        raise ValueError("No data fetched for the given parameters.")
    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    stem = f"{symbol.replace('/', '')}_{timeframe}_{start_date}_{end_date}"
    if fmt == "csv":
        out = path / f"{stem}.csv"
        df.to_csv(out, index=True)
    else:
        out = path / f"{stem}.feather"
        df.reset_index().to_feather(out, compression="zstd")
    return out


def main():
    parser = argparse.ArgumentParser(description="Download Binance OHLCV data and save as CSV")
    parser.add_argument(
        "symbol", help="Trading pair symbol (e.g., BTC-USD, ETH-USD, BTCUSDT, ETHUSDT)"
    )
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    parser.add_argument("--start_date", default=None, help="Start date (YYYY-MM-DD or ISO format)")
    parser.add_argument("--end_date", default=None, help="End date (YYYY-MM-DD or ISO format)")
    parser.add_argument("--output_dir", default="tests/data", help="Directory to save data file")
    parser.add_argument(
        "--format", choices=["feather", "csv"], default="feather", help="Output format"
    )
    args = parser.parse_args()
    try:
        out_file = download_data(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            fmt=args.format,
        )
        print(f"Saved to {out_file}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
