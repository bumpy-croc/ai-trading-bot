#!/usr/bin/env python3
"""
Prefill Binance cache for background backtests.

Downloads and caches historical OHLCV data by year using the existing
CachedDataProvider year-based caching scheme. This avoids repeated downloads
for overlapping backtests and enables background agents to run quickly.

Examples:
  - Prefill last 8 years of hourly BTCUSDT and ETHUSDT:
      python scripts/prefill_binance_cache.py --symbols BTCUSDT ETHUSDT --timeframes 1h --years 8

  - Prefill specific date range for 4h candles:
      python scripts/prefill_binance_cache.py --symbols BTCUSDT --timeframes 4h \
        --start 2018-01-01 --end 2024-12-31
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

# Ensure package imports work when executed as a script
sys.path.append(str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.paths import get_cache_dir  # noqa: E402
from data_providers.binance_provider import BinanceProvider  # noqa: E402
from data_providers.cached_data_provider import CachedDataProvider  # noqa: E402
from utils.symbol_factory import SymbolFactory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefill local cache with Binance OHLCV data")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT"],
        help="Symbols to prefill (Binance format, e.g., BTCUSDT; accepts BTC-USD and normalizes)",
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h"],
        help="Timeframes to prefill (e.g., 1h 4h 1d)",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=8,
        help="Number of trailing calendar years to prefill (ignored if --start provided)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). If omitted, computed from --years.",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD). Defaults to now.",
    )
    parser.add_argument(
        "--cache-ttl-hours",
        type=int,
        default=24,
        help="TTL for current-year entries (prior years are treated immutable)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(get_cache_dir()),
        help="Cache directory override (default is project data cache)",
    )
    return parser.parse_args()


def normalize_symbols(raw_symbols: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for s in raw_symbols:
        s = s.strip().upper()
        if "-" in s or "/" in s:
            s = SymbolFactory.to_exchange_symbol(s, "binance")
        normalized.append(s)
    return normalized


def year_chunks(start: datetime, end: datetime) -> List[tuple[int, datetime, datetime]]:
    chunks: List[tuple[int, datetime, datetime]] = []
    cur = start
    while cur <= end:
        y = cur.year
        y_start = datetime(y, 1, 1)
        y_end = datetime(y + 1, 1, 1) - timedelta(seconds=1)
        if y_start < start:
            y_start = start
        if y_end > end:
            y_end = end
        chunks.append((y, y_start, y_end))
        cur = datetime(y + 1, 1, 1)
    return chunks


def prefill(symbols: List[str], timeframes: List[str], start: datetime, end: datetime, cache_ttl_hours: int, cache_dir: str) -> None:
    provider = CachedDataProvider(BinanceProvider(), cache_dir=cache_dir, cache_ttl_hours=cache_ttl_hours)

    for symbol in symbols:
        for tf in timeframes:
            # Leverage provider's internal year-based caching by calling once for each year
            for _, y_start, y_end in year_chunks(start, end):
                try:
                    df = provider.get_historical_data(symbol, tf, y_start, y_end)
                    if df is None or df.empty:
                        print(f"{symbol} {tf} {y_start.year}: no data")
                    else:
                        print(
                            f"Cached {symbol} {tf} {y_start.year}: {len(df)} candles from {df.index.min()} to {df.index.max()}"
                        )
                except Exception as e:
                    print(f"Error caching {symbol} {tf} {y_start.year}: {e}")


def main() -> int:
    args = parse_args()
    end = datetime.strptime(args.end, "%Y-%m-%d") if args.end else datetime.now()
    if args.start:
        start = datetime.strptime(args.start, "%Y-%m-%d")
    else:
        # Trailing full years: start at Jan 1 (current_year - years)
        cy = end.year
        start = datetime(cy - args.years, 1, 1)

    symbols = normalize_symbols(args.symbols)
    timeframes = [tf.strip() for tf in args.timeframes]

    print(
        f"Prefilling cache dir={args.cache_dir} symbols={symbols} timeframes={timeframes} range={start.date()}..{end.date()}"
    )
    prefill(symbols, timeframes, start, end, args.cache_ttl_hours, args.cache_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


