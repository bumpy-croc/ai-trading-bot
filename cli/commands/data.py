from __future__ import annotations
from datetime import UTC, datetime

import argparse
import os
import sys
from pathlib import Path

# Ensure project root and src are in sys.path for absolute imports
from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

import ccxt
import pandas as pd

from src.infrastructure.logging.config import configure_logging
from src.trading.symbols.factory import SymbolFactory


def _download(ns: argparse.Namespace) -> int:
    try:
        symbol = SymbolFactory.to_exchange_symbol(ns.symbol, "binance")
        if not symbol.endswith("USDT"):
            print("Only USDT pairs are supported (e.g., ETH-USD, BTC-USD, ETHUSDT, BTCUSDT)")
            return 1
        symbol = symbol.replace("USDT", "/USDT")
        binance = ccxt.binance()
        since = int(pd.Timestamp(ns.start_date).timestamp() * 1000) if ns.start_date else None
        end = int(pd.Timestamp(ns.end_date).timestamp() * 1000) if ns.end_date else None
        all_ohlcv = []
        limit = 1000
        fetch_since = since
        while True:
            ohlcv = binance.fetch_ohlcv(symbol, ns.timeframe, since=fetch_since, limit=limit)
            if not ohlcv:
                break
            all_ohlcv += ohlcv
            if len(ohlcv) < limit:
                break
            fetch_since = ohlcv[-1][0] + 1
            if end and fetch_since > end:
                break
        if not all_ohlcv:
            print("No data fetched for the given parameters.")
            return 1
        df = pd.DataFrame(
            all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        out_dir = Path(ns.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{symbol.replace('/', '')}_{ns.timeframe}_{ns.start_date}_{ns.end_date}"
        if ns.format == "csv":
            out = out_dir / f"{stem}.csv"
            df.to_csv(out, index=True)
        else:
            out = out_dir / f"{stem}.feather"
            df.reset_index().to_feather(out, compression="zstd")
        print(f"Saved to {out}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def _prefill(ns: argparse.Namespace) -> int:
    from datetime import UTC, datetime

    from src.config.paths import get_cache_dir
    from src.data_providers.binance_provider import BinanceProvider
    from src.data_providers.cached_data_provider import CachedDataProvider

    def _normalize_symbols(raw):
        normalized = []
        for s in raw:
            s = s.strip().upper()
            if "-" in s or "/" in s:
                s = SymbolFactory.to_exchange_symbol(s, "binance")
            normalized.append(s)
        return normalized

    def _year_chunks(start: datetime, end: datetime):
        from datetime import UTC, timedelta

        chunks = []
        cur = start
        while cur <= end:
            y = cur.year
            y_start = datetime(y, 1, 1, tzinfo=UTC)
            y_end = datetime(y + 1, 1, 1, tzinfo=UTC) - timedelta(seconds=1)
            if y_start < start:
                y_start = start
            if y_end > end:
                y_end = end
            chunks.append((y, y_start, y_end))
            cur = datetime(y + 1, 1, 1, tzinfo=UTC)
        return chunks

    end = (
        datetime.strptime(ns.end, "%Y-%m-%d").replace(tzinfo=UTC)
        if ns.end
        else datetime.now(UTC)
    )
    if ns.start:
        start = datetime.strptime(ns.start, "%Y-%m-%d").replace(tzinfo=UTC)
    else:
        cy = end.year
        start = datetime(cy - ns.years, 1, 1, tzinfo=UTC)

    symbols = _normalize_symbols(ns.symbols)
    timeframes = [tf.strip() for tf in ns.timeframes]
    cache_dir = ns.cache_dir or str(get_cache_dir())
    provider = CachedDataProvider(
        BinanceProvider(), cache_dir=cache_dir, cache_ttl_hours=ns.cache_ttl_hours
    )
    print(
        f"Prefilling cache dir={cache_dir} symbols={symbols} timeframes={timeframes} range={start.date()}..{end.date()}"
    )
    for symbol in symbols:
        for tf in timeframes:
            for _, y_start, y_end in _year_chunks(start, end):
                try:
                    df = provider.get_historical_data(symbol, tf, y_start, y_end)
                    if df is None or df.empty:
                        print(f"{symbol} {tf} {y_start.year}: no data")
                    else:
                        print(
                            f"Cached {symbol} {tf} {y_start.year}: {len(df)} candles from {df.index.min()} to {df.index.max()}"
                        )
                except Exception as e:
                    print(f"{symbol} {tf} {y_start.year}: error {e}")
    return 0


def _preload_offline(ns: argparse.Namespace) -> int:
    """Pre-load cache with historical data for offline backtesting."""
    import logging
    from datetime import datetime, timedelta
    from pathlib import Path

    from tqdm import tqdm

    from src.config.paths import ensure_dir_exists, get_cache_dir
    from src.data_providers.binance_provider import BinanceProvider
    from src.data_providers.cached_data_provider import CachedDataProvider
    from src.infrastructure.logging.config import configure_logging

    # Setup logging
    configure_logging(level_name="INFO")
    logger = logging.getLogger("atb.data.preload")

    # Determine cache directory
    cache_dir = getattr(ns, "cache_dir", None) or str(get_cache_dir())
    ensure_dir_exists(Path(cache_dir))
    logger.info(f"Using cache directory: {cache_dir}")

    # Get years to download
    current_year = datetime.now(UTC).year
    years = list(range(current_year - ns.years_back + 1, current_year + 1))
    logger.info(f"Pre-loading data for years: {years}")

    # Normalize symbols
    def _normalize_symbols(raw):
        normalized = []
        for s in raw:
            s = s.strip().upper()
            if "-" in s or "/" in s:
                s = SymbolFactory.to_exchange_symbol(s, "binance")
            normalized.append(s)
        return normalized

    symbols = _normalize_symbols(ns.symbols)
    timeframes = [tf.strip() for tf in ns.timeframes]

    # Create providers with extended TTL for offline preloading
    # Use very long TTL (10 years) to treat preloaded data as permanently valid
    try:
        binance_provider = BinanceProvider()
        cached_provider = CachedDataProvider(
            binance_provider,
            cache_dir=cache_dir,
            cache_ttl_hours=87600,  # 10 years = 10 * 365 * 24 hours
        )
    except Exception as e:
        logger.error(f"Failed to initialize providers: {e}")
        return 1

    # Pre-load data
    total_combinations = len(symbols) * len(timeframes) * len(years)
    successful_combinations = 0

    logger.info(f"Starting pre-load for {total_combinations} symbol/timeframe/year combinations")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Timeframes: {timeframes}")

    with tqdm(total=total_combinations, desc="Pre-loading cache") as pbar:
        for symbol in symbols:
            for timeframe in timeframes:
                for year in years:
                    try:
                        # Define year boundaries
                        year_start = datetime(year, 1, 1)
                        year_end = datetime(year + 1, 1, 1) - timedelta(seconds=1)

                        # Don't fetch beyond current time
                        current_time = datetime.now(UTC)
                        if year_end > current_time:
                            year_end = current_time

                        # Check if cache already exists and is valid (unless force refresh)
                        if not ns.force_refresh:
                            cache_key = cached_provider._generate_year_cache_key(
                                symbol, timeframe, year
                            )
                            cache_path = cached_provider._get_cache_path(cache_key)
                            if cache_path and cached_provider._is_cache_valid(cache_path, year):
                                logger.debug(
                                    f"Cache already exists for {symbol} {timeframe} {year}"
                                )
                                successful_combinations += 1
                                pbar.update(1)
                                continue

                        # Fetch data for this year
                        data = cached_provider.get_historical_data(
                            symbol, timeframe, year_start, year_end
                        )

                        if data is not None and not data.empty:
                            logger.debug(
                                f"Successfully cached {len(data)} candles for {symbol} {timeframe} {year}"
                            )
                            successful_combinations += 1
                        else:
                            logger.warning(f"No data retrieved for {symbol} {timeframe} {year}")

                    except Exception as e:
                        logger.error(f"Failed to preload {symbol} {timeframe} {year}: {e}")

                    pbar.update(1)

    # Summary
    logger.info(f"Pre-loading completed: {successful_combinations}/{total_combinations} successful")

    # Test offline access if requested
    if ns.test_offline:
        logger.info("Testing offline access...")
        success = _test_offline_access(cache_dir, symbols[0], timeframes[0])
        if not success:
            logger.warning("Offline test failed - cache may not be working properly")
            return 1

    # Check cache directory size
    try:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            total_size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            logger.info(f"Total cache size: {total_size / (1024*1024):.1f} MB")
    except Exception as e:
        logger.warning(f"Could not calculate cache size: {e}")

    # Return appropriate exit code based on success/failure
    if successful_combinations == total_combinations:
        logger.info("✓ All data successfully pre-loaded!")
        return 0
    else:
        logger.warning(
            f"⚠ Some data failed to pre-load ({successful_combinations}/{total_combinations})"
        )
        return 1  # Return non-zero exit code to indicate partial failure


def _test_offline_access(cache_dir: str, symbol: str = "BTCUSDT", timeframe: str = "1h") -> bool:
    """Test that cached data can be accessed in offline mode."""
    import logging
    from datetime import datetime, timedelta

    from src.data_providers.binance_provider import BinanceProvider
    from src.data_providers.cached_data_provider import CachedDataProvider

    logger = logging.getLogger("atb.data.preload")
    logger.info(f"Testing offline access for {symbol} {timeframe}")

    try:
        # Create an offline provider (stub that returns empty data)
        offline_provider = BinanceProvider()
        offline_provider._client = offline_provider._create_offline_client()

        # Wrap with cached provider using extended TTL for offline testing
        cached_provider = CachedDataProvider(
            offline_provider,
            cache_dir=cache_dir,
            cache_ttl_hours=87600,  # 10 years = 10 * 365 * 24 hours
        )

        # Try to fetch recent data (should come from cache)
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=30)

        data = cached_provider.get_historical_data(symbol, timeframe, start_date, end_date)

        if data is not None and not data.empty:
            logger.info(f"✓ Offline test successful: Retrieved {len(data)} candles from cache")
            return True
        else:
            logger.warning("✗ Offline test failed: No data retrieved from cache")
            return False

    except Exception as e:
        logger.error(f"✗ Offline test failed with error: {e}")
        return False


def _cache_manager(ns: argparse.Namespace) -> int:
    import pickle
    from datetime import datetime

    from src.config.paths import get_cache_dir
    from src.data_providers.binance_provider import BinanceProvider
    from src.data_providers.cached_data_provider import CachedDataProvider

    def _format_size(num):
        units = ["B", "KB", "MB", "GB"]
        i = 0
        while num >= 1024 and i < len(units) - 1:
            num /= 1024.0
            i += 1
        return f"{num:.1f} {units[i]}"

    cache_dir = ns.cache_dir or str(get_cache_dir())
    cmd = ns.subcmd
    if cmd == "info":
        provider = CachedDataProvider(BinanceProvider(), cache_dir=cache_dir)
        info = provider.get_cache_info()
        print("Cache Information:\n" + "=" * 40)
        print(f"Cache Directory: {cache_dir}")
        print(f"Total Files: {info['total_files']}")
        print(f"Total Size: {_format_size(info['total_size_mb'] * 1024 * 1024)}")
        if info["oldest_file"]:
            print(f"Oldest File: {info['oldest_file']}")
        if info["newest_file"]:
            print(f"Newest File: {info['newest_file']}")
        return 0
    if cmd == "list":
        if not os.path.exists(cache_dir):
            print(f"Cache directory {cache_dir} does not exist.")
            return 1
        files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]
        if not files:
            print("No cache files found.")
            return 0
        print(f"\nCache Files ({len(files)} total):\n" + "=" * 60)
        file_info = []
        for filename in files:
            path = os.path.join(cache_dir, filename)
            size = os.path.getsize(path)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            file_info.append({"name": filename, "size": size, "modified": mtime, "path": path})
        file_info.sort(key=lambda x: x["modified"], reverse=True)
        for info in file_info:
            if ns.detailed:
                try:
                    with open(info["path"], "rb") as f:
                        data = pickle.load(f)
                    data_info = ""
                    if hasattr(data, "shape"):
                        data_info = f" - {data.shape[0]} rows"
                    if hasattr(data, "index") and len(getattr(data, "index", [])) > 0:
                        start_date = data.index.min().strftime("%Y-%m-%d")
                        end_date = data.index.max().strftime("%Y-%m-%d")
                        data_info += f" ({start_date} to {end_date})"
                    print(
                        f"{info['name'][:20]:<20} {_format_size(info['size']):<8} {info['modified'].strftime('%Y-%m-%d %H:%M')}{data_info}"
                    )
                except Exception as e:
                    print(
                        f"{info['name'][:20]:<20} {_format_size(info['size']):<8} {info['modified'].strftime('%Y-%m-%d %H:%M')} - Error reading: {e}"
                    )
            else:
                print(
                    f"{info['name'][:20]:<20} {_format_size(info['size']):<8} {info['modified'].strftime('%Y-%m-%d %H:%M')}"
                )
        return 0
    if cmd == "clear":
        if not os.path.exists(cache_dir):
            print(f"Cache directory {cache_dir} does not exist.")
            return 1
        files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]
        if not files:
            print("No cache files to clear.")
            return 0
        if not ns.force:
            resp = input(f"Are you sure you want to delete {len(files)} cache files? (y/N): ")
            if resp.lower() != "y":
                print("Cache clear cancelled.")
                return 0
        deleted = 0
        total = 0
        for filename in files:
            path = os.path.join(cache_dir, filename)
            try:
                size = os.path.getsize(path)
                os.remove(path)
                deleted += 1
                total += size
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        print(f"Deleted {deleted} cache files, freed {_format_size(total)}")
        return 0
    if cmd == "clear-old":
        if not os.path.exists(cache_dir):
            print(f"Cache directory {cache_dir} does not exist.")
            return 1
        files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]
        from datetime import datetime

        now = datetime.now(UTC)
        deleted = 0
        total = 0
        for filename in files:
            path = os.path.join(cache_dir, filename)
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
                age_h = (now - mtime).total_seconds() / 3600
                if age_h > ns.hours:
                    size = os.path.getsize(path)
                    os.remove(path)
                    deleted += 1
                    total += size
                    print(f"Deleted {filename} (age: {age_h:.1f} hours)")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        if deleted:
            print(f"Deleted {deleted} old cache files, freed {_format_size(total)}")
        else:
            print(f"No cache files older than {ns.hours} hours found.")
        return 0
    print("Unknown cache subcommand")
    return 1


def _populate_dummy(ns: argparse.Namespace) -> int:
    # Inline minimal port of DummyDataPopulator main
    import logging
    import random
    from datetime import datetime, timedelta

    from src.database.manager import DatabaseManager
    from src.database.models import PositionSide, TradeSource

    configure_logging()
    logger = logging.getLogger("atb.data")

    if not ns.confirm:
        response = input(
            f"This will populate the database with {ns.trades} trades and related data. Continue? (y/N): "
        )
        if response.lower() != "y":
            print("Operation cancelled.")
            return 0

    try:
        db = DatabaseManager(ns.database_url)
        symbols = ["BTCUSDT", "ETHUSDT"]
        strategies = ["MlBasic", "BullStrategy", "BearStrategy", "TestStrategy"]
        timeframes = ["1h", "4h", "1d"]
        base_balance = 10000.0

        def _price(sym):
            ranges = {"BTCUSDT": (25000, 65000), "ETHUSDT": (1500, 4000)}
            lo, hi = ranges.get(sym, (10, 100))
            return random.uniform(lo, hi)

        # sessions
        session_ids = []
        for _i in range(2):
            sid = db.create_trading_session(
                strategy_name=random.choice(strategies),
                symbol=random.choice(symbols),
                timeframe=random.choice(timeframes),
                mode=random.choice([TradeSource.LIVE, TradeSource.BACKTEST, TradeSource.PAPER]),
                initial_balance=base_balance,
                strategy_config={},
            )
            session_ids.append(sid)

        # trades
        for i in range(ns.trades):
            sid = random.choice(session_ids)
            sym = random.choice(symbols)
            side = random.choice([PositionSide.LONG, PositionSide.SHORT])
            entry_time = datetime.now(UTC) - timedelta(
                days=random.randint(1, 10), hours=random.randint(0, 23)
            )
            exit_time = entry_time + timedelta(hours=random.randint(1, 24))
            entry_price = _price(sym)
            exit_price = _price(sym)
            size = random.uniform(0.01, 0.1)
            quantity = (base_balance * size) / entry_price
            pnl = (
                (exit_price - entry_price) * quantity
                if side == PositionSide.LONG
                else (entry_price - exit_price) * quantity
            )
            db.log_trade(
                session_id=sid,
                symbol=sym,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                quantity=quantity,
                entry_time=entry_time,
                exit_time=exit_time,
                pnl=pnl,
                exit_reason=random.choice(
                    ["take_profit", "stop_loss", "manual_close", "time_exit"]
                ),
                strategy_name=random.choice(strategies),
                confidence_score=random.uniform(0.3, 0.95),
                order_id=f"order_{i}_{random.randint(1000, 9999)}",
                stop_loss=entry_price * 0.95,
                take_profit=entry_price * 1.05,
                strategy_config={},
            )

        print("✅ Database population completed successfully!")
        return 0
    except Exception as e:
        logger = logging.getLogger("atb.data")
        logger.error(f"Failed to populate database: {e}")
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("data", help="Data utilities (download, cache, populate)")
    sub = p.add_subparsers(dest="data_cmd", required=True)

    p_dl = sub.add_parser("download", help="Download market data from Binance")
    p_dl.add_argument("symbol", help="Trading pair symbol (e.g., BTC-USD, ETH-USD, BTCUSDT)")
    p_dl.add_argument("--timeframe", default="1h")
    p_dl.add_argument("--start_date", default=None)
    p_dl.add_argument("--end_date", default=None)
    p_dl.add_argument("--output_dir", default="tests/data")
    p_dl.add_argument("--format", choices=["feather", "csv"], default="feather")
    p_dl.set_defaults(func=_download)

    p_prefill = sub.add_parser("prefill-cache", help="Prefill Binance cache for symbols/timeframes")
    p_prefill.add_argument("--symbols", nargs="+", required=True)
    p_prefill.add_argument("--timeframes", nargs="+", required=True)
    p_prefill.add_argument("--start")
    p_prefill.add_argument("--end")
    p_prefill.add_argument("--years", type=int, default=3)
    p_prefill.add_argument("--cache_dir")
    p_prefill.add_argument("--cache_ttl_hours", type=int, default=24)
    p_prefill.set_defaults(func=_prefill)

    p_preload = sub.add_parser(
        "preload-offline", help="Pre-load cache with 10 years of data for offline backtesting"
    )
    p_preload.add_argument(
        "--symbols",
        nargs="+",
        default=[
            "BTCUSDT",
            "ETHUSDT",
            "BNBUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "XRPUSDT",
            "DOTUSDT",
            "LINKUSDT",
            "LTCUSDT",
            "AVAXUSDT",
        ],
        help="Trading pair symbols to download (default: top 10 coins)",
    )
    p_preload.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h", "4h", "1d"],
        help="Timeframes to download (default: 1h, 4h, 1d)",
    )
    p_preload.add_argument(
        "--years-back", type=int, default=10, help="Number of years back to download (default: 10)"
    )
    p_preload.add_argument("--cache-dir", help="Cache directory override")
    p_preload.add_argument(
        "--force-refresh", action="store_true", help="Force refresh existing cache entries"
    )
    p_preload.add_argument(
        "--test-offline", action="store_true", help="Test offline access after pre-loading"
    )
    p_preload.set_defaults(func=_preload_offline)

    p_cache = sub.add_parser("cache-manager", help="Cache manager")
    p_cache.add_argument(
        "subcmd", choices=["info", "list", "clear", "clear-old"], help="Cache action"
    )
    p_cache.add_argument("--cache-dir", default=None)
    p_cache.add_argument("--detailed", action="store_true")
    p_cache.add_argument("--force", action="store_true")
    p_cache.add_argument("--hours", type=int, default=24)
    p_cache.set_defaults(func=_cache_manager)

    p_dummy = sub.add_parser("populate-dummy", help="Populate DB with dummy data")
    p_dummy.add_argument("--trades", type=int, default=50)
    p_dummy.add_argument("--database_url")
    p_dummy.add_argument("--confirm", action="store_true")
    p_dummy.set_defaults(func=_populate_dummy)
