from __future__ import annotations

import argparse
import os
from pathlib import Path

import ccxt
import pandas as pd

from utils.symbol_factory import SymbolFactory


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
    from datetime import datetime

    from config.paths import get_cache_dir
    from data_providers.binance_provider import BinanceProvider
    from data_providers.cached_data_provider import CachedDataProvider

    def _normalize_symbols(raw):
        normalized = []
        for s in raw:
            s = s.strip().upper()
            if "-" in s or "/" in s:
                s = SymbolFactory.to_exchange_symbol(s, "binance")
            normalized.append(s)
        return normalized

    def _year_chunks(start: datetime, end: datetime):
        from datetime import timedelta

        chunks = []
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

    end = datetime.strptime(ns.end, "%Y-%m-%d") if ns.end else datetime.now()
    if ns.start:
        start = datetime.strptime(ns.start, "%Y-%m-%d")
    else:
        cy = end.year
        start = datetime(cy - ns.years, 1, 1)

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
                    print(f"Error caching {symbol} {tf} {y_start.year}: {e}")
    print("Done.")
    return 0


def _cache_manager(ns: argparse.Namespace) -> int:
    import pickle
    from datetime import datetime

    from config.paths import get_cache_dir
    from data_providers.binance_provider import BinanceProvider
    from data_providers.cached_data_provider import CachedDataProvider

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

        now = datetime.now()
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

    from database.manager import DatabaseManager
    from database.models import PositionSide, TradeSource

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

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
            entry_time = datetime.utcnow() - timedelta(
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
        logger = logging.getLogger(__name__)
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
    p_prefill.add_argument("--symbols", nargs="+", default=["BTCUSDT"])
    p_prefill.add_argument("--timeframes", nargs="+", default=["1h"])
    p_prefill.add_argument("--years", type=int, default=8)
    p_prefill.add_argument("--start", type=str, default=None)
    p_prefill.add_argument("--end", type=str, default=None)
    p_prefill.add_argument("--cache-ttl-hours", type=int, default=24)
    p_prefill.add_argument("--cache-dir", type=str, default=None)
    p_prefill.set_defaults(func=_prefill)

    p_cache = sub.add_parser("cache-manager", help="Cache manager")
    p_cache.add_argument(
        "subcmd", choices=["info", "list", "clear", "clear-old"], help="Cache action"
    )
    p_cache.add_argument("--cache-dir", default=None)
    p_cache.add_argument("--detailed", action="store_true")
    p_cache.add_argument("--force", action="store_true")
    p_cache.add_argument("--hours", type=int, default=24)
    p_cache.set_defaults(func=_cache_manager)

    p_dummy = sub.add_parser("populate-dummy", help="Populate dummy data into DB")
    p_dummy.add_argument("--trades", type=int, default=100)
    p_dummy.add_argument("--database-url", type=str, default=None)
    p_dummy.add_argument("--confirm", action="store_true")
    p_dummy.set_defaults(func=_populate_dummy)
