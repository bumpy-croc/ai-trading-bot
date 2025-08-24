#!/usr/bin/env python3
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Import project modules
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.backtesting.engine import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.strategies.ml_basic import MlBasic
from src.utils.symbol_factory import SymbolFactory


def parse_args():
    p = argparse.ArgumentParser(description="Compare MlBasic on BTCUSDT vs ETHUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--initial-balance", type=float, default=10000.0)
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--cache-ttl", type=int, default=24)
    p.add_argument("--no-db", action="store_true")
    return p.parse_args()


def get_date_range(args):
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.utcnow()
    else:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=args.days)
    return start_date, end_date


def get_provider(use_cache: bool, cache_ttl: int):
    # Try Binance, else Coinbase
    primary = BinanceProvider()
    provider = primary
    if use_cache:
        provider = CachedDataProvider(provider, cache_ttl_hours=cache_ttl)
    return provider


def run_backtest_for_symbol(
    symbol: str,
    timeframe: str,
    start,
    end,
    initial_balance: float,
    use_cache: bool,
    cache_ttl: int,
    no_db: bool,
):
    # Try with Binance; if empty, retry with Coinbase directly (no cache swap complexity)
    provider = get_provider(use_cache, cache_ttl)
    strategy = MlBasic()
    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=None,
        initial_balance=initial_balance,
        log_to_database=not no_db,
    )
    results = backtester.run(symbol=symbol, timeframe=timeframe, start=start, end=end)
    if results.get("total_trades") == 0 and results.get("final_balance") == initial_balance:
        # Possibly no data; try Coinbase
        cb = CoinbaseProvider()
        backtester_cb = Backtester(
            strategy=MlBasic(),
            data_provider=cb,
            sentiment_provider=None,
            risk_parameters=None,
            initial_balance=initial_balance,
            log_to_database=False,
        )
        try:
            # Convert symbol to Coinbase product id format using SymbolFactory (e.g., ETH-USD)
            sym = SymbolFactory.to_exchange_symbol(symbol, "coinbase")
            return backtester_cb.run(symbol=sym, timeframe=timeframe, start=start, end=end)
        except Exception:
            return results
    return results


def main():
    args = parse_args()
    start, end = get_date_range(args)
    print(
        f"Running MlBasic comparison | timeframe={args.timeframe} | {start.date()} → {end.date()}"
    )

    btc_results = run_backtest_for_symbol(
        "BTCUSDT",
        args.timeframe,
        start,
        end,
        args.initial_balance,
        not args.no_cache,
        args.cache_ttl,
        args.no_db,
    )
    eth_results = run_backtest_for_symbol(
        "ETHUSDT",
        args.timeframe,
        start,
        end,
        args.initial_balance,
        not args.no_cache,
        args.cache_ttl,
        args.no_db,
    )

    def fmt(res):
        # Handle empty/partial dictionaries
        if not res or "final_balance" not in res:
            return {
                "trades": 0,
                "win_rate": "0.00%",
                "total_return": "0.00%",
                "annualized": "0.00%",
                "max_dd": "0.00%",
                "final_balance": "$0.00",
                "sharpe": "0.00",
            }
        return {
            "trades": res.get("total_trades", 0),
            "win_rate": f"{res.get('win_rate', 0.0):.2f}%",
            "total_return": f"{res.get('total_return', 0.0):.2f}%",
            "annualized": f"{res.get('annualized_return', 0.0):.2f}%",
            "max_dd": f"{res.get('max_drawdown', 0.0):.2f}%",
            "final_balance": f"${res.get('final_balance', 0.0):.2f}",
            "sharpe": f"{res.get('sharpe_ratio', 0.0):.2f}",
        }

    print("\nResults (MlBasic):")
    print("Symbol  | Trades | Win%   | TotalRet | Annualized | MaxDD  | Sharpe | Final Balance")
    print("--------|--------|--------|----------|------------|--------|--------|---------------")
    fbtc, feth = fmt(btc_results), fmt(eth_results)
    print(
        f"BTCUSDT | {fbtc['trades']:^6} | {fbtc['win_rate']:^6} | {fbtc['total_return']:^8} | {fbtc['annualized']:^10} | {fbtc['max_dd']:^6} | {fbtc['sharpe']:^6} | {fbtc['final_balance']:^13}"
    )
    print(
        f"ETHUSDT | {feth['trades']:^6} | {feth['win_rate']:^6} | {feth['total_return']:^8} | {feth['annualized']:^10} | {feth['max_dd']:^6} | {feth['sharpe']:^6} | {feth['final_balance']:^13}"
    )


if __name__ == "__main__":
    main()
