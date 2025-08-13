#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from backtesting import Backtester
from config.constants import DEFAULT_INITIAL_BALANCE
from data_providers.cryptocompare_sentiment import CryptoCompareSentimentProvider
from risk import RiskParameters
from utils.logging_config import configure_logging
from utils.symbol_factory import SymbolFactory

logger = logging.getLogger("run_backtest")
project_root = Path(__file__).resolve().parents[1]


def load_strategy(strategy_name: str):
    """Load a strategy by name"""
    try:
        # Import strategies
        if strategy_name == "ml_basic":
            from strategies.ml_basic import MlBasic

            strategy = MlBasic()
        elif strategy_name == "ml_with_sentiment":
            from strategies.ml_with_sentiment import MlWithSentiment

            strategy = MlWithSentiment(use_sentiment=True)
        elif strategy_name == "ml_adaptive":
            from strategies.ml_adaptive import MlAdaptive

            strategy = MlAdaptive()
        elif strategy_name == "bear":
            from strategies.bear import BearStrategy

            strategy = BearStrategy()
        else:
            print(f"Unknown strategy: {strategy_name}")
            available_strategies = ["ml_basic", "ml_with_sentiment", "ml_adaptive", "bear"]
            print(f"Available strategies: {', '.join(available_strategies)}")
            sys.exit(1)

        # Create and return strategy instance
        return strategy

    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("strategy", help="Strategy name (e.g., ml_basic, ml_with_sentiment)")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE, help="Initial balance"
    )
    parser.add_argument(
        "--risk-per-trade", type=float, default=0.01, help="Risk per trade (1% = 0.01)"
    )
    parser.add_argument(
        "--max-risk-per-trade", type=float, default=0.02, help="Maximum risk per trade"
    )
    parser.add_argument(
        "--use-sentiment", action="store_true", help="Use sentiment analysis in backtest"
    )
    parser.add_argument("--no-cache", action="store_true", help="Disable data caching")
    parser.add_argument(
        "--cache-ttl", type=int, default=24, help="Cache TTL in hours (default: 24)"
    )
    parser.add_argument(
        "--no-db", action="store_true", help="Disable database logging for this backtest"
    )
    parser.add_argument(
        "--provider",
        choices=["coinbase", "binance"],
        default="binance",
        help="Exchange provider to use (default: binance)",
    )
    parser.add_argument(
        "--enable-short-trading",
        action="store_true",
        help="Enable short entries (recommended for bear strategy)",
    )
    return parser.parse_args()


def get_date_range(args):
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.now()
    elif args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Default to 30 days
    return start_date, end_date


def main() -> int:
    args = parse_args()

    try:
        configure_logging()
        # Calculate date range
        start_date, end_date = get_date_range(args)

        # Load the strategy
        strategy = load_strategy(args.strategy)
        logger.info(f"Loaded strategy: {strategy.name}")

        # Initialize data provider with caching
        if args.provider == "coinbase":
            from data_providers.coinbase_provider import CoinbaseProvider

            provider = CoinbaseProvider()
        else:
            from data_providers.binance_provider import BinanceProvider

            provider = BinanceProvider()
        if args.no_cache:
            data_provider = provider
            logger.info("Data caching disabled")
        else:
            from data_providers.cached_data_provider import CachedDataProvider

            data_provider = CachedDataProvider(provider, cache_ttl_hours=args.cache_ttl)
            logger.info(f"Using cached data provider (TTL: {args.cache_ttl} hours)")

            # Show cache info
            cache_info = data_provider.get_cache_info()
            logger.info(
                f"Cache info: {cache_info['total_files']} files, {cache_info['total_size_mb']} MB"
            )

        # Initialize sentiment provider if requested
        sentiment_provider = None
        if args.use_sentiment:
            sentiment_provider = CryptoCompareSentimentProvider()
            logger.info("Using sentiment analysis in backtest")

        # Set up risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=args.risk_per_trade, max_risk_per_trade=args.max_risk_per_trade
        )

        # Create and run backtester
        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            sentiment_provider=sentiment_provider,
            risk_parameters=risk_params,
            initial_balance=args.initial_balance,
            enable_short_trading=args.enable_short_trading,
            log_to_database=not args.no_db,  # Disable DB logging if --no-db is passed
        )

        # Use strategy-specific trading pair if not overridden by command line
        if args.symbol != "BTC-USD":
            trading_symbol = SymbolFactory.to_exchange_symbol(args.symbol, args.provider)
        else:
            trading_symbol = strategy.get_trading_pair()

        # Run backtest
        results = backtester.run(
            symbol=trading_symbol, timeframe=args.timeframe, start=start_date, end=end_date
        )

        # Print results
        print("\nBacktest Results:")
        print("=" * 50)
        print(f"Strategy: {strategy.name}")
        print(f"Symbol: {trading_symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Using Sentiment: {args.use_sentiment}")
        print(f"Using Cache: {not args.no_cache}")
        print(f"Database Logging: {not args.no_db}")
        print("-" * 50)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print("=" * 50)

        if not args.no_db and "session_id" in results and results["session_id"]:
            print(f"Database Session ID: {results['session_id']}")
            print("=" * 50)

        if "yearly_returns" in results and results["yearly_returns"]:
            print("Yearly Returns:")
            print(f"{'Year':<8} {'Return (%)':>12}")
            for year in sorted(results["yearly_returns"].keys()):
                print(f"{year:<8} {results['yearly_returns'][year]:>12.2f}")
            print("=" * 50)

        if not args.no_cache:
            final_cache_info = data_provider.get_cache_info()
            logger.info(
                f"Final cache info: {final_cache_info['total_files']} files, {final_cache_info['total_size_mb']} MB"
            )

        # Persist backtest run to log file
        try:
            import re

            duration_years = round((end_date - start_date).days / 365.25, 2)
            timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_strategy_name = re.sub(r"[^a-zA-Z0-9_-]", "_", strategy.name)
            filename = f"{timestamp_for_file}_{sanitized_strategy_name}_{duration_years}yrs.json"
            logs_dir = project_root / "logs" / "backtest"
            logs_dir.mkdir(parents=True, exist_ok=True)
            filepath = logs_dir / filename
            with open(filepath, "w") as _f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "strategy": strategy.name,
                        "symbol": trading_symbol,
                        "timeframe": args.timeframe,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "duration_years": duration_years,
                        "initial_balance": args.initial_balance,
                        "use_sentiment": args.use_sentiment,
                        "use_cache": not args.no_cache,
                        "database_logging": not args.no_db,
                        "results": results,
                    },
                    indent=2,
                )
            logger.info(f"Backtest log saved to {filepath.relative_to(project_root)}")
        except Exception as log_err:
            logger.warning(f"Failed to write backtest log: {log_err}")

        # Save aligned sentiment data if used
        if sentiment_provider is not None:
            df = data_provider.get_historical_data(
                args.symbol, args.timeframe, start_date, end_date
            )
            sentiment_df = sentiment_provider.get_historical_sentiment(
                args.symbol, start_date, end_date
            )
            if not sentiment_df.empty:
                sentiment_df = sentiment_provider.aggregate_sentiment(
                    sentiment_df, window=args.timeframe
                )
                aligned_df = df.join(sentiment_df, how="left")
                print(f"Shape of aligned DataFrame: {aligned_df.shape}")
                if aligned_df.empty:
                    print("Warning: aligned DataFrame is empty. No file will be written.")
                output_path = project_root / "data" / "sentiment_aligned_output.csv"
                try:
                    aligned_df.to_csv(output_path)
                    print(f"Aligned sentiment and price data saved to {output_path}")
                except Exception as file_err:
                    print(f"Error writing {output_path}: {file_err}")
                    logger.error(f"Error writing {output_path}: {file_err}")

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        logging.getLogger(__name__).exception("Backtest run failed: %s", e)
        raise
