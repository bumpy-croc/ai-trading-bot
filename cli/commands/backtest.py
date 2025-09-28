from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta

# Ensure project root and src are in sys.path for absolute imports
from src.utils.project_paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

from src.utils.logging_config import configure_logging
from src.utils.symbol_factory import SymbolFactory

logger = logging.getLogger("atb.backtest")


def _load_strategy(strategy_name: str):
    try:
        if strategy_name == "ml_basic":
            from src.strategies.ml_basic import MlBasic

            return MlBasic()
        if strategy_name == "ml_sentiment":
            from src.strategies.ml_sentiment import MlSentiment

            return MlSentiment()
        if strategy_name == "bear":
            from src.strategies.bear import BearStrategy

            return BearStrategy()
        if strategy_name == "bull":
            from src.strategies.bull import Bull

            return Bull()
        if strategy_name == "ml_adaptive":
            from src.strategies.ml_adaptive import MlAdaptive

            return MlAdaptive()
        if strategy_name == "ensemble_weighted":
            from src.strategies.ensemble_weighted import EnsembleWeighted

            return EnsembleWeighted()
        if strategy_name == "momentum_leverage":
            from src.strategies.momentum_leverage import MomentumLeverage

            return MomentumLeverage()
        print(f"Unknown strategy: {strategy_name}")
        print("Available strategies: ml_basic, ml_sentiment, ml_adaptive, bear, bull, ensemble_weighted, momentum_leverage")
        raise SystemExit(1)
    except Exception as exc:
        logger.error(f"Error loading strategy: {exc}")
        raise


def _get_date_range(args):
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
        start_date = end_date - timedelta(days=30)
    return start_date, end_date


def _handle(ns: argparse.Namespace) -> int:
    try:
        from src.backtesting.engine import Backtester
        from src.data_providers.feargreed_provider import FearGreedProvider
        from src.risk.risk_manager import RiskParameters

        configure_logging()

        start_date, end_date = _get_date_range(ns)
        
        # Check if regime-aware backtesting is requested
        enable_regime_switching = hasattr(ns, 'regime_aware') and ns.regime_aware
        
        strategy = _load_strategy(ns.strategy)
        logger.info(f"Loaded strategy: {strategy.name}")

        # Provider
        if ns.provider == "coinbase":
            from src.data_providers.coinbase_provider import CoinbaseProvider

            provider = CoinbaseProvider()
        else:
            from src.data_providers.binance_provider import BinanceProvider

            provider = BinanceProvider()
        if ns.no_cache:
            data_provider = provider
            logger.info("Data caching disabled")
        else:
            from src.data_providers.cached_data_provider import CachedDataProvider

            # Determine appropriate cache TTL based on provider state
            from src.utils.cache_utils import get_cache_ttl_for_provider
            
            cache_ttl = get_cache_ttl_for_provider(provider, ns.cache_ttl)
            data_provider = CachedDataProvider(provider, cache_ttl_hours=cache_ttl)
            logger.info(f"Using cached data provider (TTL: {cache_ttl} hours)")
            cache_info = data_provider.get_cache_info()
            logger.info(
                f"Cache info: {cache_info['total_files']} files, {cache_info['total_size_mb']} MB"
            )

        sentiment_provider = None
        if ns.use_sentiment:
            sentiment_provider = FearGreedProvider()
            logger.info("Using sentiment analysis in backtest")

        risk_params = RiskParameters(
            base_risk_per_trade=ns.risk_per_trade,
            max_risk_per_trade=ns.max_risk_per_trade,
            max_drawdown=ns.max_drawdown,
        )

        # Default to no database logging for performance, unless explicitly enabled
        enable_db_logging = ns.log_to_db
        
        # Setup regime switching parameters if enabled
        regime_config = None
        strategy_mapping = None
        switching_config = None
        
        if enable_regime_switching:
            from src.live.regime_strategy_switcher import RegimeStrategyMapping, SwitchingConfig
            from src.regime.detector import RegimeConfig
            
            regime_config = RegimeConfig()
            strategy_mapping = RegimeStrategyMapping()
            switching_config = SwitchingConfig()
            logger.info("Regime-aware backtesting enabled")
        
        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            sentiment_provider=sentiment_provider,
            risk_parameters=risk_params,
            initial_balance=ns.initial_balance,
            log_to_database=enable_db_logging,
            enable_regime_switching=enable_regime_switching,
            regime_config=regime_config,
            strategy_mapping=strategy_mapping,
            switching_config=switching_config,
        )

        trading_symbol = (
            SymbolFactory.to_exchange_symbol(ns.symbol, ns.provider)
            if ns.symbol != "BTC-USD"
            else strategy.get_trading_pair()
        )

        results = backtester.run(
            symbol=trading_symbol, timeframe=ns.timeframe, start=start_date, end=end_date
        )

        # Display results based on whether regime switching was enabled
        if enable_regime_switching:
            print("\nRegime-Aware Backtest Results:")
            print("=" * 60)
            print(f"Initial Strategy: {strategy.name}")
            print(f"Final Strategy: {results.get('final_strategy', 'N/A')}")
            print(f"Total Strategy Switches: {results.get('total_strategy_switches', 0)}")
        else:
            print("\nBacktest Results:")
            print("=" * 50)
            print(f"Strategy: {strategy.name}")
            
        print(f"Symbol: {trading_symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {ns.timeframe}")
        print(f"Using Sentiment: {ns.use_sentiment}")
        print(f"Using Cache: {not ns.no_cache}")
        print(f"Database Logging: {enable_db_logging}")
        print(f"Regime Switching: {enable_regime_switching}")
        print("-" * (60 if enable_regime_switching else 50))
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print(f"Hold Return: {results['hold_return']:.2f}%")
        print(f"Trading vs Hold: {results['trading_vs_hold_difference']:+.2f}%")
        print("=" * (60 if enable_regime_switching else 50))

        # Show strategy switches if any occurred
        if enable_regime_switching and results.get('strategy_switches'):
            print("\nStrategy Switches:")
            print("-" * 60)
            for switch in results['strategy_switches']:
                print(f"{switch['timestamp']}: {switch['old_strategy']} -> {switch['new_strategy']} "
                      f"(regime: {switch['regime']}, confidence: {switch['confidence']:.2f})")
            print("=" * 60)

        if enable_db_logging and results.get("session_id"):
            print(f"Database Session ID: {results['session_id']}")
            print("=" * 50)

        if results.get("yearly_returns"):
            print("Yearly Returns:")
            print(f"{'Year':<8} {'Return (%)':>12}")
            for year in sorted(results["yearly_returns"].keys()):
                print(f"{year:<8} {results['yearly_returns'][year]:>12.2f}")
            print("=" * 50)

        if not ns.no_cache:
            final_cache_info = data_provider.get_cache_info()
            logger.info(
                f"Final cache info: {final_cache_info['total_files']} files, {final_cache_info['total_size_mb']} MB"
            )

        try:
            import re

            duration_years = round((end_date - start_date).days / 365.25, 2)
            timestamp_for_file = datetime.now().strftime("%Y%m%d_%H%M%S")
            sanitized_strategy_name = re.sub(r"[^a-zA-Z0-9_-]", "_", strategy.name)
            filename = f"{timestamp_for_file}_{sanitized_strategy_name}_{duration_years}yrs.json"
            logs_dir = PROJECT_ROOT / "logs" / "backtest"
            logs_dir.mkdir(parents=True, exist_ok=True)
            filepath = logs_dir / filename
            with open(filepath, "w") as _f:
                json.dump(
                    {
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "strategy": strategy.name,
                        "symbol": trading_symbol,
                        "timeframe": ns.timeframe,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "duration_years": duration_years,
                        "initial_balance": ns.initial_balance,
                        "use_sentiment": ns.use_sentiment,
                        "use_cache": not ns.no_cache,
                        "database_logging": enable_db_logging,
                        "results": results,
                    },
                    _f,
                    indent=2,
                )
            logger.info(f"Backtest log saved to {filepath.relative_to(PROJECT_ROOT)}")
        except Exception as log_err:
            logger.warning(f"Failed to write backtest log: {log_err}")

        if sentiment_provider is not None:
            df = data_provider.get_historical_data(ns.symbol, ns.timeframe, start_date, end_date)
            sentiment_df = sentiment_provider.get_historical_sentiment(
                ns.symbol, start_date, end_date
            )
            if not sentiment_df.empty:
                sentiment_df = sentiment_provider.aggregate_sentiment(
                    sentiment_df, window=ns.timeframe
                )
                aligned_df = df.join(sentiment_df, how="left")
                print(f"Shape of aligned DataFrame: {aligned_df.shape}")
                if aligned_df.empty:
                    print("Warning: aligned DataFrame is empty. No file will be written.")
                output_path = PROJECT_ROOT / "data" / "sentiment_aligned_output.csv"
                try:
                    aligned_df.to_csv(output_path)
                    print(f"Aligned sentiment and price data saved to {output_path}")
                except Exception as file_err:
                    print(f"Error writing {output_path}: {file_err}")
                    logger.error(f"Error writing {output_path}: {file_err}")

        return 0
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"Error running backtest: {exc}")
        return 1




def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("backtest", help="Run strategy backtest")
    p.add_argument("strategy", help="Strategy name - e.g., ml_basic")
    p.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    p.add_argument("--timeframe", default="1h", help="Candle timeframe")
    p.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    p.add_argument("--start", help="Start date - YYYY-MM-DD")
    p.add_argument("--end", help="End date - YYYY-MM-DD")
    from src.config.constants import DEFAULT_INITIAL_BALANCE

    p.add_argument(
        "--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE, help="Initial balance"
    )
    p.add_argument("--risk-per-trade", type=float, default=0.01, help="Risk per trade - 1 percent equals 0.01")
    p.add_argument("--max-risk-per-trade", type=float, default=0.02, help="Maximum risk per trade")
    p.add_argument(
        "--use-sentiment", action="store_true", help="Use sentiment analysis in backtest"
    )
    p.add_argument("--no-cache", action="store_true", help="Disable data caching")
    p.add_argument("--cache-ttl", type=int, default=24, help="Cache TTL in hours - default: 24")
    p.add_argument(
        "--log-to-db", action="store_true", help="Enable database logging for this backtest - slower but provides detailed logs"
    )
    p.add_argument(
        "--provider",
        choices=["coinbase", "binance"],
        default="binance",
        help="Exchange provider to use - default: binance",
    )
    p.add_argument(
        "--max-drawdown",
        type=float,
        default=0.5,
        help="Maximum drawdown before stopping - default: 0.5 (50 percent)",
    )
    p.add_argument(
        "--regime-aware",
        action="store_true",
        help="Enable regime-aware backtesting with automatic strategy switching",
    )
    p.set_defaults(func=_handle)
