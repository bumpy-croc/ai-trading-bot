#!/usr/bin/env python3
"""
Live Trading Bot Runner

This module provides the live trading engine with proper configuration and safety checks.
It supports both paper trading (simulation) and live trading with real money.
"""

import argparse
import logging
import sys

from src.config.constants import DEFAULT_INITIAL_BALANCE
from src.data_providers.mock_data_provider import MockDataProvider
from src.live.trading_engine import LiveTradingEngine
from src.risk.risk_manager import RiskParameters

# Import strategies
from src.strategies.ml_basic import create_ml_basic_strategy
from src.utils.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger("live_trading")


def load_strategy(strategy_name: str):
    """Load a strategy by name"""
    strategies = {
        "ml_basic": create_ml_basic_strategy,
    }

    # Lazy import for optional strategies to keep startup fast

    if strategy_name not in strategies or strategies[strategy_name] is None:
        logger.error(f"Unknown strategy: {strategy_name}")
        logger.info(
            f"Available strategies: {list(k for k, v in strategies.items() if v is not None)}"
        )
        sys.exit(1)

    try:
        strategy_factory = strategies[strategy_name]
        if not callable(strategy_factory):
            msg = f"Strategy factory for {strategy_name} must be callable"
            logger.error(msg)
            raise TypeError(msg)
        strategy = strategy_factory()
        logger.info(f"Loaded strategy: {strategy.name}")
        return strategy
    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run live trading bot")

    # Strategy selection
    parser.add_argument("strategy", help="Strategy name (e.g., ml_basic)")

    # Trading parameters
    parser.add_argument(
        "--symbol", default="BTCUSDT", help="Trading pair symbol (e.g., BTCUSDT, ETHUSDT)"
    )
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument(
        "--balance", type=float, default=DEFAULT_INITIAL_BALANCE, help="Initial balance"
    )
    parser.add_argument(
        "--max-position", type=float, default=0.1, help="Max position size (0.1 = 10% of balance)"
    )
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds")

    # Trading mode
    parser.add_argument(
        "--paper-trading", action="store_true", help="Run in paper trading mode (safe)"
    )
    parser.add_argument(
        "--live-trading",
        action="store_true",
        help="Run in live trading mode (requires --i-understand-the-risks)",
    )
    parser.add_argument(
        "--i-understand-the-risks",
        action="store_true",
        help="Explicitly acknowledge the risks of live trading",
    )

    # Data provider options
    parser.add_argument(
        "--provider",
        choices=["binance", "coinbase"],
        default="binance",
        help="Data provider to use",
    )
    parser.add_argument("--mock-data", action="store_true", help="Use mock data for testing")
    parser.add_argument("--no-cache", action="store_true", help="Disable data caching")

    # Risk management
    parser.add_argument(
        "--risk-per-trade",
        type=float,
        default=0.02,
        help="Base risk per trade (0.02 = 2% of balance)",
    )
    parser.add_argument(
        "--max-risk-per-trade",
        type=float,
        default=0.05,
        help="Maximum risk per trade (0.05 = 5% of balance)",
    )
    parser.add_argument(
        "--max-drawdown",
        type=float,
        default=0.20,
        help="Maximum drawdown before stopping (0.20 = 20%)",
    )

    # Logging and monitoring
    parser.add_argument("--log-trades", action="store_true", help="Log individual trades to file")
    parser.add_argument("--webhook-url", help="Webhook URL for trade notifications")
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=3600,
        help="Account snapshot interval in seconds",
    )

    # Sentiment analysis
    parser.add_argument("--use-sentiment", action="store_true", help="Enable sentiment analysis")

    return parser.parse_args()


def validate_configuration(args):
    """Validate configuration and safety checks"""
    if args.live_trading and not args.i_understand_the_risks:
        logger.error("❌ Live trading requires explicit risk acknowledgment")
        logger.error("   Use --i-understand-the-risks flag")
        return False

    if args.live_trading:
        logger.warning("🚨 LIVE TRADING MODE - REAL MONEY WILL BE USED")
        logger.warning("   Strategy: %s", args.strategy)
        logger.warning("   Symbol: %s", args.symbol)
        logger.warning("   Balance: $%.2f", args.balance)
        logger.warning("   Max position: %.1f%%", args.max_position * 100)
    else:
        logger.info("📄 PAPER TRADING MODE - No real money will be used")

    # Validate strategy exists
    try:
        load_strategy(args.strategy)
    except Exception:
        return False

    return True


def print_startup_info(args, strategy):
    """Print startup information"""
    logger.info("🤖 AI Trading Bot Starting")
    logger.info("   Strategy: %s", strategy.name)
    logger.info("   Symbol: %s", args.symbol)
    logger.info("   Timeframe: %s", args.timeframe)
    logger.info("   Initial Balance: $%.2f", args.balance)
    logger.info("   Max Position Size: %.1f%%", args.max_position * 100)
    logger.info("   Check Interval: %d seconds", args.check_interval)
    logger.info("   Mode: %s", "LIVE TRADING" if args.live_trading else "Paper Trading")


def main():
    """Main entry point for live trading"""
    try:
        args = parse_args()

        # Validate configuration
        if not validate_configuration(args):
            sys.exit(1)

        # Load strategy
        strategy = load_strategy(args.strategy)

        # Show startup information
        print_startup_info(args, strategy)

        # Initialize data provider
        logger.info("Initializing data providers...")
        if args.mock_data:
            data_provider = MockDataProvider(interval_seconds=5)  # 5s candles for rapid testing
            logger.info("Using MockDataProvider for rapid testing")
        else:
            if args.provider == "coinbase":
                from src.data_providers.coinbase_provider import CoinbaseProvider

                provider = CoinbaseProvider()
            else:
                from src.data_providers.binance_provider import BinanceProvider

                provider = BinanceProvider()
            if args.no_cache:
                data_provider = provider
                logger.info("Data caching disabled")
            else:
                from src.data_providers.cached_data_provider import CachedDataProvider

                data_provider = CachedDataProvider(provider, cache_ttl_hours=1)
                logger.info("Data caching enabled (1 hour TTL)")

        # Initialize sentiment provider if requested
        sentiment_provider = None
        if args.use_sentiment:
            logger.warning(
                "❌ Sentiment analysis not available - sentiment providers have been removed"
            )
            logger.info("Continuing without sentiment analysis...")

        # Set up risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=args.risk_per_trade,
            max_risk_per_trade=args.max_risk_per_trade,
            max_drawdown=args.max_drawdown,
        )

        # Create trading engine
        logger.info("Creating live trading engine...")
        engine = LiveTradingEngine(
            strategy=strategy,
            data_provider=data_provider,
            sentiment_provider=sentiment_provider,
            risk_parameters=risk_params,
            check_interval=args.check_interval,
            initial_balance=args.balance,
            max_position_size=args.max_position,
            enable_live_trading=args.live_trading,
            log_trades=args.log_trades,
            alert_webhook_url=args.webhook_url,
            account_snapshot_interval=args.snapshot_interval,
            provider=args.provider,
        )

        # Final safety check for live trading
        if args.live_trading:
            logger.critical("🚨 FINAL WARNING: STARTING LIVE TRADING IN 10 SECONDS")
            logger.critical("🚨 Press Ctrl+C now to cancel")
            import time

            for i in range(10, 0, -1):
                print(f"Starting in {i}...", end="\r")
                time.sleep(1)
            print("\n🚀 LIVE TRADING STARTED")

        # Start trading
        logger.info(f"Starting trading engine for {args.symbol} on {args.timeframe}")
        engine.start(args.symbol, args.timeframe)

    except KeyboardInterrupt:
        logger.info("🛑 Trading stopped by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("🏁 Trading session ended")


if __name__ == "__main__":
    main()
