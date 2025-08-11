#!/usr/bin/env python3
"""
Live Trading Bot Runner

This script starts the live trading engine with proper configuration and safety checks.
It supports both paper trading (simulation) and live trading with real money.

Usage:
    # Paper trading (default - safe)
    python run_live_trading.py ml_basic --symbol BTCUSDT --paper-trading
    
    # Live trading (requires explicit confirmation)
    python run_live_trading.py ml_basic --symbol BTCUSDT --live-trading --i-understand-the-risks
    
    # With custom settings
    python run_live_trading.py ml_basic --symbol BTCUSDT --balance 5000 --max-position 0.05
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
# Also add the 'src' directory to PYTHONPATH so that modules like 'config', 'data_providers', etc. can be imported correctly when this script is executed directly.
root_dir = Path(__file__).parent.parent
src_dir = root_dir / "src"
sys.path.append(str(src_dir))

from config import get_config
from config.constants import DEFAULT_INITIAL_BALANCE
from data_providers.binance_provider import BinanceProvider
from data_providers.cached_data_provider import CachedDataProvider
from data_providers.senticrypt_provider import SentiCryptProvider
from data_providers.mock_data_provider import MockDataProvider
from risk.risk_manager import RiskParameters
from live.trading_engine import LiveTradingEngine

# Import strategies
from strategies.ml_basic import MlBasic
from strategies.test_high_frequency import TestHighFrequencyStrategy

# Configure logging - ensure the logs directory exists at project root
project_root = Path(__file__).parent.parent  # ai-trading-bot/
logs_dir = project_root / "logs"
logs_dir.mkdir(exist_ok=True)
log_file_path = logs_dir / f"live_trading_{datetime.now().strftime('%Y%m%d')}.log"

from utils.logging_config import configure_logging
configure_logging()
logger = logging.getLogger('live_trading')

def load_strategy(strategy_name: str):
    """Load a strategy by name"""
    strategies = {
        'ml_basic': MlBasic,
        'test_high_frequency': TestHighFrequencyStrategy
    }
    
    if strategy_name not in strategies:
        logger.error(f"Unknown strategy: {strategy_name}")
        logger.info(f"Available strategies: {list(strategies.keys())}")
        sys.exit(1)
    
    try:
        strategy_class = strategies[strategy_name]
        strategy = strategy_class() if callable(strategy_class) else strategy_class()
        logger.info(f"Loaded strategy: {strategy.name}")
        return strategy
    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Run live trading bot')
    
    # Strategy selection
    parser.add_argument('strategy', help='Strategy name (e.g., ml_basic)')
    
    # Trading parameters
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol (e.g., BTCUSDT, ETHUSDT)')
    parser.add_argument('--timeframe', default='1h', help='Candle timeframe')
    parser.add_argument('--balance', type=float, default=DEFAULT_INITIAL_BALANCE, help='Initial balance')
    parser.add_argument('--max-position', type=float, default=0.1, help='Max position size (0.1 = 10% of balance)')
    parser.add_argument('--check-interval', type=int, default=60, help='Check interval in seconds')
    
    # Trading mode
    parser.add_argument('--paper-trading', action='store_true', help='Run in paper trading mode (safe)')
    parser.add_argument('--live-trading', action='store_true', help='Run with real money (DANGEROUS)')
    parser.add_argument('--i-understand-the-risks', action='store_true', 
                       help='Required confirmation for live trading')
    
    # Risk management
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (1% = 0.01)')
    parser.add_argument('--max-risk-per-trade', type=float, default=0.02, help='Maximum risk per trade')
    parser.add_argument('--max-drawdown', type=float, default=0.2, help='Maximum drawdown before stopping')
    
    # Data providers
    parser.add_argument('--use-sentiment', action='store_true', help='Use sentiment analysis')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching')
    parser.add_argument('--mock-data', action='store_true', help='Use mock data provider for rapid testing')
    parser.add_argument('--provider', choices=['coinbase', 'binance'], default='binance', help='Exchange provider to use (default: binance)')
    
    # Monitoring
    parser.add_argument('--webhook-url', help='Webhook URL for alerts')
    parser.add_argument('--log-trades', action='store_true', default=True, help='Log trades to file')
    parser.add_argument('--snapshot-interval', type=int, default=1800, 
                       help='Account snapshot interval in seconds (default: 1800 = 30 minutes, 0 = disable)')
    
    return parser.parse_args()

def validate_configuration(args):
    """Validate trading configuration and safety checks"""
    logger.info("Validating configuration...")
    
    # Skip API credential checks when running in paper trading mode
    if args.paper_trading and not args.live_trading:
        logger.info("Paper trading mode detected – skipping API credential validation")
    else:
        # Check API credentials
        config = get_config()
        try:
            api_key = config.get('BINANCE_API_KEY')
            secret = config.get('BINANCE_API_SECRET')

            if not api_key or not secret:
                raise ValueError("Missing key or secret")

            logger.info("✅ Binance API credentials detected")
        except ValueError:
            logger.error("❌ Binance API credentials not found!")
            logger.error("Please ensure BOTH variables are set in Railway (or other providers):")
            logger.error("  - BINANCE_API_KEY")
            logger.error("  - BINANCE_API_SECRET")
            return False
    
    # Validate trading mode
    if args.live_trading:
        if not args.i_understand_the_risks:
            logger.error("❌ LIVE TRADING REQUIRES EXPLICIT RISK ACKNOWLEDGMENT")
            logger.error("Add --i-understand-the-risks flag if you want to trade with real money")
            return False
        
        logger.warning("⚠️  LIVE TRADING MODE ENABLED - REAL MONEY AT RISK")
        logger.warning("⚠️  This bot will execute real trades on your Binance account")
        
        # Additional confirmation for live trading
        confirmation = input("\nType 'I UNDERSTAND' to continue with live trading: ")
        if confirmation != "I UNDERSTAND":
            logger.info("Live trading cancelled by user")
            return False
    else:
        logger.info("✅ Paper trading mode - no real orders will be executed")
    
    # Validate parameters
    if args.max_position > 0.5:
        logger.error("Maximum position size too large (>50%). This is dangerous.")
        return False
    
    if args.balance < 100:
        logger.error("Balance too small. Minimum recommended: $100")
        return False
    
    if args.check_interval < 30:
        logger.warning("Check interval very short. May cause rate limiting.")
    
    logger.info("✅ Configuration validated")
    return True

def print_startup_info(args, strategy):
    """Print startup information"""
    print("\n" + "="*70)
    print("🤖 LIVE TRADING BOT STARTUP")
    print("="*70)
    print(f"Strategy: {strategy.name}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Max Position Size: {args.max_position*100:.1f}% of balance")
    print(f"Check Interval: {args.check_interval}s")
    print(f"Risk Per Trade: {args.risk_per_trade*100:.1f}%")
    print(f"Trading Mode: {'🔴 LIVE TRADING' if args.live_trading else '📄 PAPER TRADING'}")
    print(f"Sentiment Analysis: {'✅ Enabled' if args.use_sentiment else '❌ Disabled'}")
    print(f"Data Caching: {'❌ Disabled' if args.no_cache else '✅ Enabled'}")
    print(f"Trade Logging: {'✅ Enabled' if args.log_trades else '❌ Disabled'}")
    print(f"Alerts: {'✅ Configured' if args.webhook_url else '❌ Not configured'}")
    snapshot_info = f"{args.snapshot_interval}s ({args.snapshot_interval//60}min)" if args.snapshot_interval > 0 else "Disabled"
    print(f"Account Snapshots: {snapshot_info}")
    print("="*70)

def main():
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
            if args.provider == 'coinbase':
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
                data_provider = CachedDataProvider(provider, cache_ttl_hours=1)
                logger.info("Data caching enabled (1 hour TTL)")
        
        # Initialize sentiment provider if requested
        sentiment_provider = None
        if args.use_sentiment:
            try:
                sentiment_provider = SentiCryptProvider(
                    csv_path='data/senticrypt_sentiment_data.csv',
                    live_mode=True,
                    cache_duration_minutes=15
                )
                logger.info("✅ Sentiment provider initialized for live trading")
            except Exception as e:
                logger.error(f"❌ Failed to initialize sentiment provider: {e}")
                logger.info("Continuing without sentiment analysis...")
        
        # Set up risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=args.risk_per_trade,
            max_risk_per_trade=args.max_risk_per_trade,
            max_drawdown=args.max_drawdown
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
            provider=args.provider
        )
        
        # Final safety check for live trading
        if args.live_trading:
            logger.critical("🚨 FINAL WARNING: STARTING LIVE TRADING IN 10 SECONDS")
            logger.critical("🚨 Press Ctrl+C now to cancel")
            import time
            for i in range(10, 0, -1):
                print(f"Starting in {i}...", end='\r')
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