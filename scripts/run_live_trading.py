#!/usr/bin/env python3
"""
Live Trading Runner

This script runs live trading using the new architecture with:
- TradeExecutor for consistent trade management
- SignalGenerator for standardized signal generation
- TradingDataRepository for data access
"""

import os
import sys
import asyncio
import argparse
import logging
import signal
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from live.trading_engine import LiveTradingEngine
from data.repository import TradingDataRepository
from data_providers.cached_data_provider import CachedDataProvider
from data_providers.binance_data_provider import BinanceDataProvider
from database.manager import DatabaseManager
from strategies import get_strategy_class
from config.config_manager import ConfigManager
from config.providers.env_provider import EnvVarProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run live trading')
    parser.add_argument('strategy', help='Strategy name to run')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to trade (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe (default: 1d)')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance for paper trading (default: 10000)')
    parser.add_argument('--paper', action='store_true', help='Paper trading mode (default: live)')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--max-position-size', type=float, default=0.1, help='Max position size as fraction of balance (default: 0.1)')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss percentage (default: 0.02)')
    parser.add_argument('--take-profit', type=float, default=0.04, help='Take profit percentage (default: 0.04)')
    
    return parser.parse_args()


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(level)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def create_data_repository(config: ConfigManager) -> TradingDataRepository:
    """Create data repository with cached data provider"""
    # Use cached provider for efficiency
    base_provider = BinanceDataProvider()
    cached_provider = CachedDataProvider(base_provider)
    
    # Create database manager
    db_manager = DatabaseManager(config.get('DATABASE_URL'))
    
    return TradingDataRepository(db_manager, cached_provider)


class TradingApp:
    """Main application class to manage the trading engine"""
    
    def __init__(self, engine: LiveTradingEngine, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.engine = engine
        self.shutdown_event = asyncio.Event()
        self.loop = loop or asyncio.get_event_loop()
        
    async def run(self):
        """Start the trading engine with proper signal handling"""
        # Setup signal handlers
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self.shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Setup event callbacks
        self.engine.set_on_trade_opened(self._on_trade_opened)
        self.engine.set_on_trade_closed(self._on_trade_closed)
        self.engine.set_on_signal_generated(self._on_signal_generated)
        self.engine.set_on_error(self._on_error)
        
        try:
            # Start the engine
            engine_task = asyncio.create_task(self.engine.start())
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
            # Stop the engine
            await self.engine.stop()
            
            # Cancel engine task if still running
            if not engine_task.done():
                engine_task.cancel()
                try:
                    await engine_task
                except asyncio.CancelledError:
                    pass
                    
        except Exception as e:
            logger.error(f"Error in trading engine controller: {str(e)}")
            raise
    
    async def _on_trade_opened(self, result):
        """Handle trade opened event"""
        logger.info(f"🔵 TRADE OPENED: {result.position_id} at ${result.executed_price:.2f}")
        
    async def _on_trade_closed(self, result):
        """Handle trade closed event"""
        pnl_emoji = "🟢" if result.pnl > 0 else "🔴"
        logger.info(f"{pnl_emoji} TRADE CLOSED: P&L ${result.pnl:.2f}")
        
    async def _on_signal_generated(self, signal):
        """Handle signal generated event"""
        if signal.action != "hold":
            logger.info(f"📊 SIGNAL: {signal.action.upper()} - {signal.side.value if signal.side else 'N/A'} "
                       f"(Confidence: {signal.confidence:.2f})")
        
    async def _on_error(self, error):
        """Handle error event"""
        logger.error(f"❌ TRADING ERROR: {str(error)}")

    async def shutdown(self):
        """Shutdown the trading engine"""
        await self.engine.stop()


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load configuration
        config = ConfigManager(providers=[EnvVarProvider()])
        
        # Get API credentials
        api_key = config.get('binance.api_key')
        api_secret = config.get('binance.api_secret')
        
        if not api_key or not api_secret:
            if not args.paper:
                logger.error("Binance API credentials not found. Use --paper for paper trading.")
                return 1
            else:
                logger.info("Running in paper trading mode (no real API credentials needed)")
                api_key = "dummy_key"
                api_secret = "dummy_secret"
        
        # Create data repository
        data_repository = create_data_repository(config)
        
        # Get strategy class
        strategy_class = get_strategy_class(args.strategy)
        if not strategy_class:
            logger.error(f"Strategy '{args.strategy}' not found")
            return 1
        
        # Create live trading engine
        engine = LiveTradingEngine(
            strategy=strategy_class(),
            data_provider=data_repository.data_provider,
            api_key=api_key,
            api_secret=api_secret,
            symbol=args.symbol,
            timeframe=args.timeframe,
            paper_trading=args.paper
        )
        
        # Create and run the application
        app = TradingApp(engine)
        
        # Print startup info
        mode = "PAPER" if args.paper else "LIVE"
        logger.info(f"🚀 Starting {mode} trading engine")
        logger.info(f"Strategy: {args.strategy}")
        logger.info(f"Symbol: {args.symbol}")
        logger.info(f"Timeframe: {args.timeframe}")
        logger.info(f"Initial Balance: ${args.initial_balance:,.2f}")
        logger.info(f"Max Position Size: {args.max_position_size*100:.1f}%")
        logger.info(f"Stop Loss: {args.stop_loss*100:.1f}%")
        logger.info(f"Take Profit: {args.take_profit*100:.1f}%")
        
        if not args.paper:
            logger.warning("⚠️  LIVE TRADING MODE - REAL MONEY AT RISK!")
            logger.warning("⚠️  Press Ctrl+C to stop trading safely")
        
        # Start the application
        asyncio.run(app.run())
        
        logger.info("Trading engine stopped successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Application interrupted")
        return 1
    except Exception as e:
        logger.error(f"Error running live trading: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 