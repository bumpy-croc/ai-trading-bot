#!/usr/bin/env python3
"""
Backtest Runner

This script runs backtests using the new architecture with:
- TradeExecutor for consistent trade management
- SignalGenerator for standardized signal generation
- TradingDataRepository for data access
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from backtesting.engine import Backtester
from data.repository import TradingDataRepository
from data_providers.cached_data_provider import CachedDataProvider
from data_providers.binance_data_provider import BinanceDataProvider
from database.manager import DatabaseManager
from strategies import get_strategy_class
from config.config_manager import ConfigManager
from config.providers.env_provider import EnvVarProvider
from performance.monitor import get_monitor, set_monitoring_enabled

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run backtest')
    parser.add_argument('strategy', help='Strategy name to backtest')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest (default: 30)')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to trade (default: BTCUSDT)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe (default: 1d)')
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance (default: 10000)')
    parser.add_argument('--no-db', action='store_true', help='Skip database logging')
    parser.add_argument('--fast', action='store_true', help='Fast mode: minimal logging, no database, optimized for speed')
    parser.add_argument('--config', help='Path to custom config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    return parser.parse_args()


def setup_logging(verbose: bool = False, fast_mode: bool = False):
    """Setup logging configuration"""
    if fast_mode:
        level = logging.WARNING  # Minimal logging for speed
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
        
    logging.getLogger().setLevel(level)
    
    # Reduce noise from external libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)


def create_data_repository(config: ConfigManager) -> TradingDataRepository:
    """Create data repository with cached data provider"""
    # Use cached provider to avoid API rate limits during backtesting
    base_provider = BinanceDataProvider()
    cached_provider = CachedDataProvider(base_provider)
    
    # Create database manager (optional for backtesting)
    db_manager = None
    try:
        db_manager = DatabaseManager(config.get('DATABASE_URL'))
    except Exception as e:
        logger.warning(f"Could not connect to database: {e}")
    
    return TradingDataRepository(db_manager, cached_provider)


def print_results(results: Dict[str, Any]):
    """Print backtest results in a formatted way"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    # Basic info
    print(f"Strategy: {results.get('strategy_name', 'Unknown')}")
    print(f"Symbol: {results.get('symbol', 'Unknown')}")
    print(f"Timeframe: {results.get('timeframe', 'Unknown')}")
    print(f"Period: {results.get('start_date', 'Unknown')} to {results.get('end_date', 'Unknown')}")
    print(f"Session ID: {results.get('session_id', 'N/A')}")
    
    print("\n" + "-"*40)
    print("PERFORMANCE METRICS")
    print("-"*40)
    
    # Performance metrics
    print(f"Total Trades: {results.get('total_trades', 0)}")
    print(f"Win Rate: {results.get('win_rate', 0):.2f}%")
    print(f"Total Return: {results.get('total_return', 0):.2f}%")
    print(f"Annualized Return: {results.get('annualized_return', 0):.2f}%")
    print(f"Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
    print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
    
    print(f"\nInitial Balance: ${results.get('initial_balance', 0):,.2f}")
    print(f"Final Balance: ${results.get('final_balance', 0):,.2f}")
    print(f"Total P&L: ${results.get('total_pnl', 0):,.2f}")
    
    # Trade statistics
    print(f"\nAverage Trade P&L: ${results.get('avg_trade_pnl', 0):,.2f}")
    print(f"Max Win: ${results.get('max_win', 0):,.2f}")
    print(f"Max Loss: ${results.get('max_loss', 0):,.2f}")
    print(f"Profit Factor: {results.get('profit_factor', 0):.2f}")
    
    # Yearly returns
    yearly_returns = results.get('yearly_returns', {})
    if yearly_returns:
        print("\n" + "-"*30)
        print("YEARLY RETURNS")
        print("-"*30)
        for year, return_pct in yearly_returns.items():
            print(f"{year}: {return_pct:.2f}%")
    
    print("\n" + "="*60)


def main():
    """Main function"""
    args = parse_arguments()
    
    # Setup logging (disable most logging in fast mode)
    setup_logging(args.verbose, args.fast)
    
    # Disable performance monitoring in fast mode for maximum speed
    if args.fast:
        set_monitoring_enabled(False)
    
    try:
        # Load configuration
        config = ConfigManager(providers=[EnvVarProvider()])
        
        # Create data repository
        data_repository = create_data_repository(config)
        
        # Get strategy class
        strategy_class = get_strategy_class(args.strategy)
        if not strategy_class:
            logger.error(f"Strategy '{args.strategy}' not found")
            return 1
        
        # Create strategy instance
        strategy = strategy_class()
        
        # Create backtester (fast mode disables database logging)
        use_database = not args.no_db and not args.fast
        backtester = Backtester(
            strategy=strategy,
            data_provider=data_repository.data_provider,
            initial_balance=args.initial_balance,
            database_url=config.get('DATABASE_URL') if use_database else None,
            log_to_database=use_database
        )
        
        # Calculate date range - exclude incomplete current day
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=args.days)
        
        logger.info(f"Starting backtest: {args.strategy} on {args.symbol}")
        logger.info(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        logger.info(f"Initial balance: ${args.initial_balance:,.2f}")
        
        # Run backtest
        results = backtester.run(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start_date,
            end=end_date
        )
        
        # Print results
        print_results(results)
        
        # Additional information
        if use_database:
            print(f"\nTrade history and detailed analytics available in database.")
            print(f"Session ID: {results.get('session_id')}")
        
        # Show performance metrics (skip in fast mode)
        if not args.fast:
            monitor = get_monitor()
            print(f"\nCurrent system stats: {monitor.get_current_system_stats()}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 