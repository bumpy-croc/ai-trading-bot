#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta
import logging
import importlib
import sys
from pathlib import Path

# Add project root and its 'src' directory to Python path so that imports like `config.*` work when running this script directly
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_path = project_root / 'src'
if src_path.exists():
    sys.path.append(str(src_path))


from config.config_manager import get_config
from config.constants import DEFAULT_INITIAL_BALANCE

from data_providers import BinanceDataProvider
from data_providers.cached_data_provider import CachedDataProvider
from data_providers.cryptocompare_sentiment import CryptoCompareSentimentProvider
from risk import RiskParameters
from backtesting import Backtester

from strategies import MlBasic  # Direct import after removing deprecated strategies
from src.utils.symbol_factory import SymbolFactory
import pandas as pd
from pathlib import Path

from utils.logging_config import configure_logging
configure_logging()
logger = logging.getLogger('backtest')

def load_strategy(strategy_name: str):
    """Load a strategy by name"""
    try:
        # Import strategies
        if strategy_name == 'ml_basic':
            from strategies.ml_basic import MlBasic
            strategy = MlBasic()
        else:
            print(f"Unknown strategy: {strategy_name}")
            available_strategies = ['ml_basic']
            print(f"Available strategies: {', '.join(available_strategies)}")
            sys.exit(1)
        
        # Create and return strategy instance
        return strategy
        
    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('strategy', help='Strategy name (e.g., ml_basic)')
    parser.add_argument('--symbol', default='BTCUSDT', help='Trading pair symbol')
    parser.add_argument('--timeframe', default='1h', help='Candle timeframe')
    parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
    parser.add_argument('--start', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', help='End date (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=DEFAULT_INITIAL_BALANCE, help='Initial balance')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (1% = 0.01)')
    parser.add_argument('--max-risk-per-trade', type=float, default=0.02, help='Maximum risk per trade')
    parser.add_argument('--use-sentiment', action='store_true', help='Use sentiment analysis in backtest')
    parser.add_argument('--no-cache', action='store_true', help='Disable data caching')
    parser.add_argument('--cache-ttl', type=int, default=24, help='Cache TTL in hours (default: 24)')
    parser.add_argument('--no-db', action='store_true', help='Disable database logging for this backtest')
    parser.add_argument('--provider', choices=['coinbase', 'binance'], default='binance', help='Exchange provider to use (default: binance)')
    return parser.parse_args()

def get_date_range(args):
    if args.start and args.end:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.strptime(args.end, '%Y-%m-%d')
    elif args.start:
        start_date = datetime.strptime(args.start, '%Y-%m-%d')
        end_date = datetime.now()
    elif args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # Default to 30 days
    return start_date, end_date

def main():
    args = parse_args()
    
    try:
        # Calculate date range
        start_date, end_date = get_date_range(args)
        
        # Load the strategy
        strategy = load_strategy(args.strategy)
        logger.info(f"Loaded strategy: {strategy.name}")
        
        # Initialize data provider with caching
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
            data_provider = CachedDataProvider(provider, cache_ttl_hours=args.cache_ttl)
            logger.info(f"Using cached data provider (TTL: {args.cache_ttl} hours)")
            
            # Show cache info
            cache_info = data_provider.get_cache_info()
            logger.info(f"Cache info: {cache_info['total_files']} files, {cache_info['total_size_mb']} MB")
        
        # Initialize sentiment provider if requested
        sentiment_provider = None
        if args.use_sentiment:
            sentiment_provider = CryptoCompareSentimentProvider()
            logger.info("Using sentiment analysis in backtest")
        
        # Set up risk parameters
        risk_params = RiskParameters(
            base_risk_per_trade=args.risk_per_trade,
            max_risk_per_trade=args.max_risk_per_trade
        )
        
        # Create and run backtester
        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            sentiment_provider=sentiment_provider,
            risk_parameters=risk_params,
            initial_balance=args.initial_balance,
            log_to_database=not args.no_db  # Disable DB logging if --no-db is passed
        )
        
        # Use strategy-specific trading pair if not overridden by command line
        # If user provided a symbol explicitly, use it; otherwise use strategy's default
        if args.symbol != 'BTC-USD':  # User provided a specific symbol
            trading_symbol = SymbolFactory.to_exchange_symbol(args.symbol, args.provider)
        else:  # Use strategy's default trading pair
            trading_symbol = strategy.get_trading_pair()
        
        # Run backtest; if provider returns empty, fallback to local CSV for BTCUSDT 1d
        results = {}
        used_csv_fallback = False
        effective_timeframe = args.timeframe
        try:
            results = backtester.run(
                symbol=trading_symbol,
                timeframe=args.timeframe,
                start=start_date,
                end=end_date
            )
        except Exception as e:
            logger.error(f"Primary backtest attempt failed: {e}")
            results = {}
 
        # Only attempt CSV fallback when the primary attempt produced no results at all
        if not results:
            # Attempt CSV fallback only for BTC daily
            csv_path = Path(__file__).parent.parent / 'src' / 'data' / 'BTCUSDT_1d.csv'
            if csv_path.exists() and args.symbol.upper() in ['BTCUSDT', 'BTC-USD'] and args.timeframe == '1d':
                logger.info(f"Falling back to local CSV: {csv_path}")
                df = pd.read_csv(csv_path)
                # Parse timestamp
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.set_index('timestamp')
                # Clip date range
                mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
                df = df.loc[mask].copy()
                # If filtered CSV is empty, skip fallback to avoid errors
                if df.empty:
                    logger.warning("CSV fallback data is empty after filtering by date range. Skipping fallback.")
                else:
                    from data_providers.data_provider import DataProvider
                    class _CsvProvider(DataProvider):
                        def __init__(self, data: pd.DataFrame):
                            super().__init__()
                            self._data = data
                        def get_historical_data(self, symbol, timeframe, start, end=None):
                            return self._data.copy()
                        def get_live_data(self, symbol, timeframe, limit=100):
                            return self._data.tail(limit).copy()
                        def update_live_data(self, symbol, timeframe):
                            return self._data.copy()
                        def get_current_price(self, symbol: str) -> float:
                            return float(self._data.iloc[-1]['close']) if not self._data.empty else 0.0
                    csv_provider = _CsvProvider(df)
                    backtester_csv = Backtester(
                        strategy=strategy,
                        data_provider=csv_provider,
                        sentiment_provider=None,
                        risk_parameters=risk_params,
                        initial_balance=args.initial_balance,
                        log_to_database=False
                    )
                    results = backtester_csv.run(
                        symbol=trading_symbol,
                        timeframe='1d',
                        start=start_date,
                        end=end_date
                    )
                    used_csv_fallback = True
                    effective_timeframe = '1d'

        # Print results
        print("\nBacktest Results:")
        print("=" * 50)
        print(f"Strategy: {strategy.name}")
        print(f"Symbol: {trading_symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {effective_timeframe}")
        print(f"Using Sentiment: {args.use_sentiment}")
        print(f"Using Cache: {not args.no_cache}")
        print(f"Database Logging: {not args.no_db}")
        print("-" * 50)
        if results:
            print(f"Total Trades: {results.get('total_trades', 0)}")
            print(f"Win Rate: {results.get('win_rate', 0.0):.2f}%")
            print(f"Total Return: {results.get('total_return', 0.0):.2f}%")
            if 'annualized_return' in results:
                print(f"Annualized Return: {results['annualized_return']:.2f}%")
            print(f"Max Drawdown: {results.get('max_drawdown', 0.0):.2f}%")
            print(f"Sharpe Ratio: {results.get('sharpe_ratio', 0.0):.2f}")
            print(f"Final Balance: ${results.get('final_balance', 0.0):.2f}")
        else:
            print("No results produced.")
        print("=" * 50)

        # Print session ID if database logging was enabled
        if not args.no_db and 'session_id' in results and results['session_id']:
            print(f"Database Session ID: {results['session_id']}")
            print("=" * 50)

        # Print early stop information if backtest was stopped early
        if results.get('early_stop_reason'):
            print("⚠️  BACKTEST STOPPED EARLY ⚠️")
            print(f"Reason: {results['early_stop_reason']}")
            print(f"Date: {results['early_stop_date']}")
            print(f"Candle: {results['early_stop_candle_index']} of {len(df) if 'df' in locals() else 'unknown'}")
            print("=" * 50)
        
        # Print yearly returns if available
        if 'yearly_returns' in results and results['yearly_returns']:
            print("Yearly Returns:")
            print(f"{'Year':<8} {'Return (%)':>12}")
            for year in sorted(results['yearly_returns'].keys()):
                print(f"{year:<8} {results['yearly_returns'][year]:>12.2f}")
            print("=" * 50)
        
        # Show final cache info if using cache
        if not args.no_cache:
            final_cache_info = data_provider.get_cache_info()
            logger.info(f"Final cache info: {final_cache_info['total_files']} files, {final_cache_info['total_size_mb']} MB")

        # --------------------------------------
        # NEW: Persist backtest run to log file
        # --------------------------------------
        try:
            import json
            import subprocess
            # Calculate duration in years (float with 2 decimals)
            duration_years = round((end_date - start_date).days / 365.25, 2)

            # Resolve git information (commit & branch) – fallback to 'unknown'
            try:
                commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                commit_hash = 'unknown'
            try:
                branch_name = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
            except Exception:
                branch_name = 'unknown'

            # Prepare payload
            log_payload = {
                'timestamp': datetime.now().isoformat(timespec='seconds'),
                'strategy': strategy.name,
                'symbol': trading_symbol,
                'timeframe': effective_timeframe,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'duration_years': duration_years,
                'initial_balance': args.initial_balance,
                'use_sentiment': args.use_sentiment,
                'use_cache': not args.no_cache,
                'database_logging': not args.no_db,
                'risk_per_trade': args.risk_per_trade,
                'max_risk_per_trade': args.max_risk_per_trade,
                'cache_ttl_hours': None if args.no_cache else args.cache_ttl,
                'git_commit': commit_hash,
                'git_branch': branch_name,
                'results': results,
                'strategy_config': getattr(strategy, 'config', {})
            }

            # Build filename and ensure directory exists
            import re
            timestamp_for_file = datetime.now().strftime('%Y%m%d_%H%M%S')
            sanitized_strategy_name = re.sub(r'[^a-zA-Z0-9_-]', '_', strategy.name)
            filename = f"{timestamp_for_file}_{sanitized_strategy_name}_{duration_years}yrs.json"
            logs_dir = Path(project_root) / 'logs' / 'backtest'
            logs_dir.mkdir(parents=True, exist_ok=True)
            filepath = logs_dir / filename

            with open(filepath, 'w') as f:
                json.dump(log_payload, f, indent=2)
            logger.info(f"Backtest log saved to {filepath.relative_to(project_root)}")
        except Exception as log_err:
            logger.warning(f"Failed to write backtest log: {log_err}")

        # --- Save aligned sentiment data to file if sentiment is used ---
        if sentiment_provider is not None:
            # Fetch price data
            df = data_provider.get_historical_data(args.symbol, args.timeframe, start_date, end_date)
            sentiment_df = sentiment_provider.get_historical_sentiment(args.symbol, start_date, end_date)
            if not sentiment_df.empty:
                sentiment_df = sentiment_provider.aggregate_sentiment(sentiment_df, window=args.timeframe)
                aligned_df = df.join(sentiment_df, how='left')
                print(f"Shape of aligned DataFrame: {aligned_df.shape}")
                if aligned_df.empty:
                    print("Warning: aligned DataFrame is empty. No file will be written.")
                output_path = '../data/sentiment_aligned_output.csv'
                try:
                    aligned_df.to_csv(output_path)
                    print(f'Aligned sentiment and price data saved to {output_path}')
                except Exception as file_err:
                    print(f'Error writing {output_path}: {file_err}')
                    logger.error(f'Error writing {output_path}: {file_err}')
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 