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

from strategies import AdaptiveStrategy, EnhancedStrategy, HighRiskHighRewardStrategy, MlBasic  # Direct imports
from strategies.ml_adaptive import MlAdaptive

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')

def load_strategy(strategy_name: str):
    """Load a strategy by name"""
    try:
        # Import strategies
        if strategy_name == 'adaptive':
            from strategies.adaptive import AdaptiveStrategy
            strategy = AdaptiveStrategy()

        elif strategy_name == 'enhanced':
            from strategies.enhanced import EnhancedStrategy
            strategy = EnhancedStrategy()
        elif strategy_name == 'high_risk_high_reward':
            from strategies.high_risk_high_reward import HighRiskHighRewardStrategy
            strategy = HighRiskHighRewardStrategy()
        elif strategy_name == 'ml_basic':
            from strategies.ml_basic import MlBasic
            strategy = MlBasic()
        elif strategy_name == 'ml_with_sentiment':
            from strategies.ml_with_sentiment import MlWithSentiment
            strategy = MlWithSentiment(use_sentiment=True)
        elif strategy_name == 'ml_adaptive':
            from strategies.ml_adaptive import MlAdaptive
            strategy = MlAdaptive()
        else:
            print(f"Unknown strategy: {strategy_name}")
            available_strategies = ['adaptive', 'enhanced', 'high_risk_high_reward', 'ml_basic', 'ml_with_sentiment', 'ml_adaptive']
            print(f"Available strategies: {', '.join(available_strategies)}")
            sys.exit(1)
        
        # Create and return strategy instance
        return strategy
        
    except Exception as e:
        logger.error(f"Error loading strategy: {e}")
        raise

def parse_args():
    parser = argparse.ArgumentParser(description='Run strategy backtest')
    parser.add_argument('strategy', help='Strategy name (e.g., adaptive, enhanced)')
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
        binance_provider = BinanceDataProvider()
        if args.no_cache:
            data_provider = binance_provider
            logger.info("Data caching disabled")
        else:
            data_provider = CachedDataProvider(
                binance_provider, 
                cache_ttl_hours=args.cache_ttl
            )
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
        if args.symbol != 'BTCUSDT':  # User provided a specific symbol
            trading_symbol = args.symbol
        else:  # Use strategy's default trading pair
            trading_symbol = strategy.get_trading_pair()
        
        # Run backtest
        results = backtester.run(
            symbol=trading_symbol,
            timeframe=args.timeframe,
            start=start_date,
            end=end_date
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