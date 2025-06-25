#!/usr/bin/env python3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import argparse
from datetime import datetime, timedelta
import logging
import importlib
import sys
from pathlib import Path

from core.data import BinanceDataProvider
from core.data.cryptocompare_sentiment import CryptoCompareSentimentProvider
from core.risk import RiskParameters
from backtesting import Backtester
from strategies import AdaptiveStrategy, EnhancedStrategy, AdaptiveStrategy2, HighRiskHighRewardStrategy, MlModelStrategy  # Direct imports

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')

def load_strategy(strategy_name: str):
    """Load a strategy by name"""
    try:
        # Map strategy names to classes
        strategy_map = {
            'enhanced': EnhancedStrategy,
            'adaptive': AdaptiveStrategy,
            'adaptive2': AdaptiveStrategy2,
            'high_risk_high_reward': HighRiskHighRewardStrategy,
            'ml_model_strategy': MlModelStrategy
        }
        
        # Get the strategy class
        strategy_class = strategy_map.get(strategy_name.lower())
        if strategy_class is None:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available strategies: {list(strategy_map.keys())}")
        
        # Create and return strategy instance
        return strategy_class()
        
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
    parser.add_argument('--initial-balance', type=float, default=10000, help='Initial balance')
    parser.add_argument('--risk-per-trade', type=float, default=0.01, help='Risk per trade (1% = 0.01)')
    parser.add_argument('--max-risk-per-trade', type=float, default=0.02, help='Maximum risk per trade')
    parser.add_argument('--use-sentiment', action='store_true', help='Use sentiment analysis in backtest')
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
        
        # Initialize data provider
        data_provider = BinanceDataProvider()
        
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
            initial_balance=args.initial_balance
        )
        
        # Run backtest
        results = backtester.run(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start_date,
            end=end_date
        )
        
        # Print results
        print("\nBacktest Results:")
        print("=" * 50)
        print(f"Strategy: {strategy.name}")
        print(f"Symbol: {args.symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {args.timeframe}")
        print(f"Using Sentiment: {args.use_sentiment}")
        print("-" * 50)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print("=" * 50)
        
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
                output_path = 'data/sentiment_aligned_output.csv'
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