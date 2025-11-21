#!/usr/bin/env python3
"""
Confidence Threshold Optimization Script

This script runs a grid search to find the optimal confidence threshold
for the ml_basic strategy across multiple symbols and timeframes.

Usage:
    python scripts/optimize_confidence_threshold.py --symbol BTCUSDT --timeframe 1h --days 365
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtester import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.strategies.ml_basic import create_ml_basic_strategy


def run_threshold_optimization(
    symbol: str,
    timeframe: str,
    days: int,
    thresholds: list[float],
    output_file: str = None,
) -> pd.DataFrame:
    """
    Run backtest grid search across confidence thresholds

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        days: Number of days to backtest
        thresholds: List of confidence thresholds to test
        output_file: Optional path to save results CSV

    Returns:
        DataFrame with results for each threshold
    """
    print(f"\n{'='*80}")
    print(f"Confidence Threshold Optimization")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}, Days: {days}")
    print(f"Testing {len(thresholds)} thresholds: {thresholds}")
    print(f"{'='*80}\n")

    results = []

    for i, threshold in enumerate(thresholds, 1):
        print(f"\n[{i}/{len(thresholds)}] Testing threshold: {threshold}")
        print("-" * 60)

        try:
            # Create strategy with this confidence threshold
            # Note: This requires modifying ml_basic.py to accept confidence_threshold parameter
            # For now, this is a template - implementation needed
            strategy = create_ml_basic_strategy(
                name=f"ml_basic_conf_{threshold}",
                # confidence_threshold=threshold,  # TODO: Add this parameter
            )

            # Set up data provider
            binance_provider = BinanceProvider()
            data_provider = CachedDataProvider(binance_provider)

            # Create backtester
            backtester = Backtester(
                strategy=strategy,
                data_provider=data_provider,
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                initial_balance=10000.0,
            )

            # Run backtest
            metrics = backtester.run()

            # Extract key metrics
            result = {
                "threshold": threshold,
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "total_return": metrics.get("total_return", 0),
                "total_return_pct": metrics.get("total_return", 0) * 100,
                "max_drawdown": metrics.get("max_drawdown", 0),
                "max_drawdown_pct": metrics.get("max_drawdown", 0) * 100,
                "win_rate": metrics.get("win_rate", 0),
                "win_rate_pct": metrics.get("win_rate", 0) * 100,
                "total_trades": metrics.get("total_trades", 0),
                "avg_trade_return": metrics.get("avg_trade_return", 0),
                "profit_factor": metrics.get("profit_factor", 0),
                "expectancy": metrics.get("expectancy", 0),
            }

            results.append(result)

            # Print results
            print(f"  Sharpe Ratio:     {result['sharpe_ratio']:.3f}")
            print(f"  Total Return:     {result['total_return_pct']:.2f}%")
            print(f"  Max Drawdown:     {result['max_drawdown_pct']:.2f}%")
            print(f"  Win Rate:         {result['win_rate_pct']:.1f}%")
            print(f"  Total Trades:     {result['total_trades']}")
            print(f"  Profit Factor:    {result['profit_factor']:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "threshold": threshold,
                    "error": str(e),
                    "sharpe_ratio": None,
                    "total_return": None,
                }
            )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Sort by Sharpe ratio (descending)
    results_df = results_df.sort_values("sharpe_ratio", ascending=False, na_position="last")

    # Save to file if specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    return results_df


def print_summary(results_df: pd.DataFrame):
    """Print optimization summary"""
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}\n")

    # Best by Sharpe ratio
    best_sharpe = results_df.iloc[0]
    print("Best Configuration by Sharpe Ratio:")
    print(f"  Threshold:        {best_sharpe['threshold']}")
    print(f"  Sharpe Ratio:     {best_sharpe['sharpe_ratio']:.3f}")
    print(f"  Total Return:     {best_sharpe['total_return_pct']:.2f}%")
    print(f"  Max Drawdown:     {best_sharpe['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:         {best_sharpe['win_rate_pct']:.1f}%")
    print(f"  Total Trades:     {best_sharpe['total_trades']}")

    # Best by total return (within max DD constraint)
    valid_results = results_df[results_df["max_drawdown"] <= 0.25]
    if not valid_results.empty:
        best_return = valid_results.sort_values("total_return", ascending=False).iloc[0]
        print(f"\nBest Configuration by Return (Max DD ≤ 25%):")
        print(f"  Threshold:        {best_return['threshold']}")
        print(f"  Total Return:     {best_return['total_return_pct']:.2f}%")
        print(f"  Sharpe Ratio:     {best_return['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:     {best_return['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:         {best_return['win_rate_pct']:.1f}%")

    # Print full results table
    print(f"\n{'='*80}")
    print("Full Results (sorted by Sharpe Ratio):")
    print(f"{'='*80}\n")
    print(
        results_df[
            [
                "threshold",
                "sharpe_ratio",
                "total_return_pct",
                "max_drawdown_pct",
                "win_rate_pct",
                "total_trades",
            ]
        ].to_string(index=False)
    )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Optimize confidence threshold for ml_basic strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=365, help="Number of days to backtest")
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.50,0.52,0.54,0.56,0.58,0.60,0.65,0.70",
        help="Comma-separated list of thresholds to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: results/threshold_optimization_{timestamp}.csv)",
    )

    args = parser.parse_args()

    # Parse thresholds
    thresholds = [float(x.strip()) for x in args.thresholds.split(",")]

    # Set default output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir / f"threshold_optimization_{args.symbol}_{args.timeframe}_{timestamp}.csv"
        )

    # Run optimization
    results_df = run_threshold_optimization(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        thresholds=thresholds,
        output_file=args.output,
    )

    # Print summary
    print_summary(results_df)

    # Save summary report
    summary_file = args.output.replace(".csv", "_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Confidence Threshold Optimization Report\n")
        f.write(f"{'='*80}\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Timeframe: {args.timeframe}\n")
        f.write(f"Days: {args.days}\n")
        f.write(f"Thresholds Tested: {thresholds}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{results_df.to_string()}\n")

    print(f"\nSummary report saved to: {summary_file}")


if __name__ == "__main__":
    main()
