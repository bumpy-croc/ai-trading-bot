#!/usr/bin/env python3
"""
Position Size Optimization Script

This script runs a grid search to find the optimal base position fraction
for the ml_basic strategy, balancing returns against risk (max drawdown).

Usage:
    python scripts/optimize_position_size.py --symbol BTCUSDT --timeframe 1h --days 365
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtester import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.strategies.ml_basic import create_ml_basic_strategy


def run_position_size_optimization(
    symbol: str,
    timeframe: str,
    days: int,
    base_fractions: list[float],
    output_file: str = None,
) -> pd.DataFrame:
    """
    Run backtest grid search across position size fractions

    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        timeframe: Candle timeframe (e.g., '1h', '4h', '1d')
        days: Number of days to backtest
        base_fractions: List of position size fractions to test (e.g., [0.05, 0.10, 0.15])
        output_file: Optional path to save results CSV

    Returns:
        DataFrame with results for each fraction
    """
    print(f"\n{'='*80}")
    print(f"Position Size Optimization")
    print(f"Symbol: {symbol}, Timeframe: {timeframe}, Days: {days}")
    print(f"Testing {len(base_fractions)} position sizes: {base_fractions}")
    print(f"{'='*80}\n")

    results = []

    for i, fraction in enumerate(base_fractions, 1):
        print(f"\n[{i}/{len(base_fractions)}] Testing base fraction: {fraction} ({fraction*100:.1f}%)")
        print("-" * 60)

        try:
            # Create strategy with this position size
            # Note: This requires modifying ml_basic.py to accept base_fraction parameter
            strategy = create_ml_basic_strategy(
                name=f"ml_basic_size_{fraction}",
                # base_fraction=fraction,  # TODO: Add this parameter
            )

            # Set up data provider
            binance_provider = BinanceProvider()
            data_provider = CachedDataProvider(binance_provider)

            # Create backtester
            initial_balance = 10000.0
            backtester = Backtester(
                strategy=strategy,
                data_provider=data_provider,
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                initial_balance=initial_balance,
            )

            # Run backtest
            metrics = backtester.run()

            # Extract key metrics
            result = {
                "base_fraction": fraction,
                "base_fraction_pct": fraction * 100,
                "sharpe_ratio": metrics.get("sharpe_ratio", 0),
                "sortino_ratio": metrics.get("sortino_ratio", 0),
                "calmar_ratio": metrics.get("calmar_ratio", 0),
                "total_return": metrics.get("total_return", 0),
                "total_return_pct": metrics.get("total_return", 0) * 100,
                "annualized_return": metrics.get("annualized_return", 0),
                "annualized_return_pct": metrics.get("annualized_return", 0) * 100,
                "max_drawdown": metrics.get("max_drawdown", 0),
                "max_drawdown_pct": metrics.get("max_drawdown", 0) * 100,
                "win_rate": metrics.get("win_rate", 0),
                "win_rate_pct": metrics.get("win_rate", 0) * 100,
                "total_trades": metrics.get("total_trades", 0),
                "avg_trade_return": metrics.get("avg_trade_return", 0),
                "avg_trade_return_pct": metrics.get("avg_trade_return", 0) * 100,
                "profit_factor": metrics.get("profit_factor", 0),
                "expectancy": metrics.get("expectancy", 0),
                "max_consecutive_losses": metrics.get("max_consecutive_losses", 0),
                "avg_position_size": metrics.get("avg_position_size", 0),
                "avg_position_pct": (metrics.get("avg_position_size", 0) / initial_balance) * 100,
                "volatility": metrics.get("volatility", 0),
                "volatility_pct": metrics.get("volatility", 0) * 100,
            }

            results.append(result)

            # Print results
            print(f"  Sharpe Ratio:       {result['sharpe_ratio']:.3f}")
            print(f"  Sortino Ratio:      {result['sortino_ratio']:.3f}")
            print(f"  Total Return:       {result['total_return_pct']:.2f}%")
            print(f"  Annualized Return:  {result['annualized_return_pct']:.2f}%")
            print(f"  Max Drawdown:       {result['max_drawdown_pct']:.2f}%")
            print(f"  Win Rate:           {result['win_rate_pct']:.1f}%")
            print(f"  Total Trades:       {result['total_trades']}")
            print(f"  Avg Position Size:  {result['avg_position_pct']:.2f}%")
            print(f"  Volatility:         {result['volatility_pct']:.2f}%")

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append(
                {
                    "base_fraction": fraction,
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


def create_visualization(results_df: pd.DataFrame, output_dir: Path):
    """Create visualization charts for position size optimization"""
    print("\nGenerating visualizations...")

    # Set style
    sns.set_style("darkgrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Sort by fraction for plotting
    plot_df = results_df.sort_values("base_fraction")

    # 1. Sharpe Ratio vs Position Size
    ax1 = axes[0, 0]
    ax1.plot(
        plot_df["base_fraction_pct"],
        plot_df["sharpe_ratio"],
        marker="o",
        linewidth=2,
        markersize=8,
        color="blue",
    )
    ax1.set_xlabel("Base Position Size (%)", fontsize=12)
    ax1.set_ylabel("Sharpe Ratio", fontsize=12)
    ax1.set_title("Sharpe Ratio vs Position Size", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    # 2. Return vs Drawdown (Efficient Frontier)
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        plot_df["max_drawdown_pct"],
        plot_df["total_return_pct"],
        c=plot_df["sharpe_ratio"],
        s=200,
        cmap="RdYlGn",
        edgecolors="black",
        linewidth=1.5,
        alpha=0.8,
    )
    # Add labels for each point
    for _, row in plot_df.iterrows():
        ax2.annotate(
            f"{row['base_fraction_pct']:.0f}%",
            (row["max_drawdown_pct"], row["total_return_pct"]),
            fontsize=8,
            ha="center",
            va="bottom",
        )
    ax2.set_xlabel("Max Drawdown (%)", fontsize=12)
    ax2.set_ylabel("Total Return (%)", fontsize=12)
    ax2.set_title("Efficient Frontier: Return vs Risk", fontsize=14, fontweight="bold")
    ax2.axvline(x=25, color="red", linestyle="--", alpha=0.5, label="25% DD Limit")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.colorbar(scatter, ax=ax2, label="Sharpe Ratio")

    # 3. Win Rate and Total Trades
    ax3 = axes[1, 0]
    ax3_twin = ax3.twinx()
    line1 = ax3.plot(
        plot_df["base_fraction_pct"],
        plot_df["win_rate_pct"],
        marker="s",
        linewidth=2,
        markersize=8,
        color="green",
        label="Win Rate",
    )
    line2 = ax3_twin.plot(
        plot_df["base_fraction_pct"],
        plot_df["total_trades"],
        marker="^",
        linewidth=2,
        markersize=8,
        color="orange",
        label="Total Trades",
    )
    ax3.set_xlabel("Base Position Size (%)", fontsize=12)
    ax3.set_ylabel("Win Rate (%)", fontsize=12, color="green")
    ax3_twin.set_ylabel("Total Trades", fontsize=12, color="orange")
    ax3.tick_params(axis="y", labelcolor="green")
    ax3_twin.tick_params(axis="y", labelcolor="orange")
    ax3.set_title("Win Rate and Trade Count vs Position Size", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc="upper left")

    # 4. Risk-Adjusted Returns (Multiple Metrics)
    ax4 = axes[1, 1]
    x = plot_df["base_fraction_pct"]
    ax4.plot(x, plot_df["sharpe_ratio"], marker="o", linewidth=2, label="Sharpe Ratio")
    ax4.plot(x, plot_df["sortino_ratio"], marker="s", linewidth=2, label="Sortino Ratio")
    ax4.plot(x, plot_df["calmar_ratio"], marker="^", linewidth=2, label="Calmar Ratio")
    ax4.set_xlabel("Base Position Size (%)", fontsize=12)
    ax4.set_ylabel("Risk-Adjusted Return", fontsize=12)
    ax4.set_title("Risk-Adjusted Return Metrics", fontsize=14, fontweight="bold")
    ax4.legend(loc="best")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save figure
    output_path = output_dir / "position_size_optimization_charts.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Charts saved to: {output_path}")

    plt.close()


def print_summary(results_df: pd.DataFrame):
    """Print optimization summary with recommendations"""
    print(f"\n{'='*80}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'='*80}\n")

    # Best by Sharpe ratio
    best_sharpe = results_df.sort_values("sharpe_ratio", ascending=False).iloc[0]
    print("Best Configuration by Sharpe Ratio:")
    print(f"  Base Position Size:   {best_sharpe['base_fraction_pct']:.1f}%")
    print(f"  Sharpe Ratio:         {best_sharpe['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio:        {best_sharpe['sortino_ratio']:.3f}")
    print(f"  Total Return:         {best_sharpe['total_return_pct']:.2f}%")
    print(f"  Max Drawdown:         {best_sharpe['max_drawdown_pct']:.2f}%")
    print(f"  Win Rate:             {best_sharpe['win_rate_pct']:.1f}%")
    print(f"  Total Trades:         {best_sharpe['total_trades']}")
    print(f"  Avg Position:         {best_sharpe['avg_position_pct']:.2f}%")

    # Best by return (within max DD constraint)
    valid_results = results_df[results_df["max_drawdown"] <= 0.25]
    if not valid_results.empty:
        best_return = valid_results.sort_values("total_return", ascending=False).iloc[0]
        print(f"\nBest Configuration by Return (Max DD ≤ 25%):")
        print(f"  Base Position Size:   {best_return['base_fraction_pct']:.1f}%")
        print(f"  Total Return:         {best_return['total_return_pct']:.2f}%")
        print(f"  Annualized Return:    {best_return['annualized_return_pct']:.2f}%")
        print(f"  Sharpe Ratio:         {best_return['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:         {best_return['max_drawdown_pct']:.2f}%")
        print(f"  Win Rate:             {best_return['win_rate_pct']:.1f}%")

    # Most conservative (lowest drawdown with positive Sharpe)
    positive_sharpe = results_df[results_df["sharpe_ratio"] > 0]
    if not positive_sharpe.empty:
        most_conservative = positive_sharpe.sort_values("max_drawdown").iloc[0]
        print(f"\nMost Conservative Configuration (Positive Sharpe):")
        print(f"  Base Position Size:   {most_conservative['base_fraction_pct']:.1f}%")
        print(f"  Max Drawdown:         {most_conservative['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio:         {most_conservative['sharpe_ratio']:.3f}")
        print(f"  Total Return:         {most_conservative['total_return_pct']:.2f}%")

    # Recommendation based on risk tolerance
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")

    print("Based on risk tolerance:\n")

    # Conservative: target DD < 15%
    conservative = results_df[
        (results_df["max_drawdown"] < 0.15) & (results_df["sharpe_ratio"] > 0)
    ]
    if not conservative.empty:
        best_conservative = conservative.sort_values("sharpe_ratio", ascending=False).iloc[0]
        print(f"Conservative (Max DD < 15%): {best_conservative['base_fraction_pct']:.1f}%")
        print(f"  - Sharpe: {best_conservative['sharpe_ratio']:.2f}")
        print(f"  - Return: {best_conservative['total_return_pct']:.1f}%")
        print(f"  - Max DD: {best_conservative['max_drawdown_pct']:.1f}%\n")

    # Moderate: target DD 15-25%
    moderate = results_df[
        (results_df["max_drawdown"] >= 0.15)
        & (results_df["max_drawdown"] <= 0.25)
        & (results_df["sharpe_ratio"] > 0)
    ]
    if not moderate.empty:
        best_moderate = moderate.sort_values("sharpe_ratio", ascending=False).iloc[0]
        print(f"Moderate (Max DD 15-25%): {best_moderate['base_fraction_pct']:.1f}%")
        print(f"  - Sharpe: {best_moderate['sharpe_ratio']:.2f}")
        print(f"  - Return: {best_moderate['total_return_pct']:.1f}%")
        print(f"  - Max DD: {best_moderate['max_drawdown_pct']:.1f}%\n")

    # Aggressive: highest return with DD < 30%
    aggressive = results_df[
        (results_df["max_drawdown"] <= 0.30) & (results_df["sharpe_ratio"] > 0)
    ]
    if not aggressive.empty:
        best_aggressive = aggressive.sort_values("total_return", ascending=False).iloc[0]
        print(f"Aggressive (Max DD < 30%): {best_aggressive['base_fraction_pct']:.1f}%")
        print(f"  - Return: {best_aggressive['total_return_pct']:.1f}%")
        print(f"  - Sharpe: {best_aggressive['sharpe_ratio']:.2f}")
        print(f"  - Max DD: {best_aggressive['max_drawdown_pct']:.1f}%\n")

    # Print full results table
    print(f"{'='*80}")
    print("Full Results (sorted by Sharpe Ratio):")
    print(f"{'='*80}\n")
    print(
        results_df[
            [
                "base_fraction_pct",
                "sharpe_ratio",
                "total_return_pct",
                "max_drawdown_pct",
                "win_rate_pct",
                "total_trades",
            ]
        ]
        .rename(
            columns={
                "base_fraction_pct": "Size %",
                "sharpe_ratio": "Sharpe",
                "total_return_pct": "Return %",
                "max_drawdown_pct": "Max DD %",
                "win_rate_pct": "Win Rate %",
                "total_trades": "Trades",
            }
        )
        .to_string(index=False)
    )


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Optimize position size for ml_basic strategy")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading pair symbol")
    parser.add_argument("--timeframe", type=str, default="1h", help="Candle timeframe")
    parser.add_argument("--days", type=int, default=365, help="Number of days to backtest")
    parser.add_argument(
        "--fractions",
        type=str,
        default="0.05,0.08,0.10,0.12,0.15,0.20",
        help="Comma-separated list of position size fractions to test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: results/position_optimization_{timestamp}.csv)",
    )
    parser.add_argument(
        "--no-charts", action="store_true", help="Skip chart generation"
    )

    args = parser.parse_args()

    # Parse fractions
    base_fractions = [float(x.strip()) for x in args.fractions.split(",")]

    # Set default output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)
        args.output = str(
            output_dir / f"position_optimization_{args.symbol}_{args.timeframe}_{timestamp}.csv"
        )

    # Run optimization
    results_df = run_position_size_optimization(
        symbol=args.symbol,
        timeframe=args.timeframe,
        days=args.days,
        base_fractions=base_fractions,
        output_file=args.output,
    )

    # Print summary
    print_summary(results_df)

    # Create visualizations
    if not args.no_charts:
        output_dir = Path(args.output).parent
        try:
            create_visualization(results_df, output_dir)
        except Exception as e:
            print(f"\nWarning: Could not generate charts: {e}")

    # Save summary report
    summary_file = args.output.replace(".csv", "_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Position Size Optimization Report\n")
        f.write(f"{'='*80}\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Timeframe: {args.timeframe}\n")
        f.write(f"Days: {args.days}\n")
        f.write(f"Fractions Tested: {base_fractions}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"\n{results_df.to_string()}\n")

    print(f"\nSummary report saved to: {summary_file}")
    print(f"\n✅ Optimization complete! Review results in: {Path(args.output).parent}")


if __name__ == "__main__":
    main()
