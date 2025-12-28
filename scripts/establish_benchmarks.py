#!/usr/bin/env python3
"""
Comprehensive Buy-and-Hold Benchmark Script

This script establishes buy-and-hold performance benchmarks that active
trading strategies must beat. It runs comprehensive backtests across:

- Multiple symbols (BTC, ETH)
- Multiple timeframes (1h, 4h, 1d)
- Multiple time periods (1 year, 3 years, 5 years, 10 years)
- Calculates comprehensive metrics (returns, Sharpe, drawdown, etc.)

Results are saved to docs/buy_and_hold_benchmark.md for reference.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.infrastructure.logging.config import configure_logging
from src.strategies.buy_and_hold import create_buy_and_hold_strategy


def run_benchmark(
    symbol: str,
    timeframe: str,
    years: int,
    initial_balance: float = 10_000,
) -> dict:
    """
    Run a single benchmark backtest

    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframe: Candle timeframe (e.g., "1h", "4h", "1d")
        years: Number of years to backtest
        initial_balance: Starting capital

    Returns:
        Dictionary containing benchmark results
    """
    print(f"\n{'='*70}")
    print(f"Running benchmark: {symbol} | {timeframe} | {years} years")
    print(f"{'='*70}")

    # Create strategy and data provider
    # IMPORTANT: Uses default position_fraction=1.0 (100% invested)
    # This is TRUE buy-and-hold - fully invested, not a hybrid portfolio
    strategy = create_buy_and_hold_strategy()
    provider = BinanceProvider()
    cached_provider = CachedDataProvider(provider, cache_ttl_hours=24)

    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        data_provider=cached_provider,
        initial_balance=initial_balance,
        log_to_database=False,  # Don't log benchmarks to database
    )

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)

    try:
        # Run backtest
        results = backtester.run(
            symbol=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
        )

        # Extract key metrics
        benchmark = {
            "symbol": symbol,
            "timeframe": timeframe,
            "period_years": years,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "initial_balance": initial_balance,
            "final_balance": results.get("final_balance", 0),
            "total_return_pct": results.get("total_return", 0),
            "annualized_return_pct": results.get("annualized_return", 0),
            "max_drawdown_pct": results.get("max_drawdown", 0),
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "sortino_ratio": results.get("sortino_ratio", 0),
            "calmar_ratio": results.get("calmar_ratio", 0),
            "total_trades": results.get("total_trades", 1),  # Buy-and-hold = 1 trade
            "win_rate_pct": results.get("win_rate", 100),
            "profit_factor": results.get("profit_factor", 0),
            "avg_trade_return_pct": results.get("avg_trade_return", 0),
            "yearly_returns": results.get("yearly_returns", {}),
            "volatility_annualized": results.get("volatility_annualized", 0),
            "days_traded": results.get("total_days", years * 365),
        }

        # Print summary
        print(f"\n✅ RESULTS:")
        print(f"   Total Return: {benchmark['total_return_pct']:.2f}%")
        print(f"   Annualized Return (CAGR): {benchmark['annualized_return_pct']:.2f}%")
        print(f"   Max Drawdown: {benchmark['max_drawdown_pct']:.2f}%")
        print(f"   Sharpe Ratio: {benchmark['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio: {benchmark['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio: {benchmark['calmar_ratio']:.2f}")
        print(f"   Final Balance: ${benchmark['final_balance']:,.2f}")

        return benchmark

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "period_years": years,
            "error": str(e),
            "status": "failed",
        }


def generate_markdown_report(benchmarks: list[dict], output_path: Path):
    """
    Generate a markdown report from benchmark results

    Args:
        benchmarks: List of benchmark result dictionaries
        output_path: Path to write the markdown file
    """
    md = []
    md.append("# Buy-and-Hold Performance Benchmarks")
    md.append("")
    md.append("**Generated:** " + datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC"))
    md.append("")
    md.append(
        "This document establishes buy-and-hold performance benchmarks for BTC and ETH across multiple timeframes and time periods."
    )
    md.append(
        "These benchmarks represent the baseline that active trading strategies must beat to justify their complexity and risk."
    )
    md.append("")
    md.append("## Executive Summary")
    md.append("")

    # Calculate summary statistics
    successful = [b for b in benchmarks if "error" not in b]
    failed = [b for b in benchmarks if "error" in b]

    md.append(f"- **Total Benchmarks Run:** {len(benchmarks)}")
    md.append(f"- **Successful:** {len(successful)}")
    md.append(f"- **Failed:** {len(failed)}")
    md.append("")

    if successful:
        avg_return = sum(b["total_return_pct"] for b in successful) / len(successful)
        max_return = max(b["total_return_pct"] for b in successful)
        min_return = min(b["total_return_pct"] for b in successful)
        avg_sharpe = sum(b["sharpe_ratio"] for b in successful) / len(successful)

        md.append(f"- **Average Total Return:** {avg_return:.2f}%")
        md.append(f"- **Best Return:** {max_return:.2f}%")
        md.append(f"- **Worst Return:** {min_return:.2f}%")
        md.append(f"- **Average Sharpe Ratio:** {avg_sharpe:.2f}")
        md.append("")

    # Group by symbol
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        symbol_benchmarks = [b for b in successful if b.get("symbol") == symbol]
        if not symbol_benchmarks:
            continue

        md.append(f"## {symbol} Benchmarks")
        md.append("")

        # Group by time period
        for years in [1, 3, 5, 10]:
            period_benchmarks = [b for b in symbol_benchmarks if b.get("period_years") == years]
            if not period_benchmarks:
                continue

            md.append(f"### {years}-Year Period")
            md.append("")
            md.append("| Timeframe | Total Return | CAGR | Max DD | Sharpe | Sortino | Calmar | Volatility |")
            md.append("|-----------|--------------|------|--------|--------|---------|--------|------------|")

            for bench in sorted(period_benchmarks, key=lambda x: x.get("timeframe", "")):
                md.append(
                    f"| {bench['timeframe']} "
                    f"| {bench['total_return_pct']:.2f}% "
                    f"| {bench['annualized_return_pct']:.2f}% "
                    f"| {bench['max_drawdown_pct']:.2f}% "
                    f"| {bench['sharpe_ratio']:.2f} "
                    f"| {bench['sortino_ratio']:.2f} "
                    f"| {bench['calmar_ratio']:.2f} "
                    f"| {bench.get('volatility_annualized', 0):.2f}% |"
                )

            md.append("")

            # Yearly returns if available
            if period_benchmarks and period_benchmarks[0].get("yearly_returns"):
                md.append("**Yearly Returns Breakdown:**")
                md.append("")
                # Use the first benchmark's yearly returns as representative
                yearly = period_benchmarks[0]["yearly_returns"]
                for year in sorted(yearly.keys()):
                    md.append(f"- {year}: {yearly[year]:.2f}%")
                md.append("")

    # Add failed benchmarks section if any
    if failed:
        md.append("## Failed Benchmarks")
        md.append("")
        md.append("The following benchmarks failed to complete:")
        md.append("")
        for bench in failed:
            md.append(
                f"- {bench['symbol']} | {bench['timeframe']} | {bench['period_years']} years: `{bench['error']}`"
            )
        md.append("")

    # Add notes and interpretation
    md.append("## Key Insights")
    md.append("")
    md.append("### What These Benchmarks Tell Us")
    md.append("")
    md.append(
        "1. **Total Returns**: Buy-and-hold returns represent the baseline. Any active strategy must beat these returns to justify its complexity."
    )
    md.append(
        "2. **Maximum Drawdown**: Shows the worst peak-to-trough decline. Active strategies should aim for lower drawdowns while maintaining competitive returns."
    )
    md.append(
        "3. **Sharpe Ratio**: Risk-adjusted returns. Higher is better. Active strategies should target Sharpe > 1.5."
    )
    md.append(
        "4. **Calmar Ratio**: Return per unit of drawdown risk. Measures how much return you get for the pain endured."
    )
    md.append("")
    md.append("### Success Criteria for Active Strategies")
    md.append("")
    md.append("To justify active trading, strategies must:")
    md.append("")
    md.append("✅ **BEAT** buy-and-hold total returns over 5+ year periods")
    md.append("✅ **REDUCE** maximum drawdown below buy-and-hold levels")
    md.append("✅ **ACHIEVE** Sharpe ratio > 1.5 (preferably > 2.0)")
    md.append("✅ **MAINTAIN** consistency across different market regimes")
    md.append("✅ **SURVIVE** validation on out-of-sample data")
    md.append("")

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(md))
    print(f"\n📝 Report saved to: {output_path}")


def main():
    """Main benchmark execution"""
    configure_logging()

    print("\n" + "=" * 70)
    print("BUY-AND-HOLD BENCHMARK ESTABLISHMENT")
    print("=" * 70)
    print("\nThis script will establish comprehensive buy-and-hold benchmarks")
    print("for BTC and ETH across multiple timeframes and time periods.")
    print("\nThis will take a while (downloading and processing historical data)...")
    print("Results will be saved to docs/buy_and_hold_benchmark.md")

    # Define test matrix
    test_matrix = [
        # BTC benchmarks
        ("BTCUSDT", "1d", 1),  # 1 year, daily candles
        ("BTCUSDT", "1d", 3),  # 3 years, daily candles
        ("BTCUSDT", "1d", 5),  # 5 years, daily candles
        ("BTCUSDT", "4h", 5),  # 5 years, 4-hour candles
        ("BTCUSDT", "1h", 3),  # 3 years, 1-hour candles (large dataset)
        # ETH benchmarks
        ("ETHUSDT", "1d", 1),  # 1 year, daily candles
        ("ETHUSDT", "1d", 3),  # 3 years, daily candles
        ("ETHUSDT", "1d", 5),  # 5 years, daily candles
        ("ETHUSDT", "4h", 5),  # 5 years, 4-hour candles
        ("ETHUSDT", "1h", 3),  # 3 years, 1-hour candles
        # Stretch: 10-year benchmarks (may fail if data not available)
        # ("BTCUSDT", "1d", 10),  # 10 years if available
        # ("ETHUSDT", "1d", 10),  # 10 years if available
    ]

    # Run all benchmarks
    benchmarks = []
    for symbol, timeframe, years in test_matrix:
        result = run_benchmark(symbol, timeframe, years)
        benchmarks.append(result)

    # Save results to JSON
    results_file = PROJECT_ROOT / "results" / "buy_and_hold_benchmarks.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(benchmarks, f, indent=2)
    print(f"\n💾 Raw results saved to: {results_file}")

    # Generate markdown report
    report_file = PROJECT_ROOT / "docs" / "buy_and_hold_benchmark.md"
    generate_markdown_report(benchmarks, report_file)

    print("\n" + "=" * 70)
    print("✅ BENCHMARK ESTABLISHMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to:")
    print(f"  - Markdown report: {report_file}")
    print(f"  - JSON data: {results_file}")
    print("\nNext steps:")
    print("  1. Review the benchmarks in docs/buy_and_hold_benchmark.md")
    print("  2. Run existing strategies against these benchmarks")
    print("  3. Identify periods where active trading can add value")
    print("  4. Design aggressive strategies to beat buy-and-hold")


if __name__ == "__main__":
    main()
