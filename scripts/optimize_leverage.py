#!/usr/bin/env python3
"""
Leverage Optimization Script

This script systematically tests different leverage levels (position sizes) to find
the optimal balance between maximum returns and acceptable risk.

System Constraint: Maximum position size is 0.5 (50%) per FixedFractionSizer

Strategy:
1. Test each strategy with multiple leverage levels (10%, 20%, 30%, 40%, 50%)
2. Compare key metrics: Total Return, Max Drawdown, Sharpe Ratio, Recovery Factor
3. Identify the optimal leverage for each strategy type
4. Generate comprehensive comparison reports

Usage:
    python scripts/optimize_leverage.py --symbol BTCUSDT --timeframe 4h --days 365
    python scripts/optimize_leverage.py --symbol BTCUSDT --start 2020-01-01 --end 2024-12-31
    python scripts/optimize_leverage.py --all-strategies --days 730
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.infrastructure.logging.config import configure_logging
from src.infrastructure.runtime.cache import get_cache_ttl_for_provider
from src.risk.risk_manager import RiskParameters
from src.strategies import (
    create_aggressive_regime_adaptive_strategy,
    create_aggressive_trend_strategy,
    create_ultra_aggressive_trend_strategy,
    create_ultra_volatile_exploiter_strategy,
    create_volatility_exploiter_strategy,
)

# Test leverage levels (as position size fractions)
LEVERAGE_LEVELS = [0.10, 0.20, 0.30, 0.40, 0.50]  # 10% to 50%

# Strategies to test (most aggressive variants)
TEST_STRATEGIES = {
    "aggressive_trend": create_aggressive_trend_strategy,
    "ultra_aggressive_trend": create_ultra_aggressive_trend_strategy,
    "volatility_exploiter": create_volatility_exploiter_strategy,
    "ultra_volatile_exploiter": create_ultra_volatile_exploiter_strategy,
    "aggressive_regime_adaptive": create_aggressive_regime_adaptive_strategy,
}


def create_leverage_variant(
    base_strategy_func: callable, leverage: float, name_suffix: str
) -> Any:
    """
    Create a strategy variant with custom leverage (position size)

    Args:
        base_strategy_func: Base strategy creation function
        leverage: Position size fraction (0.1-0.5)
        name_suffix: Suffix to add to strategy name

    Returns:
        Strategy instance with modified position sizing
    """
    strategy = base_strategy_func(name=f"{base_strategy_func.__name__}_{name_suffix}")

    # Override position sizing in risk overrides
    if hasattr(strategy, "_risk_overrides"):
        strategy._risk_overrides["base_fraction"] = leverage
        strategy._risk_overrides["max_fraction"] = leverage
        # Also update config if present
        if hasattr(strategy, "config"):
            strategy.config["base_fraction"] = leverage
            strategy.config["max_fraction"] = leverage

    return strategy


def run_leverage_test(
    strategy_name: str,
    strategy_func: callable,
    leverage: float,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float,
    data_provider: Any,
) -> dict:
    """
    Run a single backtest with specific leverage level

    Returns:
        Dictionary with test results
    """
    print(f"\n{'=' * 80}")
    print(f"Testing {strategy_name} with {leverage*100:.0f}% leverage")
    print(f"{'=' * 80}")

    # Create strategy variant with custom leverage
    strategy = create_leverage_variant(
        strategy_func, leverage, f"{int(leverage*100)}pct"
    )

    # Risk parameters (minimal - we're testing pure leverage impact)
    risk_params = RiskParameters(
        base_risk_per_trade=0.02,
        max_risk_per_trade=0.05,
        max_drawdown=0.90,  # Allow high drawdown for testing
    )

    # Create backtester
    backtester = Backtester(
        strategy=strategy,
        data_provider=data_provider,
        risk_parameters=risk_params,
        initial_balance=initial_balance,
        log_to_database=False,  # Disable DB logging for speed
    )

    # Run backtest
    results = backtester.run(
        symbol=symbol, timeframe=timeframe, start=start_date, end=end_date
    )

    # Calculate additional metrics
    recovery_factor = (
        results["total_return"] / abs(results["max_drawdown"])
        if results["max_drawdown"] != 0
        else 0
    )

    # Return key metrics
    return {
        "strategy": strategy_name,
        "leverage": leverage,
        "leverage_pct": f"{leverage*100:.0f}%",
        "total_return": results["total_return"],
        "annualized_return": results["annualized_return"],
        "max_drawdown": results["max_drawdown"],
        "sharpe_ratio": results["sharpe_ratio"],
        "win_rate": results["win_rate"],
        "total_trades": results["total_trades"],
        "final_balance": results["final_balance"],
        "recovery_factor": recovery_factor,
        "profit_factor": results.get("profit_factor", 0),
        "hold_return": results["hold_return"],
        "vs_hold": results["trading_vs_hold_difference"],
    }


def optimize_strategy_leverage(
    strategy_name: str,
    strategy_func: callable,
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
    initial_balance: float,
    data_provider: Any,
) -> list[dict]:
    """
    Test a strategy across all leverage levels

    Returns:
        List of results for each leverage level
    """
    results = []

    for leverage in LEVERAGE_LEVELS:
        try:
            result = run_leverage_test(
                strategy_name,
                strategy_func,
                leverage,
                symbol,
                timeframe,
                start_date,
                end_date,
                initial_balance,
                data_provider,
            )
            results.append(result)

            # Print quick summary
            print(f"\nResults for {leverage*100:.0f}% leverage:")
            print(f"  Total Return: {result['total_return']:>8.2f}%")
            print(f"  Max Drawdown: {result['max_drawdown']:>8.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:>8.2f}")
            print(f"  Recovery Factor: {result['recovery_factor']:>8.2f}")
            print(f"  vs Buy-Hold: {result['vs_hold']:>8.2f}%")

        except Exception as e:
            print(f"ERROR testing {leverage*100:.0f}% leverage: {e}")
            continue

    return results


def find_optimal_leverage(results: list[dict], metric: str = "sharpe_ratio") -> dict:
    """
    Find the optimal leverage level based on specified metric

    Args:
        results: List of leverage test results
        metric: Metric to optimize ('sharpe_ratio', 'total_return', 'recovery_factor')

    Returns:
        Result dict with optimal leverage
    """
    if not results:
        return {}

    # Find result with best metric value
    optimal = max(results, key=lambda x: x.get(metric, float("-inf")))
    return optimal


def generate_report(
    all_results: dict[str, list[dict]], output_dir: Path
) -> None:
    """
    Generate comprehensive leverage optimization report

    Args:
        all_results: Dict mapping strategy name to list of results
        output_dir: Directory to save reports
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate markdown report
    md_path = output_dir / "leverage_optimization_report.md"
    with open(md_path, "w") as f:
        f.write("# Leverage Optimization Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for strategy_name, results in all_results.items():
            if not results:
                continue

            f.write(f"## {strategy_name}\n\n")

            # Results table
            f.write("| Leverage | Total Return | Max DD | Sharpe | Recovery | vs Buy-Hold |\n")
            f.write("|----------|--------------|--------|--------|----------|-------------|\n")

            for r in results:
                f.write(
                    f"| {r['leverage_pct']:>8} | "
                    f"{r['total_return']:>11.2f}% | "
                    f"{r['max_drawdown']:>6.2f}% | "
                    f"{r['sharpe_ratio']:>6.2f} | "
                    f"{r['recovery_factor']:>8.2f} | "
                    f"{r['vs_hold']:>11.2f}% |\n"
                )

            # Optimal configurations
            f.write("\n**Optimal Leverage Levels:**\n\n")

            optimal_sharpe = find_optimal_leverage(results, "sharpe_ratio")
            f.write(
                f"- **Best Sharpe Ratio**: {optimal_sharpe.get('leverage_pct', 'N/A')} "
                f"(Sharpe: {optimal_sharpe.get('sharpe_ratio', 0):.2f})\n"
            )

            optimal_return = find_optimal_leverage(results, "total_return")
            f.write(
                f"- **Best Total Return**: {optimal_return.get('leverage_pct', 'N/A')} "
                f"(Return: {optimal_return.get('total_return', 0):.2f}%)\n"
            )

            optimal_recovery = find_optimal_leverage(results, "recovery_factor")
            f.write(
                f"- **Best Recovery Factor**: {optimal_recovery.get('leverage_pct', 'N/A')} "
                f"(Recovery: {optimal_recovery.get('recovery_factor', 0):.2f})\n"
            )

            f.write("\n---\n\n")

    # Generate JSON data
    json_path = output_dir / "leverage_optimization_data.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "leverage_levels": LEVERAGE_LEVELS,
                "strategies": all_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'=' * 80}")
    print(f"Reports generated:")
    print(f"  Markdown: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"  JSON:     {json_path.relative_to(PROJECT_ROOT)}")
    print(f"{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Optimize leverage levels for strategies")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="4h", help="Timeframe")
    parser.add_argument("--days", type=int, help="Number of days to test")
    parser.add_argument("--start", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", help="End date YYYY-MM-DD")
    parser.add_argument("--initial-balance", type=float, default=10000, help="Initial balance")
    parser.add_argument(
        "--all-strategies",
        action="store_true",
        help="Test all strategies (default: test all)",
    )
    parser.add_argument("--strategy", help="Test specific strategy only")
    parser.add_argument(
        "--output-dir",
        default="logs/leverage_optimization",
        help="Output directory for reports",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Determine date range
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
        end_date = datetime.strptime(args.end, "%Y-%m-%d")
    elif args.days:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        # Default: 1 year
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

    print(f"\n{'=' * 80}")
    print("LEVERAGE OPTIMIZATION")
    print(f"{'=' * 80}")
    print(f"Symbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Initial Balance: ${args.initial_balance:,.2f}")
    print(f"Leverage Levels: {[f'{int(l*100)}%' for l in LEVERAGE_LEVELS]}")
    print(f"{'=' * 80}\n")

    # Setup data provider
    provider = BinanceProvider()
    cache_ttl = get_cache_ttl_for_provider(provider)
    data_provider = CachedDataProvider(provider, cache_ttl_hours=cache_ttl)

    # Determine which strategies to test
    if args.strategy:
        if args.strategy not in TEST_STRATEGIES:
            print(f"ERROR: Unknown strategy '{args.strategy}'")
            print(f"Available: {', '.join(TEST_STRATEGIES.keys())}")
            return 1
        strategies_to_test = {args.strategy: TEST_STRATEGIES[args.strategy]}
    else:
        strategies_to_test = TEST_STRATEGIES

    # Run optimization for each strategy
    all_results = {}

    for strategy_name, strategy_func in strategies_to_test.items():
        print(f"\n{'#' * 80}")
        print(f"# Optimizing: {strategy_name}")
        print(f"{'#' * 80}\n")

        results = optimize_strategy_leverage(
            strategy_name,
            strategy_func,
            args.symbol,
            args.timeframe,
            start_date,
            end_date,
            args.initial_balance,
            data_provider,
        )

        all_results[strategy_name] = results

    # Generate reports
    output_dir = PROJECT_ROOT / args.output_dir
    generate_report(all_results, output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
