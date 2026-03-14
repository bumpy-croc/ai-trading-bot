from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.infrastructure.runtime.paths import get_project_root
from src.optimizer.walk_forward import FoldResult, WalkForwardResult

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _print_fold_table(folds: list[FoldResult]) -> None:
    """Pretty-print a per-fold results table."""
    header = (
        f"{'Fold':>4}  {'IS Sharpe':>10}  {'OOS Sharpe':>11}  {'Ratio':>7}  "
        f"{'IS Ret%':>8}  {'OOS Ret%':>9}  {'IS DD%':>7}  {'OOS DD%':>8}  "
        f"{'IS Trades':>9}  {'OOS Trades':>10}"
    )
    print(header)
    print("-" * len(header))
    for f in folds:
        print(
            f"{f.fold_index + 1:>4}  {f.is_sharpe:>10.2f}  {f.oos_sharpe:>11.2f}  "
            f"{f.robustness_ratio:>7.2f}  {f.is_return:>8.2f}  {f.oos_return:>9.2f}  "
            f"{f.is_max_drawdown:>7.2f}  {f.oos_max_drawdown:>8.2f}  "
            f"{f.is_total_trades:>9}  {f.oos_total_trades:>10}"
        )


def _print_summary(result: WalkForwardResult) -> None:
    """Print the overall robustness assessment."""
    print()
    print("=" * 60)
    print("Walk-Forward Analysis Summary")
    print("=" * 60)
    print(f"  Strategy:              {result.config.strategy_name}")
    print(f"  Symbol:                {result.config.symbol}")
    print(f"  Timeframe:             {result.config.timeframe}")
    print(f"  Folds:                 {result.num_folds}")
    print(f"  Train window:          {result.config.train_days}d")
    print(f"  Test window:           {result.config.test_days}d")
    print()
    print(f"  Mean IS Sharpe:        {result.mean_is_sharpe:.3f}")
    print(f"  Mean OOS Sharpe:       {result.mean_oos_sharpe:.3f}")
    print(f"  Mean Robustness Ratio: {result.mean_robustness_ratio:.3f}")
    print(f"  Median Robustness:     {result.median_robustness_ratio:.3f}")
    print(f"  Overfitting Detected:  {'YES' if result.overfitting_detected else 'No'}")
    print(f"  Robustness:            {result.robustness_label}")
    print("=" * 60)


def _handle(ns: argparse.Namespace) -> int:
    try:
        from datetime import UTC, datetime, timedelta

        from src.optimizer.walk_forward import WalkForwardAnalyzer, WalkForwardConfig

        end = datetime.now(UTC)

        cfg = WalkForwardConfig(
            strategy_name=ns.strategy,
            symbol=ns.symbol,
            timeframe=ns.timeframe,
            train_days=ns.train_days,
            test_days=ns.test_days,
            num_folds=ns.folds,
            initial_balance=ns.initial_balance,
            provider=ns.provider,
            use_cache=not ns.no_cache,
            random_seed=ns.seed,
        )

        analyzer = WalkForwardAnalyzer(cfg)
        result = analyzer.run(end=end)

        _print_fold_table(result.folds)
        _print_summary(result)

        return 0
    except Exception as exc:
        if getattr(ns, "debug", False):
            raise
        logging.exception("Walk-forward analysis failed")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    from src.config.constants import DEFAULT_INITIAL_BALANCE

    p = subparsers.add_parser(
        "walk-forward",
        help="Run walk-forward analysis to validate strategy robustness",
    )
    p.add_argument("strategy", nargs="?", default="ml_basic", help="Strategy name")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--train-days", type=int, default=180, help="In-sample window (days)")
    p.add_argument("--test-days", type=int, default=30, help="Out-of-sample window (days)")
    p.add_argument("--folds", type=int, default=6, help="Number of rolling folds")
    p.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    p.add_argument(
        "--provider",
        choices=["binance", "coinbase", "mock", "fixture"],
        default="mock",
    )
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--debug", action="store_true", help="Re-raise exceptions with full traceback")
    p.set_defaults(func=_handle)
