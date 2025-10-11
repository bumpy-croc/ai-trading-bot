#!/usr/bin/env python3
"""Legacy strategy baseline benchmarking utilities.

This script captures Phase 0 benchmarking artefacts for the
`strategy_migration_proposal`. It exercises the existing legacy
backtesting engine and the live trading engine in paper-trading mode
using deterministic synthetic market data so that results are
reproducible offline.

Outputs are written to
`artifacts/strategy-migration/baseline/` and include JSON summaries,
trade logs, and a Markdown roll-up that can be used as regression
targets for later phases of the migration.

Example usage::

    python scripts/benchmark_legacy_baseline.py --strategies ml_basic ml_adaptive

The command above will run both backtest and live (paper) baselines for
the listed strategies. See `docs/strategy_migration_baseline.md` for
detailed documentation.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Ensure local src package is importable when running as a script
import sys

PROJECT_ROOT = _project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtesting.engine import Backtester
from src.data_providers.mock_data_provider import MockDataProvider
from src.live.trading_engine import LiveTradingEngine
from src.risk import RiskParameters
from src.strategies.ml_adaptive import MlAdaptive
from src.strategies.ml_basic import MlBasic

# Conservative trading constants so results stay comparable across runs
DEFAULT_INITIAL_BALANCE = 10_000.0
DEFAULT_TIMEFRAME = "1h"
DEFAULT_BACKTEST_DAYS = 30
DEFAULT_LIVE_STEPS = 50

# Global variable to store custom output directory
_OUTPUT_DIR: Path | None = None


def _baseline_dir() -> Path:
    if _OUTPUT_DIR is not None:
        out = _OUTPUT_DIR
    else:
        out = _project_root() / "artifacts" / "strategy-migration" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_strategy(name: str):
    """Instantiate a legacy strategy by slug."""

    normalized = name.lower()
    if normalized == "ml_basic":
        return MlBasic()
    if normalized == "ml_adaptive":
        return MlAdaptive()

    raise ValueError(
        f"Unsupported strategy '{name}'. Supported: ml_basic, ml_adaptive"
    )


def _json_safe(value: Any) -> Any:
    """Convert values into JSON-serialisable forms."""

    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if is_dataclass(value):
        return _json_safe(asdict(value))

    try:
        json.dumps(value)
    except TypeError:
        return str(value)
    return value


def _serialize_trades(trades: Iterable[Any]) -> list[dict[str, Any]]:
    """Convert trade records (dicts or dataclasses) into serialisable dicts."""

    serialised: list[dict[str, Any]] = []
    for trade in trades:
        if isinstance(trade, dict):
            serialised.append({key: _json_safe(value) for key, value in trade.items()})
        elif is_dataclass(trade):
            serialised.append(_json_safe(asdict(trade)))
        else:
            # Fallback to object attributes for unexpected types
            data: dict[str, Any] = {}
            for attr in dir(trade):
                if attr.startswith("_"):
                    continue
                value = getattr(trade, attr)
                if callable(value):
                    continue
                data[attr] = _json_safe(value)
            serialised.append(data)
    return serialised


def _write_trade_log(trades: Iterable[Any], path: Path) -> None:
    trade_dicts = _serialize_trades(trades)
    if not trade_dicts:
        path.write_text("trade_id,timestamp,side,entry_price,exit_price,pnl\n", encoding="utf-8")
        return
    df = pd.DataFrame(trade_dicts)
    df.to_csv(path, index=False)


def run_backtest_baseline(strategy_name: str, timeframe: str, days: int) -> dict[str, Any]:
    strategy = load_strategy(strategy_name)
    provider = MockDataProvider(interval_seconds=3600, num_candles=days * 24, seed=42)
    end = datetime.utcnow()
    start = end - timedelta(days=days)

    risk_params = RiskParameters(base_risk_per_trade=0.01, max_risk_per_trade=0.02)
    backtester = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=risk_params,
        initial_balance=DEFAULT_INITIAL_BALANCE,
        log_to_database=False,
    )

    out_dir = _baseline_dir()
    log_path = out_dir / f"baseline_backtest_{strategy_name}.log"

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file):
        results = backtester.run(
            symbol=strategy.get_trading_pair(),
            timeframe=timeframe,
            start=start,
            end=end,
        )
    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    dataset = provider.get_historical_data(
        symbol=strategy.get_trading_pair(), timeframe=timeframe, start=start, end=end
    )
    rows = len(dataset)

    output = {
        "mode": "backtest",
        "strategy": strategy_name,
        "timeframe": timeframe,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "rows": rows,
        "wall_time_seconds": wall_elapsed,
        "cpu_time_seconds": cpu_elapsed,
        "results": results,
    }

    json_path = out_dir / f"baseline_backtest_{strategy_name}.json"
    trades_path = out_dir / f"baseline_backtest_{strategy_name}_trades.csv"

    output.update(
        {
            "artifact_json": json_path.relative_to(_project_root()).as_posix(),
            "artifact_trades": trades_path.relative_to(_project_root()).as_posix(),
            "artifact_log": log_path.relative_to(_project_root()).as_posix(),
        }
    )

    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_trade_log(backtester.trades, trades_path)
    return output


def run_live_baseline(strategy_name: str, timeframe: str, steps: int) -> dict[str, Any]:
    strategy = load_strategy(strategy_name)
    provider = MockDataProvider(interval_seconds=3600, num_candles=steps * 5, seed=1337)

    risk_params = RiskParameters(base_risk_per_trade=0.01, max_risk_per_trade=0.02)
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=risk_params,
        check_interval=1,
        initial_balance=DEFAULT_INITIAL_BALANCE,
        max_position_size=0.2,
        enable_live_trading=False,
        log_trades=False,
        alert_webhook_url=None,
        enable_hot_swapping=False,
        resume_from_last_balance=False,
        database_url="sqlite:///:memory:",
        max_consecutive_errors=3,
        account_snapshot_interval=0,
        enable_dynamic_risk=False,
        enable_partial_operations=False,
    )

    out_dir = _baseline_dir()
    log_path = out_dir / f"baseline_live_{strategy_name}.log"

    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    with log_path.open("w", encoding="utf-8") as log_file, redirect_stdout(log_file):
        engine.start(
            symbol=strategy.get_trading_pair(),
            timeframe=timeframe,
            max_steps=steps,
        )
    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    summary = engine.get_performance_summary()
    trades = _serialize_trades(engine.completed_trades)

    json_path = out_dir / f"baseline_live_{strategy_name}.json"
    trades_path = out_dir / f"baseline_live_{strategy_name}_trades.csv"

    payload = {
        "mode": "live_paper",
        "strategy": strategy_name,
        "timeframe": timeframe,
        "steps": steps,
        "wall_time_seconds": wall_elapsed,
        "cpu_time_seconds": cpu_elapsed,
        "summary": summary,
        "trades": trades,
    }

    payload.update(
        {
            "artifact_json": json_path.relative_to(_project_root()).as_posix(),
            "artifact_trades": trades_path.relative_to(_project_root()).as_posix(),
            "artifact_log": log_path.relative_to(_project_root()).as_posix(),
        }
    )

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_trade_log(trades, trades_path)
    return payload


def write_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return

    out_dir = _baseline_dir()
    summary_json = out_dir / "baseline_summary.json"
    summary_md = out_dir / "baseline_summary.md"

    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    header = [
        "# Strategy Migration Baseline (Legacy Contract)",
        "",  # blank line
        "| Scenario | Strategy | Timeframe | Dataset/Steps | Trades | Final Balance | Return % | Wall Time (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    rows: list[str] = []
    for result in results:
        scenario = result["mode"]
        strategy = result["strategy"]
        timeframe = result.get("timeframe", "-")
        if scenario == "backtest":
            dataset = f"{result.get('rows', 0)} rows"
            trades = result.get("results", {}).get("total_trades", 0)
            final_balance = result.get("results", {}).get("final_balance", 0.0)
            total_return = result.get("results", {}).get("total_return", 0.0)
        else:
            dataset = f"{result.get('steps', 0)} steps"
            trades = result.get("summary", {}).get("total_trades", 0)
            final_balance = result.get("summary", {}).get("current_balance", 0.0)
            total_return = result.get("summary", {}).get("total_return", 0.0)

        row = (
            f"| {scenario} | {strategy} | {timeframe} | {dataset} | {trades} | "
            f"${final_balance:,.2f} | {total_return:.2f}% | {result.get('wall_time_seconds', 0.0):.2f} |"
        )
        rows.append(row)

    summary_md.write_text("\n".join(header + rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture legacy baseline benchmarks.")
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["ml_basic", "ml_adaptive"],
        help="Strategies to benchmark (default: ml_basic ml_adaptive)",
    )
    parser.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help="Candle timeframe for both backtests and live simulation",
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=DEFAULT_BACKTEST_DAYS,
        help="Number of days to include in the backtest benchmark",
    )
    parser.add_argument(
        "--live-steps",
        type=int,
        default=DEFAULT_LIVE_STEPS,
        help="Number of loop iterations to execute in the live engine",
    )
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip live (paper trading) baseline generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory (relative to project root or absolute path)",
    )
    return parser.parse_args()


def main() -> None:
    global _OUTPUT_DIR
    args = parse_args()
    
    # Set the output directory if provided
    if args.output_dir:
        output_path = Path(args.output_dir)
        if not output_path.is_absolute():
            output_path = _project_root() / output_path
        _OUTPUT_DIR = output_path
    
    results: list[dict[str, Any]] = []

    for strategy in args.strategies:
        print(f"Running backtest baseline for {strategy}...")
        backtest_result = run_backtest_baseline(strategy, args.timeframe, args.backtest_days)
        results.append(backtest_result)

        if args.skip_live:
            continue

        print(f"Running live baseline for {strategy}...")
        live_result = run_live_baseline(strategy, args.timeframe, args.live_steps)
        results.append(live_result)

    write_summary(results)
    print("Baseline benchmarking complete. Artefacts written to", _baseline_dir())


if __name__ == "__main__":
    main()
