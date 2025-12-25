from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterable
from contextlib import redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure project root and src are importable when packaged
from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

from src.engines.backtest.engine import Backtester  # noqa: E402
from src.data_providers.mock_data_provider import MockDataProvider  # noqa: E402
from src.engines.live.trading_engine import LiveTradingEngine  # noqa: E402
from src.risk import RiskParameters  # noqa: E402
from src.strategies.ml_adaptive import create_ml_adaptive_strategy  # noqa: E402
from src.strategies.ml_basic import create_ml_basic_strategy  # noqa: E402

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
        out = PROJECT_ROOT / "artifacts" / "strategy-migration" / "baseline"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_strategy(name: str):
    normalized = name.lower()
    if normalized == "ml_basic":
        return create_ml_basic_strategy()
    if normalized == "ml_adaptive":
        return create_ml_adaptive_strategy()
    raise ValueError("Supported strategies: ml_basic, ml_adaptive")


def _json_safe(value: Any) -> Any:
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
    serialised: list[dict[str, Any]] = []
    for trade in trades:
        if isinstance(trade, dict):
            serialised.append({key: _json_safe(val) for key, val in trade.items()})
        elif is_dataclass(trade):
            serialised.append(_json_safe(asdict(trade)))
        else:
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
    serialised = _serialize_trades(trades)
    if not serialised:
        path.write_text("trade_id,timestamp,side,entry_price,exit_price,pnl\n", encoding="utf-8")
        return
    import pandas as pd

    df = pd.DataFrame(serialised)
    df.to_csv(path, index=False)


def _run_backtest(strategy_name: str, timeframe: str, days: int) -> dict[str, Any]:
    strategy = _load_strategy(strategy_name)
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
    json_path = out_dir / f"baseline_backtest_{strategy_name}.json"
    trades_path = out_dir / f"baseline_backtest_{strategy_name}_trades.csv"
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
        "artifact_json": json_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_trades": trades_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_log": log_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_trade_log(backtester.trades, trades_path)
    return output


def _run_live(strategy_name: str, timeframe: str, steps: int) -> dict[str, Any]:
    strategy = _load_strategy(strategy_name)
    provider = MockDataProvider(interval_seconds=3600, num_candles=steps * 2, seed=99)
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
        engine.start(symbol=strategy.get_trading_pair(), timeframe=timeframe, max_steps=steps)
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
        "artifact_json": json_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_trades": trades_path.relative_to(PROJECT_ROOT).as_posix(),
        "artifact_log": log_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_trade_log(trades, trades_path)
    return payload


def _write_summary(results: list[dict[str, Any]]) -> None:
    if not results:
        return
    out_dir = _baseline_dir()
    summary_json = out_dir / "baseline_summary.json"
    summary_md = out_dir / "baseline_summary.md"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    header = [
        "# Strategy Migration Baseline (Legacy Contract)",
        "",
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


def _handle_baseline(ns: argparse.Namespace) -> int:
    global _OUTPUT_DIR

    # Set the output directory if provided
    if hasattr(ns, "output_dir") and ns.output_dir:
        output_path = Path(ns.output_dir)
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path
        _OUTPUT_DIR = output_path

    results: list[dict[str, Any]] = []
    for strategy in ns.strategies:
        print(f"Running backtest baseline for {strategy}...")
        try:
            results.append(_run_backtest(strategy, ns.timeframe, ns.backtest_days))
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Backtest baseline failed for {strategy}: {exc}")
            return 1
        if ns.skip_live:
            continue
        print(f"Running live baseline for {strategy}...")
        try:
            results.append(_run_live(strategy, ns.timeframe, ns.live_steps))
        except Exception as exc:  # noqa: BLE001
            print(f"❌ Live baseline failed for {strategy}: {exc}")
            return 1
    _write_summary(results)
    print("Baseline benchmarking complete. Artefacts written to", _baseline_dir())
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("migration", help="Strategy migration utilities")
    sub = parser.add_subparsers(dest="migration_cmd", required=True)
    p_baseline = sub.add_parser(
        "baseline",
        help="Capture legacy baseline benchmark artefacts for selected strategies",
    )
    p_baseline.add_argument(
        "--strategies",
        nargs="+",
        default=["ml_basic", "ml_adaptive"],
        help="Strategies to benchmark (default: ml_basic ml_adaptive)",
    )
    p_baseline.add_argument(
        "--timeframe",
        default=DEFAULT_TIMEFRAME,
        help="Candle timeframe for benchmarks (default: 1h)",
    )
    p_baseline.add_argument(
        "--backtest-days",
        type=int,
        default=DEFAULT_BACKTEST_DAYS,
        help="Number of days to include in backtest benchmark",
    )
    p_baseline.add_argument(
        "--live-steps",
        type=int,
        default=DEFAULT_LIVE_STEPS,
        help="Number of loop iterations for live (paper) benchmark",
    )
    p_baseline.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip live (paper trading) baseline generation",
    )
    p_baseline.add_argument(
        "--output-dir",
        type=str,
        help="Custom output directory (relative to project root or absolute path)",
    )
    p_baseline.set_defaults(func=_handle_baseline)
