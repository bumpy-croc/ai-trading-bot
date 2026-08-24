"""GH #1081 Step 2 item 1 rerun — short-suppression counterfactual, provenance-verified.

Thin research-only runner reusing the same construction path as
`cli.commands.backtest._handle` (RiskParameters, Backtester, model-version
pinning via `resolve_strategy_max_position_size`), diverging only to thread
`allow_shorts` explicitly since the CLI has no --allow-shorts flag (matches
the methodology of the original counterfactual and the tier-restore
reproduction, both of which used the same in-process bypass).

Not committed to src/ — session scratchpad artifact, per the original
study's own Sec 10 convention.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime

# ---- Provenance banner: MUST print before anything else touches src/cli ----
import src  # noqa: E402

_SRC_FILE = src.__file__
_SRC_ROOT = _SRC_FILE.rsplit("/src/", 1)[0]
_GIT_HEAD = subprocess.run(
    ["git", "-C", _SRC_ROOT, "rev-parse", "HEAD"], capture_output=True, text=True
).stdout.strip()
_GIT_BRANCH = subprocess.run(
    ["git", "-C", _SRC_ROOT, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
).stdout.strip()
import os

_PYTHONPATH = os.environ.get("PYTHONPATH", "<unset>")


def print_provenance_banner() -> None:
    print("=" * 70)
    print("PROVENANCE BANNER")
    print(f"  src.__file__ resolved root : {_SRC_ROOT}")
    print(f"  git rev-parse HEAD          : {_GIT_HEAD}")
    print(f"  git branch                  : {_GIT_BRANCH}")
    print(f"  effective PYTHONPATH        : {_PYTHONPATH}")
    print(f"  python executable           : {sys.executable}")
    print("=" * 70)


print_provenance_banner()

from src.engines.backtest.engine import Backtester  # noqa: E402
from src.engines.shared.risk_configuration import resolve_strategy_max_position_size  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402
from src.strategies import call_strategy_factory  # noqa: E402
from src.strategies.hyper_growth import create_hyper_growth_strategy  # noqa: E402

MODEL_VERSION = "2026-07-04_22h_v1"
SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"


def build_data_provider():
    from src.data_providers.cached_data_provider import CachedDataProvider
    from src.data_providers.provider_factory import create_data_provider

    provider = create_data_provider(provider_type="auto")
    return CachedDataProvider(provider, cache_ttl_hours=24 * 365)


def run_arm(
    *,
    label: str,
    allow_shorts: bool,
    start: datetime,
    end: datetime,
    initial_balance: float,
    max_position_size_override: float | None = None,
    risk_kwargs: dict | None = None,
) -> dict:
    """Construct HyperGrowth in-process (allow_shorts explicit) and run.

    Mirrors cli.commands.backtest._handle: risk_params_kwargs built the same
    way (explicit CLI-equivalent overrides win, else strategy's own
    max_fraction via resolve_strategy_max_position_size, else ratified
    default — never the bare-Backtester #1088 default of 0.20 by accident).
    """
    strategy = create_hyper_growth_strategy(
        symbol=SYMBOL, model_version=MODEL_VERSION, allow_shorts=allow_shorts
    )

    risk_params_kwargs: dict = dict(risk_kwargs or {})
    if max_position_size_override is not None:
        risk_params_kwargs["max_position_size"] = max_position_size_override
    else:
        strategy_max_position = resolve_strategy_max_position_size(strategy)
        if strategy_max_position is not None:
            risk_params_kwargs["max_position_size"] = strategy_max_position
    risk_params = RiskParameters(**risk_params_kwargs)

    data_provider = build_data_provider()

    backtester = Backtester(
        strategy=strategy,
        data_provider=data_provider,
        sentiment_provider=None,
        risk_parameters=risk_params,
        initial_balance=initial_balance,
        log_to_database=False,
        enable_engine_risk_exits=True,
        enable_dynamic_risk=True,
    )

    results = backtester.run(symbol=SYMBOL, timeframe=TIMEFRAME, start=start, end=end)

    # BaseTrade.side is a PositionSide enum (or the raw string in some
    # paths) — normalize with str() before comparing.
    completed_trades = backtester.trades
    short_trades = [t for t in completed_trades if str(t.side) == "short"]
    long_trades = [t for t in completed_trades if str(t.side) == "long"]
    short_pnl_sum = sum(t.pnl_percent for t in short_trades if t.pnl_percent is not None)

    summary = {
        "label": label,
        "allow_shorts": allow_shorts,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "initial_balance": initial_balance,
        "effective_max_position_size": risk_params.max_position_size,
        "total_trades": results.get("total_trades"),
        "short_trades": len(short_trades),
        "long_trades": len(long_trades),
        "total_return_pct": results.get("total_return"),
        "max_drawdown_pct": results.get("max_drawdown"),
        "win_rate_pct": results.get("win_rate"),
        "sharpe_ratio": results.get("sharpe_ratio"),
        "profit_factor": results.get("profit_factor"),
        "final_balance": results.get("final_balance"),
        "short_side_pnl_pct_sum": short_pnl_sum,
    }
    print(json.dumps(summary, indent=2, default=str))
    return summary


def dt(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)


ALL_RESULTS: list[dict] = []


def main() -> None:
    # ---- 1. Segment B (live-matched, original study Sec 7.1) ----
    for allow_shorts, label in [(True, "segB_shorts_enabled"), (False, "segB_long_only")]:
        ALL_RESULTS.append(
            run_arm(
                label=label,
                allow_shorts=allow_shorts,
                start=dt("2026-07-05"),
                end=dt("2026-07-12"),
                initial_balance=84.40,
            )
        )

    # ---- 2. Supplementary folds F1/F2/F3 (original study Sec 7.2) ----
    folds = [
        ("F1_2023H1", "2023-01-01", "2023-06-30"),
        ("F2_2024H1", "2024-01-01", "2024-06-30"),
        ("F3_2025H1", "2025-01-01", "2025-06-30"),
    ]
    for fold_name, start_s, end_s in folds:
        for allow_shorts, arm_label in [(True, "shorts_enabled"), (False, "long_only")]:
            ALL_RESULTS.append(
                run_arm(
                    label=f"{fold_name}_{arm_label}",
                    allow_shorts=allow_shorts,
                    start=dt(start_s),
                    end=dt(end_s),
                    initial_balance=10_000.0,
                )
            )

    # ---- 3. D-2026-08-13-06 window (12-month, most recent, explicit 0.20 cap) ----
    d0813_kwargs = {"base_risk_per_trade": 0.02, "max_risk_per_trade": 0.03}
    for allow_shorts, label in [
        (False, "d0813_window_long_only_cap20"),
        (True, "d0813_window_shorts_enabled_cap20"),
    ]:
        ALL_RESULTS.append(
            run_arm(
                label=label,
                allow_shorts=allow_shorts,
                start=dt("2025-07-04"),
                end=dt("2026-07-04"),
                initial_balance=85.0,
                max_position_size_override=0.20,
                risk_kwargs=d0813_kwargs,
            )
        )

    print("\n\n" + "=" * 70)
    print("ALL RESULTS (JSON)")
    print("=" * 70)
    print(json.dumps(ALL_RESULTS, indent=2, default=str))

    with open("rerun_results.json", "w") as f:
        json.dump(
            {
                "provenance": {
                    "src_root": _SRC_ROOT,
                    "git_head": _GIT_HEAD,
                    "git_branch": _GIT_BRANCH,
                    "pythonpath": _PYTHONPATH,
                    "run_at_utc": datetime.now(UTC).isoformat(),
                },
                "results": ALL_RESULTS,
            },
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
