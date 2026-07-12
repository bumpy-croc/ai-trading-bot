#!/usr/bin/env python3
"""EXIT-GEOMETRY ROUND 2 driver (docs/research/experiments/2026-07-12_exit-geometry-round2.md).

Tests the arms round 1 (docs/research/experiments/2026-07-12_exit-geometry-honest.md,
#970/#971) flagged as inexpressible without a src/ change: MFE-conditioned
early-cut (new EarlyCutPolicy), and trailing-stop/breakeven ablation --
now real create_hyper_growth_strategy factory kwargs via PR #976. Also
reruns tp_06 (round 1's only directionally-positive arm) with 2 additional
folds for statistical power.

Reuses round 1's ExitGeometryRunner pattern (subclasses ExperimentRunner to
capture the full raw results dict + Trade objects) and its capture_ratio
mechanism-metric helper verbatim -- no src/ file is modified by this script.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# CRITICAL: running this file as `python3 experiments/foo.py` (a script PATH,
# not `python3 -c`/a module) sets sys.path[0] to this file's own directory
# (experiments/), NOT the current working directory -- so a bare `import src`
# falls through past this worktree's own repo root and resolves against
# whatever editable "atb" install is on site-packages, which points at the
# MAIN CHECKOUT (/Users/alex/Sites/ai-trading-bot, branch `main`, months
# behind `develop`). Verified directly: `python3 experiments/pathcheck.py`
# from this worktree resolved `src.strategies.hyper_growth.__file__` to the
# main checkout's copy, not this worktree's. Forcing this worktree's repo
# root to the FRONT of sys.path makes every `src.*` import resolve locally,
# regardless of invocation style. Without this, EVERY experiments/*.py driver
# invoked as a script path silently tests main's stale code, not the
# worktree's -- a serious, previously-undetected harness bug (see prereg
# Sec. 5 addendum; filed as a GH issue).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "Strategy.HyperGrowth",
    "atb.Strategy.HyperGrowth",
    "src.engines.backtest",
    "src.engines",
    "src.strategies",
    "src.strategies.components.ml_signal_generator",
    "src.prediction",
    "src.data_providers",
    "atb.src.engines.backtest",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.src.data_providers",
    "MLBasicSignalGenerator",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)

from src.engines.backtest.engine import Backtester  # noqa: E402
from src.experiments.runner import ExperimentRunner  # noqa: E402
from src.experiments.schemas import ExperimentConfig  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402

SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
INITIAL_BALANCE = 85.0  # matches live prod balance scale (round 1 precedent)

FOLDS: dict[str, tuple[datetime, datetime]] = {
    "F1_2023H1": (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    "F2_2024H1": (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC)),
    "F3_2025H1": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    "F0a_2021H1": (datetime(2021, 1, 3, tzinfo=UTC), datetime(2021, 6, 30, tzinfo=UTC)),
    "F0b_2022H1": (datetime(2022, 1, 3, tzinfo=UTC), datetime(2022, 6, 30, tzinfo=UTC)),
}
PRIMARY_FOLDS = ["F1_2023H1", "F2_2024H1", "F3_2025H1"]
EXTENSION_FOLDS = ["F0a_2021H1", "F0b_2022H1"]

# Pre-registered arms (Sec. 7 of the prereg). Every arm changes ONLY
# exit/trade-management factory kwargs; entries, model, sizing identical to
# control. `trailing_atr_multiplier=None` is applied globally via
# risk_parameters (Sec. 4's gotcha fix) -- provably inert for every arm
# except breakeven_only, where it is required to get a true "no
# distance-based trailing" policy instead of an accidental ATR-based one.
ARM_FACTORY_KWARGS: dict[str, dict[str, Any]] = {
    "control": {},
    "early_cut_1p5_12h": {
        "early_cut_mfe_threshold_pct": 0.015,
        "early_cut_evaluation_window_hours": 12,
    },
    "early_cut_1p5_18h": {
        "early_cut_mfe_threshold_pct": 0.015,
        "early_cut_evaluation_window_hours": 18,
    },
    "early_cut_1p0_24h": {
        "early_cut_mfe_threshold_pct": 0.01,
        "early_cut_evaluation_window_hours": 24,
    },
    "trailing_only": {"breakeven_threshold": None},
    "breakeven_only": {"trailing_distance_pct": None},
    "tp_06_rerun": {"take_profit_pct": 0.06},
}
ALL_ARM_NAMES = list(ARM_FACTORY_KWARGS.keys())
GLOBAL_RISK_PARAMETERS: dict[str, Any] = {"trailing_atr_multiplier": None}


class ExitGeometryRound2Runner(ExperimentRunner):
    """Runs a config and returns the FULL raw backtester.run() dict plus trades.

    Identical pattern to round 1's ExitGeometryRunner: ExperimentResult (the
    base class's return type) narrows away profit_factor and per-trade
    mfe/mae/metadata, all of which this experiment needs.
    """

    def run_full(self, config: ExperimentConfig) -> dict[str, Any]:
        strategy = self._load_strategy(
            config.strategy_name, factory_kwargs=config.factory_kwargs or None
        )
        self._apply_parameter_overrides(strategy, config)
        self._validate_post_override_invariants(strategy)
        provider = self._load_provider(
            config.provider,
            config.use_cache,
            start=config.start,
            end=config.end,
            timeframe=config.timeframe,
            seed=config.random_seed,
        )
        risk_params = (
            RiskParameters(**config.risk_parameters) if config.risk_parameters else RiskParameters()
        )
        backtester = Backtester(
            strategy=strategy,
            data_provider=provider,
            sentiment_provider=None,
            risk_parameters=risk_params,
            initial_balance=config.initial_balance,
            log_to_database=False,
        )
        results = backtester.run(
            symbol=config.symbol, timeframe=config.timeframe, start=config.start, end=config.end
        )
        trades = list(getattr(backtester, "trades", []) or [])
        return {
            "results": results,
            "trades": [
                {
                    "entry_time": getattr(t, "entry_time", None),
                    "exit_time": getattr(t, "exit_time", None),
                    "pnl_percent": getattr(t, "pnl_percent", None),
                    "mfe": getattr(t, "mfe", None),
                    "mae": getattr(t, "mae", None),
                    "side": getattr(getattr(t, "side", None), "value", str(getattr(t, "side", None))),
                    "exit_reason": getattr(t, "exit_reason", None),
                    "early_cut_window_mfe_pct": (getattr(t, "metadata", None) or {}).get(
                        "early_cut_window_mfe_pct"
                    ),
                }
                for t in trades
            ],
        }


def build_config(
    arm_name: str, start: datetime, end: datetime, random_seed: int | None = None
) -> ExperimentConfig:
    # CRITICAL: ExperimentRunner._load_strategy does NOT thread config.symbol
    # into the strategy factory (unlike src/engines/live/runner.py and
    # cli/commands/backtest.py, which both explicitly thread symbol for this
    # exact reason -- "the symbol must reach ML signal generators so model
    # registry selection matches"). Without this, create_hyper_growth_strategy
    # defaults symbol=None -> MLBasicSignalGenerator.DEFAULT_SYMBOL="BTCUSDT",
    # silently scoring ETHUSDT candles with BTCUSDT's model. Verified directly
    # against develop@3721a835 before locking this script (see prereg Sec. 5
    # addendum) -- round 1's checked-in exit_geometry_sweep.py has this same
    # gap. Filed as a harness bug (GH issue, see prereg RESULTS).
    factory_kwargs = {"symbol": SYMBOL, **ARM_FACTORY_KWARGS.get(arm_name, {})}
    return ExperimentConfig(
        strategy_name="hyper_growth",
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        initial_balance=INITIAL_BALANCE,
        factory_kwargs=factory_kwargs,
        risk_parameters=dict(GLOBAL_RISK_PARAMETERS),
        use_cache=True,
        provider="binance",
        random_seed=random_seed,
    )


def profit_factor_from_result(result: dict[str, Any]) -> float:
    return float(result.get("profit_factor", 0.0))


def capture_ratio(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Winner MFE-capture and loser MAE-ride fractions (round 1 method, verbatim)."""
    winners = [t for t in trades if (t.get("pnl_percent") or 0) > 0 and t.get("mfe") is not None]
    losers = [t for t in trades if (t.get("pnl_percent") or 0) < 0 and t.get("mae") is not None]

    out: dict[str, Any] = {
        "n_winners_with_mfe": len(winners),
        "mean_realized_win_pct": None,
        "mean_mfe_pct": None,
        "capture_ratio": None,
        "n_losers_with_mae": len(losers),
        "mean_realized_loss_pct": None,
        "mean_mae_pct": None,
        "mae_ride_fraction": None,
    }
    if winners:
        mean_realized = sum(t["pnl_percent"] for t in winners) / len(winners)
        mean_mfe = sum(t["mfe"] for t in winners) / len(winners)
        out["mean_realized_win_pct"] = mean_realized
        out["mean_mfe_pct"] = mean_mfe
        out["capture_ratio"] = (mean_realized / mean_mfe) if mean_mfe else None
    if losers:
        mean_realized_loss = sum(abs(t["pnl_percent"]) for t in losers) / len(losers)
        mean_mae = sum(abs(t["mae"]) for t in losers) / len(losers)
        out["mean_realized_loss_pct"] = mean_realized_loss
        out["mean_mae_pct"] = mean_mae
        out["mae_ride_fraction"] = (mean_realized_loss / mean_mae) if mean_mae else None
    return out


def early_cut_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Cut rate and window-MFE distribution for early-cut arms (Sec. 8)."""
    cut_trades = [t for t in trades if (t.get("exit_reason") or "").startswith("Early cut")]
    n_total = len(trades)
    mfes = [t["early_cut_window_mfe_pct"] for t in cut_trades if t.get("early_cut_window_mfe_pct") is not None]
    out: dict[str, Any] = {
        "n_cut": len(cut_trades),
        "cut_rate": (len(cut_trades) / n_total) if n_total else None,
        "mean_window_mfe_pct_at_cut": (sum(mfes) / len(mfes)) if mfes else None,
    }
    return out


def cut_precision(cut_trades: list[dict[str, Any]], control_trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Matched-entry cut-precision proxy (Sec. 8, pre-committed before locking).

    Matches each cut trade to a control trade in the SAME fold with an
    identical entry_time. Reports the fraction of MATCHED cut trades whose
    control counterpart eventually closed as a loser, plus the match rate
    (coverage), so the metric's completeness is disclosed, not assumed.
    """
    control_by_entry = {t["entry_time"]: t for t in control_trades if t.get("entry_time") is not None}
    matched = []
    for t in cut_trades:
        et = t.get("entry_time")
        if et is not None and et in control_by_entry:
            matched.append(control_by_entry[et])
    n_cut = len(cut_trades)
    n_matched = len(matched)
    n_would_be_losers = sum(1 for m in matched if (m.get("pnl_percent") or 0) < 0)
    return {
        "n_cut": n_cut,
        "n_matched": n_matched,
        "match_rate": (n_matched / n_cut) if n_cut else None,
        "n_matched_would_be_losers": n_would_be_losers,
        "cut_precision": (n_would_be_losers / n_matched) if n_matched else None,
    }


def run_one(
    runner: ExitGeometryRound2Runner,
    arm_name: str,
    fold_name: str,
    start: datetime,
    end: datetime,
    seed=None,
):
    cfg = build_config(arm_name, start, end, random_seed=seed)
    t0 = time.time()
    payload = runner.run_full(cfg)
    dt = time.time() - t0
    r = payload["results"]
    trades = payload["trades"]
    record = {
        "arm": arm_name,
        "fold": fold_name,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "total_trades": r.get("total_trades"),
        "win_rate": r.get("win_rate"),
        "total_return": r.get("total_return"),
        "max_drawdown": r.get("max_drawdown"),
        "sharpe_ratio": r.get("sharpe_ratio"),
        "profit_factor": profit_factor_from_result(r),
        "final_balance": r.get("final_balance"),
        "trade_pnl_pcts": r.get("trade_pnl_pcts"),
        "capture": capture_ratio(trades),
        "early_cut": early_cut_metrics(trades) if arm_name.startswith("early_cut") else None,
        "trades_raw": [
            {
                "entry_time": str(t.get("entry_time")),
                "pnl_percent": t.get("pnl_percent"),
                "exit_reason": t.get("exit_reason"),
                "early_cut_window_mfe_pct": t.get("early_cut_window_mfe_pct"),
            }
            for t in trades
        ],
        "runtime_s": round(dt, 1),
    }
    print(
        f"[{fold_name}] {arm_name:<20} trades={record['total_trades']:>4} "
        f"ret%={record['total_return']:>8.2f} PF={record['profit_factor']:>6.3f} "
        f"maxDD%={record['max_drawdown']:>6.2f} winR%={record['win_rate']:>6.2f} "
        f"capture={record['capture']['capture_ratio']} mae_ride={record['capture']['mae_ride_fraction']} "
        f"[{dt:.0f}s]"
    )
    return record


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=list(FOLDS.keys()))
    parser.add_argument("--arms", nargs="+", default=ALL_ARM_NAMES)
    parser.add_argument("--out", default="experiments/exit_geometry_round2_results.jsonl")
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    runner = ExitGeometryRound2Runner()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for fold_name in args.folds:
        start, end = FOLDS[fold_name]
        for arm_name in args.arms:
            rec = run_one(runner, arm_name, fold_name, start, end)
            records.append(rec)

    if args.determinism_check:
        start, end = FOLDS["F1_2023H1"]
        rec_a = run_one(runner, "control", "F1_2023H1_DETERMINISM_A", start, end)
        rec_b = run_one(runner, "control", "F1_2023H1_DETERMINISM_B", start, end)
        match = (
            rec_a["total_trades"] == rec_b["total_trades"]
            and rec_a["total_return"] == rec_b["total_return"]
            and rec_a["profit_factor"] == rec_b["profit_factor"]
        )
        print(f"DETERMINISM CHECK: {'PASS' if match else 'FAIL'}")
        records.append({"determinism_check": "PASS" if match else "FAIL", "a": rec_a, "b": rec_b})

    with open(out_path, "a") as f:
        for rec in records:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\nWrote {len(records)} records to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
