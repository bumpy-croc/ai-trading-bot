#!/usr/bin/env python3
"""EXIT-GEOMETRY experiment driver (docs/research/experiments/2026-07-12_exit-geometry-honest.md).

Runs HyperGrowth (ETHUSDT/1h, live prod defaults as control) through a small,
pre-registered set of exit/trade-management variants across fixed exam
folds, using the checked-in ``src.experiments`` framework so no strategy or
engine source is modified. Only ``stop_loss_pct``, ``take_profit_pct``, and
``time_exits.max_holding_hours`` are varied -- the only exit-geometry knobs
confirmed overridable for hyper_growth without a src/ change (see prereg
Sec. 3). Trailing-stop distance/activation, breakeven-move threshold, and
partial-exit ladder targets/sizes are hardcoded inside
``create_hyper_growth_strategy``'s ``set_risk_overrides`` call and are NOT
exposed as factory kwargs or runner-override attributes -- confirmed via
direct reading of hyper_growth.py, src/experiments/runner.py's
``_apply_strategy_attribute`` component-target map, and
src/engines/shared/risk_configuration.py's ``build_trailing_stop_policy``
(strategy-declared cfg wins over RiskParameters fallback whenever cfg is
present, which it always is for hyper_growth). Those variants are SKIPPED
per instruction, not faked by patching src/.

Subclasses ExperimentRunner only to capture the full raw results dict
(profit_factor, mfe/mae-bearing Trade objects) that ExperimentResult
narrows away -- no src/ files are modified.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb.Strategy.HyperGrowth",
    "atb.src.engines.backtest",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.src.data_providers",
    "MLBasicSignalGenerator",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

from src.engines.backtest.engine import Backtester  # noqa: E402
from src.experiments.runner import ExperimentRunner  # noqa: E402
from src.experiments.schemas import ExperimentConfig, ParameterSet  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402

SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
INITIAL_BALANCE = 85.0  # matches live prod balance scale (see 2026-07-04 exit-geometry precedent)

FOLDS: dict[str, tuple[datetime, datetime]] = {
    "F1_2023H1": (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    "F2_2024H1": (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC)),
    "F3_2025H1": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
    "F4_2026H1_CONFIRMATORY": (datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 30, tzinfo=UTC)),
}

# Pre-registered arms. Every arm changes ONLY exit/trade-management knobs;
# entries, model, and sizing are identical to control on every arm.
# Weighting (per Lane D's 2026-07-12 live-trade-review mid-flight evidence,
# incorporated before this prereg was locked/committed): stop-side arms
# outnumber TP-side arms 4:2, because the live-fill autopsy found winners
# already capture ~72% of MFE (TP/trailing is not the main bleed) while
# losers ride ~91% of MAE to the full 10% stop (the stop side is).
ARMS: dict[str, dict[str, Any]] = {
    "control": {},
    "sl_08": {"hyper_growth.stop_loss_pct": 0.08},
    "sl_06": {"hyper_growth.stop_loss_pct": 0.06},
    "sl_04": {"hyper_growth.stop_loss_pct": 0.04},
    "tp_06": {"hyper_growth.take_profit_pct": 0.06},
    "combo_sl06_tp15": {"hyper_growth.stop_loss_pct": 0.06, "hyper_growth.take_profit_pct": 0.15},
}
# maxhold_18 is expressed via risk_parameters (time_exits), not `parameters`
# overrides -- hyper_growth's set_risk_overrides dict has no "time_exits" key,
# so RiskParameters.time_exits is honored as a fallback (confirmed by reading
# build_time_exit_policy in src/engines/shared/risk_configuration.py). This
# is the closest EXPRESSIBLE proxy to Lane D's "early-cut if MFE < ~1.5%
# within 12-18h" hypothesis: a plain unconditional time cutoff at 18h, not a
# true MFE-conditioned rule (that would need new ExitHandler logic -- a src/
# change -- so it is explicitly NOT built here; see prereg Sec. 3/8).
ARM_RISK_PARAMETERS: dict[str, dict[str, Any]] = {
    "maxhold_18": {"time_exits": {"max_holding_hours": 18}},
}
ALL_ARM_NAMES = list(ARMS.keys()) + list(ARM_RISK_PARAMETERS.keys())


class ExitGeometryRunner(ExperimentRunner):
    """Runs a config and returns the FULL raw backtester.run() dict plus trades.

    ExperimentResult (the base class's return type) narrows away
    ``profit_factor`` and per-trade MFE/MAE, both of which this experiment
    needs. Reuses the base class's private overrides/validation machinery
    rather than duplicating it -- this is a driver script, not a src/ change.
    """

    def run_full(self, config: ExperimentConfig) -> dict[str, Any]:
        strategy = self._load_strategy(config.strategy_name, factory_kwargs=config.factory_kwargs or None)
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
        # Capture-ratio mechanism metric needs raw Trade objects (mfe/mae),
        # not just the aggregate dict.
        trades = list(getattr(backtester, "trades", []) or [])
        return {
            "results": results,
            "trades": [
                {
                    "pnl_percent": getattr(t, "pnl_percent", None),
                    "mfe": getattr(t, "mfe", None),
                    "mae": getattr(t, "mae", None),
                    "side": getattr(getattr(t, "side", None), "value", str(getattr(t, "side", None))),
                    "exit_reason": getattr(t, "exit_reason", None),
                }
                for t in trades
            ],
        }


def build_config(
    arm_name: str, start: datetime, end: datetime, random_seed: int | None = None
) -> ExperimentConfig:
    # BUG FIX (GH #997/#998, found during round 2): ExperimentRunner._load_strategy
    # builds the strategy via builder(**factory_kwargs) and does NOT thread
    # config.symbol in on its own (fixed at the runner level by PR #1004,
    # merged separately) -- without an explicit factory_kwargs["symbol"],
    # create_hyper_growth_strategy defaults symbol=None, so
    # MLBasicSignalGenerator falls back to DEFAULT_SYMBOL="BTCUSDT" and every
    # run in this script actually scored ETHUSDT candles with BTCUSDT's
    # model. Round 1's original numbers (docs/research/experiments/
    # 2026-07-12_exit-geometry-honest.md) predate this fix and are addended,
    # not silently corrected in place (see that doc's RESULTS addendum).
    # Fixed here to match round 2's exit_geometry_round2_sweep.py::build_config,
    # which threaded this correctly from the start.
    overrides = ARMS.get(arm_name)
    risk_parameters = ARM_RISK_PARAMETERS.get(arm_name, {})
    parameters = ParameterSet(name=arm_name, values=overrides) if overrides else None
    return ExperimentConfig(
        strategy_name="hyper_growth",
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        initial_balance=INITIAL_BALANCE,
        parameters=parameters,
        risk_parameters=risk_parameters,
        factory_kwargs={"symbol": SYMBOL},
        use_cache=True,
        provider="binance",
        random_seed=random_seed,
    )


def profit_factor_from_result(result: dict[str, Any]) -> float:
    return float(result.get("profit_factor", 0.0))


def capture_ratio(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Winner MFE-capture and loser MAE-ride fractions (the mechanism metrics).

    Capture ratio (winners): mean realized move / mean MFE -- matches the
    2026-07-04 precedent method. How much of the peak favorable move a
    winning trade kept before exiting (trailing/TP/partial).

    MAE-ride fraction (losers): mean |realized loss| / mean |MAE| -- added
    per Lane D's 2026-07-12 live-trade-review finding that losers ride ~91%
    of MAE to the stop. 1.0 == exits essentially at the worst point reached
    (full-stop behavior); well below 1.0 == exits recover before the worst
    point (e.g. a tighter stop or time exit cutting the trade off earlier).
    """
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


def run_one(runner: ExitGeometryRunner, arm_name: str, fold_name: str, start: datetime, end: datetime, seed=None):
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
        "annualized_return": r.get("annualized_return"),
        "max_drawdown": r.get("max_drawdown"),
        "sharpe_ratio": r.get("sharpe_ratio"),
        "profit_factor": profit_factor_from_result(r),
        "final_balance": r.get("final_balance"),
        "trade_pnl_pcts": r.get("trade_pnl_pcts"),
        "capture": capture_ratio(trades),
        "runtime_s": round(dt, 1),
    }
    print(
        f"[{fold_name}] {arm_name:<18} trades={record['total_trades']:>4} "
        f"ret%={record['total_return']:>8.2f} PF={record['profit_factor']:>6.3f} "
        f"maxDD%={record['max_drawdown']:>6.2f} winR%={record['win_rate']:>6.2f} "
        f"capture={record['capture']['capture_ratio']} mae_ride={record['capture']['mae_ride_fraction']} "
        f"[{dt:.0f}s]"
    )
    return record


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=["F1_2023H1", "F2_2024H1", "F3_2025H1"])
    parser.add_argument("--arms", nargs="+", default=ALL_ARM_NAMES)
    parser.add_argument("--out", default="experiments/exit_geometry_results.jsonl")
    parser.add_argument("--determinism-check", action="store_true")
    args = parser.parse_args()

    runner = ExitGeometryRunner()
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
