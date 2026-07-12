#!/usr/bin/env python3
"""EV-CONDITIONING (GH #984 part 2) -- extract control-arm trades with
entry/exit timestamps for the entry-observable analysis.

This is a LIGHT, control-arm-only rerun of the exact same
`hyper_growth` config and fold definitions as
`docs/research/experiments/2026-07-12_exit-geometry-honest.md`'s
control arm (no sweep, no arms besides control -- 3 backtest runs total,
not 21). It exists only because the round-1/round-2 exit-geometry result
JSONLs persist fold-level aggregates and (round 2 only) a thin
per-trade `{entry_time, pnl_percent, exit_reason}` slice, but neither
persists `entry_time`/`side` together for round 1, and neither
persists model-confidence/predicted_return per trade (verified by
direct inspection before writing this script -- see the analysis
note's Data section). `BaseTrade` (src/engines/shared/models.py)
already carries `entry_time`, `exit_time`, `side`, `entry_price`,
`exit_price`, `pnl_percent` natively -- no engine change needed to get
those. Predicted-return/confidence/regime/vol are computed
independently in `ev_conditioning_features.py` against the same cached
OHLCV, keyed off the `entry_time` this script emits.

Uses the checked-in `src.experiments.ExperimentRunner` machinery only
-- no strategy or engine source is modified (same discipline as
`exit_geometry_sweep.py`).
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

# CRITICAL (GH #997, independently reconfirmed while building this script):
# running this file as `python3 experiments/foo.py` sets sys.path[0] to this
# file's own directory, not the caller's cwd, so a bare `import src` can
# resolve against whatever editable "atb" install is on site-packages (the
# shared venv's install, which may point at the main checkout's stale src/)
# instead of this worktree's own repo root. Force this worktree's root to the
# front of sys.path so every `src.*` import resolves locally regardless of
# invocation style.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
from src.experiments.schemas import ExperimentConfig  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402

SYMBOL = "ETHUSDT"
TIMEFRAME = "1h"
INITIAL_BALANCE = 85.0  # matches live prod balance scale + the exit-geometry precedent

# Same fold definitions as docs/research/experiments/2026-07-12_exit-geometry-honest.md
# Sec. 4, so this analysis is directly comparable to (not a re-litigation of) that study.
FOLDS: dict[str, tuple[datetime, datetime]] = {
    "F1_2023H1": (datetime(2023, 1, 1, tzinfo=UTC), datetime(2023, 6, 30, tzinfo=UTC)),
    "F2_2024H1": (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 6, 30, tzinfo=UTC)),
    "F3_2025H1": (datetime(2025, 1, 1, tzinfo=UTC), datetime(2025, 6, 30, tzinfo=UTC)),
}


class ControlTradesRunner(ExperimentRunner):
    """Runs the control config and returns per-trade BaseTrade fields.

    Reuses the base class's private overrides/validation machinery --
    a driver script, not a src/ change.
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
                    "symbol": t.symbol,
                    "side": getattr(
                        getattr(t, "side", None), "value", str(getattr(t, "side", None))
                    ),
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                    "entry_price": float(t.entry_price),
                    "exit_price": float(t.exit_price),
                    "size": float(t.size),
                    "pnl": float(t.pnl),
                    "pnl_percent": float(t.pnl_percent) if t.pnl_percent is not None else None,
                    "exit_reason": t.exit_reason,
                    "mfe": float(t.mfe) if t.mfe is not None else None,
                    "mae": float(t.mae) if t.mae is not None else None,
                }
                for t in trades
            ],
        }


def build_config(start: datetime, end: datetime) -> ExperimentConfig:
    return ExperimentConfig(
        strategy_name="hyper_growth",
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start=start,
        end=end,
        initial_balance=INITIAL_BALANCE,
        parameters=None,  # control: no overrides
        risk_parameters={},
        use_cache=True,
        provider="binance",
        random_seed=None,
        # GH #997: ExperimentRunner._load_strategy never threads config.symbol
        # into the strategy factory on its own -- a bare create_hyper_growth_strategy()
        # call resolves signal_generator.symbol to MLBasicSignalGenerator.DEFAULT_SYMBOL
        # ("BTCUSDT"), silently scoring ETHUSDT candles with the wrong model (GH #998
        # found this invalidates round 1's exit-geometry absolute numbers). Must be
        # threaded explicitly here.
        factory_kwargs={"symbol": SYMBOL},
    )


def main() -> int:
    parser_folds = sys.argv[1:] or list(FOLDS.keys())
    runner = ControlTradesRunner()
    out_path = Path("experiments/ev_conditioning_control_trades.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_records = []
    for fold_name in parser_folds:
        start, end = FOLDS[fold_name]
        cfg = build_config(start, end)
        t0 = time.time()
        payload = runner.run_full(cfg)
        dt = time.time() - t0
        r = payload["results"]
        trades = payload["trades"]
        print(
            f"[{fold_name}] control trades={r.get('total_trades')} "
            f"ret%={r.get('total_return'):.2f} PF={r.get('profit_factor')} "
            f"[{dt:.0f}s]"
        )
        for t in trades:
            t["fold"] = fold_name
            all_records.append(t)

    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, default=str) + "\n")
    print(f"\nWrote {len(all_records)} trade records to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
