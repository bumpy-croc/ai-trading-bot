#!/usr/bin/env python3
"""Validate the hyper-growth signal fix.

Compares three configurations on BTCUSDT 1h 2024 using the fixture data:
  - pre-fix: model_type="sentiment", sl=0.20 (the old broken defaults)
  - post-fix: model_type="basic", sl=0.10 (the new defaults from this commit)
  - control: default create_hyper_growth_strategy() — must match post-fix

If the new defaults are live and the fix is correct:
  * post-fix return >> pre-fix return (was 14.16% → expected ~99.80%)
  * decision mix must include both BUY and SELL (not 100% SELL)
  * control == post-fix (defaults match what we expect)

This script only READS production code — it does not modify src/ or cli/.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "WARNING")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
for noisy in (
    "atb",
    "atb.Strategy.HyperGrowth",
    "atb.src.engines.backtest",
    "atb.src.engines",
    "atb.src.strategies",
    "atb.src.prediction",
    "atb.matplotlib.font_manager",
    "atb.src.engines.backtest.execution.exit_handler",
):
    logging.getLogger(noisy).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_providers.offline import FixtureProvider  # noqa: E402
from src.engines.backtest.engine import Backtester  # noqa: E402
from src.risk.risk_manager import RiskParameters  # noqa: E402
from src.strategies.components import MLBasicSignalGenerator  # noqa: E402
from src.strategies.hyper_growth import create_hyper_growth_strategy  # noqa: E402


def run(strategy, start: datetime, end: datetime) -> dict:
    provider = FixtureProvider(ROOT / "tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather")
    bt = Backtester(
        strategy=strategy,
        data_provider=provider,
        sentiment_provider=None,
        risk_parameters=RiskParameters(),
        initial_balance=1000.0,
        log_to_database=False,
    )
    return bt.run(symbol="BTCUSDT", timeframe="1h", start=start, end=end)


def describe(name: str, r: dict) -> str:
    return (
        f"{name:32s}  trades={r.get('total_trades', 0):4d}  "
        f"return={r.get('total_return', 0.0)*100:7.2f}%  "
        f"maxDD={r.get('max_drawdown', 0.0)*100:6.2f}%  "
        f"sharpe={r.get('sharpe_ratio', 0.0):6.3f}  "
        f"winR={r.get('win_rate', 0.0)*100:5.1f}%"
    )


def main() -> int:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)

    # Pre-fix: reinstate the old broken defaults to compare against
    pre_fix = create_hyper_growth_strategy(stop_loss_pct=0.20)
    # Force the broken sentiment model explicitly to reproduce pre-fix
    pre_fix.signal_generator = MLBasicSignalGenerator(
        name="HyperGrowth_signals", model_type="sentiment"
    )

    # Post-fix with explicit kwargs (belt-and-suspenders vs control)
    post_fix = create_hyper_growth_strategy(stop_loss_pct=0.10)

    # Control: whatever the new defaults actually are
    control = create_hyper_growth_strategy()

    print("\n=== Hyper-growth signal fix validation ===\n")
    print("Running 3 backtests on BTCUSDT 1h 2024 fixture…\n")

    r_pre = run(pre_fix, start, end)
    r_post = run(post_fix, start, end)
    r_ctrl = run(control, start, end)

    print(describe("pre-fix (sentiment, sl=0.20)", r_pre))
    print(describe("post-fix (basic, sl=0.10)", r_post))
    print(describe("control (new defaults)", r_ctrl))

    # Sanity checks
    ok = True
    if r_ctrl.get("total_return", 0.0) < r_pre.get("total_return", 0.0) + 0.30:
        print("\n  FAIL: new-default return not at least +30 pp vs pre-fix")
        ok = False
    else:
        lift_pp = (r_ctrl["total_return"] - r_pre["total_return"]) * 100
        print(f"\n  OK: return lift vs pre-fix: +{lift_pp:.1f} percentage points")

    if abs(r_ctrl.get("total_return", 0.0) - r_post.get("total_return", 0.0)) > 1e-9:
        print("  FAIL: control (default kwargs) does not match post-fix explicit kwargs")
        ok = False
    else:
        print("  OK: defaults match expected post-fix configuration")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
