"""Supplement: re-run the arms whose current-default 20% early-stop (from
#1073's ratified RiskParameters hydration, merged AFTER the original 07-12
and 08-13 studies) truncated the backtest relative to the un-capped
originals. max_drawdown=1.0 explicitly disables the early-stop so these
match the originals' effective configuration -- apples-to-apples.
"""
from __future__ import annotations
from run_counterfactual_rerun import run_arm, dt, print_provenance_banner, ALL_RESULTS
import json

print_provenance_banner()

UNCAPPED = {"base_risk_per_trade": 0.02, "max_risk_per_trade": 0.03, "max_drawdown": 1.0}

results = []
# F3 fold, shorts-enabled, uncapped (matches original methodology: no
# --max-drawdown flag passed => original's pre-#1073 default had no
# effective early-stop at 20%)
results.append(run_arm(
    label="F3_2025H1_shorts_enabled_UNCAPPED",
    allow_shorts=True,
    start=dt("2025-01-01"), end=dt("2025-06-30"),
    initial_balance=10_000.0,
    risk_kwargs={"max_drawdown": 1.0},
))

# D-2026-08-13-06 window, both arms, uncapped (matches tier-restore doc's
# methodology exactly: --max-position-size 0.20 explicit, no --max-drawdown)
for allow_shorts, label in [
    (False, "d0813_window_long_only_UNCAPPED"),
    (True, "d0813_window_shorts_enabled_UNCAPPED"),
]:
    results.append(run_arm(
        label=label, allow_shorts=allow_shorts,
        start=dt("2025-07-04"), end=dt("2026-07-04"),
        initial_balance=85.0,
        max_position_size_override=0.20,
        risk_kwargs=UNCAPPED,
    ))

with open("rerun_results_uncapped.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\n\nUNCAPPED SUPPLEMENT RESULTS:")
print(json.dumps(results, indent=2, default=str))
