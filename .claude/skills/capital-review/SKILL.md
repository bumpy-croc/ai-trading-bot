---
name: capital-review
description: Monthly Board pack — equity curve vs charter KPIs, full cost accounting, risk posture, live-expectancy evidence, and an explicit scale/hold/reduce recommendation with pre-committed gates. Use on the monthly cadence, when the Board asks "should we add capital?", or before any proposal that changes capital under management.
---

# Capital Review

The monthly sitting where the Board decides scale/hold/reduce on evidence, not narrative. Reads
layers 1+2, produces a pack that is itself layer-2 record (dated file + log entry). The daemon
recommends; capital changes above the charter's autonomy envelope are the human's call.
See `docs/architecture/memory_system.md`.

## 1. Equity truth first

Build the equity curve from `account_history` — but apply the pinned-book-value check before
trusting ANY historical point (`prod-forensics`): true equity reads begin 2026-06-03 (#655);
the drawdown baseline is peak TRUE equity since the last reconciled reset (2026-06-05 /
session 20 ≈ $84.40 — pm policy, log.md 2026-07-04). A review that quotes the phantom $103.82
April "peak" repeats the exact error the 2026-07-04 13:55 correction withdrew. Cross-check the
month's net change against the `account_balances` ledger decomposition (where the money actually
went: P&L vs fees vs sync corrections).

## 2. Charter KPI scorecard (charter.md §KPIs, in priority order)

| KPI | Target (charter v0.1) | This month | Evidence |
|---|---|---|---|
| Capital preservation | no risk-limits.json breach | | ledger + guard logs |
| Backtest/live parity | variance ≤15% | | `trade-review` pass 3 |
| Sharpe (rolling 30d) | target 1.5 / min 0.5 | | equity curve |
| Win rate | target 55% / min 45% | | closed trades (state n!) |
| Max drawdown (rolling) | target <15%, hard 20% | | true-equity baseline |
| Cost per decision | <$0.50 | | see cost accounting |

Every cell carries its artifact (file/query/log ref) or is marked UNVERIFIED — the pm.md
confidence-cap rule applies to Board packs too.

## 3. Full cost accounting

Cost/decision includes EVERYTHING: exchange fees (ledger `entry_fee_*` + `orders.actual_
commission`, mind the received-asset denomination), margin interest, Railway hosting, AWS/
SageMaker training spend (cloud training ≈ $0.10/candidate, $0.37 full retrain — real July
numbers), and inference/agent costs. The charter caps inference spend >$50/24h as a
human-approval item — report actuals against it.

## 4. Risk posture

Current sizing vs limits (as-deployed AND as-written — the risk-limits.json 0.10 vs deployed
0.20 divergence stayed open for weeks; if a divergence exists, this pack escalates it to
`risk-ratification`), guard status (drawdown guard armed at correct peak, tiers), tripwire table
status (standup tripwires: soft $80.18 / reduce $75.96 / hard floor $67.52), open incidents,
event-window calendar (FOMC/CPI pauses armed?).

## 5. Live-expectancy evidence

The live record's n, win rate, avg win/loss, and what it does NOT yet prove (5 wins =
"favorable sample", per the 2026-07-03 log). Latest shared-exam results for the deployed
model/strategy (`docs/research/experiments/`, scoreboard) and staging paper status. Structural
findings that reframe expectancy (e.g. the cross-symbol-model P0: 8 months of prod decisions
were noise-trading — a fact any capital decision that month had to know).

## 6. The recommendation — gated, not vibed

**The standing capital gate:** new capital is added only when ALL hold —
1. the deployed model/strategy is a current shared-exam winner (L2 of
   `docs/architecture/model_evaluation_system.md`), not just incumbent-by-default;
2. ≥48h clean staging paper on the exact deployed configuration (L3a);
3. 3–4 weeks AND 20+ trades of live-positive performance at current size (L3b evidence);
4. no open P0/P1 incident, no unratified risk-limit divergence, no breached tripwire.

**Reduce/halt triggers:** tripwire breach, parity blown past 15% unexplained, a
`kill-switch-drill` FAIL on a delivery/trip path, or the structural-expectancy class of finding
(the −20.15%/365d backtest with a 21.84% MaxDD breach was treated as a reduce-candidate signal
even while live was green). Recommendation is one of SCALE / HOLD / REDUCE with the gate
checklist shown pass/fail. Precedent for honesty: the 2026-07-03 assessment told the Board the
£85→£1,000-in-9-days ask was infeasible (~8.5% success even at zero-edge all-in) rather than
pretending — that candor is the product.

## Record

Pack → `docs/research/notes/YYYY-MM-DD_capital-review.md`; decision + rationale → log.md via
`decision-record` (`[D-…]`); anything needing a charter/risk-limits edit → `risk-ratification`.
