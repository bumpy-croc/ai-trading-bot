---
name: trade-review
description: Periodic autopsy of live trades — MFE/MAE capture ratios, exit-reason P&L decomposition, live-vs-backtest divergence spot checks, regime attribution. Use on a weekly/biweekly cadence or after any notable win/loss streak; output is hypotheses that feed experiment-preregister, never direct strategy changes.
---

# Trade Review

Where live P&L evidence turns into research hypotheses. Read-only against prod; output goes to
layer 2 (a dated review note + hypotheses handed to `experiment-preregister`). This skill never
changes a parameter — the 2026-07-04 exit-geometry sweep exists because a trade review asked
"are we exiting too early?" and the preregistered answer was NO (every tighter variant strictly
worse; the problem was the signal, #867). Hunches go through the pipeline.

## Data access

Read-only prod psql per `prod-forensics` (public proxy URL, `SET default_transaction_read_only
= on;` first). Tables: `trades` (exit_reason, pnl; commission/quantity only post-#731),
`positions`, `strategy_executions` (per-decision signal/confidence trail), `account_balances`
(ledger truth), `account_history` (hourly equity). Candle data from the local parquet cache
(`atb data prefill-cache`) for MFE/MAE reconstruction.

## The four passes

**1. MFE/MAE capture per trade.** For each closed trade, reconstruct from cached 1h candles:
max favorable excursion, max adverse excursion, and capture ratio (realized P&L / MFE).
Reference finding: all 5 winners of the June–July streak exited via trailing stop at +3.1–3.8%
— below the first partial-exit target (8%), which is why live had zero partials ever (log.md
2026-07-03 verification entry). Low capture across many trades = exit-geometry hypothesis;
deep MAE on winners = entry-timing/stop hypothesis. Either way: hypothesis, not tweak.

**2. Exit-reason P&L decomposition.** Group realized P&L by `exit_reason` (trailing stop, SL,
TP, emergency close, external_close_recovery). Emergency-close P&L is an ops signal, not a
strategy signal (the 2026-06-02 cascade). Fees from the ledger, not `trades.commission`
(pre-#731 rows are 0 — see `prod-forensics` for denominations).

**3. Live-vs-backtest divergence spot check.** Replay the review window through the corrected
backtest engine (post-#838 ONLY — every pre-#838 partial-exit backtest return is fabricated)
with prod-matched flags, and compare trade-by-trade: same entries taken? same exits? Charter KPI:
parity variance ≤15%. Known structural divergences to check against before alarming: backtest
force-injected default partial targets ignoring strategy overrides (fixed in #838); forming-bar
decisions — live evaluates against a mutating tail candle, so live entries can fire intra-bar
on decisions a closed-bar backtest never sees (`2026-07-06_forming-bar-fliprate.md`: large flip
rates, mostly direction reversals; churn risk, not hidden edge).

**4. Regime attribution.** Tag each trade with the regime slice (bull/bear/chop by monthly
return — the simple deterministic labels, not the live detector, per
`docs/architecture/model_evaluation_system.md`). A strategy positive in one regime and bleeding
in another is a leverage-map/gating hypothesis. Check conviction too: the #913 forensics
reconstructed position 22's full 30-min strengthening decision trail from
`strategy_executions` — entry conviction (confidence/strength at entry) vs outcome is a real
axis; prod's confidence median has sat at noise level (~0.03–0.04), which reframed a "weak
strategy" as "no signal" (the cross-symbol model P0).

## Sample-size honesty

The live record is TINY (tens of trades). 5 consecutive wins moved the account +$1.11 and was
correctly logged as "a short favorable win-streak sample", not edge. Never conclude expectancy
from <20 trades; flag direction-of-evidence only. The standing capital gate (see
`capital-review`) requires 20+ live trades before scaling — this review is how those trades get
counted and characterized.

## Output

`docs/research/notes/YYYY-MM-DD_trade-review.md`: per-trade table (entry/exit, reason, P&L,
MFE/MAE/capture, regime, entry confidence), the four-pass findings, and an explicit
**Hypotheses** section — each one phrased ready for `experiment-preregister` (falsifiable, with
a candidate metric). Material findings (parity breach, unexplained exit class, tripwire-adjacent
drawdown) → log.md entry via `decision-record`; capital-risk findings → `incident-response`.

## Red flags

- Reviewing with pre-#838 backtest numbers as the comparison baseline.
- A "fix" ships from this review without passing through `experiment-preregister`.
- Trusting `trades` rows alone for money math — the ledger (`account_balances`) is truth.
- Concluding from an exit-reason mix that includes emergency closes without separating ops
  failures from strategy behavior.
