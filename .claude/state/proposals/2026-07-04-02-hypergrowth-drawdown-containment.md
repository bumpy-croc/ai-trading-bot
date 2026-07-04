---
id: 2026-07-04-02-hypergrowth-drawdown-containment
from: risk-officer
to: pm
status: open
risk_review_required: true
risk_verdict: approve   # authored by risk-officer; code changes still require independent code review before merge
code_review_required: true
board_required: false   # entry-pause + parameter tightening are live-capital changes within the charter autonomy envelope; NOT a kill-switch action
created: 2026-07-04T13:05:00Z
updated: 2026-07-04T13:05:00Z
---

## Ask

Contain HyperGrowth/ETHUSDT's drawdown exposure **now**, in four ordered steps, following the confirmed hard-cap breach analysis in `docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md` and incident `2026-07-04T1300-P1-hypergrowth-drawdown-cap-breach`.

## Context (one paragraph)

Live equity is $83.92 vs a true peak of $103.82 (19.18% DD; the 20% line is $83.06 — less than one stop-out away, and it was already crossed once on 2026-06-06 at 20.33%, undetected). The strategy's honest full-year backtest reproduces the breach (-20.15% / 21.84% MaxDD, verified twice). All four control layers that should react are broken or miscalibrated: live hard-cap halt is dead code (#749), backtest hard-cap default is 50%, HyperGrowth loosens the graduated breakers ~3x with the second tier past the kill line, and live's drawdown input resets to ~0 on every deploy restart.

## Proposed change (ordered)

1. **Ops, immediate, no code**: set `FEATURE_ENTRY_PAUSE=true` on the prod HyperGrowth service **now** (preferred; the $0.86 margin to the line is smaller than the open position's ~$1.33 stop-out) — or, minimum, on the first close below $83.06. Exits/SL/reconciliation continue; the open SHORT keeps its stop. Lift the pause only on explicit pm decision with equity recovered above the 15%-DD line ($88.25) or on human instruction. Recalibrate the daily-standup tripwires to the true $103.82 peak (currently derived from the post-reset $84.40).
2. **Code, P1 (owner: existing `fix/live-max-drawdown-halt` branch, currently zero commits)**: wire a live max-drawdown halt per #749 — halt new entries + alert at `RiskParameters.max_drawdown` (0.20), with peak sourced from a **persistent** store (all-time/rolling peak over `account_history`), not the in-process tracker, so restarts can't amnesia it. CF-B evidence: even in backtest the per-candle check overshoots (halted at 20.50%), so treat 20% as a *detection* line, not a guarantee.
3. **Code, P1-P2, one block**: remove HyperGrowth's `dynamic_risk` threshold loosening (`src/strategies/hyper_growth.py:294-300`) so both engines inherit `risk-limits.json`'s `[0.05, 0.10, 0.15]` / `[0.8, 0.6, 0.4]`. CF-A (this review, same 365d window): MaxDD 21.84% → **17.01% (no breach)**, return -20.15% → **-16.08%** ($3.70 of capital saved on the bad year; protection was free on this path).
4. **Config, P2**: change backtest CLI `--max-drawdown` default 0.5 → `DEFAULT_MAX_DRAWDOWN` (0.20) so every future backtest enforces the book's hard line by default (`cli/commands/backtest.py:326-330`).

## Evidence

- Experiment doc: `docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md` (reproduction table, live equity scan, four-layer analysis, counterfactuals).
- Incident: `.claude/state/incidents/2026-07-04T1300-P1-hypergrowth-drawdown-cap-breach.md`.
- Prior signals now corroborated: #749 (2026-06-10), observability audit 2026-06-08, #807 (account-level circuit breakers).

## How this could lose money

1. **Entry pause = opportunity cost, not capital risk.** While paused, the strategy can't enter; if ETH trends favorably, gains are foregone. Given negative full-year expectancy (PF 0.47) at current calibration, the expected cost of pausing is low; the expected cost of NOT pausing is a hard-cap re-breach on the next ordinary stop-out.
2. **Tighter breakers cut position size during drawdowns; if the strategy's edge is real and mean-reverting, recovery is slower.** CF-A quantifies the trade on the observed bad year. If pm believes in HyperGrowth's 5-year story, note that story predates #838 and its partial-exit returns are known-fabricated (#839) — the honest evidence for "edge that would be throttled" is currently thin.
3. **A persistent-peak halt can pin the strategy paused after any deep drawdown** until a human reviews — that is the designed behavior of a hard line per charter ("halt new entries and page human"; existing stops keep running).

## What pm should decide

- Execute step 1 now vs on-touch of $83.06 (risk-officer recommends **now**).
- Sequencing/ownership of steps 2-4 (step 2's branch exists but is empty; consider re-dispatching with the persistent-peak requirement added).
- Whether the strategic question — HyperGrowth fails charter KPIs on honest full-year evidence (Sharpe 0.12 vs 0.5 min) — goes to the bear-market-2026 workstream (#801-#807) or gets its own review.

## Reviews

### risk-officer

**Verdict**: approve (author). Steps are graduated, reversible (pause is a flag; threshold change is one dict; halt blocks entries without liquidating), and none touches exchange orders directly. The only irreversible path here is *inaction* followed by another leg down.
