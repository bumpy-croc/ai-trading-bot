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

> **Revised 2026-07-04 (same day)**: the original context claimed live had already breached the cap (20.33% from a $103.82 peak) and was one stop-out from re-breach. That peak is phantom-era book value (ledger-verified: `account_history.balance` software-pinned ~$100 Mar–May; true reads begin 2026-06-03 at $84.14) — the claim is withdrawn. Under pm's adopted baseline policy (peak = true equity since the last reconciled reset, 2026-06-05 / session 20 ≈ $84.40), current live DD ≈ 0.6%.

The strategy's honest full-year backtest breaches the cap (-20.15% / 21.84% MaxDD, verified twice, structural slow bleed). All four control layers that should contain a repeat are broken or miscalibrated: live hard-cap halt is dead code (#749), backtest hard-cap default is 50%, HyperGrowth loosens the graduated breakers ~3x with the second tier past the kill line, and live's drawdown input resets to ~0 on every deploy restart. There is no urgency-of-the-hour, but the account is running without any functioning automated drawdown control.

## Proposed change (ordered)

1. **Ops, no code** *(revised — no immediate pause)*: treat the standup tripwires from the post-reconciled-reset peak ($84.40: soft $80.18 / reduce $75.96 / hard $67.52 → `FEATURE_ENTRY_PAUSE` + page human) as **binding**, not advisory — they are the only functioning drawdown control in production until step 2 lands. Exits/SL/reconciliation continue through any pause; the open SHORT keeps its stop.
2. **Code, P1 (owner: existing `fix/live-max-drawdown-halt` branch, zero commits at review time)**: wire a live max-drawdown halt per #749 — halt new entries + alert at `RiskParameters.max_drawdown` (0.20), with the peak surviving process restarts and scoped per pm's adopted baseline policy (peak **true** equity since the last reconciled reset/session — an unqualified all-time peak would resurrect phantom-era book values). Residual gap to file as follow-up: a future clean restart that creates a new session re-baselines the peak ("20% per session") — needs a durable cross-session anchor. CF-B evidence: even in backtest the per-candle check overshoots (halted at 20.50%), so treat 20% as a *detection* line, not a guarantee.
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
