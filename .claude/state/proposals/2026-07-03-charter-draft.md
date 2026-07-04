---
status: proposed
board_required: true
title: Charter v1 draft — max-growth-within-limits mandate
created: 2026-07-03
owner: daemon(PM)
risk_review_required: false
---

# Proposal: fill in charter.md (Board-owned — copy/edit/commit yourself)

The charter is all TODOs, which per CLAUDE.md blocks material daemon decisions. Below is a
complete draft reflecting the mandate agreed in-session on 2026-07-03. Values marked
**(suggested)** are my invention — adjust freely. The daemon never edits `charter.md` itself.

---

## Mission

Grow the live Binance cross-margin account (≈ $84 on 2026-07-03) as fast as the agreed risk
framework allows, compounding toward **$1,270 (≈ £1,000)** as a **multi-month** target
(realistic horizon 6–14 months). Capital preservation outranks speed at all times.

## Operating mode

- Current trading mode: **live**
- Capital under management (USD): **≈ $84 live** (USDT, Binance cross-margin)
- Environments in use: development (`develop`), staging (`staging`), production (`main`, Railway)
- Active symbols: **ETHUSDT** (BTCUSDT under evaluation via 2026-07-03 tournament)

## Risk tolerance

Concrete numbers live in `risk-limits.json`; this is the why: a small account rebuilding trust
after the June 2026 capital-erosion incidents — losses must stay survivable and diagnosable.

- Maximum acceptable drawdown before human halts: **20%** (matches risk-limits.json)
- Maximum acceptable daily loss: **6%**
- Maximum single-position exposure: **10%** of equity (large-position review threshold 20%)
- Leverage policy: **cross-margin up to 3×**, long and short
- On any breach: **halt new entries, keep stops on open positions, page human** (matches
  `escalation.breach_action`)

## Autonomy envelope

(Existing MAY / MUST-approve / MUST-NEVER lists in charter.md are already accurate — keep them.)

- Inference spend cap: **$25 per 24h (suggested)**

## KPIs the Board cares about

1. **Capital preservation** — no risk-limits.json breach
2. **Backtest/live parity** — variance within **10% (suggested)**
3. **Sharpe (rolling 30d)** — target **1.5**, minimum **0.5 (suggested)**
4. **Win rate** — target **55%**, minimum **40% (suggested)**
5. **Max drawdown (rolling)** — target **<10%**, hard limit 20%
6. **Cost per decision** — target **<$0.05 (suggested)**

## Escalation

- **Method**: incident file in `.claude/state/incidents/` + GitHub issue with `type:incident`.
  NOTE: the 2026-06-08 observability audit found the alert webhook UNSET (double-blind) —
  fixing alert delivery should be part of accepting this charter. **(Board: pick a channel —
  Telegram/Slack/email.)**
- **Response SLA expected**: **24h (suggested)**
- **While waiting**: freeze new entries; maintain existing stops; paper trading may continue.

## Known constraints & preferences

- Run backtests sequentially on the dev Mac (thermal limits).
- Prod promotes are surgical per-fix cherry-picks onto `main` — never wholesale develop→main.
- `HyperGrowth` is the live incumbent (session 20); strategy swaps need tournament evidence +
  risk-officer review + human sign-off.
- Prod DB reads: read-only psql via saved URL, human-approved per session; never `railway ssh`
  writes without explicit per-action approval.

---

*If accepted: copy the sections above into `.claude/state/charter.md`, set "Last updated by
human: 2026-07-03", bump charter version to 1.0, and commit.*
