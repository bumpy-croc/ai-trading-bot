# Company Charter

> **This file is owned by the human Board.** The daemon reads it; the daemon does not edit it. Update it when your priorities, risk tolerance, or capital situation changes — the daemon will reflect the new reality on the next cycle.
>
> Fill in the `TODO` placeholders before relying on the daemon for anything material.

## Mission

Grow $1000 live account 

## Operating mode

- Current trading mode: **live**
- Capital under management (USD): $1,000 paper, $87 live
- Environments in use: development / staging / production
- Active symbols: ETHUSDT

## Risk tolerance

High-level statement of appetite. The concrete numeric limits live in `risk-limits.json`; this is the *why*.

- Maximum acceptable drawdown before human decides to halt: **20%** (matches `risk-limits.json` `max_drawdown_pct` — given the high risk appetite, the hard system limit doubles as the human decision point)
- Maximum acceptable daily loss: **6%** (matches `max_daily_risk_pct`)
- Maximum single-position exposure: **10%** of capital (matches `max_position_size_pct`); positions above 20% are flagged as large (`large_single_position_threshold_pct`)
- Leverage policy: **up to 3x on futures** (matches `max_leverage`); spot preferred, leverage used only when the strategy's signal and sizing justify it
- On any breach: **halt new entries and page human** (matches `risk-limits.json` `escalation.breach_action`); existing positions run their own stop/exit logic unless the breach itself requires an emergency close

## Autonomy envelope

What the daemon **MAY do without asking**:
- Produce research, briefs, backtests, post-mortems
- Draft and open PRs
- Deploy to **staging**
- Any change affecting live capital (sizing, strategy activation, parameter change)
- Promotion of a model's `latest` symlink for a live-trading symbol
- Deployment to **production**
- Run paper-mode experiments
- Restart the paper-trading process
- Update docs under `docs/research/`, `docs/`, and `.claude/state/`

What the daemon **MUST get human approval for**:
- Changes to `charter.md`
- Triggering the kill-switch
- Any action the daemon itself classifies as "irreversible"
- Spending more than **$50** in inference cost per 24h

What the daemon **MUST NEVER do**:
- Execute trades manually (all trades go through the bot engines)
- Modify closed incidents or past `log.md` entries
- Act on a proposal that lacks a risk-officer verdict, when `risk_review_required` is true
- Continue operating if `charter.md` or `risk-limits.json` is missing/invalid

## KPIs the Board cares about

List in priority order. The daemon optimizes for these, in this order:

1. **Capital preservation** — do not breach risk-limits.json
2. **Backtest/live parity** — variance between the two stays within **15%**
3. **Sharpe ratio** (rolling 30-day) — target **1.5**, minimum **0.5**
4. **Win rate** — target **55%**, minimum **45%**
5. **Maximum drawdown** (rolling) — target **<15%**, hard limit in risk-limits.json (20%)
6. **Cost per decision** — inference + exchange fees, target **<$0.50**

## Escalation

When something needs the human:

- **Method**: Create incident file in `.claude/state/incidents/` + matching GitHub Issue with `type:incident` label; ping Slack webhook if configured; human checks async
- **Response SLA expected**: 1 hour for P0/critical, 24 hours otherwise
- **What the daemon does while waiting**: Freeze new entries on the affected symbol; maintain existing stops; continue paper trading and other symbols normally

## Review cadence

- Daily: `/standup` produces a brief
- Weekly: a `/standup` with a weekly-review prompt + charter re-read (Board amends if needed)
- Monthly: post-mortem on all closed incidents, review KPI trend, review calibration of each agent

## Known constraints & preferences

Freeform section. Things the daemon should always remember:

- High risk tolerance applies to position sizing, drawdown, and leverage — it does not relax the CODE.md bar for thread safety, financial correctness, or reconciliation
- Sentiment and other experimental features must beat the current baseline by a clear, out-of-sample margin (see `docs/research/`) before promotion to live
- Prefer strategies with demonstrated robustness across multiple backtested market regimes over single-regime overfits
- Avoid deploying to production on Fridays after 18:00 UTC or immediately ahead of major macro events (FOMC, CPI)

---

*Last updated by human: 2026-07-03*
*Charter version: 0.1*
