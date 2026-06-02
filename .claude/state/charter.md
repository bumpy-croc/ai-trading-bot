# Company Charter

> **This file is owned by the human Board.** The daemon reads it; the daemon does not edit it. Update it when your priorities, risk tolerance, or capital situation changes — the daemon will reflect the new reality on the next cycle.
>
> Fill in the `TODO` placeholders before relying on the daemon for anything material.

## Mission

TODO: 1–2 sentences. Example: "Grow a $1,000 paper-trading account to $10,000 over 12 months via ML-driven crypto strategies, learning which approaches work before committing live capital."

## Operating mode

- Current trading mode: **paper** / **live** — TODO
- Capital under management (USD): **TODO** (e.g., $1,000 paper, $0 live)
- Environments in use: development / staging / production — TODO
- Active symbols: **TODO** (e.g., BTCUSDT, ETHUSDT)

## Risk tolerance

High-level statement of appetite. The concrete numeric limits live in `risk-limits.json`; this is the *why*.

- Maximum acceptable drawdown before human decides to halt: **TODO%**
- Maximum acceptable daily loss: **TODO%**
- Maximum single-position exposure: **TODO%**
- Leverage policy: **TODO** (e.g., "spot only, no leverage" / "up to 3× on futures")
- On any breach: **TODO** (e.g., "halt new entries, close risky positions, page human")

## Autonomy envelope

What the daemon **MAY do without asking**:
- Produce research, briefs, backtests, post-mortems
- Draft and open PRs
- Deploy to **staging** (never production)
- Run paper-mode experiments
- Restart the paper-trading process
- Update docs under `docs/research/`, `docs/`, and `.claude/state/`

What the daemon **MUST get human approval for**:
- Any change affecting live capital (sizing, strategy activation, parameter change)
- Deployment to **production**
- Promotion of a model's `latest` symlink for a live-trading symbol
- Changes to `risk-limits.json` or `charter.md`
- Triggering the kill-switch
- Any action the daemon itself classifies as "irreversible"
- Spending more than $TODO in inference cost per 24h

What the daemon **MUST NEVER do**:
- Execute trades manually (all trades go through the bot engines)
- Modify closed incidents or past `log.md` entries
- Act on a proposal that lacks a risk-officer verdict, when `risk_review_required` is true
- Continue operating if `charter.md` or `risk-limits.json` is missing/invalid

## KPIs the Board cares about

List in priority order. The daemon optimizes for these, in this order:

1. **Capital preservation** — do not breach risk-limits.json
2. **Backtest/live parity** — variance between the two stays within TODO%
3. **Sharpe ratio** (rolling 30-day) — target TODO, minimum TODO
4. **Win rate** — target TODO%, minimum TODO%
5. **Maximum drawdown** (rolling) — target <TODO%, hard limit in risk-limits.json
6. **Cost per decision** — inference + exchange fees, target <TODO

## Escalation

When something needs the human:

- **Method**: TODO (e.g., "Create incident file in `.claude/state/incidents/` + matching GitHub Issue with `type:incident` label; ping Slack webhook at TODO; human checks every TODO hours")
- **Response SLA expected**: TODO
- **What the daemon does while waiting**: TODO (e.g., "Freeze new entries; maintain existing stops; continue paper trading normally")

## Review cadence

- Daily: `/standup` produces a brief
- Weekly: a `/standup` with a weekly-review prompt + charter re-read (Board amends if needed)
- Monthly: post-mortem on all closed incidents, review KPI trend, review calibration of each agent

## Known constraints & preferences

Freeform section. Things the daemon should always remember:

- TODO (e.g., "Prefer momentum over mean-reversion for crypto.")
- TODO (e.g., "Do not use sentiment features until they beat baseline by 10% OOS.")
- TODO (e.g., "Never retire `ml_basic` — it's the control arm.")
- TODO (e.g., "Avoid deploying Fridays after 18:00 UTC.")

---

*Last updated by human: YYYY-MM-DD*
*Charter version: 0.1*
