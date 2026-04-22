# Company State

The daemon's persistent memory. Human-owned files set policy; daemon-owned files record what happened.

## Files

| File | Owner | Purpose |
|---|---|---|
| `charter.md` | **human** | Mandate, risk tolerance, autonomy envelope, escalation method |
| `risk-limits.json` | **human** | Canonical risk thresholds (must match `src/config/constants.py`) |
| `log.md` | daemon | Chronological log of every material action — one scannable file |
| `proposals/*.md` | originating agent | Active back-and-forth on changes awaiting review/approval |
| `incidents/*.md` | any detector | Operational issues — P0 / P1 / P2 / P3 |

## Daemon cycle

**Start of every `/standup` or `/triage`:**
1. Read `charter.md` — if any `TODO` remains in mission / autonomy envelope / escalation, refuse material decisions and ask the human.
2. Read `risk-limits.json` — know the lines.
3. Tail `log.md` (last ~50 lines) — remember recent activity.
4. List `proposals/` and `incidents/`, filter `status: open` from frontmatter — see the active queue.

**End of every material action:**
1. Append a dated section to `log.md`.
2. Update `status` frontmatter in proposal / incident files as they move through their lifecycle.

## `log.md` format

One append-only markdown file. Each entry is an H2 section. H2 line pattern: `YYYY-MM-DD HH:MM · kind · actor-or-severity`.

Kinds: `decision`, `escalation`, `proposal-open`, `incident-open`, `incident-close`, `post-mortem`, `note`.

```
## 2026-04-21 10:00 · decision · pm
Approved BTC model v4 promotion to staging.
Rationale: risk review clean; 48h paper passed.
Ref: proposals/2026-04-20-01-promote-btc-v4.md
```

## Backlog — lives on GitHub

The live work queue is **GitHub Issues + the "Agent PM Board" Project** (`bumpy-croc/projects/6`). Files above are for durable audit and human-owned config; the backlog itself is on GitHub so humans can triage from anywhere and PR linkage is native.

**Label taxonomy** (every card carries one each of `state`, `priority`, `type`, `owned-by`, `source`, plus ≥1 `area`, plus 0+ `needs`):

- `state:idea → researching → proposed → building → paper → shipped → monitoring → closed | icebox`
- `priority:p0..p3`
- `type:research | feature | fix | experiment | model-promotion | strategy-change | infra | incident | post-mortem-action`
- `area:strategy | ml-model | risk | live-ops | data | infra | backtest | sentiment`
- `owned-by:pm | quant-researcher | ml-engineer | risk-officer | live-ops | market-analyst | code-reviewer | architecture-reviewer | human`
- `needs:risk-review | code-review | human-approval | human-input | data`
- `source:ideation | forensics | human | market-anomaly | parity-gap | incident`

**Custom fields** (on the Project): `Priority Score` (number), `Wake At` (date), `Effort Days` (number), `Cost Tokens` (number), `Confidence` (Low/Med/High).

**Agent queries** (read-only for most):

```bash
# What's actionable right now?
gh issue list --state open --label state:researching,state:proposed,state:building --json number,title,labels,updatedAt

# What's blocked on the human?
gh issue list --state open --label needs:human-approval --json number,title

# Open incidents?
gh issue list --state open --label type:incident --json number,title,labels

# Due paper tests (requires Project field lookup — use the "In Paper" Project view for humans)
```

**See** `.claude/docs/project-setup.md` for the Project views the human needs to create in the UI and related setup notes.

## Rules

- **Human-owned files never change without a Board decision.** Daemon proposes; human edits `charter.md` and `risk-limits.json`.
- **`log.md` is append-only.** Corrections are new entries referencing the earlier one. Never edit history.
- **Timestamps are UTC, ISO-like.** Always.
- **Missing or invalid state blocks material decisions.** If `charter.md` or `risk-limits.json` is missing or has unfilled TODOs, daemon stops and pages the human.
- **Status is canonical in frontmatter (files) and in `state:*` labels (issues).** If they disagree, the label wins and the PM logs a cleanup.
