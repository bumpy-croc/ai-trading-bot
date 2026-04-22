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
## 2026-04-21 10:00 · decision · ceo
Approved BTC model v4 promotion to staging.
Rationale: risk review clean; 48h paper passed.
Ref: proposals/2026-04-20-01-promote-btc-v4.md
```

## Rules

- **Human-owned files never change without a Board decision.** Daemon proposes; human edits `charter.md` and `risk-limits.json`.
- **`log.md` is append-only.** Corrections are new entries referencing the earlier one. Never edit history.
- **Timestamps are UTC, ISO-like.** Always.
- **Missing or invalid state blocks material decisions.** If `charter.md` or `risk-limits.json` is missing or has unfilled TODOs, daemon stops and pages the human.
