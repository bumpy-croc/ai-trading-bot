# Company State

This directory is the persistent memory of the trading-bot operation. The daemon (CEO) reads from here at the start of every cycle and writes decisions back at the end. Every agent has a piece of state it owns.

**Rule of thumb**: if something would be worse to forget than to recompute, it belongs here.

## Directory layout

```
.claude/state/
├── README.md                    # this file
├── charter.md                   # human-owned: mandate, risk tolerance, KPIs
├── risk-limits.json             # canonical risk thresholds (single source of truth)
├── baselines.json               # "what normal looks like" — live-ops baseline metrics
├── decisions.jsonl              # append-only log of every material decision
├── proposals/                   # open/closed proposals awaiting or past review
│   ├── README.md
│   ├── open/                    # *.md files, one per proposal
│   ├── approved/
│   └── rejected/
├── incidents/                   # open/closed incidents
│   ├── README.md
│   ├── open/
│   └── closed/
├── track-records/               # per-agent calibration history
│   ├── README.md
│   ├── market-analyst.jsonl
│   ├── quant-researcher.jsonl
│   ├── risk-officer.jsonl
│   ├── live-ops.jsonl
│   └── ml-engineer.jsonl
└── registries/
    ├── experiments.jsonl        # owned by quant-researcher
    └── models.jsonl             # owned by ml-engineer
```

## Ownership

| Path | Owner | Writers | Readers |
|---|---|---|---|
| `charter.md` | **Human Board** | Human only | Everyone |
| `risk-limits.json` | **Human Board** | Human only | Everyone (risk-officer in particular) |
| `baselines.json` | `live-ops` | `live-ops` | Everyone |
| `decisions.jsonl` | `ceo` | `ceo` only | Everyone |
| `proposals/` | originating agent | originating agent + `ceo` (on verdict) | Everyone |
| `incidents/` | `live-ops` to open; `ceo`/specialists to update | various | Everyone |
| `track-records/{agent}.jsonl` | that agent | that agent | Everyone |
| `registries/experiments.jsonl` | `quant-researcher` | `quant-researcher` | Everyone |
| `registries/models.jsonl` | `ml-engineer` | `ml-engineer` | Everyone |

## Daemon cycle (the CEO's loop)

At the start of every `/standup` or `/triage`:

1. Read `charter.md` — reconfirm priorities.
2. Read `risk-limits.json` — know the lines.
3. Tail `decisions.jsonl` (last 20) — remember what was decided recently.
4. List `proposals/open/` and `incidents/open/` — know what's awaiting attention.

At the end:

1. Append to `decisions.jsonl` for every material decision or escalation.
2. Move proposal files between `open/` → `approved/` or `rejected/`.
3. Move incidents `open/` → `closed/` (after post-mortem).

## Schemas

### `decisions.jsonl` — append-only JSON lines

One JSON object per line:

```json
{"ts": "2026-04-21T14:03:00Z", "actor": "ceo", "kind": "decision|escalation|proposal|incident|post-mortem", "ref": "proposal-2026-04-21-01", "summary": "Approved model v4 promotion to staging", "rationale": "Risk review clean; 48h paper passed.", "followups": ["verify 48h paper monitoring"], "board_required": false}
```

Required fields: `ts`, `actor`, `kind`, `summary`. Optional: `ref`, `rationale`, `followups`, `board_required`.

### `risk-limits.json`

Single source of truth for the risk thresholds the bot currently operates under. `risk-officer` reads this; the code reads `src/config/constants.py`; any divergence between the two is itself a P0 finding and must be reconciled.

### `track-records/{agent}.jsonl` — append-only

```json
{"ts": "2026-04-21T10:00:00Z", "agent": "market-analyst", "call": "regime=trending-up", "confidence": "med", "horizon_hours": 24, "outcome_ts": "2026-04-22T10:00:00Z", "outcome": "correct|partial|wrong|inconclusive", "notes": "..."}
```

Calls are appended at the time of the call with `outcome: null`; a later sweep (run weekly by `/weekly-strategy-review`) fills in the outcome. This gives each agent a calibration history.

### `proposals/*.md`

See `proposals/README.md`. Each proposal is a single markdown file with frontmatter (id, from, to, status, created, updated), the ask, supporting data, and a verdict section.

### `incidents/*.md`

See `incidents/README.md`. Each incident is a single markdown file with severity, timeline, actions taken, and eventual post-mortem.

## Rules

- **Never rewrite history.** `decisions.jsonl`, `track-records/*.jsonl`, and `registries/*.jsonl` are append-only. Fix mistakes by appending a correction, not by editing.
- **Timestamps are UTC, ISO 8601.** Always. No local time. No naive datetimes.
- **IDs are stable.** A proposal's id stays the same when it moves between `open/` and `approved/rejected/`.
- **Humans own the charter and risk-limits.** The daemon proposes changes to these via a board-required decision; only the human edits these files.
- **When the state is missing, the daemon stops.** If `charter.md` is empty or `risk-limits.json` doesn't exist, the daemon should refuse to make material decisions and page the human.
