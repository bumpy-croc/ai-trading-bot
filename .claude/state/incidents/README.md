# Incidents

An incident is a deviation from expected operation that required attention. Opened by `live-ops` (or any agent that detects it); classified, mitigated, and eventually post-mortem'd.

## Severity

| | Definition | Who decides | Auto-response |
|---|---|---|---|
| **P0** | Live capital at risk RIGHT NOW. Duplicate orders, DB/memory divergence, auth failure with open positions, data corruption, kill-switch condition. | `live-ops` or `risk-officer` | Page human immediately. Do not attempt automatic recovery on live-capital processes. |
| **P1** | Trading degraded but not bleeding. Paper process down, stale data > 2× timeframe, elevated error rate, health endpoint down. | `live-ops` | Capture evidence, attempt paper-safe mitigation, notify PM on next cycle. |
| **P2** | Slow or non-critical. High API latency, minor deploy issue, warning thresholds approached but not breached. | `pm` | Log, monitor, address in next standup. |
| **P3** | Observation. Baseline drift, agent disagreement worth noting. | any | Log; no action required. |

## Lifecycle

```
open/*.md  --mitigated-->  open/*.md (with `status: mitigated`)
                                 |
                           post-mortem
                                 |
                                 v
                         closed/*.md
```

## File naming

`YYYY-MM-DDThhmm-severity-short-slug.md` — e.g., `2026-04-21T1430-P0-db-memory-divergence.md`.

## Template

```markdown
---
id: 2026-04-21T1430-P0-db-memory-divergence
opened_by: live-ops
severity: P0
status: open        # open | mitigated | closed
opened_at: 2026-04-21T14:30:00Z
mitigated_at: null
closed_at: null
human_paged: true
affected_components: [live-engine, positions-table]
affected_symbols: [BTCUSDT]
---

## What happened

(Observable facts, with timestamps. No speculation.)

## Detection

How it was caught. What signal. Was this signal expected or a surprise?

## Impact

Real money / paper only? Positions affected? Estimated $ impact if any?

## Timeline

```
14:30:00 UTC  — [detection] …
14:32:00 UTC  — [escalation] paged human via …
14:35:00 UTC  — [mitigation] …
```

## Actions taken

Concrete actions with who/what. Include commands run verbatim.

## Current state

Is the bleeding stopped? What is the system doing right now?

## Post-mortem (filled after close)

### Root cause
### Contributing factors
### What went well
### What went poorly
### Action items (each links to a proposal or tracker)
```

## Rules

- **P0 auto-pages the human.** No exceptions. The daemon does not wait for the next standup.
- **Never close an incident without a post-mortem** for P0/P1. P2/P3 may close with a one-line note.
- **Post-mortems are blameless.** The question is "what in the system allowed this" not "who did this."
- **Action items from post-mortems become proposals.** Each one gets a proposal id.
- The daemon should **refuse new material decisions while any P0 is open** except those directly related to the incident.
