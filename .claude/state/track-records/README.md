# Agent track records

Every specialist agent keeps a calibration history. This is the agent's memory across sessions and the human's evidence for trusting (or distrusting) a specific agent's calls.

## Schema — `{agent}.jsonl` (append-only)

```json
{"id": "mkt-2026-04-21-01", "ts": "2026-04-21T10:00:00Z", "agent": "market-analyst", "call": "regime=trending-up,conf=med", "horizon_hours": 24, "linked_brief": "docs/research/market-briefs/2026-04-21.md", "outcome_ts": null, "outcome": null, "outcome_notes": null}
```

## Call types by agent

| Agent | Example call | Outcome dimensions |
|---|---|---|
| `market-analyst` | `regime=trending-up,conf=med` | Did price actually trend? Was conf appropriate? |
| `quant-researcher` | `proposal=promote-params-X,expected_sharpe=1.2` | Was proposal approved? If executed, did live Sharpe match? |
| `risk-officer` | `verdict=approve,scenario=2022-collapse-drawdown<8%` | If approved, did a similar scenario actually stay <8%? |
| `live-ops` | `anomaly=none` or `anomaly=X` | Did a real incident occur within next 24h that was missed/caught? |
| `ml-engineer` | `model=BTCUSDT/v4,OOS_accuracy=61%` | Did live-prediction accuracy match? |

## Lifecycle

1. **Agent makes a call** → appends a row with `outcome: null`.
2. **Scheduled sweep** (weekly, part of `/weekly-strategy-review`) → fills in `outcome`, `outcome_ts`, `outcome_notes` for calls whose horizon has passed.
3. **Quarterly review** → summarize each agent's calibration: percentage correct by confidence level. If an agent is systematically overconfident, the CEO flags it for prompt tuning.

## Rules

- **Append-only.** Never edit a prior call's outcome. If wrong, append a correction with a `correction_for` field pointing at the prior id.
- **Link the artifact.** Always include a path to the brief/report/proposal the call came from. Unlinked calls are unverifiable and should not count toward calibration.
- **Honest outcomes.** The sweep should grade as `wrong` or `inconclusive` when appropriate — do not reclassify misses as "partially right."
- **Self-review.** Each agent should read the last 20 of its own track-record rows at the start of a session to stay calibrated (i.e., if you've been overconfident recently, temper the next call).
