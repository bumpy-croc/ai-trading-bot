---
name: risk-ratification
description: Guided Board sitting for the human-owned identity files (charter.md, risk-limits.json) — assemble every pending divergence and proposed change into one reviewable diff with evidence, verify constants.py mirrors risk-limits.json, get the human's edit/merge, stamp $last_reviewed, log the ratification. Use when a limit divergence is flagged, when charter TODOs block decisions, or on the periodic charter re-read.
---

# Risk Ratification

`charter.md` and `risk-limits.json` are layer 1 — IDENTITY, human-owned, read-only to every
agent (`docs/architecture/memory_system.md`). The daemon's job is to make the Board sitting
CHEAP: one reviewable diff, every line backed by evidence, so ratification takes minutes instead
of rotting for weeks. The cautionary tale: risk-limits.json `max_position_size_pct` 0.10 vs
deployed 0.20 was flagged 2026-07-03, re-flagged twice on 07-04 by two different agents, and
stayed unreconciled — with `$last_reviewed: 1970-01-01` — because nobody packaged the decision.

## 1. Assemble the pending set

Sweep for every open item that needs a layer-1 edit:
- log.md entries with "escalated to Board" / divergence flags still unresolved;
- risk-officer verdicts conditioned on a limit change;
- charter TODOs (an unfilled mission/autonomy/escalation TODO BLOCKS all material decisions —
  state README rule; the daemon must refuse and page rather than improvise);
- `capital-review` / `kill-switch-drill` findings tagged for ratification.

## 2. Verify the constants mirror — divergence is P0 by the file's own words

risk-limits.json's `$source_of_truth_note`: "Must match src/config/constants.py. Any divergence
is a P0." Check EVERY key against its constant, plus the as-deployed value (railway.json
startCommand flags and env vars can silently override both — prod ran `--max-position 0.5` via
startCommand for weeks before #835/#836 caught it). Known traps from the 2026-07-03/04 reviews:
- `DEFAULT_MAX_CORRELATED_RISK` (0.10) and `DEFAULT_MAX_CORRELATED_EXPOSURE` (0.15) are TWO
  distinct constants — verify which one a JSON key means before declaring a match;
- strategy-level hardcodes can shadow both (kelly_momentum's `fallback_fraction=0.03` vs
  `DEFAULT_KELLY_FALLBACK_FRACTION=0.02`; HyperGrowth's dynamic_risk override loosening the
  drawdown tiers past the kill line — the #845 control-failure #3);
- `kill_switch.manual_trigger_command` must name a command that EXISTS and acts
  (`atb live-control halt` does not exist; `emergency-stop` is simulated — `kill-switch-drill`
  finding). A dead reference in the risk file is itself a ratification item.

Three-way rule: JSON ↔ constants.py ↔ as-deployed. Any mismatch goes in the diff with its
evidence line.

## 3. Package one reviewable diff

For each proposed change: current value → proposed value → evidence (issue/experiment/incident
ref, the risk-officer verdict, the deployed reality) → consequence of NOT changing. Two delivery
modes, human's choice:
- **PR the human merges** (preferred — the merge IS the ratification signature), branch
  containing ONLY layer-1 edits + the constants.py mirror updates in the same diff so they
  cannot diverge; or
- **exact file edits** spelled out (old line → new line) for the human to apply by hand.

Include the stamp in the same diff: `$last_reviewed: YYYY-MM-DD`, `$last_reviewer`, and
charter.md's "*Last updated by human:*" footer. A ratification that doesn't move the stamp
didn't happen.

## 4. The sitting

Present the diff + a one-screen summary (what changes, why, what it unblocks). The human edits
or merges — the daemon NEVER commits to layer 1 itself, even with in-session verbal approval;
the file history must show the human's hand (charter hard rule + CLAUDE.md daemon rules). If
the human rejects an item, that's a decision too — record it so the flag stops re-firing.

## 5. Record

log.md entry via `decision-record` (`[D-…]`, kind `decision`): items ratified/rejected/deferred,
the new stamp values, refs. Close the loop: update any GH issues waiting on `needs:human-input`
/ `needs:human-approval`; notify `capital-review` inputs if limits changed; if a limit change
alters guard behavior, schedule a `kill-switch-drill` pass on staging.

## Red flags

- An agent "temporarily" editing risk-limits.json or charter.md to unblock work. Never — halt
  and page instead.
- Ratifying the JSON without the constants.py mirror in the same diff (creates the next P0).
- A verbal "yes, change it" with no file edit by the human — that ratifies nothing.
- `$last_reviewed` older than the last known divergence flag: the backlog exists; run this skill.
