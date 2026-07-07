---
name: weekly-retro
description: The weekly consolidation pass — review the week's log, incidents, agent failures, and prediction-vs-outcome record, then distill them into LESSONS.md appends and skill amendments. Use on the weekly cadence (charter review rhythm) or after an unusually dense week. This is the ONLY routine editor of the distillate layer; output is concrete process diffs, not prose.
---

# Weekly Retro

Episodic → semantic. The week's events live in layer 2 (log.md, incidents, experiments); this
skill distills them into layer 3 (LESSONS.md, the skills themselves) — and it is the only skill
that routinely edits layer 3, which is what keeps the distillate curated instead of accreted
(`docs/architecture/memory_system.md`, discipline 4). A retro that produces "we should be more
careful" has failed; the unit of output is a DIFF — a LESSONS append, a skill amendment, a
checklist line, a new tripwire.

## Inputs (all layer 2 + working state)

1. **log.md** — the week's entries end to end, not just the tail.
2. **Incidents + corrections** — anything opened/closed/corrected. Corrections are retro gold:
   the phantom-peak withdrawal (2026-07-04 13:55) became LESSONS §5.6 (the distinct-count check)
   — that's the pipeline working as designed.
3. **Agent failures** — stalls, wake-losses, kills, review collisions, fabricated/wrong relayed
   claims (`agent-fleet-health` findings; the 2026-07-05 ml-engineer session-integrity note).
4. **Prediction-vs-outcome record** — every dated prediction made in the log (deploy will be
   healthy, model will beat incumbent, "no midnight heroics" calls) vs what happened. Agents'
   calibration is a charter review item (monthly: "review calibration of each agent"); the
   risk-officer's own calibration note (2026-07-04: "consistent, not a reversal") is the format.
5. **Experiments** — verdicts vs their preregistered thresholds; any p-hacking drift.
6. **Scheduled tasks** — `ls ~/.claude/scheduled-tasks` + each task's expected trace (did the
   standup run daily? did the FOMC pause flip fire?). Tasks run only while the app is open —
   silent non-firing is a known failure mode; a task that didn't fire is a finding.
7. **Model scoreboard** — new rows this week; stale `latest` claims.

## The distillation pass

For each notable event, ask: **which layer-3 artifact would have prevented it or captured it?**

| Event class | Distills into |
|---|---|
| New bug class / trap → rule | LESSONS.md append (its "the trap → the rule" format, newest at section bottom) |
| A skill's procedure was wrong or incomplete | Amend the SKILL.md (PR like any code change) |
| A monitoring signature learned | LESSONS §5 (NOT the bot-monitor-live skill — method vs specifics, one-definition rule) |
| A process gap with no owner skill | Proposal for a new skill or checklist (Board-visible) |
| One-off with no generalizable rule | Nothing — resist the urge; a lessons file full of episodes is an unread lessons file |

Rules for editing layer 3:
- Every new rule cites its earned event (issue #, log date, incident id) — untraceable rules
  rot into superstition; the existing LESSONS/skills style is "each one paid for."
- Prefer amending an existing rule over adding a near-duplicate (consolidation, not accretion).
- Deleting a rule is allowed when its precondition died (e.g. a fixed bug's workaround) — note
  the removal in the retro log entry so the record explains the edit.
- Mid-week emergency LESSONS appends are legal; the retro reconciles/merges them.

## Scoreboard + tracker updates

Update `docs/research/model-scoreboard.md` rows for the week's exam results (append-only);
verify the deployed model's row matches reality (`atb live-control list-models` / registry
symlinks). Check the standup tripwire table is still the ratified one.

## Output (the definition of done)

A log.md entry (kind `note`, via `decision-record`) listing, as diffs:
- LESSONS.md sections appended/amended (with the earning event for each);
- skills amended (PR links);
- calibration verdicts per agent (over/under-confident, with the week's examples);
- scheduled-task audit result (fired/missed);
- process changes proposed to the Board (anything touching layer 1 → `risk-ratification`).

Zero diffs is a legitimate outcome for a quiet week — say so explicitly rather than
manufacturing lessons.

## Red flags

- A retro entry that is narrative summary without a single artifact diff.
- Editing log.md or closed incidents "while we're at it" — layer 2 is append-only, always.
- A lesson written into two places (skill + LESSONS) — pick the right layer, link the other.
- Skipping the prediction-vs-outcome pass because the week "went fine" — calibration drift is
  invisible exactly when things go fine.
