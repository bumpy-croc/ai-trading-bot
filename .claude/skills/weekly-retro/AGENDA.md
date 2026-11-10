# Weekly Retro — Agenda

Running agenda for the next weekly retro. **Any agent or the PM appends items here the moment
they're noticed** (process failures, near-misses, corrections, patterns worth codifying) instead of
carrying them in memory. One bullet per item: date, what happened, and the rule/change it suggests.

The retro reads this file FIRST, actions every item (LESSONS.md append, skill amendment, or GH
issue — or an explicit "no action, reasoning" disposition in the retro PR), then **clears the file
back to this header** in the same PR. Items must never be silently dropped: everything below either
becomes a diff or gets a written disposition.

---

## Items

_(Cleared by the 2026-09-28 retro. Empty on arrival for the **ninth** consecutive window; the only
open PR (#1263, last week's retro, superseded by this one) carried nothing new. This window's
items came from the `source:automation` issue #1265, standup comments on #1127/#1230/#1260, and
the scheduled-task transcripts. None came through here. #1099 (retire or restructure this file) is
still open and unowned. See [D-2026-09-28-01].)_

- 2026-10-04: #1242 made `latest` an explicit opt-in for local training (`TrainingConfig.set_latest`,
  `--set-latest`, `live-control train --auto-deploy`) and added the NO CHANGE-vs-`latest` CI check.
  LESSONS §1.16(a) and `weekly-retro/SKILL.md` still say the pipeline writes `latest` at training
  time; rewrite them to describe the opt-in and the CI check.
