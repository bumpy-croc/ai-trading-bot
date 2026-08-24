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

_(Cleared by the 2026-08-24 retro. Empty on arrival for the fourth consecutive window. The
2026-08-17→24 window generated at least four agenda-worthy items that reached this retro only via
session transcripts: the 08-18 time-exit trace (→ #1083), the 08-20 peak-anchor discrepancy
(→ #1084), the missed 08-19 standup slot (→ #1085), and the seven-day merge stall (→ #1079). Seven
standup runs wrote nothing here and nothing to `log.md`. See [D-2026-08-24-01] and LESSONS §2.10 —
the mechanism is now identified and the task file has been amended to name a layer-2 sink.)_

- **2026-08-13 (#1036)** — *Still open: this item lived on the #1060 branch while the 2026-08-24
  retro cleared `main`'s copy, so it was never actioned — neither rule below appears in LESSONS.md
  as of `origin/develop`. It is re-appended here rather than dropped.* Two restart-safe risk
  seeders (#1001, #1032) read `_recovered_inactive_session_id`, a field whose lifetime is owned by
  an unrelated feature (the #668 carry-forward re-entry guard, which clears it before the first
  loop iteration). Result: the seeding shipped by #1032 had never once run on the carry-forward
  boot path, and went 30 days undetected on staging because the miss was logged as a normal
  "unavailable" and the provenance field reported `self_anchored` — the value that also means
  "legitimately nothing to seed from". Two rules worth codifying: (a) a consumer must not depend
  on a field whose lifetime another feature controls — give it its own field or resolve the value
  itself; (b) a "safety feature armed" provenance/telemetry value must distinguish *nothing to do*
  from *could not do it*, or a permanently broken safety feature looks healthy in the logs. Also:
  boot verification for a restart-safety feature must exercise the carry-forward path, not just
  the session-reuse path prod happens to take.

- **2026-08-24 (#1060 review round)** — An agenda item can be silently lost to a BRANCH: the retro
  clears `AGENDA.md` on `develop` while an unmerged branch still carries an unactioned item, and
  the merge resolves to the cleared version. Suggests the retro should check open PRs for AGENDA
  additions before clearing, or that items be appended in a merge-friendly append-only form
  (e.g. one file per item under `AGENDA.d/`).
