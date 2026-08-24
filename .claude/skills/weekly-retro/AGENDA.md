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

- **2026-08-13 (#1036)** — Two restart-safe risk seeders (#1001, #1032) read
  `_recovered_inactive_session_id`, a field whose lifetime is owned by an unrelated feature
  (the #668 carry-forward re-entry guard, which clears it before the first loop iteration).
  Result: the seeding shipped by #1032 had never once run on the carry-forward boot path, and
  went 30 days undetected on staging because the miss was logged as a normal "unavailable" and
  the provenance field reported `self_anchored` — the value that also means "legitimately
  nothing to seed from". Two rules worth codifying: (a) a consumer must not depend on a field
  whose lifetime another feature controls — give it its own field or resolve the value itself;
  (b) a "safety feature armed" provenance/telemetry value must distinguish *nothing to do* from
  *could not do it*, or a permanently broken safety feature looks healthy in the logs. Also:
  boot verification for a restart-safety feature must exercise the carry-forward path, not just
  the session-reuse path prod happens to take.

_(Process note, carried forward from the 2026-08-17 retro's disposition: this file arrived EMPTY for
three consecutive windows while agenda-worthy items were being written into `log.md` instead — the
2026-08-13 session alone produced at least four, one of which literally wrote "Candidate LESSONS
entry at the next retro" into the log rather than here. The item above is the counter-example: it was
appended at the moment of discovery, which is the intended behaviour. The one-line append that
populates this file costs less than the log entry that substitutes for it. See [D-2026-08-17-01].)_

_(Disposition note for the next retro: verify whether the 2026-08-24 retro (PR #1086) already
codified the two rules in the item above — if so, mark it actioned with the LESSONS section
reference; if not, action it. Per this skill's own rule, it gets a written disposition either way,
never a silent drop.)_
