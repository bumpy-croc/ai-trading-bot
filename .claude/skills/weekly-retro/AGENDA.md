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
