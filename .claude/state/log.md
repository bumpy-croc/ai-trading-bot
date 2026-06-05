# Log

Chronological record of every material action by the daemon. Append-only. Newest last.

See `README.md` for format.

---

## 1970-01-01 00:00 · note · system
Log initialized.

---

## 2026-06-05 · incident-resolve · daemon(PM)
**Prod "orphan" was a PHANTOM, not double-exposure.** Halted the approved #677+#679 promote after prod logs showed the bot was no longer flat (opened new pos #13 at 20:07 UTC) with the old orphan SL still referenced. With explicit human approval, ran a strictly read-only reconciliation inside the prod container (`railway ssh`, SELECT + GET only):
- DB had 2 OPEN ETHUSDT rows: **#12** (0.0033 ETH, session 17, SL `47100334866`) and **#13** (0.0037 ETH, session 18, SL `47181104013`).
- Exchange truth: only **0.00378 ETH** held (= #13's 0.0037 + dust); only **one** live ETH order (`47181104013`); account equity **$83.28** vs tracked $84.13.
- ⇒ **#12 is a phantom** (stale pre-#671 close-gap row; its ETH + SL already gone). No double-exposure. Only #13 is real and SL-protected.

**Actions (human-approved):**
1. Closed phantom #12 in prod DB (guarded status-only write; re-verified exchange showed no second holding before writing). Balance NOT manually poked — layer-2 `_sync_margin_equity` books the $0.85 (1%) overstatement to true equity at the next FLAT moment (race-free).
2. Pivoted the prod promote from #677+#679 → **#677-only** (PR #682; closed #680). #677 is phantom-safe (re-adopts only most-recent inactive session; `old_session_id=self._recovered_inactive_session_id`). #679's `adopt-all` would resurrect phantoms because the margin-LONG reconcile check (`reconciliation.py:1885`) reads AGGREGATE balance — a phantom borrows the real position's holdings and survives.
3. Filed **#683** to redesign #679 as exchange-verified-before-adopt; #679 to be reverted on develop (interim parity with prod = #677-only).

Refs: #668, #677, #679, #671, #648/#15, #28 (booking-while-held), #674 (margin Decimal, in prod).
