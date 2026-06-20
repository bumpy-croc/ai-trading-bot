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

---

## 2026-06-10 · refactor · claude(session, human-directed)
**#486 live-engine refactor steps 1–3 landed on `claude/live-trading-engine-refactor-09koj9` (pure refactor, parity-proven).**
- `LiveStopLossManager` (`engines/live/execution/stop_loss_manager.py`) now owns every exchange-facing stop-loss call (place/retry, cancel, fill/held queries, re-protect, offline-fill detection). Engine keeps thin wrappers; original #486 acceptance criteria met — no direct `place_stop_loss_order`/`cancel_order`/`get_open_orders`/`get_order` in the engine.
- Monitoring glue → `engines/live/monitoring/` (`LiveAccountMonitor` + dataframe extractors). Engine 6,558 → 6,110 lines.
- The 3 byte-identical entry-handler methods (AST-verified) → `engines/shared/execution/entry_handler_mixin.py`; divergent orchestration deliberately NOT merged.
- Parity proof: 3,965 unit tests green; 51 parity tests green; deterministic backtest fingerprint byte-identical before/after every commit; new end-to-end paper-session smoke test (real strategy + real in-memory DB, start→entry→exit→shutdown) asserts gross P&L equals shared `pnl_percent` on recorded fills.
- Reviews: code-reviewer (no findings), architecture-reviewer (no blockers; P2+nits applied), risk-officer **APPROVE, high confidence** (all UNPROTECTED paths verified equivalent to origin/develop).
- Residual pre-existing risks noted by risk-officer (refactor-neutral): webhook alert delivery is fire-and-forget; SL retry budget hardcoded (3×, 1s backoff) in two places; `position_still_held` defers to ~120s reconciler on API errors.
Commits: 0e3c0c5, 9a8a1c0, d49b3d5, a5729d3, d12eb12. Remaining #486 scope (recovery extraction, config dataclass, <1,500-line target) intentionally deferred to follow-up PRs.
