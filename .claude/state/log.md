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

## 2026-06-10 00:00 · track-record · risk-officer
Branch review (recovery extraction #486): verdict=approve, confidence=high
Scenarios checked: corrupt-balance fail-fast propagation, offline-SL balance-basis + double/zero-booking, atomic_balance_update parity, phantom/orphan reload (#657/#668/#677), close-only escalation variants (write vs _enter_close_only_mode), Protocol-mutation aliasing to engine object, init-before-start lifecycle, eager-capture window. Method/helper diffs byte-identical modulo self.->state. + one cosmetic wrap. Ref: src/engines/live/recovery.py, src/engines/live/trade_close_accounting.py

---

## 2026-06-10 · refactor · claude(session, human-directed)
**#486 step 4: startup recovery extracted to `engines/live/recovery.py` on stacked branch `claude/live-trading-engine-refactor-09koj9-recovery`; PR #796 opened for steps 1–3.**
- PR #796 (steps 1–3) opened against develop; first CI run failed on the new paper-session smoke (hardcoded SQLite URL rejected in CI) — fixed in 5cf5862 (conftest-provisioned DATABASE_URL + recovery disabled in the smoke to avoid shared-CI-DB session leakage); re-run pending.
- `LiveSessionRecoverer`: session balance recovery (#668/#681 fail-fast preserved), persisted-position reload with #657 self-heal, risk-manager re-registration, startup reconciliation incl. legacy offline-SL bookkeeping. Close-accounting helpers → `trade_close_accounting.py` (re-exported). Engine cumulative: 6,558 → 5,368 lines.
- Parity proof repeated: full unit suite green, 82 integration (parity+live) green, backtest fingerprint byte-identical, 5 new recoverer wiring contract tests.
- Reviews: code-reviewer *correct* (normalized diff of all 4 methods = zero semantic differences); risk-officer **APPROVE high confidence** (condition: CI green before merge).
- Remaining #486 scope: (d) config dataclass + <1,500-line target — next stacked PR after these merge.
Commits: 2ec7a8b, 2e5eb75, 2487564 (+ 5cf5862 on the PR branch).

---

## 2026-07-03 21:25 · decision · pm
Human (Board) filled charter.md TODOs and confirmed a high-risk-appetite autonomy envelope: daemon may change live capital, deploy to production, and promote a live-trading symbol's model `latest` symlink without per-action human approval. Risk-tolerance numbers set to match risk-limits.json (20% max drawdown, 6% max daily loss, 10% max position, 3x leverage; breach = halt new entries + page human).
Per explicit human instruction, relaxed the conflicting hard rules in CLAUDE.md and `.claude/agents/ml-engineer.md`: model promotion for live-trading symbols no longer requires human sign-off or `board_required: true`. The eval bar (held-out temporal split, per-regime breakdown, calibration check, >=48h paper validation) and a clean risk-officer review (`risk_review_required: true`) remain mandatory — self-certifying without running them does not count as "verified." All other `board_required: true` gates (e.g. kill-switch, charter.md changes) are unchanged.
Ref: charter.md (risk tolerance, autonomy envelope), CLAUDE.md (daemon hard rules), .claude/agents/ml-engineer.md

---

## 2026-07-04 13:20 · incident-open · risk-officer
**P1: the 20% max-drawdown hard cap was already breached in production (20.33% peak-to-trough, 2026-04-22 $103.82 → 2026-06-06 $82.71) — undetected; current equity $83.92 is 19.18% below true peak, $0.86 above the line.** Independent review of the HyperGrowth 365d benchmark from #844: reproduced exactly (-20.15% / 21.84% MaxDD / 104 trades, fresh worktree at develop@e1d24239); verdict **genuine tail risk, not a backtest artifact** (drawdown = 12-month slow bleed, PF 0.47 at 71% win rate; live already realized the same tail).
Four stacked control failures verified: (1) live `check_drawdown` dead code (#749); (2) backtest CLI `--max-drawdown` defaults 0.5 vs constants/risk-limits 0.20; (3) HyperGrowth loosens `dynamic_risk` to [0.15,0.30,0.45]/[0.8,0.5,0.2] in both engines — second tier past the kill line; (4) live drawdown input peak-resets on every restart (no `account_history` rehydration) — live currently perceives ~0.6% DD.
Counterfactuals on the same window: risk-limits.json thresholds (override removed) → **17.01% MaxDD, no breach, -16.08% return** (protection free on this path); `--max-drawdown 0.20` enforced → halt at 20.50% (overshoot; backstop only).
Recommended to pm (proposal 2026-07-04-02): entry-pause prod HyperGrowth now (margin < one stop-out), land #749 with persistent peak, drop the threshold loosening, fix the CLI default. Production accessed strictly read-only; no mutation performed by this session.
Ref: incidents/2026-07-04T1300-P1-hypergrowth-drawdown-cap-breach.md · proposals/2026-07-04-02-hypergrowth-drawdown-containment.md · docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md · GH #845, #844, #749

---

## 2026-07-04 13:55 · note · risk-officer
**CORRECTION to the 13:20 incident-open entry (ledger-verified, same day).** The claim "20% cap already breached in production (20.33% from a $103.82 peak)" is **withdrawn**: pm challenged the peak's provenance and re-verification confirms `account_history.balance` was software-pinned in the pre-#655 era (May 2026: one distinct balance value across 451 hourly rows) — the April "$103.82 peak" was phantom-era book value, not a true exchange read. True equity reads begin 2026-06-03 ($84.14). No true-equity 20% breach can be established; the "one stop-out from re-breach" imminence claim is also withdrawn. Adopted baseline (pm): drawdown peak = peak true equity since the last reconciled reset (2026-06-05 / session 20 ≈ $84.40) → current live DD ≈ 0.6%; standup tripwires stand ($80.18/$75.96/$67.52).
Unaffected and still standing: the exact 365d backtest reproduction (-20.15%/21.84% MaxDD — structural breach for the live config), the four control-layer failures, both counterfactuals (risk-limits.json thresholds → 17.01% MaxDD, no breach), and proposal steps 2-4. Proposal step 1 revised: tripwires binding, no immediate entry-pause. Recommend pm reclassify the incident P1 → P2.
Process lesson appended to the incident file: verify a balance column varies like a market-tracking value (distinct-count sanity check) before treating any equity peak as real — the phantom-balance failure mode claimed this review's first draft as a victim.
Ref: corrects log entry 2026-07-04 13:20 · incidents/2026-07-04T1300-P1-hypergrowth-drawdown-cap-breach.md (CORRECTION section) · proposals/2026-07-04-02-hypergrowth-drawdown-containment.md (revised) · GH #845

## 2026-07-05 08:10 · track-record · live-ops
Severity: yellow  Top anomaly: session had no Railway/DB access (no `railway` CLI, no `DATABASE_URL`, no `atb`) — process/DB/log checks not directly verifiable; reported from state files + GitHub only. Pre-existing #669/#689 (staging observability gap, silent-outage risk) still open.
Ref: docs/research/ops-snapshots/2026-07-05_0810.md

---

## 2026-07-05 08:16 · track-record · live-ops
Severity: yellow  Top anomaly: session has no live production telemetry access (no railway CLI, no DATABASE_URL, no atb) — snapshot is state-file/GitHub coverage only, not a fresh DB/process read; last live figures are ~18.5h stale (2026-07-04 13:55, equity $83.92, true DD ≈0.6%, tripwires $80.18/$75.96/$67.52). Only open incident concerning the live bot is GH #845 (HyperGrowth structural 21.84% backtest MaxDD, containment proposal not yet landed). No new P0/P1 found or filed.
Ref: docs/research/ops-snapshots/2026-07-05_0816.md
