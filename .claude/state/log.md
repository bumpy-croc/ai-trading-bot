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

## 2026-07-03 · decision · daemon(PM)
**New Board mandate (human, in-session): maximize growth within existing risk limits.** Human initially asked for £85→£1,000 by ~2026-07-12; PM assessment: infeasible (11.8x in ~9 days ≈ +31%/day compounded; risk-limits.json caps the theoretical best case at ~1.06^9 ≈ 1.7x; zero-edge all-in optional-stopping bound ≈ 8.5% success). Human explicitly chose "Max growth within limits"; £1,000 becomes a multi-month compounding target.
**Read-only prod verification (human-approved, psql via saved prod URL, session set read-only):**
- Bot HEALTHY: hourly account_history heartbeat fresh (last 18:46 UTC today).
- Equity **$84.06** (balance $84.40), margin unused, session 20 `HyperGrowth` ETHUSDT 1h LIVE since 2026-06-05.
- **5 consecutive winning live trades** (2026-06-07→07-02): +3.77/+3.48/+1.66/+3.20/+3.14% per-trade; balance 83.29→84.40.
- Open position: SHORT 0.0079 ETH @ 1696.83 (now ~1739, uPnL −$0.33), SL 1864.65 (risk ≈ $1.33 ≈ 1.6% equity), TP 1186.60.
- Growth bottleneck = **position size × trade frequency** (~+1.3%/month at current pace), NOT win rate.
**Actions:** dispatched market-analyst (7–10 day regime brief) + quant-researcher (sequential backtest tournament incl. incumbent HyperGrowth; frequency/exposure reported per entrant). Charter draft written to `proposals/2026-07-03-charter-draft.md` for Board fill-in (charter.md is human-owned). Any resulting live change (sizing, strategy swap, symbol add) goes to human sign-off per autonomy envelope.

## 2026-07-03 20:45 · track-record · quant-researcher
Experiment #1: Strategy tournament (8 strategies × 2 symbols × 2 windows = 25 runs) → concluded
Evidence: docs/research/experiments/2026-07-03_strategy-tournament.md
Result: kelly_momentum/ETHUSDT best risk-adjusted (Sharpe 0.21/0.32, +19.25%/+16.67% vs incumbent HyperGrowth -0.86% 30d). 
Incumbent HyperGrowth/ETHUSDT confirmed positive 90d (+10.66%) but flat/negative 30d. 
Dead strategies (0 trades): ml_adaptive, ml_sentiment, ensemble_weighted. 
Catastrophic: adaptive_trend (-33.07%, DD 35.77%). 
Win-rate metric unreliable for non-ML strategies (balance-based total_return used instead).
Proposal: kelly_momentum/ETHUSDT as challenger. Needs risk-officer review before live change.

## 2026-07-03 20:30 · track-record · risk-officer
Proposal 2026-07-03-01-kelly-momentum-ethusdt: reviewed as 3 options (a) HyperGrowth sizing raise, (b) swap to kelly_momentum live, (c) hybrid.
Verdict: (a) approve-with-conditions [confidence med] — chartered 2-3% risk/trade is NOT a clean param bump: needs notional 20-30% (breaches max_position_size 10% + large_single 20%) OR stop widened to 20-30% (>=2.5% breaches max_stop_loss_pct 20%). Only clean in-charter point is ~2% via 10% notional x 20% stop (stop at the max-allowed boundary, 4 daily-sigma) OR a small notional bump to ~1.6-1.8% risk staying under 10% notional at current ~10% stop. Recommend capped step-up, not full 3%.
Verdict: (b) reject [confidence high] — KellyCriterionSizer.record_trade() has ZERO callers in live OR backtest engine (only unit tests). Sizer's _trades deque never populates => strategy runs in permanent cold-start fallback (0.03 x confidence x strength) forever; Kelly edge never activates. Backtest +19% return with 0.03% MaxDD + 0% win_rate is mutually inconsistent => return is an accounting artifact, not tradeable P&L. Proposal also mis-states min_trades=10 (code uses DEFAULT_KELLY_MIN_TRADES=30). No live record. Do not put live capital behind unverified metrics + dead sizer wiring.
Verdict: (c) approve [confidence high] — keep HyperGrowth live with capped (a) sizing; run kelly_momentum PAPER in staging to build a real track record AND expose the record_trade wiring gap before any live consideration.
Divergence flags: risk-limits.json vs constants.py: kelly_max_fraction 0.20 matches; BUT proposal min_trades=10 != code 30, and kelly_momentum.py fallback_fraction=0.03 != DEFAULT_KELLY_FALLBACK_FRACTION=0.02. constants.py has DEFAULT_MAX_CORRELATED_RISK=0.10 (line 99) alongside DEFAULT_MAX_CORRELATED_EXPOSURE=0.15 (line 278, matches JSON) — two constants, verify intended. risk-limits.json $last_reviewed=1970-01-01 (never formally reviewed).
Model-risk P1: kelly record_trade wiring gap (file GH issue). Ops finding: live account at 15.95% notional > constants default max_position_size 10% — confirm session-20 --max-position config.
Scenarios checked: 5-consecutive-loss drawdown path, worst-case-day vs 6% daily cap, ETH 5%/day vol vs stop distance, Jul8 FOMC / Jul14 CPI event windows, Kelly cold-start sizing, cap-collision analysis.
Ref: .claude/state/proposals/2026-07-03-01-kelly-momentum-ethusdt.md

---

## 2026-07-03 · verification+decision · daemon(PM)
**Backtest tournament returns are FABRICATED for partial-exit strategies; live record unaffected; sizing raise approved by human.**
- Independent verifier reproduced kelly_momentum ETHUSDT 30d: reported +16.67% = **+$14.19 of phantom partial-exit credits** (units bug: fraction-of-POSITION passed where fraction-of-BALANCE is consumed; inflation ~800–3,900× per event). Corrected result ≈ **0.0%**. Root causes at exit_handler.py:363-379, partial_exit_executor.py:123-137, position_tracker.py:193 (phantom zeroing → 0% win-rate artifact), exit_handler.py:405-459 (zombie scale-ins), engine.py:1114-1119 (MaxDD realized-cash-only), engine.py:308-319 (strategy partial configs never hydrate; defaults injected).
- ⇒ ENTIRE tournament return column void where positions gained ≥3% intratrade (incl. HyperGrowth's +10.66%/90d). **Live prod record (5 wins, 83.29→84.40, DB-verified) is a different engine/code path and stands.** Risk-officer's (a)-verdict logic (live-position stress math) unaffected.
- Risk-officer verdicts: (a) sizing raise approve-with-conditions ≤2% risk/trade [med]; (b) kelly_momentum live swap **REJECT [high]** — PM independently confirmed `KellyCriterionSizer.record_trade` has zero engine callers (permanent cold-start); (c) hybrid approve [high].
- **Human (Board) approvals in-session:** (1) raise HyperGrowth sizing to ~2% risk/trade cap (base/risk_fraction 0.20→0.25, explicit --max-position 0.20, stop 10%); (2) add FEATURE_ENTRY_PAUSE flag for Jul-8 FOMC / Jul-14 CPI windows.
- Also found: live entry paths do NOT enforce max_position_size (prod positions 9.2–16% notional vs documented 0.10) — P1, fix included in sizing PR.
- Dispatched: PR agent feature/hypergrowth-sizing-2pct (sizing+cap+pause), PR agent fix/backtest-partial-exit-units (5 root causes, TDD), Explore agent (does LIVE share the partial-exit units bug — pending).
- Pending human actions: update risk-limits.json max_position_size_pct 0.10→0.20 + $last_reviewed (human-owned); fill charter from proposals/2026-07-03-charter-draft.md; `railway login` (CLI auth expired — needed for deploy monitoring).

---

## 2026-07-03 · verification · daemon(PM)
**LIVE engine confirmed SAFE from the partial-exit units bug.** Live path converts correctly (`validation.py:256-296` convert_exit_fraction_to_current → `live/execution/exit_handler.py:807-812` → executor consumes coherent units); bug is backtest-only (`backtest/execution/exit_handler.py:375-379` passes unconverted). Prod `main` and develop byte-identical on all partial-exit paths. Mystery of zero live partials resolved: live HyperGrowth hydrates its own targets [0.08, 0.15, 0.30] (strategy override wins in live, `trading_engine.py:524-532`), and all 5 winners exited via trailing stop at +3.1–3.8% — below the first 8% target. Backtest force-injects defaults [0.03,0.06,0.10] ignoring strategy overrides — a backtest-live PARITY gap, in scope for fix/backtest-partial-exit-units. Sizing PR cleared to proceed.

---

## 2026-07-03 · deploy-prep · daemon(PM)
**#835 (HyperGrowth 2% sizing + enforced 0.20 cap + FEATURE_ENTRY_PAUSE) merged to develop (squash 39984f93) after code-reviewer APPROVE (no P0/P1), architecture-reviewer APPROVE (no blockers), CI fully green.** Review follow-ups included: scale-ins gated under pause flag (EntryPauseGate), entry_pause in feature_flags.json, accepted-gap notes. Findings during build: prod's real cap was `--max-position 0.5` via railway.json startCommand (not the 0.10 default; issue #836); scale-in path bypassed caps entirely via daily-risk-budget reset. **Promote PR #841 opened to main** (cherry-pick, patch-id e129442b verified vs develop squash). Human approval for prod promote given in-session (option text: "Ships as a reviewed PR, then surgical promote to prod").
**#838 opened (fix/backtest-partial-exit-units)**: 5 root causes + 2 more family members fixed (backtest partials ignored fee/slippage config; LIVE scale-ins applied policy fractions as balance deltas — live partial ops behind OFF flag, no prod money misbooked, ledger reconciliation consistent). Corrected kelly_momentum 30d: **+0.02%** (was +16.67% fabricated), win rate 66.7%, MTM drawdown. Issues #839 (units bug family), #840 (Kelly record_trade never called). Cross-engine parity tests incl. max-position scale-in clamp (arch-review P1 closed).

---

## 2026-07-03 · deploy+charter · daemon(PM)
**#835 LIVE IN PRODUCTION (deploy 21:32:50Z, verified 21:36Z):** startup banner confirms `Max Position Size: 20.0%`; balance $84.40 recovered from active session #20 (session reused across restarts — poll criterion "new session row" was wrong, corrected here for future monitors); SHORT @1696.83 re-adopted with SL order 48096847808 tracked; reconciliation 0 critical; trading loop running. HyperGrowth now sizes at base_fraction 0.25 → realized risk/trade ~1.15–2.0% (hard ceiling 2.0% = cap 0.20 × stop 0.10).
**Charter v0.1 FILLED by Board (develop 6c2f0f45):** mission = grow live account; autonomy expanded — live-capital changes, model promotion, prod deploys now daemon-autonomous; human approval only for charter changes, kill-switch, self-classified-irreversible, >$50/24h inference. Constraints noted: no prod deploys Fri >18:00 UTC or immediately pre-macro-event (tonight's 21:32Z deploy predates charter landing; monitored closely, healthy). KPIs: preservation → parity 15% → Sharpe 1.5/0.5 → win 55/45 → DD <15/20 → cost/decision <$0.50.
**Event-window protection scheduled (human-approved "both windows"):** desktop scheduled tasks fomc-pause-on (Jul 7 18:00Z), fomc-pause-off (Jul 8 20:00Z), cpi-pause-on (Jul 13 12:30Z), cpi-pause-off (Jul 14 15:00Z) — each flips FEATURE_ENTRY_PAUSE via railway, verifies health, reports to PM session. daily-trading-standup cron 09:01 local. NOTE: tasks run only while the Claude app is open — flagged to human.
**#838 architecture review: APPROVE, no blockers** (2× P3: deprecate dead convert_exit_fraction_to_current; cosmetic level-index log parity). Code review + CI pending → merge to develop when green.

---

## 2026-07-04 · merge+dispatch · daemon(PM)
**#838 MERGED to develop (squash 3ef34ade)** after full review cycle: arch APPROVE (no blockers), code review REQUEST CHANGES → P1 fixed in 5f9c0c99 (live DB partial-exit/scale-in persistence was still fraction-of-original units while runtime moved to balance-fraction — would have corrupted crash recovery once live_partial_operations enabled; DB updaters now take executed/added_size_delta in Position.current_size units, regression-tested against a real in-memory DB incl. cap-clamped scale-in, long+short). P3s: dead convert_exit_fraction_to_current deleted repo-wide; RiskManager partial/scale param names re-unitized. CI green on final head. Backtest engine on develop is now trusted for research.
**Dispatched:** (1) fix/kelly-sizer-trade-feedback (#840) — wire closed-trade outcomes into Kelly sizer via a shared seam, both engines, parity + cold-start→warm tests; (2) tournament-v2 rerun on corrected engine (sequential, --max-position-size 0.20 prod-matched) → docs/research/experiments/2026-07-04_tournament-v2-corrected.md; will decide keep-HyperGrowth vs paper-trial challenger on honest numbers.

## 2026-07-04 11:15 · track-record · quant-researcher
Experiment #842: rerun 2026-07-03 tournament on corrected engine (develop@3ef34ade, post-#838) → hypothesis (a challenger beats HyperGrowth on corrected numbers) REJECTED.
Evidence: docs/research/experiments/2026-07-04_tournament-v2-corrected.md
Result: all prior "winners" (kelly_momentum +19.25%/+16.67%, momentum_leverage +20.95%/+14.94%) collapse to ~0% (-0.03% to +0.21%) once units bug fixed — confirms PM's independent verification and risk-officer's 2026-07-03 kelly REJECT verdict. HyperGrowth corrected: -3.29% ETHUSDT/90d, +1.22% ETHUSDT/30d, -1.98% BTCUSDT/90d — negative 90d windows are new information (not visible in the void'd +10.66%/+1.46% fabricated numbers) but consistent with live's short favorable win-streak sample. All 4 challengers were sized 0.3-2.1% of balance vs HyperGrowth's 12-20%, so their true edge is untestable at this notional — not proven bad, just too small to see. Recommendation to pm: keep HyperGrowth as sole live strategy; no challenger earns a paper trial; re-open after #840 (Kelly sizer wiring) merges, or with a sizing-matched rerun. Flagged again: risk-limits.json max_position_size_pct (0.10) still diverges from prod's 0.20, still unreconciled since first flagged 2026-07-03.
