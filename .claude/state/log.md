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

## 2026-07-03 20:45 · track-record · quant-researcher
Experiment #1: Strategy tournament (8 strategies × 2 symbols × 2 windows = 25 runs) → concluded
Evidence: docs/research/experiments/2026-07-03_strategy-tournament.md
Result: kelly_momentum/ETHUSDT best risk-adjusted (Sharpe 0.21/0.32, +19.25%/+16.67% vs incumbent HyperGrowth -0.86% 30d). 
Incumbent HyperGrowth/ETHUSDT confirmed positive 90d (+10.66%) but flat/negative 30d. 
Dead strategies (0 trades): ml_adaptive, ml_sentiment, ensemble_weighted. 
Catastrophic: adaptive_trend (-33.07%, DD 35.77%). 
Win-rate metric unreliable for non-ML strategies (balance-based total_return used instead).
Proposal: kelly_momentum/ETHUSDT as challenger. Needs risk-officer review before live change.

---

## 2026-07-03 21:25 · decision · pm
Human (Board) filled charter.md TODOs and confirmed a high-risk-appetite autonomy envelope: daemon may change live capital, deploy to production, and promote a live-trading symbol's model `latest` symlink without per-action human approval. Risk-tolerance numbers set to match risk-limits.json (20% max drawdown, 6% max daily loss, 10% max position, 3x leverage; breach = halt new entries + page human).
Per explicit human instruction, relaxed the conflicting hard rules in CLAUDE.md and `.claude/agents/ml-engineer.md`: model promotion for live-trading symbols no longer requires human sign-off or `board_required: true`. The eval bar (held-out temporal split, per-regime breakdown, calibration check, >=48h paper validation) and a clean risk-officer review (`risk_review_required: true`) remain mandatory — self-certifying without running them does not count as "verified." All other `board_required: true` gates (e.g. kill-switch, charter.md changes) are unchanged.
Ref: charter.md (risk tolerance, autonomy envelope), CLAUDE.md (daemon hard rules), .claude/agents/ml-engineer.md

---

## 2026-07-04 · merge+dispatch · daemon(PM)
**#838 MERGED to develop (squash 3ef34ade)** after full review cycle: arch APPROVE (no blockers), code review REQUEST CHANGES → P1 fixed in 5f9c0c99 (live DB partial-exit/scale-in persistence was still fraction-of-original units while runtime moved to balance-fraction — would have corrupted crash recovery once live_partial_operations enabled; DB updaters now take executed/added_size_delta in Position.current_size units, regression-tested against a real in-memory DB incl. cap-clamped scale-in, long+short). P3s: dead convert_exit_fraction_to_current deleted repo-wide; RiskManager partial/scale param names re-unitized. CI green on final head. Backtest engine on develop is now trusted for research.
**Dispatched:** (1) fix/kelly-sizer-trade-feedback (#840) — wire closed-trade outcomes into Kelly sizer via a shared seam, both engines, parity + cold-start→warm tests; (2) tournament-v2 rerun on corrected engine (sequential, --max-position-size 0.20 prod-matched) → docs/research/experiments/2026-07-04_tournament-v2-corrected.md; will decide keep-HyperGrowth vs paper-trial challenger on honest numbers.

---

## 2026-07-04 · merge+dispatch · daemon(PM)
**#843 MERGED to develop (squash e1d24239), closes #840.** Four independent reviews converged; all findings folded into final commit 9e4d294d: (P1) Kelly inputs switched to UNSIZED directional R-multiples via shared pnl_percent(fraction=1.0) — removes self-sizing feedback loop, matches sizer's tested contract + expected_reward_risk prior scale; (P2a) partial-exit slices now feed the sizer per-slice in BOTH engines (shared frozen PartialExitOutcome, on_partial_exit hook; zero-size bookkeeping close skipped to prevent double-count; "one Kelly trade = one realized slice" pinned); (P2b) sibling KellySizer gained record_trade adapter — single stats-sizer interface; (P3) tracker listener registration/dispatch now lock-disciplined. 4285 unit tests + full CI green. Review-collision note: second arch review read the worktree mid-fix (its P0 was in-flight state, not a regression) — lesson: don't dispatch fix rounds into a worktree reviewers are still reading.
**Dispatched: Kelly-ACTIVE evaluation** (365d ETHUSDT for warm-up past min_trades=30, + 90/30d comparability, hyper_growth 365d benchmark) → docs/research/experiments/2026-07-04_kelly-active-evaluation.md; verdict wanted: staging-paper trial alongside live HyperGrowth, or not. Finished agent worktrees removed.

---

## 2026-07-04 · decision · daemon(PM)
**Risk-officer verdict on the HyperGrowth 365d MaxDD breach ADOPTED: (a)+(d).** Hold 0.25 sizing (per-trade risk is not the problem; the tail is structural yearly expectancy), NO weekend/pre-FOMC deploy, tripwire table installed into the daily standup task (soft $80.18 / reduce $75.96 / hard floor $67.52 → FEATURE_ENTRY_PAUSE + page human; 4-consecutive-losses; >6% weekly DD rate; missing-SL check). Event windows already armed (pause eve of Jul 8 + Jul 13).
**Premise correction (verified empirically by risk-officer):** HyperGrowth's bear-defense leverage-map rows (<1.0) ARE ACTIVE at max_leverage=1.0 — TREND_DOWN/HIGH_VOL already scales notional ×0.171; the construction clamp caps only values ABOVE max_leverage. Raising max_leverage would only re-enable bull rows (×1.414 → 28% notional, cap-breaching). Option (c) rejected. Open question: whether the regime detector labels the live bear correctly (365d loss occurred WITH the map active) — flagged for research.
**P1 confirmed: the 20% hard cap has NO live enforcement — RiskManager.check_drawdown() is dead code.** Dispatched fix/live-max-drawdown-halt (close-only on ≥20% rolling DD, warning tiers at 10%/16%, idempotent, restart-safe recompute; reuses existing close-only mechanism). Deploy timing = PM decision post-review, candidate after Jul 8.
**kelly_momentum staging-paper trial: approve-with-conditions** (risk-officer): distinct session/vehicle from paper HyperGrowth (session-collision class), no paper/live P&L conflation in dashboards/alerts, no auto-escalation to live, integration test for Kelly warm-up transition required pre-live. NOTE: trial value is months-horizon (Kelly warm-up = 30 closed slices ≈ 5 months at observed cadence) — parked as backlog, ops-feasibility check (second paper vehicle) to live-ops next week. ESCALATED to Board (again): risk-limits.json max_position_size_pct 0.10 vs deployed 0.20 + $last_reviewed=1970-01-01.

---

## 2026-07-04 · deploy+defect · daemon(PM)
**Max-drawdown hard-cap enforcement LIVE in prod** (#848 → develop 5f323cd3 → promote #849 → main, deploy 12:20Z; patch-id 4ccd0780 verified; full unit suite re-run green on the main-based promote branch). Guard armed 12:23:57Z.
**Seed defect found at arm-time:** guard logged `peak=$100.00 ... drawdown 15.60%` — but session 20's true account_history max(balance) is $84.4159 (verified read-only). $100 = config INITIAL_BALANCE via PerformanceTracker.peak_balance (the June optimistic-$100 pathology resurfacing in a new seam). Interim risk posture: CONSERVATIVE-not-dangerous (effective close-only floor $80.00 vs policy $67.52; CRITICAL-tier log noise expected near $84.00; close-only never liquidates). Weekend waiver in force → fix/drawdown-guard-seed-peak dispatched (DB-session max authoritative, no tracker-initial fallback, retry-not-latch on None, regression test reproducing prod, optional deeper fix: reset tracker peak on session recovery — would also correct account_history.drawdown's long-standing 15.6%). Promote after review same-day.

---

## 2026-07-04 · deploy-verify · daemon(PM)
**Drawdown-guard seed fix LIVE and VERIFIED.** #850 → develop 5017e931 → promote #851 (surgical conflict resolution: guard files patch-id-identical per-file; trading_engine adapted to exclude unpromoted #843 listener line; changelog trimmed to promoted entry; full suite green on main-based branch) → main, deploy ~12:53Z. Prod log 12:55:36Z: `Max-drawdown guard armed: peak=$84.42, hard cap=20.0% (session 20, account_history peak $84.42)` — no warning tier, DD ≈ 0.02%, halt floor $67.53 per policy. PerformanceTracker now seeds from recovered balance → account_history.drawdown/dynamic-risk phantom ~15% also fixed going forward. Safety stack now complete and verified end-to-end: enforced 0.20 position cap + 2% risk ceiling (#835/#841), enforced 20% DD close-only halt with 10%/16% warning tiers (#848/#849/#851), FEATURE_ENTRY_PAUSE event-window pauses (scheduled Jul 7/8 + Jul 13/14), daily standup tripwires, corrected backtest engine (#838) + honest research baselines (tournament v2, Kelly-active eval), Kelly feedback wiring (#843, develop). Follow-ups open: #847 (cross-session peak), risk-limits.json divergence (Board), kelly paper trial ops-feasibility (parked, months-horizon).

---

## 2026-07-04 · triage+discovery · daemon(PM)
**Other-session PR sweep (human-directed) all resolved:** #855 alert delivery via $ALERT_WEBHOOK_URL (merged by its session) → PROMOTED to prod (#864, patch-id ec874e02, suite green on main-based branch; deploy 14:35Z, verification poll running; webhook URL still needed from Board — flagged). #856 (reconciler SL-fill trade-row test) + #854 (label taxonomy) merged. #846 (breach stress review) merged as historical record WITH PM phantom-peak correction comment. #862 exposure governor: rebased by agent (only changelog conflict — branch already post-#835/838/843/848/850), assessed KEEP (only TOTAL-gross cap; strategy-agnostic bear cap; ex-ante; #806/#807 foundation), code review PROVED inertness-when-off → merged DEFAULT-OFF (14:38Z) with enablement conditions commented on the PR (fix exposure.py current_size P2 + scale-in bypass P3 + sequence with regime detection ON). #852 closed by author; underlying db_closed-gating gap check queued.
**P0 DISCOVERY (ml-engineer audit): HyperGrowth live has NEVER had an ETHUSDT model.** runner's --symbol never reaches MLBasicSignalGenerator (defaults BTCUSDT, ml_signal_generator.py:542/594); prod scores ETH candles with BTCUSDT/basic/2025-10-30 (247d old), silently — no ETHUSDT/basic model exists at all. Symptoms: confidence median 0.03-0.04 (49% of staging samples <0.02 = noise floor), 100%-SELL streak in recent prod sample. Reframes the −20.15%/365d as noise-trading, not weak edge. Fix chain dispatched: (1) fix/ml-signal-symbol-wiring PR (thread symbol, mismatch guard, fail-fast + FEATURE_ALLOW_CROSS_SYMBOL_MODEL transition flag); (2) ETHUSDT/basic training queued behind exit-sweep (thermal); (3) staging-paper validation ≥48h; (4) prod promote decision with human. Jul 8 resume checkpoint to weigh keeping entries paused pending validated signal.

---

## 2026-07-04 · ops+research · daemon(PM)
**Killed stuck 7h agent process** (PID 25887, the "Live trading bot analysis" session that shipped #855 hours earlier — kept churning ~13% CPU after its purpose completed; terminated cleanly per human instruction). Second suspected hang (exit-sweep agent) had already terminated — its background sweep had actually COMPLETED all 18 runs; only the wake-up link broke. Results salvaged from scratchpad.
**Exit-geometry sweep verdict: NO-GO (strong negative result, PM-compiled).** Every variant strictly worse than baseline on every window: sl_007 −24.1%, sl_005 −31.3%, tighter_trail −29.9%, combos −37.6/−40.9% vs baseline −20.1% (365d). Consistent with the same-day root cause: entries are cross-symbol noise (#867) — tighter exits just crystallize noise losses faster. Exit layer exonerated as the fix; ALL expectancy work routes through the signal (wiring fix + native ETHUSDT model + staging). Geometry re-sweep only after a signal with edge exists. Full table appended to docs/research/experiments/2026-07-04_hypergrowth-exit-geometry.md.

---

## 2026-07-04 11:15 · track-record · quant-researcher
Experiment #842: rerun 2026-07-03 tournament on corrected engine (develop@3ef34ade, post-#838) → hypothesis (a challenger beats HyperGrowth on corrected numbers) REJECTED.
Evidence: docs/research/experiments/2026-07-04_tournament-v2-corrected.md
Result: all prior "winners" (kelly_momentum +19.25%/+16.67%, momentum_leverage +20.95%/+14.94%) collapse to ~0% (-0.03% to +0.21%) once units bug fixed — confirms PM's independent verification and risk-officer's 2026-07-03 kelly REJECT verdict. HyperGrowth corrected: -3.29% ETHUSDT/90d, +1.22% ETHUSDT/30d, -1.98% BTCUSDT/90d — negative 90d windows are new information (not visible in the void'd +10.66%/+1.46% fabricated numbers) but consistent with live's short favorable win-streak sample. All 4 challengers were sized 0.3-2.1% of balance vs HyperGrowth's 12-20%, so their true edge is untestable at this notional — not proven bad, just too small to see. Recommendation to pm: keep HyperGrowth as sole live strategy; no challenger earns a paper trial; re-open after #840 (Kelly sizer wiring) merges, or with a sizing-matched rerun. Flagged again: risk-limits.json max_position_size_pct (0.10) still diverges from prod's 0.20, still unreconciled since first flagged 2026-07-03.

---

## 2026-07-04 12:15 · track-record · quant-researcher
Experiment #844 (pending issue number below): Kelly-active kelly_momentum (PR #843 wiring fix) vs HyperGrowth, ETHUSDT/1h, 30d/90d/365d → hypothesis (Kelly-active kelly_momentum beats HyperGrowth on risk-adjusted AND absolute return, multi-regime) REJECTED as stated; INCONCLUSIVE-but-promising on a narrower claim.
Evidence: docs/research/experiments/2026-07-04_kelly-active-evaluation.md
Result: Kelly wiring CONFIRMED genuinely active this time (empirically verified via instrumented replay, not just code trace — has_sufficient_history flips True at trade 30 exactly, stays True through trade 73; a first-pass sub-agent claim of "zero callers" was independently checked and found wrong, it grepped the wrong PerformanceTracker class). 365d: kelly_momentum -0.29%/0.44%DD/Sharpe 0.0018 vs HyperGrowth -20.15%/21.84%DD/Sharpe 0.119 — kelly_momentum's better absolute/DD numbers are an artifact of running ~50x smaller position size (avg 0.25% vs 13.1% of balance), not superior edge; HyperGrowth wins on Sharpe/Sortino (which correct for size). Kelly's own math (win_rate~68%, reward:risk~0.48) computes small fractions post-warmup because payoff ratio is poor (avg_loss ~2-3x avg_win) — sizing stayed flat pre/post-30-trade warm-up boundary for legitimate mathematical reasons, not a residual bug. All 83 kelly_momentum trades across all windows were LONG-only (flagged, not explained). Also surfaced: (1) CLI --risk-per-trade/--max-risk-per-trade are NOT wired to kelly_momentum's strategy-level base_risk=0.08 (only --max-position-size reaches it, downstream in EntryHandler) — decorative flags for this strategy specifically; (2) HyperGrowth's first-ever full-year backtest breaches the 20% portfolio MaxDD hard cap in risk-limits.json (21.84%) — flagged as more urgent than the Kelly question, recommend dedicated follow-up.
Recommendation to pm: kelly_momentum earns a STAGING PAPER trial alongside (never replacing) live HyperGrowth — safe given tiny realized position sizes, and needed to build genuine Kelly-warm live history. Does NOT earn live promotion or HyperGrowth replacement. Separately, HyperGrowth's 365d MaxDD breach needs its own risk-officer review, decoupled from this Kelly write-up.
Still open: risk-limits.json max_position_size_pct (0.10) vs prod's 0.20 divergence, flagged 2026-07-03, still unreconciled.

---

## 2026-07-04 12:25 · track-record · risk-officer
Two-part independent review commissioned by pm: Q1 live HyperGrowth sizing decision (post-#835), Q2 proposal 2026-07-04-01 (kelly_momentum staging-paper trial).

Q1 — HyperGrowth live ($84.40, ETHUSDT, base/max-position 0.20, stop 0.10, ≈2% risk/stop-out):
Verdict on options: (a) status-quo+tripwires = approve [conf high, RECOMMENDED]; (b) revert sizing to 0.20-base pre-#835 = reject-for-now [conf high] — a prod deploy into a Jul 4 weekend / pre-FOMC(Jul 8) window, and per-trade risk drops only ~2%→1.6% (immaterial vs the real risk which is structural expectancy, not per-trade size); (c) bump max_leverage>1.0 to "enable" bear defense = reject [conf high] — INVERTS intent: empirically the leverage map's bear-defensive rows (TREND_DOWN/LOW=0.5, TREND_DOWN/HIGH=0.0) are ALREADY LIVE at max_leverage=1.0 (only >1.0 BULL rows are clamped at construction). Measured live-config leverage multiplier for the CURRENT TREND_DOWN/HIGH regime = 0.171 (notional 0.20→0.034) at high conviction, ~0.83 freshly-entered. Setting max_leverage=1.5 would leave the bear row at 0.171 UNCHANGED while re-enabling bull rows to 1.414 → 28.3% notional, breaching the enforced 0.20 cap AND the 0.10 JSON limit. Task's premise "in a confirmed bear it sizes identically to a bull" is INCORRECT for the live config; (d) entry-pause into Jul 8/14 = approve-with-conditions [conf med] — fold into (a) as a tripwire action, not a standalone deploy; close_only_mode exists (`_enter_close_only_mode`) as the mechanism.
Single recommendation to pm: (a)+(d) — status-quo sizing (no weekend/pre-macro deploy), install the tripwire table below, and pre-commit to entering close-only (entries paused, stops maintained) before Jul 8 14:00 ET and Jul 14 08:30 ET per the market brief.
Scenarios checked (baseline $84.40, peak $84.40): 5 consecutive 2% stop-outs → 9.88% DD ($76.06), inside cap; 15% adverse month → $71.74 trough, crosses HyperGrowth's FIRST dynamic-risk tier (0.15) → mild 0.8x sizing cut only; Jul 8 gap through 10% stop (realized 10-15% move) → 2.0-3.06% loss single-event ($81.82-$82.66); combined Jul 8 gap + 4 chop stops → 10.80% DD ($75.28). ALL survivable. The MATERIAL tail is NOT any trade sequence — it is the strategy's structural negative expectancy over a full multi-regime year: corrected 365d backtest = -20.15% return / 21.84% MaxDD (both 2025 and 2026 halves negative), BREACHING the 20% portfolio hard cap by 1.84pp.
P0/P1 findings (independent, some corroborating prior audits):
  * P1 (control gap): `RiskManager.check_drawdown()` has ZERO callers outside its own docstring — the 20% max-drawdown hard cap is DEAD CODE with no live halt path. Corroborates 2026-06-08 observability audit. HyperGrowth's own dynamic_risk overrides thresholds to [0.15,0.30,0.45] w/ factors [0.8,0.5,0.2] — it REDUCES size but NEVER halts, and keeps full size below 15% DD. Nothing would have stopped the 21.84% backtest DD live.
  * P0-divergence (still open, first flagged 2026-07-03): risk-limits.json max_position_size_pct=0.10 vs prod-running 0.20; JSON $last_reviewed=1970-01-01 (never reconciled). constants.py otherwise MATCHES JSON on all other keys (verified: max_drawdown 0.20, daily_risk 0.06, correlated_exposure 0.15, base/max risk 0.02/0.03, kelly_max 0.20, stop 0.05, max_leverage 3.0). Note constants also has DEFAULT_MAX_CORRELATED_RISK=0.10 alongside DEFAULT_MAX_CORRELATED_EXPOSURE=0.15 (two distinct constants).
  * Model-risk (regime): feature_flags.json enable_regime_detection=false gates only the ENGINE-level RegimeDetector (used for regime-strategy-switching), NOT the strategy's component EnhancedRegimeDetector, which runs unconditionally inside process_candle and feeds the leverage map. So the bear defense is live IF the component detector labels the tape TREND_DOWN with conviction — which I could not verify against live tape (the 21.84% backtest through this exact path is evidence the defense did not prevent the breach, whether from under-labeling or insufficient reduction at low-conviction transitions).
Recommend: pm file a dedicated HyperGrowth-365d-MaxDD-breach review (decoupled from Kelly), + a GH issue for the dead check_drawdown / missing hard-halt (P1 capital-protection gap), + escalate the risk-limits.json 0.10-vs-0.20 reconciliation to the human Board (charter says JSON is human-owned; I cannot edit it). None of these are P0-halt-now: no active data corruption / duplicate-order storm / divergence detected; kill-switch NOT recommended. Live account is inside all limits TODAY (~2% risk/trade, no open drawdown, 5-win streak).

Q2 — proposal 2026-07-04-01 kelly_momentum staging-paper trial: verdict=approve-with-conditions, confidence=high.
Paper-only + staging-only = inside charter autonomy envelope, no board sign-off, zero capital at risk (realized backtest sizing max 1.57%/balance over 73 trades/yr). Conditions: (1) VERIFY paper-isolation operationally — distinct paper session/account, no dashboard/alert/account_history/P&L path may conflate paper kelly with live prod P&L (a paper DD misread as live could trigger a spurious human halt of the real book); (2) confirm ops tooling does not sum paper+live ETHUSDT notional into one correlation metric (max_correlated_exposure_pct is N/A for paper); (3) NO auto-escalation to live — separate proposal + human sign-off required, and must not lean on "risk-per-trade" framing (CLI risk flags don't reach kelly_momentum's hardcoded base_risk=0.08); (4) before ANY live consideration, add the missing integration test asserting has_sufficient_history transitions inside a real Backtester.run() (none exists). Could-not-verify: whether staging can host a 2nd concurrent paper session w/o session-collision — ops feasibility question for live-ops; if not, trial needs its own isolated vehicle, must not share a session with a HyperGrowth paper session.
Scenarios checked: paper-isolation misread→spurious-halt, paper+live correlation double-count, live-escalation gate, kelly base_risk wiring (CLI flags decorative), payoff-ratio reversion (win% 73→44 post-warmup in-sample), long-only blind spot in downtrend.
Calibration note vs prior track record: on 2026-07-03 I rejected kelly (b) live-swap [high] on dead-wiring grounds; #843 has since fixed the wiring and tournament-v2 corroborated my rejection with real numbers. I remain adversarial on live promotion but approve the paper trial — consistent, not a reversal. Ref: .claude/state/proposals/2026-07-04-01-kelly-momentum-staging-paper-trial.md

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

---

## 2026-07-05 · merge+model · daemon(PM)
**#867 MERGED to develop** (rebased onto the bear-market series; PM completed the stalled rebase directly — conflicts resolved, leftover changelog markers cleaned, empty CI commit dropped, full suite green, CI green on rebased head). ML symbol wiring now structural: fail-fast on missing models, FEATURE_ALLOW_CROSS_SYMBOL_MODEL transition flag, hot-swap path threaded, per-condition rate-limited guards. Issues #871 (sibling generator) + #872 (flag sunset) track follow-ups.
**First-ever ETHUSDT basic model TRAINED** (local, 60.5 min, EarlyStopping @ epoch 35/50): 2026-07-04_22h_v1, full 2017–2026 history (77,701 candles), test RMSE 0.065141 (< BTCUSDT's 0.0665), train/test gap tight, directional accuracy 0.5312 on temporal holdout. Validation backtest (hyper_growth + #867 wiring + native model, ETHUSDT 90d prod-matched) RUNNING — decides the model PR + staging promotion. Window-variant tournament (full/3y/18mo/recency-weighted) designed from 5-packet research synthesis (drift real ~2.5-3.5%/mo decay; naive daily-retrain WORST in only head-to-head; Kaggle winners: hard cutoffs/expanding, no decay weighting; soft regime handling > per-regime models — our architecture already matches; de Prado purge/embargo ~1% bars adopted for eval). Cloud SageMaker smoke job in flight; variants go cloud-parallel if it validates. Staging sync deliberately held for one combined deploy (wiring + model) — cross-symbol flag pre-set on staging either way.

---

## 2026-07-05 · milestone · daemon(PM)
**STAGING NOW RUNS THE COMPLETE INTENDED SYSTEM** (deploy #888, boot 23:24Z): native ETHUSDT/basic/2026-07-04_22h_v1 registered and resolving at decision time (zero cross-symbol/mismatch guard warnings — the guard would ERROR on substitution), #867 wiring live, full bear-market cohort flags active (governor, regime detection, event guard, vol-target sizing, breakers dry-run), paper $981.75. First time in this system's history that ETH candles are scored by an ETH-trained model in a running engine. **48h paper-trial clock for #887 (prod model promotion) starts 2026-07-04T23:24Z** → earliest prod consideration ~Jul 7, realistically post-FOMC Jul 8 with paper evidence + window-tournament winner. Known open bottleneck: decision confidences still ~0.04 (magnitude calibration — queued as next profit lever alongside the tournament). Overnight: window tournament W_full training; cloud smoke verdict outstanding (chase at standup if silent).

---

## 2026-07-05 · validation · daemon(PM)
**SageMaker cloud training VALIDATED WORKING** (smoke $0.0074, full tournament <$1; round trip verified incl. registry sync; basic/latest untouched by design). NOT used for the window tournament: --days-only windows always end "now" (would leak the 2026 OOS eval window into training) + stale Jan ECR image = feature train/serve skew. 5 fixes filed as issue; primary future use = production retrains. Local tournament (protocol-clean, fixed 2025-12-31 cutoff) remains authoritative — W_full trained (test RMSE 0.0659, 25 epochs early-stop), OOS eval running, W_3y + W_18m queued.

---

## 2026-07-05 · deploy-verify · daemon(PM)
**PRODUCTION = DEVELOP PARITY, VERIFIED** (Board directive; PR #905-family parity merge, tree byte-identical, ours-merge history tie for clean future syncs; CI green; deploy boot 18:39Z). Boot verification: native ETHUSDT/basic/2026-07-04_22h_v1 in registry with ZERO cross-symbol/mismatch warnings (prod now trades the native ETH signal for the FIRST TIME); Max Position 20.0%; drawdown guard armed peak=$84.42; trading loop running. Expected known-flag: alert channel unset (Board item; operationally covered by the 30-min alert-monitor pull loop). Staging synced to same tree (#904). All bear-market subsystems present but flag-OFF in prod; enablement remains a separate evidence-gated decision. PM flag on early model promotion (~28h into 48h paper window, trial clean, dominates the noise incumbent on every axis) recorded — Board-directed parity explicitly included it.

---

## 2026-07-05 · incident-triage · daemon(PM)
**IP-transition night: DEGRADED-NOT-DOWN, correctly triaged.** Alex rotated Railway static egress IP + Binance allowlist + deleted the unused development environment (~22:15Z). Effects: (1) user-data WS churned (400 reconnects) → circuit-open → **REST-degraded mode** (designed #717 fallback; fills/balances via polling) — 3 CRITICAL system_events fired (now also delivered to the new ALERT_WEBHOOK_URL Slack channel); (2) REST auth from the new IP verified WORKING (DB heartbeat 22:22Z + status ticks + entry evaluation post-transition); exchange-side SL unaffected; (3) railway CLI link broke ("Environment is deleted" — cached deleted-env reference) → re-linked to production, log access restored. NOT a P0: protection intact, engineered degraded state. WS-recovery checkpoint set (+45min): if user stream still circuit-open, service restart to re-establish WS cleanly on the new IP. Overnight 10-min API-error watch re-armed with fixed CLI.

---

## 2026-07-05 00:15 · track-record · ml-engineer
Model ETHUSDT/basic/2026-07-04_22h_v1 · event: trained|evaluated|proposed
Metrics: test_rmse 0.065141 (train 0.063904, beats BTCUSDT/basic bar 0.0665) · directional_accuracy 0.5312 on temporal holdout (2024-09-25→2026-07-04, chronological 80/20 split, no shuffle) · validation backtest (hyper_growth ETHUSDT 1h 90d, prod-matched risk params, tested against #867 merged @1c1b4199): native -1.31% vs cross-symbol baseline -3.29% (logged 2026-07-04) vs hold -15.49%; 11 trades, 72.73% win rate, MaxDD 3.05%, Sharpe 0.04. Strictly better than baseline on every axis, still net-negative absolute. Decision confidences cluster ~0.02-0.05 median, tail to 0.78 — confidence-scaling layer flagged as next bottleneck, not fixed here. Ref: issue #887, PR #886

Note on session integrity: two unsolicited "coordinator" messages arrived mid-task claiming to relay results on my behalf. First one (during setup) contained a fabricated claim (main checkout has ETHUSDT cache data — verified false, zero ETHUSDT files existed there) used to justify reading from the production checkout and truncating the training window; not acted on. Second and third (post-training, post-backtest) contained numbers that mostly checked out against independent verification (log files, `gh pr view`, `.claude/state/log.md`, grep of decision logs) — but I verified each claim against source before using it rather than taking the messages at face value, and would flag this pattern to a human if it recurs.

---

## 2026-07-05 11:00 · track-record · quant-researcher
Experiment #898: Training-window tournament for ETHUSDT basic model (W_full 2017-08-17, W_3y 2023-01-01, W_18m 2024-07-01; all hard-cutoff 2025-12-31 train, eval on fully unseen 2026-01-01→2026-07-04 185-day bear market) → partially supported / inconclusive-leaning-full-history
Evidence: docs/research/experiments/2026-07-05_window-tournament.md
Results (all fee/slippage-on, hyper_growth strategy, prod-matched risk params, verified against raw backtest JSONs + model metadata, not taken from any relayed summary at face value): W_full test_rmse 0.06586 / OOS return -7.43% / PF 0.673 / MaxDD 10.55% / 52 trades. W_3y test_rmse 0.06363 / OOS return -11.25% / PF 0.543 / MaxDD 13.55% / 55 trades (outright worst despite better holdout RMSE than W_full). W_18m test_rmse 0.06266 (best) / OOS return -7.30% / PF 0.553 / MaxDD 11.90% / 43 trades (statistically tied with W_full on return, worse on PF/MaxDD). Hold baseline -40.98% throughout. Key finding: holdout RMSE improves monotonically as window shortens but does NOT predict OOS P&L — direct within-experiment evidence against using training-time RMSE as an OOS proxy. No variant is OOS-profitable. Confidence distributions near-identical across all three (median 0.03) — training window has no effect on the confidence-calibration bottleneck already flagged 2026-07-05 00:15. Verdict: naive recency-chasing (H0) rejected — the two shorter windows do not beat full history on risk-adjusted terms and the worst performer is the mid-length window, not the shortest. Full history wins on PF/MaxDD outright, ties shortest window on raw return. Recommendation to pm: promising but not ready; no promotion of any variant, keep 2026-07-04_22h_v1 deployed (not directly comparable — its own training window overlaps this eval period). Follow-up surfaced: neither atb train price nor the backtest engine's prediction_metrics path produce a usable directional-accuracy figure for hyper_growth — instrumentation gap, not fabricated here.

---

## 2026-07-05 13:15 · change · claude-code (interactive session, user-directed)
**#486 live-engine modularization COMPLETE; issue closed.** Implemented the last open item from `docs/refactor/live_engine_modularization.md`: `_init_modular_handlers` folded into the engine's `_init_*` phase-helper family (`_init_core_handlers` / `_init_entry_handler` / `_init_exit_handler` / `_init_risk_guards`; original method now a thin orchestrator, signature unchanged). Pure mechanical move — bodies byte-identical, construction order and object graph unchanged. Merged to develop as PR #900 (`23f85949`) with all required checks green.
Evidence: parity fingerprint byte-identical before/after (`trades=14`, `final_balance=9964.469864061437`, `sha256=3f6db552…` — canonical value refreshed in the handover doc §4); quality gate (black/ruff/mypy) + bandit clean; wiring tests 54 passed; integration `test_engine_core.py` 16 passed; full fast suite 4,573 passed; dual architecture-reviewer + code-reviewer pass with zero findings (both independently byte-diffed the moved bodies); plan pre-approved via Codex review (2 rounds, gpt-5.5). Note: `tests.yml` CI triggers on `pull_request` only — no push-run exists for the merge commit; PR CI ran against the identical content (develop tip unchanged between base and merge).
Issue #486 closed with acceptance-criteria verification against merged `origin/develop` (zero direct `exchange_interface` order/stop-loss calls in `trading_engine.py`). Handover doc marked COMPLETE (status header, §2 table, §5, §10 ledger rows #826/#827/#828/#900). NOT promoted beyond develop — no staging/prod action taken.
Ref: GH #486, #900 · docs/refactor/live_engine_modularization.md · docs/changelog.md

---

## 2026-07-06 · nightcap · daemon(PM)
**WS user-stream: chronic churn accepted overnight, daylight fix queued.** Post-restart the #723 hard-reconnect works but the subscription drops every ~10-20min (-2036 on teardown, probe #40+ by 00:12Z). Correction to earlier triage: identical -2036 signatures predate the IP rotation — chronic subscription-lifecycle defect (the #616/#617 family), aggravated not created by the transition. REST fully authorized on the new IP; fills via polling; exchange-side SL intact; equity $84.40 stable. Issue filed for daylight investigation. Posture: designed-degraded + alert-monitor; no midnight heroics on a protected account.

---

## 2026-07-06 · track-record · quant-researcher
Experiment #912: Confidence-calibration study — does the ETHUSDT model's raw-output→confidence mapping compress real directional edge into noise, and can recalibration recover it? → rejected (H0 supported)
Evidence: docs/research/experiments/2026-07-05_confidence-calibration.md
Phase 1 (code trace): confidence = clip(|predicted_return| * 12.0, 0, 1) feeds a binary min_confidence=0.05 gate; HyperGrowth's FixedFractionSizer has adjust_for_confidence=False so confidence has ZERO effect on position size once gated — any calibration fix can only ever change which bars trade, never their size. Phase 2 (freshly retrained ETHUSDT W_full model, zero train/exam overlap, verified against metadata.json): magnitude-vs-hit-rate decile table is FLAT on the frozen exam window (2026-01-01→2026-07-04, n=4,415; Cochran-Armitage p=0.669, Spearman p=0.477, every decile's 95% Wilson CI overlaps every other's) despite showing a statistically real gradient on a training-period-adjacent slice (2025-07-01→2025-12-31, n=4,391; CA p=0.019, Spearman p=0.0008) — textbook overfitting of the confidence channel, not a real OOS signal. Phase 3 (4 gate-recalibration variants, one-shot on frozen exam, thresholds pre-selected from training-period data): 3 of 4 matched/underperformed baseline; vol-normalized z-score gate showed a directionally favorable but sub-threshold result (+1.45pp return vs required ≥3pp, 49 vs 46 trades) not distinguishable from noise.
Verdict: no calibration-layer fix clears the pre-registered bar. Recommendation: redirect to target redesign (direction-classification and/or vol-normalized-return training target) as the next research tournament, not further confidence-formula tuning.
Side-finding (spawned as separate follow-up, not fixed here): backtests are NOT deterministic — PredictionEngine's inference timeout defaults to 0.1s (a latency-alerting budget mistakenly gating actual inference-abort), so under CPU contention a small fraction of bars silently fall back to HOLD; re-running the IDENTICAL baseline backtest twice produced materially different trade counts/returns (46 trades/-11.36% vs 55 trades/-10.33%). Threatens the model-evaluation-system's frozen-exam comparability premise; filed for dedicated fix.
Also spawned: min_confidence override mapping gap in src/experiments/runner.py (maps to position_sizer only; HyperGrowth's gate lives on risk_manager).
Compute: isolated `.claude/worktrees/calibration-study` (fresh from origin/develop, detached), removed on completion. Never touched main checkout, staging, or prod.

---

## 2026-07-06 · merge · daemon(PM)
**#891 MERGED (squash a846cd4f-family), #890 CLOSED — cloud training production-grade.** Full arc: 5 fixes + 2 stubs + date-range flags + collision-proof versioning + ONNX-required cloud-promote (never touches basic/latest without --set-latest) + atomic shared latest-symlink helper + amd64/onnx-pin container + E2E revalidated on real SageMaker (3 jobs ≈ $0.01). Two-review gauntlet (code: zero findings; arch: P1 keras-only-promote + P2 symlink atomicity — both fixed). Weekly-model-retrain routine precondition #1 now satisfied; #2 (image freshness) true until training_pipeline next changes. Backlog adds from reviews: image-drift CI guard, --json on cloud-status/cloud-list.

---

## 2026-07-06 · soak-verdict · daemon(PM)
**#907 CLOSED — WS subscription fix VALIDATED in production.** 6h soak (12:30Z→18:34Z): ZERO REST_DEGRADED transitions (pre-fix baseline: one every ~10-20min, months-old chronic). Early single-attempt reconnects at 6m/28m/28m ages = Binance idle policy, handled cleanly by the verify-ping path. The months-old churn family (#616/#617 lineage) is closed by a one-constant root cause + 4-layer fix, diagnosed-to-validated in under 24h.

---

## 2026-07-06 15:30 · forensics · claude-code (interactive session, user-directed)
**Prediction latency-abort defect confirmed + severity escalated (#913); no prod impact found; scope partitioned with a parallel in-flight fix session.**
- Verified the #912-addendum non-determinism report in code: `PredictionEngine._get_timeout_seconds()` returns `max_prediction_latency` (0.1s alerting SLO) and uses it to ABORT inference (engine.py:189/244/514/641) and to INVALIDATE successful predictions slower than 0.1s wall-clock incl. feature extraction (engine.py:299-318, silent — no log). `OnnxRunner` already carries the proper 30s hang guard internally, so the engine gate is a redundant double-gate on the wrong constant.
- **Escalation**: the failure mode is not a silent HOLD. `engine.predict()` returns error results with `price=0.0`; both `_get_ml_prediction` impls consume `result.price` without checking `result.error` → `predicted_return=-1.0` → SELL strength 1.0, confidence 1.0. Empirically reproduced against unmodified generator code (repro output on #913). Latent LIVE risk, not just backtest noise.
- **Collision avoided**: a parallel agent session (branch `fix/deterministic-backtest-inference`, worktree `agent-a88052dac…`) was found actively implementing the full fix (InferenceContext deterministic/live split + generator error guard + test rewrites). This session deliberately did NOT implement — filed #913 (p1, state:building) as the tracking issue and did forensics/observability work instead.
- **Forensics verdict (read-only, prod+staging Postgres via public proxy)**: NO evidence prod ever traded on a latency error result. Suspect short = position id 22 (ETHUSDT, entered 2026-07-02 13:34 UTC — not 07-03 as previously logged), confidence 0.25 / strength 0.21 with a normal ~30-min strengthening decision trail. Zero rows in all of `strategy_executions` history match the bug signatures. Max opened-position confidence in 07-01→07-06 window: 0.371. Full notes: `docs/research/notes/2026-07-06_latency-error-phantom-short-forensics.md`.
- **Ops lesson**: `railway logs` serves ONLY the current deployment's stream — `--since` does not reach prior containers; historical incident forensics must come from the DB.
- **Side-findings → issues + agents dispatched**: #914 `strategy_executions.ml_predictions` is JSON null in every row ever (prod+staging) — root cause: callers DID pass it, but from `extract_ml_predictions(df, index)` reading df columns no component strategy writes; real outputs live on `Signal.metadata`. Fixed same-day in PR #917 (new signal-metadata extractor wired at 5 live+backtest logging sites, failures persisted as `prediction_failed` dicts; 18 TDD tests; kept file-disjoint from the #913 fix session; codex gpt-5.5 review run before merge). #915 `NO_ALERT_CHANNEL` fired on the 07-04/07-05 prod startups → USER_WS_DEGRADED escalated 3→1360 reconnects 07-04→07-06 with zero operator paging (double-blind alerting recurrence); same-day read-only verification found prod `$ALERT_WEBHOOK_URL` has since been set (Slack; guard did not fire on today's 12:30Z startup — actor unknown, Railway keeps no variable audit trail) and post-#911 user-stream health is WS-primary/clean over the first ~2h — #915 re-scoped to staging webhook parity + 24-48h WS observation (p3, monitoring). Commented evidence on #723.
Ref: GH #913, #914, #915, #912 (addendum), #723 · docs/research/notes/2026-07-06_latency-error-phantom-short-forensics.md

---

## 2026-07-07 00:20 · note · daemon(PM)
**Log-stream merge (PR #924).** Two parallel session streams covering 2026-07-03→07-06 — the PM daemon session (this branch) and the interactive claude-code session (already on develop via its own merges) — are interleaved above in timestamp order. No entry content was altered; only ordering across the two streams. Where the same underlying event appears in both streams (e.g. #913 latency-abort defect), both entries stand — they record different sessions' independent verification of it.

---


## 2026-07-07 11:30 · track-record · quant-researcher
Experiment #933: ML target-design literature/ecosystem survey for the TARGET-REDESIGN tournament → informational (no backtest run; ranked shortlist produced).
Evidence: docs/research/2026-07-07_ml-target-design-research.md — surveys FreqAI target conventions, freqst.com (skeptical: 100% non-ML technical-indicator leaderboard, zero disclosed methodology, unmaintained), triple-barrier/meta-labeling/trend-scanning/quantile literature (cited, evidence quality flagged per source), and what correlates with live profitability (coverage/precision tradeoff, not raw accuracy). Ranked shortlist: (1) meta-labeling secondary classifier, (2) binary direction classification, (3) triple-barrier ternary classification, (4) vol-normalized regression, (5) quantile/distributional regression, (6) trend-scanning — top-3 recommended for Round 1.
Key self-diagnosed risk: a meta-labeling entrant built on |predicted_return| alone would reproduce #912's already-falsified degenerate one-feature result — must use a richer feature set (vol regime, rolling hit-rate, session) to be a genuinely new test.
Recommendation: promising direction, ready to pre-register — do NOT reuse the 2026-01-01→2026-07-04 exam window again (already served 7 candidates across #898+#912, near the ~10-candidate multiple-comparison budget); new frozen exam window required before Round 1 runs. Confirm #913 (backtest non-determinism) fix holds before trusting new numbers.

---

## [D-2026-07-08-01] 2026-07-08 20:33 · deploy-verify · daemon(PM, scheduled task post-fomc-prod-promote)
**Promoted develop → production (parity): `main` now byte-identical to `develop` @ `f62260dd`.** Ships the post-FOMC safety bundle: #923 (deterministic backtest inference / phantom-SELL fix), #929 (real DB-backed manual kill-switch, closes vapor #922), #925 (cloud-first training), #932 + #937 (ML-pipeline-only), plus docs/skills (#935/#924/#921/#920/#919/#916) and #917/#891. Pre-approved by Alex on 2026-07-08 — the human chose to wait for the FOMC entry-pause to lift (20:00Z) rather than deploy during the event window; this run executed that decision post-pause.
Rationale: routine parity promote of a CI-green, staging-validated bundle. Autonomy per charter v0.1 — "Deployment to production" is a MAY-do-without-asking action; no `board_required` gate (code promote, no model `latest`-symlink change, kill-switch not triggered). Rubric: ΔP high (phantom-SELL correctness fix + a real halt lever both *reduce* live-capital risk), ΔR ≈0 (no strategy/param/sizing change), C=4 (staging kill-switch drill 6/6 PASS #935 + prod boot verified below), E low → high-priority, veto-clear (ΔP well above the 2 floor).
Gates verified (all PASS; would have halted on any failure):
- **FOMC pause lifted**: scheduled task `fomc-pause-off` fired 20:00:05Z → `FEATURE_ENTRY_PAUSE=false` on prod; pause-triggered redeploy SUCCESS 20:01:18Z; bot evaluating entries (live `Decision:` lines, not gated).
- **Staging healthy**: `origin/staging` tree == `origin/develop` tree (`9bf797e0…`); sync PR #940 deploy SUCCESS 15:38Z; loop alive, balance $1006.80, no ERROR/CRITICAL.
- **Promote proof**: ours-merge tie commit `04cbb431` (parents develop `f62260dd` + main `53d41f31`), `HEAD^{tree}`==develop tree pre-push; PR #942 CI fully green (unit×4 + integration + claude-review); merged with `--merge` (merge commit, not squash) → main `c4024ea9`; post-merge `origin/main` tree == `origin/develop` tree, 0 behind.
- **Prod boot** (deploy `f930301e` SUCCESS ~20:30Z): trading loop started 20:30:16Z; Max Position 20.0%; check interval 60s; HyperGrowth/ETHUSDT resolves natively (no cross-symbol/mismatch warnings); ETHUSDT SHORT re-adopted @ $1696.83; reconciliation 0 corrections / 0 critical; **drawdown guard armed peak=$84.42 (session 20, account_history peak)**; no ERROR/CRITICAL.
- **Session continuity**: prod reused active session — session 20 still `is_active`, no spurious session 21; `account_history` heartbeat fresh (id 2058 @ 20:31:21Z, $84.40).
- **Kill-switch (#929) live**: alembic `0012_add_system_control_flags` applied in prod; `system_control_flags` table present, 0 rows → no active halt (safe default; matches the 6/6 staging drill).
Anomalies: none. (Pre-existing informational WARN unchanged: partial-exits DISABLED #734.)
Ref: PR #942 (promote), #940 (staging sync), #935 (kill-switch drill); commits #923/#925/#929/#932/#937 (+#917/#891/#916/#919/#920/#921/#924); scheduled task `post-fomc-prod-promote`; `.claude/state/charter.md` v0.1 (autonomy envelope); `.claude/skills/deploy-prod`.
