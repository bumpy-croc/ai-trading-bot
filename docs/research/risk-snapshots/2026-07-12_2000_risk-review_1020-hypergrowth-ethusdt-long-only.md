# Risk Review — Proposal 2026-07-12-01 (HyperGrowth/ETHUSDT long-only) — 2026-07-12 ~20:00 UTC

**Reviewer**: risk-officer (independent). Advisory input to a `board_required` decision — Alex decides.
**Proposal**: `.claude/state/proposals/2026-07-12-01-hypergrowth-ethusdt-long-only.md` (branch `claude/short-suppression-990`)
**Tracking**: GH #1020 · Evidence PR #1019 · Mechanism #990 (closed)

**Verdict**: SAFE-WITH-CONDITIONS
**Confidence**: med-high that the config change is risk-neutral-to-reducing and reversible; medium on the underlying "long-only is the right posture" claim (rests on 3 in-sample-caveated folds, 1 dissenting).
**Timing recommendation**: Ratify the DECISION now and ship to staging-paper now. Do NOT gate on #1016 observability data (see §Timing). Prod only after a staging-paper validation window.

---

## Independent view formed before leaning on the proposer's framing

I drafted my own failure modes and exposure analysis, then reconciled with the note. Where I diverge or add: the exposure-change reality (§Q1), the observability self-destruction tension (C6), the drawdown-guard true state (§Q4), and the "more longs" reallocation channel.

---

## Q1 — Does long-only-by-config change exposure vs long-only-by-accident, or is it pure honesty/parity?

**It changes exposure. It is NOT purely cosmetic.**

- Today (accident): the execution guard blocks shorts **only when free ETH > $1 dust** in the margin wallet. Shorts still slip through whenever the wallet is dust-free. **Position #22 is exactly such a trade** — a live SHORT that got through and is now −7.3% (≈ −$1.0, ~1.2% of the $84 book). The current suppression is margin-state-dependent and occasionally permeable (3 live shorts in the #990 sample).
- Config long-only (`allow_shorts=False` at signal-gen): the strategy never sets `enter_short=True`, so **no short ever enters, regardless of margin state**. This closes the residual dust-free short pathway entirely. #22-type trades would not exist.

So the change is a **real, if infrequent, reduction in short exposure**, in the risk-reducing direction on current evidence (short standalone P&L negative in all 3 folds; #22 losing live).

**But there is a second-order exposure change the proposer under-weights:** the long-only backtest arm takes **more long trades** (F2: 50 vs 12; F3: 62 vs 38) because capital no longer sits in shorts — it reallocates short capacity into longs. That is the concrete channel through which the bear-regime confound would bite: long-only isn't just "fewer shorts," it's "more longs," and more longs in a downtrend = more losing longs. In the tested (bull-ish) folds this did not worsen drawdown (long-only maxDD was equal-or-better in F2/F3), but those folds are not bear regimes.

## Q2 — Evidence quality: strong enough to ratify?

Adequate to ratify the **direction of the decision**, not strong enough to call settled:
- Rules out "suppression costs returns" with reasonable confidence (shorts beat long-only in only 1/3 folds, and that fold, F1, doesn't clear the pre-committed ±2pp bar; short-side standalone P&L negative 3/3, not outlier-driven — verified per-trade in the note).
- "Long-only saves returns" is directionally supported but soft: only F2 clears ±2pp; F3 is inside the noise band (−1.15pp); F1 dissents.
- All folds are **in-sample relative to the model's 2026-07-04 training cutoff** — relative arm-vs-arm delta is defensible, absolute P&L is not. The one live-matched segment (B) is degenerate (0 vs 1 trade). This is honestly disclosed.

The evidence is good enough for a **reversible config codification**; it would NOT be good enough for an irreversible change. It is.

## Q3 — Reversibility + kill criteria + monitoring

- **Reversibility: clean.** Flag flip, no state migration, open positions unaffected (rollback plan sound — see C2 caveat). Kill is trivial.
- **Re-enable shorts if:** (i) the underlying model is retrained/redesigned (new architecture or target) — the long/short asymmetry is model-specific and built on a ~51–53% DA signal barely above noise; (ii) a bear-regime fold shows long-only materially underperforming both-sides (the confound materializes); (iii) short-side directional accuracy demonstrably exceeds long-side OOS in ≥2 folds.
- **Monitoring that proves the change did what we think:** post-ship ETHUSDT HyperGrowth SHORT-entry count must be **exactly 0** (any short entry ⇒ flag mis-wired); the same backtest run against the live config must also show 0 short trades (parity check); and staging-paper must confirm the "more longs" reallocation does not spike drawdown.

## Q4 — Interaction with drawdown guard / circuit breakers / parity

**This is where the real risk lives — and it is mostly NOT about long-vs-short.**

- **Graduated circuit breakers are OFF in prod (#986).** The 2.5% daily-loss and 15% circuit-drawdown halts never arm; the protection ladder collapses to the single 20% hard cap (`RiskManager.check_drawdown`, balance-based).
- **Peak-anchor durability is unresolved (#847).** The active balance-based cap depends on a `peak_balance` whose durability across restarts is questioned; `CircuitBreaker._peak` is in-memory and resets at process start (session 20 open since 2026-06-05). So the running guards likely anchor on a post-drawdown recent peak (~$84) and **under-read** the true ~15.6% balance / ~19.7% equity drawdown from the $100/$103.82 all-time highs. The single active safety net may not be measuring the drawdown that has already happened.
- **Long-only does not bound drawdown below the cap.** F3 2025H1 long-only hit **20.31% max drawdown** — *exceeding* the 20% portfolio limit — with shorts fully disabled. So this proposal does **not** reduce the need for the disabled breakers; it underscores it.
- **Backtest-live parity is the primary technical risk.** If prod runs `allow_shorts=False` while backtests default to `True`, every future ETHUSDT HyperGrowth backtest re-diverges from live — recreating the exact #990 defect. Must be one config source consumed by both paths (C1).
- **Constants/limits parity check (my P0 duty):** current `risk-limits.json` and `src/config/constants.py` **agree** on the headline limits I checked (max_drawdown 0.20, max_position_size 0.10, base_risk 0.02, kelly_max 0.20, max_leverage 3.0, max_correlated_exposure 0.15). No new P0 from divergence. Caveats, both pre-existing and tracked (#986), not created here: `risk-limits.json` has never been ratification-stamped (`$last_reviewed: 1970-01-01`); the two correlated-risk constants (0.10 vs 0.15) question is open; and HyperGrowth's ~0.20–0.25 single-position cap exceeds the json's 0.10 `max_position_size_pct` / 0.20 `large_single_position_threshold`.

## Timing — ratify now vs wait for #1016 observability data

**Ratify now; do not wait.** Waiting is weak and partly self-defeating:
1. #1016 (guard-rejection observability) is merged to `origin/develop` but **not on `origin/main`** — **no prod rejection data is accruing yet.**
2. Position #22 has kept the book continuously in-position since 07-02 (segment B has **zero flat time**), so even after a prod deploy, rejection events accrue only during flat periods with a short signal and free ETH — potentially very slowly.
3. **Config long-only fires UPSTREAM of the execution guard.** Once it ships, the strategy never proposes a short, so `_record_short_guard_rejection` never fires for ETHUSDT — the #1016 data the note wants to accumulate becomes **moot for this symbol.** Shipping long-only actively destroys the future counterfactual it would be validated against (unless C6 is adopted).

## Top failure modes (my independent list)

1. **Backtest-live parity re-divergence** — if the flag isn't a shared single-source config, backtests silently re-diverge from live (#990 redux). *Early-warning signal:* any ETHUSDT HyperGrowth backtest that still books short trades after ship.
2. **Regime-drift confound materializes via the "more longs" channel** — a genuine sustained ETH bear turns the reallocated extra longs into concentrated losses with no (even illusory) short offset. *Early-warning signal:* staging-paper/live long-side loss streak coinciding with negative ETH trend; long-only maxDD exceeding both-sides in a bearish stretch. Note: #898 found HyperGrowth net-negative in a real 185-day bear even *with* shorts — long-only was not separately tested there, so this is inferred, not measured.
3. **Flag strands exit management of open shorts** — if implemented as a blanket short-side gate rather than entry-only, it could block closing #22 (a BUY-to-cover) and orphan it. *Early-warning signal:* #22 (or any short) stops receiving exit/stop management after deploy.

## Conditions

- **C1 (HARD) — parity:** `allow_shorts` must be a single config source consumed by BOTH the live and backtest construction paths (`create_hyper_growth_strategy` / `MLBasicSignalGenerator`). Any backtest representing the live ETHUSDT deployment must run `allow_shorts=False`. (CODE.md Backtest-Live Parity.)
- **C2 (HARD) — entry-only:** the flag gates SHORT *entry* only, never exit/management. Closing/stopping an existing short (a BUY-to-cover) must remain unconditional so open #22 and any pre-flip short are never stranded. code-reviewer verifies.
- **C3 (HARD) — staging-paper first:** no direct-to-prod. Validation window on staging-paper before any prod flip, given the regime confound and that long-only was never tested in the actual #898 bear window.
- **C4 (HARD) — documented re-enable + kill criteria** (see Q3) recorded in the proposal/log before prod.
- **C5 (HARD) — do not touch the margin guard** (`execution_engine.py` SHORT-inventory guard). Its margin-safety rationale is independent and remains a separate open question (note §8.5). Affirming the proposer's own stance.
- **C6 (RECOMMENDED, board's call) — preserve learnability:** because long-only gates upstream of the guard, it zeroes out #1016 rejection observability for ETHUSDT. Either (a) log a shadow "would-have-entered-short" event so the counterfactual stays measurable, or (b) board explicitly accepts the decision is revisited only on model retrain and forgoes ongoing validation data.
- **C7 (ESCALATION / context) — do not mistake this for handling ETH risk.** The live account is ~$84 (down ~16% from its $100 baseline), graduated breakers are OFF (#986), the durable peak anchor is unresolved (#847), and long-only still produced 20.31% drawdown in-sample (F3). Arming #986/#847 is higher-priority risk work than this proposal and should not be deferred because long-only was approved.

## What I could not verify

- **True live drawdown-guard trip state.** `RiskManager.check_drawdown` takes `peak_balance` as an argument; I could not confirm from the DB alone what peak the running prod process currently holds (durable all-time $100 vs a reset session peak ~$84). If it's the latter, the account is ~16% down with the guard reading near-zero — a live risk worth a separate live-ops check, independent of this proposal.
- **#898 long-only bear performance.** The 185-day bear window was run both-sides only; the long-only arm there was inferred, not measured. The single biggest evidence gap.
- **Whether the running live process actually picked up the 2026-07-04 model at the 07-05 symlink flip** (note §5.1 caveat; `reload_models()` has no callers) — inherited open question, not re-litigated here.
- **Segment A counterfactual** — confirmed non-reconstructable (cross-symbol substitute model, fails closed). Not a gap I can close.

---

### Suggested log.md entry (for PM to append to canonical shared state — I did not write to shared state from this isolated worktree)

```
## 2026-07-12 20:00 · track-record · risk-officer
Proposal 2026-07-12-01 (HyperGrowth/ETHUSDT long-only): verdict=approve-with-conditions, confidence=med
Scenarios checked: F1/F2/F3 2023-2025H1 folds (long-only maxDD up to 20.31%, F3), #898 bear-window (inferred only), segment-B live-matched (degenerate), backtest-live parity, entry-only-flag orphan risk on open SHORT #22, drawdown-guard/circuit-breaker interaction (#986 breakers OFF, #847 peak anchor), #1016 observability-destruction tension, constants/risk-limits parity (agree, no new P0).
Timing: ratify now, staging-paper first, do NOT gate on #1016 (not in prod; moot post-ship). Ref: .claude/state/proposals/2026-07-12-01-hypergrowth-ethusdt-long-only.md, GH #1020
```
