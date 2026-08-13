# Risk Review — GH #986 item 1, circuit-breaker arming — 2026-08-13 15:00 UTC

**Verdict**: **reject as framed** — do NOT arm prod (dry_run or enforce) today, and do NOT
record the staging window as a pass. The pre-committed gate is **UNMET on its own terms**, and
the deeper finding is that **the gate as written cannot be satisfied by waiting**. Recommended
replacement path below (verdict (c)).

**Confidence**: high on the blocking facts (deployed-commit provenance, zero restarts, #1036
open — all directly verified); med on the false-trip risk estimate (123 days of prod history,
one regime).

Read-only throughout: safe-list `railway` commands only, Postgres via public proxy with
`SET default_transaction_read_only = on`. No flag, config, or state mutation.

---

## Blocking finding first: prod does not have the fix the gate exists to validate

`origin/main` = **`bf7f45cb`**, and prod's only live deployment is `bf7f45cb`, deployed
2026-07-17T11:35Z (the entry-pause resume). `git merge-base --is-ancestor 2f6c1fe8 bf7f45cb` →
**false**. `git show bf7f45cb:src/engines/live/monitoring/circuit_breaker_enforcer.py` contains
**zero** references to `BreakerEquityFeed` / `seed_peak` / `unrealized`.

**PR #1032 was never promoted to prod.** `develop` is 25 commits ahead of `main`.

Therefore "arm prod in dry_run now" is not the action it appears to be: it would arm the
**cash-basis, unseeded-peak breaker** that `[D-2026-07-14-03]` explicitly refused to arm, for
exactly the two reasons that are still true of `bf7f45cb`. This alone forecloses option (a).

Prod flag state confirmed unchanged: no `FEATURE_ACCOUNT_CIRCUIT_BREAKERS` var on the prod
Trading Bot service → resolves to `feature_flags.json` default `off` at `bf7f45cb`.

---

## Key numbers

### Staging dry_run window — 2026-07-14 09:46Z → 2026-08-13 14:37Z (30.2 days)

Deployed commit `3cd4ce31` (PR #1035), verified to contain #1032 (`2f6c1fe8` is an ancestor;
enforcer file carries the equity-feed/seed-peak code). `FEATURE_ACCOUNT_CIRCUIT_BREAKERS=dry_run`
still set on the staging service (verified, not changed).

| Metric | Value | vs threshold |
|---|---|---|
| `CIRCUIT_BREAKER_DRY_RUN` rows | **0** | — |
| Max drawdown, **equity** basis | **1.49%** | 15% → **9.9% of limit** |
| Worst day (day-open → day-low), equity | **−0.631%** (2026-07-16) | 2.5% → **25.2% of limit** |
| Max drawdown, **balance** basis | 0.020% | (74× less sensitive than equity) |
| Max unrealized excursion `|equity−balance|/balance` | **1.178%** | — |
| Snapshots with an open position | **705 / 723 (97.5%)** | — |
| Sessions / process restarts in window | **1 / 0** | — |
| Trades | 3, +$7.16 | — |
| `account_history` heartbeat | unbroken, 23–24 rows/day | — |

Full `system_events` for the window: `SHORT_ENTRY_SUPPRESSED` ×8, `ENGINE_START` ×1,
`ENGINE_STOP` ×1 — all on 2026-07-14. Nothing since. Zero close-only, zero `SYSTEM_HALT`,
zero ERROR/CRITICAL.

### Prod, last 30 days (717 snapshots, 1 session)

| Metric | Value | vs threshold |
|---|---|---|
| Max drawdown, equity basis | 1.374% | 15% → 9.2% of limit |
| Max drawdown, balance basis | 0.020% | (69× less sensitive) |
| Worst day close-to-open | −0.601% (2026-07-31) | 2.5% → 24% of limit |
| Worst day-open → day-low | −0.709% | 2.5% → 28% of limit |

Neither threshold approached. Under `risk-limits.json`
`escalation.warning_at_pct_of_limit = 0.50`, nothing is in the warning band on either
environment — **no incident opened**.

### Prod tail check — 180 days (123 days with data, since 2026-03-29)

This is the number that actually matters for arming.

| Metric | Value |
|---|---|
| Days with intraday equity loss ≥ 2.5% | **1 of 123 (0.8%)** — 2026-06-03, −15.93% |
| Days ≥ half the limit (1.25%) | 6 of 123 (4.9%) |
| **Second-worst day** | **−1.82%** (2026-04-23) |
| Max drawdown, equity basis | **20.33%** → would have tripped the 15% breaker |

**False-trip risk from real prod volatility is low.** The distribution has a clean gap: the
worst benign day is −1.82% (73% of the limit), and the next observation is −15.93%. The 2.5%
daily and 15% drawdown thresholds sit in that gap. On 123 days of real prod data the daily
breaker fires once, on the day the account actually lost 16%.

**But I will not launder that into "one verified true positive."** Decomposing 2026-06-02→04
from `account_balances` (lag computed over the full ordered table, then filtered — the known
pitfall): the −15.93% is **−$15.75 of a single `margin_equity_sync_correction`** plus −$0.08
fees and −$0.04 realized. That is an **accounting restatement** (phantom balance being corrected
down), not a market excursion. Halting on it would have been *correct* — a 16% unexplained
equity restatement concurrent with the SL-fail/emergency-close cascade should stop trading — but
it does not demonstrate the breaker catching the adverse-mark-on-an-open-position risk class it
was rebuilt (#1032) to catch. The historical record contains **zero** instances of that class at
threshold scale.

---

## What the 30-day staging window actually proves — and what it does not

**Proves:**
1. **No false positives under benign conditions.** 30.2 days, 705 in-position snapshots, zero
   dry-run rows, with true equity moving continuously.
2. **Gap A (cash blindness) is genuinely fixed and exercised.** The window is not idle on the
   equity path: equity diverged from cash by up to **1.178%**, and equity-basis drawdown
   (1.49%) is **74× larger** than balance-basis drawdown (0.020%). The pre-#1032 breaker would
   have been staring at a number that moved 0.02% over a month. The `[D-2026-07-14-03]`
   criterion "spans ≥1 realistic open-position hold" is **richly satisfied**.

**Does not prove — and this is the whole verdict:**

1. **Zero restarts.** The pre-committed criterion required a window "long enough to span **≥1
   restart** ... so the equity-based, **restart-safe** code path is actually exercised, not
   idle." Staging has had **one deployment (2026-07-14) and no restart since** — one session,
   no `ENGINE_START`/`ENGINE_STOP` in 30 days. Gap B (restart-safe peak seeding), which is half
   of what #1032 shipped and half of what `[D-2026-07-14-03]` demanded, was **never exercised**.
   The gate is literally unmet on its own text.
2. **The one boot that did happen, failed to seed.** Per **GH #1036 (OPEN)**, the 2026-07-14
   carry-forward boot logged neither `daily baseline seeded` nor `peak seeded`; both seeders
   read `_recovered_inactive_session_id` after `startup.py` cleared it, and the first check
   raced the first snapshot by ~3s — `_seeded = True` is terminal on an empty-but-successful
   read. So the breaker has been running the entire 30 days with
   `_peak_seed_provenance = "self_anchored"`. **#1036 does undermine the peak seeding #1032
   added**, on precisely the boot path (new-session carry-forward) that a mid-drawdown restart
   would take. The evidence value of the window for gap B is not merely absent — it is negative.
3. **Zero true-positive evidence.** The trip path never executed: no `evaluate()` → tripped
   branch, no `log_risk_event`, no `CIRCUIT_BREAKER_DRY_RUN` row, no
   `equity/balance/basis/peak/peak_seed` payload ever written. The new #968 payload fields
   #1032 added have **never been observed in production data** — their correctness rests on unit
   tests alone.
4. **No evidence the equity feed was ever healthy or ever degraded.** The enforcer emits durable
   evidence **only on a trip**. `basis=equity` vs `basis=balance_degraded`, and the latch-freeze
   logic that hangs off it, are unobservable in a quiet window. Staging logs no longer reach the
   07-14 boot lines. This is LESSONS **§5.7** exactly: a silent component reads as healthy
   because it writes nothing.
5. **Not "30 days of evidence" in the way it reads.** It is *one* boot plus 30 days of a single
   uninterrupted process on a paper account that moved less than 1.5% peak-to-trough.

### Can the gate be satisfied by waiting?

**No.** To produce a true trip, staging would need a ~4× larger daily move or ~10× larger
drawdown than anything in 30 days. Prod's own 123-day base rate for a ≥2.5% day is **0.8%** —
and the single instance was an accounting correction, not a market move. Waiting for a
spontaneous trip is an unbounded wait for a ~1-in-125-day event that the strategy's own risk
controls (per-position SLs at ~1.1–2.0%) are specifically designed to prevent. **"Accumulate
clean days until confident" is not a convergent process for a control whose whole purpose is
tail events.** The honest reading is that the gate, as written, conflates *absence of false
positives* (which waiting does establish, and has) with *presence of correct behavior* (which
waiting can never establish).

---

## Top failure modes if armed now

1. **Arming a control that forgets its peak on restart.** #1036 open + prod's restart cadence
   (~13 in 30 days historically) means a restart mid-drawdown re-anchors the 15% halt to the
   depressed value — the #845/#847 peak-reset class, silently disarming the halt exactly when it
   matters. Compounded by `[D-2026-07-14-05]`: the 20% hard cap stays realized-basis, so the
   equity breakers are the *only* unrealized-excursion layer. If they are silently disarmed,
   there is no unrealized-excursion protection at all — while the Board believes there is two.
   *Early-warning signal:* `peak_seed` field in any dry-run/trip payload reading `self_anchored`
   rather than `db_session_max`; or a boot without the `peak seeded from account_history session
   max` INFO line.
2. **Arming the wrong binary.** Prod runs `bf7f45cb` (no #1032). Flipping the flag on prod today
   arms the cash-basis breaker: on prod's own 30-day data it would have measured a 0.020%
   drawdown while true equity fell 1.374% — 69× blind. Under `enforce` this is worse than off,
   because it produces documented false confidence.
   *Early-warning signal:* none from the running system — this one is only visible by checking
   deployed commit provenance, which is why it must be a hard precondition, not a monitored one.
3. **First-ever execution of the trip path happens on live capital during a real tail event.**
   The close-only latch, the `_halt_notified` de-dup, the payload write, and the degraded-basis
   freeze have never run outside unit tests. A defect there (e.g. an exception in
   `_enter_close_only_mode` under a degraded user stream) surfaces at the worst moment. Note the
   2026-08-08 prod `ALERT`: *user data stream circuit-open after 3 reconnects — REST-degraded* —
   the exact condition that drives `basis=balance_degraded`, and it occurs in prod.
   *Early-warning signal:* a forced-trip drill; there is no passive signal.
4. **(Under `enforce`, later) transient-mark false latch.** The daily breaker is equity-basis and
   latches for the UTC day. A 2.5% unrealized wick on a 12-day hold latches close-only even if it
   fully recovers — the risk `[D-2026-07-14-05]` cited when declining to move the hard cap to
   equity basis. It applies to the 2.5% daily breaker too and is not separately mitigated.
   *Early-warning signal:* dry-run rows whose `equity` recovers above the baseline within hours
   of the would-trip — measurable in dry_run *if* trips ever occur, which is the whole problem.

---

## Recommended path (verdict (c)) — replace "wait longer" with "make it trip"

Preconditions, in order, all falsifiable:

**P1 — Promote #1032 to prod.** Parity promote `develop` → `main` (25 commits) via the standard
`deploy-prod` process. Until prod runs a build containing `BreakerEquityFeed` + `seed_peak`,
the arming question is not answerable. *Criterion:* `git merge-base --is-ancestor 2f6c1fe8
<deployed-commit>` is true, verified on the deployed hash, not on `main`'s tip.

**P2 — Close #1036.** The restart-safe seeding is currently non-functional on the
carry-forward-boot path. *Criterion:* a staging boot on that path logs `Circuit-breaker drawdown
peak seeded from account_history session max: $X` with X matching the prior session's
`account_history` equity max within 0.1%, and the next payload carries
`peak_seed=db_session_max`.

**P3 — Forced-trip drill on staging (this is the missing evidence).** Use the existing
`kill-switch-drill` skill pattern: drive the breaker to a trip in `dry_run` on staging by a
sanctioned, reversible method — a temporary staging-only tightened threshold (e.g.
`daily_loss_limit=0.002` against observed ~0.6% days) is the cleanest, since it exercises the
real evaluate/trip/log path with real equity and requires no synthetic data. *Criteria:*
(a) a `CIRCUIT_BREAKER_DRY_RUN` row exists, (b) its payload carries all five #968 fields with
`equity == balance + unrealized` to the cent and `basis=equity`, (c) the row's `equity` and
`peak` reconcile against `account_history` for the same minute, (d) `_halt_notified` de-dup
holds — exactly one row per latch, (e) the latch clears on the next UTC day. Then revert the
threshold and confirm no further rows.

**P4 — Restart drill.** Restart staging while a position is open and while equity is below the
session peak. *Criteria:* post-restart payload/log shows `peak_seed=db_session_max` and the peak
equals the pre-restart durable peak, not the post-restart equity.

**P5 — Prod `dry_run`, ≥7 days, with a positive liveness assertion.** Per §5.7, silence is not
evidence: before calling the prod window clean, assert positively that the breaker ran — e.g.
confirm the seeding INFO lines at boot and that `account_history` equity/balance divergence is
non-zero over the window (proving the feed had something to see). *Criteria:* zero trips not
corresponding to a real ≥threshold move in `account_history.equity`; any trip reviewed against
the same reconciliation as P3.

**Then enforce**, with these numeric expectations pre-committed so the flip is auditable:
expected trip base rate **≤1 day per 120 trading days** (prod's own 123-day rate), and any trip
whose `equity` recovers above the daily baseline within 6h is logged as a
transient-mark false latch and counts against the mechanism. 72h heightened monitoring post-flip.

**Interim risk position (accept explicitly):** until P1–P5 complete, prod has **no
unrealized-excursion halt** — the `[D-2026-07-14-05]` residual, now 30 days older than when it
was accepted. Prod's realized-basis 20% hard cap is the only account-level layer. Given prod's
current exposure (~$83 balance, single ETHUSDT position, 1.37% 30-day drawdown), the absolute
capital at risk from this gap is small; the governance risk — an "armed" control that is not
armed — is the larger exposure, and it is the one this verdict refuses to create.

---

## What I could not verify

- **Whether the staging breaker's equity feed was healthy on any given iteration.** No durable
  positive signal exists; `railway logs` no longer reaches the 07-14 boot. Inferred healthy from
  the equity/balance divergence in `account_history`, which is a *different* code path
  (`event_logger`) than `BreakerEquityFeed` — suggestive, not proof.
- **The `peak_seed` provenance value in the running staging process.** Inferred `self_anchored`
  from #1036's boot-log evidence; no live read is possible without a trip or a restart.
- **Whether the 15% drawdown breaker would have tripped correctly on 2026-06-03.** Prod's
  180-day max equity drawdown of 20.33% exceeds 15%, but that period predates #1032 and the
  restart cadence then would have re-anchored the peak repeatedly; the counterfactual is not
  reconstructable from `account_history` alone.
- **Tail beyond 123 days.** `account_history` on prod starts 2026-03-29 in this query window;
  no 2022-collapse-class regime exists in live data. The false-trip estimate rests on one
  benign-to-mildly-adverse regime.

### Incidental findings (not blocking, flagged for the PM)

- Staging `positions` holds **12 stale `OPEN` rows** from sessions 1–22 (oldest 2026-04-06,
  129 days). Session 23 correctly tracks 1 (`account_history.open_positions = 1`), so this is
  paper-account debris rather than a live divergence — but it is the shape of a DB/memory
  divergence indicator and should be cleaned so a real one is visible against it.
- Prod logged **no `ENGINE_START`/`ENGINE_STOP`** for the 2026-07-17 restart, while staging did
  for 07-14. Restart observability on prod appears to be missing — worth confirming, since P4/P5
  depend on being able to see restarts in durable state.
