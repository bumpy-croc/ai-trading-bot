# Risk Review — proposed raise of `portfolio.max_drawdown_pct` 0.20 → 0.30 — 2026-08-13 18:00 UTC

**Reviewer**: risk-officer (independent; own worktree `.claude/worktrees/risk-dd30-0813` from `origin/develop@102fceb9`)
**Requested by**: Alex (Board), in-session 2026-08-13, choosing "raise the enforced cap" over (i) keep 20% + widen the warning ladder, (ii) time-boxed research-only raise.
**Layer**: 1 (human-owned files). This document is review content only — no ratified file was edited by this review.
**Prod DB**: read-only. **Related**: GH #986, #1036, #845, #847, PR #1032, [D-2026-07-14-03/04/05], [D-2026-08-13-01/02].

---

## Verdict

**SAFE-WITH-CONDITIONS** — **Confidence: high**

The raise is affordable in dollars at today's capital ($8.44 of additional tail). It is **not** safe *today* on mechanism, and it does not buy what it appears to buy. Two findings drive the verdict:

1. **The premise is factually wrong.** The 21.84% MaxDD does **not** establish that the ratified 20% cap is too tight for the strategy. It is a property of HyperGrowth's *override* of the Board's ratified throttle tiers. On the identical 365-day window, with the ratified tiers `[0.05,0.10,0.15]/[0.8,0.6,0.4]` restored, the same strategy produced **MaxDD 17.01% and return −16.08%** — inside the existing cap, at **4.1pp better return** (CF-A, `docs/research/experiments/2026-07-04_hypergrowth-365d-drawdown-stress-review.md` §4). A measured alternative already exists that fits the strategy inside the current limit *and* loses less money.

2. **Sequencing is the real risk, not the number.** The 20% cap is currently the **only functioning live drawdown control**. Circuit breakers are OFF in prod and cannot be armed ([D-2026-08-13-01]: PR #1032 unpromoted, #1036 P1 seeding failure, no forced-trip drill). The cap is realized/cash-basis ([D-2026-07-14-05]) so it cannot trip on an open position's unrealized excursion. The graduated throttle has exactly one live rung. Widening the sole working control by 10 percentage points, at the precise moment the layered controls are known-broken and under active repair, is the wrong order of operations.

**Recommendation: SEQUENCE — do not apply now.** Apply after P1 (promote #1032) + P2 (close #1036) + P3 (forced-trip drill) + P5 (prod dry_run with positive liveness assertion), per the path the PM already pre-committed today. At that point a 30% cap is *materially more defensible than it is today*, because the effective halt becomes the 15% equity-basis breaker and the 30% cash-basis cap is a genuine backstop rather than the only line. This is "not now, and here is exactly when" — not a veto of the Board's appetite.

If the Board applies it now regardless (its prerogative — the charter states high appetite), the conditions in §7 are firm, and **C1 (escalation-band compensation) is non-negotiable** — without it the human's first notification silently moves from 10% to 15% drawdown.

---

## 1. What actually changes in behaviour

### 1.1 The most important mechanical fact: `risk-limits.json` is inert at runtime

`src/config/risk_limits.py` (the loader) is imported by **tests only**:

```
$ grep -rn "risk_limits" --include=*.py . | grep -v "^./src/config/risk_limits.py"
tests/unit/config/test_risk_limits_schema.py:15:...
tests/unit/config/test_risk_limits.py:18:...
(no src/ consumers)
```

The operative value is **`src/config/constants.py:140  DEFAULT_MAX_DRAWDOWN = 0.20`**, reaching live via:

```
constants.DEFAULT_MAX_DRAWDOWN
  → src/engines/live/runner.py:148   argparse default for --max-drawdown
  → runner.py:271                    RiskParameters(max_drawdown=args.max_drawdown)
  → trading_engine.py:1062           MaxDrawdownGuard(self.risk_manager.params.max_drawdown)
```

Production start command (`railway.json`) is `atb live-health hyper_growth --max-position 0.20` — **no `--max-drawdown` flag**, so prod runs the constants.py default.

**Consequence the Board must understand: editing `risk-limits.json` alone changes nothing. `constants.py` IS the behavioural change, and it takes effect on the next prod deploy.** The JSON is the document of record; the constant is the control. This also means the change is not "config-reversible" — reverting requires a deploy.

### 1.2 Complete consumer trace

| Consumer | File:line | Effect of 0.20 → 0.30 |
|---|---|---|
| **MaxDrawdownGuard hard cap (LIVE)** | `drawdown_guard.py:135`, `trading_engine.py:1062` | **Close-only latch moves from 20% → 30% drawdown from session peak.** The only live behavioural change. |
| Guard WARNING tier | `drawdown_guard.py:136` (`cap × 0.50`) | 10% → **15%** DD |
| Guard CRITICAL tier | `drawdown_guard.py:137` (`cap × 0.80`) | 16% → **24%** DD |
| Backtest early-stop | `engine.py:358-359`, `:1272` | 20% → 30%, **only when `risk_parameters` is passed**. The CLI default remains `0.5`, so the standard `atb backtest` invocation is unaffected (unchanged since the 2026-07-04 review flagged it). |
| `PerformanceMonitor.max_drawdown_threshold` | `performance_monitor.py:59` | 0.20 → 0.30. **Not wired into any engine** (`grep "PerformanceMonitor(" src/engines/` → zero hits). Monitoring-only, dormant. |
| `RiskParameters.max_drawdown` validation | `risk_manager.py:187-188` | bound is `0 < x <= 1`; 0.30 passes. |
| `PortfolioRiskManager.check_drawdown` | `risk_manager.py:880` | **Zero call sites** (only a docstring example at `:72`). Dead code, as found 2026-07-04. |
| `dynamic_risk.py:305` | pass-through | comment: "Don't adjust max drawdown threshold". No effect. |
| Loader dead-tier invariant | `risk_limits.py:278` | widens the *allowed* JSON threshold space to `< 0.30`. No runtime effect (loader inert). |
| **NOT affected** | | `DEFAULT_CIRCUIT_DRAWDOWN_HALT=0.15` (breakers, OFF), `DEFAULT_REGIME_MAX_DRAWDOWN_SWITCH=0.15`, `DEFAULT_DRAWDOWN_THRESHOLD=0.15`, `EmergencyControls` thresholds 0.25/0.15 (**not wired to any engine** — zero hits in `src/engines/`) |

**Net: exactly one live control moves.** That is simultaneously reassuring (small blast radius) and the whole problem (§3).

### 1.3 What runs 10 percentage points longer

Not a crash — a bleed. The 365d reproduction's anatomy is explicit: *"slow bleed, not a crash; worst single trade −$2.19 (2.6% of capital) … death by a thousand stop losses."* Two independent estimates of the extra runway:

- **By bleed rate**: −20.15% over 365d ≈ −1.85%/month realized → 10pp ≈ **5.4 additional months** of trading before the halt.
- **By stop-outs**: max realized loss per stop-out = notional 0.20 × `stop_loss_pct` 0.10 = **2.0% of balance**. 10pp ≈ **5 additional full-size stop-outs**. At the observed live rate (15 trades / 90 days ≈ 1 per 6 days), ≈ **30 days** if every trade loses at full size.

So the honest characterisation is: *the raise buys roughly one to five more months of losing before the machine stops you.*

### 1.4 Dollar exposure

Live baseline, read-only prod query, 2026-08-13 15:47 UTC:

```
account_history latest   $83.64      session 20 (active since 2026-06-05)
session-20 peak          $84.42      ← the guard's actual baseline
session-20 min           $82.73
open positions           0           (cash ≈ equity today)
current drawdown         0.92%       (84.42 → 83.64)
trades: 2 / 30d, 15 / 90d, Σpnl +$0.46 / 90d
```

Empirical overshoot past the latch is **+0.5pp** (CF-B measured: cap enforced at 0.20 → realized MaxDD 20.50%). Theoretical worst-case overshoot is **+2.0pp** (one open position stopping out at full size after the latch: 0.20 notional × 0.10 stop; long-only per [D-2026-07-14-01], `max_leverage=1.0`).

| Capital (peak) | Trip @20% | Trip @30% | **Δ capital at risk** | Worst realized @30% (+2.0pp) |
|---|---|---|---|---|
| **$84.42** (today) | −$16.88 → $67.54 | −$25.33 → $59.09 | **+$8.44 (+50%)** | −$27.01 → $57.41 |
| **$250** | −$50.00 | −$75.00 | **+$25.00** | −$80.00 |
| **$1,000** | −$200.00 | −$300.00 | **+$100.00** | −$320.00 |
| **$1,083** (live + the £1000-equiv tranche the Board offered on 2026-08-13) | −$216.60 | −$324.90 | **+$108.30** | −$346.56 |

### 1.5 Tail check — 99th-percentile daily loss

Computed from prod `account_history`, 122 daily observations:

```
p99 daily return  = −1.647%
p95 daily return  = −0.011%      (most days flat — 15 trades/90d)
worst day         = −15.773%     (2026-06-03)
```

The worst day is **not a market move**: per [D-2026-08-13-01] it decomposes to −$15.75 of `margin_equity_sync_correction`, an accounting restatement. Excluding it, the worst genuine market day in 180d is **−1.82%**.

| Capital | p99 daily loss | Worst genuine market day (−1.82%) |
|---|---|---|
| $83.64 | **−$1.38** | −$1.52 |
| $250 | **−$4.12** | −$4.55 |
| $1,000 | **−$16.47** | −$18.20 |

**Assessment: the daily tail is not the binding risk and is unaffected by this change.** At any plausible capital the p99 day is 1.6% of balance — the cap is 12–18 p99-days away either way. The binding risk is cumulative bleed, which is exactly what the cap exists to stop, and exactly what this change delays.

---

## 2. The uncomfortable context — stated plainly

**Answer: (c) both — but the (a) component does not require this change, and the (b) component is demonstrable rather than merely arguable.**

### 2.1 The legitimate part (a)

The charter states high risk appetite. The programme is in an explicitly pre-edge research phase ([D-2026-08-13-02]: "make the system profitable first"). A halt latching mid-exploration on a $83 account is operationally annoying and carries its own cost — the guard's trip is a *process-lifetime latch* requiring an operator restart with `FEATURE_MAX_DRAWDOWN_RESET_PEAK=true`. Wanting more headroom during research is a coherent position, and "it's aggressive" is not, on its own, a finding here.

### 2.2 The goalpost-moving part (b) — the decisive evidence

The stated justification is that the strategy's known drawdown profile (21.84% MaxDD) does not fit inside a 20% cap. **That framing is incorrect.** From the same stress review that produced the 21.84% figure (§4, three runs on the identical window and params):

| Run | Config | Return | **MaxDD** | Breach? |
|---|---|---|---|---|
| Baseline | live config as-is (HyperGrowth tiers `[0.15,0.30,0.45]/[0.8,0.5,0.2]`) | −20.15% | **21.84%** | yes |
| CF-B | + hard cap enforced at 0.20 | −20.41% | 20.50% | 0.5pp overshoot |
| **CF-A** | **ratified tiers restored `[0.05,0.10,0.15]/[0.8,0.6,0.4]`** | **−16.08%** | **17.01%** | **no** |

CF-A is load-bearing. The 21.84% is caused by HyperGrowth's deliberate ~3× loosening of the Board's ratified throttle tiers (`hyper_growth.py:406`, comment: *"Wider drawdown tolerance for hyper-growth target"*). Restoring the Board's own numbers contains the same bad year inside the existing cap **and returns 4.1pp more** ($3.70 on an $85 base). The protection was free on this path.

So the choice actually in front of the Board is not "raise the cap or trip the halt." It is:

- **Option A** — raise the cap to 30%: fits the strategy inside the limit by moving the limit. Cost: +$8.44 today, +$108 at the offered capital. Measured benefit: none.
- **Option B** — re-anchor HyperGrowth's tiers to the ratified set: fits the strategy inside the *existing* limit, with **measured** −16.08% vs −20.15% (i.e. loses $3.70 *less* per $85) and MaxDD 17.01%.

Option B was explicitly left available by [D-2026-07-14-04] item 2: *"Re-anchor idea available as a future separate proposal if ever wanted."* It has never been proposed. Raising the cap without first putting Option B in front of the Board is choosing the more expensive of two measured options.

### 2.3 The asymmetry that settles it

**A wider drawdown cap has zero upside on a negative-expectancy strategy. It can only ever let you lose more.** It pays off exclusively when the expected value of continuing to trade past 20% is positive. The current evidence on that question ([D-2026-08-13-02], and I have not re-derived it):

- honest 365d backtest: **−20.15% return, PF 0.47**, both halves negative (2025 −16.72%, 2026 −4.81%)
- Sharpe 0.119 — below the charter's stated **minimum** of 0.5
- six independent nulls at the ~51–53% directional-accuracy ceiling; **every fold of every study has PF < 1.0**
- live: 15 trades / 90 days, Σ P&L **+$0.46** — indistinguishable from zero

There is no evidence that trading past 20% has positive expected value, and substantial evidence it does not. Under those conditions, raising a loss limit to accommodate the losses is definitionally relabelling an unacceptable outcome as acceptable.

**How I would put it to the Board in one sentence:** *the raise costs up to $108 of tail at the capital you just offered, purchases no measured benefit, and a cheaper alternative measured on the same data both fits the strategy inside the current cap and loses less — so if the goal is "profitable across all market conditions" ([D-2026-08-13-02]), this change moves in the opposite direction.*

I state that as clearly as I can while recognising it is the Board's call. The charter's high appetite legitimises accepting a larger loss; it does not make a change that only enlarges losses a *good* one.

---

## 3. Interaction with the currently-BROKEN protection stack

**This is the strongest concrete objection, and it is about timing, not the number.**

### 3.1 The ladder as it actually exists in prod today

| Rung | Intended | Actual state in prod |
|---|---|---|
| Graduated throttle ×0.8 | 5% DD (ratified) | **15% DD** — HyperGrowth override, `hyper_growth.py:406` |
| Graduated throttle ×0.6 / ×0.5 | 10% DD | **DEAD** — override puts it at 30%, past the 20% cap ([#986 item 4]) |
| Graduated throttle ×0.4 / ×0.2 | 15% DD | **DEAD** — override puts it at 45% |
| Daily-loss breaker 2.5% | halt for the day | **OFF** — flag off in prod; #1032 (equity-basis) never promoted ([D-2026-08-13-01] finding 1) |
| Circuit drawdown-halt 15% | halt until recovery | **OFF** — same; and cash-basis breaker on prod's last 30d reads 0.020% DD while true equity fell 1.374% (**69× blind**) |
| **Max-drawdown hard cap 20%** | close-only latch | **THE ONLY FUNCTIONING RUNG** — and realized/cash-basis ([D-2026-07-14-05]), so blind to unrealized excursion |
| Guard peak durability | rolling | **session-scoped** (#847 open); a clean restart re-baselines the peak |
| Peak seeding | `db_session_max` | **#1036 P1** — carry-forward boots silently self-anchor; staging has run 30 days at `peak_seed=self_anchored` |

The live ladder is: **one ×0.8 nudge at 15%, then a cliff at 20%.** Nothing else fires.

### 3.2 Does widening the sole functioning control materially change the risk picture?

**Yes.** Removing 10 of the 20 percentage points of the only working control is not a marginal adjustment — it is a 50% reduction in the effective protection the system currently has. In dollar terms it is small today ($8.44) because the account is small; in *control* terms it is the difference between "one broken ladder with a floor at 20%" and "one broken ladder with a floor at 30%."

Three compounding factors make now the worst moment:

1. **The cap is cash-blind.** It cannot trip during an open position's unrealized excursion. The equity-basis breakers that were supposed to cover that gap at 15% ([D-2026-07-14-05]'s explicit rationale for leaving the cap realized-basis: *"the equity-based breakers from #1032, once armed at 15%, trip BEFORE the 20% cap in unrealized scenarios — layered coverage restored"*) **do not exist in prod**. The stated mitigation for the cap's known blindness is unbuilt. Raising the cap widens a gap whose compensating control was already missing.
2. **The peak baseline is not durable.** #1036 (P1, fix dispatched today) means a mid-drawdown restart can self-anchor the peak, silently resetting the measurement. A 30% cap measured from a reset peak is not a 30% cap.
3. **The band that would warn a human has never fired.** [D-2026-08-13-01]: nothing has entered the 50%-of-limit warning band on either environment. The alerting path at the tier that matters is untested. Moving that band deeper (10% → 15%, §5) makes an untested path also a later one.

### 3.3 Sequencing recommendation

**Sequence the raise behind the already-pre-committed repair path.** From [D-2026-08-13-01]:

- **P1** — promote #1032 to prod, verified on the deployed hash
- **P2** — close #1036 (peak seeding)
- **P3** — forced-trip drill on staging (temporary staging-only tightened `daily_loss_limit`, payload reconciled against `account_history`)
- **P4** — restart drill proving `peak_seed=db_session_max`
- **P5** — prod dry_run ≥7 days with a **positive** liveness assertion
- **then** — arm breakers at 2.5% daily / 15% drawdown
- **then** — this cap raise becomes reviewable on its merits

**Why sequencing genuinely resolves most of my objection, rather than just deferring it:** once the 15% equity-basis breaker enforces, the *effective* halt is 15% on true equity — tighter and better-based than today's 20% cash-basis cap. At that point raising the cash-basis cap from 20% → 30% converts it from "the only line" into an actual backstop behind a functioning primary. **A 30% cap in a working three-rung ladder is safer than a 20% cap in a broken one-rung ladder.** The Board would be buying research headroom without giving up protection. That is a genuinely good trade; today's version is not.

Estimated gate: P1–P2 are days (both dispatched/available); P3–P4 are a drill each; P5 is 7 days. **~2–3 weeks**, not months.

---

## 4. Dynamic-risk throttle tiers

Two separate answers, because two different tier sets are in play.

### 4.1 The RATIFIED JSON tiers — leave them alone

`[0.05, 0.10, 0.15]` / `[0.8, 0.6, 0.4]` under a 0.30 cap sit at **17% / 33% / 50%** of the cap (vs 25% / 50% / 75% under 0.20).

**Recommendation: NO change.** Three reasons:

1. They remain a strictly ascending, valid ramp; the loader's dead-tier invariant (`risk_limits.py:278`, every threshold `< max_drawdown_pct`) still passes with margin.
2. They are the *conservative* set and CF-A proves they contain this exact strategy on this exact window at 17.01% MaxDD. Widening them to "keep pace" with the cap would destroy the one configuration measured to work.
3. Front-loading the de-risking (all three cuts inside the first half of the range) is the **correct** shape for a graduated throttle, not a defect. You want size cut early and hard so the deep range is never reached. A ramp whose last rung sits at 75% of the cap (today's shape) is the badly-distributed one.

Resist the intuition that raising the ceiling requires raising the rungs. It requires the opposite.

### 4.2 The LIVE tiers — this is where the raise does real damage

Production does **not** use the ratified tiers. `hyper_growth.py:406` overrides them to `[0.15, 0.30, 0.45]` / `[0.8, 0.5, 0.2]`, and both engines honour the override via `merge_dynamic_risk_config`.

| Cap | 0.15 → ×0.8 | 0.30 → ×0.5 | 0.45 → ×0.2 | Effective rungs |
|---|---|---|---|---|
| 0.20 (today) | live | dead (past cap) | dead | **1** |
| **0.30 (proposed)** | live | **fires at exactly the latch point — useless** | dead | **1** |

Raising the cap **does not resurrect a single useful rung**. It stretches one ×0.8 nudge across 30 percentage points instead of 20. The "graduated" ramp gets measurably worse: the fraction of the loss range covered by any de-risking at all falls from 25% to 17%.

Two further facts the Board should have:

- **[D-2026-07-14-04] item 2 (PRUNE-ONLY) is still unimplemented on develop.** `hyper_growth.py:406` still reads `[0.15, 0.30, 0.45]`. The companion "unrepresentability invariant" (strategy threshold ≥ `max_drawdown_pct` fails validation) also never shipped — `dynamic_risk.py.__post_init__` (lines 82–99) validates length, positivity and factor range only, with **no comparison against `max_drawdown`**. Strategy overrides bypass the loader's dead-tier invariant entirely.
- Note the prune, as ratified, is explicitly *zero behaviour change* (delete the two dead tiers, keep `0.15 → 0.8`). It does **not** deliver CF-A's 17.01%. CF-A requires the **re-anchor**, which the Board declined in July. These are different changes and I do not conflate them.

### 4.3 Concrete recommendation

**If the raise proceeds, re-spacing HyperGrowth's override is a required companion, not optional.** Proportional analogue of the ratified ramp at a 0.30 cap:

```
"drawdown_thresholds":     [0.075, 0.15, 0.225]     # 25% / 50% / 75% of a 0.30 cap
"risk_reduction_factors":  [0.8,   0.6,  0.4]       # the ratified factor set
```

This preserves the ratified ramp *shape* at the new cap and is the direct 30%-cap analogue of the only configuration measured to contain the strategy. Per [D-2026-07-14-04], **this is a new Board item, not an assumed follow-on** — it should be tabled at the same sitting rather than shipped as a rider, and it is a strategy-config change (layer 2) requiring its own backtest verification before deploy.

**Better-evidenced alternative I would rather the Board consider first**: adopt CF-A outright — delete the HyperGrowth `dynamic_risk` override so the ratified defaults apply — and **do not raise the cap**. Measured on the exact window in question: **MaxDD 17.01%, return −16.08%**, no breach, no limit change, no layer-1 sitting required for the cap. That is the cheapest path to "the strategy's drawdown profile fits inside the limit."

---

## 5. Escalation bands

Unchanged ratios `0.50 / 0.80` under a 0.30 cap move the tiers to:

| Tier | @0.20 cap | @0.30 cap, ratios unchanged | @0.30 cap, **recommended** ratios |
|---|---|---|---|
| WARNING (human first notified) | 10.0% DD | **15.0% DD** | 0.35 → **10.5% DD** |
| CRITICAL | 16.0% DD | **24.0% DD** | 0.55 → **16.5% DD** |
| BREACH (machine halt) | 20.0% | 30.0% | 30.0% |

### 5.1 Is 15% / 24% acceptable?

**On lead time alone, marginally — yes.** Lead time from WARNING to BREACH:

- by bleed rate: 15% → 30% at −1.85%/month ≈ **8 months**
- by cascade: 15pp ÷ 2.0% max realized per stop-out = **7.5 consecutive full-size stop-outs**; at 1 trade per 6 days ≈ **45 days**

The human is not going to be surprised in hours. Turnover is far too low for that.

**On principle — no, and I recommend against it.** Three reasons:

1. **It silently relocates the human decision point.** The charter (line 22) currently *conflates* the machine halt with the human decision point: *"the hard system limit doubles as the human decision point."* Leaving the ratios fixed means raising the machine cap drags the human's first notification from 10% to 15% as an unintended side effect. Nobody proposed moving the human decision point; it would move by arithmetic.
2. **The warning path is untested.** [D-2026-08-13-01]: nothing has entered the 50%-of-limit band on either environment, ever. Making an untested alerting path also a *later* one compounds two weaknesses.
3. **Raising the cap is precisely the right moment to un-conflate the two.** Separating "when the machine stops" (30%) from "when a human must decide" (10.5%) is strictly better governance than today's arrangement, and it costs nothing.

### 5.2 Recommendation

**Tighten the ratios to hold the human decision points where they are today:**

```
"warning_at_pct_of_limit":  0.50 → 0.35      # 10.5% DD  (vs 10.0% today)
"critical_at_pct_of_limit": 0.80 → 0.55      # 16.5% DD  (vs 16.0% today)
```

Verified compatible: the loader invariant `warning <= critical` (`risk_limits.py:285`) holds (0.35 ≤ 0.55); both pass `_require_fraction`; the guard's tier products (`drawdown_guard.py:136-137`) with `_TIER_EPSILON = 1e-12` handle 0.30 × 0.35 = 0.105 and 0.30 × 0.55 = 0.165 without boundary issues.

Because the PM has converted the daily-standup tripwires to compute from these ratified values dynamically, this takes effect automatically with no further code change. **The constants.py comments at lines 143–144 hardcode the old arithmetic in prose and must be updated in the same edit** (they read "(10% drawdown at 0.20)" / "(16% drawdown at 0.20)").

**This is condition C1 and I regard it as non-negotiable.** Raising the cap without it is a change the Board did not ask for.

---

## 6. Charter coherence + queued layer-1 items

Charter line 22 must change in the same sitting or the charter contradicts the ratified file. While Alex is in the file, these previously-queued layer-1 items should clear in one sitting:

| # | Item | Current text | Should be | Source |
|---|---|---|---|---|
| 1 | **charter:22 drawdown prose** | "**20%** (matches `risk-limits.json` `max_drawdown_pct` — the hard system limit doubles as the human decision point)" | 30% machine cap, **10.5% human decision point** — explicitly un-conflated | this review |
| 2 | **charter:24 stale exposure** | "Maximum single-position exposure: **10%** of capital (matches `max_position_size_pct`)" | **20%** — ratified JSON says 0.20, prod pins `--max-position 0.20` | [D-2026-07-14-04] item 4 |
| 3 | **charter path refs** | `risk-limits.json` | `src/config/risk-limits.json` | [D-2026-07-14-04] item 1 |
| 4 | **charter:9 Mission** | "Grow $1000 live account" | goal restated to "reach demonstrated profitability first"; only $83.64 is live | [D-2026-08-13-02] |
| 5 | **charter:13 capital** | "$1,000 paper, $87 live" | $83.64 live (prod `account_history`, 2026-08-13 15:47 UTC) | this review |
| 6 | **#986 item 2 — a live P0 by the file's own rule** | `constants.py:135 DEFAULT_MAX_POSITION_SIZE = 0.1` vs JSON `max_position_size_pct: 0.20` | align to 0.20 | GH #986 |
| 7 | **#986 item 3** | `FixedFractionSizer` hidden `max_fraction=0.2` silently clamps HyperGrowth's configured `base_fraction=0.25`; the 0.25 is decorative | Board to state which number is approved; make the clamp explicit | GH #986, [D-2026-07-14-04] item 4 |
| 8 | **charter:25 leverage prose** (minor) | "up to 3x on **futures**" | prod runs margin with `max_leverage=1.0`; JSON caps at 3.0. Prose is not wrong as a cap but names the wrong venue | this review |

Item 6 is worth calling out: `risk-limits.json`'s own header says *"Must match src/config/constants.py. Any divergence is a P0."* That divergence is live today and has been since at least 2026-07-12. It is benign in effect (the code default is stricter and prod overrides explicitly) but it violates the file's stated invariant, and this sitting is the natural place to clear it.

---

## 7. Conditions

**C1 — Escalation bands compensate (NON-NEGOTIABLE).** `warning_at_pct_of_limit` 0.50 → **0.35**, `critical_at_pct_of_limit` 0.80 → **0.55**, holding the human decision points at 10.5% / 16.5%. Without this the human's first notification moves from 10% to 15% DD as an unintended arithmetic side effect. Included in the §8 diff.

**C2 — Sequence behind the protection-stack repair (STRONGLY RECOMMENDED; the substance of my objection).** Apply after P1 (promote #1032, verified on the deployed hash) + P2 (close #1036) + P3 (forced-trip drill) + P5 (prod dry_run ≥7d with positive liveness assertion), per [D-2026-08-13-01]. Estimated ~2–3 weeks. If the Board applies now instead, C2 converts to a hard gate on the *reverse* direction: **the breaker-arming path must not be de-prioritised as a result of the extra headroom.** The raise makes that repair more urgent, not less.

**C3 — CF-A must be tabled before, or alongside, the raise.** The Board should see that restoring its own ratified throttle tiers produces MaxDD 17.01% / return −16.08% on the same window — inside the current cap, at better return — before choosing to move the cap instead. If the Board sees CF-A and still prefers the raise, that is an informed decision and I withdraw this condition.

**C4 — HyperGrowth tier re-spacing tabled as a companion Board item.** Per §4.3, `[0.075, 0.15, 0.225] / [0.8, 0.6, 0.4]`. A new Board item per [D-2026-07-14-04], not a rider; needs its own backtest verification before deploy. Without it the 30% cap has a single ×0.8 rung covering the whole range.

**C5 — Ship `constants.py` and `risk-limits.json` in the same commit, and understand that `constants.py` is the control.** The JSON is inert at runtime (§1.1). Shipping the JSON alone produces a documented-but-not-enforced 30% cap — the exact "armed control that isn't" failure mode [D-2026-08-13-01] called out as the real exposure. The change takes effect on the next prod deploy and reverting requires a deploy, not a config flip.

**C6 — Re-verify the guard peak after the next deploy.** The change reaches prod via restart. Given #1036 (P1, open), confirm from logs that the guard re-arms with `peak_seed=db_session_max` and the expected session-20 peak (~$84.42) — not a self-anchored current balance. A 30% cap measured from a reset peak is not a 30% cap. `grep "Max-drawdown guard armed"` in the boot logs.

**C7 — Pre-commit the review trigger.** Record in `log.md` that the 30% cap is revisited if *either* (a) live drawdown reaches the 10.5% warning band, or (b) 90 days pass without demonstrated positive expectancy. A raised limit with no expiry and no review trigger is how limits ratchet permanently in one direction.

---

## 8. Proposed diff — ready to apply verbatim

> Produced as review content. **Not applied by this review** — layer-1 files are the Board's hand.
> Includes C1 (band compensation). Charter items 4/5/8 from §6 are marked OPTIONAL — they are correct but independent of this change.

### 8.1 `src/config/risk-limits.json`

```diff
@@
   "$schema_version": "1",
   "$owner": "human_board",
   "$source_of_truth_note": "Must match src/config/constants.py. Any divergence is a P0.",
-  "$last_reviewed": "2026-07-05",
+  "$last_reviewed": "2026-08-13",
   "$last_reviewer": "alexflorisca",
 
   "portfolio": {
-    "max_drawdown_pct": 0.20,
+    "max_drawdown_pct": 0.30,
     "max_daily_risk_pct": 0.06,
     "max_correlated_exposure_pct": 0.15,
     "dynamic_drawdown_thresholds_pct": [0.05, 0.10, 0.15],
     "dynamic_risk_reduction_factors": [0.8, 0.6, 0.4]
   },
@@
   "escalation": {
-    "warning_at_pct_of_limit": 0.50,
-    "critical_at_pct_of_limit": 0.80,
+    "warning_at_pct_of_limit": 0.35,
+    "critical_at_pct_of_limit": 0.55,
     "breach_action": "halt_new_entries_and_page_human"
   },
```

Post-change tier arithmetic: WARNING 0.30 × 0.35 = **10.5% DD**; CRITICAL 0.30 × 0.55 = **16.5% DD**; BREACH **30% DD**.
Loader invariants verified: `max(dynamic_drawdown_thresholds_pct) = 0.15 < 0.30` ✓ (`risk_limits.py:278`); `warning 0.35 <= critical 0.55` ✓ (`:285`); all values pass `_require_fraction` ✓.
`dynamic_drawdown_thresholds_pct` / `dynamic_risk_reduction_factors` deliberately **unchanged** — see §4.1.

### 8.2 `src/config/constants.py`

```diff
@@ -137,9 +137,15 @@
 DEFAULT_MAX_DAILY_RISK = 0.06  # 6% maximum daily risk
 DEFAULT_MAX_CORRELATED_RISK = 0.10  # 10% maximum risk for correlated positions
-DEFAULT_MAX_DRAWDOWN = 0.20  # 20% maximum drawdown (fraction)
+# Board-ratified 2026-08-13: machine halt widened to 30% for the pre-edge research
+# phase. The HUMAN decision point stays at ~10% via the tightened escalation ratios
+# below — the cap and the human trigger are deliberately no longer the same number.
+DEFAULT_MAX_DRAWDOWN = 0.30  # 30% maximum drawdown (fraction)
 # Escalation tiers as fractions of the max-drawdown limit.
 # Must match src/config/risk-limits.json escalation.{warning,critical}_at_pct_of_limit.
-DRAWDOWN_WARNING_AT_PCT_OF_LIMIT = 0.50  # WARNING at 50% of the cap (10% drawdown at 0.20)
-DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT = 0.80  # CRITICAL at 80% of the cap (16% drawdown at 0.20)
+DRAWDOWN_WARNING_AT_PCT_OF_LIMIT = 0.35  # WARNING at 35% of the cap (10.5% drawdown at 0.30)
+DRAWDOWN_CRITICAL_AT_PCT_OF_LIMIT = 0.55  # CRITICAL at 55% of the cap (16.5% drawdown at 0.30)
 DRAWDOWN_GUARD_LOG_INTERVAL_SECONDS = 900  # Min seconds between repeated drawdown-tier logs
```

**Also recommended in the same sitting** — clears #986 item 2, a live divergence against the JSON's own "any divergence is a P0" rule (§6 item 6):

```diff
@@ -133,7 +133,7 @@
 DEFAULT_TAKE_PROFIT_PCT = 0.04  # 4% take profit
-DEFAULT_MAX_POSITION_SIZE = 0.1  # 10% max position size
+DEFAULT_MAX_POSITION_SIZE = 0.20  # 20% max position size (matches risk-limits.json)
 DEFAULT_BASE_RISK_PER_TRADE = 0.02  # 2% risk per trade
```

> Note: this second hunk loosens a default that is currently stricter than the ratified value. Prod overrides it explicitly (`--max-position 0.20`), so there is no live behaviour change — but it does relax the default for any invocation that does not pass the flag. If the Board prefers, the alternative resolution is to keep 0.10 and add a `$source_of_truth_note` exemption recording the intentional strictness. Either resolves the P0; the file's rule requires one of them.

### 8.3 `.claude/state/charter.md`

```diff
@@ -20,9 +20,11 @@
-High-level statement of appetite. The concrete numeric limits live in `risk-limits.json`; this is the *why*.
+High-level statement of appetite. The concrete numeric limits live in `src/config/risk-limits.json`; this is the *why*.
 
-- Maximum acceptable drawdown before human decides to halt: **20%** (matches `risk-limits.json` `max_drawdown_pct` — given the high risk appetite, the hard system limit doubles as the human decision point)
+- Maximum acceptable drawdown before the **human** decides to halt: **~10%** (`escalation.warning_at_pct_of_limit` 0.35 x `max_drawdown_pct` 0.30 = 10.5%). The **machine** close-only halt is a separate, wider backstop at **30%** (`src/config/risk-limits.json` `max_drawdown_pct`, raised from 20% on 2026-08-13 for the pre-edge research phase). These are deliberately no longer the same number: the wider machine cap buys research headroom without moving the point at which a human must decide. CRITICAL paging at 16.5%.
 - Maximum acceptable daily loss: **6%** (matches `max_daily_risk_pct`)
-- Maximum single-position exposure: **10%** of capital (matches `max_position_size_pct`); positions above 20% are flagged as large (`large_single_position_threshold_pct`)
+- Maximum single-position exposure: **20%** of capital (matches `max_position_size_pct`; production pins `--max-position 0.20`); positions at or above 20% are flagged as large (`large_single_position_threshold_pct`)
 - Leverage policy: **up to 3x on futures** (matches `max_leverage`); spot preferred, leverage used only when the strategy's signal and sizing justify it
-- On any breach: **halt new entries and page human** (matches `risk-limits.json` `escalation.breach_action`); existing positions run their own stop/exit logic unless the breach itself requires an emergency close
+- On any breach: **halt new entries and page human** (matches `src/config/risk-limits.json` `escalation.breach_action`); existing positions run their own stop/exit logic unless the breach itself requires an emergency close
```

**OPTIONAL (§6 items 4, 5, 8) — correct, but independent of this change:**

```diff
@@ -7,3 +7,3 @@
 ## Mission
 
-Grow $1000 live account 
+Reach demonstrated profitability first, across market conditions, then scale capital against evidence ([D-2026-08-13-02]). Live account today: ~$84.
@@ -12,3 +12,3 @@
-- Capital under management (USD): $1,000 paper, $87 live
+- Capital under management (USD): $1,000 paper, $83.64 live (prod account_history, 2026-08-13 15:47 UTC)
@@ -25,1 +25,1 @@
-- Leverage policy: **up to 3x on futures** (matches `max_leverage`); spot preferred, leverage used only when the strategy's signal and sizing justify it
+- Leverage policy: cap **3x** (matches `max_leverage`); production currently runs margin at `max_leverage=1.0` (leverage effectively disabled); spot preferred, leverage used only when the strategy's signal and sizing justify it
```

### 8.4 Deploy note

`constants.py` reaches production only on the next deploy, via `railway.json` `startCommand: atb live-health hyper_growth --max-position 0.20` (no `--max-drawdown` flag → uses the constant). Per C6, verify the guard re-arms from the durable session peak after that restart:

```
grep "Max-drawdown guard armed" <boot logs>
# expect: peak=$84.42-ish, hard cap=30.0%, "account_history peak $84.42" — NOT "unavailable"
```

---

## 9. Top failure modes introduced by this change

1. **Slow bleed runs ~5 additional months / ~5 additional full-size stop-outs before the halt, on a strategy with no demonstrated positive expectancy.** −$8.44 today, −$108 at the offered capital. *Early-warning signal*: the 10.5% WARNING tier (only if C1 is applied — at unchanged ratios the first signal is 15% and this failure mode is 5pp less visible). Secondary signal: rolling 30d realized P&L turning consistently negative while trade count stays low — visible in the standup snapshot.

2. **The 30% cap is measured from a reset peak and is not really 30%.** #1036 (P1, open) means a carry-forward boot can silently self-anchor the peak; #847 means a clean restart re-baselines it per-session. A deeper cap makes a peak-reset error proportionally more expensive — a reset at 25% drawdown now grants a further 30% from the new anchor instead of 20%. *Early-warning signal*: `"Max-drawdown guard armed"` boot log showing `account_history peak unavailable` or a peak equal to current balance. Assert positively on every deploy (C6); do not read silence as health.

3. **The breaker-arming work is de-prioritised because the headroom removes the felt urgency.** The 20% cap being close was part of what kept #1032/#1036 moving. This is the governance failure mode, and on this project's record it is the likeliest of the three. *Early-warning signal*: no movement on P1–P5 within 14 days of the raise. Concretely: `#1036` still open and `origin/main` still lacking `BreakerEquityFeed` two weeks from now.

---

## 10. What I could not verify

- **Regime-specific drawdown for a 30% cap.** No new backtests were run for this review (Board standing instruction: local training/heavy compute off-limits; backtests sequential). All drawdown figures are from the 2026-07-04 reproduction (`develop@e1d24239`, exact match to the originating session). I did **not** re-run the 365d window on `develop@102fceb9`, so post-#838/#843 engine drift since 2026-07-04 is unverified. The 2022 collapse and flash-crash regimes are **outside the available 365d ETHUSDT window** and were not tested at either cap — the "worst 3 historical regimes" requirement is only partially met, by the single 2025-07→2026-07 window (ETH hold −31.03%).
- **Whether a 30% cap would have changed any live outcome.** It would not have, on the record available: prod's honest-accounting history begins 2026-06-03, max session drawdown since is ~1.4%, and the P1 #845 20.33% "breach" is book-value, pre-sync — [D-2026-07-14-05] and the stress review's own correction establish that **no true-equity 20% breach can be demonstrated**. The task framing states prod "already breached 20% live once"; the ledger evidence says that reading was withdrawn as phantom-era book value. I flag the discrepancy rather than adopt either version — but the Board should not treat "we already breached 20%" as an established fact supporting the raise.
- **Whether `alert_webhook_url` is currently set in prod**, i.e. whether either escalation band would actually page a human. Not checked in this session (Railway env read not attempted). This is load-bearing for C1's value and should be confirmed at the sitting.
- **Expectancy under a 30% cap.** Unknowable. Every available estimate of this strategy's expectancy is negative or null; a cap change cannot improve it. This is the honest limit of what any risk review can say about the raise: I can bound the loss, I cannot find the benefit.
