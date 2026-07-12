# INPUT SCREENING (Linear) — Pre-registration

Date: 2026-07-12
Author: quant-researcher
Status: **PLANNED — pre-registration locked before any scoring run**
Issue: GH #959 (Phase 0, input-candidates audit), this doc is Lane A Phase 1
Related: `docs/research/2026-07-12_input-candidates-audit.md` (PR #958, open at time of
writing — branch `docs/input-audit`, not yet merged to develop; scripts cherry-picked into
this worktree directly since PR #958 was not blocked on), `docs/research/experiments/2026-07-10_target-redesign-tournament-results.md`
(fold definitions, linear-baseline methodology, merged to develop at `25e0a202`)
North star: the target-redesign report's conclusion, quoted verbatim: *"the feature set
itself — not the model, not the window, not the target shape — is the ceiling... The next
research lever this implies is new information sources."*

**No deep-model training accompanies this document.** This is a cheap CPU-only linear/logistic
screening gate — seconds per fit — whose sole purpose is to decide which of the audit's
shortlisted input classes earn a slot in the (expensive, GPU) deep-model input tournament.
Nothing here is proposed for staging or live capital.

---

## 1. Hypothesis

**H1 (per arm)**: Adding input class *k* to the price-only feature contract produces a linear
classifier whose next-bar directional accuracy (DA) is higher than the price-only control's DA,
by a margin that is both statistically significant (paired McNemar per fold) and practically
non-trivial (≥0.5pp averaged across F1–F3), on at least 2 of the 3 primary folds.

**H0 (per arm, the falsifier)**: Input class *k* adds no more OOS directional information than
the price-only control already has — any observed DA improvement is noise (fails the McNemar +
magnitude bar on at least 2 of 3 folds).

**Why a linear gate, not a straight jump to deep models**: three independent tournaments (window
#898, architecture #939, target-redesign, this doc's direct predecessor) each held the
price-only feature set fixed and varied a different lever, and all three found the lever wasn't
the binding constraint. A cheap linear/logistic screen on the SAME feature-addition question
(does this input help at all, linearly) is the appropriate next filter before spending GPU budget
on a deep architecture per input class — a mediocre-evidence input that a linear model can't even
detect a linear signal from is a weak candidate for a deep model to justify testing next, though
not proof a nonlinear model couldn't find something a linear one can't (named explicitly as a risk
of false negative in §5).

**Named risk of false positive** (per arm, generic): any of the 7 arms could show a spurious
"win" purely from added degrees of freedom (more coefficients to fit noise with), especially arms
whose extra features are themselves partially derived from price (Fear & Greed's known
circularity, per the audit). The Bonferroni correction (§4) and the ≥2-of-3-folds requirement are
the primary defenses; §5 names this explicitly per arm.

---

## 2. Model — same linear family as the tournament baseline

**Method chosen**: `sklearn.linear_model.LogisticRegression` (L2 penalty, `C=1.0`, `lbfgs`,
`max_iter=1000`) predicting binary next-bar direction, fit on the identical feature contract used
by the target-redesign tournament's linear/incumbent baseline
(`PriceOnlyFeatureExtractor(normalization_window=120)`, 5 columns — `close/volume/high/low/open`
rolling-min-max-normalized — flattened over a 120-bar sequence = 600 base inputs), standardized
(`StandardScaler`, fit on the training split only, applied to eval).

**Why logistic-on-direction rather than byte-for-byte `LinearRegression`-on-`close_normalized`**:
the target-redesign tournament's own linear-baseline script ran in an ephemeral worktree
(`.claude/worktrees/target-redesign-tournament`, detached at `6fc224c0`) that was never committed
to `develop` or to any PR — `methods.md`/`training_matrix.py` are not recoverable, only their
prose description in the results doc. Byte-for-byte reproduction is therefore not possible.
Logistic-on-direction is the explicitly-permitted alternative from this experiment's own dispatch
brief ("logistic regression on next-bar direction, **or** replicate the tournament's exact linear
method") and is the closest same-family analog: same feature contract, same
linear-model-family assumption, direct classification instead of a continuous-target-then-sign
reconstruction (which would have required guessing at an unrecoverable de-normalization step).

**Validity check (pre-committed, arm 0 only)**: the price-only control's averaged per-fold DA
must land within **±2.0 percentage points** of the tournament's reported linear-baseline DA
(53.24% / 53.61% / 53.18% on F1/F2/F3, `aggregate_stats_CORRECTED`). If any fold misses by more
than 2.0pp, this is reported as a **non-replication** in the results section, not silently
adjusted — all arm-vs-control comparisons in this doc still stand on their own (each arm is
compared to the SAME control run under the SAME method), but the "matches the tournament" claim
is withdrawn for whichever fold(s) miss, and flagged as a limitation rather than fixed by
changing the method after seeing the number (that would be exactly the post-hoc-threshold-move
the anti-p-hacking rule prohibits).

---

## 3. Folds — identical to the target-redesign tournament

Reused verbatim from `docs/research/experiments/2026-07-10_target-redesign-tournament-prereg.md`
§3 (same symbol, ETHUSDT, same timeframe, 1h):

| Fold | Train (cutoff) | Embargo | Eval window |
|---|---|---|---|
| F1 | 2017-08-17 → 2022-12-31 | 48h | 2023-01-03 → 2023-06-30 |
| F2 | 2017-08-17 → 2023-12-31 | 48h | 2024-01-03 → 2024-06-30 |
| F3 | 2017-08-17 → 2024-12-31 | 48h | 2025-01-03 → 2025-06-30 |

Expanding-window training (not rolling), matching #898's finding that full history ties-or-beats
shorter windows. The 48h embargo applies to every feature's lookback, not just the label: no
rolling/z-score/percentile statistic used at eval time may be fit using any data inside the
embargo gap or the eval window itself (§6 gives the per-input rule). F4 (the tournament's
confirmatory-only fold) is **not** included here — this is a screening gate, not a full
tournament, and F4 was explicitly non-deciding even there; adding a 4th fold here would only
inflate the multiple-comparison budget for no decision-relevant benefit.

---

## 4. Arms (pre-committed, no post-hoc additions or removals)

All arms 1–6 ADD their features to the arm-0 price-only contract; none replace it. Arm 7 adds
all of arms 1–6's features together.

| # | Arm | Extra features (beyond price-only) | Alignment rule |
|---|---|---|---|
| 0 | Price-only control | none (600-dim price-only contract only) | — |
| 1 | Multi-scale realized vol + range dynamics | rolling realized vol of log returns at 6h/24h/168h; Parkinson range estimator (24h window); single-bar HL-range-%-of-close and its 24h rolling mean | All rolling windows end at bar `t-1` (the last bar in the price-only sequence); computed on the full OHLCV frame, sampled only at `t-1`, never at `t` — a bar's still-forming high/low is never used to predict its own direction |
| 2 | Calendar/session | hour-of-day (sin/cos), day-of-week (sin/cos), hours-to-next-funding-settlement (00:00/08:00/16:00 UTC cycle) | Uses bar `t`'s OWN timestamp — a deterministic function of the exchange clock, known arbitrarily far in advance; this is the one arm with no lookahead surface by construction (per the audit) |
| 3 | BTC→ETH cross-asset | BTC lagged return (1h/6h/24h) and 24h realized vol, joined on BTC's own closed bar at ETH's `t-1` timestamp | BTC bar joined at ETH's `t-1`, never at `t` — BTC's `t-1` bar is closed well before ETH's `t` prediction is made, per the audit's explicit alignment warning |
| 4 | Funding rate (ETHUSDT perp) | last-settled funding rate level, rate-of-change vs previous settlement, 30-day rolling z-score (frozen: z-score mean/std fit on the training split only, applied unchanged to eval — the harness-wide no-eval-leakage rule from the target-redesign tournament), extreme-funding binary flag (|z|>2) | Forward-filled from the last **settled** print as of bar `t-1`'s timestamp — a bar at `t-1`=07:00 UTC sees the 00:00 print, never the 08:00 one; `markPrice`'s empty pre-2021 values are not coerced to 0.0 (excluded from any feature that would use them) |
| 5 | Basis / perp-spot premium proxy | premium-index CLOSE at `t-1`, and its own 24h rolling std | Uses only the closed bar's `close`-of-premium; never that bar's own `high`/`low` (those aren't known until the bar itself closes) |
| 6 | Fear & Greed | daily F&G value (forward-filled to hourly), lagged a full day past the print date, plus 7-day momentum and an extreme-value flag (<20 or >80) | Daily value for calendar day `D` is used only for bars on or after `D+1` — one full day of conservatism past the print, per the audit's alignment rule; tested as a marginal addition over the price/vol baseline specifically (not standalone), given F&G's own partial circularity with price |
| 7 | All-combined | arms 1+2+3+4+5+6's features together, added to price-only | Same per-input alignment rules as above, applied jointly |

**Explicitly excluded from this round** (per the audit's own recommendation, restated here so it
is pre-committed, not a post-hoc omission): open interest / long-short ratio (no free historical
depth past 30 days), on-chain exchange-flow/active-address metrics (no free source at the needed
depth), DXY/SPX/NDX macro (weak short-horizon evidence + non-trivial calendar-alignment cost).
`OnChainFeatureExtractor`, `MacroFeatureExtractor`, and the social-volume/news-sentiment parts of
`EnhancedSentimentExtractor` are **not used anywhere in this experiment** — per the audit's
headline finding, they are simulated no-ops and including them would silently invalidate any
"alternative data didn't help" reading.

---

## 5. Metrics, thresholds, and the graduation rule (pre-committed)

**Primary metric**: per-fold directional accuracy (DA) — plain classification accuracy of the
logistic model's `argmax` direction vs. `1[close[t] > close[t-1]]`, on that fold's eval window.

**Significance test**: per-fold **McNemar's test** (paired — every arm is scored on the identical
set of eval bars within a fold, so the correct test is paired, not an independent-samples
two-proportion z-test) comparing each arm's per-bar correctness vector against the price-only
control's, on the same fold.

**Multiple-comparison correction**: **Bonferroni, α = 0.05/7 ≈ 0.0071** across the 7 arms
(6 candidate arms + the all-combined arm; arm 0 is the reference, not itself tested against
itself). This is stricter than the target-redesign tournament's 6-pairwise-comparison correction
(0.0083) because this screen deliberately tests more candidates against one fixed reference.

**Graduation rule (pre-committed, exact numbers, not to move after seeing results)**: an input
class graduates to the deep-model input tournament if and only if, versus the price-only control:

1. McNemar p < 0.0071 (Bonferroni-corrected) on **≥ 2 of the 3 folds** (F1/F2/F3), AND
2. The **average** DA improvement across F1–F3 is **≥ +0.5 percentage points**.

Both conditions must hold. An arm that clears (1) but not (2) — statistically real but too small
to matter — does not graduate. An arm that clears (2) but not (1) — a nominally large gap that
isn't statistically distinguishable from noise at this bar count — does not graduate. This
exact wording is locked before any scoring run.

**Reported, never used to rank or gate** (per instruction 6):

- **Brier score** (mean squared error of the model's `P(up)` against the actual binary outcome),
  every arm, every fold.
- **DA restricted to the subset of eval bars where the arm's predicted direction disagrees with
  naive persistence** (`sign(close[t-1]-close[t-2])` continuing) — isolates whether an arm adds
  anything beyond trivial momentum-continuation, computed only where cheap (it is, here — no
  extra data, just a subset mask on already-computed predictions).

---

## 6. Leakage discipline (restated per input, all causal)

Every feature is computed strictly from data available at or before the bar it is attributed to,
with the specific rule stated per arm in §4's table. Two harness-wide rules apply across every
arm:

- **The 48h embargo gates every rolling/z-score statistic**, not just the label: any
  frozen-statistic feature (arm 4's funding z-score) is fit on the training split only and applied
  unchanged to the eval window — never refit or updated using eval-window data, matching the
  target-redesign tournament's harness-wide confidence-mechanism rule.
- **No same-bar high/low lookahead**: every range/volatility feature computed from a bar's own
  high/low uses that bar only after it is fully closed and only for bars strictly before the
  target bar `t` (i.e., sampled at `t-1`, never `t`) — CODE.md's top backtest-parity rule, applied
  to every new feature in this experiment, not just the ones with an external data source.

---

## 7. Data

- **ETHUSDT / BTCUSDT 1h OHLCV**: local cache (`cache/market_data/`, `CachedDataProvider` +
  `BinanceProvider`), already covering 2017-08-17 → present per `atb data cache-manager info`
  (verified this session, 34 cached files). No stale-cache concern — this experiment fetches
  through the exact fold cutoffs directly if not already cached.
- **ETHUSDT perp funding rate**: `fapi.binance.com/fapi/v1/fundingRate`, full history pull (not
  the audit's 30-day sample), paginated per the audit's proof-of-obtainability script pattern.
- **ETHUSDT perp premium index**: `fapi.binance.com/fapi/v1/premiumIndexKlines`, 1h granularity,
  full history pull, same pagination pattern.
- **Fear & Greed index**: `api.alternative.me/fng/?limit=0`, full history (already verified
  3,080 daily records, 2018-02-01 to present, by the audit).

All fetches are cached to disk under `scripts/research/.cache/` (gitignored) inside this
experiment's scripts, so re-running the screen does not re-hit any network endpoint.

---

## 8. Compute plan

Every fit is CPU-only `sklearn.LogisticRegression` on at most ~65k rows × ~610 features — seconds
per fit, `lbfgs` solver. Total: 7 arms × 3 folds = 21 fits, plus the naive-persistence baseline
(free, no fit). No GPU, no cloud, no SageMaker. Runs entirely inside this worktree
(`.claude/worktrees/input-screening-linear`), sequentially, no contention with the exit-geometry
lane's heavy-compute lock (not applicable here — nothing here is a full backtest or GPU job). If
any step turns out to need a full `atb backtest` run, that is explicitly out of this doc's scope
and will be coordinated with the PM before running, per the dispatch brief.

---

## 9. Decision this experiment feeds

This is a **screening gate**, not a strategy-change proposal. Its only output is a graduation
verdict per input class, handed to whoever scopes the next (deep-model) input tournament. No
arm's outcome here — pass or fail — authorizes any change to a live-affecting strategy, model, or
`risk-limits.json`. If every arm fails to graduate, that is itself a reportable, full-write-up
result (per the anti-p-hacking rule), not a silent gap — it would mean even a linear detector
finds no signal in any shortlisted input class, which is directly relevant to whether the "new
information sources" lever the target-redesign report recommended is worth GPU budget at all.

---

*Pre-registration locked at the above wording. Results appended below after the run, never by
editing the sections above.*
