# ML Target-Design Research — Informing the TARGET-REDESIGN Tournament

Date: 2026-07-07
Author: quant-researcher
Status: RESEARCH SURVEY (not an experiment — no backtest run here; informs the pre-registration
of the next tournament)
North star: `docs/architecture/model_evaluation_system.md` (open question #2: "Is next-bar price
the right target at all?"), `docs/research/experiments/2026-07-05_confidence-calibration.md`
(H0 supported — confidence channel information-free OOS; explicit redirect to target redesign)

## Why this document exists

Two independent, fully-preregistered experiments this cycle converged on the same wall:

1. **Window tournament** (#898, `2026-07-05_window-tournament.md`): training window choice
   (full history vs 3y vs 18m) is not the binding constraint — all three variants lose money
   OOS on the frozen 2026-01-01→2026-07-04 exam (-7.3% to -11.3%), and holdout RMSE ranked them
   *backwards* relative to OOS trading P&L.
2. **Confidence calibration** (#912, `2026-07-05_confidence-calibration.md`): the raw
   `predicted_return` magnitude from the current next-bar price-regression target carries **zero**
   OOS directional-accuracy information (Cochran-Armitage Z=+0.43, p=0.669 on the frozen exam,
   vs. a spurious p=0.019 in training-period-adjacent data — textbook overfitting of the
   confidence channel to the training distribution, not a real signal). No recalibration of the
   confidence *mapping* can fix this; the raw signal itself doesn't carry the information.
3. **Exit-geometry sweep** (`2026-07-04_hypergrowth-exit-geometry.md`): tightening
   stop-loss/trailing-stop parameters made every variant *worse* on every window, because with
   entries effectively noise (root-caused there to a since-fixed cross-symbol bug, #867), cutting
   losers earlier just crystallizes noise excursions faster. **The exits are the symptom, not the
   disease** — but the task brief's headline number (78% win rate, PF 0.69, "small wins, few
   large losses") describes exactly the shape meta-labeling and triple-barrier labeling exist to
   fix, so it is worth re-testing this diagnosis with a signal that has been trained on a target
   that actually encodes stop/target mechanics, not retested with parameter sweeps alone.

The current setup — CNN-LSTM regression on next-bar normalized close, `predicted_return =
(prediction - current_price) / current_price`, confidence `= clip(|predicted_return| × 12.0, 0,
1)` — is diagnosed, not just suspected, to be the wrong target. This document surveys what the
practitioner/academic literature and the two most relevant public trading-bot ecosystems
(Freqtrade/FreqAI, freqst.com) actually do and actually show, then proposes a ranked shortlist for
the next tournament.

**Scope note**: this is a literature/ecosystem survey, not a backtest. No code was run. All claims
below are cited; where a source is thin (single vendor blog, single unreplicated preprint,
self-reported leaderboard), that is flagged explicitly rather than presented as settled evidence.

---

## (a) Survey findings by angle

### Angle 1 — Freqtrade / FreqAI target conventions

FreqAI's target-definition contract is simple and well-documented:

- Targets are declared in a strategy's `set_freqai_targets(dataframe, metadata, **kwargs)` method;
  any column prepended with `&` is a training target ("label"). Features are separately prepended
  with `%`. [Freqtrade FreqAI configuration docs](https://docs.freqtrade.io/en/stable/freqai-configuration/),
  [FreqaiExampleStrategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/FreqaiExampleStrategy.py)
- `label_period_candles` (a `feature_parameters` config key) sets the forward horizon. The
  official example target is:
  ```python
  dataframe["&-s_close"] = (
      dataframe["close"]
      .shift(-label_period_candles)
      .rolling(label_period_candles)
      .mean()
      / dataframe["close"] - 1
  )
  ```
  i.e. a **smoothed forward return**, not a raw single-bar future close — the rolling-mean
  smoothing is FreqAI's own answer to a version of our exact problem (raw future-price regression
  is noisy at a single horizon; averaging several forward bars is FreqAI's cheap denoising move,
  at the cost of a fuzzier "when does the fill actually happen" semantics). [Source](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/FreqaiExampleStrategy.py)
- **Classifiers** use string class labels, e.g. `df['&s-up_or_down'] = np.where(df["close"].shift(-100)
  > df["close"], 'up', 'down')`, with `self.freqai.class_names` declared explicitly. [Freqtrade docs](https://www.freqtrade.io/en/stable/freqai-configuration/)
- FreqAI also auto-computes `&*_std` / `&*_mean` — the statistical distribution of each target
  over the most recent training window — specifically so strategies can build **dynamic,
  distribution-relative thresholds** instead of a hardcoded constant. This is directly relevant:
  our confidence formula's `×12.0` multiplier is exactly the kind of hardcoded, uncalibrated
  constant FreqAI's own convention is designed to avoid. [Source](https://docs.freqtrade.io/en/stable/freqai-configuration/)
- **Community strategies**: the most substantive public FreqAI+deep-learning example found,
  [Netanelshoshan/freqAI-LSTM](https://github.com/Netanelshoshan/freqAI-LSTM), does **not** train
  the LSTM to regress raw future price. It trains an `LSTMRegressor` on a hand-engineered
  **composite "target score"** — a weighted combination of multiple technical factors
  (volatility-adjusted, regime-aware, dynamically weighted), with entry/exit thresholds applied
  to the predicted score. [README](https://github.com/Netanelshoshan/freqAI-LSTM/blob/main/README.md),
  [author's writeup](https://netanel.io/posts/freqai_lstm_reg/). This is a real-world signal that
  practitioners who ship FreqAI+LSTM strategies do not trust a raw price-regression target enough
  to use it directly — they pre-shape the target into something more directly tradeable
  (volatility-adjusted, multi-factor) before ever handing it to the model. This is anecdotal (one
  repo, unverified live track record) but directionally consistent with our own finding that raw
  next-bar-price regression under-delivers.
- Freqtrade's own hyperopt documentation warns explicitly against exactly the failure mode this
  desk already guards against: "values more precise than the default settings will usually result
  in overfitted results," and community discussion threads describe brute-force hyperopt finding
  "peaks" that fit in-sample trades perfectly but "you'll never find these trades in the future."
  [Freqtrade hyperopt docs](https://www.freqtrade.io/en/stable/hyperopt/), [GitHub issue #2472 discussion](https://github.com/freqtrade/freqtrade/issues/2472)

### Angle 2 — freqst.com survey

[freqst.com](https://freqst.com/) is a public leaderboard/archive of community Freqtrade
strategies with backtest metrics attached.

- **Strategies observed** (top-10-this-week + newly-added lists at fetch time): `ichiV1`,
  `HarmonicDivergence`, `Babico_SMA5xBBmid`, `TheForce`, `Ichimoku_v37`, `keltnerchannel`,
  `WaveTrendStra`, `EMA520015_V17`, `Slowbro`, `NostalgiaForInfinityXw`, `MultiMA_TSL3`,
  `MacheteV8b`, `Ichimoku_SenkouSpanCross`, `Schism2`, `BB_RPB_TSL_Tranz`, `NfiNextModded`,
  `NASOSv5_mod3`, `NowoIchimoku5mV2`, `NostalgiaForInfinityNextV7155`, `RSI`.
- **ML/FreqAI usage: effectively none of the surveyed strategies are explicitly ML-based.** Every
  name maps to a classic technical-indicator family (Ichimoku, Bollinger Bands, moving-average
  crossovers, RSI, harmonic patterns). This is a genuinely useful negative finding for angle 2: the
  most visible public leaderboard of "top" Freqtrade strategies is **not** where FreqAI/ML
  adoption evidence lives — it's a graveyard of parameter-fit technical-indicator strategies.
- **Evaluation methodology is opaque and should be treated with active skepticism**: the site
  ranks by "% Profit Month" with win/loss counts, but discloses no backtest period, no
  out-of-sample/in-sample split, no hyperopt-vs-final-backtest distinction, and no mention of fee
  assumptions. The site itself states these strategies "are no longer actively maintained and
  will be archived" — i.e. this is explicitly a historical curiosity cabinet, not a maintained,
  validated benchmark.
- **Skeptical read**: overfit leaderboards are the norm in this space, exactly as the task brief
  anticipated. A strategy with a headline "% Profit Month" number and zero methodology disclosure
  is indistinguishable from a hyperopt-fit curve that will not reproduce out-of-sample — this is
  the same failure signature our own charter guards against (`model_evaluation_system.md`
  principle 2: "leakage is the default, honesty is engineered"). **No claim from this site should
  inform target design beyond the negative finding that ML is not what wins on public technical-
  strategy leaderboards** — it says nothing about whether ML *could* win with a better target,
  only that the current pool of publicly-shared "winners" isn't ML-driven.

### Angle 3 — Practitioner/academic label design

- **Triple-barrier labeling** (López de Prado, *Advances in Financial Machine Learning*, 2018,
  Ch. 3): label a bar by whichever of three barriers is hit first — an upper profit-taking
  barrier, a lower stop-loss barrier, or a vertical time-expiry barrier — using intra-bar
  high/low, not just the close, to check barrier touches. This directly encodes the trading
  mechanics (stop-loss %, take-profit %, max holding period) into the label itself, instead of
  measuring return at an arbitrary fixed future point regardless of what happened in between.
  [Reasonable Deviations chapter notes](https://reasonabledeviations.com/notes/adv_fin_ml/),
  [mlfinpy labelling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html),
  [O'Reilly chapter excerpt](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml)
- **Meta-labeling** (same source): a two-stage design — a high-recall **primary** model/rule
  decides direction (side: {-1, 0, +1}), and a **secondary** binary classifier decides whether to
  actually take that bet (and, via its output probability, how large), explicitly separating "is
  there a signal" from "should I bet on it and how much." [Wikipedia summary](https://en.wikipedia.org/wiki/Meta-Labeling) states this
  was formalized by López de Prado (2018) as "corrective AI."
- **Trend-scanning labels** (López de Prado, *Machine Learning for Asset Managers*): instead of a
  fixed horizon, fit a trend line (or several, at multiple forward-looking window lengths) and
  label by the **t-value** of the best-fit slope — sign gives direction, magnitude gives
  confidence/sample weight, and there is no fixed profit/stop % to hand-tune. Implemented in the
  open-source `mlfinlab` library. [LinkedIn/Hudson & Thames summary](https://www.linkedin.com/posts/hudson-thames-quantiative-research_trend-scanning-with-mlfinlab-developed-activity-7110593515889307648-TaKH),
  [mlfinlab repo](https://hudsonthames.org/mlfinlab/). **Caveat**: this is less mature tooling than
  triple-barrier — open GitHub issues on `mlfinlab` document real bugs in live/production use
  (e.g. "trend_scanning_labels.t_value is negative and huge for sample weights,"
  [issue #433](https://github.com/hudson-and-thames/mlfinlab/issues/433); a separate live-market
  bug report, [issue #551](https://github.com/hudson-and-thames/mlfinlab/issues/551)) — flag as an
  implementation-risk concern, not just a theoretical one.
- **Fixed-horizon vs. triple-barrier, empirical comparison**: a 2025 arXiv paper testing
  triple-barrier labeling with an LSTM on **all KOSPI/KOSDAQ-listed Korean stocks, daily OHLCV,
  2006–2024**, with a genuinely held-out 2022-09→2024-12 test period, found:
  - Optimal barriers (tuned on train/val, not test): 29-day horizon, ±9% barrier width, giving a
    label distribution of ~36% time-limit / ~35% take-profit / ~29% stop-loss.
  - **Out-of-sample 3-class accuracy: 43.28% (LSTM), 43.11% (XGBoost on technical indicators),
    vs. a 35.39% dummy-classifier baseline** — a real, non-trivial, but modest lift (~8pp over
    chance on a 3-class problem), AUC ≈0.62.
  - **No trading returns, drawdown, Sharpe, or profit factor were reported at all** — the paper
    evaluates classification metrics only. This is an important, honest data point: even the
    "gold standard" label design, done carefully with a genuinely unseen multi-year test period
    across an entire market, produces a modest classification lift with **zero evidence it
    translates to money**, and the paper itself does not attempt that translation.
  - The paper does **not** compare against a fixed-horizon baseline at all, despite testing
    triple-barrier extensively — a real gap; "triple-barrier beats fixed-horizon" is not
    demonstrated here, only asserted in secondary/blog sources.
  [arXiv:2504.02249](https://arxiv.org/html/2504.02249v2)
- **Volatility-normalized returns**: a 2025 arXiv paper (ReVol) proposes normalizing price
  features (return, volatility, price scale) per-instance specifically to combat distribution
  shift in stock-price data, reporting an average IC improvement of >0.03 and Sharpe-ratio
  improvement of >0.7 versus other normalization schemes across backbone models.
  [arXiv:2508.20108](https://arxiv.org/abs/2508.20108). **Extrapolation risk for us**: this
  evidence comes from **cross-sectional, multi-asset equity** setups (an attention module learns
  to down-weight noisy per-instrument observations *relative to other instruments in the same
  training batch*) — our setup is a **single symbol, single time series** with no
  cross-sectional dimension, so the specific mechanism that produces ReVol's reported gains
  (relative down-weighting across a instrument panel) may not transfer. The generic idea (predict
  return in units of realized volatility, not raw %) is still worth testing on its own merits,
  independent of ReVol's specific architecture.
- **Quantile / distributional targets**: quantile regression for financial returns has a long
  academic history (portfolio construction, VaR) but is future-facing for retail crypto bots — the
  most directly relevant recent work, "FutureQuant Transformer" ([arXiv:2505.05595](https://arxiv.org/html/2505.05595)),
  proposes predicting the **distribution** of forward returns in futures markets and using
  predicted quantiles to directly size/place stops — a single, unreplicated preprint, not
  production evidence. Treat as a promising but unproven direction, not a "known good" target.
- **Horizon selection for 1h crypto specifically**: no source found gives a crypto-specific,
  hourly-timeframe-specific horizon recommendation with real forward-tested evidence; every
  concrete empirical number found above is either daily-equity (Korean paper) or cross-sectional
  equity (ReVol) or a single unreplicated preprint (FutureQuant). **This is itself a finding**: the
  literature has essentially no direct, replicated evidence for "what horizon works for hourly
  crypto" — any horizon choice for our tournament is necessarily an empirical bet to be tested on
  our own frozen exam, not something to import from a paper.

### Angle 4 — Meta-labeling as a fit for our specific failure mode

The task brief's headline (78% win rate, PF 0.69 — small wins, few large losses) is exactly the
textbook meta-labeling use case: a primary signal fires often and is directionally right most of
the time, but the bet-*sizing*/bet-*taking* decision doesn't discriminate the trades worth taking
large from the ones that end up net-negative. That maps cleanly onto our own
`SignalGenerator`/`RiskManager`/`PositionSizer` split (`docs/architecture.md`): the primary
model stays the `SignalGenerator` (direction/side), and a secondary meta-model's output feeds the
`RiskManager` (gate) and/or `PositionSizer` (size), exactly the seam the architecture already
exposes — no new component type needed, just a new signal flowing into the existing sizer/risk
interfaces.

**However — a hard, self-diagnosed caveat that must not be glossed over**: the
confidence-calibration study (#912) already tested a *degenerate, one-feature version* of
meta-labeling — a secondary "model" that is just a threshold on `|predicted_return|` itself (the
existing raw signal's own magnitude) — and found it **information-free OOS** (H0 supported,
p=0.669 on the frozen exam). A meta-labeling model that reuses only the same scalar the primary
model already outputs is not meaningfully different from what was just falsified. For meta-
labeling to have a chance of adding real information, the secondary model's **feature set must be
genuinely richer** than the primary signal's own magnitude — e.g. realized-volatility regime,
recent rolling hit-rate, time-of-day/session, feature-store context already computed for the
primary model, and (per triple-barrier) whether the *actual* stop/target mechanics would have been
hit, not just the sign of a next-bar delta. If a meta-labeling tournament entrant is built by
just re-deriving a smarter threshold on the same `predicted_return` scalar, it should be expected,
on the balance of our own evidence, to fail the same way — this should be stated as a pre-
registered risk in that entrant's hypothesis file, not discovered again the hard way.

The other necessary condition, also self-diagnosed: **meta-labeling can only filter/size an
existing signal — it cannot manufacture edge that isn't there.** If the primary `SignalGenerator`
truly has ~zero OOS directional edge (an open question — Phase 2 falsified the *confidence*
channel, not the base 51.85% raw hit rate, which is close to a coin flip but not conclusively
proven to be exactly 50%), a meta-model has nothing real to filter and will, at best, filter noise
by chance on the training data and fail OOS. This is the primary risk to name upfront in that
entrant's pre-registration (per `experiment-preregister`'s "risks of false positive" section).

### Angle 5 — What actually correlates with live profitability

- A 2025 MDPI paper on a confidence-threshold framework for crypto direction prediction reports
  82.68% directional accuracy **on executed trades only**, at 11.99% market coverage (i.e. the
  model abstains on ~88% of bars and only trades the ~12% it's most confident about), with 151.11
  bps average net profit per executed trade, and states results are exactly reproducible across
  hardware. [MDPI 2076-3417/15/20/11145](https://www.mdpi.com/2076-3417/15/20/11145). This is
  structurally the single most relevant external data point to our situation: it is direct
  evidence that a **coverage/accuracy tradeoff** (abstain on low-confidence bars, trade only the
  high-confidence tail) is a workable strategy shape *if the confidence signal is real* — which is
  precisely the thing our own confidence channel currently is not. This paper's own reported
  87%+ accuracy at low coverage is a single self-reported result without independent replication
  found in this survey; treat the specific numbers with real skepticism (no OOS-vs-training split
  detail could be confirmed from the abstract alone), but the **shape of the claim** (accuracy
  buys a coverage tradeoff, not a fixed threshold) is directly actionable design guidance.
- Broader directional-accuracy benchmarks found in this survey cluster in the 48-65% range for
  crypto/equity ML models (Random Forest ~65% on ETH pairs per one source, 43% 3-class accuracy
  in the Korean triple-barrier paper, our own 51.85% raw hit rate) — **there is no single
  "magic threshold" reported anywhere in this survey above which profitability reliably follows.**
  Every source that reports both accuracy and profitability separately (the Hudson & Thames
  meta-labeling case study being the clearest) shows profitability moving with **precision on the
  trades actually taken**, not raw directional accuracy across all bars — consistent with the
  "coverage" framing above and with meta-labeling's whole premise.
- **Capacity/turnover**: no crypto-specific, hourly-timeframe capacity study was found in this
  survey (a real gap — flagged, not filled with a weak citation). General finance literature
  confirms turnover/fee drag scales directly with trade frequency and that fragility should be
  measured by sensitivity to fee/slippage/delay assumptions — this is already exactly our own
  `CostCalculator`-on-by-default discipline (`CODE.md`, this desk's hard rules) and needs no new
  practice, just a reminder that any target reformation that increases trade *count* materially
  (e.g. a shorter effective horizon, or a classifier that fires more often than the current
  ~50-trade/185-day baseline) must re-verify that fees/slippage as a % of gross return haven't
  silently eaten the reported edge.
- **Honest summary for angle 5**: the replicated, credible signal in this literature is "trade
  less, but trade the trades you're actually right about" (coverage/precision tradeoff via
  abstention or meta-labeling), not "find a bigger/better raw accuracy number." Our exam metrics
  contract (`model_evaluation_system.md`'s L2 metrics: return, PF, MaxDD, win rate, trade count,
  Sharpe, confidence distribution, regime slices) already captures the right axes; the only
  addition this survey motivates is **explicitly tracking accuracy-vs-coverage as a curve**, not a
  single point, for any classifier/probability-output candidate — i.e. report OOS metrics at
  several candidate abstention thresholds, not just one.

---

## (b) Ranked shortlist — 4-6 candidate target formulations for the TARGET-REDESIGN tournament

Ranked by (evidence quality × fit to our diagnosed failure mode × implementation cost). All six
should still run through the full L1→L2 pipeline; rank reflects priority/expected value, not a
recommendation to skip the lower-ranked ones outright — the north star's own rule 6 says a dumb
baseline must run in every tournament regardless of rank.

### 1. Meta-labeling secondary classifier (on top of the current primary signal)

- **Label definition**: for every bar where the (existing or reformed) primary `SignalGenerator`
  would have fired a non-HOLD signal, label = 1 if simulating that trade through the actual
  HyperGrowth exit mechanics (10% stop / 30% take-profit / partial-exit ladder / trailing stop,
  using intrabar high/low, matching `src/engines/shared/`) would have closed net-profitable after
  fees, else 0. Feature set: **must include, at minimum**, trailing realized volatility (48-bar,
  matching the vol_zscore_gate variant that showed the only directionally-favorable — if
  sub-threshold — result in #912), recent rolling hit-rate of the primary signal, time-of-day/
  session bucket, and the primary model's own predicted-return magnitude (as one feature among
  several, never the only one).
- **Model output type**: calibrated probability `P(trade profitable | signal fired)` (binary
  classifier, e.g. gradient-boosted trees or logistic regression — start simple, per the
  north-star's "always include dumb baselines" rule applied within this entrant too).
- **Strategy consumption**: feeds `RiskManager` (gate: only take signals above a probability
  threshold chosen from the OOS reliability curve, not an arbitrary constant) and optionally
  `PositionSizer` (size scaled by probability, replacing today's `adjust_for_confidence=False`
  no-op for HyperGrowth specifically — turning this on is an explicit, separate parameter change
  that itself needs its own sensitivity pass per `CODE.md`).
- **Expected failure modes**: (1) if the primary signal has ~zero real edge, the meta-model has
  nothing to filter and will overfit its own training window; (2) reusing only
  `|predicted_return|` as the dominant feature reproduces the already-falsified #912 result almost
  exactly — must be pre-registered as a named risk; (3) meta-labeling needs enough primary-signal
  trade events to train on — with ~50 trades per 185-day window, the meta-model's own training set
  may be too small; consider training the meta-model on Phase-1-style triple-barrier-labeled bars
  (not just realized trades) to get a larger training set, then validating on realized trades OOS.
- **Evidence supporting**: López de Prado's foundational proposal ([Wikipedia](https://en.wikipedia.org/wiki/Meta-Labeling));
  a vendor case study reporting OOS accuracy 48%→55% (trend-following) and 17%→63% (mean
  reversion), Sharpe 0.67→1.42 (mean reversion) after adding a meta-model
  ([Hudson & Thames](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/))
  — flagged explicitly as a small-sample, vendor-published result (the vendor sells the
  `mlfinlab` library this technique is built on), not independently replicated evidence, but
  directionally consistent with the theoretical mechanism and with the coverage/precision pattern
  in angle 5.

### 2. Binary fixed-horizon direction classification (the required baseline arm)

- **Label definition**: `y = 1 if close[t+H] > close[t] else 0` for a fixed horizon `H` (start at
  `H=1`, i.e. the same horizon as today's regression, to isolate "does changing loss function
  alone help" from "does changing horizon help").
- **Model output type**: calibrated `P(up)` (sigmoid/softmax output), directly a legitimate
  confidence signal by construction — unlike today's `|predicted_return| × 12.0` proxy, a
  well-calibrated classifier probability doesn't need an arbitrary multiplier at all.
- **Strategy consumption**: `SignalGenerator` converts `P(up) > 0.5 + margin` → long,
  `P(up) < 0.5 - margin` → short; `confidence = |P(up) - 0.5| × 2`, a natural [0,1] scale with no
  free constant to mistune.
- **Expected failure modes**: class imbalance is mild at H=1 (roughly 50/50 for hourly crypto
  returns) but classifier probabilities from neural nets are frequently **not calibrated**
  out-of-the-box (this is exactly the mechanism that produced our current bug in a different
  guise) — a calibration check (reliability diagram / Brier score on the frozen exam) must be
  a required L2 metric for this candidate, not an afterthought.
- **Evidence supporting**: this is FreqAI's own documented convention for classifier targets
  ([Freqtrade docs](https://www.freqtrade.io/en/stable/freqai-configuration/)) and the cheapest,
  lowest-risk reformation to test — same horizon, same features, only the loss function and
  output head change. Directly required by `model_evaluation_system.md` principle 6 (dumb
  baselines in every tournament) applied to *this* tournament's specific question.

### 3. Triple-barrier ternary classification

- **Label definition**: for each bar, simulate forward with an upper barrier at
  `+take_profit_pct`, lower barrier at `-stop_loss_pct` (start from HyperGrowth's own live values,
  10%/30%, or a vol-scaled variant — see risk below), and a vertical time barrier at
  `max_holding_hours` (matches `risk-limits.json`'s `operational.max_holding_hours=336`); label =
  {+1, -1, 0} for whichever barrier is hit first, using intrabar high/low (no look-ahead — matches
  our own exit-handler's existing high/low-based fill logic, `src/engines/backtest/execution/
  exit_handler.py`).
- **Model output type**: 3-class probability distribution.
- **Strategy consumption**: `argmax` class → direction, `P(argmax class)` → confidence; because
  the label already encodes the *actual* stop/target mechanics, this is a more honest match to
  "will this trade, as HyperGrowth actually executes it, work" than a generic next-bar-price
  target — directly relevant to the exit-geometry finding (78% win rate / PF 0.69) since the label
  itself is now aware that a large loss is possible before a small win locks in.
- **Expected failure modes**: barrier-width selection is itself a tunable hyperparameter with
  p-hacking risk — must be chosen from training-period data only, exactly like #912's Phase 3
  discipline, and held fixed before touching the exam; heavy class imbalance toward the
  time-barrier class is possible at 1h/336h max-hold (need to check empirically, not assume);
  requires a genuinely separate simulation pass at training-label time (compute cost, correctness
  risk — must reuse `src/engines/shared/` fill logic per backtest-live-parity, not a hand-rolled
  reimplementation).
- **Evidence supporting**: the foundational López de Prado proposal
  ([Reasonable Deviations notes](https://reasonabledeviations.com/notes/adv_fin_ml/)); the most
  rigorous empirical test found in this survey (Korean stock LSTM, 2006-2024, genuinely held-out
  2022-2024 test period) shows a real but modest lift (43.3% vs 35.4% dummy 3-class accuracy) —
  **explicitly caveated**: that paper reports **no trading P&L at all**, so "triple-barrier
  produces better classification metrics" is evidenced; "triple-barrier produces better trading
  returns" is not evidenced by that source and must be established on our own exam.
  ([arXiv:2504.02249](https://arxiv.org/html/2504.02249v2))

### 4. Volatility-normalized return regression (ReVol-style, single-instrument variant)

- **Label definition**: `y = (close[t+1] - close[t]) / close[t] / realized_vol[t]` where
  `realized_vol` is a trailing (e.g. 48-bar) realized volatility of bar-over-bar returns — the
  same normalization the #912 `vol_zscore_gate` variant used post-hoc on the *existing* model's
  output, but here applied at training time as the actual regression target instead of retrofitted
  onto a price-regression model's residual.
- **Model output type**: continuous z-score (units of "how many sigma of a move").
- **Strategy consumption**: sign → direction; `|z|` → confidence directly, with no arbitrary
  multiplier (the vol-normalization itself does the calibration work).
- **Expected failure modes**: this changes *units*, not necessarily *information content* — Phase
  2 of #912 showed the failure wasn't literally "wrong units," it was "no OOS relationship between
  magnitude and hit-rate at all." If the underlying regression target still has no OOS
  magnitude-accuracy relationship, expressing it in vol-normalized units doesn't manufacture one.
  This candidate is a genuine, cheap experiment worth running, but should be pre-registered with
  that exact null-hypothesis framing (does normalizing the target change whether magnitude
  predicts accuracy, not just whether it "looks" more calibrated).
- **Evidence supporting**: ReVol reports avg IC improvement >0.03 and Sharpe improvement >0.7 vs.
  other normalizations ([arXiv:2508.20108](https://arxiv.org/abs/2508.20108)) — but that evidence
  is from **cross-sectional multi-asset equity** panels where the normalization interacts with an
  attention mechanism across instruments; our single-symbol setup lacks that cross-sectional
  dimension, so this is a plausible-but-unproven extrapolation, flagged as such.

### 5. Quantile / distributional forward-return regression

- **Label definition**: multiple quantiles (e.g. p10/p50/p90) of the forward N-bar return
  distribution, trained via pinball/quantile loss; or a parametric (mean, vol) head.
- **Model output type**: a small vector of quantile values (or distribution parameters), not a
  scalar.
- **Strategy consumption**: rather than gating entries, this would reshape `RiskManager`'s
  stop-loss/take-profit placement to the model's own predicted dispersion at trade time (e.g. set
  the stop near the predicted p10 and target near p90) — a genuinely different lever than any
  other candidate here: it attacks the exit-geometry problem directly (the diagnosed 78%-win/
  PF-0.69 asymmetry) from the exit-placement side, rather than the entry-filtering side that every
  other candidate above uses.
- **Expected failure modes**: highest implementation complexity of the six (multi-output head,
  quantile-crossing checks, more surface area for a parity bug between backtest and live exit
  logic); thinnest evidence base — the closest analog found is a single unreplicated 2025 preprint
  on futures markets ([arXiv:2505.05595](https://arxiv.org/html/2505.05595)), not production
  evidence.
- **Evidence supporting**: theoretically the most direct fit to the diagnosed problem (reshape
  exits to match predicted dispersion, rather than another entry-side confidence proxy), but the
  evidence base is the weakest of the six — recommend as a **second-round** candidate (build it
  only if round-1 candidates above establish that *some* reformed target carries real OOS
  information at all; a distribution head trained on a target with no OOS signal is not more
  useful than a point estimate with no OOS signal).

### 6. Trend-scanning labels (flagged, lowest priority for this tournament)

- **Label definition**: t-value of the best-fit trend line over a variable forward-looking
  window (multiple candidate window lengths, pick the one with max |t-value| at each bar, per
  López de Prado's method); sign → direction, |t| → confidence/sample weight.
- **Model output type**: continuous (t-value) or classification (sign) with continuous weight.
- **Strategy consumption**: naturally aligned with HyperGrowth's `ignore_signal_reversal=True`
  (hold-through-flip) behavior, since a genuine multi-bar trend label is less likely to flip on
  single-bar noise than a fixed 1-bar-ahead target.
- **Expected failure modes**: implementation immaturity — the reference open-source
  implementation (`mlfinlab`) has documented live/production bugs including degenerate t-value
  edge cases ([issue #433](https://github.com/hudson-and-thames/mlfinlab/issues/433),
  [issue #551](https://github.com/hudson-and-thames/mlfinlab/issues/551)); computationally heavier
  (multiple regression fits per bar); would require a from-scratch implementation in
  `src/engines/shared/` rather than reusing a battle-tested library, raising both cost and
  parity-bug risk.
- **Evidence supporting**: theoretically elegant, foundational source (López de Prado,
  *Machine Learning for Asset Managers*), but no independent empirical replication found in this
  survey beyond the library's own documentation. Recommend deferring to a future tournament after
  the cheaper candidates (1-4) have been tried and, ideally, after core exit-geometry work is
  informed by whichever of those wins.

**Top-3 for a first tournament round, if forced to pick**: **(1) meta-labeling** (directly targets
the diagnosed 78%-win/PF-0.69 asymmetry, reuses existing architecture seams), **(2) binary
direction classification** (cheapest, required baseline, fixes the "arbitrary constant" root cause
of the confidence bug by construction), and **(3) triple-barrier ternary classification** (most
rigorous single-model reformation, encodes actual exit mechanics into the label, best independent
evidence quality among the pure reformations). Candidates 4-6 are legitimate but lower-priority;
5 and 6 in particular should wait for round 1's results.

---

## (c) Tournament design recommendations (compatible with `experiment-preregister`)

1. **New frozen exam window — do not reuse 2026-01-01→2026-07-04 again.** That window has already
   served 7 candidates across two tournaments (3 in #898, 4 in #912) against the north star's own
   "~10 candidates then rotate" multiple-comparison budget
   (`model_evaluation_system.md`). Adding 5-6 more target-redesign candidates on the same window
   would push past 12-13 total — pick a new cutoff/eval window (e.g. training data through
   2026-04-30, eval 2026-05-01→2026-07-07, or similar) so this tournament's numbers aren't
   competing for luck-budget against the two already run.
2. **Confirm the non-determinism fix (#913, closed) actually holds before trusting any new
   number.** Per `experiment-preregister`'s "determinism guard" rule, re-run the first candidate's
   exam twice before trusting the tournament's results — #913 (inference-timeout-driven backtest
   non-determinism, closed 2026-07-06) is exactly the kind of infrastructure bug that would
   quietly corrupt a fresh multi-candidate tournament the same way it corrupted #912's Phase 3.
3. **Structure as two rounds, not one flat comparison**:
   - **Round 1 (primary target reformation)**: naive-persistence baseline + linear baseline
     (north-star principle 6) + current price-regression (incumbent control) + candidate #2
     (binary classification) + candidate #3 (triple-barrier) + candidate #4 (vol-normalized
     regression) — 6 entrants, all through L1 (temporal holdout sanity) then L2 (shared frozen
     exam), same protocol as #898/#912 (prod-matched risk params, `CostCalculator` on, ≥30 trades
     minimum, Wilson CIs + trend tests for any magnitude-vs-accuracy claim).
   - **Round 2 (meta-labeling)**: only proceeds if at least one Round 1 candidate clears a
     pre-registered minimum bar (e.g., beats naive persistence on both OOS return and PF, per
     `model_evaluation_system.md`'s existing gate language) — meta-labeling (candidate #1) is
     built on top of Round 1's winning primary signal. If **no** Round 1 candidate clears the bar,
     the honest, reportable conclusion is "target reformation alone does not create OOS edge" and
     Round 2 should not run (per the risk named in angle 4 — meta-labeling on a signal with no
     edge has nothing to filter).
4. **Explicit numeric success threshold, pre-committed before any run** (do not let this appear
   for the first time in a results section, per the `experiment-preregister` red-flag list):
   suggest, consistent with #912's threshold shape, "OOS return improves by ≥3pp vs. the current
   incumbent baseline on the shared exam AND MaxDD does not worsen by >2pp AND trade count >30
   AND (for any classifier candidate) a reliability/calibration check — Brier score or a binned
   reliability diagram — shows real separation between predicted-probability bins, not just a
   headline accuracy number." The calibration check is new relative to #898/#912's thresholds and
   is added specifically because this tournament's whole premise is fixing a calibration failure.
5. **Report accuracy-vs-coverage as a curve for every classifier/probability candidate**, per
   angle 5's finding — at minimum, OOS return/PF/trade-count at 3-4 candidate abstention
   thresholds (e.g. trade top 10%/25%/50%/100% of bars by predicted confidence), not a single
   fixed threshold chosen after the fact.
6. **Candidate-count discipline**: this design proposes 6 Round-1 entrants + up to 1-2 Round-2
   entrants (meta-labeling variants) = 7-8 total against the new exam window, leaving headroom
   under the ~10 budget for a determinism-guard re-run of the eventual winner.
7. **Every entrant gets a full experiment-preregister file** (hypothesis, H0, metric, numeric
   threshold, risks of false positive, protocol) before it runs, per the skill — this document is
   the survey that informs those files, not a substitute for them.

---

## (d) What NOT to do — known overfit traps in this space

1. **Do not hyperopt/tune barrier widths, horizons, or confidence thresholds on the exam window
   itself.** Freqtrade's own documentation and community warn that brute-force parameter search
   finds "peaks" that fit in-sample trades perfectly and never recur
   ([Freqtrade hyperopt docs](https://www.freqtrade.io/en/stable/hyperopt/),
   [GitHub #2472](https://github.com/freqtrade/freqtrade/issues/2472)) — this is the same failure
   mode #912's Phase 3 protocol was built to avoid (thresholds chosen from training-period data,
   held fixed, run once on the exam).
2. **Do not trust a public strategy leaderboard's headline return as evidence for anything.**
   freqst.com's own top strategies are 100% non-ML technical-indicator strategies with zero
   disclosed methodology and an explicit "no longer maintained" notice — this is the textbook
   overfit-leaderboard graveyard the task brief anticipated, not a source of validated targets or
   parameters.
3. **Do not treat classification/regression accuracy as a stand-in for profitability.** The most
   rigorous empirical source found (the Korean triple-barrier LSTM paper) reports classification
   metrics only and explicitly never establishes a link to trading P&L — carrying that paper's
   "43% > 35% dummy" finding into a claim like "triple-barrier is proven to make money" would be
   citing evidence for something it doesn't show.
4. **Do not build a meta-labeling model whose dominant feature is the primary model's own
   |predicted_return|.** #912 already falsified exactly this (as a degenerate one-feature
   meta-model) — repeating it with a fancier classifier on top of the same single feature is not a
   new experiment, it's the same one with extra steps.
5. **Do not reuse the 2026-01-01→2026-07-04 exam window a third time.** Doing so would silently
   blow through the north star's own multiple-comparison budget and make any "winner" more likely
   to be a lucky draw against an increasingly well-explored window.
6. **Do not skip the determinism guard.** #913's fix is recent (closed 2026-07-06); a tournament
   that trusts single-run numbers without a repeat-run check is exposed to exactly the kind of
   silent corruption that already produced two different "identical" baseline results in #912
   before the bug was found.
7. **Do not promote any tournament winner to staging/live directly from a backtest.** Per
   `model_evaluation_system.md` principle 4 ("a backtest buys a staging ticket, never a live
   deployment") and this desk's standing rule (never auto-promote) — L3a paper validation (≥48h)
   is mandatory regardless of how good the L2 exam numbers look, and any live-affecting change
   still requires the "How this could lose money" adversarial section and risk-officer stress-test
   call-outs this desk's own workflow already mandates.
8. **Do not let a single vendor/blog case study (Hudson & Thames' meta-labeling numbers, the
   freqAI-LSTM repo's implied edge) stand in for independent replication.** Both are cited above
   because they're the best evidence found, not because they're strong evidence — Hudson & Thames
   sells the library the technique is built on, and the freqAI-LSTM repo has no disclosed live
   track record. Treat both as "plausible direction, not proof."

---

## Summary for `pm`

**Recommendation: promising direction, ready to pre-register — not ready to run without new
exam-window setup.** The literature and ecosystem survey converges with our own two closed
experiments (#898, #912) on the same conclusion the north star's open question already
anticipated: next-bar price regression is very likely the wrong target, and the two most credible
replacement directions are (1) a target that separates "is there a trade" from "is this trade
worth taking/how big" (meta-labeling, directly fits the diagnosed 78%-win/PF-0.69 exit-geometry
signature) and (2) a target that produces a genuine, calibrated probability by construction
(direction classification / triple-barrier), fixing the confidence-channel's root cause (an
uncalibrated constant) rather than patching around it again.

No backtest was run for this document — it is research-survey input to the next tournament's
pre-registration, not itself an experiment result. Next step: write the Round-1 experiment-
preregister files (one per candidate, or one umbrella file with per-candidate sub-hypotheses,
matching the `2026-07-05_window-tournament.md` shape) against a **new** frozen exam window, then
run L1+L2 per `model_evaluation_system.md`.

## Sources

- [Freqtrade FreqAI configuration docs](https://docs.freqtrade.io/en/stable/freqai-configuration/)
- [FreqaiExampleStrategy.py](https://github.com/freqtrade/freqtrade/blob/develop/freqtrade/templates/FreqaiExampleStrategy.py)
- [Freqtrade FreqAI feature engineering docs](https://docs.freqtrade.io/en/latest/freqai-feature-engineering/)
- [Freqtrade hyperopt docs](https://www.freqtrade.io/en/stable/hyperopt/)
- [Freqtrade GitHub issue #2472 — hyperopt/overfitting discussion](https://github.com/freqtrade/freqtrade/issues/2472)
- [freqst.com](https://freqst.com/)
- [Netanelshoshan/freqAI-LSTM README](https://github.com/Netanelshoshan/freqAI-LSTM/blob/main/README.md)
- [Netanelshoshan freqAI-LSTM author writeup](https://netanel.io/posts/freqai_lstm_reg/)
- [Reasonable Deviations — Advances in Financial ML chapter notes](https://reasonabledeviations.com/notes/adv_fin_ml/)
- [mlfinpy Labelling docs](https://mlfinpy.readthedocs.io/en/latest/Labelling.html)
- [O'Reilly — Advances in Financial ML Ch. 3 excerpt](https://www.oreilly.com/library/view/advances-in-financial/9781119482086/c03.xhtml)
- [Wikipedia — Meta-Labeling](https://en.wikipedia.org/wiki/Meta-Labeling)
- [Hudson & Thames — Does Meta-Labeling Add to Signal Efficacy?](https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/)
- [Hudson & Thames / LinkedIn — Trend Scanning with MLFinLab](https://www.linkedin.com/posts/hudson-thames-quantiative-research_trend-scanning-with-mlfinlab-developed-activity-7110593515889307648-TaKH)
- [mlfinlab GitHub issue #433 — trend-scanning t-value bug](https://github.com/hudson-and-thames/mlfinlab/issues/433)
- [mlfinlab GitHub issue #551 — trend-scanning live-market bug](https://github.com/hudson-and-thames/mlfinlab/issues/551)
- [arXiv:2504.02249 — Triple-Barrier Labeling, Korean market LSTM study](https://arxiv.org/html/2504.02249v2)
- [arXiv:2508.20108 — ReVol: Return-Volatility Normalization](https://arxiv.org/abs/2508.20108)
- [arXiv:2505.05595 — FutureQuant Transformer (distributional futures targets)](https://arxiv.org/html/2505.05595)
- [MDPI 2076-3417/15/20/11145 — Confidence-threshold framework for crypto direction prediction](https://www.mdpi.com/2076-3417/15/20/11145)
- Internal: `docs/architecture/model_evaluation_system.md`, `docs/research/experiments/2026-07-05_confidence-calibration.md`,
  `docs/research/experiments/2026-07-05_window-tournament.md`, `docs/research/experiments/2026-07-04_hypergrowth-exit-geometry.md`,
  GitHub issues #898, #912, #913, #867, `.claude/skills/experiment-preregister/SKILL.md`
