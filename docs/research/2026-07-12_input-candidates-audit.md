# Input Candidates Audit — INPUT Tournament Lane A, Phase 0

Date: 2026-07-12
Author: quant-researcher
Status: **Research complete — feeds the LINEAR SCREENING experiment preregistration (next phase)**
Related: window tournament (#898), architecture tournament (#939), TARGET-REDESIGN tournament
(`docs/research/experiments/2026-07-10_target-redesign-tournament-results.md`, merged to develop
at `25e0a202`)

## Why this exists

Three independent tournaments each held the feature set fixed (price-only 1h OHLCV) and varied a
different lever — training window, model architecture, target/label design. All three converged
on the same finding: every entrant collapses to predicting the unconditional/majority-class
distribution. The target-redesign report's explicit conclusion (quoted verbatim): *"the feature
set itself — not the model, not the window, not the target shape — is the ceiling... The next
research lever this implies is new information sources."* This document is Phase 0 of that next
lever: which alternative inputs have credible evidence AND are actually obtainable as historical
data for our three eval folds (2023H1 / 2024H1 / 2025H1), with training history reaching back to
~2019-2020.

**This is a research/audit document. No `src/` changes. No backtest was run.** The scripts under
`scripts/research/` in this branch are proof-of-obtainability probes only — throwaway, not wired
into any `DataProvider`, and must not be mistaken for backtest-ready code (per CODE.md's
Backtest-Live Parity rule, nothing here counts until it goes through `src/engines/shared/`).

## Headline finding before the candidate-by-candidate breakdown

**This repo already has three "alternative data" feature extractors wired into the pipeline —
`OnChainFeatureExtractor`, `MacroFeatureExtractor`, and two-thirds of `EnhancedSentimentExtractor`
(`src/prediction/features/onchain.py`, `macro.py`, `enhanced_sentiment.py`) — and every single one
of them is a documented no-op.** Reading past the class docstrings and into `_compute_*`, every
feature they produce is a deterministic transform of the same `close`/`volume` columns already in
the OHLCV frame (e.g. `exchange_netflow` = `volume_change * -returns`; `spx_trend` = a moving-average
crossover of `close`; `social_volume_zscore` = a z-scored `log1p(volume)`). Each docstring says so
explicitly: *"Uses simulated data... Replace with real \[X\] data from \[vendor\]."* They are
disabled by default (`DEFAULT_ENABLE_SENTIMENT/ONCHAIN/MACRO = False` in `src/config/constants.py`)
for good reason: **flipping `enabled=True` on any of them adds zero new information — it's a second
pass over data the model already has, dressed up as "on-chain" or "macro."** If the linear screening
experiment or any future tournament ever includes these three extractors, this must be caught before
publishing results — a "new feature" that is provably a linear function of features already in the
model, tested for incremental power, will silently reproduce the price-only ceiling and could be
misread as "alternative data didn't help" when what actually happened is alternative data was never
tested. Only the `sentiment.py` extractor's non-fallback path (real `FearGreedProvider`, gated behind
`DEFAULT_ENABLE_SENTIMENT`) and the F&G component inside `enhanced_sentiment.py` are real.

This finding alone reframes the audit: for on-chain and macro, the work is not "enable the existing
extractor," it's "replace the entire body of the existing extractor with a real data source" — full
integration cost, not near-zero.

---

## Candidate-by-candidate breakdown

### 1. Derivatives state (funding rate, open interest, long/short ratio, basis)

**Evidence.** Mixed, split cleanly by sub-signal:
- *Funding rate as a crowding/positioning signal*: practitioner and academic consensus is that
  persistently elevated funding indicates crowded long positioning and precedes squeezes/reversals;
  Binance funding Granger-causes Bybit/OKX funding in a majority of rolling windows in at least one
  cross-exchange study (arXiv:2506.08573, MDPI 14(2):346). This is evidence for funding as a
  *regime/crowding* feature, not a demonstrated directional-return predictor at 1h-4h horizon —
  no source found makes that specific claim with an out-of-sample Sharpe number.
- *Funding-rate arbitrage* (cash-and-carry, delta-neutral) has real documented returns (up to
  115.9%/6mo in one study, ScienceDirect S2096720925000818) — but that is a market-neutral carry
  strategy, structurally unrelated to directional ETH price prediction; cite it as "funding rate is
  a real economic signal," not as "funding rate predicts direction."
- *Open interest / long-short ratio*: theoretically the cleanest crowding measure (aggregate
  leveraged positioning), widely used by practitioners for liquidation-cascade risk, but I found no
  rigorous academic OOS evidence for short-horizon directional edge — treat as plausible-but-unproven.
- *Basis (perp-spot premium)*: mechanically the same signal as funding (funding is calculated FROM
  the premium), so it's correlated, not independent, evidence.

**Obtainability — verified empirically this session** (`scripts/research/pull_binance_funding_rate.py`,
`scripts/research/check_binance_derivatives_retention.py`, both run against the live Binance API,
2026-07-12):

| Signal | Endpoint | Auth | Coverage confirmed | Granularity |
|---|---|---|---|---|
| Funding rate | `fapi.binance.com/fapi/v1/fundingRate` | none | Back to at least 2000 days (~5.5y; first ETHUSDT perp funding settlements are Nov 2019) — pulled a live sample, 90 records/30d, all exactly 8.00h apart | Every 8h settlement (00:00/08:00/16:00 UTC) |
| Basis proxy (premium index) | `fapi.binance.com/fapi/v1/premiumIndexKlines` | none | Same deep history as funding rate — verified 2000 days back, standard kline pagination, no retention wall | 1h+ (kline-style, any standard interval) |
| Open interest history | `fapi.binance.com/futures/data/openInterestHist` | none | **Hard 30-day retention wall.** Empirically confirmed: `startTime` 25 days back → 200 OK; 30+ days back → `400 {"code":-1130,"msg":"parameter 'startTime' is invalid."}` on every trial up to 2000 days back | 5m-1d, but only for the last ~30 days |
| Long/short ratio | `fapi.binance.com/futures/data/globalLongShortAccountRatio` | none | **Same 30-day wall**, confirmed identically | Same |

**This is the single most important obtainability finding in this audit**: OI and long/short ratio
are *not usable at all* for the 2023H1/2024H1/2025H1 historical folds via the free API — only funding
rate and the basis/premium-index proxy have the historical depth this tournament needs. A paid vendor
(Coinalyze, CryptoQuant, Glassnode) would be required for historical OI/L-S, which is out of scope
without a separate cost decision.

**Leakage/alignment traps.**
- Funding settles at fixed 8h boundaries (00:00/08:00/16:00 UTC); on a 1h/4h bar grid, forward-fill
  the last *settled* funding rate — never use the *next* settlement's value on bars before it prints.
  The `fundingTime` in the API response is the settlement timestamp; a bar at 07:00 UTC must see the
  00:00 print, not the 08:00 one.
- `markPrice` is empty (`""`) in the oldest funding-rate records (pre-2021ish) — don't silently coerce
  to 0.0, that will corrupt any basis feature computed from this field on that period.
- Basis via `premiumIndexKlines` uses OHLC-of-premium — using the bar's own `close` premium at time
  `t` to decide at `t` is fine (it closes with the bar), but using `high`/`low` of that same bar is a
  same-bar lookahead (the extremes aren't known until the bar closes).

**Integration cost.** New `DataProvider` needed (no derivatives provider exists in
`src/data_providers/` today — confirmed via grep, only spot OHLCV/sentiment providers exist). Small:
a single REST GET, standard pagination, no auth, JSON→DataFrame. Comparable in shape to the existing
`FearGreedProvider`. Estimate: 0.5-1 day including cache-manager integration and a real vs. fake
distinction so this never collapses into another `onchain.py`-style stub.

---

### 2. Cross-asset lead-lag (BTC→ETH, DXY/SPX/NDX, BTC dominance)

**Evidence.**
- BTC→ETH: genuinely mixed in the literature. Sifat & Mohamad (2019, hourly/daily Aug2017-Sep2018)
  found no consistent lead-lag direction — largely bidirectional Granger causality. More recent work
  on cross-cryptocurrency predictability finds the five largest coins' lagged returns do predict
  smaller coins' next-period returns (a "seesaw"/size effect, ScienceDirect S0927539823000956), and
  that predictability shrinks as liquidity increases — which argues *against* a strong BTC-lead-ETH
  effect specifically, since ETH is itself one of the top-2 most liquid coins. Net read: plausible,
  weak, almost certainly time-varying (stronger during high-vol/stress regimes, weaker in calm
  regimes) rather than a stable standing effect.
- DXY/SPX/NDX: crypto-macro co-movement ("risk-on/risk-off") is well documented qualitatively but
  operates on daily/weekly macro-regime timescales, not 1h-4h; using it as a 1h feature mostly buys a
  slow-moving regime flag, not a short-horizon directional edge.
- BTC dominance: no direct evidence found bearing on short-horizon ETH prediction specifically;
  reasoned as strictly downstream of BTC and total-market-cap moves already substantially captured
  by BTC returns/vol.

**Obtainability.**
- BTC returns/volatility as ETH features: **zero incremental cost — already in our cache.** Confirmed
  via `atb data cache-manager info` against the main checkout's cache (`cache/market_data`, 34 files,
  6.5MB) — BTCUSDT OHLCV at the same timeframes as ETHUSDT is already fetched/cached by this repo's
  existing `BinanceProvider`/`CachedDataProvider` machinery. This is a pure feature-engineering
  exercise (lag BTC's own return/realized-vol columns and join on timestamp), not a new data-source
  integration.
- DXY/SPX/NDX: obtainable free via Yahoo Finance's unofficial chart API (`query1.finance.yahoo.com`),
  verified working this session with no auth (`^GSPC`... equities have decades of daily history; the
  `DX-Y.NYB` dollar-index ticker's depth via `range=max` returned an oddly short 168-point series in a
  quick check — needs `period1`/`period2` explicit params to get the true depth, or substitute FRED's
  trade-weighted dollar index for a cleaner free daily series). This is an unofficial/undocumented API
  with no SLA — acceptable for research, not for a live-trading data path per CODE.md's External API
  Calls guidance without a documented fallback.
- BTC dominance: needs total-market-cap history; CoinGecko's historical global chart is paywalled
  (Pro tier) as far as I could establish; not pursued further given weak evidence and redundancy with
  #2's BTC-returns feature.

**Leakage/alignment traps.**
- BTC and ETH bars must be joined on the *same* closed-candle boundary as the rest of the pipeline
  (no using a BTC bar that closes later than the ETH bar it's paired with).
- Equity/DXY data is daily and only prints on trading days (no weekends/holidays) against 24/7 crypto
  bars — forward-fill from the last *confirmed* daily close, and lag by enough that "today's SPX
  close" is never available before the US market actually closes (~21:00 UTC) — a naive same-calendar-
  day join risks using a close that hasn't happened yet in bars earlier that day.

**Integration cost.** BTC-derived features: near-zero (pure feature engineering on already-cached
data, no new provider). DXY/SPX/NDX: small-to-moderate (new provider + non-crypto calendar alignment
logic, which is a real, non-trivial parity concern the shared engine doesn't currently have to solve).

---

### 3. Sentiment (Fear & Greed, social metrics)

**Evidence.**
- Fear & Greed: strong evidence for *extreme-value contrarian* trades on longer holding periods
  (buy <20-25, sell >75-80 has reportedly beaten DCA/buy-and-hold by wide margins in several
  backtests — codemeetscapital.substack.com, bitcoinmagazine.com, ainvest.com), but every source
  agrees this is a low-frequency, extreme-tail signal (readings below 10-15 can be absent for years)
  with a multi-week/month holding horizon — not obviously a continuous 1h-4h directional feature.
  Using it as a dense per-bar ML input is a genuinely different claim than the backtested strategies
  in the literature, which are event-driven at the extremes.
- Social metrics (Reddit/Twitter volume, engagement): no rigorous evidence found in this pass;
  general practitioner consensus treats raw social volume as noisy and easily gamed (bot volume,
  influencer pumps) without NLP-quality filtering.

**Obtainability.**
- Fear & Greed: **already fully solved in this repo.** `src/data_providers/feargreed_provider.py`
  (`FearGreedProvider`) hits `api.alternative.me/fng/?limit=0` — verified this session
  (`scripts/research/pull_feargreed_index.py`): 3,080 daily records, 2018-02-01 to 2026-07-12
  (today), no gaps beyond normal 1-day cadence, free, no key. `SentimentFeatureExtractor` already
  wraps this provider end-to-end (resampling, ffill, freshness gating) — it's just switched off
  (`DEFAULT_ENABLE_SENTIMENT = False`).
- Social metrics: no free, deep-history API found. LunarCrush and Santiment both gate historical
  social-volume series behind paid tiers as of this research pass; the `enhanced_sentiment.py`
  extractor's `social_volume_zscore` and `news_sentiment_score` are (per the headline finding above)
  simulated volume/return proxies, not real social data — treat as unobtainable for free at the
  history depth this tournament needs; defer.

**Leakage/alignment traps.**
- Daily F&G vs 1h/4h bars: forward-fill from the *previous* day's confirmed index value; the API's
  `time_until_update` field implies the index for "today" is a running/settling value — using
  "today's F&G" on a bar from earlier the same day is a same-day lookahead risk. Lag by a full day
  to be safe, or confirm from alternative.me's methodology exactly when a given day's value is
  finalized before trusting same-day use.
- **Redundancy/circularity risk, not just leakage**: the F&G index's own methodology blends price
  momentum, volatility, and volume — i.e. it's *partially derived from the same OHLCV series already
  in the model*. It is not guaranteed to be "new information" the way funding rate or BTC-cross
  features are; the linear screening experiment should test its *marginal* contribution over a
  price/vol-only baseline specifically, not just its raw correlation with returns.

**Integration cost.** Effectively zero — flip `DEFAULT_ENABLE_SENTIMENT` and wire the extractor's
output into the training feature schema. This is the cheapest candidate in this entire audit to
actually test, which is exactly why (per the ranking weighting) it makes the shortlist despite
middling evidence for the specific 1h-4h directional task.

---

### 4. On-chain (exchange flows, active addresses)

**Evidence.** Plausible mechanistically (exchange inflows preceding sell pressure is a commonly
cited on-chain thesis) but this pass did not turn up rigorous short-horizon OOS evidence specific to
ETH at 1h-4h; on-chain research is more commonly framed at daily/weekly macro-cycle horizons
(accumulation/distribution phases), similar caveat to Fear & Greed's extreme-value framing.

**Obtainability.** Poor, for the depth this tournament needs. Glassnode's free tier is heavily
capped (recent data only, most metrics paywalled at Advanced/Institutional tiers); CryptoQuant
similarly gates historical exchange-flow series. Blockchain.com's free charts API has deep BTC-only
history (hash rate, active addresses, tx volume) but nothing for ETH. Etherscan's stats endpoints
exist for basic chain metrics (gas price, daily tx count) but historical *chart-data* access has
historically required their paid tier. **No free path found to real ETH on-chain history covering
2019-2025.** Per the headline finding, `OnChainFeatureExtractor` in this repo is 100% simulated and
contributes nothing today.

**Leakage/alignment traps.** On-chain data is typically block-time or daily-snapshot; joining to
hourly bars needs the same forward-fill-from-last-confirmed discipline as sentiment, plus care that
"active addresses today" isn't finalized until the day closes.

**Integration cost.** High — no viable free source at the needed depth means this either needs a
paid subscription (separate proposal/cost decision, outside this research task's scope) or stays
deprioritized. **Recommendation: do not include in the linear screening experiment; revisit only if
a paid on-chain vendor is separately approved.**

---

### 5. Market microstructure from OUR OWN OHLCV data (realized vol, volume profile, HL range, seasonality)

**Evidence.** This is the strongest evidence-to-cost ratio in the whole audit, but the evidence is
predominantly for *volatility forecasting*, not directly for *directional* return prediction — an
important distinction given our target is next-bar direction. HAR-RV and its crypto-specific
extensions (SA-Log-HAR-RS, Lasso-SA-Log variants) are a mature, well-replicated literature showing
multi-scale realized volatility (fine/medium/coarse windows) has strong, high-frequency-persistent
predictive power *for future volatility*, with short (1-step) horizons dominated by recent
high-frequency persistence (arXiv:2507.22409, and multiple corroborating 2025-2026 papers). The
read-through to our problem: these features are excellent *regime/conditioning* inputs (separating
"model is confident and right" from "model is guessing," which is exactly the axis the
target-redesign tournament's own mechanism finding flagged — triple_barrier's one real edge, in
fold F2, showed up specifically as a non-degenerate confidence distribution tied to a vol-regime-like
split) rather than a demonstrated source of *direction* signal on their own. That is a meaningfully
different, more modest claim than "this predicts up/down" — and should be framed that way in the
screening experiment's hypothesis, not oversold.

Do not underweight this class per the brief's own instruction — it is the only candidate that is
simultaneously well-evidenced (for its actual claim), zero-cost, and available for the entire
backtest history with no vendor risk at all.

**Obtainability.** Trivial — 100% derivable from the OHLCV already cached (confirmed: 34 cached
files, 6.5MB, covering our symbols/timeframes). No network call, no new provider, no rate limit, no
retention wall. Candidate features: realized vol at multiple lookback scales (e.g. 6h/24h/7d rolling
std of log returns sampled at the bar's own timeframe), Parkinson/Garman-Klass range-based vol
estimators from high/low, rolling high-low range as % of close, volume z-score vs its own rolling
mean (a *real* version of what `enhanced_sentiment.py` fakes as "social volume"), and intraday/
day-of-week seasonality dummies.

**Leakage/alignment traps.** The obvious one, worth stating plainly because it's exactly CODE.md's
top rule: every rolling window must use data strictly *before* bar `t`'s close to decide at `t` —
e.g. a `high`/`low`-based range feature for bar `t` must use `t-1`'s completed range, never `t`'s own
still-forming high/low. This is the same closed-candle discipline the existing technical-indicator
pipeline already enforces elsewhere in the codebase; extending it to new rolling-window features must
reuse the same shared helper, not reimplement the shift/lag logic ad hoc per CODE.md's
Backtest-Live-Parity rule (features must live in `src/engines/shared/` or reuse what's there).

**Integration cost.** Low. `src/tech/features/technical.py` already computes some rolling
statistics; multi-scale realized vol and range estimators are incremental additions to an extractor
that already exists and is already enabled, not a new subsystem. Fastest path to a testable feature
in this entire audit.

---

### 6. Time/calendar features (session, day-of-week, funding-settlement windows)

**Evidence.** Weaker/thinner literature specific to crypto (day-of-week and weekend effects are
debated and have reportedly weakened as crypto markets matured/institutionalized), but the cost of
testing is essentially zero, so a null result costs almost nothing to obtain. Funding-settlement-
window dummies (is this bar within N hours of a funding print) are a natural complement *if* the
funding-rate candidate is adopted — worth testing jointly, not standalone.

**Obtainability.** Trivial — derived purely from the bar's own timestamp (hour-of-day, day-of-week,
distance-to-next-funding-settlement). No external data at all.

**Leakage/alignment traps.** None beyond the standard timezone-consistency rule already in CODE.md
(all timestamps UTC, no local-time ambiguity) — this is about the only candidate in the audit with
no lookahead surface, since it's a deterministic function of the bar's own already-known timestamp.

**Integration cost.** Trivial — a handful of `pd.Series` derived from the DataFrame's own index.

---

## Ranked shortlist for the LINEAR SCREENING experiment

Ranked by evidence × obtainability × integration cost, per the brief's explicit weighting (a
mediocre-evidence input testable today beats a great-evidence input needing a paid API):

| Rank | Candidate | Data source + coverage | Feature sketch | Alignment rule | Evidence one-liner |
|---|---|---|---|---|---|
| 1 | **Multi-scale realized volatility + range dynamics (own OHLCV)** | Already-cached OHLCV, full history, zero cost | Rolling-window realized vol at 2-3 scales (e.g. 6h/24h/7d), Parkinson/GK range estimator, HL-range % of close | Strict `t-1`-and-earlier windows only; reuse existing closed-candle shift helper | Mature, replicated HAR-RV literature — strong for vol/regime, not direction; frame hypothesis accordingly |
| 2 | **Time/calendar features** | Bar's own timestamp, zero cost | Hour-of-day, day-of-week one-hot or cyclical encode, hours-to-next-funding-settlement | None needed (fully known at bar close) | Thin/debated crypto-specific evidence, but free to test and pairs naturally with #1 and #4 |
| 3 | **BTC→ETH cross-asset features** | Already-cached BTCUSDT OHLCV, same timeframes | Lagged BTC return/realized-vol at matching lookbacks, joined on closed-candle timestamp | Join on same bar-close boundary as ETH; no using a later-closing BTC bar | Mixed/weak lead-lag evidence, weakens with liquidity, but zero incremental data cost — cheap to falsify |
| 4 | **Funding rate (ETHUSDT perp)** | `fapi.binance.com/fapi/v1/fundingRate`, free, no key, confirmed back to ~Nov 2019 (2000+ days), 8h settlement | Level, rate-of-change, and rolling z-score of funding; extreme-funding binary flags | Forward-fill last *settled* print only; never the next scheduled settlement; watch empty `markPrice` in pre-2021 records | Real crowding/positioning signal (cross-exchange Granger evidence), no direct short-horizon directional OOS result found — moderate evidence, good obtainability, small new-provider cost |
| 5 | **Basis / perp-spot premium proxy** | `fapi.binance.com/fapi/v1/premiumIndexKlines`, free, no key, same deep history as funding, 1h+ granularity | Premium level and its own short-window realized vol | Use only the *closed* bar's `close`-of-premium; don't use same-bar high/low | Mechanically tied to funding rate (not independent evidence) but higher-frequency; bundle with #4 rather than test standalone |
| 6 | **Fear & Greed index** | `api.alternative.me/fng/`, free, no key, confirmed 3,080 daily records 2018-02-01 to today; already wired via `FearGreedProvider`/`SentimentFeatureExtractor` (currently disabled) | `sentiment_primary` level + momentum + extreme-value flags, forward-filled to bar timeframe | Lag by a full day past the print; explicitly test *marginal* contribution over price/vol baseline given circularity risk (F&G is partly price-derived itself) | Strong evidence only at extreme-tail, multi-week-holding-period framing; weak/unproven as a dense continuous 1h feature; but integration cost is ~zero (already built) |

**Explicitly deferred, not recommended for this round**: open interest and long/short ratio (good
theoretical evidence, but empirically confirmed unobtainable for 2023-2025 history via any free API —
would need a paid vendor decision first); on-chain exchange-flow/active-address metrics (real data is
paywalled at the depth needed; the in-repo extractor is simulated and must not be mistaken for real);
DXY/SPX/NDX macro (weak short-horizon evidence, non-trivial calendar-alignment cost, and the in-repo
extractor is likewise simulated); BTC dominance and social-media volume (no free historical source
found at the needed depth).

## What the next phase (linear screening prereg) should do differently

1. Test candidates 1-2-3 (own-data microstructure, calendar, BTC cross-asset) first — they cost
   nothing to obtain and can be screened this week without any new `DataProvider` work.
2. Build the funding-rate provider (candidate 4) as the first *new external data* addition, since
   it's the only derivatives signal that actually clears the historical-coverage bar; fold basis
   (candidate 5) in alongside it rather than as a separate screening slot, given the mechanical
   correlation between the two.
2b. Frame candidate 1's hypothesis honestly as a regime/confidence-conditioning feature, not a
    direct directional-edge feature — that framing is falsifiable and matches what the literature
    actually supports, avoiding a repeat of the target-redesign tournament's "reshaped the target,
    still price-only" trap in a new "reshaped the feature set, but every 'feature' is still a
    function of price" form.
3. Before running any tournament that uses `OnChainFeatureExtractor`, `MacroFeatureExtractor`, or
   the social-volume/news-sentiment components of `EnhancedSentimentExtractor`, confirm they are
   NOT included (or confirm they've been replaced with a real data source) — otherwise a "no
   improvement from alternative data" result would be invalid, since those extractors don't add any
   data.
4. Fear & Greed (candidate 6) is worth including for its near-zero cost, but the prereg should set a
   specific test for *marginal* contribution beyond a price/vol-only baseline (not just standalone
   correlation), given its partial circularity with price data.

## Proof-of-obtainability scripts (this branch, not production code)

- `scripts/research/pull_binance_funding_rate.py` — pulls ETHUSDT funding-rate history from the
  public Binance futures API; run this session, confirmed 90 records/30 days, exactly 8h apart.
- `scripts/research/pull_feargreed_index.py` — independently re-verifies `FearGreedProvider`'s
  source; run this session, confirmed 3,080 daily records, 2018-02-01 to 2026-07-12.
- `scripts/research/check_binance_derivatives_retention.py` — the script that produced the OI/
  long-short 30-day-wall finding; run this session across 10/25/30/45/90/365/2000-day probes,
  confirms the cutoff falls between 25 and 30 days for both endpoints, with funding-rate and
  premium-index-klines as clean deep-history controls.

None of these touch `src/` and none are wired into the real `DataProvider`/cache-manager path —
that integration work belongs to whichever candidates the linear screening prereg actually adopts.
