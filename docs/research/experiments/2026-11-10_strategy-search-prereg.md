# Wide strategy search (Phase 1b): preregistration

**Date**: 2026-11-10
**Status**: PREREGISTERED. Families, parameters, windows and gates below are locked before any strategy result is computed. Corrections go in new dated sections, never edits.
**Author**: quant research session, at the Board's request (Alex, 2026-11-10: the bake-off outcome "is not good enough"; search wider for a mix of strategies and symbols that earns in all market conditions).
**Builds on**: `2026-11-10_strategy-bakeoff-results.md` (#1315), `2026-08-13_model-free-predictability-scan.md` (#1105), `../2026-07-12_returns-levers-synthesis.md`.
**Scripts**: `experiments/strategy_search/` (`data.py`, `engine.py`, `strategies.py`, `run_search.py`).

## 0. Why this experiment, and what it is not

Everything tested so far falls in three boxes: predicting the next bar of one coin (no edge after costs, six experiments), long-only spot rules on twelve large caps (trend and momentum fail the 2025-26 holdout because the market fell and they cannot short), and BTC/ETH funding carry (passes, earns 0.9%/yr).

Untested: **long/short and market-neutral portfolios across the whole perpetual-futures universe.** These can earn when the market falls, and they harvest premia that exist between coins rather than in the direction of any one coin. That is what this experiment covers. It does not revisit single-coin direction prediction, long-only spot, or ML.

## 1. Hypotheses

**H1**: at least one cross-sectional or long/short family below earns a net premium on Binance USDⓈ-M perpetuals that survives realistic costs, funding and a one-day execution delay, in a validation period and then in the untouched-by-these-strategies 2025-26 holdout.

Mechanism per family:

| ID | Family | Why it might pay |
|---|---|---|
| T | Time-series trend, long/short | Crypto trends persist (slow information diffusion, reflexive leverage); shorts earn in bear markets, which long-only S2 could not |
| XM | Cross-sectional momentum, market-neutral | Relative winners keep winning for 1-8 weeks (attention, slow rotation); documented in Liu-Tsyvinski-Wu (2022) |
| XR | Cross-sectional reversal, market-neutral | Liquidity provision after overreaction; liquidation cascades overshoot |
| XC | Funding carry across perps, market-neutral | High funding = crowded leveraged longs; short them, long the cheap-to-hold coins, collect the spread |
| CC | Cash-and-carry rotation (spot long + perp short on the highest-funding coins) | S1's premium, harvested where funding is highest instead of only BTC/ETH |
| AB | Long BTC / short alt basket | Token inflation, unlocks and lottery demand make alts bleed against BTC |
| LV | Low-volatility / lottery factor, market-neutral | Retail overpays for lottery-like coins (high vol, recent extreme up-days) |
| NL | Short recent listings, hedged with BTC | Listing hype plus unlock supply; new perps drift down |

**H0**: none does. Either the gross premium is not there, or it is smaller than costs plus funding, or it existed in 2020-22 and was arbitraged away.

- *H1 supported if*: at least one candidate clears the holdout gate in section 5.
- *H0 supported if*: no family clears the validation gate in section 4 (then nothing touches the holdout), or every shortlisted candidate fails the holdout gate.

## 2. Data

- Source: the public Binance archive (data.binance.vision). Daily klines, funding history, and daily spot klines for the matching spot pair.
- **Universe: every USDT-margined perpetual that ever listed, including delisted ones** (LUNA, FTT and friends are in). This removes the survivorship bias the bake-off had to caveat.
- Excluded by name (not crypto-asset perps): stablecoin pairs, index products (BTCDOM, DEFI, FOOTBALL, BLUEBIRD, ALL), tokenised metals, BTCST. The list is in `engine.EXCLUDE`. If the download shows other non-crypto perps (tokenised equities, FX), they are added to the list before the first run and noted in the results file.
- Range: 2019-09-01 to 2026-10-31.
- **Point-in-time tradable set**, recomputed on every signal day from data known at that close: a perp is eligible once it has 60 daily closes; the universe is the **top 30 eligible perps by trailing 30-day median quote volume**. NL is the only family that trades outside it (by design).

## 3. Shared rules

- **Timing**: signals use day t's close (UTC); fills happen at day t+1's close. This one-day delay is deliberately harsh on fast strategies: anything that only works when filled at the same price it was signalled on is not accepted. A same-close fill is reported as a secondary sensitivity, never used for a gate.
- **Costs per side on traded notional**: perp fee 5 bps + slippage 5 bps (liquidity rank 1-10), 10 bps (rank 11-30), 20 bps (rank 31+). Spot fee 10 bps + the same slippage tier.
- **Funding**: real historical funding is paid or received on every perp position each day. If a print is missing while a position is held, longs are charged 3 bps/day and shorts receive nothing.
- **Delisting**: a held contract that stops trading is closed at its last close with normal costs.
- **Sizing**: every family except CC is scaled to **15% annualised volatility** using its own trailing 60-day realised vol (known at the signal close), gross exposure capped at 2.0x equity. CC is reported at its natural size (hedged notional = 50% of capital).
- **Hurdle**: idle USDT at **4%/yr**. Sharpe ratios are computed on returns in excess of it. Strategy cash earns nothing (conservative, since perp margin cannot sit in a savings product).
- **Quintile sorts**: long and short legs each hold max(3, round(N/5)) coins, equal weight inside a leg, legs sized 0.5 / 0.5 of gross (dollar-neutral) unless stated.
- vol30 = 30-day standard deviation of daily returns. "Weekly" = signal on Monday's close.

## 4. Families, variants and the discovery protocol

21 variants in 8 families. No other variant will be run against the gates.

| ID | Rule | Variants |
|---|---|---|
| T | For each universe coin, s = mean over L in the set of sign(close_t / close_{t-L} - 1); weight ∝ s / vol30, normalised to gross 1. Weekly. Net exposure floats. | L set {7, 14, 28}; {30, 60, 120} |
| XM | Rank by return from t-L to t-1 (skips the last day). Long top quintile, short bottom. Weekly. | L = 7, 14, 28, 56 |
| XR | Rank by return over the last L days. Long the losers' quintile, short the winners'. | L = 1 (daily), 3 (daily), 7 (weekly) |
| XC | Rank by mean daily funding over the last L days. Long lowest quintile, short highest. Weekly. | L = 3, 7, 14 |
| CC | Among universe coins with a spot pair: K = 5 slots, each 1/K of capital, half in spot and half as perp margin. Enter a coin when it is in the top 5 by trailing L-day funding and that funding is above 15%/yr; exit when it leaves the top 10 or its trailing funding drops below 0. Checked weekly. Costs: spot + perp cost on entry and exit, plus a daily hedge-rebalancing charge of \|spot return\| x notional x spot cost (as in the bake-off). | L = 7, 14 |
| AB | Long BTC, short an equal-weight basket of the other universe coins. Weekly. | dollar-neutral (0.5 / 0.5); beta-neutral (short leg scaled by 1/β, β = 90-day beta of the basket to BTC, renormalised to gross 1) |
| LV | Weekly quintile sorts. | (a) long lowest vol30, short highest, dollar-neutral; (b) same, each leg scaled to equal ex-ante vol; (c) "MAX": short the quintile with the highest single-day return in the last 30 days, long the lowest |
| NL | Short, equal weight, every non-excluded perp between D1 and D2 days after its first trading day (any liquidity rank, 20 bps slippage tier applies); long BTC for the same notional. Weekly. | (D1, D2) = (7, 60); (7, 120) |

**Windows** (hard cutoffs; the scripts truncate the data at 2024-12-31 for discovery so the holdout cannot leak in):

- Fit: **2020-07-01 to 2022-12-31** (the universe reaches 30 seasoned perps around mid-2020).
- Validation: **2023-01-01 to 2024-12-31**.
- Holdout: **2025-01-01 to 2026-10-31** (the same exam window as #1315).

**Selection** (per family, using fit and validation only):

1. The family's representative is the variant with the highest net excess Sharpe in the **fit** window.
2. The family reaches the shortlist if its representative has net excess Sharpe > 0.7 in fit **and** > 0.7 in validation, validation max drawdown < 25%, and every variant in the family has a positive net return over fit + validation combined (no sign flips across neighbours).
3. At most 5 families go forward, ranked by validation Sharpe.
4. **Mix**: if two or more families are shortlisted, one extra candidate is their combination with weights inversely proportional to each one's fit+validation volatility, the sum rescaled to 15% vol the same way. Weights are frozen before the holdout is read.

## 5. Primary metric, holdout gate and candidate budget

**Primary metric**: net Sharpe in excess of idle USDT, on the holdout, with one-day-delayed fills.
Secondaries (reported, never decisive): CAGR, max drawdown, worst day, calendar-year returns, turnover, cost and funding drag, same-close-fill sensitivity, correlation to BTC.

**Holdout gate** (a candidate must meet all):

1. Net excess Sharpe ≥ 1.0.
2. Net CAGR ≥ 10%/yr (CC: ≥ 8%/yr, i.e. twice the idle-USDT hurdle, at its natural size).
3. Max drawdown ≤ 20% (CC: ≤ 10%).
4. Positive net return in both 2025 and 2026 (Jan-Oct).
5. Positions held on at least 80% of holdout days (activity floor, the analogue of a minimum trade count; CC: at least 50%, since it is flat by design when funding is low).

**Statistical honesty.** The holdout is 22 months, so a Sharpe estimate has a standard error of about 0.74. A pass at 1.0 is suggestive, not proof. A candidate is labelled a **strong pass** only at Sharpe ≥ 1.8, which is the one-sided 5% level after a Bonferroni correction for the six candidates this experiment may send. Results report both labels.

**Candidate count on the 2025-01 → 2026-10 exam window**: 3 so far (S1, S2, S3 from #1315, plus the BTC control). This experiment may add at most 6 (five families + the mix), for 9 in total. The window is retired after this experiment; future work needs data after 2026-10-31.

**Known contamination**: #1315 showed that this holdout was a falling market (BTC -6% in 2025, -12% in 2026 to October). Knowing that favours strategies that can short. The market-neutral families (XM, XR, XC, CC, LV) are least exposed to that hint; T, AB and NL are more exposed, and the results will say so.

## 6. Decision each outcome triggers

- **One or more candidates pass (or strong-pass)**: write a proposal for an 8-week paper trial on staging of the passing mix. That needs a small build (a multi-coin long/short perp portfolio executor) and it is the Board's decision; nothing is deployed and the kill switch stays on until Alex says otherwise in the thread.
- **Shortlist is non-empty but nothing passes the holdout**: no build. Write up which premia were real in 2020-24 and decayed, keep the funding watcher as the only live process, and say plainly that the crypto search has not found an edge. Any further crypto idea then needs a new mechanism, not a re-parameterisation of these families.
- **No family passes validation**: same as above, and the holdout stays untouched by this experiment.
- **Inconclusive** (a candidate passes gates 1-3 but fails gate 4 or 5 narrowly): no build; the candidate may be paper-traded only if the Board asks, with the weak evidence stated.

## 7. Risks of a false positive

- **Short holdout, one regime.** 22 months of a falling market. Handled by the strong-pass label and by requiring fit and validation (which include the 2020-21 bull run and the 2022 crash) to agree.
- **Multiple comparisons.** 21 variants in discovery; handled by the fit → validation → holdout chain, the neighbour sign-stability rule and the 6-candidate cap.
- **Execution optimism on small coins.** Handled by the top-30 liquidity filter, tiered slippage and the one-day delay. Not handled: capacity above roughly $1-2m, and exchange outages on the days that matter.
- **Short-squeeze tails.** A short in a coin that doubles in a day is a real loss the daily bar shows but a stop would not have prevented. Worst-day loss is reported for every candidate.
- **Funding data gaps.** Covered by the missing-print rule; coverage is reported.
- **Archive drift.** The archive is immutable per month; the run is repeated twice and the two result files must match byte for byte before any number is trusted.
- **Counterparty.** Everything sits on one exchange. Not in the numbers.

## 8. Protocol

1. Download and cache the data; print coverage. No strategy code runs in this step.
2. Commit this file.
3. `python experiments/strategy_search/run_search.py discover` twice (data truncated at 2024-12-31); compare outputs; apply section 4; record the shortlist and the mix weights in a results file and commit it **before** step 4.
4. `python experiments/strategy_search/run_search.py holdout` (reads the committed shortlist), twice; compare; apply section 5.
5. Append results, verdict and a `log.md` entry. Negative results get the full write-up.
