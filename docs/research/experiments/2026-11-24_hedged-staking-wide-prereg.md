# Staking with a hedge on 16 coins, and two mixes: preregistration

**Date**: 2026-11-24
**Status**: PREREGISTERED. Definitions fixed and pushed before any funding, basis or combined yield for the new coins or the mixes has been computed. Corrections go in new dated sections.
**Authority**: Alex, 2026-11-24 10:46 UTC, project thread, about the ETH and SOL result of 2026-11-23: "Why is this not worth building? That seems like ok returns to me? Maybe we can also try it with other coins or a mix until we find something even more profitable?"
**Script**: `experiments/strategy_search/yields.py hedged-wide` (output `results/hedged_staking_wide.json`).
**Follows**: `2026-11-23_hedged-staking-yield-prereg.md` and its results (ETH 3.4%, SOL 3.7% in 2026; gate 7%).

## 0. What is measured

The same structure as before, on every coin where it can be measured: hold a liquid staking token and short the same amount of the coin's Binance perpetual. The price cancels. What is left is the staking yield plus the funding the short receives or pays. Coins with high staking yields often have perpetuals whose funding runs against shorts, because this trade is known; the measurement shows what is left after that.

Two mixes are then built from the per-coin series with rules fixed here.

## 1. Hypotheses

**H1**: a mix of hedged staking positions earned at least 7% a year on total capital, net of costs, in both 2025 and January-October 2026, with a drawdown of at most 5%.

**H0**: across coins the staking yield is offset by funding paid, switching costs and the occasional squeeze on the short, and a mix earns about what idle dollars earn (4%) or less.

## 2. Coins (fixed by rule, before any outcome was computed)

Rule: every pool that DefiLlama classes as liquid staking, single-asset, whose coin has a Binance USDT perpetual in the Phase 1b cache and whose published yield covers at least 180 days inside 2025-01-01 to 2026-10-31; one pool per coin, the largest by value locked on 2026-11-23, at least 1 million dollars. ETH and SOL keep the pools used on 2026-11-23.

| Coin | Staking token | Yield history in the window (days) | Median published yield in the window |
|---|---|---|---|
| ETH | Lido stETH | 669 | 2.6% |
| SOL | Jito JitoSOL | 585 | 6.0% |
| HYPE | Kinetiq kHYPE | 245 | 2.0% |
| BNB | Lista slisBNB | 527 | 7.0% |
| AVAX | Benqi sAVAX | 668 | 5.1% |
| LINK | stake.link stLINK | 653 | 4.9% |
| FIL | GLIF iFIL | 475 | 6.5% |
| TON | Hipo hTON | 669 | 3.9% |
| DOT | Bifrost vDOT | 627 | 11.0% |
| BERA | Infrared iBERA | 457 | 2.9% |
| ICP | WaterNeuron nICP | 604 | 12.3% |
| STRK | Endur xSTRK | 669 | 7.4% |
| MON | shMonad shMON | 334 | 13.2% |
| FLOW | Ankr ankrFLOW | 182 | 7.9% |
| POL | Stader MaticX | 669 | 2.4% |
| APT | Amnis amAPT | 608 | 9.5% |

Left out by the rule: NEAR (44 days of history), STX (37 days), FLR and ZIG (no Binance perpetual). The published yields above are inputs and were looked at to apply the rule; funding and basis for the 14 new coins have not been looked at.

Several of these pools are small (2 to 15 million dollars). That does not matter for an account of this size, but it does mean real smart-contract risk, which the numbers do not contain.

## 3. Definitions (fixed)

- **Staking yield for a day** = the 7-day trailing median of the pool's published APY, limited to between 0% and 25%, divided by 365. The median and the cap remove single-day reporting spikes (several series print days above 40%).
- **Daily return per hedged dollar** = staking yield + funding received by the short that day + (spot return − perp return) − liquidation penalty. Where Binance has no spot pair for the coin, the spot-perp term is zero.
- **Liquidation penalty.** With 25 cents of margin per hedged dollar, a rise of 20% or more inside one day (perp high against the previous close) is treated as the short being liquidated and re-opened: 1.5% of the hedged dollar is charged that day. The staking token gains what the short loses, so the price itself still cancels.
- **Capital.** Every hedged dollar carries 25 cents of idle margin, so returns are divided by 1.25, as in the primary version on 2026-11-23.
- **Per-coin table (descriptive)**: each coin held throughout, with 0.30% a year in costs as before. Reported by year: staking, funding, basis change, liquidation days, net on total capital, drawdown.
- **M1, equal weight.** On the first day of each month, equal weight across every eligible coin. A coin is eligible when it has at least 25 days of data in the previous 30 and data on the day.
- **M2, top three by trailing yield.** On the first day of each month, rank eligible coins by their average daily return per hedged dollar over the previous 30 days (annualised, on total capital). Three slots of one third each go to the top three. A slot whose coin's trailing figure is not above 4% a year holds idle dollars at 4% instead.
- **Switching cost (both mixes)**: 0.25% of every dollar moved into or out of a hedged position (spot 0.10%, perp 0.05%, staking-token swap 0.10%), charged on the day of the change, including the first entry. No separate annual cost is charged to the mixes.
- **Window**: 2025-01-01 to 2026-10-31. Prices and funding from the Phase 1b cache; staking yields from DefiLlama.

## 4. Metric and gate

**Primary**: net return on total capital for M1 and for M2, in 2025 and in January-October 2026 (annualised as on 2026-11-23).

H1 is supported for a mix if both years are at least 7% and the largest drawdown on daily marks over the window is at most 5%. Otherwise H0.

The per-coin table is descriptive. A single coin clearing 7% in both years is not a pass: it would have been picked in hindsight from sixteen. It is listed as such and nothing is built on it.

Also reported: what each mix held month by month, switching costs, share of capital hedged (M2), worst day, and the count of liquidation days per coin.

## 5. Decision per outcome

- **H1 for a mix**: propose to Alex a forward paper record of that mix (daily yields recorded as they are published, no money) before any trial, with the unmeasured risks listed. Where the staking tokens live (several different chains) and what moving a small account between them costs would be worked out at that point.
- **H0 for both**: reported as such, with the table. No build, and no further mixes of the same series (other lookbacks, other slot counts) on this window.

Either way the bot stays paused.

## 6. Window accounting and limits

- These two mixes are candidates twelve and thirteen on the 2025-26 window, under Alex's decision of 2026-11-24 that it may be used once per candidate. The inputs are realised yields and funding, not price forecasts, so the scope for a lucky pass is smaller than for a trading rule, but M2 does select coins and can still flatter itself; hence the paper record before anything else.
- The staking token is assumed to track its coin. Small staking tokens can trade well below their coin when exits queue up. Not in the numbers.
- Costs of using several chains (bridges, gas, minimum sizes) are not in the numbers. On an account of about 85 dollars they could exceed the yield.
- DefiLlama's yields are what the protocols publish, including any reward tokens they fold into the base figure. Some series are short or flat for long stretches (FLOW prints the same value every day).
- Moving margin between the two legs after large moves is represented only by the liquidation penalty.
- Funding is Binance's.

## Amendment 1 (2026-11-24, after the results): a forward check of LINK

The results left one coin above 7% in both years, LINK (stLINK long, LINKUSDT perp short), picked in hindsight from sixteen. A hindsight pick can only be judged on data that did not exist when it was picked, so this section fixes a forward check. It costs nothing to run: the inputs are published daily and the calculation is `yields.py hedged-wide`, unchanged.

- **Definition**: the per-coin LINK series exactly as in section 3 (7-day median yield, funding, basis, liquidation penalty, 0.30% a year in costs, 25% margin).
- **Window**: from 2026-11-01, the first day after the data used above.
- **Check dates**: 2027-01-20 (with the trend paper trader's week 8 check) and 2027-05-26 (with its week 26 check).
- **Bar at the second date** (about seven months of new data): at least 7% a year on total capital, annualised, and a drawdown of at most 2%. If met, propose a small trial to Alex with the unmeasured risks listed (stLINK trading below LINK, the staking contract, the exchange). If not, closed.
- The first date is a progress reading only. No decision is taken on two and a half months.
- No money and nothing deployed in either case.
