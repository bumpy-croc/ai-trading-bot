# Board diagnostic: at what account balance do position-sizing constraints start to bind?

**Date:** 2026-08-13
**Requested by:** Alex (Board), 2026-08-13
**Author:** quant-researcher
**Type:** Analysis only — no engine or config changes. No deposits.

## Hypothesis

"The live account (~$83.44 equity) is structurally handicapped by its size: percentage fees
eat a disproportionate share of returns, and exchange minimum-notional / lot-size constraints
cause trades to be rejected or skipped that would execute cleanly on a larger account." This is
falsifiable: if the sizing distribution HyperGrowth actually produces at $83 clears exchange
minimums with comfortable margin, and fees are confirmed to scale proportionally rather than
regressively, the hypothesis is rejected for the current strategy/config.

## Metric

For each balance in {$80, $150, $250, $500, $1,000, $5,000}:
1. Whether the strategy's realized entry-notional range clears Binance's `NOTIONAL` filter
   (`minNotional`) for ETHUSDT.
2. Lot-step (`LOT_SIZE.stepSize`) quantization error as a % of intended position.
3. Whether HyperGrowth's configured partial-exit / scale-in sub-fractions, applied to a
   position of that balance, clear `minNotional` — computed as a labeled hypothetical, since
   this code path is currently hard-disabled in live.

## Success threshold

A "materially binding" constraint is defined as: >10% of a representative sizing distribution
rejected/skipped, or >2% distortion from quantization, or a scale-in/partial-exit sub-order
structurally below `minNotional` at the current live balance. Below that, the constraint is
"not binding" at that balance.

## Risks of false positive

- Conflating a *deliberate* zero-size decision (confidence gate declines to trade) with a
  *capital-driven* zero-size decision (sizer produces a non-zero-but-sub-minimum quantity).
  These have different remedies (accept idleness vs. add capital) and must not be merged.
- Reusing a stale historical incident (GH #700) without checking which sizing architecture was
  live at the time — the strategy has changed since.
- Assuming backtest units (P&L/fees) transfer to live when the partial-ops path has documented
  live/backtest divergence (GH #734).

---

## 1. Fee model — verifying the PM's claim

`src/engines/shared/cost_calculator.py` (`CostCalculator.calculate_entry_costs` /
`calculate_exit_costs`) computes:

```python
fee = notional * fee_rate   # DEFAULT_FEE_RATE = 0.001 (0.1%), src/config/constants.py:153
```

There is **no minimum fee floor, no fixed per-order fee, and no tiered discount for small
notional** anywhere in the cost path. `fee_rate` and `slippage_rate` (`DEFAULT_SLIPPAGE_RATE =
0.0005`) are pure percentages of `notional`, applied identically at $16 notional or $1,000
notional. **PM's claim confirmed: percentage-based exchange fees do not scale regressively
with account size in this codebase.** A $16 position pays the same 0.1%/0.05% as a $1,000
position, in relative terms. (In absolute terms a small account obviously pays a smaller dollar
fee — the point is the *rate* isn't worse.)

## 2. HyperGrowth's real sizing behaviour (verified in code, not assumed)

`src/strategies/hyper_growth.py::create_hyper_growth_strategy`:

- **Risk manager:** `FlatRiskManager` (own file, lines 71–165). `calculate_position_size`
  returns `balance * risk_fraction` (default `risk_fraction=0.25`) as a **binary gate**: `0.0`
  if `signal.confidence < min_confidence` (default `0.05`), else the full flat fraction — no
  proportional confidence scaling.
- **Position sizer:** `FixedFractionSizer(fraction=base_fraction=0.25, adjust_for_confidence=False,
  adjust_for_strength=False)` wrapped in `LeveragedPositionSizer`. Per
  `FixedFractionSizer.calculate_size` (`position_sizer.py:189-220`), `risk_amount` from the risk
  manager is used **only as a >0 veto**, not as a scaling numerator — the actual position size is
  `balance * 0.25 * leverage_multiplier`, confidence-independent once past the gate. (GH #938
  independently documents this: "HyperGrowth's flat position sizing makes it structurally blind
  to ML model quality above a low confidence gate.")
- **Leverage multiplier:** `LeverageManager` with `max_leverage=1.0` (leverage disabled by
  design — the docstring notes leverage cut returns -32% in testing). Regime-conviction ramp
  produces a multiplier in **[0.0, 1.0]**; the code's own comment (`hyper_growth.py:186-188`)
  states realized notional lands at **0.46–0.80× base_fraction (11.5%–20% of balance)** in
  practice, confirmed independently by `.claude/state/log.md` ops entries ("prod positions
  9.2–16% notional").
- **Engine-level cap:** `src/risk/risk_manager.py::_parse_position_sizing_params` clamps
  `max_fraction` to `self.params.max_position_size`. Live prod's `railway.json` passes
  `atb live-health hyper_growth --max-position 0.20` — matching the ratified
  `src/config/risk-limits.json` `position.max_position_size_pct = 0.20`. (GH #836, shipped,
  fixed an earlier divergence where prod ran at 0.5; GH #1021 covers a separate backtest-harness
  clamp bug, not live.) So the effective ceiling is 0.20, below the strategy's own 0.25 request.

**Modeled sizing distribution used below:** binary — either **$0 (declined trade)** when
confidence < 0.05 or the regime map returns leverage 0 (confirmed bear + high vol), or an
**active entry in the range 11.5%–20% of balance**, saturating at the 20% ratified cap. There
is no continuous middle; HyperGrowth never produces, say, a 3% or 7% position. This matters
because the whole "sizing distribution partially falls below the minimum" framing only applies
if positions get small — HyperGrowth's active positions are always large relative to balance by
construction.

## 3. Real ETHUSDT exchange filters (live, fetched 2026-08-13)

```
$ curl https://api.binance.com/api/v3/exchangeInfo?symbol=ETHUSDT
PRICE_FILTER   tickSize   = 0.01
LOT_SIZE       minQty     = 0.0001   stepSize = 0.0001
NOTIONAL       minNotional = 5.00    (applyMinToMarket: true)
```
ETHUSDT spot price at fetch time: **$1,880.71**.

**Finding (flagged, not in scope to fix here):** `src/data_providers/binance_provider.py:2041-2070`
(`get_symbol_info`) reads `filters.get("MIN_NOTIONAL", {})`. Binance renamed this filter type
from `MIN_NOTIONAL` to `NOTIONAL` in its 2023 API update — confirmed live for ETHUSDT and BTCUSDT
above. The bot's own pre-trade minimum-notional guard (`execution_engine.py:1357-1364`) therefore
always reads `min_notional=0` and never fires; it is silently dead code. The test suite mirrors
the same stale key (`tests/unit/data_providers/test_binance_provider.py:2098`), so it doesn't
catch this. **This does not change the numbers below** — the exchange itself still enforces
`NOTIONAL` server-side (any sub-$5 order gets rejected with error -1013, not silently
undersized), and every entry computed here clears $5 by 1.8×+ regardless. It just means the bot
would find out via an exchange rejection rather than a preemptive skip, if it ever got close.
Opened as GH issue (see below) — not fixed in this diagnostic per the "analysis only" scope.

## 4. Computed table

Entry notional range = balance × [11.5%, 20%] (the strategy's real active-sizing band, not an
assumed constant). All partial/scale-in figures are **hypothetical — `live_partial_operations`
is confirmed hard-disabled** (see §5).

| Balance | Entry notional (low–high) | Min-notional headroom (low end) | Qty @ 20% cap (ETH) | Lot-quantization error | Smallest partial/scale-in sub-order (low end, 11.5%) | Smallest sub-order (high end, 20%) |
|---:|---:|---:|---:|---:|---:|---:|
| $80    | $9.20 – $16.00   | 1.84× | 0.00850 | −0.09% | **$1.84 (FAIL)** | **$3.20 (FAIL)** |
| $150   | $17.25 – $30.00  | 3.45× | 0.01600 | +0.31% | **$3.45 (FAIL)** | $6.00 (pass) |
| $250   | $28.75 – $50.00  | 5.75× | 0.02660 | +0.05% | $5.75 (pass, thin) | $10.00 (pass) |
| $500   | $57.50 – $100.00 | 11.5× | 0.05320 | +0.05% | $11.50 (pass) | $20.00 (pass) |
| $1,000 | $115.00 – $200.00| 23.0× | 0.10630 | −0.04% | $23.00 (pass) | $40.00 (pass) |
| $5,000 | $575.00 – $1,000.00| 115×| 0.53170 | ≈0.00% | $115.00 (pass) | $200.00 (pass) |

"Smallest sub-order" = the smallest of HyperGrowth's configured partial-exit fractions
`[0.20, 0.30, 0.50]` and scale-in fractions `[0.40, 0.25]` (`hyper_growth.py:412-415`, applied
to the entry position) — the first one to hit `minNotional` as balance shrinks.

## 5. Partial-exit / scale-in: is this path even active?

**No.** `src/engines/live/trading_engine.py:559-580` and `src/engines/live/config.py:77`
gate the entire partial-operations subsystem behind the `live_partial_operations` feature flag,
which resolves via `flag_lookup("live_partial_operations", False)` — **default OFF**, and no
override was found in the repo's config/flag files. This was an interim mitigation for GH #734
(P0: partial exits/scale-ins are bookkeeping-only in the live engine — no real exchange order is
placed, and a fraction-of-original vs fraction-of-balance unit mismatch produces phantom PnL and
desyncs tracked size from real holdings). GH #734 remains **OPEN**; the underlying bug is not
fixed, only the live path is disabled. **The $150–$250 knee computed above is therefore
inoperative today** — it describes when the feature would become viable *if* re-enabled, not a
present constraint.

## 6. Cross-check against production history

Searched `.claude/state/log.md` and GitHub issues for evidence of minimum-notional / lot-size
rejections or zero-sized orders on this account:

- **GH #700** ("Prod bot idle: risk×confidence sizing falls below exchange minimum on small
  $82 account") — a **real, confirmed historical zero-qty rejection**:
  `Calculated quantity 0.00000000 below minimum 0.00010000 for ETHUSDT`. Opened
  2026-06-05T22:31Z, the *same day* session 20 (HyperGrowth, still running today) started
  (2026-06-05T20:59Z per GH #734's ops confirmation). Its described mechanism —
  "risk × confidence sizing," size ≈0.08 at confidence 0.08–0.15 — is a *proportional*
  confidence-scaled sizer, not HyperGrowth's binary-gated flat sizer. Git history confirms
  `FlatRiskManager` existed since 2026-03-14 (commit `35695837`) but the live strategy switched
  to it (session 20) right around when #700 was filed — consistent with #700 being the incident
  that motivated the cutover away from confidence-scaled sizing, on whatever strategy was live
  immediately before HyperGrowth. **This is real evidence the constraint used to bind, under an
  architecture the account no longer runs.** It does not describe HyperGrowth's current behavior,
  which the computation above shows clears the minimum by 1.8×+ even at $80.
- **GH #1045** (staging, $1,019 balance, 30 min of `Size: 0.00` BUY decisions, no logged
  reason) — explicitly **not** a capital-size effect; the issue itself notes "#700 does not
  explain it (balance ~$1019, not ~$82)." Root cause still open, unrelated to this diagnostic.
- No log.md entry or GH issue found describing a HyperGrowth-era (post 2026-06-05) minimum-
  notional or lot-size rejection on the live ETHUSDT account. Consistent with the computed
  table: HyperGrowth's active entries never get small enough to approach $5.

## Verdict

**Minimum-notional rejections on entries: do NOT bind at $83, or at any balance tested up to
$5,000.** HyperGrowth's flat, gate-then-full-size architecture means entries are always $9–$17
at the current balance — 1.8×+ above the $5 `NOTIONAL` floor — because there is no small/partial
sizing tier to fall into. The historically real rejection (GH #700) happened under a *different*,
now-retired confidence-scaled sizer, not the one currently trading.

**Lot-step quantization error: never material.** ETHUSDT's `stepSize` (0.0001 ETH ≈ $0.19) is
small relative to any position HyperGrowth takes; error stays under ±0.31% at every balance
tested — smaller than one side of the trading fee.

**Partial-exit / scale-in minimums: would bind between ~$150 and ~$250 if re-enabled**, but the
feature is hard-disabled today (`live_partial_operations=False`, GH #734 open) specifically
because of a P0 correctness bug unrelated to sizing. This is the one place a knee exists in the
math, and it is currently moot.

**One-line answer:** at the current balance and strategy configuration, **no position-sizing
constraint is binding — capital is not the reason the account trades or doesn't trade.** The
"small account is structurally handicapped" hypothesis is **rejected** for HyperGrowth as
configured. The nearest latent knee (partial-ops re-enablement, ~$150–$250) is inactive and
gated on a correctness fix (#734), not on account size.

## What this doesn't tell you

This diagnostic is scoped to sizing/notional mechanics only. It says nothing about whether
HyperGrowth has positive expectancy at any balance (log.md already documents six independent
null results and a corrected -20.15%/365d honest backtest) — a bigger, unconstrained account
trading a negative-expectancy strategy loses money faster, it doesn't start winning. Sizing
headroom is not evidence for adding capital; it only retires one specific objection to adding
capital.

## Follow-ups opened

- GH issue: `binance_provider.get_symbol_info` reads the stale `MIN_NOTIONAL` filter key;
  Binance now returns `NOTIONAL` — the bot's own pre-trade min-notional guard is silently dead
  code (exchange-side enforcement still applies). Filed, not fixed here (out of scope for an
  analysis-only diagnostic).

## Data sources

- `src/strategies/hyper_growth.py`, `src/strategies/components/{position_sizer,risk_manager}.py`,
  `src/strategies/components/leverage_manager.py`
- `src/risk/risk_manager.py` (`_parse_position_sizing_params`)
- `src/engines/shared/cost_calculator.py`, `src/config/constants.py`
- `src/engines/live/trading_engine.py`, `src/engines/live/config.py` (partial-ops flag)
- `railway.json` (`--max-position 0.20`)
- `src/config/risk-limits.json`
- Binance public API `exchangeInfo` for ETHUSDT/BTCUSDT, fetched 2026-08-13
- `.claude/state/log.md`; GH #700, #734, #836, #938, #1021, #1045
