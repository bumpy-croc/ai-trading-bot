# Exit taxonomy: design rationale and historical mapping (GH #1115)

Date: 2026-08-25 · Branch: `fix/1115-exit-taxonomy` · Status: implemented

## 1. The defect

Two separate problems shared one root cause — `trades.exit_reason` was free text with no
typed companion.

**Control flow branched on prose.** `src/engines/backtest/execution/exit_handler.py:247` did
`if "Stop loss" in exit_reason:` to choose `OrderType.STOP_LOSS` and, in the live twin, to
apply worst-case gap pricing. The spellings `stop_loss` and `stop_loss_filled_offline` do not
contain `"Stop loss"`, so those exits silently fell through to a market order priced at mid
candle. The live engine had a *different* matcher —
`_is_stop_loss_reason` accepted `"stop loss" | "stop_loss" | "stop-loss"` case-insensitively —
so the two engines disagreed about what even counted as a stop exit. Every substring match
found in the sweep:

| Site | Matched on | Drove |
|---|---|---|
| `backtest/execution/exit_handler.py:247` | `"Stop loss" in …` | exit order type |
| `backtest/execution/exit_handler.py:250` | `"Take profit" in …` | exit order type |
| `live/execution/exit_handler.py:200-206` | `_is_stop_loss_reason` (3 spellings) | order type **and** stop-gap pricing |
| `live/execution/exit_handler.py:224` | `"take profit" \| "take_profit"` | order type |
| `dashboards/monitoring/dashboard.py:1474` | `exit_reason = 'failed'` | "failed orders" tile |

The dashboard filter is its own evidence: no producer has ever written `'failed'`, so that
tile read zero permanently. A vocabulary nobody owns drifts, and nothing tells you when.

**The same logical exit was written several ways.** Producer inventory as of `develop`:

| Producer | Emitted string |
|---|---|
| `backtest/execution/exit_handler.py:682` | `Stop loss` |
| `live/execution/exit_handler.py:375` | `Stop loss` |
| `live/ws_health.py:283` (deferred exchange-stop fill) | `stop_loss` |
| `live/reconciliation.py:2173` (offline SL fill, periodic) | `stop_loss_filled_offline` |
| `live/recovery.py:800,822` (offline SL fill, startup) | `stop_loss_offline` |
| `live/execution/entry_coordinator.py:1006` | `Stop-loss placement failed - emergency close` |
| `live/execution/entry_coordinator.py:877` | `Risk manager sync failure` |
| `live/trading_engine.py:1458` | `Engine shutdown` |
| `live/strategy_hot_swap.py:140` | `Strategy change - close requested` |
| `live/reconciliation.py:2262` | `external_close_recovery` |
| `live/reconciliation.py:1114` | `exit_order_recovery` |
| `live/account_sync.py:752` | `recovered_from_exchange` |

Four different spellings for "a stop level was hit".

## 2. The design decision that matters

The issue observed that in prod `Stop loss` was net **+5.52** while `stop_loss` was net
**−1.58**, and hypothesised that trailing stops taking profit were being recorded under the
same label as protective stops taking losses.

**The hypothesis is right about the conflation and wrong about the split.** The producer that
wrote the string is not the discriminator — prod trade #15 was written by the `stop_loss`
producer *and* had a fully trailed stop (+0.41). Reading the sign of the P&L would have
produced a plausible, wrong mapping. What actually discriminates is position state:
`trailing_stop_activated` and `breakeven_triggered`, which both engines maintain through the
shared `TrailingStopManager` and which are persisted on the `positions` row.

So the taxonomy splits a stop-level exit three ways, and one function
(`classify_stop_exit`) is the only place that decides:

- `stop_loss` — the stop was never moved off its initial level. Capital was **protected**;
  the planned loss was taken.
- `breakeven_stop` — the stop reached breakeven but trailing never activated. Roughly flat.
- `trailing_stop` — trailing had activated. This is a **profit-taking** exit.

Everything else in the enum names a distinct logical event: `take_profit`, `signal_exit`,
`time_exit`, `early_cut`, `partial_exit_complete`, `emergency_close`, `engine_shutdown`,
`strategy_change`, `external_close`, `recovered`, `unknown`.

### Deliberate deviation from the issue's proposal

The issue suggested `STOP_FILLED_OFFLINE` as its own category. It is not one here. "Filled
while the bot was down" is a *circumstance*, not a kind of exit — the position hit its stop
either way. Giving it a category would re-create the exact fragmentation the taxonomy exists
to end: an analyst grouping by category would once again see stop exits split across buckets.
The circumstance stays in the free-text `exit_reason` (`stop_loss_filled_offline`), which is
preserved verbatim, and the category is the stop kind, classified from the recovered
position's own flags. Same argument applies to `stop_loss_offline` and to the ws_health
deferred-drain path.

### Why `exit_reason` keeps its exact values

`exit_reason` is not only read by humans. `live/execution/exit_coordinator.py:520` interpolates
it into the balance-ledger key `realized_pnl_<SYM>_<reason>` in
`account_balances.update_reason`, which `prod-forensics` money-flow decomposition groups on.
It also backs a pre-registered experiment metric (`exit_geometry_round2_sweep.py:233` matches
the `Early cut` prefix, defined in `docs/research/experiments/2026-07-12_exit-geometry-round2.md`).
Renaming the strings would split historical ledger groupings and retroactively invalidate a
pre-registered metric. The change is therefore strictly **additive**: `exit_reason` byte-identical,
`exit_category` new and typed. Every existing consumer keeps working.

## 3. What changed

- `src/trading/exit_reason.py` — the closed `ExitReason` StrEnum, `STOP_EXIT_CATEGORIES`,
  `classify_stop_exit`, `coerce_exit_category`, and the legacy mapping. Placed in
  `src/trading/` rather than `src/engines/shared/` so the database layer can import it without
  depending on the engines.
- Every substring match replaced by an enum comparison. `_is_stop_loss_reason` is deleted.
- `ExitCheckResult` (backtest) and `LiveExitCheck` (live) carry `exit_category`; it threads
  through `execute_exit` / `_execute_exit` / `execute_filled_exit` /
  `_log_reconciliation_trade` to `log_trade`.
- `trades.exit_category VARCHAR(32)` + index, via migration `0014_add_exit_category`.
- `v_trades_exit_category` view — resolves a category for every row and flags inferred ones.
- Dashboard reads and displays the category (detail as tooltip); the dead `'failed'` filter now
  counts `emergency_close`.
- `.claude/skills/trade-review/SKILL.md` pass 2 rewritten to group on the category.

`classify_stop_exit` reads its flags with `is True`, not truthiness: a `MagicMock` position in
a test exposes every attribute as truthy and would otherwise classify every stop as trailing
(`.claude/LESSONS.md`).

## 4. Historical mapping — the 18 prod rows

Read-only from prod (`SET default_transaction_read_only = on;`) on 2026-08-25, joining
`trades` to `positions` on `position_id`. **No rows were written.** The migration does not
backfill: the view resolves categories at read time and marks every pre-#1115 row
`exit_category_inferred = true`, because `exit_category IS NULL` on all of them.

The table below is the *authoritative* mapping, and it is better than the view can be — the
view only sees the prose, whereas this reconstruction consulted `positions.trailing_stop_activated`.
"KNOWN" means the position row was joinable and carried the flags.

| id | date | `exit_reason` | PnL | trailing / BE | Category | Confidence |
|---:|---|---|---:|---|---|---|
| 1 | 06-02 | Stop-loss placement failed - emergency close | +0.0033 | — | `emergency_close` | KNOWN (prose is unambiguous) |
| 2 | 06-02 | Stop-loss placement failed - emergency close | −0.0011 | — | `emergency_close` | KNOWN |
| 3 | 06-02 | Stop-loss placement failed - emergency close | −0.0045 | — | `emergency_close` | KNOWN |
| 4 | 06-02 | Engine shutdown | −0.0019 | — | `engine_shutdown` | KNOWN |
| 5 | 06-05 | stop_loss | −0.6530 | False / False | `stop_loss` | KNOWN (position 13) |
| 6 | 06-05 | Stop-loss placement failed - emergency close | −0.0094 | — | `emergency_close` | KNOWN |
| 7 | 06-07 | Stop loss | +0.2392 | True / False | `trailing_stop` | KNOWN (position 16) |
| 8 | 06-07 | Stop loss | +0.3597 | True / False | `trailing_stop` | KNOWN (position 17) |
| 9 | 06-14 | Stop loss | +0.4174 | True / False | `trailing_stop` | KNOWN (position 18) |
| 10 | 06-18 | Stop loss | +0.1599 | True / False | `trailing_stop` | KNOWN (position 19) |
| 11 | 06-23 | Stop loss | +0.3857 | True / False | `trailing_stop` | KNOWN (position 20) |
| 12 | 07-02 | Stop loss | +0.2422 | True / False | `trailing_stop` | KNOWN (position 21) |
| 13 | 07-14 | stop_loss | −1.3327 | False / False | `stop_loss` | KNOWN (position 22) |
| 14 | 07-20 | stop_loss_filled_offline | +0.2510 | *(no FK)* | `trailing_stop` | **INFERRED** — see below |
| 15 | 07-21 | stop_loss | +0.4091 | True / False | `trailing_stop` | KNOWN (position 24) |
| 16 | 08-19 | Stop loss | +1.3382 | True / True | `trailing_stop` | KNOWN (position 25) |
| 17 | 08-19 | Stop loss | +1.7492 | True / True | `trailing_stop` | KNOWN (position 26) |
| 18 | 08-20 | Stop loss | +0.6287 | True / False | `trailing_stop` | KNOWN (position 27) |

### The one inferred row

**Trade 14** is the only ambiguous row. It was written by the reconciler's offline-SL path,
which did not set `position_id`, so the FK is NULL and the flags cannot be read directly.
Corroboration, recorded so a future reader can judge it:

- `positions` row 23 (`entry_time 2026-07-17 12:40`, `entry_price 1829.55`, `stop_loss 1863.06`,
  `trailing_stop_price 1863.06`, `trailing_stop_activated = true`) matches trade 14's entry
  price exactly, and its stop level matches the trade's exit price of 1863.02 to within
  four cents.
- The trade is LONG with exit (1863.02) above entry (1829.55), so the stop *must* have been
  moved above entry — an untouched protective stop sits below entry on a long.

Both point to `trailing_stop`, and the second argument alone is close to conclusive. It is
still marked INFERRED because the join is reconstructed rather than recorded, and this
document should not launder a reconstruction into a fact.

Note that the three `emergency_close` rows and the `engine_shutdown` row are marked KNOWN on
prose alone — those strings have exactly one producer each and never described a stop.

### What the corrected decomposition says

| Category | Trades | Net PnL |
|---|---:|---:|
| `trailing_stop` | 11 (incl. 1 inferred) | **+6.18** |
| `stop_loss` | 2 | **−1.99** |
| `emergency_close` | 4 | −0.01 |
| `engine_shutdown` | 1 | −0.00 |

Total +4.18, reconciling with the issue's +4.18 across the old labels.

Protective stops in production are **0 for 2**; trailing stops are **11 for 11**. The old
vocabulary made this invisible in both directions — it split the trailing stops across two
labels and merged the protective ones into the larger of them. This is the first number the
exit-quality work should have, and it is the reason the instrument had to be repaired first.

Two protective stops is not a sample. It is a starting instrument reading, not a finding.

## 5. Reading historical data

```sql
SELECT exit_category_resolved,
       bool_or(exit_category_inferred) AS any_inferred,
       count(*),
       round(sum(pnl), 4) AS gross_pnl
FROM v_trades_exit_category
GROUP BY 1 ORDER BY 4 DESC;
```

The view maps prose to categories for rows the engines never labelled. For pre-#1115 stop
exits it necessarily returns `stop_loss` for every spelling — the prose cannot distinguish
protective from trailing. **Use the table in §4 for the 18 prod rows**; the view is the
general fallback, and `exit_category_inferred` is what tells you which is which.
