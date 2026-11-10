"""Part B of the other-markets bake-off: rule-based sources of return outside crypto.

Preregistration: docs/research/experiments/2026-11-10_other-markets-bakeoff-prereg.md

Nine canonical published rules (FX carry, FX trend, multi-asset trend long/cash and
long/short, dual momentum, sector rotation, RSI-2 mean reversion, turn of the month,
volatility risk premium) plus passive controls. Parameters are fixed; nothing is swept.

Timing: a signal computed on day t's close is filled at day t+1's close and first earns
on day t+2, unless a rule says otherwise (B7 fills at the next open, B8 is calendar-only).

Usage: python experiments/markets_bakeoff.py
Writes experiments/markets_results/markets_bakeoff.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import markets_common as mc
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "markets_results" / "markets_bakeoff.json"
DESIGN_END = "2014-12-31"
TRADING_DAYS = 252


# --------------------------------------------------------------------------- ETF plumbing


def load_etf_panel(symbols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(total-return price panel, daily return panel) on the union of trading days."""
    prices = pd.DataFrame({s: mc.load_yahoo(s)["adjclose"] for s in symbols}).sort_index()
    return prices, prices.pct_change(fill_method=None)


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().to_numpy())


def week_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    iso = index.isocalendar()
    return pd.DatetimeIndex(
        s.groupby([iso.year.to_numpy(), iso.week.to_numpy()]).last().sort_values().to_numpy()
    )


def year_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby(index.year).last().to_numpy())


def simulate(target: pd.DataFrame, rets: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    """Hold drifting ETF weights; rebalance to target[t] at the close of the day after t.

    `target` has one row per signal date (weights as fractions of equity; negative = short).
    Unallocated capital earns T-bills minus the cash haircut. Shorts pay the borrow fee and
    free up no cash. Returns total and excess daily returns, turnover and gross exposure.
    """
    dates = rets.index
    r_mat = rets.reindex(columns=target.columns).fillna(0.0).to_numpy()
    tgt = target.reindex(dates).fillna(0.0).to_numpy()
    is_signal = dates.isin(target.index)
    rf_arr = rf.reindex(dates).fillna(0.0).to_numpy()
    yrs = (pd.Series(dates, index=dates).diff().dt.days.fillna(1).clip(lower=1) / 365.0).to_numpy()
    w = np.zeros(tgt.shape[1])
    total, turnover, gross = np.zeros(len(dates)), np.zeros(len(dates)), np.zeros(len(dates))
    for i in range(len(dates)):
        long_w, short_w = w.clip(min=0).sum(), (-w).clip(min=0).sum()
        gross[i] = long_w + short_w
        cash_w = max(0.0, 1.0 - long_w)
        r = (
            float(w @ r_mat[i])
            + cash_w * (rf_arr[i] - mc.CASH_HAIRCUT * yrs[i])
            - short_w * mc.SHORT_BORROW * yrs[i]
        )
        w = w * (1.0 + r_mat[i]) / (1.0 + r)
        if i >= 1 and is_signal[i - 1]:
            turnover[i] = np.abs(tgt[i - 1] - w).sum()
            r -= turnover[i] * mc.ETF_COST
            w = tgt[i - 1].copy()
        total[i] = r
    out = pd.DataFrame({"total": total, "turnover": turnover, "gross": gross}, index=dates)
    out["excess"] = out["total"] - rf_arr
    return out


def trailing_return(prices: pd.DataFrame, lookback: int) -> pd.DataFrame:
    return prices / prices.shift(lookback) - 1.0


def tbill_trailing(rf: pd.Series, lookback: int) -> pd.Series:
    return (1.0 + rf).rolling(lookback).apply(np.prod, raw=True) - 1.0


# --------------------------------------------------------------------------- B3-B6, B9 and controls (ETF weights)


def b3_gtaa(prices: pd.DataFrame, lookback: int = 210) -> pd.DataFrame:
    cols = ["SPY", "EFA", "IEF", "DBC", "VNQ"]
    px = prices[cols]
    above = (
        px > px.rolling(lookback).mean()
    )  # NaN average (too little history) compares False -> cash
    return (above.astype(float) * 0.20).loc[month_ends(px.index)]


def b4_trend_ls(
    prices: pd.DataFrame, rets: pd.DataFrame, rf: pd.Series, lookback: int = 252
) -> pd.DataFrame:
    cols = ["SPY", "EFA", "EEM", "IEF", "TLT", "GLD", "DBC", "VNQ"]
    px = prices[cols]
    mom = trailing_return(px, lookback).sub(tbill_trailing(rf, lookback), axis=0)
    vol = rets[cols].rolling(60).std() * math.sqrt(TRADING_DAYS)
    w = np.sign(mom) * (0.10 / vol) / len(cols)
    w = w.where(mom.notna() & vol.notna(), 0.0)
    gross = w.abs().sum(axis=1)
    w = w.div(gross.clip(lower=1.0), axis=0)  # cap total gross at 1.0
    return w.loc[month_ends(px.index)]


def b5_gem(prices: pd.DataFrame, rf: pd.Series, lookback: int = 252) -> pd.DataFrame:
    px = prices[["SPY", "EFA", "AGG"]]
    mom = trailing_return(px, lookback)
    rf_mom = tbill_trailing(rf, lookback)
    rows = {}
    for t in month_ends(px.index):
        m = mom.loc[t]
        if m.isna().any():
            continue
        if m["SPY"] > rf_mom.loc[t]:
            pick = "SPY" if m["SPY"] >= m["EFA"] else "EFA"
        else:
            pick = "AGG"
        rows[t] = {c: float(c == pick) for c in px.columns}
    return pd.DataFrame.from_dict(rows, orient="index")


SECTORS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def b6_sectors(prices: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    px = prices[SECTORS]
    mom = trailing_return(px, lookback)
    spy = prices["SPY"]
    risk_on = spy > spy.rolling(210).mean()
    rows = {}
    for t in month_ends(px.index):
        m = mom.loc[t]
        if m.isna().any() or pd.isna(spy.rolling(210).mean().loc[t]):
            continue
        top = m.sort_values(ascending=False, kind="stable").index[:3] if risk_on.loc[t] else []
        rows[t] = {c: (1.0 / 3.0 if c in top else 0.0) for c in SECTORS}
    return pd.DataFrame.from_dict(rows, orient="index")


def b9_vrp(index: pd.DatetimeIndex) -> pd.DataFrame:
    vix = mc.load_yahoo("^VIX")["close"].reindex(index).ffill()
    vix3m = mc.load_yahoo("^VIX3M")["close"].reindex(index).ffill()
    svxy_live = mc.load_yahoo("SVXY")["adjclose"].reindex(index).notna()
    state = ((vix < vix3m) & svxy_live).astype(float) * 0.25
    change = state.ne(
        state.shift(1)
    )  # trade only when the state flips; let the sleeve drift otherwise
    return state[change].to_frame("SVXY")


def control_weights(
    prices: pd.DataFrame, weights: dict[str, float], dates: pd.DatetimeIndex
) -> pd.DataFrame:
    live = prices[list(weights)].notna().all(axis=1)
    rows = {t: weights for t in dates if live.loc[t]}
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- B7, B8 (own loops)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / down.replace(0, np.nan))


def b7_sleeve(symbol: str, rf: pd.Series, fill: str, trend_lookback: int = 200) -> pd.DataFrame:
    """RSI-2 mean reversion on one ETF. fill='open' (next open, the gate) or 'close' (same close)."""
    y = mc.load_yahoo(symbol)
    close, open_ = y["adjclose"], y["adjopen"]
    enter = (_rsi(close, 2) < 10) & (close > close.rolling(trend_lookback).mean())
    leave = close > close.rolling(5).mean()
    rf_d = rf.reindex(close.index).fillna(0.0)
    yrs = pd.Series(close.index, index=close.index).diff().dt.days.fillna(1).clip(lower=1) / 365.0
    idle = (rf_d - mc.CASH_HAIRCUT * yrs).to_numpy()
    c, o = close.to_numpy(), open_.to_numpy()
    total, in_mkt, traded = np.zeros(len(c)), np.zeros(len(c)), np.zeros(len(c))
    holding = False
    for i in range(1, len(c)):
        buy = (not holding) and bool(enter.iloc[i - 1])
        sell = holding and bool(leave.iloc[i - 1])
        if fill == "open":  # decided at the close of day i-1, filled at the open of day i
            if buy:
                total[i] = c[i] / o[i] - 1.0 - mc.ETF_COST
            elif sell:
                total[i] = o[i] / c[i - 1] - 1.0 - mc.ETF_COST
            else:
                total[i] = c[i] / c[i - 1] - 1.0 if holding else idle[i]
        else:  # decided and filled at the close of day i-1
            if buy:
                total[i] = c[i] / c[i - 1] - 1.0 - mc.ETF_COST
            elif sell:
                total[i] = idle[i] - mc.ETF_COST
            else:
                total[i] = c[i] / c[i - 1] - 1.0 if holding else idle[i]
        holding = (holding or buy) and not sell
        traded[i] = float(buy or sell)
        in_mkt[i] = float(holding)
    return pd.DataFrame({"total": total, "gross": in_mkt, "turnover": traded}, index=close.index)


def b7_rsi2(rf: pd.Series, fill: str, trend_lookback: int = 200) -> pd.DataFrame:
    sleeves = [b7_sleeve(s, rf, fill, trend_lookback) for s in ("SPY", "QQQ")]
    idx = sleeves[0].index.intersection(sleeves[1].index)
    out = sum(s.reindex(idx) for s in sleeves) / 2.0  # half of capital in each sleeve
    out["excess"] = out["total"] - rf.reindex(idx).fillna(0.0)
    return out


def b8_turn_of_month(rets: pd.DataFrame, rf: pd.Series) -> pd.DataFrame:
    spy = rets["SPY"].dropna()
    idx = spy.index
    month_key = pd.Series(idx.year * 100 + idx.month, index=idx)
    day_in_month = month_key.groupby(month_key).cumcount() + 1
    days_left = month_key.groupby(month_key).cumcount(ascending=False)  # 0 on the last trading day
    in_mkt = (days_left == 0) | (day_in_month <= 3)
    yrs = pd.Series(idx, index=idx).diff().dt.days.fillna(1).clip(lower=1) / 365.0
    rf_d = rf.reindex(idx).fillna(0.0)
    total = spy.where(in_mkt, rf_d - mc.CASH_HAIRCUT * yrs)
    # Entry is at the close before the first held day, exit at the close of the last held day.
    entry_next_day = in_mkt.astype(float).diff().shift(-1).fillna(0.0) > 0
    exit_today = in_mkt & ~in_mkt.shift(-1, fill_value=True)
    traded = (entry_next_day | exit_today).astype(float)
    total = total - traded * mc.ETF_COST
    out = pd.DataFrame(
        {"total": total, "gross": in_mkt.astype(float), "turnover": traded}, index=idx
    )
    out["excess"] = out["total"] - rf_d
    return out


# --------------------------------------------------------------------------- B1, B2 (FX)


def fx_run(
    w_sig: pd.DataFrame, px: pd.DataFrame, rates: pd.DataFrame, rf: pd.Series, markup_on: str
) -> pd.DataFrame:
    """Unfunded FX positions vs USD. w_sig rows are signal dates; weights are fractions of capital."""
    ccys = list(px.columns)
    held = w_sig.reindex(px.index).ffill().shift(2).fillna(0.0)[ccys]
    days = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1).clip(lower=1)
    yrs = days / 365.0
    r = rates.reindex(rates.index.union(px.index)).ffill().reindex(px.index).shift(1)
    carry = (r[ccys].sub(r["USD"], axis=0) / 100.0).mul(yrs, axis=0)
    pnl = (held * (px.pct_change() + carry)).sum(axis=1)
    cost_side = pd.Series({c: mc.FX_COST_BPS[c] / 1e4 for c in ccys})
    turnover = held.diff().abs().fillna(held.abs())
    costs = (turnover * cost_side).sum(axis=1)
    long_n, short_n = held.clip(lower=0).sum(axis=1), (-held).clip(lower=0).sum(axis=1)
    if markup_on == "pairs":  # currency-against-currency pairs: one markup per pair
        usd_w = -(held.sum(axis=1))  # the USD leg implied by a dollar-neutral basket
        notional = np.maximum(long_n + usd_w.clip(lower=0), short_n + (-usd_w).clip(lower=0))
    else:
        notional = long_n + short_n
    net = pnl - costs - mc.FX_MARKUP * notional * yrs
    rf_d = rf.reindex(px.index).fillna(0.0)
    out = pd.DataFrame(
        {
            "excess": net - mc.FX_IDLE_MARGIN * rf_d,
            "total": net + (1 - mc.FX_IDLE_MARGIN) * rf_d,
            "turnover": turnover.sum(axis=1),
            "gross": long_n + short_n,
        }
    )
    first = held.abs().sum(axis=1).ne(0).idxmax()
    return out.loc[first:]


def b1_carry(px: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    r = rates.reindex(rates.index.union(px.index)).ffill().reindex(px.index)[mc.G10]
    rows = {}
    for t in month_ends(px.index):
        ranked = r.loc[t].sort_values(
            ascending=False, kind="stable"
        )  # ties keep the fixed G10 order
        w = dict.fromkeys(mc.G10, 0.0)
        for c in ranked.index[:3]:
            w[c] = 1.0 / 3.0
        for c in ranked.index[-3:]:
            w[c] = -1.0 / 3.0
        rows[t] = {c: w[c] for c in px.columns}  # the USD leg is implicit
    return pd.DataFrame.from_dict(rows, orient="index")


def b2_fx_trend(px: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    mom = px / px.shift(lookback) - 1.0
    vol = px.pct_change().rolling(60).std() * math.sqrt(TRADING_DAYS)
    w = np.sign(mom) * (0.10 / vol) / px.shape[1]
    w = w.where(mom.notna() & vol.notna(), 0.0)
    gross = w.abs().sum(axis=1)
    w = w.mul((2.0 / gross).clip(upper=1.0).fillna(1.0), axis=0)  # cap total gross at 2.0
    w = w.loc[week_ends(px.index)]
    return w[(w.abs().sum(axis=1) > 0).cummax()]


# --------------------------------------------------------------------------- reporting


def ppy(index: pd.DatetimeIndex) -> float:
    return len(index) / ((index[-1] - index[0]).days / 365.25)


def summarise(name: str, run: pd.DataFrame, bench: pd.Series | None, gated: bool) -> dict:
    periods = ppy(run.index)
    design, holdout = run.loc[:DESIGN_END], run.loc[mc.HOLDOUT[0] : mc.HOLDOUT[1]]
    g = mc.gate(run["excess"], periods, bench_excess=bench)
    weekly = None
    if bench is not None:
        wk = (
            lambda s: (1.0 + s.loc[mc.HOLDOUT[0] : mc.HOLDOUT[1]]).resample("W-FRI").prod() - 1.0
        )  # noqa: E731
        weekly = mc.alpha_vs(wk(run["excess"]), wk(bench), 52.0)
    return {
        "name": name,
        "gated": gated,
        "start": str(run.index[0].date()),
        "design": (
            {**mc.perf(design["excess"], periods), **mc.total_stats(design["total"], periods)}
            if len(design) > 250
            else None
        ),
        "holdout": {**g["window"], **mc.total_stats(holdout["total"], periods)},
        "full": {**mc.perf(run["excess"], periods), **mc.total_stats(run["total"], periods)},
        "alpha_vs_6040": g.get("alpha"),
        "alpha_vs_6040_weekly": weekly,
        "recent_mean_ann": g["recent_mean_ann"],
        "gates": {k: bool(g[k]) for k in ("G1", "G2", "G3", "G4", "G5") if k in g},
        "verdict": g["verdict"],
        "holdout_turnover_per_year": float(holdout["turnover"].sum() / (len(holdout) / periods)),
        "holdout_avg_gross": float(holdout["gross"].mean()),
        "total_by_year": mc.yearly(run["total"]),
    }


def main() -> None:
    all_syms = [
        "SPY",
        "EFA",
        "EEM",
        "QQQ",
        "IEF",
        "TLT",
        "AGG",
        "GLD",
        "DBC",
        "VNQ",
        "SVXY",
        *SECTORS,
    ]
    prices, rets = load_etf_panel(all_syms)
    rf = mc.load_tbill_daily(prices.index)
    fx = mc.load_ecb_usd_crosses()
    rates = mc.load_policy_rates()
    rf_fx = mc.load_tbill_daily(fx.index)
    m_ends, y_ends = month_ends(prices.index), year_ends(prices.index)

    def etf(target: pd.DataFrame) -> pd.DataFrame:
        run = simulate(target, rets, rf)
        return run.loc[target.index[0] :]

    controls = {
        "Control: SPY buy and hold": etf(control_weights(prices, {"SPY": 1.0}, y_ends)),
        "Control: 60/40 SPY/IEF": etf(control_weights(prices, {"SPY": 0.6, "IEF": 0.4}, m_ends)),
        "Control: permanent portfolio": etf(
            control_weights(prices, {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25}, y_ends)
        ),
    }
    bench = controls["Control: 60/40 SPY/IEF"]["excess"]

    builders = {
        "B1 FX carry": lambda k: fx_run(b1_carry(fx, rates), fx, rates, rf_fx, "pairs"),
        "B2 FX trend": lambda k: fx_run(b2_fx_trend(fx, round(252 * k)), fx, rates, rf_fx, "gross"),
        "B3 Multi-asset trend, long/cash": lambda k: etf(b3_gtaa(prices, round(210 * k))),
        "B4 Multi-asset trend, long/short": lambda k: etf(
            b4_trend_ls(prices, rets, rf, round(252 * k))
        ),
        "B5 Dual momentum": lambda k: etf(b5_gem(prices, rf, round(252 * k))),
        "B6 Sector rotation": lambda k: etf(b6_sectors(prices, round(252 * k))),
        "B7 RSI-2 mean reversion": lambda k: b7_rsi2(rf, "open", round(200 * k)),
        "B8 Turn of the month": lambda k: b8_turn_of_month(rets, rf),
        "B9 Volatility risk premium": lambda k: etf(b9_vrp(prices.index)),
    }
    no_lookback = {"B1 FX carry", "B8 Turn of the month", "B9 Volatility risk premium"}

    results, runs = [], {}
    for name, build in builders.items():
        runs[name] = build(1.0)
        results.append(summarise(name, runs[name], bench, gated=True))
    for name, run in controls.items():
        results.append(summarise(name, run, bench, gated=False))

    secondary = summarise(
        "B7 RSI-2, same-close fill (secondary, not gated)", b7_rsi2(rf, "close"), bench, gated=False
    )

    survivors = [
        r["name"] for r in results if r["gated"] and r["verdict"] in ("PASS", "PROVISIONAL")
    ]
    robustness = {}
    for name in survivors:
        if name in no_lookback:
            robustness[name] = "no lookback parameter"
            continue
        robustness[name] = {}
        for k in (0.5, 1.5):
            alt = builders[name](k)
            robustness[name][f"x{k}"] = mc.perf(
                alt["excess"].loc[mc.HOLDOUT[0] : mc.HOLDOUT[1]], ppy(alt.index)
            )["sharpe"]
        if min(robustness[name].values()) <= 0:
            next(r for r in results if r["name"] == name)["verdict"] = "FAIL (robustness sign flip)"
    survivors = [
        r["name"] for r in results if r["gated"] and r["verdict"] in ("PASS", "PROVISIONAL")
    ]

    blend = None
    if len(survivors) >= 2:
        ex = (
            pd.concat([runs[n]["excess"] for n in survivors], axis=1)
            .loc[mc.HOLDOUT[0] : mc.HOLDOUT[1]]
            .fillna(0.0)
        )
        blend_ex = ex.mean(axis=1)
        blend = {
            "members": survivors,
            "holdout": mc.perf(blend_ex, ppy(blend_ex.index)),
            "note": "descriptive only: members chosen on the holdout",
        }

    corr = pd.concat(
        [runs[n]["excess"].rename(n) for n in runs] + [bench.rename("60/40")], axis=1
    ).loc[mc.HOLDOUT[0] : mc.HOLDOUT[1]]
    corr_weekly = ((1.0 + corr.fillna(0.0)).resample("W-FRI").prod() - 1.0).corr().round(2)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "results": results,
                "b7_secondary": secondary,
                "robustness": robustness,
                "blend": blend,
                "weekly_correlation_holdout": corr_weekly.to_dict(),
                "bonferroni_z": mc.BONFERRONI_Z,
                "holdout": mc.HOLDOUT,
            },
            indent=1,
        )
    )
    print(
        f"{'strategy':36s} {'verdict':12s} {'design SR':>9s} {'hold SR':>8s} {'t':>6s} {'excess/yr':>9s} {'CAGR':>7s} {'maxDD':>7s} {'alpha t':>8s} {'2022+':>7s}"
    )
    for r in [*results, secondary]:
        h, d = r["holdout"], r["design"]
        a = r["alpha_vs_6040"] or {"alpha_t": float("nan")}
        print(
            f"{r['name'][:36]:36s} {r['verdict'][:12]:12s} {(d['sharpe'] if d else float('nan')):9.2f} {h['sharpe']:8.2f} {h['tstat']:6.2f} "
            f"{h['ann_excess']:9.2%} {h['cagr']:7.2%} {h['max_dd']:7.1%} {a['alpha_t']:8.2f} {r['recent_mean_ann']:7.2%}"
        )
    print("robustness:", robustness)
    print("blend:", blend)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
