"""What seven plain portfolios returned in pounds, 1997-2026.

Preregistration: docs/research/experiments/2026-11-24_gbp-portfolios-10pct-prereg.md
(pushed before this file existed). A measurement of known sources of return, not a
search: portfolios, costs, window and the 10% bar are fixed there.

Run: python experiments/gbp_portfolios.py
Output: experiments/markets_results/gbp_portfolios.json
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from markets_common import DATA_DIR, _get, load_yahoo

OUT = Path(__file__).resolve().parent / "markets_results" / "gbp_portfolios.json"

COMMON = ("1997-01-01", "2026-10-31")
HALVES = (("1997-01-01", "2011-12-31"), ("2012-01-01", "2026-10-31"))
LONG = ("1991-01-01", "2026-10-31")

SP_COST = 0.0030  # per year: 0.07% fund charge + 15% withholding on dividends
SWITCH_COST = 0.0010  # of the amount moved
CASH_HAIRCUT = 0.0010  # pound cash earns Bank Rate minus this
LEV = 2.0
LEV_BORROW_SPREAD = 0.0050  # per year over the dollar cash rate, on the borrowed half
LEV_FEE = 0.0060  # per year
TREND_MONTHS = 10
TARGET = 0.10
DD_BAR = 0.40
POT = 5000.0

BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom=01/Jan/1975&Dateto=31/Oct/2026"
    "&SeriesCodes=XUDLUSS,IUDBEDR&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)
SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"


# --------------------------------------------------------------------------- data


def load_boe() -> pd.DataFrame:
    """Dollars per pound (XUDLUSS) and Bank Rate in percent (IUDBEDR), daily."""
    path = DATA_DIR / "boe" / "usd_bankrate.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(BOE_URL).content)
    df = pd.read_csv(path)
    df["DATE"] = pd.to_datetime(df["DATE"], format="%d %b %Y")
    return df.set_index("DATE").sort_index().apply(pd.to_numeric, errors="coerce")


def load_shiller() -> pd.DataFrame:
    """Monthly US stock price and annualised dividend, 1871 on (dollars)."""
    path = DATA_DIR / "shiller" / "ie_data.xls"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get(SHILLER_URL).content)
    raw = pd.read_excel(io.BytesIO(path.read_bytes()), sheet_name="Data", header=7)
    df = raw.iloc[:, :3].copy()
    df.columns = ["date", "price", "dividend"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    year = df["date"].astype(int)
    month = ((df["date"] - year) * 100).round().astype(int).clip(1, 12)
    df.index = pd.to_datetime({"year": year, "month": month, "day": 1})
    return df[["price", "dividend"]]


def accrual(rate: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Interest over each interval ending at index[i], at the previous day's annual rate."""
    days = pd.Series(index, index=index).diff().dt.days.fillna(1).clip(lower=1)
    prev = rate.reindex(index.union(rate.index)).ffill().reindex(index).shift(1).bfill()
    return prev * days / 365.0


def build_inputs() -> pd.DataFrame:
    """Daily returns on the S&P 500 trading calendar. Dollar assets converted to pounds."""
    sp = load_yahoo("^SP500TR")["adjclose"]
    idx = sp.index
    days = pd.Series(idx, index=idx).diff().dt.days.fillna(1).clip(lower=1)
    year_frac = days / 365.0

    def ret(symbol: str) -> pd.Series:
        return load_yahoo(symbol)["adjclose"].reindex(idx).pct_change()

    boe = load_boe()
    gbp_per_usd = (1.0 / boe["XUDLUSS"]).reindex(idx.union(boe.index)).ffill().reindex(idx)
    fx = gbp_per_usd / gbp_per_usd.shift(1)
    usd_cash = accrual(load_yahoo("^IRX")["close"] / 100.0, idx)
    uk_cash = accrual(boe["IUDBEDR"] / 100.0, idx)

    sp_usd = sp.pct_change()
    intl_usd = ret("VGTSX")
    bond_usd = ret("VFITX")
    long_usd = ret("VUSTX")

    def lev(underlying: pd.Series) -> pd.Series:
        borrow = (LEV - 1.0) * (usd_cash + LEV_BORROW_SPREAD * year_frac)
        return LEV * underlying - borrow - LEV_FEE * year_frac

    def to_gbp(usd_ret: pd.Series) -> pd.Series:
        return (1.0 + usd_ret) * fx - 1.0

    out = pd.DataFrame(
        {
            "sp_index": sp,
            "sp": to_gbp(sp_usd - SP_COST * year_frac),
            "intl": to_gbp(intl_usd),
            "bond_hedged": bond_usd + uk_cash - usd_cash,
            "cash": uk_cash - CASH_HAIRCUT * year_frac,
            "sp2x": to_gbp(lev(sp_usd)),
            "mix2x": to_gbp(lev(0.6 * sp_usd + 0.4 * long_usd)),
            "sp2x_usd": lev(sp_usd),
            "sp_usd": sp_usd,
            "fx": fx - 1.0,
        }
    )
    return out


# --------------------------------------------------------------------------- portfolios


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().values)


def rebalanced(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Hold sleeves, let them drift, reset to target at each month-end close (cost on turnover)."""
    cols = list(weights)
    target = np.array([weights[c] for c in cols])
    r = returns[cols].to_numpy()
    ends = set(month_ends(returns.index))
    w = target.copy()
    out = np.empty(len(r))
    for i, day in enumerate(returns.index):
        grown = w * (1.0 + r[i])
        total = grown.sum()
        port = total - 1.0
        w = grown / total
        if day in ends:
            port -= SWITCH_COST * np.abs(target - w).sum()
            w = target.copy()
        out[i] = port
    return pd.Series(out, index=returns.index)


def trend_signal(sp_index: pd.Series) -> pd.Series:
    """True for a day whose return is earned in the risky asset.

    Month-end value above the mean of the last ten month-end values; the switch is made at the
    close of the first trading day of the next month, so it applies from the day after that.
    """
    idx = sp_index.index
    me = sp_index.loc[month_ends(idx)]
    on = me > me.rolling(TREND_MONTHS).mean()
    first_days = pd.Series(idx, index=idx).groupby([idx.year, idx.month]).first().values
    # signal from month m is acted on at the close of first trading day of month m+1
    acted = pd.Series(on.to_numpy()[:-1], index=pd.DatetimeIndex(first_days[1:]))
    held_at_close = acted.reindex(idx).ffill()
    return held_at_close.shift(1)  # position during day d = what was held at the close of d-1


def filtered(risky: pd.Series, cash: pd.Series, signal: pd.Series) -> pd.Series:
    sig = signal.fillna(False).astype(bool)
    r = risky.where(sig, cash)
    switch = sig.ne(sig.shift(1)).fillna(False)
    switch.iloc[0] = False
    # the cost is charged on the day of the trade, which is the day before the new position earns
    cost = switch.shift(-1).fillna(False).astype(float) * SWITCH_COST
    return r - cost


def build_portfolios(x: pd.DataFrame) -> pd.DataFrame:
    sig = trend_signal(x["sp_index"])
    x = x.copy()
    return pd.DataFrame(
        {
            "P1": x["sp"],
            "P2": rebalanced(x.fillna(0.0), {"sp": 0.6, "intl": 0.4}),
            "P3": rebalanced(x.fillna(0.0), {"sp": 0.36, "intl": 0.24, "bond_hedged": 0.40}),
            "P4": filtered(x["sp"], x["cash"], sig),
            "P5": x["sp2x"],
            "P6": filtered(x["sp2x"], x["cash"], sig),
            "P7": x["mix2x"],
            "cash": x["cash"],
            "trend_on": sig.fillna(False).astype(float),
        }
    )


NAMES = {
    "P1": "US stocks",
    "P2": "World stocks",
    "P3": "60/40",
    "P4": "US stocks with a trend filter",
    "P5": "US stocks at 2x",
    "P6": "US stocks at 2x with the trend filter",
    "P7": "Stocks and bonds at 2x",
    "cash": "Pound cash",
}


# --------------------------------------------------------------------------- metrics


def cagr(r: pd.Series, start: str, end: str) -> float:
    w = r.loc[start:end]
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    return float((1.0 + w).prod() ** (1.0 / years) - 1.0)


def drawdown(r: pd.Series) -> dict:
    eq = (1.0 + r).cumprod()
    peak = eq.cummax()
    dd = 1.0 - eq / peak
    trough = dd.idxmax()
    under = dd > 0
    # longest run of days below a previous peak, in years
    run_id = (~under).cumsum()
    longest = 0.0
    start_longest = end_longest = None
    for _, g in under[under].groupby(run_id[under]):
        span = (g.index[-1] - g.index[0]).days / 365.25
        if span > longest:
            longest, start_longest, end_longest = span, g.index[0], g.index[-1]
    return {
        "max_dd": float(dd.max()),
        "trough": str(trough.date()),
        "peak_before": str(eq.loc[:trough].idxmax().date()),
        "longest_underwater_years": round(longest, 2),
        "longest_underwater_from": str(start_longest.date()) if start_longest is not None else None,
        "longest_underwater_to": str(end_longest.date()) if end_longest is not None else None,
        "still_underwater_at_end": bool(under.iloc[-1]),
    }


def rolling(r: pd.Series, years: int) -> dict:
    eq = (1.0 + r).cumprod()
    me = eq.loc[month_ends(eq.index)]
    n = years * 12
    c = (me / me.shift(n)) ** (1.0 / years) - 1.0
    c = c.dropna()
    if c.empty:
        return {}
    return {
        "windows": int(len(c)),
        "share_at_or_above_target": float((c >= TARGET).mean()),
        "share_negative": float((c < 0).mean()),
        "worst": float(c.min()),
        "worst_ending": str(c.idxmin().date()),
        "median": float(c.median()),
        "best": float(c.max()),
    }


def by_year(r: pd.Series) -> dict:
    return {
        str(int(y)): round(float((1.0 + g).prod() - 1.0), 4) for y, g in r.groupby(r.index.year)
    }


def describe(r: pd.Series, window: tuple[str, str], halves: bool) -> dict:
    w = r.loc[window[0] : window[1]]
    years = by_year(w)
    full = cagr(r, *window)
    out = {
        "window": list(window),
        "cagr": full,
        "final_value_of_pot": float(POT * (1.0 + w).prod()),
        **drawdown(w),
        "vol": float(w.std() * np.sqrt(252)),
        "worst_year": min(years.items(), key=lambda kv: kv[1]),
        "best_year": max(years.items(), key=lambda kv: kv[1]),
        "years_negative": int(sum(v < 0 for v in years.values())),
        "years_at_or_above_target": int(sum(v >= TARGET for v in years.values())),
        "years_total": len(years),
        "rolling_5y": rolling(w, 5),
        "rolling_10y": rolling(w, 10),
        "by_year": years,
    }
    if halves:
        h = [cagr(r, *hv) for hv in HALVES]
        out["cagr_first_half"], out["cagr_second_half"] = h
        returns_ok = full >= TARGET and min(h) >= TARGET
        if returns_ok and out["max_dd"] <= DD_BAR:
            out["verdict"] = "MEETS"
        elif returns_ok:
            out["verdict"] = "MEETS WITH DEEP LOSSES"
        else:
            out["verdict"] = "SHORT"
    return out


def leverage_check(x: pd.DataFrame) -> dict:
    """The 2x model against a real 2x fund (SSO), in dollars, over the fund's life."""
    sso = load_yahoo("SSO")["adjclose"].reindex(x.index).pct_change().dropna()
    start, end = str(sso.index[0].date()), COMMON[1]
    model = x["sp2x_usd"].reindex(sso.index)
    return {
        "from": start,
        "fund_cagr": cagr(sso, start, end),
        "model_cagr": cagr(model, start, end),
        "sp500_cagr": cagr(x["sp_usd"].reindex(sso.index), start, end),
        "fund_max_dd": drawdown(sso.loc[:end])["max_dd"],
        "model_max_dd": drawdown(model.loc[:end])["max_dd"],
        "daily_correlation": float(sso.loc[:end].corr(model.loc[:end])),
    }


def shiller_look() -> dict:
    """US stocks in dollars since 1871, monthly, dividends reinvested, no costs."""
    s = load_shiller()
    r = ((s["price"] + s["dividend"] / 12.0) / s["price"].shift(1) - 1.0).dropna()
    eq = (1.0 + r).cumprod()
    out = {
        "from": str(r.index[0].date()),
        "to": str(r.index[-1].date()),
        "cagr": float(eq.iloc[-1] ** (12.0 / len(r)) - 1.0),
    }
    for years in (5, 10, 20):
        c = ((eq / eq.shift(years * 12)) ** (1.0 / years) - 1.0).dropna()
        out[f"rolling_{years}y"] = {
            "windows": int(len(c)),
            "share_at_or_above_target": float((c >= TARGET).mean()),
            "share_negative": float((c < 0).mean()),
            "worst": float(c.min()),
            "median": float(c.median()),
        }
    return out


def main() -> None:
    x = build_inputs()
    p = build_portfolios(x)
    ids = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "cash"]
    result = {
        "prereg": "docs/research/experiments/2026-11-24_gbp-portfolios-10pct-prereg.md",
        "target": TARGET,
        "dd_bar": DD_BAR,
        "names": NAMES,
        "common": {k: describe(p[k], COMMON, halves=True) for k in ids},
        "long_us_only": {
            k: describe(p[k], LONG, halves=False) for k in ["P1", "P4", "P5", "P6", "P7", "cash"]
        },
        "trend_filter": {
            "share_of_days_in_stocks": float(p["trend_on"].loc[COMMON[0] : COMMON[1]].mean()),
            "switches": int(p["trend_on"].loc[COMMON[0] : COMMON[1]].diff().abs().sum()),
        },
        "pound": {
            "gbp_per_usd_start": float(1.0 / load_boe()["XUDLUSS"].loc[: COMMON[0]].iloc[-1]),
            "gbp_per_usd_end": float(1.0 / load_boe()["XUDLUSS"].loc[: COMMON[1]].iloc[-1]),
            "contribution_to_p1_cagr": cagr(x["fx"], *COMMON),
        },
        "leverage_check": leverage_check(x),
        "shiller": shiller_look(),
    }
    daily = p.loc[COMMON[0] : COMMON[1], ids].round(10)
    result["daily_return_hash"] = hashlib.sha256(daily.to_csv().encode()).hexdigest()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")

    print(
        f"{'':4}{'portfolio':40}{'CAGR':>7}{'H1':>7}{'H2':>7}{'maxDD':>7}{'worstY':>8}{'5y>=10%':>9}{'worst5y':>9}{'£5k->':>10}  verdict"
    )
    for k in ids:
        d = result["common"][k]
        print(
            f"{k:4}{NAMES[k]:40}{d['cagr']:7.1%}{d['cagr_first_half']:7.1%}{d['cagr_second_half']:7.1%}"
            f"{d['max_dd']:7.1%}{d['worst_year'][1]:8.1%}{d['rolling_5y']['share_at_or_above_target']:9.0%}"
            f"{d['rolling_5y']['worst']:9.1%}{d['final_value_of_pot']:10,.0f}  {d['verdict']}"
        )
    print("leverage check", json.dumps(result["leverage_check"]))
    print("shiller", json.dumps(result["shiller"]))
    print("hash", result["daily_return_hash"])


if __name__ == "__main__":
    main()
