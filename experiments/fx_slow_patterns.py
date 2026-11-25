"""Slow forex patterns (weeks, months, years): a scan on 1987-2014 and one exam on 2015-2026.

Preregistration: docs/research/experiments/2026-11-24_fx-slow-patterns-prereg.md
(pushed before this file existed). The simulator is fx_run from markets_bakeoff.py, unchanged.

    python experiments/fx_slow_patterns.py scan   # every series is cut at 2014-12-31 first
    python experiments/fx_slow_patterns.py exam   # runs only what fx_slow_shortlist.json names

Outputs: experiments/markets_results/fx_slow_{scan,shortlist,exam}.json
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path

import markets_common as mc
import numpy as np
import pandas as pd
from markets_bakeoff import b1_carry, fx_run, month_ends, ppy, week_ends
from scipy.stats import norm

RESULTS = Path(__file__).resolve().parent / "markets_results"
SCAN = ("1987-01-01", "2014-12-31")
SCAN_HALVES = (("1987-01-01", "2000-12-31"), ("2001-01-01", "2014-12-31"))
EXAM = ("2015-01-01", "2026-10-31")
RECENT_FROM = "2022-01-01"
EURO_START = "1999-01-04"
CAD_BOE_END = "2021-06-30"
RATE_CAP = 20.0
SHORTLIST_SHARPE = 0.40
SHORTLIST_MAX = 5
PRIOR_CANDIDATES = 15
EXAM_SHARPE = 0.5
EXAM_DD = 0.25
SEASON_MIN_YEARS = 10
VALUE_MONTHS = 60
CCYS = [c for c in mc.G10 if c != "USD"]

BOE_CODES = {
    "GBP": "XUDLUSS",  # dollars per pound; every other series is currency per dollar
    "EUR": "XUDLERD",
    "DEM": "XUDLDMD",
    "JPY": "XUDLJYD",
    "CHF": "XUDLSFD",
    "AUD": "XUDLADD",
    "NZD": "XUDLNDD",
    "CAD": "XUDLCDD",
    "SEK": "XUDLSKD",
    "NOK": "XUDLNKD",
}
BOE_URL = (
    "https://www.bankofengland.co.uk/boeapps/database/_iadb-fromshowcolumns.asp"
    "?csv.x=yes&Datefrom=01/Jan/1975&Dateto=31/Oct/2026"
    "&SeriesCodes=" + ",".join(BOE_CODES.values()) + "&CSVF=TN&UsingCodes=Y&VPD=Y&VFD=N"
)


# --------------------------------------------------------------------------- data


def _splice(early: pd.Series, late: pd.Series, at: str) -> pd.Series:
    """`late` from `at` on; before it, `early` scaled to meet `late` on the splice day."""
    day = late.loc[at:].dropna().index[0]
    scale = late.loc[day] / early.loc[day]
    return pd.concat([early.loc[: day - pd.Timedelta(days=1)] * scale, late.loc[day:]])


def load_prices() -> pd.DataFrame:
    """Dollars per one unit of each currency, daily from 1975 (Bank of England fixes)."""
    path = mc.DATA_DIR / "boe" / "fx_g10.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(mc._get(BOE_URL).content)
    raw = pd.read_csv(path)
    raw["DATE"] = pd.to_datetime(raw["DATE"], format="%d %b %Y")
    raw = raw.set_index("DATE").sort_index().apply(pd.to_numeric, errors="coerce")
    usd_per = {c: (raw[code] if c == "GBP" else 1.0 / raw[code]) for c, code in BOE_CODES.items()}
    px = pd.DataFrame({c: usd_per[c] for c in CCYS if c not in ("EUR", "CAD")})
    px["EUR"] = _splice(usd_per["DEM"].dropna(), usd_per["EUR"].dropna(), EURO_START)
    ecb_cad = mc.load_ecb_usd_crosses()["CAD"]
    boe_cad = usd_per["CAD"].dropna().loc[:CAD_BOE_END]
    both = boe_cad.index.intersection(ecb_cad.index)
    px["CAD"] = _splice(boe_cad, ecb_cad.reindex(px.index).ffill(), str(both[-1].date()))
    return px[CCYS].dropna(how="any").loc[: mc.END]


def load_rates() -> pd.DataFrame:
    """Policy rates in percent, daily, G10 columns; Germany stands in for the euro before 1999."""
    path = mc.DATA_DIR / "bis_cbpol_long.csv"
    if not path.exists():
        raw = mc._get("https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip").content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            full = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        col = {
            k: next(c for c in full.columns if c.upper().startswith(k))
            for k in ("REF_AREA", "FREQ", "TIME_PERIOD", "OBS_VALUE")
        }
        full["area"] = full[col["REF_AREA"]].astype(str).str.split(":").str[0].str.strip()
        full["freq"] = full[col["FREQ"]].astype(str).str.split(":").str[0].str.strip()
        keep = full[(full["freq"] == "D") & full["area"].isin([*mc.BIS_AREA.values(), "DE"])]
        keep[["area", col["TIME_PERIOD"], col["OBS_VALUE"]]].set_axis(
            ["area", "date", "rate"], axis=1
        ).to_csv(path, index=False)
    tidy = pd.read_csv(path, parse_dates=["date"])
    wide = tidy.pivot_table(
        index="date", columns="area", values="rate", aggfunc="last"
    ).sort_index()
    wide = wide.ffill()
    wide["XM"] = wide["XM"].where(wide.index >= "1999-01-01", wide["DE"])
    wide = wide.rename(columns={v: k for k, v in mc.BIS_AREA.items()})[mc.G10]
    return wide.clip(upper=RATE_CAP).loc["1975-01-01" : mc.END]


def load_reer() -> pd.DataFrame:
    """BIS real effective exchange rate, narrow basket, monthly (period index, G10 columns)."""
    path = mc.DATA_DIR / "bis_reer_narrow.csv"
    if not path.exists():
        raw = mc._get("https://data.bis.org/static/bulk/WS_EER_csv_flat.zip").content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            full = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        col = {
            k: next(c for c in full.columns if c.upper().startswith(k))
            for k in ("REF_AREA", "FREQ", "EER_TYPE", "EER_BASKET", "TIME_PERIOD", "OBS_VALUE")
        }
        first = lambda k: full[col[k]].astype(str).str.split(":").str[0].str.strip()  # noqa: E731
        keep = full[
            (first("FREQ") == "M") & (first("EER_TYPE") == "R") & (first("EER_BASKET") == "N")
        ]
        keep = keep.assign(area=first("REF_AREA"))
        keep = keep[keep["area"].isin(mc.BIS_AREA.values())]
        keep[["area", col["TIME_PERIOD"], col["OBS_VALUE"]]].set_axis(
            ["area", "month", "reer"], axis=1
        ).to_csv(path, index=False)
    tidy = pd.read_csv(path)
    wide = tidy.pivot_table(index="month", columns="area", values="reer", aggfunc="last")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    return wide.sort_index().rename(columns={v: k for k, v in mc.BIS_AREA.items()})[mc.G10]


def load_all(end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Everything cut at `end` before any signal or return is computed."""
    px = load_prices().loc[:end]
    rates = load_rates().loc[:end]
    reer = load_reer().loc[: pd.Period(end, freq="M")]
    days = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1).clip(lower=1)
    usd = rates["USD"].reindex(rates.index.union(px.index)).ffill().reindex(px.index).shift(1)
    rf = (usd / 100.0 * days / 365.0).fillna(0.0)
    return px, rates, rf, reer


# --------------------------------------------------------------------------- signals -> weights


def top_bottom(scores: pd.DataFrame) -> pd.DataFrame:
    """+1/3 in the three highest scores, -1/3 in the three lowest. Columns must be G10 (USD
    included); ties keep the fixed G10 order. Rows with any missing score are dropped."""
    rows = {}
    for t, row in scores[mc.G10].dropna(how="any").iterrows():
        ranked = row.sort_values(ascending=False, kind="stable")
        w = dict.fromkeys(mc.G10, 0.0)
        for c in ranked.index[:3]:
            w[c] = 1.0 / 3.0
        for c in ranked.index[-3:]:
            w[c] = -1.0 / 3.0
        rows[t] = {c: w[c] for c in CCYS}
    return pd.DataFrame.from_dict(rows, orient="index")


def with_usd(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["USD"] = 0.0
    out.loc[df.isna().any(axis=1), "USD"] = np.nan
    return out[mc.G10]


def trailing(px: pd.DataFrame, ends: pd.DatetimeIndex, periods: int) -> pd.DataFrame:
    p = px.loc[ends]
    return p / p.shift(periods) - 1.0


def seasonal_scores(px: pd.DataFrame) -> pd.DataFrame:
    """At each month end: each currency's average return in the coming calendar month over all
    earlier years (at least SEASON_MIN_YEARS of them)."""
    me = month_ends(px.index)
    monthly = px.loc[me].pct_change()
    out = pd.DataFrame(index=me, columns=CCYS, dtype=float)
    for i, t in enumerate(me):
        coming = t.month % 12 + 1
        past = monthly.iloc[: i + 1]
        past = past[past.index.month == coming].dropna(how="any")
        if len(past) >= SEASON_MIN_YEARS:
            out.loc[t] = past.mean()
    return out


def value_scores(px: pd.DataFrame, reer: pd.DataFrame) -> pd.DataFrame:
    """At each month end: minus (log real rate last month less its mean over the 60 before)."""
    log = np.log(reer)
    dev = log - log.shift(1).rolling(VALUE_MONTHS).mean()
    me = month_ends(px.index)
    lagged = dev.shift(1)  # the value known at the end of month m is month m-1's
    rows = lagged.reindex(pd.PeriodIndex(me, freq="M"))
    rows.index = me
    return -rows[mc.G10]


def build_weights(px: pd.DataFrame, rates: pd.DataFrame, reer: pd.DataFrame) -> dict:
    """Rule id -> (weights on signal days, markup basis). One sign each; the scan flips them."""
    me, we = month_ends(px.index), week_ends(px.index)
    out: dict[str, tuple[pd.DataFrame, str]] = {}
    for k in (1, 4):
        out[f"W{k}"] = (top_bottom(with_usd(trailing(px, we, k))), "pairs")
    for k in (1, 3, 6, 12):
        r = trailing(px, me, k).dropna(how="any")
        out[f"T{k}"] = (np.sign(r) / len(CCYS), "gross")
        out[f"X{k}"] = (top_bottom(with_usd(trailing(px, me, k))), "pairs")
    season = seasonal_scores(px)
    out["S1"] = (top_bottom(with_usd(season)), "pairs")
    basket = season.dropna(how="any").mean(axis=1)
    s2 = pd.DataFrame({c: np.sign(basket) / len(CCYS) for c in CCYS})
    out["S2"] = (s2, "gross")
    out["V1"] = (top_bottom(value_scores(px, reer)), "pairs")
    out["V2"] = (top_bottom(with_usd(-trailing(px, me, VALUE_MONTHS))), "pairs")
    carry = b1_carry(px, rates)
    parts = [carry, out["X3"][0], out["V1"][0]]
    common = parts[0].index.intersection(parts[1].index).intersection(parts[2].index)
    out["K1"] = (sum(p.loc[common, CCYS] for p in parts) / 3.0, "pairs")
    return out


NAMES = {
    "W1": "last week's winners against losers",
    "W4": "last four weeks' winners against losers",
    "T1": "each pair, direction of last month",
    "T3": "each pair, direction of last 3 months",
    "T6": "each pair, direction of last 6 months",
    "T12": "each pair, direction of last 12 months",
    "X1": "last month's winners against losers",
    "X3": "last 3 months' winners against losers",
    "X6": "last 6 months' winners against losers",
    "X12": "last 12 months' winners against losers",
    "S1": "best against worst currencies for the coming calendar month",
    "S2": "all currencies against the dollar by calendar month",
    "V1": "cheap against dear on real exchange rates",
    "V2": "five-year losers against five-year winners",
    "K1": "carry + 3-month winners + cheapness, one third each",
}
ONE_SIGN = {"K1"}


# --------------------------------------------------------------------------- running


def gross_returns(w: pd.DataFrame, px: pd.DataFrame, rates: pd.DataFrame) -> pd.Series:
    """Same timing as fx_run, with no spread, no markup and no idle margin."""
    held = w.reindex(px.index).ffill().shift(2).fillna(0.0)[CCYS]
    days = pd.Series(px.index, index=px.index).diff().dt.days.fillna(1).clip(lower=1)
    r = rates.reindex(rates.index.union(px.index)).ffill().reindex(px.index).shift(1)
    carry = (r[CCYS].sub(r["USD"], axis=0) / 100.0).mul(days / 365.0, axis=0)
    return (held * (px.pct_change() + carry)).sum(axis=1)


def stats(x: pd.Series, window: tuple[str, str]) -> dict:
    w = x.loc[window[0] : window[1]]
    p = mc.perf(w, ppy(w.index))
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in p.items()}


def by_year(x: pd.Series) -> dict:
    return {
        str(int(y)): round(float((1.0 + g).prod() - 1.0), 4) for y, g in x.groupby(x.index.year)
    }


def run_candidate(w, basis, sign, px, rates, rf) -> tuple[pd.DataFrame, pd.Series]:
    ws = w * sign
    return fx_run(ws, px, rates, rf, basis), gross_returns(ws, px, rates)


def descriptive(px: pd.DataFrame, window: tuple[str, str]) -> dict:
    p = px.loc[window[0] : window[1]]
    out: dict = {"autocorrelation": {}, "calendar_month": {}, "weekday": {}}
    we, me = week_ends(p.index), month_ends(p.index)
    series = {
        "1w": p.loc[we].pct_change(),
        "1m": p.loc[me].pct_change(),
        "3m": p.loc[me[::3]].pct_change(),
        "12m": p.loc[me[::12]].pct_change(),
    }
    for h, r in series.items():
        r = r.dropna()
        ac = {c: float(r[c].autocorr(1)) for c in CCYS}
        out["autocorrelation"][h] = {
            "n": int(len(r)),
            "mean": round(float(np.mean(list(ac.values()))), 4),
            "two_se": round(2.0 / math.sqrt(len(r)), 4),
            "by_pair": {c: round(v, 4) for c, v in ac.items()},
        }
    monthly = series["1m"].dropna()
    cells, beyond = {}, 0
    for c in CCYS:
        for m in range(1, 13):
            x = monthly.loc[monthly.index.month == m, c]
            t = float(x.mean() / (x.std() / math.sqrt(len(x))))
            cells[f"{c}-{m:02d}"] = {"mean": round(float(x.mean()), 5), "t": round(t, 2)}
            beyond += abs(t) >= 2.0
    out["calendar_month"] = {
        "cells": cells,
        "beyond_2": int(beyond),
        "expected_by_chance": round(108 * 0.0455, 1),
    }
    daily = p.pct_change().dropna()
    cells, beyond = {}, 0
    for c in CCYS:
        for d in range(5):
            x = daily.loc[daily.index.dayofweek == d, c]
            t = float(x.mean() / (x.std() / math.sqrt(len(x))))
            cells[f"{c}-{d}"] = {"mean_bp": round(float(x.mean()) * 1e4, 2), "t": round(t, 2)}
            beyond += abs(t) >= 2.0
    out["weekday"] = {
        "cells": cells,
        "beyond_2": int(beyond),
        "expected_by_chance": round(45 * 0.0455, 1),
    }
    return out


def scan() -> None:
    px, rates, rf, reer = load_all(SCAN[1])
    assert px.index[-1] <= pd.Timestamp(SCAN[1]) and reer.index[-1] <= pd.Period(SCAN[1], freq="M")
    weights = build_weights(px, rates, reer)
    table = {}
    for rule, (w, basis) in weights.items():
        for sign, tag in ((1.0, "+"), (-1.0, "-")):
            if rule in ONE_SIGN and sign < 0:
                continue
            run, gross = run_candidate(w, basis, sign, px, rates, rf)
            full = stats(run["excess"], SCAN)
            halves = [stats(run["excess"], h) for h in SCAN_HALVES]
            table[f"{rule}{tag}"] = {
                "rule": rule,
                "sign": tag,
                "name": NAMES[rule] + (" (reversed)" if sign < 0 else ""),
                "sharpe": full["sharpe"],
                "ann_excess": full["ann_excess"],
                "vol": full["vol"],
                "max_dd": full["max_dd"],
                "tstat": full["tstat"],
                "first_half": {k: halves[0][k] for k in ("sharpe", "ann_excess")},
                "second_half": {k: halves[1][k] for k in ("sharpe", "ann_excess")},
                "sharpe_before_costs": stats(gross, SCAN)["sharpe"],
                "turnover_per_year": round(
                    float(run["turnover"].loc[SCAN[0] :].sum() / full["years"]), 2
                ),
                "starts": str(run.index[0].date()),
            }
    qualifies = [
        k
        for k, v in table.items()
        if v["sharpe"] >= SHORTLIST_SHARPE
        and v["first_half"]["ann_excess"] > 0
        and v["second_half"]["ann_excess"] > 0
    ]
    qualifies.sort(key=lambda k: -table[k]["sharpe"])
    shortlist, seen = [], set()
    for k in qualifies:
        if table[k]["rule"] in seen:
            continue
        shortlist.append(k)
        seen.add(table[k]["rule"])
        if len(shortlist) == SHORTLIST_MAX:
            break
    result = {
        "prereg": "docs/research/experiments/2026-11-24_fx-slow-patterns-prereg.md",
        "window": list(SCAN),
        "data_cut_at": str(px.index[-1].date()),
        "price_rows": int(len(px.loc[SCAN[0] :])),
        "candidates": table,
        "qualifies": qualifies,
        "shortlist": shortlist,
        "descriptive": descriptive(px, SCAN),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fx_slow_scan.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    (RESULTS / "fx_slow_shortlist.json").write_text(
        json.dumps({"shortlist": shortlist}, indent=1) + "\n"
    )
    print(
        f"{'id':5}{'Sharpe':>8}{'gross':>8}{'%/yr':>8}{'H1%':>7}{'H2%':>7}{'DD':>7}{'turn':>7}  name"
    )
    for k, v in sorted(table.items(), key=lambda kv: -kv[1]["sharpe"]):
        print(
            f"{k:5}{v['sharpe']:8.2f}{v['sharpe_before_costs']:8.2f}{v['ann_excess']:8.1%}"
            f"{v['first_half']['ann_excess']:7.1%}{v['second_half']['ann_excess']:7.1%}"
            f"{v['max_dd']:7.1%}{v['turnover_per_year']:7.1f}  {v['name']}"
        )
    print("qualifies", qualifies, "shortlist", shortlist)
    d = result["descriptive"]
    print("autocorr", {h: (v["mean"], v["two_se"]) for h, v in d["autocorrelation"].items()})
    print(
        "calendar beyond 2:",
        d["calendar_month"]["beyond_2"],
        "of 108; weekday:",
        d["weekday"]["beyond_2"],
        "of 45",
    )


def exam() -> None:
    shortlist = json.loads((RESULTS / "fx_slow_shortlist.json").read_text())["shortlist"]
    if not shortlist:
        print("no shortlist: exam not run")
        return
    n = PRIOR_CANDIDATES + len(shortlist)
    bar = float(norm.ppf(1.0 - 0.05 / n))
    px, rates, rf, reer = load_all(EXAM[1])
    weights = build_weights(px, rates, reer)
    scan_px = px.loc[: SCAN[1]]
    out = {}
    for key in shortlist:
        rule, sign = key[:-1], (1.0 if key.endswith("+") else -1.0)
        w, basis = weights[rule]
        run, gross = run_candidate(w, basis, sign, px, rates, rf)
        ex = run["excess"]
        s, recent = stats(ex, EXAM), ex.loc[RECENT_FROM : EXAM[1]]
        ok = s["sharpe"] >= EXAM_SHARPE and s["max_dd"] <= EXAM_DD and recent.mean() > 0
        total = run["total"].loc[EXAM[0] : EXAM[1]]
        years = s["years"]
        out[key] = {
            "name": NAMES[rule] + (" (reversed)" if sign < 0 else ""),
            **s,
            "total_per_year": float((1.0 + total).prod() ** (1.0 / years) - 1.0),
            "excess_per_year_at_10pct_vol": float(s["sharpe"] * 0.10),
            "sharpe_before_costs": stats(gross, EXAM)["sharpe"],
            "recent_mean_ann": float(recent.mean() * ppy(recent.index)),
            "scan_sharpe": stats(ex, SCAN)["sharpe"],
            "turnover_per_year": round(
                float(run["turnover"].loc[EXAM[0] : EXAM[1]].sum() / years), 2
            ),
            "by_year": by_year(total),
            "verdict": "PASS" if ok and s["tstat"] >= bar else ("PROVISIONAL" if ok else "FAIL"),
        }
    cash = rf.loc[EXAM[0] : EXAM[1]]
    result = {
        "prereg": "docs/research/experiments/2026-11-24_fx-slow-patterns-prereg.md",
        "window": list(EXAM),
        "candidates_on_window": n,
        "t_bar": bar,
        "dollar_cash_per_year": float(
            (1.0 + cash).prod() ** (1.0 / (len(cash) / ppy(cash.index))) - 1.0
        ),
        "results": out,
        "descriptive_exam_window": descriptive(px, EXAM),
        "scan_rows_unchanged": bool(scan_px.equals(load_all(SCAN[1])[0])),
    }
    blob = json.dumps(result, indent=1, sort_keys=True)
    result["hash"] = hashlib.sha256(blob.encode()).hexdigest()
    (RESULTS / "fx_slow_exam.json").write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(f"candidates on window {n}, t bar {bar:.2f}")
    for k, v in out.items():
        print(
            f"{k:5} scan {v['scan_sharpe']:5.2f} | exam Sharpe {v['sharpe']:5.2f} (gross {v['sharpe_before_costs']:5.2f}) "
            f"t {v['tstat']:5.2f} excess {v['ann_excess']:6.1%} total {v['total_per_year']:6.1%} DD {v['max_dd']:5.1%} "
            f"2022+ {v['recent_mean_ann']:6.1%}  {v['verdict']}  {v['name']}"
        )
        print("      ", " ".join(f"{y[2:]}:{r * 100:+.0f}" for y, r in v["by_year"].items()))
    d = result["descriptive_exam_window"]
    print("autocorr", {h: (x["mean"], x["two_se"]) for h, x in d["autocorrelation"].items()})
    print(
        "calendar beyond 2:",
        d["calendar_month"]["beyond_2"],
        "of 108; weekday:",
        d["weekday"]["beyond_2"],
        "of 45",
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "scan":
        scan()
    elif mode == "exam":
        exam()
    else:
        sys.exit("usage: fx_slow_patterns.py scan|exam")
