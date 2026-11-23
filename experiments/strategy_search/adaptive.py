"""Slow, adaptive, many-coin system: the books and the Stage A design run
(docs/research/experiments/2026-11-11_slow-adaptive-trend-prereg.md).

  python experiments/strategy_search/adaptive.py design
      Data truncated at 2024-12-31. Runs the baseline B0 and the four pre-specified changes
      (V1 selective, V2 broad, V3 regime switch, V4 ML ranker), applies the selection rule and
      writes results/adaptive_design.json.

The functions here are also what the forward recorder (tools/paper_trend_recorder.py) calls, so the
weights it records are produced by the same code that was judged in Stage A.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402
import engine as eng  # noqa: E402

import strategies as strat  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
FIT = ("2020-07-01", "2022-12-31")
VAL = ("2023-01-01", "2024-12-31")
DESIGN_CUTOFF = "2024-12-31"

LOOKBACKS = (30, 60, 120)
BROAD_N = 50
ER_WINDOW = 60  # days in the efficiency ratio
ER_MIN_HISTORY = 180  # past values needed before the regime switch can be on

ML_HORIZON = 7  # days held after the fill
ML_LABEL_LAG = 1 + ML_HORIZON  # a Monday's outcome is known this many days later
ML_MIN_TRAIN_WEEKS = 52
ML_REFIT_WEEKS = 13
ML_MIN_COINS = 6  # a week needs this many labelled coins to count (same floor as the quintile legs)
ML_PARAMS = {"max_depth": 3, "max_iter": 200, "learning_rate": 0.05, "min_samples_leaf": 50,
             "random_state": 0, "early_stopping": False}

EDGE = 0.15  # V1-V3 must beat B0's Sharpe by this much in both windows
TWEAK_MAX_DD = 0.20
ML_MIN_SHARPE = 0.7
ML_MAX_DD = 0.25


@dataclass
class Book:
    result: eng.Result
    weights: pd.DataFrame  # final target weights on signal days (after volatility scaling and any switch)
    info: dict


# ------------------------------------------------------------------------ trend books


def trend_signal(m: data_mod.Market) -> pd.DataFrame:
    """Average of the signs of the 30-, 60- and 120-day returns: one of -1, -1/3, 1/3, 1."""
    return sum(np.sign(m.close / m.close.shift(L) - 1.0) for L in LOOKBACKS) / len(LOOKBACKS)


def selective_targets(m: data_mod.Market, univ: pd.DataFrame) -> pd.DataFrame:
    """V1: a coin is held only when all three lookbacks agree."""
    sig = trend_signal(m)
    sig = sig.where(sig.abs() > 0.99, 0.0)
    raw = (sig / strat._vol30(m)).where(univ)
    return eng.on_days(strat._normalise(raw.fillna(0.0)), eng.weekly_days(m.close.index))


def trendiness(m: data_mod.Market, univ: pd.DataFrame) -> pd.Series:
    """Efficiency ratio of the equal-weight universe index over 60 days: |sum of daily log returns|
    divided by the sum of their absolute values. 1 = a straight line, 0 = all noise."""
    r = m.close.pct_change(fill_method=None)
    index_r = r.where(univ.shift(1, fill_value=False)).mean(axis=1)  # membership known the day before
    lr = np.log1p(index_r)
    return lr.rolling(ER_WINDOW, min_periods=ER_WINDOW).sum().abs() / lr.abs().rolling(ER_WINDOW, min_periods=ER_WINDOW).sum()


def regime_on(er: pd.Series) -> pd.Series:
    """True when today's trendiness is above the median of all its PAST values (today excluded)."""
    past_median = er.expanding(min_periods=ER_MIN_HISTORY).median().shift(1)
    return er > past_median  # False while either side is unknown


def run_book(m: data_mod.Market, rank: pd.DataFrame, targets: pd.DataFrame, gate: pd.Series | None = None,
             lag: int = 1) -> Book:
    """Phase 1b's volatility scaling, then (V3 only) the on/off switch. The scale comes from the
    un-switched book, so time spent flat cannot inflate leverage."""
    res = eng.vol_targeted(m.close, m.funding, rank, targets, lag=lag)
    weights = targets.mul(res.extra["leverage"], axis=0)
    if gate is None:
        return Book(res, weights, {})
    weights = weights.mul(gate.astype(float), axis=0)
    gated = eng.simulate(m.close, m.funding, rank, weights, lag)
    gated.extra.update(res.extra)
    return Book(gated, weights, {})


# ------------------------------------------------------------------------ ML ranker (V4)

FEATURES = ("ret_7", "ret_14", "ret_30", "ret_60", "ret_120", "vol_30", "funding_7", "funding_30",
            "volume_7_over_60", "below_high_120", "liquidity_rank")


def ml_features(m: data_mod.Market, rank: pd.DataFrame, univ: pd.DataFrame, peek_days: int = 0) -> dict[str, pd.DataFrame]:
    """Each feature as a percentile within the day's universe. Every input is known at that day's close.

    `peek_days` > 0 exists only for the leakage control in selfcheck.py: it hands the model features
    dated that many days in the future, which must look brilliant if the test has power."""
    c = m.close
    raw = {f"ret_{L}": c / c.shift(L) - 1.0 for L in (7, 14, 30, 60, 120)}
    raw["vol_30"] = strat._vol30(m)
    raw["funding_7"] = m.funding.rolling(7, min_periods=7).mean()
    raw["funding_30"] = m.funding.rolling(30, min_periods=30).mean()
    raw["volume_7_over_60"] = m.qvol.rolling(7, min_periods=7).median() / m.qvol.rolling(60, min_periods=60).median()
    raw["below_high_120"] = c / c.rolling(120, min_periods=120).max() - 1.0
    raw["liquidity_rank"] = rank
    out = {}
    for name in FEATURES:
        x = raw[name].shift(-peek_days) if peek_days else raw[name]
        out[name] = x.where(univ).rank(axis=1, pct=True)
    return out


def ml_labels(m: data_mod.Market, univ: pd.DataFrame) -> pd.DataFrame:
    """Return from the fill (next day's close) to seven days later, minus the universe average.
    A coin that stops trading inside the week is settled at its last price, as in the simulator."""
    last = m.close.ffill()
    fwd = (last.shift(-ML_LABEL_LAG) / m.close.shift(-1) - 1.0).where(univ)
    return fwd.sub(fwd.mean(axis=1), axis=0)


def ml_scores(m: data_mod.Market, rank: pd.DataFrame, univ: pd.DataFrame, peek_days: int = 0) -> tuple[pd.DataFrame, dict]:
    """Predicted relative return per coin on each Monday, from an expanding-window model refit every 13 weeks."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    feats = ml_features(m, rank, univ, peek_days)
    label = ml_labels(m, univ)
    mondays = eng.weekly_days(m.close.index)
    x_all = pd.concat({k: v.loc[mondays].stack(future_stack=True) for k, v in feats.items()}, axis=1)[list(FEATURES)]
    in_univ = univ.loc[mondays].stack(future_stack=True)
    x_all = x_all[in_univ.reindex(x_all.index).fillna(False).to_numpy()]
    y_all = label.loc[mondays].stack(future_stack=True).reindex(x_all.index)
    row_day = x_all.index.get_level_values(0)
    labelled = y_all.notna().to_numpy()
    week_ok = pd.Series(labelled, index=row_day).groupby(level=0).sum() >= ML_MIN_COINS

    scores = pd.DataFrame(np.nan, index=m.close.index, columns=m.close.columns)
    model, weeks_since_fit, refits, first = None, 0, [], None
    for t in mondays:
        known = row_day + pd.Timedelta(days=ML_LABEL_LAG) < t  # the outcome ended before the prediction date
        train_weeks = week_ok.index[(week_ok.index + pd.Timedelta(days=ML_LABEL_LAG) < t) & week_ok.to_numpy()]
        if len(train_weeks) < ML_MIN_TRAIN_WEEKS:
            continue
        if model is None or weeks_since_fit >= ML_REFIT_WEEKS:
            rows = known & labelled & row_day.isin(train_weeks)
            assert (row_day[rows] + pd.Timedelta(days=ML_LABEL_LAG)).max() < t, "training label overlaps the prediction date"
            model = HistGradientBoostingRegressor(**ML_PARAMS).fit(x_all[rows].to_numpy(), y_all[rows].to_numpy())
            weeks_since_fit = 0
            refits.append({"date": str(t.date()), "rows": int(rows.sum()), "weeks": int(len(train_weeks))})
        today = row_day == t
        if today.any():
            scores.loc[t, x_all.index.get_level_values(1)[today]] = model.predict(x_all[today].to_numpy())
            first = first or t
        weeks_since_fit += 1
    return scores, {"first_prediction": str(first.date()) if first is not None else None, "refits": refits}


def ml_targets(m: data_mod.Market, univ: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    hi, lo = strat._quintile_legs(scores, univ)
    return eng.on_days(strat._long_short(hi, lo), eng.weekly_days(m.close.index))


def ml_diagnostics(m: data_mod.Market, univ: pd.DataFrame, scores: pd.DataFrame, start: str, end: str) -> dict:
    """Rank correlation of predicted vs realised weekly relative return, and how often the long leg beat the short leg."""
    label = ml_labels(m, univ)
    hi, lo = strat._quintile_legs(scores, univ)
    ics, calls = [], []
    for t in eng.window_frame(scores, start, end).dropna(how="all").index:
        y, s = label.loc[t], scores.loc[t]
        ok = y.notna() & s.notna()
        if ok.sum() < ML_MIN_COINS:
            continue
        ics.append(s[ok].rank().corr(y[ok].rank()))
        spread = y[hi.loc[t] & ok].mean() - y[lo.loc[t] & ok].mean()
        if np.isfinite(spread):
            calls.append(spread > 0)
    return {"weeks": len(ics), "rank_correlation": float(np.mean(ics)) if ics else None,
            "long_beat_short_share": float(np.mean(calls)) if calls else None}


# ------------------------------------------------------------------------ all books


def build_books(m: data_mod.Market, names: tuple[str, ...] = ("B0", "V1", "V2", "V3", "V4"), lag: int = 1) -> dict[str, Book]:
    rank, univ = strat.prepare(m)
    books: dict[str, Book] = {}
    base_targets = strat.trend(m, univ, LOOKBACKS)
    if "B0" in names:
        books["B0"] = run_book(m, rank, base_targets, lag=lag)
    if "V1" in names:
        books["V1"] = run_book(m, rank, selective_targets(m, univ), lag=lag)
    if "V2" in names:
        books["V2"] = run_book(m, rank, strat.trend(m, eng.universe(rank, BROAD_N), LOOKBACKS), lag=lag)
    if "V3" in names:
        er = trendiness(m, univ)
        on = regime_on(er)
        books["V3"] = run_book(m, rank, base_targets, gate=on, lag=lag)
        books["V3"].info = {"trendiness": er, "on": on}
    if "V4" in names:
        scores, info = ml_scores(m, rank, univ)
        books["V4"] = run_book(m, rank, ml_targets(m, univ, scores), lag=lag)
        books["V4"].info = {**info, "scores": scores}
    return books


def _total_return(res: eng.Result, start: str, end: str) -> float:
    return float((1.0 + eng.window(res.returns, start, end)).prod() - 1.0)


def design() -> dict:
    import sklearn

    m = data_mod.build().truncate(DESIGN_CUTOFF)
    assert m.close.index.max() <= pd.Timestamp(DESIGN_CUTOFF, tz="UTC")
    rank, univ = strat.prepare(m)
    books = build_books(m)
    same_close = build_books(m, lag=0)
    btc = m.close["BTCUSDT"].pct_change(fill_method=None)
    mondays = eng.weekly_days(m.close.index)

    ml_start = books["V4"].info["first_prediction"]
    ml_fit = (max(ml_start, FIT[0]), FIT[1])
    report: dict = {"windows": {"fit": FIT, "validation": VAL, "V4_fit": ml_fit}, "books": {},
                    "sklearn": sklearn.__version__, "ml_params": ML_PARAMS}
    for name, book in books.items():
        fit_w = ml_fit if name == "V4" else FIT
        res = book.result
        report["books"][name] = {
            "fit": eng.metrics(res, *fit_w),
            "validation": eng.metrics(res, *VAL),
            "total_return_fit_plus_validation": _total_return(res, fit_w[0], VAL[1]),
            "years": eng.yearly(res),
            "btc_correlation": float(eng.window(res.returns, fit_w[0], VAL[1]).corr(eng.window(btc, fit_w[0], VAL[1]))),
            "same_close_fill_sharpe": {"fit": eng.metrics(same_close[name].result, *fit_w).get("sharpe"),
                                       "validation": eng.metrics(same_close[name].result, *VAL).get("sharpe")},
        }
    on = books["V3"].info["on"]
    er = books["V3"].info["trendiness"]
    report["books"]["V3"]["switch"] = {
        "first_day_it_can_be_on": str(er.expanding(min_periods=ER_MIN_HISTORY).median().shift(1).first_valid_index().date()),
        "share_of_weeks_on": {k: float(eng.window(on.loc[mondays], *w).mean()) for k, w in (("fit", FIT), ("validation", VAL))},
    }
    v4 = books["V4"].info
    report["books"]["V4"]["model"] = {
        "first_prediction": v4["first_prediction"], "refits": v4["refits"],
        "fit": ml_diagnostics(m, univ, v4["scores"], *ml_fit),
        "validation": ml_diagnostics(m, univ, v4["scores"], *VAL),
        # the literal fit window starts at the first prediction, which includes the 60 days the volatility
        # scaling needs before it takes any exposure; this is the same figure from the first day with exposure
        "fit_sharpe_from_first_exposure": eng.metrics(
            books["V4"].result, str(books["V4"].result.gross.loc[lambda g: g > 0].index[0].date()), FIT[1]).get("sharpe"),
    }

    b = report["books"]
    b0_fit, b0_val = b["B0"]["fit"]["sharpe"], b["B0"]["validation"]["sharpe"]
    selection: dict = {"tweaks": {}}
    for name in ("V1", "V2", "V3"):
        checks = {
            f"fit Sharpe beats B0 by {EDGE}": b[name]["fit"]["sharpe"] >= b0_fit + EDGE,
            f"validation Sharpe beats B0 by {EDGE}": b[name]["validation"]["sharpe"] >= b0_val + EDGE,
            "validation drawdown at most 20%": b[name]["validation"]["max_dd"] <= TWEAK_MAX_DD,
        }
        selection["tweaks"][name] = {"checks": checks, "qualifies": all(checks.values())}
    qualified = [n for n, s in selection["tweaks"].items() if s["qualifies"]]
    best = max(qualified, key=lambda n: b[n]["validation"]["sharpe"]) if qualified else None
    ml_checks = {
        "fit Sharpe above 0.7": b["V4"]["fit"]["sharpe"] > ML_MIN_SHARPE,
        "validation Sharpe above 0.7": b["V4"]["validation"]["sharpe"] > ML_MIN_SHARPE,
        "validation drawdown under 25%": b["V4"]["validation"]["max_dd"] < ML_MAX_DD,
        "positive over both windows together": b["V4"]["total_return_fit_plus_validation"] > 0,
        "beats B0 validation Sharpe": b["V4"]["validation"]["sharpe"] > b0_val,
    }
    selection["V4"] = {"checks": ml_checks, "qualifies": all(ml_checks.values())}
    selection["forward_books"] = ["B0"] + ([best] if best else []) + (["V4"] if selection["V4"]["qualifies"] else [])
    report["selection"] = selection

    phase1b = json.loads((RESULTS / "discovery.json").read_text())["families"]["T"]["variants"]["T_slow_30_60_120"]
    report["baseline_matches_phase_1b"] = bool(
        np.isclose(b0_fit, phase1b["fit"]["sharpe"]) and np.isclose(b0_val, phase1b["validation"]["sharpe"]))
    report["benchmarks"] = {
        "BTC buy and hold": {"fit": _series_metrics(btc.fillna(0.0), *FIT), "validation": _series_metrics(btc.fillna(0.0), *VAL)},
    }
    daily = pd.DataFrame({n: bk.result.returns for n, bk in books.items()})
    report["daily_returns_sha256"] = hashlib.sha256(daily.round(12).to_csv().encode()).hexdigest()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "adaptive_design.json").write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    daily.loc[FIT[0]:].to_csv(RESULTS / "adaptive_design_daily_returns.csv")
    _print(report)
    return report


def _series_metrics(r: pd.Series, start: str, end: str) -> dict:
    return eng.metrics(eng.Result(r, r, r * 0, r * 0, r * 0, r * 0 + 1, r * 0 + 1), start, end)


def _print(report: dict) -> None:
    pct = lambda x: f"{x * 100:6.1f}%"  # noqa: E731
    print(f"{'book':4} {'fitShp':>7} {'fitCAGR':>8} {'fitDD':>7} | {'valShp':>7} {'valCAGR':>8} {'valDD':>7} | "
          f"{'cost/y':>7} {'fund/y':>7} {'turn/y':>7} {'gross':>6} {'corrBTC':>8} | same-close fit/val")
    for name, v in report["books"].items():
        f, va, sc = v["fit"], v["validation"], v["same_close_fill_sharpe"]
        print(f"{name:4} {f['sharpe']:7.2f} {pct(f['cagr'])}  {pct(f['max_dd'])} | {va['sharpe']:7.2f} {pct(va['cagr'])}  "
              f"{pct(va['max_dd'])} | {pct(va['cost_per_year'])} {pct(va['funding_per_year'])} {va['turnover_per_year']:6.1f}x "
              f"{va['avg_gross']:6.2f} {v['btc_correlation']:8.2f} | {sc['fit']:5.2f} / {sc['validation']:5.2f}")
        print("     years:", {y: f"{r * 100:.1f}%" for y, r in v["years"].items()})
    print("V3 switch:", json.dumps(report["books"]["V3"]["switch"]))
    mdl = report["books"]["V4"]["model"]
    print("V4 model:", json.dumps({k: mdl[k] for k in ("first_prediction", "fit", "validation", "fit_sharpe_from_first_exposure")}),
          f"{len(mdl['refits'])} refits")
    print("selection:", json.dumps(report["selection"], indent=1))
    print("baseline matches Phase 1b:", report["baseline_matches_phase_1b"], "| daily returns sha256:", report["daily_returns_sha256"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["design"])
    ap.parse_args()
    design()
