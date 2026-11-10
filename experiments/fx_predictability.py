"""Part A of the other-markets bake-off: can a model predict forex (and ETF) direction?

Preregistration: docs/research/experiments/2026-11-10_other-markets-bakeoff-prereg.md

Walk-forward (expanding window, refit each calendar year) logistic regression and
LightGBM on price/indicator features, pooled per market family. Trades the top and
bottom decile of predictions, net of spreads and financing. Crypto (BTC, ETH) runs
through the same pipeline as a reference.

Usage: python experiments/fx_predictability.py
Writes experiments/markets_results/fx_predictability.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import lightgbm as lgb
import markets_common as mc
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

OUT = Path(__file__).resolve().parent / "markets_results" / "fx_predictability.json"
RET_LAGS = [1, 2, 3, 6, 12, 24, 48, 120]
MIN_TRAIN_YEARS = 3
ETFS = ["SPY", "QQQ", "EFA", "EEM", "IEF", "TLT", "GLD"]


# --------------------------------------------------------------------------- features


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100.0 - 100.0 / (1.0 + up / down.replace(0, np.nan))


def build_features(bars: pd.DataFrame, hourly: bool) -> pd.DataFrame:
    """Features known at the close of bar t, plus the next bar's return (the thing predicted)."""
    close, high, low = bars["close"], bars["high"], bars["low"]
    logc = np.log(close)
    lr = logc.diff()
    f = pd.DataFrame(index=bars.index)
    for k in RET_LAGS:
        f[f"ret_{k}"] = logc - logc.shift(k)
    f["rsi14"] = _rsi(close)
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / close
    for span in (20, 50, 200):
        f[f"ema_{span}"] = close / close.ewm(span=span, adjust=False).mean() - 1.0
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    f["atr14"] = true_range.ewm(alpha=1.0 / 14, adjust=False).mean() / close
    f["vol_24"] = lr.rolling(24).std()
    f["vol_120"] = lr.rolling(120).std()
    f["bb_z"] = (close - close.rolling(20).mean()) / close.rolling(20).std()
    hi24, lo24 = high.rolling(24).max(), low.rolling(24).min()
    f["range_pos"] = (close - lo24) / (hi24 - lo24).replace(0, np.nan)
    if hourly:
        hour = bars.index.hour + bars.index.minute / 60.0
        f["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
        f["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
        f["dow"] = bars.index.day_of_week.astype(float)
    f = f.iloc[250:]  # warm-up for the 200-bar EMA and 120-bar windows
    f["next_ret"] = (close.shift(-1) / close - 1.0).reindex(f.index)
    f["dt_days"] = (
        pd.Series(bars.index, index=bars.index).shift(-1) - pd.Series(bars.index, index=bars.index)
    ).dt.total_seconds().reindex(f.index) / 86400.0
    return f.replace([np.inf, -np.inf], np.nan).dropna()


# --------------------------------------------------------------------------- models


def _fit_predict(
    model: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, min_leaf: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (in-sample probabilities, test probabilities) of an up move."""
    if model == "logit":
        mu, sd = x_train.mean(axis=0), x_train.std(axis=0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(C=1.0, max_iter=500)
        clf.fit((x_train - mu) / sd, y_train)
        return (
            clf.predict_proba((x_train - mu) / sd)[:, 1],
            clf.predict_proba((x_test - mu) / sd)[:, 1],
        )
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=min_leaf,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=7,
        deterministic=True,
        force_row_wise=True,
        n_jobs=4,
        verbose=-1,
    )
    clf.fit(x_train, y_train)
    return clf.predict_proba(x_train)[:, 1], clf.predict_proba(x_test)[:, 1]


NON_FEATURES = ("next_ret", "dt_days", "rf")


def walk_forward(
    frames: dict[str, pd.DataFrame], model: str, min_leaf: int, first_oos_year: int
) -> pd.DataFrame:
    """Out-of-sample predictions for every instrument. Adds columns inst, p, lo, hi."""
    feat_cols = [c for c in next(iter(frames.values())).columns if c not in NON_FEATURES]
    pooled = pd.concat([f.assign(inst=name) for name, f in frames.items()]).sort_index(
        kind="stable"
    )
    out = []
    for year in range(first_oos_year, int(pooled.index.year.max()) + 1):
        cutoff = pd.Timestamp(year=year, month=1, day=1)
        train = pooled[pooled.index < cutoff]
        # Purge: a training row's target must be fully realised before the cutoff.
        train = train[train.index + pd.to_timedelta(train["dt_days"], unit="D") < cutoff]
        test = pooled[(pooled.index >= cutoff) & (pooled.index < cutoff + pd.DateOffset(years=1))]
        if len(test) == 0 or train.index.year.nunique() < MIN_TRAIN_YEARS:
            continue
        y_train = (train["next_ret"] > 0).astype(int).to_numpy()
        p_in, p_out = _fit_predict(
            model,
            train[feat_cols].to_numpy(float),
            y_train,
            test[feat_cols].to_numpy(float),
            min_leaf,
        )
        res = test[[c for c in test.columns if c in NON_FEATURES or c == "inst"]].copy()
        res["p"] = p_out
        res["lo"], res["hi"] = np.quantile(p_in, 0.10), np.quantile(p_in, 0.90)
        out.append(res)
    return pd.concat(out)


# --------------------------------------------------------------------------- trading the predictions


def _fx_rate_diff(pair: str, index: pd.DatetimeIndex, rates: pd.DataFrame) -> pd.Series:
    """Annual rate differential earned by a long position in the pair (base minus quote)."""
    base, quote = pair[:3], pair[3:]
    diff = (rates[base] - rates[quote]) / 100.0
    days = index.normalize()
    daily = diff.reindex(diff.index.union(days.unique())).ffill()
    return pd.Series(daily.loc[days].to_numpy(), index=index)


def trade(preds: pd.DataFrame, family: str, rates: pd.DataFrame, always_in: bool = False) -> dict:
    """Per-instrument net P&L of the decile rule, then an equal-weight portfolio."""
    net_cols, gross_cols, trades, costs_rt, abs_moves = {}, {}, 0, [], []
    for inst, g in preds.groupby("inst"):
        g = g.sort_index()
        if always_in:
            pos = np.sign(g["p"] - 0.5)
        else:
            pos = pd.Series(
                np.where(g["p"] > g["hi"], 1.0, np.where(g["p"] < g["lo"], -1.0, 0.0)),
                index=g.index,
            )
        dpos = pos.diff().abs().fillna(pos.abs())
        gross = pos * g["next_ret"]
        years = g["dt_days"] / 365.0
        if family == "fx":
            side = mc.FX_COST_BPS[mc.pair_ccy(inst)] / 1e4
            carry = pos * _fx_rate_diff(inst, g.index, rates) * years
            net = gross - side * dpos + carry - mc.FX_MARKUP * pos.abs() * years
        elif family == "etf":
            side = mc.ETF_COST
            # Excess-return bookkeeping per unit of capital: long earns R - rf, flat pays the
            # cash haircut, short earns -R and pays the haircut plus the borrow fee.
            long_w, short_w = pos.clip(lower=0), (-pos).clip(lower=0)
            net = (
                gross
                - side * dpos
                - long_w * g["rf"]
                - (1 - long_w) * mc.CASH_HAIRCUT * years
                - short_w * mc.SHORT_BORROW * years
            )
        else:  # crypto reference
            side = mc.CRYPTO_COST_RT / 2.0
            net = gross - side * dpos
        net_cols[inst], gross_cols[inst] = net, gross
        trades += int((dpos > 0).sum())
        costs_rt.append(2 * side)
        abs_moves.append(float(g["next_ret"].abs().mean()))
    net_df = pd.DataFrame(net_cols).fillna(0.0)
    gross_df = pd.DataFrame(gross_cols).fillna(0.0)
    return {
        "net": net_df.mean(axis=1),
        "gross": gross_df.mean(axis=1),
        "trades": trades,
        "cost_rt": float(np.mean(costs_rt)),
        "mean_abs_move": float(np.mean(abs_moves)),
    }


def to_daily(x: pd.Series) -> pd.Series:
    return x.groupby(x.index.normalize()).sum()


def ppy(x: pd.Series) -> float:
    span_years = (x.index[-1] - x.index[0]).days / 365.25
    return len(x) / span_years


def evaluate(
    cell: str, preds: pd.DataFrame, family: str, rates: pd.DataFrame, rf_daily_rate: pd.Series
) -> dict:
    moved = preds[preds["next_ret"] != 0]
    hit = (np.sign(moved["p"] - 0.5) == np.sign(moved["next_ret"])).to_numpy()
    n = len(hit)
    da = float(hit.mean())
    picked = moved[(moved["p"] > moved["hi"]) | (moved["p"] < moved["lo"])]
    da_picked = (
        float((np.sign(picked["p"] - 0.5) == np.sign(picked["next_ret"])).mean())
        if len(picked)
        else float("nan")
    )
    res = trade(preds, family, rates)
    res_always = trade(preds, family, rates, always_in=True)
    net_d, gross_d, always_d = (
        to_daily(res["net"]),
        to_daily(res["gross"]),
        to_daily(res_always["net"]),
    )
    if family == "fx":  # idle margin: 25% of capital earns nothing instead of T-bills
        rate = (
            rf_daily_rate.reindex(rf_daily_rate.index.union(net_d.index))
            .ffill()
            .reindex(net_d.index)
            .fillna(0.0)
        )
        day_gap = pd.Series(net_d.index, index=net_d.index).diff().dt.days.fillna(1)
        drag = mc.FX_IDLE_MARGIN * rate * day_gap / 365.0
        net_d, always_d = net_d - drag, always_d - drag
    periods = ppy(net_d)
    span_years = (net_d.index[-1] - net_d.index[0]).days / 365.25
    g = mc.gate(net_d, periods, window=(str(net_d.index[0].date()), str(net_d.index[-1].date())))
    return {
        "cell": cell,
        "oos_start": str(net_d.index[0].date()),
        "oos_end": str(net_d.index[-1].date()),
        "oos_bars": int(len(preds)),
        "directional_accuracy": da,
        "da_ci95": [da - 1.96 * math.sqrt(0.25 / n), da + 1.96 * math.sqrt(0.25 / n)],
        "da_on_traded_deciles": da_picked,
        "share_of_bars_traded": float(len(picked) / max(len(moved), 1)),
        "breakeven_accuracy": 0.5 + res["cost_rt"] / (2 * res["mean_abs_move"]),
        "cost_rt_bps": res["cost_rt"] * 1e4,
        "mean_abs_move_bps": res["mean_abs_move"] * 1e4,
        "trades_per_year": res["trades"] / span_years,
        "gross_sharpe": mc.perf(gross_d, periods)["sharpe"],
        "net": g["window"],
        "always_in_net_sharpe": mc.perf(always_d, periods)["sharpe"],
        "recent_mean_ann": g["recent_mean_ann"],
        "gates": {k: bool(g[k]) for k in ("G1", "G2", "G4", "G5")},
        "verdict": g["verdict"],
        "net_by_year": mc.yearly(net_d),
    }


# --------------------------------------------------------------------------- descriptive


def describe(name: str, bars: pd.DataFrame, bars_per_year: float, cost_rt: float) -> dict:
    r = np.log(bars["close"]).diff().dropna()

    def variance_ratio(k: int) -> float:
        return float(r.rolling(k).sum().dropna().var() / (k * r.var()))

    return {
        "inst": name,
        "ann_vol": float(r.std() * math.sqrt(bars_per_year)),
        "autocorr_1": float(r.autocorr(1)),
        "vr_5": variance_ratio(5),
        "vr_20": variance_ratio(20),
        "mean_abs_move_bps": float(r.abs().mean() * 1e4),
        "cost_rt_over_move": float(cost_rt / r.abs().mean()),
    }


# --------------------------------------------------------------------------- main


def main() -> None:
    rates = mc.load_policy_rates()
    irx = mc.load_yahoo("^IRX")["close"] / 100.0

    fx_h = {p: mc.load_fx_minute_resampled(p)[0] for p in mc.FX_PAIRS}
    fx_d = {p: mc.hourly_to_daily(h) for p, h in fx_h.items()}
    crypto_h = {s: mc.load_binance_1h(s) for s in ("BTCUSDT", "ETHUSDT")}
    etf_d = {}
    for sym in ETFS:
        y = mc.load_yahoo(sym).loc["2005-01-01":]
        etf_d[sym] = pd.DataFrame(
            {"open": y["adjopen"], "high": y["adjhigh"], "low": y["adjlow"], "close": y["adjclose"]}
        ).dropna()

    # family -> (cost model, feature frames, LightGBM min leaf, first out-of-sample year)
    families = {
        "FX-1h": ("fx", {p: build_features(b, True) for p, b in fx_h.items()}, 500, 2013),
        "FX-1d": ("fx", {p: build_features(b, False) for p, b in fx_d.items()}, 200, 2013),
        "ETF-1d": ("etf", {s: build_features(b, False) for s, b in etf_d.items()}, 200, 2010),
        "crypto-1h": (
            "crypto",
            {s: build_features(b, True) for s, b in crypto_h.items()},
            500,
            2021,
        ),
    }
    # ETF rows need the T-bill accrual over the holding bar (yield known at the bar's close).
    for frame in families["ETF-1d"][1].values():
        rate = irx.reindex(irx.index.union(frame.index)).ffill().reindex(frame.index)
        frame["rf"] = rate * frame["dt_days"] / 365.0

    cells = []
    for fam_name, (family, frames, min_leaf, first_oos_year) in families.items():
        for model in ("logit", "GBM"):
            preds = walk_forward(frames, model.lower(), min_leaf, first_oos_year)
            preds.to_pickle(
                mc.DATA_DIR / f"preds_{fam_name}-{model}.pkl"
            )  # for diagnostics; not committed
            cell = evaluate(f"{fam_name}-{model}", preds, family, rates, irx)
            cell["gated"] = family != "crypto"
            cells.append(cell)
            print(
                f"{cell['cell']:18s} DA {cell['directional_accuracy']:.4f} (b/e {cell['breakeven_accuracy']:.4f})  gross SR {cell['gross_sharpe']:+.2f}  net SR {cell['net']['sharpe']:+.2f}  DD {cell['net']['max_dd']:.1%}  {cell['verdict']}",
                flush=True,
            )

    descriptive = []
    for p in mc.FX_PAIRS:
        rt = 2 * mc.FX_COST_BPS[mc.pair_ccy(p)] / 1e4
        descriptive.append({"bar": "1h", **describe(p, fx_h[p], ppy(fx_h[p]["close"]), rt)})
        descriptive.append({"bar": "1d", **describe(p, fx_d[p], ppy(fx_d[p]["close"]), rt)})
    for s, bars in crypto_h.items():
        descriptive.append(
            {"bar": "1h", **describe(s, bars, ppy(bars["close"]), mc.CRYPTO_COST_RT)}
        )
        daily = (
            bars.resample("1D")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
            .dropna()
        )
        descriptive.append({"bar": "1d", **describe(s, daily, 365.0, mc.CRYPTO_COST_RT)})

    data_meta = {p: mc.load_fx_minute_resampled(p)[1] for p in mc.FX_PAIRS}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "cells": cells,
                "descriptive": descriptive,
                "fx_data": data_meta,
                "bonferroni_z": mc.BONFERRONI_Z,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
