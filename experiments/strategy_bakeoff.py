"""Phase 1 strategy bake-off, as preregistered in
docs/research/experiments/2026-11-10_strategy-bakeoff-prereg.md.

Self-contained: pandas/numpy/requests only. Downloads Binance spot daily klines,
USD-M perp daily klines (BTC/ETH, for the carry hedge's basis) and USD-M funding
history, caches them under experiments/bakeoff_data/, then runs S1 carry, S2 trend
with vol targeting, S3 cross-sectional momentum and BTC buy-and-hold.

Timing: a signal computed on day t's close is filled at day t+1's close.

Usage: python experiments/strategy_bakeoff.py [--refresh]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import requests

COINS = ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "LINK", "LTC", "AVAX", "DOT", "TRX"]
START = pd.Timestamp("2019-01-01", tz="UTC")
END = pd.Timestamp("2026-10-31", tz="UTC")
DATA_DIR = Path(__file__).resolve().parent / "bakeoff_data"

SPOT_COST = 0.0015  # 10 bps taker + 5 bps slippage, per side
PERP_COST = 0.0010  # 5 bps taker + 5 bps slippage, per side
MIN_HISTORY_DAYS = 200
ANN = 365

IS_WINDOW = ("2020-01-01", "2024-12-31")
HOLDOUT_WINDOW = ("2025-01-01", "2026-10-31")
IS_YEARS = [2020, 2021, 2022, 2023, 2024]

SPOT_URL = "https://api.binance.com/api/v3/klines"
PERP_URL = "https://fapi.binance.com/fapi/v1/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


# --------------------------------------------------------------------------- data


def _get(url: str, params: dict) -> list:
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 400:
            return []  # unknown symbol (not listed)
        time.sleep(2**attempt)
    resp.raise_for_status()
    return []


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


def fetch_daily_klines(url: str, symbol: str) -> pd.DataFrame:
    rows: list = []
    start = _ms(START)
    end = _ms(END + pd.Timedelta(days=1)) - 1
    while start <= end:
        batch = _get(url, {"symbol": symbol, "interval": "1d", "startTime": start, "endTime": end, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        start = batch[-1][0] + 86_400_000
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["date", "close"])
    df = pd.DataFrame([(r[0], float(r[4])) for r in rows], columns=["open_ms", "close"])
    df["date"] = pd.to_datetime(df["open_ms"], unit="ms", utc=True).dt.normalize()
    return df[["date", "close"]].drop_duplicates("date")


def fetch_funding(symbol: str) -> pd.DataFrame:
    rows: list = []
    start = _ms(START)
    end = _ms(END + pd.Timedelta(days=1)) - 1
    while start <= end:
        batch = _get(FUNDING_URL, {"symbol": symbol, "startTime": start, "endTime": end, "limit": 1000})
        if not batch:
            break
        rows.extend(batch)
        start = int(batch[-1]["fundingTime"]) + 1
        if len(batch) < 1000:
            break
    if not rows:
        return pd.DataFrame(columns=["time", "rate"])
    df = pd.DataFrame({"time": [int(r["fundingTime"]) for r in rows], "rate": [float(r["fundingRate"]) for r in rows]})
    df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    return df.drop_duplicates("time").sort_values("time")


def _cached(path: Path, fetch, refresh: bool) -> pd.DataFrame:
    if path.exists() and not refresh:
        df = pd.read_csv(path)
        for col in ("date", "time"):
            if col in df:
                df[col] = pd.to_datetime(df[col], utc=True, format="ISO8601")
        return df
    df = fetch()
    df.to_csv(path, index=False)
    return df


@dataclass
class Market:
    spot: pd.DataFrame  # date x coin close, NaN before listing
    perp: pd.DataFrame  # date x coin perp close (BTC, ETH only)
    funding: dict[str, pd.DataFrame]  # coin -> (time, rate)


def load_market(refresh: bool) -> Market:
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / ".gitignore").write_text("*\n")
    idx = pd.date_range(START, END, freq="D")
    spot, perp, funding = {}, {}, {}
    for coin in COINS:
        sym = f"{coin}USDT"
        s = _cached(DATA_DIR / f"spot_{sym}.csv", lambda: fetch_daily_klines(SPOT_URL, sym), refresh)
        spot[coin] = s.set_index("date")["close"].reindex(idx)
        funding[coin] = _cached(DATA_DIR / f"funding_{sym}.csv", lambda: fetch_funding(sym), refresh)
        if coin in ("BTC", "ETH"):
            p = _cached(DATA_DIR / f"perp_{sym}.csv", lambda: fetch_daily_klines(PERP_URL, sym), refresh)
            perp[coin] = p.set_index("date")["close"].reindex(idx)
    return Market(pd.DataFrame(spot), pd.DataFrame(perp), funding)


# --------------------------------------------------------------------- simulator


@dataclass
class Result:
    returns: pd.Series  # daily net return on total capital
    fees: pd.Series  # daily fees as fraction of equity
    turnover: pd.Series  # daily traded notional / equity (one side)
    exposure: pd.Series  # daily gross exposure
    extra: dict = field(default_factory=dict)


def simulate_spot(prices: pd.DataFrame, signal_weights: pd.DataFrame) -> Result:
    """Long-only spot portfolio.

    `signal_weights` holds target weights on signal days (NaN rows = no rebalance).
    A target set at close t is executed at close t+1; holdings drift with prices between rebalances.
    """
    rets = prices.pct_change(fill_method=None).fillna(0.0).to_numpy()
    targets = signal_weights.shift(1).to_numpy()  # fill one day after the signal
    n_days, n_assets = rets.shape
    w = np.zeros(n_assets)  # current weights (value / equity) after the close
    out_ret = np.zeros(n_days)
    out_fee = np.zeros(n_days)
    out_turn = np.zeros(n_days)
    out_gross = np.zeros(n_days)
    for d in range(n_days):
        # mark-to-market from close d-1 to close d
        growth = w * rets[d]
        port_ret = growth.sum()
        if w.any():
            w = (w + growth) / (1.0 + port_ret)
        fee = 0.0
        turn = 0.0
        tgt = targets[d]
        if not np.isnan(tgt).all():
            tgt = np.nan_to_num(tgt)
            traded = np.abs(tgt - w).sum()
            fee = traded * SPOT_COST
            turn = traded
            w = tgt.copy()
        out_ret[d] = (1.0 + port_ret) * (1.0 - fee) - 1.0
        out_fee[d] = fee
        out_turn[d] = turn
        out_gross[d] = w.sum()
    i = prices.index
    return Result(pd.Series(out_ret, i), pd.Series(out_fee, i), pd.Series(out_turn, i), pd.Series(out_gross, i))


def weekly_signal_days(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index[index.dayofweek == 0]  # Monday close


def eligible(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.notna().cumsum() >= MIN_HISTORY_DAYS


# --------------------------------------------------------------------- strategies


def strat_buy_hold_btc(m: Market) -> Result:
    p = m.spot[["BTC"]]
    w = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    w.iloc[0] = 1.0
    return simulate_spot(p, w)


def strat_trend_vol(m: Market, sma_len: int = 100, vol_len: int = 30, vol_target: float = 0.20) -> Result:
    p = m.spot
    sma = p.rolling(sma_len, min_periods=sma_len).mean()
    vol = p.pct_change(fill_method=None).rolling(vol_len, min_periods=vol_len).std() * np.sqrt(ANN)
    elig = eligible(p) & sma.notna() & vol.notna()
    n_elig = elig.sum(axis=1)
    raw = (vol_target / n_elig.replace(0, np.nan)).to_numpy()[:, None] / vol
    raw = raw.where(elig & (p > sma), 0.0).fillna(0.0)
    gross = raw.sum(axis=1)
    scale = np.where(gross > 1.0, 1.0 / gross.replace(0, np.nan), 1.0)
    w = raw.mul(pd.Series(scale, index=raw.index).fillna(1.0), axis=0)
    targets = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    days = weekly_signal_days(p.index)
    targets.loc[days] = w.loc[days]
    return simulate_spot(p, targets)


def strat_xs_momentum(m: Market, sma_len: int = 100, mom_len: int = 28, top_n: int = 3) -> Result:
    p = m.spot
    btc = p["BTC"]
    regime = btc > btc.rolling(sma_len, min_periods=sma_len).mean()
    mom = p / p.shift(mom_len) - 1.0
    elig = eligible(p) & mom.notna()
    targets = pd.DataFrame(np.nan, index=p.index, columns=p.columns)
    for day in weekly_signal_days(p.index):
        row = pd.Series(0.0, index=p.columns)
        if regime.loc[day]:
            ranked = mom.loc[day][elig.loc[day]].sort_values(ascending=False)
            picks = ranked.index[:top_n]
            if len(picks):
                row[picks] = 1.0 / len(picks)
        targets.loc[day] = row
    return simulate_spot(p, targets)


def daily_funding(m: Market, coin: str, index: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    """Funding received by a short perp on day d (prints in (00:00 d, 00:00 d+1]), and the
    trailing-mean input known at close d. Returns (daily_sum, prints_series)."""
    f = m.funding[coin].set_index("time")["rate"]
    # Prints are stamped a few ms after 00:00/08:00/16:00; the 00:00 print of d+1 settles at day d's close.
    day = (f.index - pd.Timedelta(hours=1)).floor("D")
    daily_sum = f.groupby(day).sum().reindex(index).fillna(0.0)
    return daily_sum, f


def strat_carry(
    m: Market, lookback_prints: int = 21, enter_thr: float = 0.00005, coins: tuple[str, ...] = ("BTC", "ETH")
) -> Result:
    """Long spot + short perp, equal notional. Each coin gets half the capital; half of a coin's
    capital buys spot, the other half is perp collateral, so notional = 0.5 * coin capital."""
    idx = m.spot.index
    alloc = 1.0 / len(coins)
    notional = 0.5 * alloc  # per coin, as fraction of total capital at last rebalance
    total_ret = pd.Series(0.0, index=idx)
    total_fee = pd.Series(0.0, index=idx)
    total_turn = pd.Series(0.0, index=idx)
    total_gross = pd.Series(0.0, index=idx)
    in_pos_frac = {}
    for coin in coins:
        spot_r = m.spot[coin].pct_change(fill_method=None)
        perp_r = m.perp[coin].pct_change(fill_method=None)
        fund_day, prints = daily_funding(m, coin, idx)
        # trailing mean of the last N prints known at each day's close
        trail = prints.rolling(lookback_prints, min_periods=lookback_prints).mean()
        trail_at_close = trail.groupby((trail.index - pd.Timedelta(hours=1)).floor("D")).last()
        trail_at_close = trail_at_close.reindex(idx).ffill()
        tradable = m.perp[coin].notna() & m.spot[coin].notna()

        state = False
        signal = np.zeros(len(idx), dtype=bool)
        for i, (val, ok) in enumerate(zip(trail_at_close.to_numpy(), tradable.to_numpy())):
            if not ok or np.isnan(val):
                state = False
            elif state and val < 0:
                state = False
            elif not state and val > enter_thr:
                state = True
            signal[i] = state
        held = pd.Series(signal, index=idx).shift(1, fill_value=False)  # filled at next close
        # position during day d is the one held after close d-1
        pos = held.shift(1, fill_value=False)
        leg_ret = (spot_r.fillna(0.0) - perp_r.fillna(0.0) + fund_day).where(pos, 0.0)
        change = held.astype(int).diff().abs().fillna(held.astype(int))
        fee = change * notional * (SPOT_COST + PERP_COST)
        # Keeping the hedge at equal notional and the perp margin topped up means moving the day's
        # spot P&L between legs; charge it as a spot trade of that size.
        fee = fee + (spot_r.abs().fillna(0.0) * notional * SPOT_COST).where(pos, 0.0)
        total_ret += leg_ret * notional
        total_fee += fee
        total_turn += change * notional * 2
        total_gross += pos.astype(float) * notional * 2
        in_pos_frac[coin] = held
    net = (1.0 + total_ret) * (1.0 - total_fee) - 1.0
    res = Result(net, total_fee, total_turn, total_gross)
    res.extra["in_position"] = in_pos_frac
    return res


# ------------------------------------------------------------------------ metrics


def window(s: pd.Series, start: str, end: str) -> pd.Series:
    return s.loc[pd.Timestamp(start, tz="UTC") : pd.Timestamp(end, tz="UTC")]


def metrics(res: Result, start: str, end: str) -> dict:
    r = window(res.returns, start, end)
    eq = (1.0 + r).cumprod()
    years = len(r) / ANN
    total = eq.iloc[-1] - 1.0
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else -1.0
    sd = r.std()
    sharpe = (r.mean() / sd) * np.sqrt(ANN) if sd > 0 else float("nan")
    dd = (eq / eq.cummax().clip(lower=1.0) - 1.0).min()
    fees = window(res.fees, start, end)
    eq_before = eq.shift(1, fill_value=1.0)
    return {
        "total_return": total,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_dd": -dd,
        "fees_pct_of_start_capital": float((fees * eq_before).sum()),
        "turnover_per_year": float(window(res.turnover, start, end).sum() / years),
        "avg_gross": float(window(res.exposure, start, end).mean()),
    }


def yearly_returns(res: Result) -> dict[int, float]:
    out = {}
    for year, r in res.returns.groupby(res.returns.index.year):
        out[int(year)] = float((1.0 + r).prod() - 1.0)
    return out


def gate(res: Result, robustness: dict[str, Result]) -> dict:
    is_m = metrics(res, *IS_WINDOW)
    ho_m = metrics(res, *HOLDOUT_WINDOW)
    yr = yearly_returns(res)
    pos_years = sum(1 for y in IS_YEARS if yr.get(y, 0.0) > 0)
    rob = {k: metrics(v, *IS_WINDOW)["sharpe"] for k, v in robustness.items()}
    g = {
        "G1 IS Sharpe > 1.0": is_m["sharpe"] > 1.0,
        "G2 IS MaxDD < 25%": is_m["max_dd"] < 0.25,
        "G3 >=4/5 positive years": pos_years >= 4,
        "G4 Holdout Sharpe > 0.5 and MaxDD < 25%": ho_m["sharpe"] > 0.5 and ho_m["max_dd"] < 0.25,
        "G5 Robustness (IS Sharpe >= 0 for all +/-50%)": all(s >= 0 for s in rob.values()) if rob else None,
    }
    passed = all(v for v in g.values() if v is not None)
    return {"is": is_m, "holdout": ho_m, "years": yr, "pos_years": pos_years, "robustness": rob, "gates": g, "pass": passed}


def half_and_half(n: int) -> tuple[int, int]:
    """+/-50% of an integer lookback, rounded half up."""
    return int(np.floor(n * 0.5 + 0.5)), int(np.floor(n * 1.5 + 0.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="re-download data")
    ap.add_argument("--json", type=Path, help="write full results as JSON")
    args = ap.parse_args()

    m = load_market(args.refresh)

    lo21, hi21 = half_and_half(21)
    lo100, hi100 = half_and_half(100)
    lo30, hi30 = half_and_half(30)
    lo28, hi28 = half_and_half(28)

    runs = {
        "S1 carry": (
            strat_carry(m),
            {f"funding lookback {lo21} prints": strat_carry(m, lookback_prints=lo21),
             f"funding lookback {hi21} prints": strat_carry(m, lookback_prints=hi21)},
        ),
        "S2 trend+vol": (
            strat_trend_vol(m),
            {f"SMA {lo100}": strat_trend_vol(m, sma_len=lo100), f"SMA {hi100}": strat_trend_vol(m, sma_len=hi100),
             f"vol {lo30}": strat_trend_vol(m, vol_len=lo30), f"vol {hi30}": strat_trend_vol(m, vol_len=hi30)},
        ),
        "S3 XS momentum": (
            strat_xs_momentum(m),
            {f"SMA {lo100}": strat_xs_momentum(m, sma_len=lo100), f"SMA {hi100}": strat_xs_momentum(m, sma_len=hi100),
             f"mom {lo28}": strat_xs_momentum(m, mom_len=lo28), f"mom {hi28}": strat_xs_momentum(m, mom_len=hi28)},
        ),
        "BTC buy&hold": (strat_buy_hold_btc(m), {}),
    }

    report = {}
    for name, (res, rob) in runs.items():
        report[name] = gate(res, rob)
        report[name]["full"] = metrics(res, "2019-01-01", "2026-10-31")

    # data coverage
    first = {c: str(m.spot[c].first_valid_index().date()) for c in COINS}
    elig_date = {c: str(eligible(m.spot)[c].idxmax().date()) for c in COINS}
    fund_first = {c: (str(m.funding[c]["time"].min().date()) if len(m.funding[c]) else None) for c in COINS}
    gaps = {c: int(m.spot[c].loc[m.spot[c].first_valid_index():].isna().sum()) for c in COINS}
    carry = runs["S1 carry"][0]
    in_pos = {c: float(window(s.astype(float), *IS_WINDOW).mean()) for c, s in carry.extra["in_position"].items()}
    report["_data"] = {"spot_first": first, "eligible_from": elig_date, "funding_first": fund_first,
                       "spot_missing_days_after_listing": gaps,
                       "perp_first": {c: str(m.perp[c].first_valid_index().date()) for c in m.perp},
                       "carry_time_in_position_IS": in_pos}

    pct = lambda x: f"{x * 100:7.1f}%"  # noqa: E731
    print(f"{'strategy':16} {'IS CAGR':>8} {'IS Shp':>7} {'IS DD':>8} {'HO CAGR':>8} {'HO Shp':>7} {'HO DD':>8}  verdict")
    for name, r in report.items():
        if name.startswith("_"):
            continue
        i, h = r["is"], r["holdout"]
        print(f"{name:16} {pct(i['cagr'])} {i['sharpe']:7.2f} {pct(i['max_dd'])} {pct(h['cagr'])} {h['sharpe']:7.2f} "
              f"{pct(h['max_dd'])}  {'PASS' if r['pass'] else 'FAIL'}")
    for name, r in report.items():
        if name.startswith("_"):
            continue
        print(f"\n{name}")
        print("  years: " + ", ".join(f"{y}: {v * 100:.1f}%" for y, v in r["years"].items()))
        print("  robustness IS Sharpe: " + ", ".join(f"{k}: {v:.2f}" for k, v in r["robustness"].items()))
        for k, v in r["gates"].items():
            print(f"  {k}: {'n/a' if v is None else ('PASS' if v else 'FAIL')}")
    print("\ndata:", json.dumps(report["_data"], indent=1))

    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
