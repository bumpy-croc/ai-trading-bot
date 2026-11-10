"""Shared data loaders, cost tables, metrics and gates for the other-markets bake-off.

Preregistration: docs/research/experiments/2026-11-10_other-markets-bakeoff-prereg.md
Used by experiments/fx_predictability.py (Part A) and experiments/markets_bakeoff.py (Part B).

Self-contained (pandas/numpy/requests). Everything downloaded is cached under
experiments/markets_data/ (git-ignored, re-downloadable).
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "markets_data"
END = pd.Timestamp("2026-10-31")
HOLDOUT = ("2015-01-01", "2026-10-31")
RECENT = ("2022-01-01", "2026-10-31")
N_GATED_CANDIDATES = 15  # 6 in Part A + 9 in Part B, fixed by the preregistration
BONFERRONI_Z = 2.7131  # one-sided 5% / 15 -> norm.ppf(1 - 0.05 / 15)
UA = "Mozilla/5.0"

FX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]
# Per-side cost (half spread + slippage) in bps of notional, by non-USD currency.
FX_COST_BPS = {
    "EUR": 0.6,
    "JPY": 0.7,
    "GBP": 0.9,
    "AUD": 0.9,
    "CAD": 1.0,
    "CHF": 1.0,
    "NZD": 1.2,
    "SEK": 2.5,
    "NOK": 2.5,
}
FX_MARKUP = 0.01  # broker financing markup per year on pair notional
FX_IDLE_MARGIN = 0.25  # share of capital parked at the broker earning nothing
ETF_COST = 0.0005  # per side
CASH_HAIRCUT = 0.0020  # cash earns T-bill minus this, per year
SHORT_BORROW = 0.015  # per year on short ETF notional
CRYPTO_COST_RT = 0.0011  # round trip

G10 = ["USD", "EUR", "JPY", "GBP", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"]
BIS_AREA = {
    "USD": "US",
    "EUR": "XM",
    "JPY": "JP",
    "GBP": "GB",
    "CHF": "CH",
    "AUD": "AU",
    "NZD": "NZ",
    "CAD": "CA",
    "SEK": "SE",
    "NOK": "NO",
}


def pair_ccy(pair: str) -> str:
    """The non-USD currency of a USD pair."""
    return pair[:3] if pair[3:] == "USD" else pair[3:]


# --------------------------------------------------------------------------- http


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = UA
    return s


def _get(url: str, session: requests.Session | None = None, **kwargs) -> requests.Response:
    s = session or _session()
    last: Exception | None = None
    for attempt in range(5):
        try:
            resp = s.get(url, timeout=120, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code == 404:
                return resp
            last = RuntimeError(f"HTTP {resp.status_code} for {url}")
        except requests.RequestException as exc:  # network hiccup: retry
            last = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"giving up on {url}: {last}")


# --------------------------------------------------------------------------- histdata (FX 1-minute)


def _histdata_zip(pair: str, period: str, session: requests.Session) -> bytes | None:
    """period is 'YYYY' or 'YYYY/M'. Returns zip bytes, or None if the period is not offered."""
    page = f"https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/{pair.lower()}/{period}"
    resp = _get(page, session)
    fields = dict(
        re.findall(
            r'name="(tk|date|datemonth|platform|timeframe|fxpair)"[^>]*value="([^"]*)"', resp.text
        )
    )
    if "tk" not in fields:
        return None
    for attempt in range(5):
        out = session.post(
            "https://www.histdata.com/get.php",
            data=fields,
            headers={"Referer": page, "Origin": "https://www.histdata.com"},
            timeout=180,
        )
        if out.status_code == 200 and out.content[:2] == b"PK":
            return out.content
        time.sleep(2**attempt)
    return None


def _parse_histdata(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(
            z.open(name),
            sep=";",
            header=None,
            names=["ts", "open", "high", "low", "close", "vol"],
            dtype={"ts": str},
        )
    df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    return df.drop(columns="vol").set_index("ts").sort_index()


def load_fx_minute_resampled(pair: str, start_year: int = 2010) -> tuple[pd.DataFrame, dict]:
    """Hourly OHLC bars (fixed EST clock) built from histdata 1-minute bid bars. Cached per pair."""
    out_path = DATA_DIR / "fx_1h" / f"{pair}.csv"
    meta_path = DATA_DIR / "fx_1h" / f"{pair}.json"
    if out_path.exists() and meta_path.exists():
        df = pd.read_csv(out_path, index_col=0, parse_dates=True)
        return df, json.loads(meta_path.read_text())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = DATA_DIR / "histdata_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = _session()
    periods = [str(y) for y in range(start_year, END.year)] + [
        f"{END.year}/{m}" for m in range(1, END.month + 1)
    ]
    frames, missing = [], []
    for period in periods:
        path = raw_dir / f"{pair}_{period.replace('/', '_')}.zip"
        if not path.exists():
            raw = _histdata_zip(pair, period, session)
            if raw is None:
                missing.append(period)
                continue
            path.write_bytes(raw)
            time.sleep(0.5)
        frames.append(_parse_histdata(path.read_bytes()))
    m1 = pd.concat(frames).sort_index()
    m1 = m1[~m1.index.duplicated(keep="first")]
    m1 = m1[m1.index <= END + pd.Timedelta(days=1)]
    # Bad ticks: a 1-minute close-to-close move above 3% is not a real print in a major.
    jump = m1["close"].pct_change().abs() > 0.03
    bad = int(jump.sum())
    m1 = m1[~jump]
    hourly = (
        m1.resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    meta = {
        "minute_bars": int(len(m1)),
        "bad_ticks_dropped": bad,
        "missing_periods": missing,
        "first": str(m1.index[0]),
        "last": str(m1.index[-1]),
    }
    hourly.to_csv(out_path)
    meta_path.write_text(json.dumps(meta))
    return hourly, meta


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily bars cut at 17:00 on the feed's fixed EST clock (the FX day boundary)."""
    shifted = hourly.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=7)  # 17:00 -> 00:00 next day
    daily = (
        shifted.resample("1D")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
    )
    return daily[daily.index.day_of_week < 5]


# --------------------------------------------------------------------------- ECB fixes, BIS policy rates


def load_ecb_usd_crosses() -> pd.DataFrame:
    """USD per one unit of each G10 currency (date index), from ECB euro reference rates."""
    path = DATA_DIR / "ecb_eurofxref_hist.csv"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw = _get("https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip").content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            path.write_bytes(z.read(z.namelist()[0]))
    df = pd.read_csv(path, na_values=["N/A"]).rename(columns={"Date": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df.loc[:END]
    usd_per_eur = df["USD"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["EUR"] = usd_per_eur
    for ccy in G10:
        if ccy in ("USD", "EUR"):
            continue
        out[ccy] = usd_per_eur / df[ccy].astype(float)
    return out.dropna(how="any")


def load_policy_rates() -> pd.DataFrame:
    """Central bank policy rates, percent per year, daily, forward-filled (date index, G10 columns)."""
    path = DATA_DIR / "bis_cbpol_g10.csv"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        raw = _get("https://data.bis.org/static/bulk/WS_CBPOL_csv_flat.zip").content
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            full = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
        area_col = next(c for c in full.columns if c.upper().startswith("REF_AREA"))
        freq_col = next(c for c in full.columns if c.upper().startswith("FREQ"))
        time_col = next(c for c in full.columns if c.upper().startswith("TIME_PERIOD"))
        val_col = next(c for c in full.columns if c.upper().startswith("OBS_VALUE"))
        full["area"] = full[area_col].astype(str).str.split(":").str[0].str.strip()
        full["freq"] = full[freq_col].astype(str).str.split(":").str[0].str.strip()
        keep = full[(full["freq"] == "D") & full["area"].isin(BIS_AREA.values())]
        tidy = keep[["area", time_col, val_col]].rename(columns={time_col: "date", val_col: "rate"})
        tidy.to_csv(path, index=False)
    tidy = pd.read_csv(path, parse_dates=["date"])
    wide = tidy.pivot_table(
        index="date", columns="area", values="rate", aggfunc="last"
    ).sort_index()
    wide = wide.rename(columns={v: k for k, v in BIS_AREA.items()})[G10]
    return wide.ffill().loc["1998-06-01":END]


# --------------------------------------------------------------------------- Yahoo daily


def load_yahoo(symbol: str) -> pd.DataFrame:
    """Daily bars with total-return (adjusted) open and close. Index = trading date."""
    path = DATA_DIR / "yahoo" / f"{symbol.replace('^', '_')}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{requests.utils.quote(symbol)}"
        params = {
            "period1": 631152000,
            "period2": int((END + pd.Timedelta(days=2)).timestamp()),
            "interval": "1d",
            "events": "div,splits",
        }
        res = _get(url, params=params).json()["chart"]["result"][0]
        quote = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", quote["close"])
        df = pd.DataFrame(
            {
                "ts": res["timestamp"],
                "open": quote["open"],
                "high": quote["high"],
                "low": quote["low"],
                "close": quote["close"],
                "adjclose": adj,
            }
        )
        offset = res["meta"].get("gmtoffset", 0)
        df["date"] = pd.to_datetime(df["ts"] + offset, unit="s").dt.normalize()
        df.drop(columns="ts").dropna(subset=["close"]).drop_duplicates("date").to_csv(
            path, index=False
        )
        time.sleep(1.0)  # be polite to a free endpoint
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index().loc[:END]
    ratio = df["adjclose"] / df["close"]
    for col in ("open", "high", "low"):
        df[f"adj{col}"] = df[col] * ratio
    return df


def load_tbill_daily(index: pd.DatetimeIndex) -> pd.Series:
    """T-bill return accrued over each interval ending at index[i] (13-week yield, actual/365)."""
    irx = load_yahoo("^IRX")["close"] / 100.0
    days = pd.Series(index, index=index).diff().dt.days.fillna(1).clip(lower=1)
    rate = irx.reindex(index.union(irx.index)).ffill().reindex(index).shift(1).bfill()
    return (rate * days / 365.0).rename("rf")


# --------------------------------------------------------------------------- Binance (crypto reference)


def load_binance_1h(symbol: str, start: str = "2018-01") -> pd.DataFrame:
    path = DATA_DIR / "binance_1h" / f"{symbol}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for month in pd.period_range(start, END.strftime("%Y-%m"), freq="M"):
            url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1h/{symbol}-1h-{month}.zip"
            resp = _get(url)
            if resp.status_code != 200:
                continue
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                part = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 1, 2, 3, 4])
            part = part[pd.to_numeric(part[0], errors="coerce").notna()].astype(float)
            part.columns = ["t", "open", "high", "low", "close"]
            unit_ms = part["t"].where(
                part["t"] < 1e14, part["t"] / 1000.0
            )  # 2025+ files use microseconds
            part["ts"] = pd.to_datetime(unit_ms, unit="ms")
            frames.append(part.drop(columns="t"))
        pd.concat(frames).drop_duplicates("ts").sort_values("ts").to_csv(path, index=False)
    df = pd.read_csv(path)
    df["ts"] = pd.to_datetime(df["ts"], format="ISO8601")
    return df.set_index("ts").sort_index()


# --------------------------------------------------------------------------- metrics and gates


def perf(excess: pd.Series, periods_per_year: float) -> dict:
    """Stats on a series of per-period excess returns (already net of costs and the cash rate)."""
    x = excess.dropna()
    if len(x) < 2 or x.std() == 0:
        return {
            "n": int(len(x)),
            "years": 0.0,
            "ann_excess": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "tstat": 0.0,
            "max_dd": 0.0,
        }
    years = len(x) / periods_per_year
    sharpe = x.mean() / x.std() * math.sqrt(periods_per_year)
    equity = (1.0 + x).cumprod()
    dd = float((1.0 - equity / equity.cummax()).max())
    return {
        "n": int(len(x)),
        "years": round(years, 2),
        "ann_excess": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "vol": float(x.std() * math.sqrt(periods_per_year)),
        "sharpe": float(sharpe),
        "tstat": float(sharpe * math.sqrt(years)),
        "max_dd": dd,
    }


def total_stats(total: pd.Series, periods_per_year: float) -> dict:
    """CAGR and drawdown of total (not excess) returns: what the account balance would show."""
    x = total.dropna()
    years = len(x) / periods_per_year
    equity = (1.0 + x).cumprod()
    return {
        "cagr": float(equity.iloc[-1] ** (1.0 / years) - 1.0),
        "max_dd_total": float((1.0 - equity / equity.cummax()).max()),
    }


def alpha_vs(excess: pd.Series, bench_excess: pd.Series, periods_per_year: float) -> dict:
    """OLS intercept of excess on benchmark excess (annualised) and its t-statistic."""
    df = pd.concat([excess.rename("y"), bench_excess.rename("x")], axis=1).dropna()
    if len(df) < 30:
        return {"alpha": 0.0, "alpha_t": 0.0, "beta": 0.0}
    x = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    y = df["y"].to_numpy()
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    sigma2 = resid @ resid / (len(df) - 2)
    cov = sigma2 * np.linalg.inv(x.T @ x)
    return {
        "alpha": float(coef[0] * periods_per_year),
        "alpha_t": float(coef[0] / math.sqrt(cov[0, 0])),
        "beta": float(coef[1]),
    }


def gate(
    excess: pd.Series,
    periods_per_year: float,
    bench_excess: pd.Series | None = None,
    window: tuple[str, str] = HOLDOUT,
) -> dict:
    """Apply the preregistered gates G1-G5 to per-period excess returns."""
    win = excess.loc[window[0] : window[1]]
    stats = perf(win, periods_per_year)
    recent = excess.loc[RECENT[0] : RECENT[1]]
    out = {
        "window": stats,
        "recent_mean_ann": float(recent.mean() * periods_per_year) if len(recent) else 0.0,
        "G1": stats["sharpe"] >= 0.5,
        "G2": stats["max_dd"] <= 0.25,
        "G4": bool(len(recent) and recent.mean() > 0),
        "G5": stats["tstat"] >= BONFERRONI_Z,
    }
    if bench_excess is not None:
        a = alpha_vs(win, bench_excess.loc[window[0] : window[1]], periods_per_year)
        out["alpha"] = a
        out["G3"] = a["alpha"] > 0 and a["alpha_t"] >= 1.65
    base = out["G1"] and out["G2"] and out["G4"] and out.get("G3", True)
    out["verdict"] = "PASS" if base and out["G5"] else ("PROVISIONAL" if base else "FAIL")
    return out


def yearly(returns: pd.Series) -> dict:
    return {
        str(int(y)): round(float((1.0 + g).prod() - 1.0), 4)
        for y, g in returns.dropna().groupby(returns.dropna().index.year)
    }
