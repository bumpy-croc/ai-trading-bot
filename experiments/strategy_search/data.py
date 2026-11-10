"""Data layer for the wide strategy search
(docs/research/experiments/2026-11-10_strategy-search-prereg.md).

Source: the public Binance archive at data.binance.vision (reachable where the REST API is
geo-blocked). It keeps delisted symbols, so the universe is point-in-time rather than
"today's survivors".

Downloads, for every USDT-margined perpetual that ever listed:
  - daily perp klines (open, high, low, close, quote volume)
  - funding-rate history
  - daily spot klines for the matching spot pair, where one exists

and caches them as pickled wide frames under experiments/strategy_search/data/.

Usage: python experiments/strategy_search/data.py [--refresh]
"""

from __future__ import annotations

import argparse
import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
DATA_DIR = Path(__file__).resolve().parent / "data"
START = pd.Timestamp("2019-09-01", tz="UTC")
END = pd.Timestamp("2026-10-31", tz="UTC")
WORKERS = 24

_session = requests.Session()
_adapter = requests.adapters.HTTPAdapter(pool_connections=WORKERS, pool_maxsize=WORKERS, max_retries=3)
_session.mount("https://", _adapter)


def _list(prefix: str) -> tuple[list[str], list[str]]:
    """Return (sub-prefixes, keys) directly under an archive prefix."""
    prefixes: list[str] = []
    keys: list[str] = []
    marker = ""
    while True:
        resp = _session.get(BUCKET, params={"delimiter": "/", "prefix": prefix, "marker": marker}, timeout=60)
        resp.raise_for_status()
        root = ElementTree.fromstring(resp.content)
        prefixes += [e.text for e in root.findall("s3:CommonPrefixes/s3:Prefix", NS)]
        keys += [e.text for e in root.findall("s3:Contents/s3:Key", NS)]
        if root.findtext("s3:IsTruncated", namespaces=NS) != "true":
            return prefixes, keys
        marker = root.findtext("s3:NextMarker", namespaces=NS) or (keys[-1] if keys else prefixes[-1])


def _zip_keys(prefix: str) -> list[str]:
    return [k for k in _list(prefix)[1] if k.endswith(".zip")]


def _read_zip_csv(key: str) -> pd.DataFrame:
    resp = _session.get(f"{FILES}/{key}", timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        raw = zf.read(zf.namelist()[0])
    first = raw[:1]
    header = None if first.isdigit() else 0
    return pd.read_csv(io.BytesIO(raw), header=header)


def _to_ts(ms: pd.Series) -> pd.Series:
    ms = ms.astype("int64")
    # spot files switched to microseconds in 2025
    ms = ms.where(ms < 10**14, ms // 1000)
    return pd.to_datetime(ms, unit="ms", utc=True)


def _klines(prefix: str) -> pd.DataFrame:
    frames = []
    for key in _zip_keys(prefix):
        df = _read_zip_csv(key).iloc[:, :8]
        df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["date"] = _to_ts(out["open_time"]).dt.normalize()
    out = out.drop_duplicates("date").set_index("date").sort_index()
    return out[["open", "high", "low", "close", "quote_volume"]].astype(float)


def _funding(symbol: str) -> pd.Series:
    frames = []
    for key in _zip_keys(f"data/futures/um/monthly/fundingRate/{symbol}/"):
        df = _read_zip_csv(key).iloc[:, :3]
        df.columns = ["calc_time", "interval_hours", "rate"]
        frames.append(df)
    if not frames:
        return pd.Series(dtype=float)
    out = pd.concat(frames, ignore_index=True)
    out["time"] = _to_ts(out["calc_time"])
    return out.drop_duplicates("time").set_index("time")["rate"].astype(float).sort_index()


def spot_symbol(perp: str) -> str:
    """Spot pair matching a perp symbol (1000SHIBUSDT -> SHIBUSDT)."""
    return re.sub(r"^1000+", "", perp)


def _fetch_symbol(symbol: str) -> dict:
    out = {"symbol": symbol}
    try:
        out["perp"] = _klines(f"data/futures/um/monthly/klines/{symbol}/1d/")
        out["funding"] = _funding(symbol)
        out["spot"] = _klines(f"data/spot/monthly/klines/{spot_symbol(symbol)}/1d/")
    except Exception as exc:  # noqa: BLE001 - recorded and reported, never silently dropped
        out["error"] = repr(exc)
    return out


@dataclass
class Market:
    """Wide daily frames, index = UTC date, columns = perp symbol."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    qvol: pd.DataFrame  # perp quote volume (USDT)
    funding: pd.DataFrame  # sum of funding prints settled during the UTC day (long pays if > 0)
    spot_close: pd.DataFrame  # matching spot pair close, NaN where no spot market

    def truncate(self, end: str) -> "Market":
        """Hard cutoff: drop every row after `end`. Used to keep the holdout physically out of reach."""
        cut = pd.Timestamp(end, tz="UTC")
        return Market(**{k: v.loc[:cut] for k, v in self.__dict__.items()})


def list_perp_symbols() -> list[str]:
    prefixes, _ = _list("data/futures/um/monthly/klines/")
    syms = [p.rstrip("/").split("/")[-1] for p in prefixes]
    # USDT-margined perpetuals only: no USDC/BUSD quotes, no dated contracts, no index products
    return sorted(s for s in syms if re.fullmatch(r"[A-Z0-9]+USDT", s))


def build(refresh: bool = False) -> Market:
    DATA_DIR.mkdir(exist_ok=True)
    (DATA_DIR / ".gitignore").write_text("*\n")
    cache = DATA_DIR / "market.pkl"
    if cache.exists() and not refresh:
        return Market(**pd.read_pickle(cache))

    symbols = list_perp_symbols()
    print(f"{len(symbols)} USDT perp symbols in the archive; downloading...", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(_fetch_symbol, symbols))
    errors = {r["symbol"]: r["error"] for r in results if "error" in r}
    if errors:
        raise RuntimeError(f"download failed for {len(errors)} symbols: {dict(list(errors.items())[:5])}")

    idx = pd.date_range(START, END, freq="D")
    cols: dict[str, dict[str, pd.Series]] = {k: {} for k in ("open", "high", "low", "close", "qvol", "funding", "spot_close")}
    for r in results:
        perp = r["perp"]
        if perp.empty:
            continue
        s = r["symbol"]
        cols["open"][s] = perp["open"]
        cols["high"][s] = perp["high"]
        cols["low"][s] = perp["low"]
        cols["close"][s] = perp["close"]
        cols["qvol"][s] = perp["quote_volume"]
        f = r["funding"]
        if len(f):
            # a print stamped 00:00 on day d+1 settles at day d's close
            day = (f.index - pd.Timedelta(hours=1)).floor("D")
            cols["funding"][s] = f.groupby(day).sum()
        if not r["spot"].empty:
            cols["spot_close"][s] = r["spot"]["close"]
    frames = {k: pd.DataFrame(v).reindex(idx) for k, v in cols.items()}
    for k in ("funding", "spot_close"):
        frames[k] = frames[k].reindex(columns=frames["close"].columns)
    pd.to_pickle(frames, cache)
    return Market(**frames)


def coverage(m: Market) -> pd.DataFrame:
    first = m.close.apply(pd.Series.first_valid_index)
    last = m.close.apply(pd.Series.last_valid_index)
    gaps = pd.Series({c: int(m.close[c].loc[first[c] : last[c]].isna().sum()) for c in m.close.columns})
    fund_days = m.funding.notna().sum()
    return pd.DataFrame(
        {
            "first": first,
            "last": last,
            "days": m.close.notna().sum(),
            "gap_days": gaps,
            "funding_days": fund_days,
            "has_spot": m.spot_close.notna().any(),
        }
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    market = build(args.refresh)
    cov = coverage(market)
    print(cov.describe(include="all").to_string())
    print("symbols:", len(cov), "| delisted before END:", int((cov["last"] < END - pd.Timedelta(days=5)).sum()))
    print("listed per year:", cov["first"].dt.year.value_counts().sort_index().to_dict())
    print("symbols with gaps:", cov[cov.gap_days > 0][["first", "last", "gap_days"]].head(20).to_string())
    print("funding missing entirely:", int((cov.funding_days == 0).sum()))
    print("NaN check close>0:", bool((market.close.stack() > 0).all()))
