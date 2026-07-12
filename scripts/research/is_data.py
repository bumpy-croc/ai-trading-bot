#!/usr/bin/env python3
"""Data loading for the LINEAR INPUT-SCREENING experiment (Lane A, Phase 1).

Research script only — not wired into any DataProvider/cache-manager path, per
CODE.md's Backtest-Live Parity rule. Fetches are disk-cached under
scripts/research/.cache/ (gitignored) so repeat runs never re-hit the network.

Sources:
  - ETHUSDT / BTCUSDT 1h OHLCV: the repo's own CachedDataProvider(BinanceProvider())
    (same machinery the training pipeline uses).
  - ETHUSDT perp funding rate: fapi.binance.com/fapi/v1/fundingRate (free, no key).
  - ETHUSDT perp premium index (basis proxy): fapi.binance.com/fapi/v1/premiumIndexKlines
    (free, no key).
  - Fear & Greed index: api.alternative.me/fng/ (free, no key).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

CACHE_DIR = Path(__file__).parent / ".cache"
CACHE_DIR.mkdir(exist_ok=True)

FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
FNG_URL = "https://api.alternative.me/fng/"


def _project_root() -> Path:
    # scripts/research/is_data.py -> repo root is two parents up
    return Path(__file__).resolve().parents[2]


def load_ohlcv(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Load OHLCV via the repo's own CachedDataProvider (reuses local cache/market_data)."""
    import sys

    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.data_providers.binance_provider import BinanceProvider
    from src.data_providers.cached_data_provider import CachedDataProvider

    provider = CachedDataProvider(BinanceProvider())
    df = provider.get_historical_data(symbol, timeframe, start, end)
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        # Normalize to a tz-aware UTC DatetimeIndex regardless of what column/index
        # the provider returns timestamps in.
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
        elif "open_time" in df.columns:
            df = df.set_index(pd.to_datetime(df["open_time"], utc=True))
        else:
            df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize(UTC)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def _paginate_binance(url: str, params: dict, time_key: str, start_ms: int, end_ms: int) -> list:
    out: list = []
    cursor = start_ms
    while cursor < end_ms:
        p = dict(params)
        p["startTime"] = cursor
        p["endTime"] = end_ms
        p["limit"] = 1000
        resp = requests.get(url, params=p, timeout=(5, 30))
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        last_ts = batch[-1][time_key] if isinstance(batch[-1], dict) else batch[-1][0]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return out


def load_funding_rate(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Full ETHUSDT perp funding-rate history, disk-cached."""
    cache_file = CACHE_DIR / f"funding_{symbol}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
    else:
        start_ms = int(datetime(2019, 1, 1, tzinfo=UTC).timestamp() * 1000)
        end_ms = int(datetime.now(UTC).timestamp() * 1000)
        records = _paginate_binance(
            FUNDING_URL, {"symbol": symbol}, "fundingTime", start_ms, end_ms
        )
        df = pd.DataFrame(records)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
        df["fundingRate"] = df["fundingRate"].astype(float)
        # markPrice is "" in some pre-2021 records; keep as NaN, never coerce to 0.0
        df["markPrice"] = pd.to_numeric(df["markPrice"], errors="coerce")
        df = df.drop_duplicates(subset="fundingTime").sort_values("fundingTime")
        df.to_parquet(cache_file)
    df = df.set_index("fundingTime")
    # start/end are accepted for API-shape consistency with the other loaders; this
    # loader always returns the full cached history and lets the caller align/forward-fill,
    # since funding-rate feature construction needs the pre-fold-cutoff tail for lookback.
    del start, end
    return df


def load_premium_index(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Full ETHUSDT premium-index (basis proxy) klines, 1h, disk-cached."""
    cache_file = CACHE_DIR / f"premium_{symbol}.parquet"
    if cache_file.exists():
        df = pd.read_parquet(cache_file)
        return df
    start_ms = int(datetime(2019, 1, 1, tzinfo=UTC).timestamp() * 1000)
    end_ms = int(datetime.now(UTC).timestamp() * 1000)
    out = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            PREMIUM_URL,
            params={
                "symbol": symbol,
                "interval": "1h",
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
            timeout=(5, 30),
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        last_open = batch[-1][0]
        if last_open <= cursor:
            break
        cursor = last_open + 3600_000  # advance one hour past last open time
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "ignore1",
        "close_time",
        "ignore2",
        "ignore3",
        "ignore4",
        "ignore5",
        "ignore6",
    ]
    df = pd.DataFrame(out, columns=cols[: len(out[0])] if out else cols)
    if df.empty:
        raise RuntimeError(f"No premium-index data returned for {symbol}")
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close"):
        df[c] = df[c].astype(float)
    df = df.drop_duplicates(subset="open_time").sort_values("open_time")
    df = df.set_index("open_time")[["open", "high", "low", "close"]]
    df.to_parquet(cache_file)
    return df


def load_fear_greed() -> pd.DataFrame:
    """Full Fear & Greed daily history, disk-cached."""
    cache_file = CACHE_DIR / "feargreed.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    resp = requests.get(FNG_URL, params={"limit": 0, "format": "json"}, timeout=(5, 30))
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["value"] = df["value"].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")
    df = df.set_index("timestamp")[["value"]]
    df.to_parquet(cache_file)
    return df


if __name__ == "__main__":
    # Smoke check: pull everything once so subsequent runs hit the disk cache.
    fr = load_funding_rate("ETHUSDT", datetime(2019, 1, 1, tzinfo=UTC), datetime.now(UTC))
    print(f"funding rate: {len(fr)} records, {fr.index.min()} -> {fr.index.max()}")
    pi = load_premium_index("ETHUSDT", datetime(2019, 1, 1, tzinfo=UTC), datetime.now(UTC))
    print(f"premium index: {len(pi)} records, {pi.index.min()} -> {pi.index.max()}")
    fg = load_fear_greed()
    print(f"fear&greed: {len(fg)} records, {fg.index.min()} -> {fg.index.max()}")
