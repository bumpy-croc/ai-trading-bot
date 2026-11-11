"""Recent daily candles (and funding) for every live Binance USDT perpetual, from the public archive.

data.build() reads complete monthly files up to a fixed END for the research cache. The forward paper
test (docs/research/experiments/2026-11-11_slow-adaptive-trend-prereg.md, Stage B) needs the day that
has just closed, so this module reads complete months from the monthly files and the current month
day by day. The archive publishes a day's file at about 09:00 UTC the next morning.

Used by tools/paper_trend_recorder.py (candles only) and paper_score.py (candles and funding).
"""

from __future__ import annotations

import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod  # noqa: E402

DAILY = "data/futures/um/daily/klines"
MONTHLY = "data/futures/um/monthly/klines"
FUNDING_MONTHLY = "data/futures/um/monthly/fundingRate"
PROBE_SYMBOL = "BTCUSDT"
EXCHANGE_INFO = ("https://www.binance.com/fapi/v1/exchangeInfo", "https://fapi.binance.com/fapi/v1/exchangeInfo")
KNOWN_FILE = Path(__file__).resolve().parent / "known_symbols.json"  # every USDT perp in the archive on 2026-11-10
ATTEMPTS = 4
FIELDS = ("open", "high", "low", "close", "quote_volume")


def _get(key: str) -> pd.DataFrame | None:
    """One archive zip as a frame, or None when the archive has no such file (404)."""
    last: Exception | None = None
    for attempt in range(ATTEMPTS):
        try:
            return data_mod._read_zip_csv(key)
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                return None
            last = exc
        except requests.RequestException as exc:
            last = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"archive unreachable for {key}: {last}")


def _candles(key: str) -> pd.DataFrame | None:
    df = _get(key)
    if df is None:
        return None
    df = df.iloc[:, :8]
    df.columns = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume"]
    df["date"] = data_mod._to_ts(df["open_time"]).dt.normalize()
    return df.set_index("date")[list(FIELDS)].astype(float)


def _months(first: pd.Timestamp, last: pd.Timestamp) -> pd.PeriodIndex:
    return pd.period_range(first.tz_localize(None), last.tz_localize(None), freq="M")


def day_published(day: pd.Timestamp) -> bool:
    """True once the archive holds the daily candle for `day` (checked on BTC)."""
    key = f"{DAILY}/{PROBE_SYMBOL}/1d/{PROBE_SYMBOL}-1d-{day:%Y-%m-%d}.zip"
    for attempt in range(ATTEMPTS):
        try:
            code = data_mod._session.head(f"{data_mod.FILES}/{key}", timeout=60).status_code
        except requests.RequestException:
            time.sleep(2**attempt)
            continue
        if code in (200, 404):
            return code == 200
        time.sleep(2**attempt)
    raise RuntimeError("archive unreachable")


def list_symbols() -> list[str]:
    prefixes, _ = data_mod._list(f"{DAILY}/")
    names = (p.rstrip("/").split("/")[-1] for p in prefixes)
    return sorted(s for s in names if re.fullmatch(r"[A-Z0-9]+USDT", s))


def fetch_symbol(symbol: str, day: pd.Timestamp, history_days: int) -> pd.DataFrame:
    """Daily candles for the `history_days` up to and including `day`. A contract that was settled inside
    the window is kept up to its last file, so past universes and past positions can still be rebuilt."""
    latest = _candles(f"{DAILY}/{symbol}/1d/{symbol}-1d-{day:%Y-%m-%d}.zip")
    start = day - pd.Timedelta(days=history_days)
    frames = [] if latest is None else [latest]
    months = _months(start, day)
    recent = months[-2:]  # the month in progress, and the one before it (its monthly file may not be out yet)
    for month in months:
        first = max(start, month.start_time.tz_localize("UTC"))
        last = min(day - pd.Timedelta(days=1), month.end_time.normalize().tz_localize("UTC"))
        monthly = None
        if month != months[-1]:
            monthly = _candles(f"{MONTHLY}/{symbol}/1d/{symbol}-1d-{month}.zip")
            # a month that has only just ended has no monthly file yet; older gaps mean "not listed then"
            if monthly is None and month not in recent:
                continue
        if monthly is not None:
            frames.append(monthly)
            continue
        for d in pd.date_range(first, last, freq="D"):
            one = _candles(f"{DAILY}/{symbol}/1d/{symbol}-1d-{d:%Y-%m-%d}.zip")
            if one is not None:
                frames.append(one)
    if not frames:
        return pd.DataFrame(columns=list(FIELDS))
    out = pd.concat(frames)
    out = out[~out.index.duplicated()].sort_index()
    return out.loc[start:day]


def fetch_market(day: pd.Timestamp, history_days: int, symbols: list[str] | None = None,
                 workers: int = data_mod.WORKERS) -> tuple[data_mod.Market, dict]:
    """Wide daily frames for every perp with a candle in the window. Funding is set to zero and there is no
    spot leg: callers that need funding add it with fetch_funding()."""
    symbols = symbols if symbols is not None else list_symbols()
    with ThreadPoolExecutor(workers) as pool:
        got = list(pool.map(lambda s: fetch_symbol(s, day, history_days), symbols))
    live = {s: df for s, df in zip(symbols, got, strict=True) if len(df)}
    idx = pd.date_range(day - pd.Timedelta(days=history_days), day, freq="D")
    frames = {f: pd.DataFrame({s: df[f] for s, df in live.items()}).reindex(idx) for f in FIELDS}
    # Same cleaning as data.repair(): a settled (delisted) contract keeps printing flat zero-volume candles,
    # so every series is cut after its last day with real volume; short interior gaps are carried forward.
    traded = frames["quote_volume"].fillna(0.0) > 0
    alive = traded.iloc[::-1].cummax().iloc[::-1]
    frames = {f: v.where(alive) for f, v in frames.items()}
    listed = frames["close"].ffill().notna() & frames["close"].bfill().notna()
    gap = frames["close"].isna() & listed
    for f in ("open", "high", "low", "close"):
        frames[f] = frames[f].ffill(limit=5).where(listed)
    frames["quote_volume"] = frames["quote_volume"].where(~gap, 0.0)
    close = frames["close"]
    info = {
        "symbols_in_archive": len(symbols),
        "perps_with_data": int(close.loc[day].notna().sum()),
        "perps_previous_day": int(close.loc[day - pd.Timedelta(days=1)].notna().sum()),
        "missing_candles_carried_forward": int((gap & close.notna()).sum().sum()),
    }
    market = data_mod.Market(frames["open"], frames["high"], frames["low"], close, frames["quote_volume"],
                             close * 0.0, close * np.nan)
    return market, info


def _exchange_info() -> dict | None:
    """The exchange's own contract list, or None when no host answers (GitHub's runners are geo-blocked
    from some of them)."""
    for url in EXCHANGE_INFO:
        try:
            resp = data_mod._session.get(url, timeout=30)
        except requests.RequestException:
            continue
        if resp.status_code == 200:
            body = resp.json()
            if isinstance(body, dict) and len(body.get("symbols", [])) > 100:
                return body
    return None


def extra_exclusions(symbols: list[str]) -> tuple[list[str], str]:
    """Symbols to keep out of the universe on top of the committed non-crypto list, and where the answer
    came from. The committed list (tokenised stocks, metals, index products) was made on 2026-11-10.

    "exchange": the exchange's contract list was reachable; anything it classifies as not a crypto-asset
    perpetual is excluded, and so is a contract listed after 2026-11-10 that it no longer lists.
    "fallback": it was not reachable, so every contract listed after 2026-11-10 stays out until it can be
    classified. That keeps new stock and commodity perps out at the price of also keeping new coins out."""
    committed = set(json.loads(data_mod.NON_CRYPTO_FILE.read_text()))
    known = set(json.loads(KNOWN_FILE.read_text()))
    new = {s for s in symbols if s not in known}
    info = _exchange_info()
    if info is None:
        return sorted(new), "fallback"
    listed = {x["symbol"] for x in info["symbols"]}
    non_crypto = {x["symbol"] for x in info["symbols"] if x.get("quoteAsset") == "USDT"
                  and (x.get("contractType") != "PERPETUAL" or x.get("underlyingType") != "COIN")}
    extra = (non_crypto - committed) | (new - listed)
    return sorted(extra & set(symbols)), "exchange"


def _funding_rest(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    out: dict[pd.Timestamp, float] = {}
    cursor = int(start.timestamp() * 1000)
    stop = int((end + pd.Timedelta(days=1, hours=1)).timestamp() * 1000)
    while cursor < stop:
        rows = data_mod._rest("fundingRate", {"symbol": symbol, "startTime": cursor, "endTime": stop, "limit": 1000})
        if not rows:
            break
        for r in rows:
            out[pd.Timestamp(int(r["fundingTime"]), unit="ms", tz="UTC")] = float(r["fundingRate"])
        cursor = int(rows[-1]["fundingTime"]) + 1
        if len(rows) < 1000:
            break
    return pd.Series(out, dtype=float).sort_index()


def fetch_funding(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp,
                  workers: int = data_mod.WORKERS) -> tuple[pd.DataFrame, dict]:
    """Daily funding (sum of the prints settled during each UTC day; long pays if positive), NaN where unknown.
    Complete months come from the archive; months the archive does not have yet come from the exchange API."""
    months = _months(start, end)

    def one(symbol: str) -> tuple[pd.Series, int]:
        parts, from_api = [], 0
        for month in months:
            df = _get(f"{FUNDING_MONTHLY}/{symbol}/{symbol}-fundingRate-{month}.zip")
            if df is not None:
                df = df.iloc[:, :3]
                df.columns = ["calc_time", "interval_hours", "rate"]
                parts.append(pd.Series(df["rate"].astype(float).to_numpy(), index=data_mod._to_ts(df["calc_time"])))
            else:
                a = max(start, month.start_time.tz_localize("UTC"))
                b = min(end, month.end_time.normalize().tz_localize("UTC"))
                s = _funding_rest(symbol, a, b)
                from_api += int(len(s) > 0)
                parts.append(s)
        prints = pd.concat(parts) if parts else pd.Series(dtype=float)
        prints = prints[~prints.index.duplicated()].sort_index()
        if prints.empty:
            return pd.Series(dtype=float), from_api
        # a print stamped 00:00 on day d+1 settles at day d's close (same rule as data.build)
        return prints.groupby((prints.index - pd.Timedelta(hours=1)).floor("D")).sum(), from_api

    with ThreadPoolExecutor(workers) as pool:
        got = list(pool.map(one, symbols))
    idx = pd.date_range(start, end, freq="D")
    frame = pd.DataFrame({s: g[0] for s, g in zip(symbols, got, strict=True)}).reindex(idx)
    return frame, {"symbols": len(symbols), "symbol_months_from_api": int(sum(g[1] for g in got))}
