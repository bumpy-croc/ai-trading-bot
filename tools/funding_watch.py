"""Funding-rate watcher: alert when BTC/ETH perp funding makes carry worth building.

Context (log.md D-2026-11-10-02): the 2026-11-10 bake-off (#1315) found funding
carry (spot long + perp short) passes every in-sample gate but earned only ~0.9%/yr
in the 2025-26 holdout, because annualised funding fell from ~14-17% to ~4%. That
is below idle stablecoin yield, so the Board chose to stop and watch. This script
is the watch: it reads the trailing funding history of BTCUSDT and ETHUSDT perps
and reports when the trailing average climbs back above the threshold
(default 10%/yr), which is roughly where carry beats idle USDT after costs.

Standalone on purpose (stdlib + requests only): it runs on a bare GitHub Actions
runner without the bot's dependencies or database.

Sources are tried in order, because Binance geo-blocks US cloud runners:
Binance USD-M -> OKX swap -> Hyperliquid. Every source is normalised to
"annualised rate" so the threshold means the same thing on each.

Exit codes: 0 = below threshold, 10 = above threshold (alert), 2 = no source
reachable for some symbol (the watcher itself is blind — also worth a look).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import requests  # type: ignore[import-untyped]  # types-requests not installed

HOURS_PER_YEAR = 24 * 365
HTTP_TIMEOUT_S = 15
EXIT_OK = 0
EXIT_BLIND = 2
EXIT_ALERT = 10

# symbol -> per-source instrument name
INSTRUMENTS: dict[str, dict[str, str]] = {
    "BTCUSDT": {"binance": "BTCUSDT", "okx": "BTC-USDT-SWAP", "hyperliquid": "BTC"},
    "ETHUSDT": {"binance": "ETHUSDT", "okx": "ETH-USDT-SWAP", "hyperliquid": "ETH"},
}


@dataclass(frozen=True)
class FundingPoint:
    """One settled funding payment, normalised to an annualised rate."""

    time: datetime
    annualised: float


@dataclass(frozen=True)
class SymbolReading:
    symbol: str
    source: str
    samples: int
    mean_annualised: float
    latest_annualised: float


def _annualise(rate: float, interval_hours: float) -> float:
    return rate * (HOURS_PER_YEAR / interval_hours)


def fetch_binance(instrument: str, since: datetime) -> list[FundingPoint]:
    resp = requests.get(
        "https://fapi.binance.com/fapi/v1/fundingRate",
        params={
            "symbol": instrument,
            "startTime": int(since.timestamp() * 1000),
            "limit": 1000,
        },
        timeout=HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    # Binance USD-M BTC/ETH settle every 8h.
    return [
        FundingPoint(
            time=datetime.fromtimestamp(row["fundingTime"] / 1000, tz=UTC),
            annualised=_annualise(float(row["fundingRate"]), 8),
        )
        for row in resp.json()
    ]


def fetch_okx(instrument: str, since: datetime) -> list[FundingPoint]:
    points: list[FundingPoint] = []
    after: str | None = None
    # OKX pages newest-first, 100 rows a page; 7-30 days of 8h funding fits in 1 page.
    for _ in range(5):
        params = {"instId": instrument, "limit": "100"}
        if after:
            params["after"] = after
        resp = requests.get(
            "https://www.okx.com/api/v5/public/funding-rate-history",
            params=params,
            timeout=HTTP_TIMEOUT_S,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("code") != "0":
            raise RuntimeError(f"OKX error: {body.get('msg')}")
        rows = body.get("data", [])
        if not rows:
            break
        for row in rows:
            ts = datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=UTC)
            if ts >= since:
                points.append(
                    FundingPoint(time=ts, annualised=_annualise(float(row["realizedRate"]), 8))
                )
        oldest = datetime.fromtimestamp(int(rows[-1]["fundingTime"]) / 1000, tz=UTC)
        if oldest < since:
            break
        after = rows[-1]["fundingTime"]
    return points


def fetch_hyperliquid(instrument: str, since: datetime) -> list[FundingPoint]:
    resp = requests.post(
        "https://api.hyperliquid.xyz/info",
        json={
            "type": "fundingHistory",
            "coin": instrument,
            "startTime": int(since.timestamp() * 1000),
        },
        timeout=HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    # Hyperliquid settles hourly.
    return [
        FundingPoint(
            time=datetime.fromtimestamp(row["time"] / 1000, tz=UTC),
            annualised=_annualise(float(row["fundingRate"]), 1),
        )
        for row in resp.json()
    ]


Fetcher = Callable[[str, datetime], list[FundingPoint]]
SOURCES: list[tuple[str, Fetcher]] = [
    ("binance", fetch_binance),
    ("okx", fetch_okx),
    ("hyperliquid", fetch_hyperliquid),
]


def read_symbol(
    symbol: str,
    since: datetime,
    sources: Sequence[tuple[str, Fetcher]] = SOURCES,
    min_samples: int = 3,
) -> SymbolReading | None:
    """Return the first source's reading that has enough samples, else None."""
    for name, fetch in sources:
        try:
            points = fetch(INSTRUMENTS[symbol][name], since)
        except Exception as e:  # any network/HTTP/parse failure -> try the next source
            print(f"  {symbol}: {name} unavailable ({type(e).__name__}: {e})", file=sys.stderr)
            continue
        points = sorted((p for p in points if p.time >= since), key=lambda p: p.time)
        if len(points) < min_samples:
            print(f"  {symbol}: {name} returned {len(points)} samples, skipping", file=sys.stderr)
            continue
        mean = sum(p.annualised for p in points) / len(points)
        return SymbolReading(
            symbol=symbol,
            source=name,
            samples=len(points),
            mean_annualised=mean,
            latest_annualised=points[-1].annualised,
        )
    return None


def evaluate(readings: dict[str, SymbolReading | None], threshold: float) -> tuple[int, list[str]]:
    """Decide the exit code and build the human-readable report lines."""
    lines: list[str] = []
    above: list[str] = []
    blind: list[str] = []
    for symbol, r in readings.items():
        if r is None:
            blind.append(symbol)
            lines.append(f"- {symbol}: no data source reachable")
            continue
        flag = "ABOVE" if r.mean_annualised >= threshold else "below"
        lines.append(
            f"- {symbol}: trailing mean {r.mean_annualised:.1%}/yr "
            f"(latest {r.latest_annualised:.1%}/yr, {r.samples} samples via {r.source}) "
            f"-> {flag} {threshold:.0%}"
        )
        if r.mean_annualised >= threshold:
            above.append(symbol)
    if above:
        return EXIT_ALERT, lines
    if blind:
        return EXIT_BLIND, lines
    return EXIT_OK, lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.10, help="annualised, e.g. 0.10")
    parser.add_argument("--days", type=int, default=7, help="trailing window in days")
    parser.add_argument("--json-out", help="write the readings as JSON to this path")
    args = parser.parse_args(argv)

    since = datetime.now(UTC) - timedelta(days=args.days)
    readings = {sym: read_symbol(sym, since) for sym in INSTRUMENTS}
    code, lines = evaluate(readings, args.threshold)

    header = f"Funding watch ({args.days}d trailing, threshold {args.threshold:.0%}/yr)"
    report = "\n".join([header, *lines])
    print(report)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(f"### {header}\n\n" + "\n".join(lines) + "\n")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "exit_code": code,
                    "report": report,
                    "readings": {s: (r.__dict__ if r else None) for s, r in readings.items()},
                },
                fh,
                indent=2,
            )
    return code


if __name__ == "__main__":
    sys.exit(main())
