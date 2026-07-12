#!/usr/bin/env python3
"""Proof-of-obtainability: pull ETHUSDT perpetual funding-rate history from Binance's
public futures API (no API key required) and report exact coverage.

Usage: python scripts/research/pull_binance_funding_rate.py [--symbol ETHUSDT] [--months 1]

This is a research script, not production code. It does not use src/engines/shared/ and
must not be treated as backtest-ready until wired through a real DataProvider with the
same caching/retry/parity guarantees as the OHLCV path.
"""
import argparse
import csv
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/fundingRate"


def fetch_funding_rate(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    out = []
    cursor = start_ms
    while cursor < end_ms:
        resp = requests.get(
            BASE_URL,
            params={"symbol": symbol, "startTime": cursor, "limit": 1000},
            timeout=(5, 20),
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        out.extend(batch)
        last_ts = batch[-1]["fundingTime"]
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        if len(batch) < 1000:
            break
        time.sleep(0.2)  # be polite to the public endpoint
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--months", type=int, default=1, help="how many recent months to pull")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    end = datetime.now(UTC)
    start = end - timedelta(days=30 * args.months)
    start_ms, end_ms = int(start.timestamp() * 1000), int(end.timestamp() * 1000)

    records = fetch_funding_rate(args.symbol, start_ms, end_ms)
    print(f"Pulled {len(records)} funding-rate records for {args.symbol}")
    if records:
        first = datetime.fromtimestamp(records[0]["fundingTime"] / 1000, tz=UTC)
        last = datetime.fromtimestamp(records[-1]["fundingTime"] / 1000, tz=UTC)
        print(f"Coverage in this pull: {first.isoformat()} -> {last.isoformat()}")
        gaps = [
            (records[i + 1]["fundingTime"] - records[i]["fundingTime"]) / 3600_000
            for i in range(len(records) - 1)
        ]
        print(f"Settlement interval (hours): min={min(gaps):.2f} max={max(gaps):.2f}")

    out_path = Path(args.out or f"/tmp/{args.symbol}_funding_rate_sample.csv")
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["symbol", "fundingTime", "fundingRate", "markPrice"])
        w.writeheader()
        w.writerows(records)
    print(f"Wrote sample to {out_path}")


if __name__ == "__main__":
    main()
