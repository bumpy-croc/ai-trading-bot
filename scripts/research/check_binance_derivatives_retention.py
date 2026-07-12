#!/usr/bin/env python3
"""Proof-of-(un)obtainability: empirically check how far back Binance's free
"data statistics" futures endpoints (open interest history, long/short ratio) actually
go, versus the fundingRate and premiumIndexKlines endpoints which use normal
kline-style pagination.

Finding this script exists to document (see docs/research/2026-07-12_input-candidates-audit.md):
openInterestHist / globalLongShortAccountRatio reject any startTime older than ~30 days
(code -1130), i.e. free-tier history is NOT usable for the 2023H1/2024H1/2025H1 backtest
folds. fundingRate and premiumIndexKlines have no such restriction (verified back to 2019).
"""
from datetime import UTC, datetime, timedelta

import requests

SYMBOL = "ETHUSDT"


def probe(name: str, url: str, extra_params: dict, days_back: int) -> None:
    start = datetime.now(UTC) - timedelta(days=days_back)
    params = {"symbol": SYMBOL, "limit": 2, "startTime": int(start.timestamp() * 1000)}
    params.update(extra_params)
    resp = requests.get(url, params=params, timeout=(5, 20))
    ok = resp.status_code == 200 and isinstance(resp.json(), list)
    print(f"{name:35s} days_back={days_back:4d}  status={resp.status_code}  ok={ok}  body={resp.text[:120]}")


def main():
    endpoints = {
        "openInterestHist": ("https://fapi.binance.com/futures/data/openInterestHist", {"period": "1h"}),
        "globalLongShortAccountRatio": (
            "https://fapi.binance.com/futures/data/globalLongShortAccountRatio",
            {"period": "1h"},
        ),
        "fundingRate (control, deep history)": (
            "https://fapi.binance.com/fapi/v1/fundingRate",
            {},
        ),
        "premiumIndexKlines (control, deep history)": (
            "https://fapi.binance.com/fapi/v1/premiumIndexKlines",
            {"interval": "1h"},
        ),
    }
    for days_back in (10, 25, 30, 45, 90, 365, 2000):
        for name, (url, extra) in endpoints.items():
            probe(name, url, extra, days_back)
        print()


if __name__ == "__main__":
    main()
