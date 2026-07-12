#!/usr/bin/env python3
"""Proof-of-obtainability: pull the full Fear & Greed Index history from alternative.me
(free, no key) and report exact coverage/granularity.

Note: the repo already has a production-grade client for this exact source at
src/data_providers/feargreed_provider.py (FearGreedProvider) — this script exists only
to independently verify coverage claims for the research write-up, not to replace it.
"""
import json
from datetime import UTC, datetime

import requests

URL = "https://api.alternative.me/fng/"


def main():
    resp = requests.get(URL, params={"limit": 0, "format": "json"}, timeout=(5, 20))
    resp.raise_for_status()
    data = resp.json()["data"]

    ts = [int(r["timestamp"]) for r in data]
    ts.sort()
    oldest, newest = ts[0], ts[-1]
    print(f"Total records: {len(data)}")
    print(f"Oldest: {datetime.fromtimestamp(oldest, tz=UTC).date()}")
    print(f"Newest: {datetime.fromtimestamp(newest, tz=UTC).date()}")

    gaps_days = sorted({round((ts[i + 1] - ts[i]) / 86400, 2) for i in range(len(ts) - 1)})
    print(f"Distinct sample-interval values (days): {gaps_days[:5]} ... {gaps_days[-5:]}")
    print("Granularity: daily (one value per calendar day, per API docs)")

    with open("/tmp/feargreed_full_history.json", "w") as f:
        json.dump(data, f)
    print("Wrote full history to /tmp/feargreed_full_history.json")


if __name__ == "__main__":
    main()
