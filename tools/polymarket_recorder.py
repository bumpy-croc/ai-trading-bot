"""Polymarket recorder: save a point-in-time snapshot of every open, liquid market.

Context: docs/research/experiments/2026-11-11_polymarket-forward-underdogs-prereg.md.
A backtest suggested underdogs on Polymarket win more often than their price, but it
could only choose markets by their *final* volume, which over-represents upsets. This
script records volume and quotes as they are on the day, so the question can be
scored cleanly later (experiments/polymarket_forward_score.py).

It reads the public market listing only. No account, no orders, no money.

Standalone on purpose (stdlib + requests): it runs on a bare GitHub Actions runner
without the bot's dependencies or database.

Output: one gzip CSV per run, <out-dir>/YYYY/MM/snapshot_YYYYMMDDTHHMMSSZ.csv.gz.

Exit codes: 0 = snapshot written, 2 = listing unreachable or implausibly small
(the recorder is blind, which should be visible as a failed run).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests  # type: ignore[import-untyped]  # types-requests not installed

LISTING_URL = "https://gamma-api.polymarket.com/markets/keyset"
PAGE_SIZE = 500
MAX_PAGES = 200  # 100,000 open markets; far above anything seen (about 1,600 on 2026-11-11)
MIN_VOLUME_USD = 100_000
MIN_PLAUSIBLE_MARKETS = 100
HTTP_TIMEOUT_S = 60
HTTP_ATTEMPTS = 4
EXIT_OK = 0
EXIT_BLIND = 2

COLUMNS = [
    "snapshot_ts",
    "id",
    "event_id",
    "question",
    "n_outcomes",
    "best_bid",
    "best_ask",
    "last_trade",
    "volume",
    "liquidity",
    "start_date",
    "end_date",
    "accepting_orders",
    "fees_enabled",
    "fee_type",
    "fee_rate",
    "fee_exponent",
    "neg_risk",
    "tick_size",
]


def _count_outcomes(raw: Any) -> int:
    """`outcomes` arrives as a JSON-encoded list in a string."""
    if isinstance(raw, list):
        return len(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError:
            return 0
        return len(parsed) if isinstance(parsed, list) else 0
    return 0


def market_row(market: dict[str, Any], snapshot_ts: str) -> dict[str, Any]:
    """Flatten one listing entry to the recorded columns. Missing fields stay empty."""
    events = market.get("events") or []
    schedule = market.get("feeSchedule") or {}
    return {
        "snapshot_ts": snapshot_ts,
        "id": market.get("id"),
        "event_id": events[0].get("id") if events else None,
        "question": (market.get("question") or "")[:200],
        "n_outcomes": _count_outcomes(market.get("outcomes")),
        "best_bid": market.get("bestBid"),
        "best_ask": market.get("bestAsk"),
        "last_trade": market.get("lastTradePrice"),
        "volume": market.get("volumeNum"),
        "liquidity": market.get("liquidityNum"),
        "start_date": market.get("startDate"),
        "end_date": market.get("endDate"),
        "accepting_orders": market.get("acceptingOrders"),
        "fees_enabled": bool(market.get("feesEnabled")),
        "fee_type": market.get("feeType"),
        "fee_rate": schedule.get("rate"),
        "fee_exponent": schedule.get("exponent"),
        "neg_risk": market.get("negRisk"),
        "tick_size": market.get("orderPriceMinTickSize"),
    }


def _get_page(session: requests.Session, cursor: str | None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": PAGE_SIZE,
        "closed": "false",
        "volume_num_min": MIN_VOLUME_USD,
    }
    if cursor:
        params["after_cursor"] = cursor
    last_error: Exception | None = None
    for attempt in range(HTTP_ATTEMPTS):
        try:
            resp = session.get(LISTING_URL, params=params, timeout=HTTP_TIMEOUT_S)
            resp.raise_for_status()
            page = resp.json()
            if isinstance(page, dict):
                return page
            last_error = ValueError("listing page is not a JSON object")
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
        time.sleep(2**attempt)
    raise RuntimeError(f"listing unreachable after {HTTP_ATTEMPTS} attempts: {last_error}")


def fetch_open_markets(session: requests.Session) -> list[dict[str, Any]]:
    """Every open market above the volume floor, de-duplicated by id."""
    seen: dict[str, dict[str, Any]] = {}
    cursor: str | None = None
    for _ in range(MAX_PAGES):
        page = _get_page(session, cursor)
        markets = page.get("markets") or []
        for market in markets:
            market_id = market.get("id")
            if market_id is not None:
                seen[str(market_id)] = market
        cursor = page.get("next_cursor")
        if not markets or not cursor:
            return list(seen.values())
    raise RuntimeError(f"listing did not end within {MAX_PAGES} pages")


def write_snapshot(rows: Sequence[dict[str, Any]], out_dir: Path, now: datetime) -> Path:
    path = out_dir / f"{now:%Y}" / f"{now:%m}" / f"snapshot_{now:%Y%m%dT%H%M%SZ}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS)
    writer.writeheader()
    writer.writerows(rows)
    # mtime=0 keeps the bytes a pure function of the rows
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as fh:
        fh.write(buffer.getvalue().encode("utf-8"))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="directory to write the snapshot under"
    )
    args = parser.parse_args(argv)

    now = datetime.now(UTC)
    session = requests.Session()
    session.headers["User-Agent"] = "polymarket-recorder"
    try:
        markets = fetch_open_markets(session)
    except RuntimeError as exc:
        print(f"BLIND: {exc}", file=sys.stderr)
        return EXIT_BLIND
    if len(markets) < MIN_PLAUSIBLE_MARKETS:
        print(
            f"BLIND: only {len(markets)} open markets listed (expected at least "
            f"{MIN_PLAUSIBLE_MARKETS}); not writing a snapshot",
            file=sys.stderr,
        )
        return EXIT_BLIND

    snapshot_ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = sorted(
        (market_row(m, snapshot_ts) for m in markets),
        key=lambda r: (len(str(r["id"])), str(r["id"])),
    )
    path = write_snapshot(rows, args.out_dir, now)
    quoted = sum(1 for r in rows if r["best_bid"] is not None and r["best_ask"] is not None)
    print(f"snapshot {snapshot_ts}: {len(rows)} markets, {quoted} with a bid and ask -> {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
