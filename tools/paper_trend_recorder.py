"""Paper trend recorder: write down each book's target weights before the prices that judge them exist.

Context: docs/research/experiments/2026-11-11_slow-adaptive-trend-prereg.md (Stage B).
The slow many-coin trend books were designed on 2020-24 data. The 2025-26 window is used up, so
the only clean test left is forward: every morning this script rebuilds each book's target weights
from the daily candles up to the UTC close that has just passed and saves them. A Monday close's
weights are filled, in the scoring, at Tuesday's close, which has not happened yet when they are
recorded. experiments/strategy_search/paper_score.py scores the recorded weights later.

It reads the public Binance archive only. No account, no orders, no money.

Output: <out-dir>/records/YYYY/record_YYYY-MM-DD.json, named after the UTC day whose close the
weights come from. A day that already has a record is never rewritten. Weights appear only on
signal days (Mondays); other days get a heartbeat record, so a missed day is visible.

Exit codes: 0 = record written, already present, or the day's candles are not published yet and
it is still early; 2 = blind (candles still missing late in the day, the archive is unreachable,
or fewer than 100 perps returned data).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments" / "strategy_search"))

import adaptive  # noqa: E402
import engine as eng  # noqa: E402
import forward_data  # noqa: E402

DESIGN_FILE = REPO / "experiments" / "strategy_search" / "results" / "adaptive_design.json"
# A Monday's weights depend on about 250 days of candles: 60 days of the book's own returns for the
# volatility scale, whose weights need 120-day trends, on coins that need 60 days of history to qualify.
HISTORY_DAYS = 420
MIN_PERPS = 100
GIVE_UP_HOUR_UTC = (
    15  # the archive publishes at about 09:00 UTC; after this hour a missing day is a failure
)
EXIT_OK = 0
EXIT_BLIND = 2


def forward_books() -> list[str]:
    """The books Stage A sent forward, read from its committed result."""
    return list(json.loads(DESIGN_FILE.read_text())["selection"]["forward_books"])


def record_path(out_dir: Path, day: pd.Timestamp) -> Path:
    return out_dir / "records" / f"{day:%Y}" / f"record_{day:%Y-%m-%d}.json"


def build_record(
    market, info: dict, day: pd.Timestamp, books: Sequence[str], now: datetime
) -> dict:
    """The record for `day`. Weights are included only when `day` is a signal day."""
    signal_day = bool(day in eng.weekly_days(market.close.index))
    record: dict = {
        "data_day": f"{day:%Y-%m-%d}",
        "recorded_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "signal_day": signal_day,
        "fill": "the close of the next UTC day",
        "history_days": HISTORY_DAYS,
        **info,
        "code": {
            "git_sha": os.environ.get("GITHUB_SHA", "local"),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "books": {},
    }
    if not signal_day:
        return record
    built = adaptive.build_books(market, tuple(books))
    for name in books:
        row = built[name].weights.loc[day].fillna(0.0)
        held = row[row != 0.0]
        leverage = built[name].result.extra["leverage"].loc[day]
        record["books"][name] = {
            "leverage": round(float(leverage), 8),
            "gross": round(float(held.abs().sum()), 8),
            "net": round(float(held.sum()), 8),
            "weights": {symbol: round(float(w), 8) for symbol, w in held.sort_index().items()},
        }
    return record


def main(argv: Sequence[str] | None = None, now: datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, required=True, help="checkout of the data branch")
    parser.add_argument(
        "--day",
        help="UTC day to record (default: yesterday). For dry runs only: a record "
        "written after its fill price exists is ignored by the scoring.",
    )
    parser.add_argument(
        "--books", nargs="+", help="override the books to record (default: Stage A's selection)"
    )
    args = parser.parse_args(argv)

    now = now or datetime.now(UTC)
    day = (
        pd.Timestamp(args.day, tz="UTC")
        if args.day
        else pd.Timestamp(now.date(), tz="UTC") - pd.Timedelta(days=1)
    )
    path = record_path(args.out_dir, day)
    if path.exists():
        print(f"{day:%Y-%m-%d} is already recorded; leaving {path} as it is")
        return EXIT_OK
    try:
        if not forward_data.day_published(day):
            if args.day is None and now.hour < GIVE_UP_HOUR_UTC:
                print(
                    f"candles for {day:%Y-%m-%d} are not published yet; a later run will record them"
                )
                return EXIT_OK
            print(f"BLIND: no candles for {day:%Y-%m-%d} in the archive", file=sys.stderr)
            return EXIT_BLIND
        market, info = forward_data.fetch_market(day, HISTORY_DAYS)
    except RuntimeError as exc:
        print(f"BLIND: {exc}", file=sys.stderr)
        return EXIT_BLIND
    if info["perps_with_data"] < MIN_PERPS:
        print(
            f"BLIND: only {info['perps_with_data']} perps returned data (expected at least {MIN_PERPS}); "
            "not writing a record",
            file=sys.stderr,
        )
        return EXIT_BLIND

    books = args.books or forward_books()
    record = build_record(market, info, day, books, now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    if record["signal_day"]:
        summary = ", ".join(
            f"{n}: {len(b['weights'])} coins, gross {b['gross']:.2f}, net {b['net']:+.2f}"
            for n, b in record["books"].items()
        )
        print(f"{day:%Y-%m-%d} (signal day): {summary} -> {path}")
    else:
        print(f"{day:%Y-%m-%d}: heartbeat, {info['perps_with_data']} perps -> {path}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
