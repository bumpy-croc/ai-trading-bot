"""Tick data for the naive market-maker test: download BTCUSDT perp aggregated trades and pack them.

Context: docs/research/experiments/2026-11-23_naive-market-maker-prereg.md.

For each chunk (a calendar month, or the November 2026 days published so far) this writes three
arrays under <dir>/npy/: trade time in ms (int64), price in 0.1-dollar ticks (int32), and whether
the buyer was the maker (bool, i.e. the aggressor was a seller). Zips are kept under <dir>/zips/
unless --delete-zips is given.

Needs pyarrow. Usage: python mm_data.py --dir /path/with/30GB
"""
from __future__ import annotations

import argparse
import io
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv as pacsv

SYMBOL = "BTCUSDT"
BASE = "https://data.binance.vision/data/futures/um"
MONTHS = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04",
          "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]
FRESH_DAYS = [f"2026-11-{d:02d}" for d in range(1, 22)]
TICK = 0.1
COLUMNS = ["agg_trade_id", "price", "quantity", "first_trade_id", "last_trade_id", "transact_time", "is_buyer_maker"]


def fetch(url: str, dest: Path) -> bool:
    """Download with curl (resumes, retries). False when the file does not exist in the archive."""
    if dest.with_suffix(dest.suffix + ".done").exists():
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(4):
        out = subprocess.run(["curl", "-sS", "-f", "-C", "-", "-m", "3000", "-o", str(dest), url], capture_output=True, text=True)
        if out.returncode == 0:
            dest.with_suffix(dest.suffix + ".done").touch()
            return True
        if "404" in out.stderr:
            return False
    raise RuntimeError(f"could not download {url}: {out.stderr.strip()}")


def read_zip(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(time ms, price ticks, buyer-is-maker) from one archive zip. Handles files with and without a header."""
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        with zf.open(name) as raw:
            has_header = raw.read(1) == b"a"
        with zf.open(name) as raw:
            reader = pacsv.open_csv(
                io.BufferedReader(raw, buffer_size=1 << 24),
                read_options=pacsv.ReadOptions(column_names=None if has_header else COLUMNS, block_size=1 << 26),
                convert_options=pacsv.ConvertOptions(
                    include_columns=["price", "transact_time", "is_buyer_maker"],
                    column_types={"price": pa.float64(), "transact_time": pa.int64(), "is_buyer_maker": pa.bool_()},
                ),
            )
            ts, px, bm = [], [], []
            for batch in reader:
                ts.append(batch.column("transact_time").to_numpy())
                px.append(np.rint(batch.column("price").to_numpy() / TICK).astype(np.int32))
                bm.append(batch.column("is_buyer_maker").to_numpy(zero_copy_only=False))
    return np.concatenate(ts), np.concatenate(px), np.concatenate(bm)


def pack(chunk: str, zips: list[Path], out: Path) -> None:
    if (out / f"{chunk}_ts.npy").exists():
        return
    parts = [read_zip(z) for z in zips]
    ts, px, bm = (np.concatenate([p[i] for p in parts]) for i in range(3))
    if not (np.diff(ts) >= 0).all():
        order = np.argsort(ts, kind="stable")
        ts, px, bm = ts[order], px[order], bm[order]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{chunk}_px.npy", px)
    np.save(out / f"{chunk}_bm.npy", bm)
    np.save(out / f"{chunk}_ts.npy", ts)  # written last: its presence marks the chunk complete
    print(f"{chunk}: {len(ts):,} trades, price {px.min() * TICK:,.1f} to {px.max() * TICK:,.1f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--delete-zips", action="store_true")
    ap.add_argument("--only", nargs="+", help="chunks to build (default: all)")
    args = ap.parse_args()
    zips, out = args.dir / "zips", args.dir / "npy"
    for month in MONTHS:
        if args.only and month not in args.only:
            continue
        z = zips / f"{SYMBOL}-aggTrades-{month}.zip"
        if not (out / f"{month}_ts.npy").exists():
            if not fetch(f"{BASE}/monthly/aggTrades/{SYMBOL}/{z.name}", z):
                raise RuntimeError(f"{z.name} is not in the archive")
        pack(month, [z], out)
        if args.delete_zips and z.exists():
            z.unlink()
    if not args.only or "2026-11" in args.only:
        days = []
        for day in FRESH_DAYS:
            z = zips / f"{SYMBOL}-aggTrades-{day}.zip"
            if fetch(f"{BASE}/daily/aggTrades/{SYMBOL}/{z.name}", z):
                days.append(z)
        print(f"2026-11: {len(days)} of {len(FRESH_DAYS)} daily files found")
        pack("2026-11", days, out)


if __name__ == "__main__":
    main()
