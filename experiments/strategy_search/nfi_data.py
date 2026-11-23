"""Pair list, price data and market metadata for the NostalgiaForInfinity out-of-sample test
(docs/research/experiments/2026-11-11_nfi-frozen-oos-prereg.md).

Everything comes from the public Binance archive (data.binance.vision) and the reachable REST
mirror, because api.binance.com is geo-blocked from the cloud container.

  python experiments/strategy_search/nfi_data.py pairs --nfi-repo PATH
      Rank Binance spot USDT pairs by December 2024 quote volume, apply the strategy's own
      blacklist (read from the frozen commit) and a 30-day age filter, and write the top 80 to
      results/nfi_pairs.json. Uses no data after 2024-12-31.
  python experiments/strategy_search/nfi_data.py download
      5m/15m/1h/4h/1d candles for those pairs plus BTC/USDT, written as freqtrade feather files.
  python experiments/strategy_search/nfi_data.py markets
      Save the exchange's symbol metadata for the offline stub used by nfi_run.py.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import FILES, WORKERS, _list, _read_zip_csv, _session, _to_ts  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
NFI_DIR = HERE / "data" / "nfi"
OHLCV_DIR = NFI_DIR / "user_data" / "data" / "binance"

NFI_COMMIT = "e5b345a8d81e4559572cc0806d86d9178d0719e1"  # 2024-12-30, last commit of 2024 (v15.1.312)
N_PAIRS = 80
RANK_MONTH = "2024-12"
AGE_MONTH = "2024-11"  # a monthly file here means the pair listed at least 30 days before 2024-12-31
EXAM_END_MONTH = "2026-10"
# First month to download per timeframe. The strategy asks for 800 start-up candles, which freqtrade
# applies per timeframe, so the slower informative timeframes need a longer run-up.
FIRST_MONTH = {"5m": "2024-11", "15m": "2024-11", "1h": "2024-10", "4h": "2024-06", "1d": "2022-08"}
PREFIX = "data/spot/monthly/klines"
MIRROR = "https://www.binance.com/api/v3/exchangeInfo"


def _blacklist(nfi_repo: Path) -> list[re.Pattern]:
    raw = subprocess.run(
        ["git", "-C", str(nfi_repo), "show", f"{NFI_COMMIT}:configs/blacklist-binance.json"],
        check=True, capture_output=True, text=True,
    ).stdout
    body = "\n".join(line for line in raw.splitlines() if not line.strip().startswith("//"))
    return [re.compile(p, re.IGNORECASE) for p in json.loads(body)["exchange"]["pair_blacklist"]]


def _month_volume(symbol: str) -> tuple[str, float | None, bool]:
    """(symbol, December 2024 quote volume or None if no file, listed by November 2024)."""
    try:
        df = _read_zip_csv(f"{PREFIX}/{symbol}/1d/{symbol}-1d-{RANK_MONTH}.zip")
    except requests.HTTPError:
        return symbol, None, False
    aged = _session.head(f"{FILES}/{PREFIX}/{symbol}/1d/{symbol}-1d-{AGE_MONTH}.zip", timeout=60).status_code == 200
    return symbol, float(df.iloc[:, 7].astype(float).sum()), aged


def pairs(nfi_repo: Path) -> dict:
    prefixes, _ = _list(f"{PREFIX}/")
    symbols = sorted(s for s in (p.rstrip("/").split("/")[-1] for p in prefixes) if re.fullmatch(r"[A-Z0-9]+USDT", s))
    black = _blacklist(nfi_repo)
    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(_month_volume, symbols))
    table = []
    for symbol, volume, aged in rows:
        pair = f"{symbol[:-4]}/USDT"
        table.append({
            "pair": pair,
            "quote_volume_2024_12": volume,
            "traded_2024_12": volume is not None and volume > 0,
            "listed_30_days": aged,
            "blacklisted": any(p.fullmatch(pair) for p in black),
        })
    ok = [r for r in table if r["traded_2024_12"] and r["listed_30_days"] and not r["blacklisted"]]
    ok.sort(key=lambda r: (-r["quote_volume_2024_12"], r["pair"]))
    chosen = ok[:N_PAIRS]
    out = {
        "nfi_commit": NFI_COMMIT,
        "rule": f"top {N_PAIRS} Binance spot USDT pairs by {RANK_MONTH} quote volume, strategy blacklist applied, "
                f"listed by {AGE_MONTH}",
        "symbols_in_archive": len(symbols),
        "traded_in_rank_month": sum(r["traded_2024_12"] for r in table),
        "eligible": len(ok),
        "pairs": [r["pair"] for r in chosen],
        "quote_volume_2024_12": {r["pair"]: r["quote_volume_2024_12"] for r in chosen},
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "nfi_pairs.json").write_text(json.dumps(out, indent=1))
    return out


def _months(first: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(first, EXAM_END_MONTH, freq="M")]


def _one(job: tuple[str, str]) -> dict:
    pair, tf = job
    symbol = pair.replace("/", "")
    target = OHLCV_DIR / f"{pair.replace('/', '_')}-{tf}.feather"
    if target.exists():
        df = pd.read_feather(target)
        return {"pair": pair, "tf": tf, "rows": len(df), "first": str(df["date"].iloc[0]), "last": str(df["date"].iloc[-1])}
    frames, missing = [], []
    for month in _months(FIRST_MONTH[tf]):
        try:
            df = _read_zip_csv(f"{PREFIX}/{symbol}/{tf}/{symbol}-{tf}-{month}.zip").iloc[:, :6]
        except requests.HTTPError:
            missing.append(month)
            continue
        df.columns = ["date", "open", "high", "low", "close", "volume"]
        frames.append(df)
    if not frames:
        return {"pair": pair, "tf": tf, "rows": 0, "missing": missing}
    out = pd.concat(frames, ignore_index=True)
    out["date"] = _to_ts(out["date"]).astype("datetime64[ms, UTC]")
    out = out.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    out[["open", "high", "low", "close", "volume"]] = out[["open", "high", "low", "close", "volume"]].astype(float)
    out.to_feather(target, compression_level=9, compression="lz4")
    return {"pair": pair, "tf": tf, "rows": len(out), "first": str(out["date"].iloc[0]),
            "last": str(out["date"].iloc[-1]), "missing_months": missing}


def download() -> list[dict]:
    chosen = json.loads((RESULTS / "nfi_pairs.json").read_text())["pairs"]
    wanted = sorted(set(chosen) | {"BTC/USDT"})
    OHLCV_DIR.mkdir(parents=True, exist_ok=True)
    jobs = [(p, tf) for p in wanted for tf in FIRST_MONTH]
    with ThreadPoolExecutor(WORKERS) as pool:
        report = list(pool.map(_one, jobs))
    (NFI_DIR / "download_report.json").write_text(json.dumps(report, indent=1))
    return report


def markets() -> int:
    """Symbol metadata from the reachable mirror. Delisted pairs come back with status BREAK; the
    offline stub flips them to TRADING so freqtrade will backtest them over the months they traded."""
    info = requests.get(MIRROR, timeout=120).json()
    keep = [s for s in info["symbols"] if s["quoteAsset"] == "USDT"]
    for s in keep:
        s["status"] = "TRADING"
    info["symbols"] = keep
    NFI_DIR.mkdir(parents=True, exist_ok=True)
    (NFI_DIR / "exchangeInfo.json").write_text(json.dumps(info))
    return len(keep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "download", "markets"])
    ap.add_argument("--nfi-repo", type=Path, default=Path("/home/user/iterativv/nostalgiaforinfinity"))
    args = ap.parse_args()
    if args.mode == "pairs":
        res = pairs(args.nfi_repo)
        print(json.dumps({k: v for k, v in res.items() if k != "quote_volume_2024_12"}, indent=1))
    elif args.mode == "download":
        rep = download()
        five = [r for r in rep if r["tf"] == "5m"]
        print(f"{len(rep)} files; 5m rows min/median/max = "
              f"{min(r['rows'] for r in five)}/{sorted(r['rows'] for r in five)[len(five) // 2]}/{max(r['rows'] for r in five)}")
        short = [(r["pair"], r.get("last")) for r in five if r["rows"] and not str(r.get("last", "")).startswith("2026-10-31")]
        print("5m series ending before 2026-10-31:", short)
    else:
        print(f"{markets()} USDT symbols saved")
