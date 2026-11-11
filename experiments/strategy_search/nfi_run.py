"""Run the frozen NostalgiaForInfinity strategy in freqtrade and score it
(docs/research/experiments/2026-11-11_nfi-frozen-oos-prereg.md).

  python nfi_run.py prepare
      Export the strategy file and its config files from the frozen commit into data/nfi/frozen/.
  python nfi_run.py backtest --tag exam --timerange 20250101-20261101 [--pairs 80] [--fee 0.001]
      Run freqtrade backtesting offline. Binance's API is geo-blocked from the cloud container, so a
      local stub serves the exchange's symbol metadata (saved by nfi_data.py markets) to ccxt.
  python nfi_run.py analyse --tag exam
      Rebuild daily marked-to-market equity from the exported orders, compute the preregistered
      metrics, benchmarks and gates, and write results/nfi_<tag>.json and results/nfi_<tag>_trades.csv.

Needs the freqtrade 2024.11 environment described in the preregistration (FREQTRADE_BIN, or
`freqtrade` on PATH).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from nfi_data import NFI_COMMIT, NFI_DIR, OHLCV_DIR, RESULTS  # noqa: E402

FROZEN = NFI_DIR / "frozen"
USERDIR = NFI_DIR / "user_data"
STRATEGY = "NostalgiaForInfinityX5"
WALLET = 10_000.0
RF_ANNUAL = 0.04
EXAM = (pd.Timestamp("2025-01-01", tz="UTC"), pd.Timestamp("2026-10-31", tz="UTC"))
FROZEN_FILES = [f"{STRATEGY}.py", "configs/trading_mode-spot.json", "configs/exampleconfig.json",
                "configs/blacklist-binance.json"]


def prepare(nfi_repo: Path) -> None:
    for rel in FROZEN_FILES:
        body = subprocess.run(["git", "-C", str(nfi_repo), "show", f"{NFI_COMMIT}:{rel}"],
                              check=True, capture_output=True).stdout
        target = FROZEN / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(body)
    print(f"frozen files written to {FROZEN}")


class _Stub(BaseHTTPRequestHandler):
    payload = b""

    def do_GET(self):  # noqa: N802 - http.server API
        if self.path.split("?")[0].endswith("/exchangeInfo"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        return


def backtest(tag: str, timerange: str, n_pairs: int, fee: float) -> None:
    pairs = json.loads((RESULTS / "nfi_pairs.json").read_text())["pairs"][:n_pairs]
    _Stub.payload = (NFI_DIR / "exchangeInfo.json").read_bytes()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/v3"
    ccxt = {"urls": {"api": {"public": base}}, "options": {"fetchMarkets": ["spot"], "fetchCurrencies": False}}
    overlay = {
        "exchange": {"name": "binance", "pair_whitelist": pairs, "pair_blacklist": [],
                     "ccxt_config": ccxt, "ccxt_async_config": ccxt},
        "pairlists": [{"method": "StaticPairList"}],
    }
    out_dir = NFI_DIR / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "overlay.json").write_text(json.dumps(overlay, indent=1))
    cmd = [
        os.environ.get("FREQTRADE_BIN", "freqtrade"), "backtesting",
        "--strategy", STRATEGY, "--strategy-path", str(FROZEN),
        "-c", str(FROZEN / "configs/trading_mode-spot.json"),
        "-c", str(FROZEN / "configs/exampleconfig.json"),
        "-c", str(out_dir / "overlay.json"),
        "--timerange", timerange, "--fee", str(fee),
        "--userdir", str(USERDIR), "--datadir", str(OHLCV_DIR),
        "--export", "trades", "--export-filename", str(out_dir / "result.json"),
        "--cache", "none", "--breakdown", "month",
        "--logfile", str(out_dir / "freqtrade.log"),
    ]
    env = dict(os.environ, NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
    started = time.time()
    with (out_dir / "stdout.txt").open("w") as fh:
        code = subprocess.run(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    peak_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
    meta = {"tag": tag, "timerange": timerange, "pairs": len(pairs), "fee": fee, "exit_code": code,
            "minutes": round((time.time() - started) / 60, 1), "peak_memory_gb": round(peak_gb, 2)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))
    server.shutdown()


def _load_result(tag: str) -> dict:
    out_dir = NFI_DIR / "runs" / tag
    latest = json.loads((out_dir / ".last_result.json").read_text())["latest_backtest"]
    path = out_dir / latest
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".json") and "_config" not in n and ".meta" not in n)
            return json.loads(zf.read(name))["strategy"][STRATEGY]
    return json.loads(path.read_text())["strategy"][STRATEGY]


def _daily_close(pair: str) -> pd.Series:
    df = pd.read_feather(OHLCV_DIR / f"{pair.replace('/', '_')}-1d.feather")
    return df.set_index(pd.to_datetime(df["date"], utc=True))["close"].astype(float)


def _metrics(equity: pd.Series) -> dict:
    ret = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.0
    excess = ret - RF_ANNUAL / 365.0
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0),
        "max_drawdown": float(-(equity / equity.cummax() - 1.0).min()),
        "sharpe_excess": float(excess.mean() / ret.std() * np.sqrt(365.0)) if ret.std() > 0 else 0.0,
        "worst_day": float(ret.min()),
    }


def analyse(tag: str, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    res = _load_result(tag)
    meta = json.loads((NFI_DIR / "runs" / tag / "run_meta.json").read_text())
    fee = meta["fee"]
    trades = res["trades"]
    pairs = json.loads((RESULTS / "nfi_pairs.json").read_text())["pairs"][: meta["pairs"]]

    # Every filled order as a cash flow and a change in coins held.
    rows = []
    for t in trades:
        for o in t["orders"]:
            side = 1.0 if o["ft_is_entry"] else -1.0
            rows.append({"time": pd.Timestamp(o["order_filled_timestamp"], unit="ms", tz="UTC"), "pair": t["pair"],
                         "coins": side * o["amount"],
                         "cash": -side * o["amount"] * o["safe_price"] - fee * o["amount"] * o["safe_price"]})
    orders = pd.DataFrame(rows).sort_values("time")
    days = pd.date_range(start, end + pd.Timedelta(days=1), freq="D", tz="UTC")  # equity at each 00:00 UTC
    cash = WALLET + orders.set_index("time")["cash"].cumsum()
    cash_daily = cash.groupby(level=0).last().reindex(cash.index.unique().union(days)).ffill().reindex(days).fillna(WALLET)
    held_value = pd.Series(0.0, index=days)
    exposure_pairs = pd.Series(0, index=days)
    for pair, grp in orders.groupby("pair"):
        coins = grp.set_index("time")["coins"].cumsum().groupby(level=0).last()
        coins = coins.reindex(coins.index.union(days)).ffill().reindex(days).fillna(0.0)
        coins = coins.where(coins.abs() > 1e-9, 0.0)
        close = _daily_close(pair)
        # value at 00:00 on day D uses the close of day D-1 (the last price known at that moment)
        px = close.shift(1, freq="D").reindex(close.index.union(days)).ffill().reindex(days)
        held_value += (coins * px).fillna(0.0)
        exposure_pairs += (coins > 0).astype(int)
    equity = cash_daily + held_value

    out = {
        "tag": tag, "run": meta, "nfi_commit": NFI_COMMIT,
        "window": [str(start.date()), str(end.date())],
        "freqtrade": {k: res.get(k) for k in (
            "total_trades", "profit_total", "profit_total_abs", "final_balance", "max_drawdown_account",
            "profit_factor", "winrate", "holding_avg", "cagr", "sharpe", "trade_count_long", "max_open_trades")},
        "reconstruction_check": {
            "final_equity_rebuilt": float(equity.iloc[-1]),
            "final_balance_freqtrade": res.get("final_balance"),
        },
        "mark_to_market": _metrics(equity),
        "by_period": {},
        "avg_share_of_equity_in_positions": float((held_value / equity).mean()),
        "avg_pairs_held": float(exposure_pairs.mean()),
    }
    for label, a, b in (("2025", "2025-01-01", "2026-01-01"), ("2026_jan_oct", "2026-01-01", "2026-11-01")):
        seg = equity.loc[a:b]
        if len(seg) > 1:
            out["by_period"][label] = float(seg.iloc[-1] / seg.iloc[0] - 1.0)

    # Benchmarks over the same days.
    def _hold(pair: str) -> pd.Series:
        px = _daily_close(pair).shift(1, freq="D")
        return px.reindex(px.index.union(days)).ffill().reindex(days)

    btc = _hold("BTC/USDT")
    basket = pd.concat({p: _hold(p) for p in pairs}, axis=1)
    basket = (basket / basket.iloc[0]).mean(axis=1)  # equal weight at the start, never rebalanced; dead coins keep their last price
    idle = pd.Series((1 + RF_ANNUAL) ** (np.arange(len(days)) / 365.0), index=days)
    out["benchmarks"] = {"idle_usdt_4pct": {k: v for k, v in _metrics(idle).items() if k in ("total_return", "cagr")},
                         "btc_buy_and_hold": _metrics(btc), "equal_weight_basket_of_same_pairs": _metrics(basket)}

    closed = pd.DataFrame([{
        "pair": t["pair"], "open": t["open_date"], "close": t["close_date"], "profit_abs": t["profit_abs"],
        "profit_ratio": t["profit_ratio"], "enter_tag": t.get("enter_tag"), "exit_reason": t.get("exit_reason"),
        "orders": len(t["orders"]), "max_stake": t.get("max_stake_amount"),
        "days_held": t.get("trade_duration", 0) / 1440.0,
    } for t in trades])
    closed.to_csv(RESULTS / f"nfi_{tag}_trades.csv", index=False)
    equity.rename("equity").to_csv(RESULTS / f"nfi_{tag}_equity.csv", header=True)
    out["trades"] = {
        "count": int(len(closed)), "win_rate": float((closed["profit_abs"] > 0).mean()),
        "median_profit_ratio": float(closed["profit_ratio"].median()),
        "median_days_held": float(closed["days_held"].median()), "max_days_held": float(closed["days_held"].max()),
        "worst_10": closed.nsmallest(10, "profit_abs")[["pair", "open", "close", "profit_abs", "profit_ratio", "exit_reason"]].to_dict("records"),
        "by_exit_reason": closed.groupby("exit_reason")["profit_abs"].agg(["count", "sum"]).round(2).to_dict("index"),
        "profit_by_pair_worst_5": closed.groupby("pair")["profit_abs"].sum().nsmallest(5).round(2).to_dict(),
        "profit_by_pair_best_5": closed.groupby("pair")["profit_abs"].sum().nlargest(5).round(2).to_dict(),
        "profit_by_month": closed.assign(m=closed["close"].str[:7]).groupby("m")["profit_abs"].sum().round(2).to_dict(),
    }
    m = out["mark_to_market"]
    out["gates"] = {
        "1_cagr_at_least_10pct": m["cagr"] >= 0.10,
        "2_max_drawdown_at_most_30pct": m["max_drawdown"] <= 0.30,
        "3_positive_in_both_periods": len(out["by_period"]) == 2 and all(v > 0 for v in out["by_period"].values()),
        "4_sharpe_at_least_0_8": m["sharpe_excess"] >= 0.8,
        "5_at_least_100_trades": len(closed) >= 100,
    }
    out["gates_1_to_5_pass"] = all(out["gates"].values())
    (RESULTS / f"nfi_{tag}.json").write_text(json.dumps(out, indent=1, default=float))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "backtest", "analyse"])
    ap.add_argument("--nfi-repo", type=Path, default=Path("/home/user/iterativv/nostalgiaforinfinity"))
    ap.add_argument("--tag", default="exam")
    ap.add_argument("--timerange", default="20250101-20261101")
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--start", default=str(EXAM[0].date()))
    ap.add_argument("--end", default=str(EXAM[1].date()))
    args = ap.parse_args()
    if args.mode == "prepare":
        prepare(args.nfi_repo)
    elif args.mode == "backtest":
        backtest(args.tag, args.timerange, args.pairs, args.fee)
    else:
        report = analyse(args.tag, pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"))
        print(json.dumps({k: v for k, v in report.items() if k != "trades"}, indent=1, default=float))
