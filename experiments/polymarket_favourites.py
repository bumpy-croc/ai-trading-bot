"""Prediction-market favourites test (Polymarket), as preregistered in
docs/research/experiments/2026-11-10_prediction-market-favourites-prereg.md.

Buys the favourite side of a two-outcome market the day after its price first enters
a band, holds to resolution, and measures the excess return over T-bills after a
per-share entry cost.

Usage: python experiments/polymarket_favourites.py
Writes experiments/markets_results/polymarket_favourites.json
Data is cached under experiments/markets_data/polymarket/ (git-ignored).
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import markets_common as mc
import numpy as np
import pandas as pd
import requests

PM_DIR = mc.DATA_DIR / "polymarket"
OUT = Path(__file__).resolve().parent / "markets_results" / "polymarket_favourites.json"
GAMMA = "https://gamma-api.polymarket.com/markets/keyset"
CLOB = "https://clob.polymarket.com/prices-history"

MIN_LISTING_VOLUME = 100_000
MIN_VOLUME = 250_000
MIN_LIFE_DAYS = 7
END_FROM, END_TO = "2022-01-01", "2026-09-30"
BANDS = {
    "PM1 0.90-0.97": (0.90, 0.97),
    "PM2 0.97-0.99": (0.97, 0.99),
    "PM3 0.70-0.90": (0.70, 0.90),
}
COST = 0.01
COST_SENSITIVITY = (0.005, 0.02)
NO_FILL_AT = 0.995
BONFERRONI_Z_18 = 2.7729  # one-sided 5% / 18 candidates
KEEP = (
    "id",
    "question",
    "outcomes",
    "outcomePrices",
    "clobTokenIds",
    "volumeNum",
    "startDate",
    "endDate",
    "closedTime",
    "createdAt",
    "feesEnabled",
)


# --------------------------------------------------------------------------- download


def fetch_listing() -> list[dict]:
    path = PM_DIR / "markets.json"
    if path.exists():
        return json.loads(path.read_text())
    PM_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = mc.UA
    out, cursor = [], None
    while True:
        params = {"limit": 500, "closed": "true", "volume_num_min": MIN_LISTING_VOLUME}
        if cursor:
            params["after_cursor"] = cursor
        page = session.get(GAMMA, params=params, timeout=90).json()
        for m in page.get("markets", []):
            row = {k: m.get(k) for k in KEEP}
            row["event_ids"] = [e.get("id") for e in (m.get("events") or [])]
            out.append(row)
        cursor = page.get("next_cursor")
        if not page.get("markets") or not cursor:
            break
        time.sleep(0.25)
    path.write_text(json.dumps(out))
    return out


def universe(listing: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(listing)
    for col in ("startDate", "endDate", "createdAt", "closedTime"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
    df["life_days"] = (
        df["endDate"] - df["startDate"].fillna(df["createdAt"])
    ).dt.total_seconds() / 86400.0
    df["n_outcomes"] = df["outcomes"].apply(
        lambda o: len(json.loads(o)) if isinstance(o, str) else 0
    )
    payouts = df["outcomePrices"].apply(
        lambda p: [float(x) for x in json.loads(p)] if isinstance(p, str) else []
    )
    df["yes_payout"] = payouts.apply(
        lambda p: p[0] if len(p) == 2 and abs(sum(p) - 1.0) < 0.01 else np.nan
    )
    df["yes_token"] = df["clobTokenIds"].apply(
        lambda t: json.loads(t)[0] if isinstance(t, str) and t.startswith("[") else None
    )
    df["event"] = df["event_ids"].apply(lambda e: e[0] if e else None)
    keep = (
        (df["n_outcomes"] == 2)
        & ~df["feesEnabled"].fillna(False).astype(bool)
        & (df["volumeNum"] >= MIN_VOLUME)
        & (df["life_days"] >= MIN_LIFE_DAYS)
        & (df["endDate"] >= pd.Timestamp(END_FROM, tz="UTC"))
        & (df["endDate"] < pd.Timestamp(END_TO, tz="UTC") + pd.Timedelta(days=1))
        & df["yes_payout"].notna()
        & df["yes_token"].notna()
        & df["closedTime"].notna()
    )
    df = df[keep].copy()
    df["event"] = df["event"].fillna("m" + df["id"].astype(str))
    return df.sort_values("id", key=lambda s: s.astype(int)).reset_index(drop=True)


def fetch_histories(uni: pd.DataFrame) -> dict[str, list]:
    """market id -> [[unix seconds, yes price], ...] at daily fidelity. Cached, resumable."""
    path = PM_DIR / "histories.jsonl"
    have: dict[str, list] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            have[rec["id"]] = rec["h"]
    todo = [(r.id, r.yes_token) for r in uni.itertuples() if r.id not in have]
    if not todo:
        return have
    session = requests.Session()
    session.headers["User-Agent"] = mc.UA

    def one(item: tuple[str, str]) -> tuple[str, list | None]:
        market_id, token = item
        for attempt in range(5):
            try:
                resp = session.get(
                    CLOB, params={"market": token, "interval": "max", "fidelity": 1440}, timeout=60
                )
                if resp.status_code == 200:
                    return market_id, [[p["t"], p["p"]] for p in resp.json().get("history", [])]
                if resp.status_code in (400, 404):
                    return market_id, []
            except requests.RequestException:
                pass
            time.sleep(2**attempt)
        return market_id, None

    with path.open("a") as fh, ThreadPoolExecutor(max_workers=4) as pool:
        for n, (market_id, hist) in enumerate(pool.map(one, todo), 1):
            if hist is None:
                continue  # left for the next run
            have[market_id] = hist
            fh.write(json.dumps({"id": market_id, "h": hist}) + "\n")
            if n % 1000 == 0:
                print(f"  histories {n}/{len(todo)}", flush=True)
    return have


# --------------------------------------------------------------------------- bets


def make_bets(
    uni: pd.DataFrame,
    hist: dict[str, list],
    band: tuple[float, float],
    lower_open: bool = False,
    upper_open: bool = False,
) -> pd.DataFrame:
    """One bet per market: signal at the first daily point in the band, fill at the next point."""
    lo, hi = band
    rows = []
    for m in uni.itertuples():
        points = hist.get(m.id) or []
        closed = m.closedTime.timestamp()
        for i in range(len(points) - 1):
            p_yes = points[i][1]
            fav = max(p_yes, 1.0 - p_yes)
            in_band = (fav > lo if lower_open else fav >= lo) and (
                fav < hi if upper_open else fav <= hi
            )
            if not in_band:
                continue
            t_fill, p_fill_yes = points[i + 1]
            if t_fill >= closed:
                break  # no tradable point after the signal
            side_yes = p_yes >= 0.5
            fill = p_fill_yes if side_yes else 1.0 - p_fill_yes
            payout = m.yes_payout if side_yes else 1.0 - m.yes_payout
            rows.append(
                {
                    "id": m.id,
                    "event": m.event,
                    "signal_price": fav,
                    "fill": fill,
                    "payout": payout,
                    "t_fill": pd.Timestamp(t_fill, unit="s"),
                    "t_close": m.closedTime.tz_convert(None),
                    "volume": m.volumeNum,
                }
            )
            break
    bets = pd.DataFrame(rows)
    bets["days"] = (
        bets["t_close"] - bets["t_fill"]
    ).dt.total_seconds() / 86400.0 + 1.0  # one day to redeem
    return bets


def score(bets: pd.DataFrame, cost: float, irx: pd.Series) -> dict:
    b = bets[
        (bets["fill"] + cost < NO_FILL_AT)
        & (bets["t_fill"] >= "2022-01-01")
        & (bets["t_fill"] < "2026-10-01")
    ].copy()
    skipped = len(bets) - len(b)
    stake = b["fill"] + cost
    days = b["t_fill"].dt.normalize()
    rf = irx.reindex(irx.index.union(days.unique())).ffill().loc[days].to_numpy()
    b["ret"] = b["payout"] / stake - 1.0
    b["excess"] = b["ret"] - rf * b["days"] / 365.0
    b["stake"], b["profit"], b["rf"] = stake, b["payout"] - stake, rf

    def tstat(groups: pd.Series) -> float:
        means = b.groupby(groups)["excess"].mean()
        return (
            float(means.mean() / means.std(ddof=1) * math.sqrt(len(means)))
            if len(means) > 2 and means.std() > 0
            else 0.0
        )

    week = b["t_fill"].dt.to_period("W").astype(str)
    t_event, t_week = tstat(b["event"]), tstat(week)
    cap_days = float((b["stake"] * b["days"]).sum())
    roc = float(b["profit"].sum() / cap_days * 365.0)
    rf_weighted = float((b["rf"] * b["stake"] * b["days"]).sum() / cap_days)
    by_year = b.groupby(b["t_fill"].dt.year).agg(
        n=("excess", "size"),
        mean_excess=("excess", "mean"),
        win=("payout", "mean"),
        fill=("fill", "mean"),
    )
    month = b["t_close"].dt.to_period("M").astype(str)
    monthly = b.groupby(month).agg(
        n=("profit", "size"), profit=("profit", "sum"), stake=("stake", "sum")
    )
    monthly["ret"] = monthly["profit"] / monthly["stake"]
    big = monthly[monthly["n"] >= 20]
    worst = big["ret"].idxmin() if len(big) else None
    return {
        "bets": int(len(b)),
        "no_fill_skipped": int(skipped),
        "events": int(b["event"].nunique()),
        "mean_fill": float(b["fill"].mean()),
        "win_rate": float(b["payout"].mean()),
        "mean_ret": float(b["ret"].mean()),
        "mean_excess": float(b["excess"].mean()),
        "t_event": t_event,
        "t_week": t_week,
        "return_on_capital_ann": roc,
        "return_on_capital_excess_ann": roc - rf_weighted,
        "median_days": float(b["days"].median()),
        "mean_days": float(b["days"].mean()),
        "worst_month": (
            {"month": worst, "ret": float(big["ret"].min()), "n": int(big.loc[worst, "n"])}
            if worst
            else None
        ),
        "by_year": {
            str(int(y)): {
                "n": int(r.n),
                "mean_excess": float(r.mean_excess),
                "win": float(r.win),
                "fill": float(r.fill),
            }
            for y, r in by_year.iterrows()
        },
    }


def verdict(base: dict, at_2c: dict) -> dict:
    t_min = min(base["t_event"], base["t_week"])
    years = [base["by_year"].get(y) for y in ("2024", "2025", "2026")]
    gates = {
        "G1": base["mean_excess"] > 0 and base["return_on_capital_excess_ann"] >= 0.05,
        "G2": t_min >= BONFERRONI_Z_18,
        "G3": base["worst_month"] is not None and base["worst_month"]["ret"] >= -0.10,
        "G4": all(y is not None and y["n"] >= 200 and y["mean_excess"] > 0 for y in years),
        "G5": at_2c["mean_excess"] > 0,
    }
    others = gates["G1"] and gates["G3"] and gates["G4"] and gates["G5"]
    label = (
        "PASS"
        if others and gates["G2"]
        else ("PROVISIONAL" if others and t_min >= 1.65 else "FAIL")
    )
    return {"gates": gates, "t_min": t_min, "verdict": label}


def calibration(uni: pd.DataFrame, hist: dict[str, list]) -> list[dict]:
    """Descriptive: first-touch bets in 5-cent bands, win rate against fill price (no costs)."""
    out = []
    edges = [round(0.50 + 0.05 * k, 2) for k in range(11)]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        bets = make_bets(uni, hist, (lo, hi), upper_open=hi < 1.0)
        bets = bets[bets["fill"] < NO_FILL_AT]
        out.append(
            {
                "band": f"{lo:.2f}-{hi:.2f}",
                "n": int(len(bets)),
                "mean_fill": float(bets["fill"].mean()),
                "win_rate": float(bets["payout"].mean()),
            }
        )
    return out


def main() -> None:
    listing = fetch_listing()
    uni = universe(listing)
    print(f"listing {len(listing)}, universe {len(uni)}", flush=True)
    hist = fetch_histories(uni)
    with_hist = sum(1 for m in uni["id"] if len(hist.get(m) or []) >= 2)
    irx = mc.load_yahoo("^IRX")["close"] / 100.0

    results = {}
    for name, band in BANDS.items():
        # PM1 [0.90, 0.97], PM2 (0.97, 0.99], PM3 [0.70, 0.90)
        bets = make_bets(
            uni, hist, band, lower_open=name.startswith("PM2"), upper_open=name.startswith("PM3")
        )
        base = score(bets, COST, irx)
        sens = {f"{c * 100:g}c": score(bets, c, irx) for c in COST_SENSITIVITY}
        results[name] = {"at_1c": base, "sensitivity": sens, **verdict(base, sens["2c"])}
        v = results[name]
        print(
            f"{name}: bets {base['bets']} events {base['events']} fill {base['mean_fill']:.3f} win {base['win_rate']:.3f} "
            f"excess/bet {base['mean_excess']:+.4f} t_event {base['t_event']:+.2f} t_week {base['t_week']:+.2f} "
            f"RoC excess {base['return_on_capital_excess_ann']:+.1%} worst month {base['worst_month']} -> {v['verdict']} {v['gates']}",
            flush=True,
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "listing": len(listing),
                "universe": len(uni),
                "universe_with_history": with_hist,
                "universe_by_end_year": {
                    str(int(y)): int(n)
                    for y, n in uni.groupby(uni["endDate"].dt.year).size().items()
                },
                "results": results,
                "calibration": calibration(uni, hist),
                "bonferroni_z": BONFERRONI_Z_18,
            },
            indent=1,
        )
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
