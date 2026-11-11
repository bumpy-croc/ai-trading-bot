"""Score the Polymarket underdog forward test, as preregistered in
docs/research/experiments/2026-11-11_polymarket-forward-underdogs-prereg.md.

Reads the snapshots written by tools/polymarket_recorder.py (a checkout of the
`data/polymarket-recorder` branch), builds one bet per market per band at the first
eligible snapshot, looks up how each market resolved, and applies the gates.

Usage:
  git fetch origin data/polymarket-recorder
  git worktree add /tmp/pm-data origin/data/polymarket-recorder
  python experiments/polymarket_forward_score.py --snapshots /tmp/pm-data/snapshots

Not to be run for a verdict before the preregistered scoring date (2027-02-08).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import markets_common as mc
import numpy as np
import pandas as pd
import requests

OUT = Path(__file__).resolve().parent / "markets_results" / "polymarket_forward.json"
GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
MIN_VOLUME = 250_000
MIN_HOURS_TO_END = 24
MAX_SPREAD = 0.05
BANDS = {"U1 0.03-0.10": (0.03, 0.10), "U2 0.10-0.30": (0.10, 0.30), "U3 0.30-0.50": (0.30, 0.50)}
ENTRY_CUTOFF, RESOLVE_CUTOFF = "2027-01-10", "2027-02-07"
EXT_ENTRY_CUTOFF, EXT_RESOLVE_CUTOFF = "2027-04-11", "2027-05-09"
BONFERRONI_Z_21 = 2.8227  # one-sided 5% / 21 candidates
MIN_BETS, MIN_EVENTS, POWER_BETS = 300, 100, 2500
MIN_EDGE = 0.02


def load_snapshots(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("snapshot_*.csv.gz"))
    if not files:
        raise SystemExit(f"no snapshots under {root}")
    df = pd.concat(
        [pd.read_csv(f, dtype={"id": str, "event_id": str}) for f in files], ignore_index=True
    )
    df["snapshot_ts"] = pd.to_datetime(df["snapshot_ts"], utc=True)
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce", format="mixed")
    return df


def in_band(u: pd.Series, name: str) -> pd.Series:
    lo, hi = BANDS[name]
    if name.startswith("U1"):
        return (u >= lo) & (u <= hi)
    if name.startswith("U2"):
        return (u > lo) & (u <= hi)
    return (u > lo) & (u < hi)


def build_bets(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per market per band: the first eligible snapshot with the underdog's ask in the band."""
    s = snaps.copy()
    bid, ask = s["best_bid"].astype(float), s["best_ask"].astype(float)
    eligible = (
        (s["n_outcomes"] == 2)
        & s["accepting_orders"].astype(str).str.lower().eq("true")
        & (s["volume"] >= MIN_VOLUME)
        & (s["end_date"] >= s["snapshot_ts"] + pd.Timedelta(hours=MIN_HOURS_TO_END))
        & (bid > 0)
        & (ask < 1)
        & (bid < ask)
        & (ask - bid <= MAX_SPREAD + 1e-9)
    )
    s = s[eligible].copy()
    yes_ask, no_ask = s["best_ask"].astype(float), 1.0 - s["best_bid"].astype(float)
    s["side_yes"] = yes_ask <= no_ask
    s["u"] = np.minimum(yes_ask, no_ask)
    s["fav_ask"] = np.maximum(yes_ask, no_ask)
    rate = (
        s["fee_rate"]
        .astype(float)
        .where(s["fees_enabled"].astype(str).str.lower().eq("true"), 0.0)
        .fillna(0.0)
    )
    exponent = s["fee_exponent"].astype(float).fillna(1.0)
    s["fee"] = rate * (s["u"] * (1.0 - s["u"])) ** exponent
    s["fav_fee"] = rate * (s["fav_ask"] * (1.0 - s["fav_ask"])) ** exponent
    s["event"] = s["event_id"].fillna("m" + s["id"])
    out = []
    for name in BANDS:
        hit = (
            s[in_band(s["u"], name)].sort_values("snapshot_ts").drop_duplicates("id", keep="first")
        )
        out.append(hit.assign(band=name))
    cols = [
        "band",
        "id",
        "event",
        "snapshot_ts",
        "side_yes",
        "u",
        "fee",
        "fav_ask",
        "fav_fee",
        "fees_enabled",
        "volume",
    ]
    return pd.concat(out, ignore_index=True)[cols]


def fetch_resolutions(ids: list[str]) -> pd.DataFrame:
    """Final payout of the first outcome and the close time, for markets that have closed."""
    session = requests.Session()
    session.headers["User-Agent"] = mc.UA
    rows = []
    for i in range(0, len(ids), 50):
        params = [("closed", "true"), ("limit", 50)] + [("id", m) for m in ids[i : i + 50]]
        for attempt in range(5):
            resp = session.get(GAMMA_MARKETS, params=params, timeout=60)
            if resp.status_code == 200:
                break
            time.sleep(2**attempt)
        resp.raise_for_status()
        for m in resp.json():
            prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
            # Final payouts only (1, 0, or an even split): a closed market that has not
            # settled yet still shows its last traded prices and counts as open.
            final = len(prices) == 2 and sorted(prices) in ([0.0, 1.0], [0.5, 0.5])
            if final and m.get("closedTime"):
                rows.append(
                    {"id": str(m["id"]), "yes_payout": prices[0], "closed": m["closedTime"]}
                )
        time.sleep(0.2)
    res = pd.DataFrame(rows, columns=["id", "yes_payout", "closed"])
    res["closed"] = pd.to_datetime(res["closed"], utc=True, format="mixed")
    return res


def _tstat(values: pd.Series, groups: pd.Series) -> float:
    means = values.groupby(groups).mean()
    if len(means) < 3 or means.std(ddof=1) == 0:
        return 0.0
    return float(means.mean() / means.std(ddof=1) * math.sqrt(len(means)))


def score_band(bets: pd.DataFrame, irx: pd.Series, entry_cutoff: str, resolve_cutoff: str) -> dict:
    """`bets` already joined to resolutions (yes_payout, closed may be NaN/NaT for open markets)."""
    entered = bets[
        bets["snapshot_ts"] < pd.Timestamp(entry_cutoff, tz="UTC") + pd.Timedelta(days=1)
    ]
    done = entered["closed"].notna() & (
        entered["closed"] < pd.Timestamp(resolve_cutoff, tz="UTC") + pd.Timedelta(days=1)
    )
    b = entered[done].copy()
    out: dict = {
        "entered": int(len(entered)),
        "still_open_excluded": int((~done).sum()),
        "bets": int(len(b)),
    }
    if b.empty:
        return {**out, "gates": {}, "verdict": "INCONCLUSIVE"}
    days = (b["closed"] - b["snapshot_ts"]).dt.total_seconds() / 86400.0 + 1.0
    entry_day = b["snapshot_ts"].dt.tz_convert(None).dt.normalize()
    rf = irx.reindex(irx.index.union(entry_day.unique())).ffill().loc[entry_day].to_numpy()
    payout = np.where(b["side_yes"], b["yes_payout"], 1.0 - b["yes_payout"])
    stake = b["u"] + b["fee"]
    b["excess"] = payout / stake - 1.0 - rf * days / 365.0
    fav_stake = b["fav_ask"] + b["fav_fee"]
    fav_excess = (1.0 - payout) / fav_stake - 1.0 - rf * days / 365.0
    half = entry_day <= entry_day.min() + (entry_day.max() - entry_day.min()) / 2
    t_min = min(_tstat(b["excess"], b["event"]), _tstat(b["excess"], entry_day))
    cap_days = float((stake * days).sum())
    gates = {
        "G1": bool(b["excess"].mean() >= MIN_EDGE),
        "G2": bool(t_min >= BONFERRONI_Z_21),
        "G3": bool(len(b) >= MIN_BETS and b["event"].nunique() >= MIN_EVENTS),
        "G4": bool(b.loc[half, "excess"].mean() > 0 and b.loc[~half, "excess"].mean() > 0),
    }
    base = gates["G1"] and gates["G3"] and gates["G4"]
    if base and gates["G2"]:
        verdict = "PASS"
    elif base and t_min >= 1.65:
        verdict = "PROVISIONAL"
    elif not gates["G3"] or (gates["G1"] and gates["G4"] and len(b) < POWER_BETS):
        verdict = "INCONCLUSIVE"
    else:
        verdict = "FAIL"
    by_fee = b.groupby(b["fees_enabled"].astype(str).str.lower().eq("true"))["excess"].agg(
        ["size", "mean"]
    )
    return {
        **out,
        "events": int(b["event"].nunique()),
        "mean_price_paid": float(stake.mean()),
        "win_rate": float(np.mean(payout)),
        "mean_excess": float(b["excess"].mean()),
        "t_event": _tstat(b["excess"], b["event"]),
        "t_entry_day": _tstat(b["excess"], entry_day),
        "return_on_capital_excess_ann": float(
            (payout - stake).sum() / cap_days * 365.0 - (rf * stake * days).sum() / cap_days
        ),
        "first_half_mean": float(b.loc[half, "excess"].mean()),
        "second_half_mean": float(b.loc[~half, "excess"].mean()),
        "by_fee_market": {
            ("fee" if k else "no_fee"): {"n": int(r["size"]), "mean_excess": float(r["mean"])}
            for k, r in by_fee.iterrows()
        },
        "favourite_reference_mean_excess": float(np.mean(fav_excess)),
        "gates": gates,
        "verdict": verdict,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument(
        "--extension", action="store_true", help="use the one preregistered extension window"
    )
    args = parser.parse_args()
    entry_cutoff, resolve_cutoff = (
        (EXT_ENTRY_CUTOFF, EXT_RESOLVE_CUTOFF) if args.extension else (ENTRY_CUTOFF, RESOLVE_CUTOFF)
    )

    snaps = load_snapshots(args.snapshots)
    bets = build_bets(snaps)
    res = fetch_resolutions(sorted(bets["id"].unique(), key=int))
    bets = bets.merge(res, on="id", how="left")
    irx = mc.load_yahoo("^IRX")["close"] / 100.0
    results = {
        name: score_band(bets[bets["band"] == name], irx, entry_cutoff, resolve_cutoff)
        for name in BANDS
    }
    report = {
        "snapshots": int(snaps["snapshot_ts"].nunique()),
        "first_snapshot": str(snaps["snapshot_ts"].min()),
        "last_snapshot": str(snaps["snapshot_ts"].max()),
        "entry_cutoff": entry_cutoff,
        "resolve_cutoff": resolve_cutoff,
        "results": results,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1))
    for name, r in results.items():
        print(
            name,
            {
                k: r.get(k)
                for k in (
                    "entered",
                    "bets",
                    "events",
                    "mean_excess",
                    "t_event",
                    "t_entry_day",
                    "verdict",
                )
            },
        )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
