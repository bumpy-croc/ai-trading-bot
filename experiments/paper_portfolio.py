"""Value the paper portfolio managed by Claude and score it against a world index fund.

Preregistration: docs/research/experiments/2026-11-24_paper-portfolio-prereg.md
Decisions:       docs/research/paper_portfolio/decisions/NNN_YYYY-MM-DD.json

Standard library only, so it runs in a fresh container with no installs.

    python3 experiments/paper_portfolio.py            # value and score; writes ledger.json, ledger.md
    python3 experiments/paper_portfolio.py funds      # recent returns of every allowed fund
    python3 experiments/paper_portfolio.py selftest   # mechanics on made-up prices, no network
"""

from __future__ import annotations

import json
import math
import sys
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "docs" / "research" / "paper_portfolio"
DECISIONS = BOOK / "decisions"
LONDON = ZoneInfo("Europe/London")

UNIVERSE = {
    "VUAG": "S&P 500",
    "CNX1": "Nasdaq 100",
    "VUKG": "FTSE 100",
    "VMIG": "FTSE 250",
    "VERG": "Europe excluding the UK",
    "VJPB": "Japan",
    "VFEG": "Emerging markets",
    "IWFV": "World value",
    "IWFM": "World momentum",
    "IWFQ": "World quality",
    "CSH2": "Sterling cash fund",
    "IGLS": "Gilts up to 5 years",
    "IGLT": "All gilts",
    "INXG": "Index-linked gilts",
    "VAGS": "Global bonds hedged to pounds",
    "IBTM": "US Treasuries 7 to 10 years",
    "SGLN": "Gold",
    "IUKP": "UK property shares",
}
BENCH = "VWRP"
CASH_FUND = "CSH2"
POT = 5000.0
COST = 0.0010  # of every pound bought or sold
MAX_WEIGHT = 0.35
CUTOFF = time(16, 30)  # London; a decision recorded later fills at the next close
END = date(2027, 5, 26)
STOP_FALL = 0.20
BAD_MOVE = 0.15
AHEAD = 0.03
FALL_ALLOWANCE = 0.03
WEEK = 5  # trading days


# --------------------------------------------------------------------------- prices


def fetch(symbol: str, start: date, now: datetime) -> dict[date, float]:
    """Daily adjusted closes from Yahoo. Today's bar is dropped until the London close has printed."""
    p1 = int(datetime(start.year, start.month, start.day, tzinfo=UTC).timestamp())
    p2 = int((now + timedelta(days=2)).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(symbol)}.L"
        f"?period1={p1}&period2={p2}&interval=1d&events=div"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last: Exception | None = None
    for _ in range(4):
        try:
            with urllib.request.urlopen(req, timeout=40) as resp:
                res = json.load(resp)["chart"]["result"][0]
            break
        except Exception as exc:  # noqa: BLE001 - network hiccup: retry, then fail loudly
            last = exc
    else:
        raise RuntimeError(f"could not fetch {symbol}: {last}")
    adj = res["indicators"]["adjclose"][0]["adjclose"]
    today = now.astimezone(LONDON)
    out: dict[date, float] = {}
    for ts, value in zip(res["timestamp"], adj, strict=True):
        if value is None:
            continue
        day = datetime.fromtimestamp(ts, LONDON).date()
        if day == today.date() and today.time() < time(16, 45):
            continue
        out[day] = float(value)
    return out


def check_prices(prices: dict[str, dict[date, float]]) -> None:
    for symbol, series in prices.items():
        days = sorted(series)
        for a, b in zip(days, days[1:], strict=False):
            move = series[b] / series[a] - 1.0
            if abs(move) > BAD_MOVE:
                raise RuntimeError(f"{symbol} moved {move:+.1%} on {b}: check the price by hand")


# --------------------------------------------------------------------------- decisions


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def validate(decision: dict) -> None:
    w = decision.get("weights")
    if w is None:
        return
    unknown = set(w) - set(UNIVERSE)
    if unknown:
        raise ValueError(f"decision {decision['id']}: funds not allowed: {sorted(unknown)}")
    if any(v < 0 for v in w.values()):
        raise ValueError(f"decision {decision['id']}: negative weight")
    if sum(w.values()) > 1.0 + 1e-9:
        raise ValueError(f"decision {decision['id']}: weights sum to {sum(w.values()):.4f}")
    over = {k: v for k, v in w.items() if v > MAX_WEIGHT + 1e-9 and k != CASH_FUND}
    if over:
        raise ValueError(f"decision {decision['id']}: above {MAX_WEIGHT:.0%}: {over}")


def load_decisions(folder: Path = DECISIONS) -> list[dict]:
    out = []
    for path in sorted(folder.glob("*.json")):
        d = json.loads(path.read_text())
        d["file"] = path.name
        validate(d)
        out.append(d)
    out.sort(key=lambda d: parse_time(d["logged_at"]))
    return out


def fill_day(logged_at: datetime, trading_days: list[date]) -> date | None:
    """The first trading day whose London close comes after the decision was recorded."""
    local = logged_at.astimezone(LONDON)
    first = local.date() if local.time() < CUTOFF else local.date() + timedelta(days=1)
    for day in trading_days:
        if day >= first:
            return day
    return None


# --------------------------------------------------------------------------- simulation


def price_on(series: dict[date, float], day: date) -> float:
    """The close on `day`, or the latest one before it if the fund did not print that day."""
    while day not in series:
        day -= timedelta(days=1)
        if day < min(series):
            raise RuntimeError("no price on or before the requested day")
    return series[day]


def simulate(prices: dict[str, dict[date, float]], decisions: list[dict]) -> dict:
    days_all = sorted(prices[BENCH])
    fills = []
    for d in decisions:
        day = fill_day(parse_time(d["logged_at"]), days_all)
        fills.append((day, d))
    filled = [(day, d) for day, d in fills if day is not None]
    if not filled:
        return {
            "status": "no decision has reached a close yet",
            "pending": [d["file"] for d in decisions],
        }
    start = filled[0][0]
    days = [x for x in days_all if x >= start]
    by_day: dict[date, list[dict]] = {}
    for day, d in filled:
        by_day.setdefault(day, []).append(d)

    holdings: dict[str, float] = {}  # pounds in each fund
    cash = POT
    bench = POT * (1.0 - COST)
    ref_mix = {BENCH: 0.6 * POT * (1.0 - COST), "VAGS": 0.4 * POT * (1.0 - COST)}
    ref_cash = POT * (1.0 - COST)
    rows, log, costs = [], [], 0.0
    prev = None
    for day in days:
        if prev is not None:
            for k in holdings:
                holdings[k] *= price_on(prices[k], day) / price_on(prices[k], prev)
            bench *= price_on(prices[BENCH], day) / price_on(prices[BENCH], prev)
            for k in ref_mix:
                ref_mix[k] *= price_on(prices[k], day) / price_on(prices[k], prev)
            ref_cash *= price_on(prices[CASH_FUND], day) / price_on(prices[CASH_FUND], prev)
        for d in by_day.get(day, []):
            entry = {
                "id": d["id"],
                "file": d["file"],
                "logged_at": d["logged_at"],
                "filled_at_close_of": str(day),
            }
            if d.get("weights") is None:
                entry.update({"change": False, "traded": 0.0, "cost": 0.0})
            else:
                total = cash + sum(holdings.values())
                target = {k: v * total for k, v in d["weights"].items() if v > 0}
                traded = sum(
                    abs(target.get(k, 0.0) - holdings.get(k, 0.0))
                    for k in set(target) | set(holdings)
                )
                cost = COST * traded
                after = total - cost
                holdings = {k: v * after for k, v in d["weights"].items() if v > 0}
                cash = after - sum(holdings.values())
                costs += cost
                entry.update(
                    {
                        "change": True,
                        "traded": round(traded, 2),
                        "cost": round(cost, 2),
                        "weights": d["weights"],
                    }
                )
            log.append(entry)
        rows.append(
            {
                "day": day,
                "portfolio": cash + sum(holdings.values()),
                "benchmark": bench,
                "mix_60_40": sum(ref_mix.values()),
                "cash_fund": ref_cash,
            }
        )
        prev = day

    def fall(key: str) -> float:
        peak, worst = 0.0, 0.0
        for r in rows:
            peak = max(peak, r[key])
            worst = max(worst, 1.0 - r[key] / peak)
        return worst

    last = rows[-1]
    total = last["portfolio"]
    weekly = rows[::WEEK]
    diffs = [
        (b["portfolio"] / a["portfolio"] - 1.0) - (b["benchmark"] / a["benchmark"] - 1.0)
        for a, b in zip(weekly, weekly[1:], strict=False)
    ]
    tstat = None
    if len(diffs) >= 3:
        mean = sum(diffs) / len(diffs)
        sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1))
        tstat = mean / sd * math.sqrt(len(diffs)) if sd > 0 else None
    ret_p, ret_b = total / POT - 1.0, last["benchmark"] / POT - 1.0
    fall_p, fall_b = fall("portfolio"), fall("benchmark")
    finished = last["day"] >= END
    verdict = None
    if finished:
        gap = ret_p - ret_b
        if gap >= AHEAD and fall_p <= fall_b + FALL_ALLOWANCE:
            verdict = "AHEAD"
        elif gap <= -AHEAD:
            verdict = "BEHIND"
        else:
            verdict = "NO CLEAR DIFFERENCE"
    return {
        "status": "finished" if finished else "running",
        "start_close": str(start),
        "as_of_close": str(last["day"]),
        "end_close": str(END),
        "trading_days": len(rows),
        "portfolio_value": round(total, 2),
        "benchmark_value": round(last["benchmark"], 2),
        "mix_60_40_value": round(last["mix_60_40"], 2),
        "cash_fund_value": round(last["cash_fund"], 2),
        "portfolio_return": round(ret_p, 5),
        "benchmark_return": round(ret_b, 5),
        "gap_points": round((ret_p - ret_b) * 100, 2),
        "portfolio_largest_fall": round(fall_p, 5),
        "benchmark_largest_fall": round(fall_b, 5),
        "weekly_gap_tstat": None if tstat is None else round(tstat, 2),
        "weeks": len(diffs),
        "stop_rule_hit": fall_p > STOP_FALL,
        "trading_costs": round(costs, 2),
        "decisions": log,
        "pending": [d["file"] for day, d in fills if day is None],
        "current_weights": {k: round(v / total, 4) for k, v in sorted(holdings.items())}
        | {"uninvested": round(cash / total, 4)},
        "verdict": verdict,
        "daily": [
            {"day": str(r["day"]), **{k: round(r[k], 2) for k in r if k != "day"}} for r in rows
        ],
    }


# --------------------------------------------------------------------------- output


def write_ledger(result: dict) -> None:
    BOOK.mkdir(parents=True, exist_ok=True)
    (BOOK / "ledger.json").write_text(json.dumps(result, indent=1) + "\n")
    lines = ["# Paper portfolio ledger", ""]
    if "portfolio_value" not in result:
        lines += [result["status"], ""]
    else:
        lines += [
            f"Status: {result['status']}. Valued at the close of {result['as_of_close']} "
            f"(started at the close of {result['start_close']}, ends at the close of {result['end_close']}).",
            "",
            "| | Value | Return | Largest fall |",
            "|---|---|---|---|",
            f"| Managed portfolio | £{result['portfolio_value']:,.2f} | {result['portfolio_return']:+.2%} | {result['portfolio_largest_fall']:.2%} |",
            f"| World index fund (benchmark) | £{result['benchmark_value']:,.2f} | {result['benchmark_return']:+.2%} | {result['benchmark_largest_fall']:.2%} |",
            f"| 60/40 mix, left alone | £{result['mix_60_40_value']:,.2f} | {result['mix_60_40_value'] / POT - 1:+.2%} | |",
            f"| Cash fund | £{result['cash_fund_value']:,.2f} | {result['cash_fund_value'] / POT - 1:+.2%} | |",
            "",
            f"Gap to benchmark: {result['gap_points']:+.2f} points. Weekly t-statistic: {result['weekly_gap_tstat']} "
            f"over {result['weeks']} weeks. Trading costs so far: £{result['trading_costs']:.2f}.",
            "",
            "Current weights: "
            + ", ".join(f"{k} {v:.1%}" for k, v in result["current_weights"].items() if v > 0.0005)
            + ".",
            "",
            "| Decision | Recorded (UTC) | Filled at close of | Change | Traded | Cost |",
            "|---|---|---|---|---|---|",
        ]
        for d in result["decisions"]:
            lines.append(
                f"| {d['file']} | {d['logged_at']} | {d['filled_at_close_of']} | "
                f"{'yes' if d['change'] else 'no change'} | £{d['traded']:,.2f} | £{d['cost']:.2f} |"
            )
        if result["pending"]:
            lines += ["", "Recorded, not yet filled: " + ", ".join(result["pending"]) + "."]
        if result["verdict"]:
            lines += ["", f"**Verdict: {result['verdict']}**"]
    (BOOK / "ledger.md").write_text("\n".join(lines) + "\n")


def run() -> None:
    now = datetime.now(UTC)
    decisions = load_decisions()
    if not decisions:
        sys.exit("no decision files")
    first = parse_time(decisions[0]["logged_at"]).date() - timedelta(days=10)
    symbols = sorted(
        {BENCH, CASH_FUND, "VAGS"} | {k for d in decisions for k in (d.get("weights") or {})}
    )
    prices = {s: fetch(s, first, now) for s in symbols}
    check_prices(prices)
    result = simulate(prices, decisions)
    write_ledger(result)
    print(json.dumps({k: v for k, v in result.items() if k != "daily"}, indent=1))


def funds() -> None:
    now = datetime.now(UTC)
    start = now.date() - timedelta(days=400)
    print(
        f"{'':6}{'fund':32}{'1w':>7}{'1m':>7}{'3m':>7}{'6m':>7}{'12m':>7}{'vol':>7}{'from high':>10}  last close"
    )
    for s, name in {BENCH: "World index (benchmark)", **UNIVERSE}.items():
        series = fetch(s, start, now)
        days = sorted(series)
        v = [series[d] for d in days]
        rets = [b / a - 1.0 for a, b in zip(v, v[1:], strict=False)]

        def back(n: int, v: list[float] = v) -> str:
            return f"{v[-1] / v[-1 - n] - 1.0:7.1%}" if len(v) > n else f"{'':>7}"

        vol = math.sqrt(sum(x * x for x in rets[-252:]) / len(rets[-252:])) * math.sqrt(252)
        print(
            f"{s:6}{name:32}{back(5)}{back(21)}{back(63)}{back(126)}{back(252)}{vol:7.1%}{v[-1] / max(v[-252:]) - 1.0:10.1%}  {days[-1]}"
        )


def selftest() -> None:
    d0 = date(2026, 11, 25)
    days = [d0 + timedelta(days=i) for i in range(0, 30) if (d0 + timedelta(days=i)).weekday() < 5]
    flat = {d: 100.0 for d in days}
    up = {d: 100.0 * (1.0 + 0.01 * i) for i, d in enumerate(days)}
    prices = {
        BENCH: dict(up),
        CASH_FUND: dict(flat),
        "VAGS": dict(flat),
        "VUAG": dict(up),
        "VUKG": dict(flat),
    }

    # cut-off: 16:29 London fills the same day, 16:31 the next trading day, a Friday evening the Monday
    assert fill_day(datetime(2026, 11, 25, 16, 29, tzinfo=LONDON), days) == date(2026, 11, 25)
    assert fill_day(datetime(2026, 11, 25, 16, 31, tzinfo=LONDON), days) == date(2026, 11, 26)
    assert fill_day(datetime(2026, 11, 27, 22, 0, tzinfo=LONDON), days) == date(2026, 11, 30)
    assert fill_day(datetime(2026, 11, 24, 22, 55, tzinfo=UTC), days) == date(2026, 11, 25)

    # all in the cash fund at flat prices: the pot less one purchase cost
    one = [{"id": 1, "file": "a", "logged_at": "2026-11-24T22:00:00Z", "weights": {CASH_FUND: 1.0}}]
    r = simulate(prices, one)
    assert abs(r["portfolio_value"] - POT * (1 - COST)) < 0.01, r["portfolio_value"]

    # 35% in a fund that tracks the benchmark, the rest uninvested: 35% of the benchmark's gain
    two = [{"id": 1, "file": "a", "logged_at": "2026-11-24T22:00:00Z", "weights": {"VUAG": 0.35}}]
    r = simulate(prices, two)
    gain = up[days[-1]] / up[days[0]] - 1.0
    want = POT - COST * 0.35 * POT + 0.35 * (POT - COST * 0.35 * POT) * gain
    assert abs(r["portfolio_value"] - want) < 0.01, (r["portfolio_value"], want)
    assert abs(r["benchmark_value"] - POT * (1 - COST) * (1 + gain)) < 0.01

    # a switch pays the cost on both sides; "no change" pays nothing
    three = two + [
        {"id": 2, "file": "b", "logged_at": "2026-11-30T06:41:00Z", "weights": None},
        {"id": 3, "file": "c", "logged_at": "2026-12-07T06:41:00Z", "weights": {"VUKG": 0.35}},
    ]
    r = simulate(prices, three)
    assert [x["change"] for x in r["decisions"]] == [True, False, True]
    assert r["decisions"][2]["traded"] > 2 * 0.35 * POT * 0.99 and r["decisions"][1]["cost"] == 0.0

    for bad in (
        {"VWRP": 0.5},
        {"VUAG": 0.36},
        {"VUAG": 0.3, "VUKG": 0.3, "CSH2": 0.5},
        {"VUAG": -0.1},
    ):
        try:
            validate({"id": 9, "weights": bad})
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    validate({"id": 9, "weights": {CASH_FUND: 1.0}})

    jump = {BENCH: {days[0]: 100.0, days[1]: 120.0}}
    try:
        check_prices(jump)
    except RuntimeError:
        pass
    else:
        raise AssertionError("bad price not caught")
    print("selftest ok")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"run": run, "funds": funds, "selftest": selftest}[mode]()
