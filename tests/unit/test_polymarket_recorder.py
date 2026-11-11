"""Tests for tools/polymarket_recorder.py and experiments/polymarket_forward_score.py.

Context: docs/research/experiments/2026-11-11_polymarket-forward-underdogs-prereg.md.
"""

from __future__ import annotations

import gzip
import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rec = _load("polymarket_recorder", _ROOT / "tools" / "polymarket_recorder.py")
sys.path.insert(0, str(_ROOT / "experiments"))  # the scorer imports markets_common
try:
    score = _load("polymarket_forward_score", _ROOT / "experiments" / "polymarket_forward_score.py")
finally:
    sys.path.remove(str(_ROOT / "experiments"))

NOW = datetime(2026, 11, 11, 12, 15, 18, tzinfo=UTC)
TS = "2026-11-11T12:15:18Z"


def _market(market_id: str, **overrides) -> dict:
    market = {
        "id": market_id,
        "question": "Will it happen?",
        "outcomes": '["Yes", "No"]',
        "bestBid": 0.14,
        "bestAsk": 0.15,
        "lastTradePrice": 0.15,
        "volumeNum": 400_000.0,
        "liquidityNum": 20_000.0,
        "startDate": "2026-10-01T00:00:00Z",
        "endDate": "2026-12-31T00:00:00Z",
        "acceptingOrders": True,
        "feesEnabled": True,
        "feeType": "politics_fees",
        "feeSchedule": {"rate": 0.04, "exponent": 1},
        "negRisk": False,
        "orderPriceMinTickSize": 0.01,
        "events": [{"id": "900"}],
    }
    market.update(overrides)
    return market


def _response(payload) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestMarketRow:
    def test_flattens_listing_fields(self):
        row = rec.market_row(_market("1"), TS)
        assert set(row) == set(rec.COLUMNS)
        assert row["event_id"] == "900"
        assert row["n_outcomes"] == 2
        assert row["best_bid"] == 0.14 and row["best_ask"] == 0.15
        assert row["fee_rate"] == 0.04 and row["fee_exponent"] == 1
        assert row["fees_enabled"] is True

    def test_missing_fields_stay_empty(self):
        row = rec.market_row({"id": "2", "outcomes": "not json"}, TS)
        assert row["event_id"] is None
        assert row["n_outcomes"] == 0
        assert row["best_bid"] is None and row["fee_rate"] is None
        assert row["fees_enabled"] is False
        assert row["question"] == ""


class TestFetchOpenMarkets:
    def test_follows_cursor_and_deduplicates(self):
        session = MagicMock()
        session.get.side_effect = [
            _response({"markets": [_market("1"), _market("2")], "next_cursor": "abc"}),
            _response({"markets": [_market("2"), _market("3")], "next_cursor": None}),
        ]
        markets = rec.fetch_open_markets(session)
        assert [m["id"] for m in markets] == ["1", "2", "3"]
        first, second = (call.kwargs["params"] for call in session.get.call_args_list)
        assert "after_cursor" not in first
        assert second["after_cursor"] == "abc"
        assert first["closed"] == "false" and first["volume_num_min"] == rec.MIN_VOLUME_USD

    def test_listing_that_never_ends_is_an_error(self, monkeypatch):
        monkeypatch.setattr(rec, "MAX_PAGES", 3)
        session = MagicMock()
        session.get.return_value = _response({"markets": [_market("1")], "next_cursor": "x"})
        with pytest.raises(RuntimeError, match="did not end"):
            rec.fetch_open_markets(session)

    def test_unreachable_listing_is_an_error(self, monkeypatch):
        monkeypatch.setattr(rec.time, "sleep", lambda _s: None)
        session = MagicMock()
        session.get.side_effect = rec.requests.ConnectionError("down")
        with pytest.raises(RuntimeError, match="unreachable"):
            rec.fetch_open_markets(session)
        assert session.get.call_count == rec.HTTP_ATTEMPTS


class TestWriteSnapshot:
    def test_path_and_bytes_are_deterministic(self, tmp_path):
        rows = [rec.market_row(_market(str(i)), TS) for i in (1, 2)]
        first = rec.write_snapshot(rows, tmp_path / "a", NOW)
        second = rec.write_snapshot(rows, tmp_path / "b", NOW)
        assert first.relative_to(tmp_path / "a").as_posix() == (
            "2026/11/snapshot_20261111T121518Z.csv.gz"
        )
        assert first.read_bytes() == second.read_bytes()
        lines = gzip.decompress(first.read_bytes()).decode().splitlines()
        assert lines[0] == ",".join(rec.COLUMNS)
        assert len(lines) == 3


class TestMain:
    def test_writes_snapshot(self, tmp_path, monkeypatch):
        markets = [_market(str(i)) for i in range(rec.MIN_PLAUSIBLE_MARKETS)]
        monkeypatch.setattr(rec, "fetch_open_markets", lambda _session: markets)
        assert rec.main(["--out-dir", str(tmp_path)]) == rec.EXIT_OK
        assert len(list(tmp_path.rglob("snapshot_*.csv.gz"))) == 1

    def test_blind_when_listing_is_implausibly_small(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rec, "fetch_open_markets", lambda _session: [_market("1")])
        assert rec.main(["--out-dir", str(tmp_path)]) == rec.EXIT_BLIND
        assert not list(tmp_path.rglob("*.csv.gz"))

    def test_blind_when_listing_is_unreachable(self, tmp_path, monkeypatch):
        def boom(_session):
            raise RuntimeError("listing unreachable")

        monkeypatch.setattr(rec, "fetch_open_markets", boom)
        assert rec.main(["--out-dir", str(tmp_path)]) == rec.EXIT_BLIND


def _snapshots(tmp_path: Path, runs: list[tuple[datetime, list[dict]]]) -> pd.DataFrame:
    for now, markets in runs:
        ts = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        rec.write_snapshot([rec.market_row(m, ts) for m in markets], tmp_path, now)
    return score.load_snapshots(tmp_path)


class TestBuildBets:
    def test_underdog_side_price_and_fee(self, tmp_path):
        snaps = _snapshots(
            tmp_path,
            [
                (
                    NOW,
                    [
                        _market("1"),  # Yes is the underdog at its ask, 0.15
                        _market("2", bestBid=0.84, bestAsk=0.85, feesEnabled=False),  # No at 0.16
                    ],
                )
            ],
        )
        bets = score.build_bets(snaps).set_index("id")
        assert set(bets["band"]) == {"U2 0.10-0.30"}
        assert bool(bets.loc["1", "side_yes"]) is True
        assert bets.loc["1", "u"] == pytest.approx(0.15)
        assert bets.loc["1", "fee"] == pytest.approx(0.04 * 0.15 * 0.85)
        assert bool(bets.loc["2", "side_yes"]) is False
        assert bets.loc["2", "u"] == pytest.approx(0.16)
        assert bets.loc["2", "fee"] == 0.0
        assert bets.loc["2", "fav_ask"] == pytest.approx(0.85)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"volumeNum": 249_999.0},
            {"outcomes": '["A", "B", "C"]'},
            {"acceptingOrders": False},
            {"endDate": "2026-11-12T06:00:00Z"},  # under 24 hours away
            {"bestBid": 0.09, "bestAsk": 0.15},  # spread 0.06
            {"bestBid": None},
        ],
    )
    def test_ineligible_markets_are_skipped(self, tmp_path, overrides):
        snaps = _snapshots(tmp_path, [(NOW, [_market("1", **overrides)])])
        assert score.build_bets(snaps).empty

    def test_entry_is_the_first_snapshot_in_the_band(self, tmp_path):
        later = datetime(2026, 11, 11, 18, 15, 18, tzinfo=UTC)
        snaps = _snapshots(
            tmp_path,
            [
                (NOW, [_market("1", bestBid=0.39, bestAsk=0.40)]),
                (later, [_market("1", bestBid=0.19, bestAsk=0.20)]),
                (datetime(2026, 11, 12, 0, 15, 18, tzinfo=UTC), [_market("1")]),
            ],
        )
        bets = score.build_bets(snaps).set_index("band")
        assert len(bets) == 2  # one bet per band the market has visited
        assert bets.loc["U3 0.30-0.50", "u"] == pytest.approx(0.40)
        assert bets.loc["U2 0.10-0.30", "u"] == pytest.approx(0.20)
        assert bets.loc["U2 0.10-0.30", "snapshot_ts"] == pd.Timestamp(later)

    def test_band_edges(self):
        u = pd.Series([0.03, 0.10, 0.30, 0.50])
        assert score.in_band(u, "U1 0.03-0.10").tolist() == [True, True, False, False]
        assert score.in_band(u, "U2 0.10-0.30").tolist() == [False, False, True, False]
        assert score.in_band(u, "U3 0.30-0.50").tolist() == [False, False, False, False]


def _bets(n: int, win_every: int, closed: str = "2026-12-20") -> pd.DataFrame:
    """n no-fee Yes-underdog bets at 0.20 over 37 entry days, one event each."""
    entry = pd.Timestamp("2026-11-12", tz="UTC")
    return pd.DataFrame(
        {
            "band": "U2 0.10-0.30",
            "id": [str(i) for i in range(n)],
            "event": [f"e{i}" for i in range(n)],
            "snapshot_ts": [entry + pd.Timedelta(days=i % 37) for i in range(n)],
            "side_yes": True,
            "u": 0.20,
            "fee": 0.0,
            "fav_ask": 0.81,
            "fav_fee": 0.0,
            "fees_enabled": False,
            "volume": 500_000.0,
            "yes_payout": [1.0 if i % win_every == 0 else 0.0 for i in range(n)],
            "closed": pd.Timestamp(closed, tz="UTC"),
        }
    )


IRX = pd.Series(0.0, index=pd.date_range("2026-11-01", "2027-02-28"))


class TestScoreBand:
    def test_fairly_priced_underdogs_fail(self):
        result = score.score_band(_bets(3000, win_every=5), IRX, "2027-01-10", "2027-02-07")
        assert result["bets"] == 3000 and result["events"] == 3000
        assert result["win_rate"] == pytest.approx(0.20)
        assert result["mean_excess"] == pytest.approx(0.0, abs=1e-9)
        assert result["gates"]["G1"] is False and result["gates"]["G3"] is True
        assert result["verdict"] == "FAIL"

    def test_large_edge_with_enough_bets_passes(self):
        result = score.score_band(_bets(3000, win_every=3), IRX, "2027-01-10", "2027-02-07")
        assert result["mean_excess"] == pytest.approx(1 / 3 / 0.20 - 1.0)
        assert all(result["gates"].values())
        assert result["verdict"] == "PASS"
        # the favourite side of the same bets loses
        assert result["favourite_reference_mean_excess"] < 0

    def test_too_few_bets_is_inconclusive(self):
        result = score.score_band(_bets(120, win_every=3), IRX, "2027-01-10", "2027-02-07")
        assert result["gates"]["G3"] is False
        assert result["verdict"] == "INCONCLUSIVE"

    def test_open_markets_and_late_entries_are_excluded(self):
        bets = _bets(400, win_every=5)
        bets.loc[:9, "closed"] = pd.NaT  # still open
        bets.loc[10:19, "closed"] = pd.Timestamp("2027-02-08", tz="UTC")  # resolved too late
        bets.loc[20:29, "snapshot_ts"] = pd.Timestamp("2027-01-11", tz="UTC")  # entered too late
        result = score.score_band(bets, IRX, "2027-01-10", "2027-02-07")
        assert result["entered"] == 390
        assert result["still_open_excluded"] == 20
        assert result["bets"] == 370

    def test_tbill_accrual_is_charged(self):
        irx = pd.Series(0.0365, index=IRX.index)
        free = score.score_band(_bets(400, win_every=5), IRX, "2027-01-10", "2027-02-07")
        charged = score.score_band(_bets(400, win_every=5), irx, "2027-01-10", "2027-02-07")
        assert charged["mean_excess"] < free["mean_excess"]
        # 0.01% a day for between 2 and 39 days, so well under 1% of stake
        assert free["mean_excess"] - charged["mean_excess"] == pytest.approx(0.002, abs=0.002)


class TestFetchResolutions:
    def test_keeps_only_final_payouts(self, monkeypatch):
        payload = [
            {"id": 1, "outcomePrices": '["1", "0"]', "closedTime": "2026-12-01 10:00:00+00"},
            {"id": 2, "outcomePrices": '["0.5", "0.5"]', "closedTime": "2026-12-02 10:00:00+00"},
            {"id": 3, "outcomePrices": '["0.97", "0.03"]', "closedTime": "2026-12-03 10:00:00+00"},
            {"id": 4, "outcomePrices": '["0", "1"]', "closedTime": None},
        ]
        session = MagicMock()
        session.headers = {}
        resp = _response(payload)
        resp.status_code = 200
        session.get.return_value = resp
        monkeypatch.setattr(score.requests, "Session", lambda: session)
        monkeypatch.setattr(score.time, "sleep", lambda _s: None)
        res = score.fetch_resolutions(["1", "2", "3", "4"])
        assert res["id"].tolist() == ["1", "2"]
        assert res["yes_payout"].tolist() == [1.0, 0.5]
        assert str(res["closed"].dt.tz) == "UTC"
