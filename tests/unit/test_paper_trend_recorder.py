"""Tests for tools/paper_trend_recorder.py and the forward-test modules it shares with the scorer.

Context: docs/research/experiments/2026-11-11_slow-adaptive-trend-prereg.md (Stage B).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_ROOT = Path(__file__).resolve().parents[2]
_SEARCH = _ROOT / "experiments" / "strategy_search"
# The research scripts import each other by bare name ("data", "engine", "strategies"), and
# "strategies" is also a package under src/. Load them, keep the module objects, and take the
# bare names back out of the interpreter so no other test can pick them up by accident.
_BARE = (
    "data",
    "engine",
    "strategies",
    "adaptive",
    "forward_data",
    "paper_score",
    "paper_trend_recorder",
)
_before = {name: sys.modules.get(name) for name in _BARE}


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_path_before = list(sys.path)
try:
    rec = _load("paper_trend_recorder", _ROOT / "tools" / "paper_trend_recorder.py")
    score = _load("paper_score", _SEARCH / "paper_score.py")
    fd = rec.forward_data
    eng = rec.eng
    adaptive = rec.adaptive
    data_mod = fd.data_mod
finally:
    sys.path[:] = _path_before
    for _name, _module in _before.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module

MONDAY = pd.Timestamp("2026-11-09", tz="UTC")
NOW = datetime(2026, 11, 10, 10, 23, tzinfo=UTC)


def _market(
    end: pd.Timestamp, n_days: int = rec.HISTORY_DAYS + 1, n_coins: int = 40, seed: int = 5
):
    """`n_days` up to `end`, cut from one fixed random-walk path per coin, zero funding."""
    full = pd.date_range(
        "2024-01-01", "2026-12-31", freq="D", tz="UTC"
    )  # the same path whatever the window
    rng = np.random.default_rng(seed)
    cols = ["BTCUSDT"] + [f"C{i}USDT" for i in range(n_coins - 1)]
    close = pd.DataFrame(
        100 * np.cumprod(1 + rng.normal(0.0, 0.04, size=(len(full), n_coins)), axis=0),
        index=full,
        columns=cols,
    )
    qvol = pd.DataFrame(rng.uniform(1e6, 1e8, size=(len(full), n_coins)), index=full, columns=cols)
    window = slice(end - pd.Timedelta(days=n_days - 1), end)
    close, qvol = close.loc[window], qvol.loc[window]
    return data_mod.Market(close, close, close, close, qvol, close * 0.0, close * np.nan)


INFO = {"symbols_in_archive": 40, "perps_with_data": 150, "missing_candles_carried_forward": 0}


class TestBuildRecord:
    def test_signal_day_records_scaled_weights(self):
        record = rec.build_record(_market(MONDAY), INFO, MONDAY, ["B0"], NOW)
        book = record["books"]["B0"]
        assert record["signal_day"] is True and record["data_day"] == "2026-11-09"
        assert record["recorded_at"] == "2026-11-10T10:23:00Z"
        assert 0 < len(book["weights"]) <= eng.UNIVERSE_N
        assert all(np.isfinite(w) and w != 0 for w in book["weights"].values())
        assert sum(abs(w) for w in book["weights"].values()) == pytest.approx(
            book["gross"], abs=1e-6
        )
        assert book["gross"] == pytest.approx(
            book["leverage"], abs=1e-6
        )  # the unscaled book has gross 1
        assert 0 < book["gross"] <= eng.MAX_GROSS

    def test_other_days_are_heartbeats(self):
        tuesday = MONDAY + pd.Timedelta(days=1)
        record = rec.build_record(_market(tuesday), INFO, tuesday, ["B0"], NOW)
        assert record["signal_day"] is False and record["books"] == {}
        assert record["perps_with_data"] == 150

    def test_window_is_long_enough_to_match_a_longer_history(self):
        """The recorder sees 420 days. Its weights must equal what 700 days of history give."""
        short = rec.build_record(_market(MONDAY), INFO, MONDAY, ["B0"], NOW)["books"]["B0"]
        long = rec.build_record(_market(MONDAY, n_days=700), INFO, MONDAY, ["B0"], NOW)["books"][
            "B0"
        ]
        assert short["weights"].keys() == long["weights"].keys()
        assert short["leverage"] == pytest.approx(long["leverage"], rel=1e-6)
        for symbol, weight in short["weights"].items():
            assert weight == pytest.approx(long["weights"][symbol], abs=1e-7)

    def test_weights_ignore_everything_after_the_data_day(self):
        market = _market(MONDAY + pd.Timedelta(days=7))
        later = adaptive.build_books(market, ("B0",))["B0"].weights.loc[MONDAY].fillna(0.0)
        recorded = rec.build_record(_market(MONDAY), INFO, MONDAY, ["B0"], NOW)["books"]["B0"][
            "weights"
        ]
        assert later[later != 0].round(8).to_dict() == pytest.approx(recorded, abs=1e-7)


class TestMain:
    def _patch(self, monkeypatch, published=True, perps=150):
        calls = {"fetch": 0}

        def fetch(day, history_days):
            calls["fetch"] += 1
            return _market(day), {**INFO, "perps_with_data": perps}

        monkeypatch.setattr(fd, "day_published", lambda day: published)
        monkeypatch.setattr(fd, "fetch_market", fetch)
        monkeypatch.setattr(rec, "forward_books", lambda: ["B0"])
        return calls

    def test_writes_yesterdays_record(self, tmp_path, monkeypatch):
        self._patch(monkeypatch)
        assert rec.main(["--out-dir", str(tmp_path)], now=NOW) == rec.EXIT_OK
        record = json.loads((tmp_path / "records" / "2026" / "record_2026-11-09.json").read_text())
        assert record["signal_day"] is True and "B0" in record["books"]

    def test_never_rewrites_a_day(self, tmp_path, monkeypatch):
        calls = self._patch(monkeypatch)
        path = rec.record_path(tmp_path, MONDAY)
        path.parent.mkdir(parents=True)
        path.write_text("first record")
        assert rec.main(["--out-dir", str(tmp_path)], now=NOW) == rec.EXIT_OK
        assert path.read_text() == "first record" and calls["fetch"] == 0

    def test_waits_quietly_while_candles_are_unpublished(self, tmp_path, monkeypatch):
        calls = self._patch(monkeypatch, published=False)
        assert rec.main(["--out-dir", str(tmp_path)], now=NOW) == rec.EXIT_OK
        assert calls["fetch"] == 0 and not (tmp_path / "records").exists()

    def test_fails_when_candles_are_still_missing_late_in_the_day(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, published=False)
        late = NOW.replace(hour=rec.GIVE_UP_HOUR_UTC, minute=23)
        assert rec.main(["--out-dir", str(tmp_path)], now=late) == rec.EXIT_BLIND

    def test_fails_without_writing_when_too_few_perps_return_data(self, tmp_path, monkeypatch):
        self._patch(monkeypatch, perps=rec.MIN_PERPS - 1)
        assert rec.main(["--out-dir", str(tmp_path)], now=NOW) == rec.EXIT_BLIND
        assert not (tmp_path / "records").exists()

    def test_fails_when_the_archive_is_unreachable(self, tmp_path, monkeypatch):
        def down(day):
            raise RuntimeError("archive unreachable")

        monkeypatch.setattr(fd, "day_published", down)
        assert rec.main(["--out-dir", str(tmp_path)], now=NOW) == rec.EXIT_BLIND


def _candle(day: pd.Timestamp, close: float = 10.0, volume: float = 1e6) -> pd.DataFrame:
    row = {"open": close, "high": close, "low": close, "close": close, "quote_volume": volume}
    return pd.DataFrame([row], index=pd.DatetimeIndex([day], name="date"))


class TestForwardData:
    def test_fetch_symbol_uses_monthly_files_then_days(self, monkeypatch):
        day = pd.Timestamp("2026-11-03", tz="UTC")
        asked: list[str] = []

        def candles(key: str):
            asked.append(key)
            name = key.rsplit("/", 1)[-1]
            if "/monthly/" in key:
                month = name[len("XUSDT-1d-") : -len(".zip")]
                if month == "2026-10":  # just ended: no monthly file yet
                    return None
                days = pd.date_range(
                    f"{month}-01", periods=pd.Period(month).days_in_month, freq="D", tz="UTC"
                )
                return pd.concat([_candle(d) for d in days])
            return _candle(pd.Timestamp(name[len("XUSDT-1d-") : -len(".zip")], tz="UTC"))

        monkeypatch.setattr(fd, "_candles", candles)
        out = fd.fetch_symbol("XUSDT", day, 90)
        assert out.index[0] == day - pd.Timedelta(days=90) and out.index[-1] == day
        assert len(out) == 91 and out.index.is_unique
        daily = [k for k in asked if "/daily/" in k]
        assert not any(
            "2026-08" in k or "2026-09" in k for k in daily
        )  # complete months come from monthly files
        assert sum("2026-10" in k for k in daily) == 31 and sum("2026-11" in k for k in daily) == 3

    def test_fetch_symbol_is_empty_for_a_symbol_with_no_candle_on_the_day(self, monkeypatch):
        monkeypatch.setattr(fd, "_candles", lambda key: None)
        assert fd.fetch_symbol("XUSDT", MONDAY, 90).empty

    def test_fetch_market_cuts_delisted_tails_and_carries_short_gaps(self, monkeypatch):
        days = pd.date_range(end=MONDAY, periods=10, freq="D")

        def symbol(name, day, history_days):
            frame = pd.concat([_candle(d, close=10.0 + i) for i, d in enumerate(days)])
            if name == "DEADUSDT":  # settled five days ago: flat price, zero volume ever since
                frame.loc[days[5] :, "quote_volume"] = 0.0
            if name == "GAPUSDT":
                frame = frame.drop(days[4])
            return frame

        monkeypatch.setattr(fd, "fetch_symbol", symbol)
        market, info = fd.fetch_market(
            MONDAY, 9, symbols=["LIVEUSDT", "DEADUSDT", "GAPUSDT"], workers=2
        )
        assert market.close["DEADUSDT"].last_valid_index() == days[4]
        assert market.close["GAPUSDT"].loc[days[4]] == market.close["GAPUSDT"].loc[days[3]]
        assert market.qvol["GAPUSDT"].loc[days[4]] == 0.0
        assert info == {
            "symbols_in_archive": 3,
            "perps_with_data": 2,
            "missing_candles_carried_forward": 1,
        }
        assert (market.funding.fillna(0.0) == 0.0).all().all()


class TestScoring:
    def test_record_counts_only_if_written_before_its_fill(self):
        assert score.on_time({"data_day": "2026-11-09", "recorded_at": "2026-11-10T10:23:00Z"})
        assert score.on_time({"data_day": "2026-11-09", "recorded_at": "2026-11-10T23:59:59Z"})
        assert not score.on_time({"data_day": "2026-11-09", "recorded_at": "2026-11-11T00:00:00Z"})

    def test_recorded_targets_rebalance_only_on_recorded_signal_days(self):
        idx = pd.date_range("2026-11-07", periods=6, freq="D", tz="UTC")
        cols = pd.Index(["AUSDT", "BUSDT"])
        records = [
            {
                "data_day": "2026-11-09",
                "signal_day": True,
                "books": {"B0": {"weights": {"AUSDT": 0.2}}},
            },
            {"data_day": "2026-11-10", "signal_day": False, "books": {}},
        ]
        targets = score.recorded_targets(records, "B0", idx, cols)
        assert targets.loc[MONDAY].to_dict() == {"AUSDT": 0.2, "BUSDT": 0.0}
        assert targets.drop(MONDAY).isna().all().all()
        with pytest.raises(RuntimeError, match="missing from the archive"):
            score.recorded_targets(
                [{**records[0], "books": {"B0": {"weights": {"ZUSDT": 0.1}}}}], "B0", idx, cols
            )

    @pytest.mark.parametrize(
        ("weeks", "ret", "dd", "plumbing", "expect"),
        [
            (3, 0.50, 0.21, True, "STOP"),
            (3, 0.01, 0.02, True, "before the first checkpoint"),
            (8, -0.03, 0.05, True, "plumbing passes"),
            (8, 0.03, 0.05, False, "FAILS"),
            (26, 0.06, 0.10, True, "small live trial"),
            (26, 0.06, 0.16, True, "between the bars"),
            (26, -0.01, 0.05, True, "leans against"),
            (52, 0.12, 0.18, True, "live allocation"),
            (52, 0.02, 0.05, True, "below idle dollars"),
            (52, 0.06, 0.05, True, "Alex decides"),
        ],
    )
    def test_checkpoints_follow_the_preregistration(self, weeks, ret, dd, plumbing, expect):
        text = score.checkpoint(
            weeks, {"net_return": ret, "max_drawdown": dd}, idle=0.04, plumbing_ok=plumbing
        )
        assert expect in text

    def test_summarise_measures_drawdown_from_the_starting_capital(self):
        stats = score.summarise(pd.Series([-0.10, 0.05, 0.02]))
        assert stats["max_drawdown"] == pytest.approx(0.10)
        assert stats["net_return"] == pytest.approx(0.9 * 1.05 * 1.02 - 1)
