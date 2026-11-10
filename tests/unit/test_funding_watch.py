"""Tests for tools/funding_watch.py (log.md D-2026-11-10-02)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.fast]

_TOOL = Path(__file__).resolve().parents[2] / "tools" / "funding_watch.py"
_spec = importlib.util.spec_from_file_location("funding_watch", _TOOL)
assert _spec is not None and _spec.loader is not None
fw = importlib.util.module_from_spec(_spec)
sys.modules["funding_watch"] = fw  # dataclasses resolve annotations via sys.modules
_spec.loader.exec_module(fw)

NOW = datetime(2026, 11, 10, 12, tzinfo=UTC)
SINCE = NOW - timedelta(days=7)


def _points(annualised: float, n: int = 21) -> list:
    return [
        fw.FundingPoint(time=NOW - timedelta(hours=8 * i), annualised=annualised) for i in range(n)
    ]


def _response(payload) -> MagicMock:
    resp = MagicMock()
    resp.json.return_value = payload
    resp.raise_for_status.return_value = None
    return resp


class TestAnnualise:
    def test_eight_hour_baseline_is_about_eleven_percent(self):
        # 0.01% per 8h is the standard perp baseline: 0.0001 * 3 * 365 = 10.95%/yr
        assert fw._annualise(0.0001, 8) == pytest.approx(0.1095)

    def test_hourly_rate_annualises_by_8760(self):
        assert fw._annualise(0.0000125, 1) == pytest.approx(0.1095)


class TestFetchers:
    def test_binance_parses_and_annualises(self):
        rows = [{"fundingTime": int(NOW.timestamp() * 1000), "fundingRate": "0.0001"}]
        with patch.object(fw.requests, "get", return_value=_response(rows)):
            pts = fw.fetch_binance("BTCUSDT", SINCE)
        assert pts == [fw.FundingPoint(time=NOW, annualised=pytest.approx(0.1095))]

    def test_okx_drops_rows_older_than_window(self):
        old = NOW - timedelta(days=8)
        body = {
            "code": "0",
            "data": [
                {"fundingTime": str(int(NOW.timestamp() * 1000)), "realizedRate": "0.0002"},
                {"fundingTime": str(int(old.timestamp() * 1000)), "realizedRate": "0.0009"},
            ],
        }
        with patch.object(fw.requests, "get", return_value=_response(body)) as get:
            pts = fw.fetch_okx("BTC-USDT-SWAP", SINCE)
        assert [p.time for p in pts] == [NOW]
        assert pts[0].annualised == pytest.approx(0.219)
        assert get.call_count == 1  # oldest row predates the window, no further paging

    def test_okx_error_code_raises(self):
        with patch.object(
            fw.requests, "get", return_value=_response({"code": "51000", "msg": "x"})
        ):
            with pytest.raises(RuntimeError):
                fw.fetch_okx("BTC-USDT-SWAP", SINCE)

    def test_hyperliquid_is_hourly(self):
        rows = [{"time": int(NOW.timestamp() * 1000), "fundingRate": "0.0000125"}]
        with patch.object(fw.requests, "post", return_value=_response(rows)):
            pts = fw.fetch_hyperliquid("BTC", SINCE)
        assert pts[0].annualised == pytest.approx(0.1095)


class TestReadSymbol:
    def test_falls_through_to_next_source_on_failure(self):
        def blocked(_instrument, _since):
            raise ConnectionError("451 restricted location")

        sources = [("binance", blocked), ("okx", lambda i, s: _points(0.05))]
        reading = fw.read_symbol("BTCUSDT", SINCE, sources=sources)
        assert reading is not None
        assert reading.source == "okx"
        assert reading.mean_annualised == pytest.approx(0.05)

    def test_skips_source_with_too_few_samples(self):
        sources = [("binance", lambda i, s: _points(0.5, n=1)), ("okx", lambda i, s: _points(0.04))]
        reading = fw.read_symbol("ETHUSDT", SINCE, sources=sources)
        assert reading is not None and reading.source == "okx"

    def test_returns_none_when_every_source_fails(self):
        assert fw.read_symbol("BTCUSDT", SINCE, sources=[("binance", lambda i, s: [])]) is None


def _reading(symbol: str, mean: float) -> object:
    return fw.SymbolReading(symbol, "binance", 21, mean, mean)


class TestEvaluate:
    def test_below_threshold_is_ok(self):
        code, _ = fw.evaluate(
            {"BTCUSDT": _reading("BTCUSDT", 0.04), "ETHUSDT": _reading("ETHUSDT", 0.036)}, 0.10
        )
        assert code == fw.EXIT_OK

    def test_either_symbol_above_threshold_alerts(self):
        code, lines = fw.evaluate(
            {"BTCUSDT": _reading("BTCUSDT", 0.04), "ETHUSDT": _reading("ETHUSDT", 0.12)}, 0.10
        )
        assert code == fw.EXIT_ALERT
        assert "ABOVE" in lines[1]

    def test_blind_symbol_reports_blind(self):
        code, _ = fw.evaluate({"BTCUSDT": _reading("BTCUSDT", 0.04), "ETHUSDT": None}, 0.10)
        assert code == fw.EXIT_BLIND

    def test_alert_wins_over_blind(self):
        code, _ = fw.evaluate({"BTCUSDT": _reading("BTCUSDT", 0.2), "ETHUSDT": None}, 0.10)
        assert code == fw.EXIT_ALERT


def test_main_writes_summary_and_json(tmp_path, monkeypatch):
    summary = tmp_path / "summary.md"
    out = tmp_path / "funding.json"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setattr(fw, "read_symbol", lambda sym, since: _reading(sym, 0.15))
    assert fw.main(["--json-out", str(out)]) == fw.EXIT_ALERT
    assert "Funding watch" in summary.read_text()
    assert '"exit_code": 10' in out.read_text()
