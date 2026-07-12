"""Tests that the backtest CLI threads --symbol into strategy factories.

Backtest-live parity counterpart of the live runner symbol wiring tests:
both runners must hand the trading symbol to ML strategies so model
registry selection matches the traded pair.
"""

import argparse
from unittest.mock import patch

import pytest

from cli.commands.backtest import _handle, _load_strategy

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]


class TestLoadStrategySymbolThreading:
    def test_threads_symbol_to_hyper_growth(self):
        with patch(
            "cli.commands.backtest.create_hyper_growth_strategy", autospec=True
        ) as mock_create:
            _load_strategy("hyper_growth", symbol="ETHUSDT")

            mock_create.assert_called_once_with(symbol="ETHUSDT")

    def test_threads_symbol_to_ml_basic(self):
        with patch("cli.commands.backtest.create_ml_basic_strategy", autospec=True) as mock_create:
            _load_strategy("ml_basic", symbol="ETHUSDT")

            mock_create.assert_called_once_with(symbol="ETHUSDT")

    def test_factory_without_symbol_param_called_plain(self):
        """Factories that do not accept symbol are invoked without it."""
        with patch(
            "cli.commands.backtest.create_momentum_leverage_strategy", autospec=True
        ) as mock_create:
            _load_strategy("momentum_leverage", symbol="ETHUSDT")

            _, kwargs = mock_create.call_args
            assert "symbol" not in kwargs


class TestHandleSymbolWiring:
    def test_handle_passes_cli_symbol_to_load_strategy(self, monkeypatch):
        recorded = {}

        def fake_load(name, symbol=None, model_version=None):
            recorded["call"] = (name, symbol, model_version)
            raise RuntimeError("stop after strategy load")

        monkeypatch.setattr("cli.commands.backtest._load_strategy", fake_load)
        ns = argparse.Namespace(
            strategy="hyper_growth",
            symbol="ETHUSDT",
            timeframe="1h",
            days=30,
            start=None,
            end=None,
            initial_balance=1000.0,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            max_position_size=None,
            use_sentiment=False,
            no_cache=True,
            cache_ttl=24,
            log_to_db=False,
            provider="binance",
            disable_engine_sl=False,
        )

        rc = _handle(ns)

        assert rc == 1
        # model_version None: no pin flags on the namespace means the pin
        # machinery must stay entirely out of the way (GH #988).
        assert recorded["call"] == ("hyper_growth", "ETHUSDT", None)
