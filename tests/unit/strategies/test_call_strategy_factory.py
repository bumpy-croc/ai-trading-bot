"""Tests for the shared symbol-threading helpers in ``src.strategies``.

``factory_accepts_symbol`` is the single source of truth every runner (live,
backtest CLI, ExperimentRunner) uses to decide whether a strategy factory
should receive ``symbol=``. These tests pin its behavior directly against
synthetic factories so the "does this factory accept symbol" check stays
correct independent of any specific strategy module.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.strategies import call_strategy_factory, factory_accepts_symbol

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]


def _factory_with_symbol(name: str = "x", *, symbol: str | None = None):
    return {"name": name, "symbol": symbol}


def _factory_without_symbol(name: str = "x"):
    return {"name": name}


def _factory_with_var_keyword(name: str = "x", **kwargs):
    return {"name": name, **kwargs}


class TestFactoryAcceptsSymbol:
    def test_true_when_symbol_is_a_named_parameter(self):
        assert factory_accepts_symbol(_factory_with_symbol) is True

    def test_false_when_no_symbol_parameter(self):
        """Mirrors real non-ML factories like create_chaos_test_strategy,
        which has no symbol parameter at all."""
        assert factory_accepts_symbol(_factory_without_symbol) is False

    def test_true_when_factory_accepts_var_keyword(self):
        assert factory_accepts_symbol(_factory_with_var_keyword) is True

    def test_false_for_uninspectable_callable(self):
        """Builtins/C callables raise ValueError from inspect.signature; the
        helper must fall back to False rather than propagating that error."""
        assert factory_accepts_symbol(dict) is False


class TestCallStrategyFactory:
    def test_threads_symbol_when_accepted(self):
        result = call_strategy_factory(_factory_with_symbol, symbol="ETHUSDT")
        assert result == {"name": "x", "symbol": "ETHUSDT"}

    def test_omits_symbol_when_not_accepted(self):
        """A factory with no symbol parameter must be called unchanged
        rather than raising TypeError."""
        result = call_strategy_factory(_factory_without_symbol, symbol="ETHUSDT")
        assert result == {"name": "x"}

    def test_none_symbol_never_forwarded(self):
        factory = MagicMock(return_value="strategy")
        result = call_strategy_factory(factory, symbol=None)
        factory.assert_called_once_with()
        assert result == "strategy"
