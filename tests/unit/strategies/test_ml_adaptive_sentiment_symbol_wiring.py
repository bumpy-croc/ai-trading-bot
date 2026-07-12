"""Regression tests for GH #1002.

``create_ml_adaptive_strategy`` and ``create_ml_sentiment_strategy`` had no
``symbol`` parameter at all, and the ``MLSignalGenerator`` component they both
construct had no ``symbol`` attribute — so ``call_strategy_factory``'s
``inspect.signature`` check always found "does this factory accept symbol?"
to be False, silently dropping the trading symbol for every caller (live
runner, backtest CLI, experiments framework), unlike the sibling
``ml_basic``/``hyper_growth`` strategies which already threaded it through.
"""

import pytest

from src.strategies.ml_adaptive import create_ml_adaptive_strategy
from src.strategies.ml_sentiment import create_ml_sentiment_strategy

pytestmark = [pytest.mark.unit, pytest.mark.fast]


class TestMlAdaptiveSymbolWiring:
    def test_symbol_reaches_signal_generator(self):
        strategy = create_ml_adaptive_strategy(symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    def test_bare_call_defaults_to_btcusdt(self):
        """Existing unspecified-symbol callers must see no behavior change."""
        strategy = create_ml_adaptive_strategy()

        assert strategy.signal_generator.symbol == "BTCUSDT"


class TestMlSentimentSymbolWiring:
    def test_symbol_reaches_signal_generator(self):
        strategy = create_ml_sentiment_strategy(symbol="ETHUSDT")

        assert strategy.signal_generator.symbol == "ETHUSDT"

    def test_bare_call_defaults_to_btcusdt(self):
        """Existing unspecified-symbol callers must see no behavior change."""
        strategy = create_ml_sentiment_strategy()

        assert strategy.signal_generator.symbol == "BTCUSDT"
