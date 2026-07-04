"""Kelly-cap and bear-unsafe strategy warnings (#805)."""

from __future__ import annotations

import logging

import pytest

from src.config.constants import DEFAULT_MAX_KELLY_FRACTION
from src.strategies.kelly_momentum import create_kelly_momentum_strategy

pytestmark = pytest.mark.unit


def test_kelly_fraction_clamped_to_bear_cap(caplog):
    with caplog.at_level(logging.WARNING):
        strategy = create_kelly_momentum_strategy(kelly_fraction=0.9)
    assert strategy.position_sizer.kelly_fraction == DEFAULT_MAX_KELLY_FRACTION
    assert any("clamping" in r.message for r in caplog.records)


def test_kelly_fraction_within_cap_unchanged():
    strategy = create_kelly_momentum_strategy(kelly_fraction=0.25)
    assert strategy.position_sizer.kelly_fraction == 0.25


def test_momentum_leverage_warns_bear_unsafe(caplog):
    from src.strategies.momentum_leverage import create_momentum_leverage_strategy

    with caplog.at_level(logging.WARNING):
        create_momentum_leverage_strategy()
    assert any("NOT recommended for bear" in r.message for r in caplog.records)


def test_hyper_growth_warns_bear_unsafe(caplog):
    from src.strategies.hyper_growth import create_hyper_growth_strategy

    with caplog.at_level(logging.WARNING):
        create_hyper_growth_strategy()
    assert any("NOT recommended for bear" in r.message for r in caplog.records)
