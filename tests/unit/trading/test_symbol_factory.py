"""Tests for shared symbol-parsing helpers in src.trading.symbols.factory."""

from __future__ import annotations

import pytest

from src.trading.symbols.factory import base_asset_from_symbol


@pytest.mark.unit
@pytest.mark.fast
class TestBaseAssetFromSymbol:
    """Quote-suffix strip used to size closing SELLs against free base holdings.

    A wrong base asset makes the -2010 free-base cap read the wrong balance, so
    the longest-first ordering (BUSD before USD) and the unknown-quote fallback
    are load-bearing, not cosmetic.
    """

    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("ETHUSDT", "ETH"),
            ("BTCUSDT", "BTC"),
            ("ETHUSDC", "ETH"),
            ("ETHUSD", "ETH"),
            # BUSD must be stripped as a unit — checking USD first would leave "ETHB".
            ("ETHBUSD", "ETH"),
            # Multi-character/base with digits still resolves.
            ("1000SATSUSDT", "1000SATS"),
        ],
    )
    def test_strips_known_quote_suffix(self, symbol, expected):
        assert base_asset_from_symbol(symbol) == expected

    def test_longest_quote_wins_over_shorter_substring(self):
        """BUSD is checked before USD so a BUSD pair keeps its full base asset."""
        assert base_asset_from_symbol("XRPBUSD") == "XRP"

    @pytest.mark.parametrize("symbol", ["BTCEUR", "SOLGBP", "ADABTC"])
    def test_unknown_quote_returns_symbol_unchanged(self, symbol):
        assert base_asset_from_symbol(symbol) == symbol

    @pytest.mark.parametrize("symbol", ["USDT", "USD", "USDC"])
    def test_bare_quote_with_no_base_returns_unchanged(self, symbol):
        """``len(symbol) > len(quote)`` guards against stripping to an empty base."""
        assert base_asset_from_symbol(symbol) == symbol
