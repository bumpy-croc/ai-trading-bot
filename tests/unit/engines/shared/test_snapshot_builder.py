"""Tests for snapshot_builder utilities."""

import math
from datetime import datetime

import pandas as pd

from src.engines.shared.execution.snapshot_builder import (
    build_snapshot_from_candle,
    build_snapshot_from_ohlc,
    build_snapshot_from_price,
    coerce_float,
)


class TestCoerceFloat:
    """Test coerce_float handles edge cases correctly."""

    def test_coerce_valid_float(self) -> None:
        """Valid float values should be returned as-is."""
        assert coerce_float(100.5, 0.0) == 100.5

    def test_coerce_int(self) -> None:
        """Integer values should be converted to float."""
        assert coerce_float(100, 0.0) == 100.0

    def test_coerce_string_number(self) -> None:
        """Numeric strings should be converted to float."""
        assert coerce_float("100.5", 0.0) == 100.5

    def test_coerce_none_returns_fallback(self) -> None:
        """None should return the fallback value."""
        assert coerce_float(None, 99.0) == 99.0

    def test_coerce_invalid_string_returns_fallback(self) -> None:
        """Non-numeric strings should return the fallback value."""
        assert coerce_float("invalid", 99.0) == 99.0

    def test_coerce_nan_returns_fallback(self) -> None:
        """NaN should return the fallback value to prevent state corruption."""
        assert coerce_float(math.nan, 99.0) == 99.0

    def test_coerce_positive_infinity_returns_fallback(self) -> None:
        """Positive infinity should return the fallback value."""
        assert coerce_float(math.inf, 99.0) == 99.0

    def test_coerce_negative_infinity_returns_fallback(self) -> None:
        """Negative infinity should return the fallback value."""
        assert coerce_float(-math.inf, 99.0) == 99.0


class TestBuildSnapshotFromPrice:
    """Test build_snapshot_from_price creates valid snapshots."""

    def test_all_ohlc_set_to_current_price(self) -> None:
        """All OHLC values should equal current price."""
        snapshot = build_snapshot_from_price("BTCUSDT", 50000.0)

        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.last_price == 50000.0
        assert snapshot.high == 50000.0
        assert snapshot.low == 50000.0
        assert snapshot.close == 50000.0


class TestBuildSnapshotFromOhlc:
    """Test build_snapshot_from_ohlc creates valid snapshots."""

    def test_uses_provided_high_low(self) -> None:
        """Snapshot should use provided high/low values."""
        snapshot = build_snapshot_from_ohlc(
            symbol="BTCUSDT",
            current_price=50000.0,
            candle_high=51000.0,
            candle_low=49000.0,
        )

        assert snapshot.high == 51000.0
        assert snapshot.low == 49000.0
        assert snapshot.close == 50000.0

    def test_none_high_low_uses_current_price(self) -> None:
        """None high/low should fall back to current price."""
        snapshot = build_snapshot_from_ohlc(
            symbol="BTCUSDT",
            current_price=50000.0,
            candle_high=None,
            candle_low=None,
        )

        assert snapshot.high == 50000.0
        assert snapshot.low == 50000.0


class TestBuildSnapshotFromCandle:
    """Test build_snapshot_from_candle creates valid snapshots."""

    def test_extracts_ohlcv_from_series(self) -> None:
        """Snapshot should extract OHLCV from pandas Series."""
        candle = pd.Series({
            "open": 100.0,
            "high": 110.0,
            "low": 95.0,
            "close": 105.0,
            "volume": 1000.0,
        })

        snapshot = build_snapshot_from_candle(
            symbol="BTCUSDT",
            current_time=datetime(2024, 1, 1),
            current_price=105.0,
            candle=candle,
        )

        assert snapshot.high == 110.0
        assert snapshot.low == 95.0
        assert snapshot.close == 105.0
        assert snapshot.volume == 1000.0

    def test_nan_in_candle_uses_fallback(self) -> None:
        """NaN values in candle should use current_price as fallback."""
        candle = pd.Series({
            "open": 100.0,
            "high": math.nan,  # NaN should be replaced
            "low": 95.0,
            "close": 105.0,
            "volume": 1000.0,
        })

        snapshot = build_snapshot_from_candle(
            symbol="BTCUSDT",
            current_time=datetime(2024, 1, 1),
            current_price=105.0,
            candle=candle,
        )

        # NaN high should fall back to current_price
        assert snapshot.high == 105.0
        assert snapshot.low == 95.0

    def test_none_candle_uses_current_price(self) -> None:
        """None candle should use current_price for all values."""
        snapshot = build_snapshot_from_candle(
            symbol="BTCUSDT",
            current_time=datetime(2024, 1, 1),
            current_price=105.0,
            candle=None,
        )

        assert snapshot.high == 105.0
        assert snapshot.low == 105.0
        assert snapshot.close == 105.0
