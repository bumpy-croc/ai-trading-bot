import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.live.exit_policies import ModelOutageExitPolicy, PositionSnapshot


def _make_df(length: int = 30, base_price: float = 100.0, atr: float = 1.0) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=length, freq='1h')
    close = np.full(length, base_price, dtype=float)
    # simple OHLCV and ATR columns
    df = pd.DataFrame({
        'open': close,
        'high': close * 1.001,
        'low': close * 0.999,
        'close': close,
        'volume': np.full(length, 1000.0, dtype=float),
        'atr': np.full(length, atr, dtype=float),
    }, index=idx)
    return df


def test_should_exit_time_based():
    policy = ModelOutageExitPolicy(enabled=True)
    df = _make_df()
    index = len(df) - 1

    snapshot = PositionSnapshot(
        symbol='BTCUSDT',
        side='long',
        entry_price=100.0,
        entry_time=datetime.now() - timedelta(hours=48),  # older than max hold
        stop_loss=None,
        take_profit=None,
    )

    assert policy.should_exit(df, index, snapshot) is True


def test_should_exit_volatility_guard():
    policy = ModelOutageExitPolicy(enabled=True)
    # Create a df with low median ATR and a spike at the end
    df = _make_df(length=30, atr=1.0)
    df.iloc[-1, df.columns.get_loc('atr')] = 10.0  # spike
    index = len(df) - 1

    snapshot = PositionSnapshot(
        symbol='BTCUSDT',
        side='long',
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=None,
        take_profit=None,
    )

    assert policy.should_exit(df, index, snapshot) is True


def test_adjust_protective_stops_long():
    policy = ModelOutageExitPolicy(enabled=True)
    df = _make_df()
    index = len(df) - 1
    # push price up to trigger breakeven and trailing
    df.iloc[index, df.columns.get_loc('close')] = 105.0

    snapshot = PositionSnapshot(
        symbol='BTCUSDT',
        side='long',
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=None,
        take_profit=None,
    )

    new_sl, new_tp = policy.adjust_protective_stops(df, index, snapshot)
    assert new_sl is not None
    # should be at least breakeven or a trailed stop below current
    assert new_sl >= snapshot.entry_price * 0.99


def test_adjust_protective_stops_short():
    policy = ModelOutageExitPolicy(enabled=True)
    df = _make_df()
    index = len(df) - 1
    # push price down to trigger breakeven and trailing for short
    df.iloc[index, df.columns.get_loc('close')] = 95.0

    snapshot = PositionSnapshot(
        symbol='BTCUSDT',
        side='short',
        entry_price=100.0,
        entry_time=datetime.now(),
        stop_loss=None,
        take_profit=None,
    )

    new_sl, new_tp = policy.adjust_protective_stops(df, index, snapshot)
    assert new_sl is not None
    # for shorts, stop should be at or below entry after adjustments
    assert new_sl <= snapshot.entry_price * 1.01


