import numpy as np
import pandas as pd

from src.backtesting.utils import (
    compute_performance_metrics,
    extract_indicators,
    extract_ml_predictions,
    extract_sentiment_data,
)


def _sample_df():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h")
    df = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [101, 102, 103, 104, 105],
            "low": [99, 100, 101, 102, 103],
            "close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
            "rsi": [30, 35, 40, 45, 50],
            "macd": [0.1, 0.2, 0.3, 0.4, 0.5],
            "prediction_confidence": [0.6, 0.7, np.nan, 0.8, 0.9],
            "onnx_pred": [101, 102, 103, 104, 105],
            "sentiment_score": [0.1, np.nan, 0.2, 0.3, 0.4],
            "sentiment_confidence": [0.5, 0.6, 0.7, np.nan, 0.9],
        },
        index=idx,
    )
    return df


def test_extract_indicators_basic():
    df = _sample_df()
    result = extract_indicators(df, 1)
    assert isinstance(result, dict)
    # Includes OHLCV
    for k in ["open", "high", "low", "close", "volume"]:
        assert k in result
    # Includes some indicators
    assert "rsi" in result and result["rsi"] == float(df.iloc[1]["rsi"])
    assert "macd" in result


def test_extract_sentiment_data_basic():
    df = _sample_df()
    result = extract_sentiment_data(df, 0)
    assert "sentiment_score" in result
    assert "sentiment_confidence" in result


def test_extract_ml_predictions_basic():
    df = _sample_df()
    result = extract_ml_predictions(df, 4)
    assert "onnx_pred" in result
    assert "prediction_confidence" in result


def test_compute_performance_metrics_shapes():
    # Construct a simple balance history
    idx = pd.date_range("2024-01-01", periods=10, freq="D")
    balance = pd.Series(np.linspace(10000, 11000, len(idx)), index=idx)
    bh = pd.DataFrame({"balance": balance}, index=idx)

    total_ret, max_dd, sharpe, cagr = compute_performance_metrics(
        initial_balance=10000,
        final_balance=11000,
        start=pd.Timestamp("2024-01-01"),
        end=pd.Timestamp("2024-01-10"),
        balance_history=bh,
    )

    assert isinstance(total_ret, float)
    assert isinstance(max_dd, float)
    assert isinstance(sharpe, float)
    assert isinstance(cagr, float)
