import pandas as pd
from src.trading.shared.indicators import extract_indicators, extract_sentiment_data, extract_ml_predictions


def test_extract_indicators_basic():
    df = pd.DataFrame([
        {"open":1,"high":2,"low":0.5,"close":1.5,"volume":10, "rsi":50, "macd":0.1}
    ])
    got = extract_indicators(df, 0)
    assert got["open"] == 1.0
    assert got["close"] == 1.5
    assert got["rsi"] == 50.0
    assert got["macd"] == 0.1


def test_extract_sentiment_data():
    df = pd.DataFrame([
        {"sentiment_score": 0.2, "sentiment_primary": 0.1, "close": 1.0}
    ])
    got = extract_sentiment_data(df, 0)
    assert got["sentiment_score"] == 0.2
    assert got["sentiment_primary"] == 0.1


def test_extract_ml_predictions():
    df = pd.DataFrame([
        {"onnx_pred": 2.0, "ml_prediction": 2.0, "prediction_confidence": 0.5}
    ])
    got = extract_ml_predictions(df, 0)
    assert got["onnx_pred"] == 2.0
    assert got["prediction_confidence"] == 0.5