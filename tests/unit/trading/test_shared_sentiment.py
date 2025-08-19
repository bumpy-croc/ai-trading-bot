from datetime import datetime, timedelta

import pandas as pd

from src.trading.shared.sentiment import apply_live_sentiment, merge_historical_sentiment


class DummySentimentProvider:
    def get_historical_sentiment(self, symbol, start, end):
        idx = pd.date_range(start=start, periods=3, freq="H")
        return pd.DataFrame({"sentiment_score": [0.1, 0.2, 0.3]}, index=idx)

    def aggregate_sentiment(self, df, window):
        return df

    def get_live_sentiment(self):
        return {"sentiment_primary": 0.4, "sentiment_confidence": 0.8}


def test_merge_historical_sentiment():
    start = datetime(2024, 1, 1)
    idx = pd.date_range(start=start, periods=5, freq="H")
    df = pd.DataFrame(
        {"open": [1] * 5, "high": [1] * 5, "low": [1] * 5, "close": [1] * 5, "volume": [1] * 5},
        index=idx,
    )
    sp = DummySentimentProvider()
    out = merge_historical_sentiment(df, sp, "BTCUSDT", "1h", start, start + timedelta(hours=5))
    assert "sentiment_score" in out.columns
    assert (
        out["sentiment_score"].isna().sum() == 2 or out["sentiment_score"].ffill().isna().sum() == 0
    )


def test_apply_live_sentiment_adds_freshness():
    now = datetime.now()
    idx = pd.date_range(end=now, periods=10, freq="H")
    df = pd.DataFrame(
        {
            "open": [1] * 10,
            "high": [1] * 10,
            "low": [1] * 10,
            "close": [1] * 10,
            "volume": [1] * 10,
        },
        index=idx,
    )
    sp = DummySentimentProvider()
    out = apply_live_sentiment(df, sp, recent_hours=4)
    assert "sentiment_freshness" in out.columns
    # Last few rows should be 1
    assert out["sentiment_freshness"].tail(4).eq(1).all()
