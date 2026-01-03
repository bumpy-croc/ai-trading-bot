from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.data_providers.feargreed_provider import FearGreedProvider

pytestmark = pytest.mark.unit


def sample_api_payload():
    # Two days of data
    now = int(datetime(2024, 6, 2, tzinfo=UTC).timestamp())
    prev = int(datetime(2024, 6, 1, tzinfo=UTC).timestamp())
    return {
        "name": "Fear and Greed Index",
        "data": [
            {"timestamp": str(prev), "value": "30", "value_classification": "Fear"},
            {"timestamp": str(now), "value": "60", "value_classification": "Greed"},
        ],
        "metadata": {"error": None},
    }


@patch("requests.get")
def test_feargreed_provider_load_and_features(mock_get):
    resp = Mock()
    resp.json.return_value = sample_api_payload()
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    prov = FearGreedProvider()
    assert not prov.data.empty
    cols = prov.data.columns
    # Ensure engineered features exist
    for c in [
        "sentiment_primary",
        "sentiment_momentum",
        "sentiment_volatility",
        "sentiment_ma_3",
        "sentiment_ma_7",
        "sentiment_ma_14",
        "sentiment_extreme_positive",
        "sentiment_extreme_negative",
    ]:
        assert c in cols


@patch("requests.get")
def test_feargreed_historical_and_resample(mock_get):
    resp = Mock()
    resp.json.return_value = sample_api_payload()
    resp.raise_for_status.return_value = None
    mock_get.return_value = resp

    prov = FearGreedProvider()
    start = datetime(2024, 6, 1, tzinfo=UTC)
    end = datetime(2024, 6, 3, tzinfo=UTC)
    df = prov.get_historical_sentiment("BTCUSDT", start, end)
    assert not df.empty
    # Resample daily via aggregate
    agg = prov.aggregate_sentiment(df, window="1D")
    assert not agg.empty
    assert set(df.columns).issubset(set(agg.columns))
