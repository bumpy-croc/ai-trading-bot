import numpy as np
import pandas as pd
import pytest

from src.prediction import create_minimal_engine
from src.prediction.engine import PredictionEngine, PredictionResult


def _make_small_ohlcv(n: int = 130) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    base = 100.0
    rng = np.random.default_rng(42)
    close = base + rng.normal(0, 0.5, size=n).cumsum()
    open_ = close + rng.normal(0, 0.1, size=n)
    high = np.maximum(open_, close) + rng.random(size=n)
    low = np.minimum(open_, close) - rng.random(size=n)
    volume = rng.uniform(100, 500, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


@pytest.mark.fast
def test_engine_creation_smoke():
    engine = create_minimal_engine()
    assert isinstance(engine, PredictionEngine)
    assert engine.feature_pipeline is not None
    assert engine.model_registry is not None


@pytest.mark.fast
def test_engine_health_check_smoke():
    engine = create_minimal_engine()
    health = engine.health_check()
    assert health["status"] in {"healthy", "degraded", "error"}
    assert "components" in health


@pytest.mark.fast
def test_predict_smoke_skip_if_no_models():
    engine = create_minimal_engine()
    models = engine.get_available_models()
    if not models:
        pytest.skip("No prediction models available; skipping predict smoke test")

    data = _make_small_ohlcv(140)
    result = engine.predict(data)
    assert isinstance(result, PredictionResult)
    # Either successful prediction or an informative error without crashing
    if result.error is None:
        assert isinstance(result.price, (int, float))
        assert isinstance(result.confidence, (int, float))
        assert result.direction in [-1, 0, 1]
    else:
        assert isinstance(result.error, str)
