import pandas as pd

from src.indicators import technical


def test_calculate_atr_adds_column():
    df = pd.DataFrame(
        {
            "high": [2.0, 3.0, 4.5],
            "low": [1.0, 1.5, 3.5],
            "close": [1.5, 2.5, 4.0],
        }
    )

    result = technical.calculate_atr(df, period=2)

    assert "atr" in result
    assert not result["atr"].isna().all()
