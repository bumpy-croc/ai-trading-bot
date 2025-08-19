from src.trading.shared.sizing import normalize_position_size


def test_normalize_fraction_mode():
    assert normalize_position_size(0.2, 1000, mode="fraction") == 0.2
    assert normalize_position_size(1.5, 1000, mode="fraction") == 1.0
    assert normalize_position_size(-1, 1000, mode="fraction") == 0.0


def test_normalize_notional_mode():
    assert normalize_position_size(200, 1000, mode="notional") == 0.2
    assert normalize_position_size(2000, 1000, mode="notional") == 1.0
    assert normalize_position_size(-5, 1000, mode="notional") == 0.0
    assert normalize_position_size(100, 0, mode="notional") == 0.0
