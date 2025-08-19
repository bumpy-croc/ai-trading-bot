from src.database.manager import DatabaseManager


def test_record_and_fetch_optimization_cycle(monkeypatch):
    # Ensure SQLite fallback by not setting DATABASE_URL to postgres
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")

    db = DatabaseManager()

    cycle_id = db.record_optimization_cycle(
        strategy_name="ml_basic",
        symbol="BTCUSDT",
        timeframe="1h",
        baseline_metrics={"annualized_return": 10.0, "max_drawdown": 8.0},
        candidate_params={"MlBasic.stop_loss_pct": -0.05},
        candidate_metrics={"annualized_return": 12.0, "max_drawdown": 7.5},
        validator_report={"passed": True, "p_value": 0.05, "effect_size": 0.8},
        decision="apply",
        session_id=None,
    )

    assert isinstance(cycle_id, int) and cycle_id > 0

    rows = db.fetch_optimization_cycles(limit=10)
    assert isinstance(rows, list)
    assert any(r["id"] == cycle_id for r in rows)
    found = next(r for r in rows if r["id"] == cycle_id)
    assert found["strategy_name"] == "ml_basic"
    assert found["symbol"] == "BTCUSDT"
    assert found["decision"] == "apply"
