from datetime import UTC, datetime

import pytest

from src.database.manager import DatabaseManager
from src.database.models import Base, CorrelationMatrix, PortfolioExposure

pytestmark = pytest.mark.unit


def test_correlation_models_crud(monkeypatch):
    # Force SQLite fallback by giving an invalid Postgres URL and disabling integration
    monkeypatch.setenv("DATABASE_URL", "postgresql://invalid")
    # Instantiate manager (will fallback to SQLite in-memory)
    db = DatabaseManager(database_url="postgresql://invalid")
    # Create tables explicitly for SQLite
    engine = db.engine
    assert engine is not None
    Base.metadata.create_all(engine)

    with db.get_session() as session:
        row = CorrelationMatrix(
            symbol_pair="BTCUSDT-ETHUSDT",
            correlation_value=0.85,
            p_value=0.01,
            sample_size=120,
            last_updated=datetime.now(UTC),
            window_days=30,
        )
        session.add(row)
        px = PortfolioExposure(
            correlation_group="crypto_majors",
            total_exposure=0.12,
            position_count=2,
            symbols=["BTCUSDT", "ETHUSDT"],
            last_updated=datetime.now(UTC),
        )
        session.add(px)
        session.commit()

        # Query back
        rows = session.query(CorrelationMatrix).all()
        assert len(rows) == 1
        exposures = session.query(PortfolioExposure).all()
        assert len(exposures) == 1
