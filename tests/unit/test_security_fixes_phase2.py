"""
Tests for Phase 2 security fixes.

Tests for:
- SEC-004: Input validation for data providers
- SEC-005: Database SSL enforcement
- SEC-008/009: Admin application enhancements (CSRF, rate limiting)

Note: SEC-006 (Logging redaction) tests moved to
tests/unit/infrastructure/logging/test_config.py::TestSensitiveDataFilter
"""

import os
from unittest.mock import patch

import pytest


def test_sec_004_binance_credentials_validation():
    """SEC-004: Validate Binance credentials are properly formatted."""
    from src.data_providers.binance_provider import BinanceProvider

    # Both valid - should work
    api_key = "a" * 40
    api_secret = "b" * 40
    result = BinanceProvider._validate_credentials(api_key, api_secret)
    assert result == (api_key, api_secret)

    # Both missing - should return empty strings
    result = BinanceProvider._validate_credentials(None, None)
    assert result == ("", "")


def test_sec_004_binance_credentials_mismatch():
    """SEC-004: Reject if only one credential provided."""
    from src.data_providers.binance_provider import BinanceProvider

    with pytest.raises(ValueError):
        BinanceProvider._validate_credentials("a" * 40, None)

    with pytest.raises(ValueError):
        BinanceProvider._validate_credentials(None, "b" * 40)


def test_sec_004_binance_credentials_too_short():
    """SEC-004: Reject credentials that are too short."""
    from src.data_providers.binance_provider import BinanceProvider

    with pytest.raises(ValueError):
        BinanceProvider._validate_credentials("short", "also_short")


def test_sec_004_coinbase_credentials_validation():
    """SEC-004: Validate Coinbase credentials are properly formatted."""
    from src.data_providers.coinbase_provider import CoinbaseProvider

    # All valid - should work
    result = CoinbaseProvider._validate_credentials("a" * 40, "b" * 40, "c" * 10)
    assert result == ("a" * 40, "b" * 40, "c" * 10)

    # All missing - should return empty strings
    result = CoinbaseProvider._validate_credentials(None, None, None)
    assert result == ("", "", "")


def test_sec_004_coinbase_credentials_partial():
    """SEC-004: Reject if not all Coinbase credentials provided."""
    from src.data_providers.coinbase_provider import CoinbaseProvider

    with pytest.raises(ValueError):
        CoinbaseProvider._validate_credentials("a" * 40, "b" * 40, None)


def test_sec_005_database_ssl_production():
    """SEC-005: Verify SSL is required in production."""
    with patch.dict(os.environ, {"ENV": "production"}, clear=False):
        os.environ.pop("DATABASE_SSL_MODE", None)
        from src.database.manager import DatabaseManager

        manager = DatabaseManager()
        config = manager._get_engine_config()

        assert config["connect_args"]["sslmode"] == "require"


def test_sec_005_database_ssl_development():
    """SEC-005: Verify SSL can be flexible in development."""
    with patch.dict(os.environ, {"ENV": "development"}, clear=False):
        os.environ.pop("DATABASE_SSL_MODE", None)
        from src.database.manager import DatabaseManager

        manager = DatabaseManager()
        config = manager._get_engine_config()

        assert config["connect_args"]["sslmode"] == "prefer"


def test_sec_005_database_ssl_override():
    """SEC-005: Verify DATABASE_SSL_MODE environment variable can override."""
    with patch.dict(
        os.environ, {"ENV": "development", "DATABASE_SSL_MODE": "require"}, clear=False
    ):
        from src.database.manager import DatabaseManager

        manager = DatabaseManager()
        config = manager._get_engine_config()

        assert config["connect_args"]["sslmode"] == "require"


def test_sec_008_csrf_imports():
    """SEC-008: Verify CSRF protection is available."""
    from src.database.admin_ui.app import csrf

    assert csrf is not None
    assert hasattr(csrf, "init_app")


def test_sec_009_rate_limiter_imports():
    """SEC-009: Verify rate limiter is available."""
    from src.database.admin_ui.app import limiter

    assert limiter is not None
    assert hasattr(limiter, "limit")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
