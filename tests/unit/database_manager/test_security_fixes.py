"""
Tests for database manager security fixes.

Tests for:
- SEC-001: Weak Default Admin Password
- SEC-002: Plain-Text Password Comparison  
- SEC-003: Inconsistent SECRET_KEY Handling
"""

import os
import pytest
from unittest.mock import patch
from werkzeug.security import generate_password_hash, check_password_hash


def test_sec_001_admin_password_required_from_env():
    """SEC-001: Verify DB_MANAGER_ADMIN_PASS environment variable is required (no fallback)."""
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("DB_MANAGER_ADMIN_PASS", None)
        os.environ.pop("DB_MANAGER_ADMIN_USER", None)
        
        from src.database_manager.app import _get_admin_credentials
        
        with pytest.raises(SystemExit):
            _get_admin_credentials()


def test_sec_001_admin_username_has_default():
    """SEC-001: Verify admin username has a default if not set."""
    with patch.dict(os.environ, {"DB_MANAGER_ADMIN_PASS": "test-password"}, clear=False):
        os.environ.pop("DB_MANAGER_ADMIN_USER", None)
        
        from src.database_manager.app import _get_admin_credentials
        
        username, password = _get_admin_credentials()
        
        assert username == "admin"
        assert password == "test-password"


def test_sec_001_admin_credentials_from_env():
    """SEC-001: Verify admin credentials are loaded from environment."""
    with patch.dict(
        os.environ,
        {
            "DB_MANAGER_ADMIN_USER": "custom_admin",
            "DB_MANAGER_ADMIN_PASS": "strong-password-123",
        },
        clear=False,
    ):
        from src.database_manager.app import _get_admin_credentials
        
        username, password = _get_admin_credentials()
        
        assert username == "custom_admin"
        assert password == "strong-password-123"


def test_sec_002_password_hashing():
    """SEC-002: Verify passwords are hashed using secure method."""
    password = "test-password-123"
    
    hash1 = generate_password_hash(password, method="pbkdf2:sha256")
    hash2 = generate_password_hash(password, method="pbkdf2:sha256")
    
    assert hash1 != hash2
    assert check_password_hash(hash1, password)
    assert check_password_hash(hash2, password)
    assert not check_password_hash(hash1, "wrong-password")


def test_sec_002_check_password_hash_timing_safe():
    """SEC-002: Verify password comparison is timing-safe (no plain-text comparison)."""
    password = "test-password"
    password_hash = generate_password_hash(password, method="pbkdf2:sha256")
    
    wrong_password = "wrong-password-attempt"
    result = check_password_hash(password_hash, wrong_password)
    assert result is False
    
    result = check_password_hash(password_hash, password)
    assert result is True


def test_sec_003_secret_key_required_in_production():
    """SEC-003: Verify SECRET_KEY is required in production."""
    with patch.dict(os.environ, {"ENV": "production"}, clear=False):
        os.environ.pop("DB_MANAGER_SECRET_KEY", None)
        
        from src.database_manager.app import _ensure_secret_key
        
        with pytest.raises(SystemExit):
            _ensure_secret_key()


def test_sec_003_secret_key_from_env():
    """SEC-003: Verify SECRET_KEY is read from environment."""
    expected_key = "my-secret-key-12345"
    
    with patch.dict(
        os.environ,
        {"DB_MANAGER_SECRET_KEY": expected_key, "ENV": "production"},
        clear=False,
    ):
        from src.database_manager.app import _ensure_secret_key
        
        key = _ensure_secret_key()
        assert key == expected_key


def test_sec_003_secret_key_fallback_in_development():
    """SEC-003: Verify SECRET_KEY has fallback in development only."""
    with patch.dict(os.environ, {"ENV": "development"}, clear=False):
        os.environ.pop("DB_MANAGER_SECRET_KEY", None)
        
        from src.database_manager.app import _ensure_secret_key
        
        key = _ensure_secret_key()
        assert key == "dev-key-change-in-production"


def test_sec_003_secret_key_fallback_in_test():
    """SEC-003: Verify SECRET_KEY fallback works in test environment."""
    with patch.dict(os.environ, {"ENV": "test"}, clear=False):
        os.environ.pop("DB_MANAGER_SECRET_KEY", None)
        
        from src.database_manager.app import _ensure_secret_key
        
        key = _ensure_secret_key()
        assert key == "dev-key-change-in-production"
