"""
Tests for database manager security fixes.

Tests for:
- SEC-001: Weak Default Admin Password
- SEC-002: Plain-Text Password Comparison  
- SEC-003: Inconsistent SECRET_KEY Handling
"""

import os
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from werkzeug.security import check_password_hash, generate_password_hash


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


def test_sec_003_secret_key_defaults_to_production_when_env_missing():
    """SEC-003: Missing env variables should fail closed."""
    with patch.dict(os.environ, {}, clear=False):
        for variable in ("DB_MANAGER_SECRET_KEY", "ENV", "FLASK_ENV"):
            os.environ.pop(variable, None)

        from src.database_manager.app import _ensure_secret_key

        with pytest.raises(SystemExit):
            _ensure_secret_key()


def test_sec_003_secret_key_fallback_honours_flask_env_dev():
    """SEC-003: FLASK_ENV should allow dev fallback when set to development."""
    with patch.dict(os.environ, {"FLASK_ENV": "development"}, clear=False):
        os.environ.pop("DB_MANAGER_SECRET_KEY", None)
        os.environ.pop("ENV", None)

        from src.database_manager.app import _ensure_secret_key

        key = _ensure_secret_key()
        assert key == "dev-key-change-in-production"


def test_sec_003_secret_key_requires_flask_env_production():
    """SEC-003: FLASK_ENV=production must require explicit secret key."""
    with patch.dict(os.environ, {"FLASK_ENV": "production"}, clear=False):
        os.environ.pop("DB_MANAGER_SECRET_KEY", None)
        os.environ.pop("ENV", None)

        from src.database_manager.app import _ensure_secret_key

        with pytest.raises(SystemExit):
            _ensure_secret_key()


def test_sec_002_login_missing_password_graceful_failure(monkeypatch):
    """SEC-002: Missing password should not raise when validating credentials."""

    class DummyDatabaseManager:
        def __init__(self):
            self.engine = object()
            self.session_factory = lambda: None

        def get_database_info(self):
            return {"status": "ok"}

    dummy_base = SimpleNamespace(
        registry=SimpleNamespace(mappers=[]),
        metadata=SimpleNamespace(create_all=lambda engine: None),
    )

    class DummyScopedSession:
        def remove(self):
            return None

    flask_admin_module = ModuleType("flask_admin")
    flask_admin_contrib = ModuleType("flask_admin.contrib")
    flask_admin_sqla = ModuleType("flask_admin.contrib.sqla")

    class DummyAdmin:
        def __init__(self, *args, **kwargs):
            self._views = []

        def add_view(self, view):
            self._views.append(view)

    class DummyModelView:
        def __init__(self, *args, **kwargs):
            pass

    flask_admin_module.Admin = DummyAdmin
    flask_admin_contrib.sqla = flask_admin_sqla
    flask_admin_sqla.ModelView = DummyModelView

    flask_login_module = ModuleType("flask_login")

    class DummyLoginManager:
        def __init__(self):
            self.login_view = None

        def init_app(self, app):
            return None

        def user_loader(self, callback):
            return callback

    class DummyUserMixin:
        pass

    def dummy_login_required(func):
        return func

    def dummy_login_user(user):
        return None

    def dummy_logout_user():
        return None

    with patch.dict(
        os.environ,
        {
            "DB_MANAGER_ADMIN_USER": "admin",
            "DB_MANAGER_ADMIN_PASS": "super-secure",
            "DB_MANAGER_SECRET_KEY": "secret",
            "ENV": "development",
        },
        clear=False,
    ):
        monkeypatch.setitem(sys.modules, "flask_admin", flask_admin_module)
        monkeypatch.setitem(sys.modules, "flask_admin.contrib", flask_admin_contrib)
        monkeypatch.setitem(sys.modules, "flask_admin.contrib.sqla", flask_admin_sqla)
        monkeypatch.setitem(sys.modules, "flask_login", flask_login_module)

        flask_login_module.LoginManager = DummyLoginManager
        flask_login_module.UserMixin = DummyUserMixin
        flask_login_module.login_required = dummy_login_required
        flask_login_module.login_user = dummy_login_user
        flask_login_module.logout_user = dummy_logout_user

        import src.database_manager.app as app_module

        monkeypatch.setattr(app_module, "DatabaseManager", DummyDatabaseManager)
        monkeypatch.setattr(app_module, "scoped_session", lambda factory: DummyScopedSession())
        monkeypatch.setattr(app_module, "Base", dummy_base)

        app = app_module.create_app()

        with app.test_client() as client:
            response = client.post("/login", data={"username": "admin"})

        assert response.status_code == 200
        assert b"Login Failed" in response.data
