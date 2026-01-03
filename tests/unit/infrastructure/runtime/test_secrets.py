"""Tests for infrastructure.runtime.secrets module."""

import os
from unittest.mock import patch

import pytest

from src.infrastructure.runtime.secrets import get_secret_key


class TestGetSecretKey:
    """Tests for get_secret_key function."""

    def test_returns_env_value_when_set(self):
        """Test that environment variable value is returned when set."""
        with patch.dict(os.environ, {"FLASK_SECRET_KEY": "my-secret-key"}):
            result = get_secret_key()
            assert result == "my-secret-key"

    def test_custom_env_var_name(self):
        """Test using custom environment variable name."""
        with patch.dict(os.environ, {"MY_CUSTOM_SECRET": "custom-value"}):
            result = get_secret_key(env_var="MY_CUSTOM_SECRET")
            assert result == "custom-value"

    def test_returns_default_in_development_env(self):
        """Test default value returned in development environment."""
        with patch.dict(os.environ, {"ENV": "development"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"

    def test_returns_default_in_dev_env(self):
        """Test default value returned with ENV=dev."""
        with patch.dict(os.environ, {"ENV": "dev"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"

    def test_returns_default_in_test_env(self):
        """Test default value returned in test environment."""
        with patch.dict(os.environ, {"ENV": "test"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"

    def test_returns_default_in_testing_env(self):
        """Test default value returned with ENV=testing."""
        with patch.dict(os.environ, {"ENV": "testing"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"

    def test_flask_env_fallback(self):
        """Test FLASK_ENV is used as fallback when ENV not set."""
        with patch.dict(os.environ, {"FLASK_ENV": "development"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            os.environ.pop("ENV", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"

    def test_raises_in_production_without_secret(self):
        """Test RuntimeError raised in production without secret."""
        with patch.dict(os.environ, {"ENV": "production"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            with pytest.raises(RuntimeError) as exc_info:
                get_secret_key()
            assert "Missing required secret" in str(exc_info.value)
            assert "FLASK_SECRET_KEY" in str(exc_info.value)

    def test_raises_in_staging_without_secret(self):
        """Test RuntimeError raised in staging without secret."""
        with patch.dict(os.environ, {"ENV": "staging"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            with pytest.raises(RuntimeError):
                get_secret_key()

    def test_allow_default_in_dev_false(self):
        """Test that allow_default_in_dev=False forces requirement."""
        with patch.dict(os.environ, {"ENV": "development"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            with pytest.raises(RuntimeError):
                get_secret_key(allow_default_in_dev=False)

    def test_env_value_takes_precedence_in_production(self):
        """Test that env value is used even in production."""
        with patch.dict(
            os.environ,
            {"ENV": "production", "FLASK_SECRET_KEY": "prod-secret"},
        ):
            result = get_secret_key()
            assert result == "prod-secret"

    def test_case_insensitive_env_matching(self):
        """Test that environment matching is case-insensitive."""
        with patch.dict(os.environ, {"ENV": "DEVELOPMENT"}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"

    def test_empty_env_treated_as_development(self):
        """Test that empty/missing ENV defaults to development."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("FLASK_SECRET_KEY", None)
            os.environ.pop("ENV", None)
            os.environ.pop("FLASK_ENV", None)
            result = get_secret_key()
            assert result == "dev-key-change-in-production"


@pytest.mark.fast
class TestSecretKeyIntegration:
    """Integration tests for secret key functionality."""

    def test_multiple_env_vars(self):
        """Test with multiple environment variables set."""
        with patch.dict(
            os.environ,
            {
                "ENV": "production",
                "FLASK_SECRET_KEY": "flask-key",
                "OTHER_SECRET": "other-key",
            },
        ):
            # Should use the specified env var
            assert get_secret_key() == "flask-key"
            assert get_secret_key(env_var="OTHER_SECRET") == "other-key"

    def test_whitespace_handling(self):
        """Test that whitespace in secret is preserved."""
        with patch.dict(os.environ, {"FLASK_SECRET_KEY": "  secret with spaces  "}):
            result = get_secret_key()
            assert result == "  secret with spaces  "

    def test_special_characters_in_secret(self):
        """Test secrets with special characters."""
        special_secret = "secret!@#$%^&*()_+-=[]{}|;':\",./<>?"
        with patch.dict(os.environ, {"FLASK_SECRET_KEY": special_secret}):
            result = get_secret_key()
            assert result == special_secret
