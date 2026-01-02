"""Tests for config.config_manager module."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from src.config.config_manager import (
    ConfigManager,
    get_config,
    set_config,
)


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_init_with_default_providers(self):
        """Test initialization with default providers."""
        with patch("src.config.config_manager.RailwayProvider") as mock_railway:
            with patch("src.config.config_manager.EnvVarProvider") as mock_env:
                with patch("src.config.config_manager.DotEnvProvider") as mock_dotenv:
                    mock_railway.return_value.is_available.return_value = False
                    mock_env.return_value.is_available.return_value = True
                    mock_dotenv.return_value.is_available.return_value = True
                    mock_railway.return_value.provider_name = "railway"
                    mock_env.return_value.provider_name = "env"
                    mock_dotenv.return_value.provider_name = "dotenv"

                    manager = ConfigManager()

                    assert len(manager.providers) == 3

    def test_init_with_custom_providers(self):
        """Test initialization with custom providers."""
        mock_provider = MagicMock()
        mock_provider.is_available.return_value = True
        mock_provider.provider_name = "custom"

        manager = ConfigManager(providers=[mock_provider])

        assert len(manager.providers) == 1
        assert manager.providers[0] == mock_provider

    def test_get_returns_value_from_first_available(self):
        """Test that get returns value from first available provider."""
        provider1 = MagicMock()
        provider1.is_available.return_value = True
        provider1.get.return_value = "value1"

        provider2 = MagicMock()
        provider2.is_available.return_value = True
        provider2.get.return_value = "value2"

        manager = ConfigManager(providers=[provider1, provider2])
        result = manager.get("KEY")

        assert result == "value1"
        provider1.get.assert_called_once_with("KEY")
        provider2.get.assert_not_called()

    def test_get_falls_through_on_none(self):
        """Test that get falls through to next provider when value is None."""
        provider1 = MagicMock()
        provider1.is_available.return_value = True
        provider1.get.return_value = None

        provider2 = MagicMock()
        provider2.is_available.return_value = True
        provider2.get.return_value = "value2"

        manager = ConfigManager(providers=[provider1, provider2])
        result = manager.get("KEY")

        assert result == "value2"

    def test_get_returns_default_when_not_found(self):
        """Test that get returns default when key not found."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = None

        manager = ConfigManager(providers=[provider])
        result = manager.get("MISSING_KEY", default="default_value")

        assert result == "default_value"

    def test_get_skips_unavailable_providers(self):
        """Test that unavailable providers are skipped."""
        provider1 = MagicMock()
        provider1.is_available.return_value = False

        provider2 = MagicMock()
        provider2.is_available.return_value = True
        provider2.get.return_value = "value2"

        manager = ConfigManager(providers=[provider1, provider2])
        result = manager.get("KEY")

        assert result == "value2"
        provider1.get.assert_not_called()

    def test_get_required_returns_value(self):
        """Test that get_required returns value when found."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "required_value"
        provider.provider_name = "test"

        manager = ConfigManager(providers=[provider])
        result = manager.get_required("KEY")

        assert result == "required_value"

    def test_get_required_raises_when_not_found(self):
        """Test that get_required raises ValueError when not found."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = None
        provider.provider_name = "test"

        manager = ConfigManager(providers=[provider])

        with pytest.raises(ValueError) as exc_info:
            manager.get_required("MISSING_KEY")

        assert "Required configuration 'MISSING_KEY' not found" in str(exc_info.value)

    def test_get_int_returns_integer(self):
        """Test that get_int returns integer value."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "42"

        manager = ConfigManager(providers=[provider])
        result = manager.get_int("KEY")

        assert result == 42
        assert isinstance(result, int)

    def test_get_int_returns_default_on_invalid(self):
        """Test that get_int returns default on invalid value."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "not_a_number"

        manager = ConfigManager(providers=[provider])
        result = manager.get_int("KEY", default=10)

        assert result == 10

    def test_get_float_returns_float(self):
        """Test that get_float returns float value."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "3.14"

        manager = ConfigManager(providers=[provider])
        result = manager.get_float("KEY")

        assert result == 3.14
        assert isinstance(result, float)

    def test_get_float_returns_default_on_invalid(self):
        """Test that get_float returns default on invalid value."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "not_a_float"

        manager = ConfigManager(providers=[provider])
        result = manager.get_float("KEY", default=1.5)

        assert result == 1.5

    def test_get_bool_true_values(self):
        """Test that get_bool recognizes true values."""
        provider = MagicMock()
        provider.is_available.return_value = True

        manager = ConfigManager(providers=[provider])

        for true_val in ["true", "True", "TRUE", "1", "yes", "on", "enabled"]:
            provider.get.return_value = true_val
            assert manager.get_bool("KEY") is True

    def test_get_bool_false_values(self):
        """Test that get_bool recognizes false values."""
        provider = MagicMock()
        provider.is_available.return_value = True

        manager = ConfigManager(providers=[provider])

        for false_val in ["false", "False", "0", "no", "off", "disabled", ""]:
            provider.get.return_value = false_val
            assert manager.get_bool("KEY") is False

    def test_get_bool_returns_default_when_none(self):
        """Test that get_bool returns default when value is None."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = None

        manager = ConfigManager(providers=[provider])
        result = manager.get_bool("KEY", default=True)

        assert result is True

    def test_get_list_splits_by_comma(self):
        """Test that get_list splits values by comma."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "a,b,c"

        manager = ConfigManager(providers=[provider])
        result = manager.get_list("KEY")

        assert result == ["a", "b", "c"]

    def test_get_list_custom_delimiter(self):
        """Test that get_list respects custom delimiter."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = "a;b;c"

        manager = ConfigManager(providers=[provider])
        result = manager.get_list("KEY", delimiter=";")

        assert result == ["a", "b", "c"]

    def test_get_list_strips_whitespace(self):
        """Test that get_list strips whitespace from items."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = " a , b , c "

        manager = ConfigManager(providers=[provider])
        result = manager.get_list("KEY")

        assert result == ["a", "b", "c"]

    def test_get_list_returns_default_when_none(self):
        """Test that get_list returns default when value is None."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.get.return_value = None

        manager = ConfigManager(providers=[provider])
        result = manager.get_list("KEY", default=["default"])

        assert result == ["default"]

    def test_get_all_merges_providers(self):
        """Test that get_all merges values from all providers."""
        provider1 = MagicMock()
        provider1.is_available.return_value = True
        provider1.get_all.return_value = {"KEY1": "val1", "KEY2": "val2"}

        provider2 = MagicMock()
        provider2.is_available.return_value = True
        provider2.get_all.return_value = {"KEY2": "override", "KEY3": "val3"}

        manager = ConfigManager(providers=[provider1, provider2])
        result = manager.get_all()

        # Later providers in reverse order should override
        assert "KEY1" in result
        assert "KEY3" in result

    def test_refresh_calls_all_providers(self):
        """Test that refresh calls refresh on all available providers."""
        provider1 = MagicMock()
        provider1.is_available.return_value = True

        provider2 = MagicMock()
        provider2.is_available.return_value = True

        manager = ConfigManager(providers=[provider1, provider2])
        manager.refresh()

        provider1.refresh.assert_called_once()
        provider2.refresh.assert_called_once()

    def test_add_provider(self):
        """Test adding a provider at specific priority."""
        provider1 = MagicMock()
        provider1.is_available.return_value = True
        provider1.provider_name = "original"

        manager = ConfigManager(providers=[provider1])

        new_provider = MagicMock()
        new_provider.is_available.return_value = True
        new_provider.provider_name = "new"

        manager.add_provider(new_provider, priority=0)

        assert manager.providers[0] == new_provider
        assert len(manager.providers) == 2

    def test_remove_provider(self):
        """Test removing a provider by name."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.provider_name = "to_remove"

        manager = ConfigManager(providers=[provider])

        result = manager.remove_provider("to_remove")

        assert result is True
        assert len(manager.providers) == 0

    def test_remove_provider_returns_false_when_not_found(self):
        """Test that remove_provider returns False when provider not found."""
        provider = MagicMock()
        provider.is_available.return_value = True
        provider.provider_name = "exists"

        manager = ConfigManager(providers=[provider])

        result = manager.remove_provider("does_not_exist")

        assert result is False
        assert len(manager.providers) == 1


class TestGetConfig:
    """Tests for get_config singleton function."""

    def test_returns_config_manager(self):
        """Test that get_config returns a ConfigManager instance."""
        import src.config.config_manager as module

        # Reset singleton - use a valid lock, not None
        module._config_instance = None
        module._config_lock = threading.Lock()

        with patch.object(module, "ConfigManager") as mock_class:
            mock_class.return_value = MagicMock()
            result = get_config()
            assert result is not None

        # Reset for other tests
        module._config_instance = None

    def test_returns_same_instance(self):
        """Test that get_config returns the same instance (singleton)."""
        import src.config.config_manager as module

        # Reset singleton
        module._config_instance = MagicMock()

        result1 = get_config()
        result2 = get_config()

        assert result1 is result2

        # Reset for other tests
        module._config_instance = None


class TestSetConfig:
    """Tests for set_config function."""

    def test_sets_global_config(self):
        """Test that set_config sets the global config instance."""
        import src.config.config_manager as module

        mock_config = MagicMock()
        set_config(mock_config)

        assert module._config_instance is mock_config

        # Reset for other tests
        module._config_instance = None


@pytest.mark.fast
class TestConfigManagerIntegration:
    """Integration tests for ConfigManager."""

    def test_full_config_workflow(self):
        """Test complete configuration workflow."""
        provider1 = MagicMock()
        provider1.is_available.return_value = True
        provider1.provider_name = "primary"
        provider1.get.side_effect = lambda k: {"API_KEY": "key123", "DEBUG": "true"}.get(k)
        provider1.get_all.return_value = {"API_KEY": "key123", "DEBUG": "true"}

        provider2 = MagicMock()
        provider2.is_available.return_value = True
        provider2.provider_name = "fallback"
        provider2.get.side_effect = lambda k: {"PORT": "8080", "HOST": "localhost"}.get(k)
        provider2.get_all.return_value = {"PORT": "8080", "HOST": "localhost"}

        manager = ConfigManager(providers=[provider1, provider2])

        # Test various get methods
        assert manager.get("API_KEY") == "key123"
        assert manager.get("PORT") == "8080"
        assert manager.get_int("PORT") == 8080
        assert manager.get_bool("DEBUG") is True
        assert manager.get("MISSING", "default") == "default"

    def test_provider_priority_chain(self):
        """Test that provider priority chain works correctly."""
        # Railway (highest) -> Env -> DotEnv (lowest)
        railway = MagicMock()
        railway.is_available.return_value = True
        railway.provider_name = "railway"
        railway.get.return_value = "railway_value"

        env = MagicMock()
        env.is_available.return_value = True
        env.provider_name = "env"
        env.get.return_value = "env_value"

        manager = ConfigManager(providers=[railway, env])

        # Should use railway value since it's first
        assert manager.get("KEY") == "railway_value"

        # When railway returns None, should fall through to env
        railway.get.return_value = None
        assert manager.get("KEY") == "env_value"
