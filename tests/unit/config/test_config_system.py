import os
import tempfile
from unittest.mock import patch
import pytest

from config.config_manager import ConfigManager
from config.providers.env_provider import EnvVarProvider
from config.providers.dotenv_provider import DotEnvProvider

pytestmark = pytest.mark.unit


class TestEnvVarProvider:
    def test_get_existing_env_var(self):
        with patch.dict(os.environ, {'TEST_KEY': 'test_value'}):
            provider = EnvVarProvider()
            assert provider.get('TEST_KEY') == 'test_value'

    def test_get_nonexistent_env_var(self):
        provider = EnvVarProvider()
        os.environ.pop('NONEXISTENT_KEY_12345', None)
        assert provider.get('NONEXISTENT_KEY_12345') is None

    def test_is_available(self):
        provider = EnvVarProvider()
        assert provider.is_available() is True

    def test_get_all(self):
        with patch.dict(os.environ, {'TEST1': 'value1', 'TEST2': 'value2'}):
            provider = EnvVarProvider()
            all_vars = provider.get_all()
            assert all_vars['TEST1'] == 'value1'
            assert all_vars['TEST2'] == 'value2'


class TestDotEnvProvider:
    def test_load_env_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('TEST_KEY=test_value\nANOTHER_KEY=another_value\n# Comment\nEMPTY_KEY=\n')
            env_file = f.name
        try:
            provider = DotEnvProvider(env_file)
            assert provider.get('TEST_KEY') == 'test_value'
            assert provider.get('ANOTHER_KEY') == 'another_value'
            assert provider.get('EMPTY_KEY') == ''
            assert provider.get('NONEXISTENT') is None
        finally:
            os.unlink(env_file)

    def test_nonexistent_env_file(self):
        provider = DotEnvProvider('/no/such/.env')
        assert provider.get('ANY_KEY') is None

    def test_malformed_env_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('VALID_KEY=valid_value\nMALFORMED\n=VALUE_WITHOUT_KEY\nANOTHER_VALID=value\n')
            env_file = f.name
        try:
            provider = DotEnvProvider(env_file)
            assert provider.get('VALID_KEY') == 'valid_value'
            assert provider.get('ANOTHER_VALID') == 'value'
        finally:
            os.unlink(env_file)


class TestConfigManager:
    def test_provider_fallback_order(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('FALLBACK_TEST=dotenv_value\n')
            env_file = f.name
        try:
            with patch.dict(os.environ, {'FALLBACK_TEST': 'env_value'}):
                providers = [EnvVarProvider(), DotEnvProvider(env_file)]
                config = ConfigManager(providers=providers)
                assert config.get('FALLBACK_TEST') == 'env_value'
        finally:
            os.unlink(env_file)

    def test_type_conversion_methods(self):
        with patch.dict(os.environ, {
            'INT_VALUE': '42', 'FLOAT_VALUE': '3.14', 'BOOL_TRUE': 'true', 'BOOL_FALSE': 'false',
            'BOOL_YES': 'yes', 'BOOL_NO': 'no', 'BOOL_1': '1', 'BOOL_0': '0', 'LIST_VALUE': 'a,b,c',
            'EMPTY_LIST': '', 'INVALID_INT': 'x', 'INVALID_FLOAT': 'y'
        }):
            config = ConfigManager()
            assert config.get_int('INT_VALUE') == 42
            assert config.get_int('INVALID_INT', 99) == 99
            assert config.get_float('FLOAT_VALUE') == 3.14
            assert config.get_float('INVALID_FLOAT', 2.5) == 2.5
            assert config.get_bool('BOOL_TRUE') is True
            assert config.get_bool('BOOL_FALSE') is False
            assert config.get_bool('BOOL_YES') is True
            assert config.get_bool('BOOL_NO') is False
            assert config.get_bool('BOOL_1') is True
            assert config.get_bool('BOOL_0') is False
            assert config.get_list('LIST_VALUE') == ['a', 'b', 'c']
            assert config.get_list('EMPTY_LIST') == []

    def test_get_required_and_all(self):
        with patch.dict(os.environ, {'REQUIRED_KEY': 'required_value', 'TEST1': 'value1', 'TEST2': 'value2'}):
            config = ConfigManager()
            assert config.get_required('REQUIRED_KEY') == 'required_value'
            all_config = config.get_all()
            assert all_config['TEST1'] == 'value1'
            assert all_config['TEST2'] == 'value2'

    def test_add_remove_providers_and_refresh(self):
        config = ConfigManager()
        initial_count = len(config.providers)
        new_provider = EnvVarProvider()
        config.add_provider(new_provider, priority=0)
        assert len(config.providers) == initial_count + 1
        assert config.providers[0] == new_provider
        removed = config.remove_provider('Environment Variables')
        assert removed is True
        config.refresh()