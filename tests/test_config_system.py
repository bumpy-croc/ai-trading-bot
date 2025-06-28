import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from core.config.config_manager import ConfigManager
from core.config.providers.env_provider import EnvVarProvider
from core.config.providers.dotenv_provider import DotEnvProvider

# Skip AWS tests if boto3 is not available
boto3_available = True
try:
    import boto3
    from core.config.providers.aws_secrets_provider import AWSSecretsProvider
except ImportError:
    boto3_available = False


class TestEnvVarProvider:
    """Test the environment variable provider."""
    
    def test_get_existing_env_var(self):
        """Test getting an existing environment variable."""
        with patch.dict(os.environ, {'TEST_KEY': 'test_value'}):
            provider = EnvVarProvider()
            assert provider.get('TEST_KEY') == 'test_value'
    
    def test_get_nonexistent_env_var(self):
        """Test getting a non-existent environment variable returns None."""
        provider = EnvVarProvider()
        # Make sure this key doesn't exist
        if 'NONEXISTENT_KEY_12345' in os.environ:
            del os.environ['NONEXISTENT_KEY_12345']
        assert provider.get('NONEXISTENT_KEY_12345') is None
    
    def test_is_available(self):
        """Test that environment provider is always available."""
        provider = EnvVarProvider()
        assert provider.is_available() is True
    
    def test_get_all(self):
        """Test getting all environment variables."""
        with patch.dict(os.environ, {'TEST1': 'value1', 'TEST2': 'value2'}):
            provider = EnvVarProvider()
            all_vars = provider.get_all()
            assert 'TEST1' in all_vars
            assert 'TEST2' in all_vars
            assert all_vars['TEST1'] == 'value1'
            assert all_vars['TEST2'] == 'value2'


class TestDotEnvProvider:
    """Test the .env file provider."""
    
    def test_load_env_file(self):
        """Test loading from a .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('TEST_KEY=test_value\n')
            f.write('ANOTHER_KEY=another_value\n')
            f.write('# This is a comment\n')
            f.write('EMPTY_KEY=\n')
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
        """Test handling of non-existent .env file."""
        provider = DotEnvProvider('/path/that/does/not/exist/.env')
        assert provider.get('ANY_KEY') is None
    
    def test_malformed_env_file(self):
        """Test handling of malformed .env file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('VALID_KEY=valid_value\n')
            f.write('MALFORMED_LINE_WITHOUT_EQUALS\n')
            f.write('=VALUE_WITHOUT_KEY\n')
            f.write('ANOTHER_VALID=value\n')
            env_file = f.name
        
        try:
            provider = DotEnvProvider(env_file)
            # Should still load valid entries
            assert provider.get('VALID_KEY') == 'valid_value'
            assert provider.get('ANOTHER_VALID') == 'value'
        finally:
            os.unlink(env_file)


@pytest.mark.skipif(not boto3_available, reason="boto3 not available")
class TestAWSSecretsProvider:
    """Test the AWS Secrets Manager provider."""
    
    @patch('boto3.client')
    def test_successful_secret_retrieval(self, mock_boto_client):
        """Test successful secret retrieval from AWS."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock successful describe_secret call
        mock_client.describe_secret.return_value = {'Name': 'test-secret'}
        
        # Mock successful response
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps({
                'BINANCE_API_KEY': 'test_api_key',
                'BINANCE_API_SECRET': 'test_api_secret',
                'DATABASE_URL': 'sqlite:///test.db'
            })
        }
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'test'}):
            provider = AWSSecretsProvider()
            assert provider.get('BINANCE_API_KEY') == 'test_api_key'
            assert provider.get('BINANCE_API_SECRET') == 'test_api_secret'
            assert provider.get('DATABASE_URL') == 'sqlite:///test.db'
            assert provider.get('NONEXISTENT') is None
    
    @patch('boto3.client')
    def test_secret_not_found(self, mock_boto_client):
        """Test handling when secret is not found in AWS."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock describe_secret to succeed (provider initialization)
        mock_client.describe_secret.return_value = {'Name': 'test-secret'}
        
        from botocore.exceptions import ClientError
        mock_client.get_secret_value.side_effect = ClientError(
            {'Error': {'Code': 'ResourceNotFoundException'}}, 'GetSecretValue'
        )
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'test'}):
            provider = AWSSecretsProvider()
            assert provider.get('ANY_KEY') is None
    
    @patch('boto3.client')
    def test_aws_credentials_error(self, mock_boto_client):
        """Test handling when AWS credentials are not available."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock describe_secret to fail (provider initialization)
        from botocore.exceptions import NoCredentialsError
        mock_client.describe_secret.side_effect = NoCredentialsError()
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'test'}):
            provider = AWSSecretsProvider()
            assert provider.is_available() is False
            assert provider.get('ANY_KEY') is None
    
    @patch('boto3.client')
    def test_caching_behavior(self, mock_boto_client):
        """Test that secrets are cached and not fetched repeatedly."""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock describe_secret to succeed
        mock_client.describe_secret.return_value = {'Name': 'test-secret'}
        
        mock_client.get_secret_value.return_value = {
            'SecretString': json.dumps({'TEST_KEY': 'test_value'})
        }
        
        with patch.dict(os.environ, {'ENVIRONMENT': 'test'}):
            provider = AWSSecretsProvider()
            
            # First call should fetch from AWS
            result1 = provider.get('TEST_KEY')
            assert result1 == 'test_value'
            assert mock_client.get_secret_value.call_count == 1
            
            # Second call should use cache
            result2 = provider.get('TEST_KEY')
            assert result2 == 'test_value'
            assert mock_client.get_secret_value.call_count == 1  # No additional calls
    
    def test_environment_detection(self):
        """Test environment detection for secret naming."""
        # Test with explicit environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            provider = AWSSecretsProvider()
            assert provider.secret_name == 'ai-trading-bot/production'
        
        # Test with staging environment
        with patch.dict(os.environ, {'ENVIRONMENT': 'staging'}):
            provider = AWSSecretsProvider()
            assert provider.secret_name == 'ai-trading-bot/staging'
        
        # Test default environment (development)
        env_backup = os.environ.get('ENVIRONMENT')
        if 'ENVIRONMENT' in os.environ:
            del os.environ['ENVIRONMENT']
        try:
            provider = AWSSecretsProvider()
            assert provider.secret_name == 'ai-trading-bot/development'
        finally:
            if env_backup:
                os.environ['ENVIRONMENT'] = env_backup


class TestConfigManager:
    """Test the main configuration manager."""
    
    def test_provider_fallback_order(self):
        """Test that providers are tried in the correct fallback order."""
        # Create a temporary .env file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write('FALLBACK_TEST=dotenv_value\n')
            env_file = f.name
        
        try:
            # Set up environment where AWS fails, env var exists, and .env exists
            with patch.dict(os.environ, {'FALLBACK_TEST': 'env_value'}):
                # Create config manager with only env and dotenv providers (skip AWS)
                from core.config.providers.env_provider import EnvVarProvider
                from core.config.providers.dotenv_provider import DotEnvProvider
                
                providers = [
                    EnvVarProvider(),
                    DotEnvProvider(env_file)
                ]
                config = ConfigManager(providers=providers)
                
                # Should get value from environment variable (first in priority)
                assert config.get('FALLBACK_TEST') == 'env_value'
        finally:
            os.unlink(env_file)
    
    def test_type_conversion_methods(self):
        """Test type conversion methods."""
        with patch.dict(os.environ, {
            'INT_VALUE': '42',
            'FLOAT_VALUE': '3.14',
            'BOOL_TRUE': 'true',
            'BOOL_FALSE': 'false',
            'BOOL_YES': 'yes',
            'BOOL_NO': 'no',
            'BOOL_1': '1',
            'BOOL_0': '0',
            'LIST_VALUE': 'item1,item2,item3',
            'EMPTY_LIST': '',
            'INVALID_INT': 'not_a_number',
            'INVALID_FLOAT': 'not_a_float'
        }):
            config = ConfigManager()
            
            # Test integer conversion
            assert config.get_int('INT_VALUE') == 42
            assert config.get_int('INVALID_INT', 99) == 99
            assert config.get_int('NONEXISTENT', 123) == 123
            
            # Test float conversion
            assert config.get_float('FLOAT_VALUE') == 3.14
            assert config.get_float('INVALID_FLOAT', 2.5) == 2.5
            assert config.get_float('NONEXISTENT', 1.5) == 1.5
            
            # Test boolean conversion
            assert config.get_bool('BOOL_TRUE') is True
            assert config.get_bool('BOOL_FALSE') is False
            assert config.get_bool('BOOL_YES') is True
            assert config.get_bool('BOOL_NO') is False
            assert config.get_bool('BOOL_1') is True
            assert config.get_bool('BOOL_0') is False
            assert config.get_bool('NONEXISTENT', True) is True
            
            # Test list conversion
            assert config.get_list('LIST_VALUE') == ['item1', 'item2', 'item3']
            assert config.get_list('EMPTY_LIST') == []
            # Test with explicit default parameter
            assert config.get_list('NONEXISTENT', default=['default']) == ['default']
    
    def test_get_required_success(self):
        """Test getting required configuration that exists."""
        with patch.dict(os.environ, {'REQUIRED_KEY': 'required_value'}):
            config = ConfigManager()
            assert config.get_required('REQUIRED_KEY') == 'required_value'
    
    def test_get_required_missing(self):
        """Test getting required configuration that doesn't exist."""
        # Clear environment of the key
        if 'MISSING_REQUIRED_KEY' in os.environ:
            del os.environ['MISSING_REQUIRED_KEY']
        
        config = ConfigManager()
        with pytest.raises(ValueError) as exc_info:
            config.get_required('MISSING_REQUIRED_KEY')
        
        assert 'MISSING_REQUIRED_KEY' in str(exc_info.value)
        assert 'not found' in str(exc_info.value)
    
    def test_get_all(self):
        """Test getting all configuration values."""
        with patch.dict(os.environ, {'TEST1': 'value1', 'TEST2': 'value2'}):
            config = ConfigManager()
            all_config = config.get_all()
            
            # Should contain our test values
            assert 'TEST1' in all_config
            assert 'TEST2' in all_config
            assert all_config['TEST1'] == 'value1'
            assert all_config['TEST2'] == 'value2'
    
    def test_add_remove_providers(self):
        """Test adding and removing providers."""
        config = ConfigManager()
        initial_count = len(config.providers)
        
        # Add a new provider
        new_provider = EnvVarProvider()
        config.add_provider(new_provider, priority=0)
        assert len(config.providers) == initial_count + 1
        assert config.providers[0] == new_provider
        
        # Remove the provider
        removed = config.remove_provider('Environment Variables')
        assert removed is True
        assert len(config.providers) == initial_count
    
    def test_refresh(self):
        """Test refreshing all providers."""
        config = ConfigManager()
        # Should not raise exception
        config.refresh()


class TestConfigSystemIntegration:
    """Integration tests for the configuration system."""
    
    def test_binance_data_provider_integration(self):
        """Test that BinanceDataProvider can use the new config system."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_key',
            'BINANCE_API_SECRET': 'test_secret',
            'BINANCE_TESTNET': 'false'
        }):
            # Import here to ensure clean environment
            from core.data_providers.binance_data_provider import BinanceDataProvider
            
            # Should not raise exception during initialization
            try:
                provider = BinanceDataProvider()
                # Check that it was initialized without error
                assert provider.client is not None
                # Verify it has the data attribute from parent class
                assert hasattr(provider, 'data')
            except Exception as e:
                pytest.fail(f"BinanceDataProvider initialization failed: {e}")
    
    def test_live_trading_config_integration(self):
        """Test configuration integration with live trading components."""
        with patch.dict(os.environ, {
            'BINANCE_API_KEY': 'test_key',
            'BINANCE_API_SECRET': 'test_secret',
            'DATABASE_URL': 'sqlite:///test.db',
            'TRADING_MODE': 'paper',
            'INITIAL_BALANCE': '1000',
            'LOG_LEVEL': 'INFO'
        }):
            config = ConfigManager()
            
            # Test that all required configs are available
            assert config.get('BINANCE_API_KEY') is not None
            assert config.get('DATABASE_URL') is not None
            assert config.get('TRADING_MODE') == 'paper'
            assert config.get_float('INITIAL_BALANCE') == 1000.0
            
            # Test required key functionality
            assert config.get_required('BINANCE_API_KEY') == 'test_key'
    
    def test_global_config_functions(self):
        """Test the global configuration functions."""
        from core.config.config_manager import get_config, set_config
        
        # Test getting global config
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2  # Should be the same instance
        
        # Test setting global config
        new_config = ConfigManager()
        set_config(new_config)
        config3 = get_config()
        assert config3 is new_config


if __name__ == '__main__':
    pytest.main([__file__]) 