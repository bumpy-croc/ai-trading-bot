#!/usr/bin/env python3
"""
Test script to validate Railway environment configuration
Used to ensure proper environment variable setup
"""

import os
import sys
from config.config_manager import get_config


def test_environment_config():
    """Test Railway environment configuration"""
    environment = os.getenv('ENVIRONMENT', 'development')
    
    print(f"🔐 Testing environment configuration for: {environment}")
    
    # Get config manager
    config = get_config()
    
    # Test required configuration keys
    required_keys = [
        'DATABASE_URL',
        'BINANCE_API_KEY',
        'BINANCE_API_SECRET'
    ]
    
    missing_keys = []
    for key in required_keys:
        value = config.get(key)
        if value:
            # Mask sensitive values
            if 'SECRET' in key or 'KEY' in key:
                masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:] if len(value) > 8 else '****'
                print(f"✓ {key}: {masked_value}")
            else:
                print(f"✓ {key}: {value}")
        else:
            print(f"❌ {key}: Not found")
            missing_keys.append(key)
    
    if missing_keys:
        print(f"\n⚠️  Missing required configuration keys: {', '.join(missing_keys)}")
        print("   Set these in Railway environment variables")
        return False
    
    print("✅ Environment configuration test passed!")
    return True


def test_database_connection():
    """Test database connection"""
    print("\n🛢️  Testing database connection...")
    
    try:
        from database.manager import DatabaseManager
        
        db_manager = DatabaseManager()
        
        # Test connection
        if db_manager.test_connection():
            print("✓ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            return False
                
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        return False


def test_binance_api():
    """Test Binance API access"""
    print("\n📈 Testing Binance API access...")
    
    try:
        from data_providers.binance_provider import BinanceProvider
        
        provider = BinanceProvider()
        
        # Test getting current price (doesn't require API keys)
        price = provider.get_current_price('BTCUSDT')
        if price and price > 0:
            print(f"✓ Binance API accessible, BTC price: ${price:,.2f}")
            return True
        else:
            print("❌ Could not fetch BTC price")
            return False
            
    except Exception as e:
        print(f"❌ Binance API test failed: {str(e)}")
        return False


def main():
    """Main test function"""
    print("🧪 AI Trading Bot - Railway Environment Test")
    print("=" * 50)
    
    # Test environment configuration
    config_ok = test_environment_config()
    
    # Test database connection
    db_ok = test_database_connection()
    
    # Test Binance API
    api_ok = test_binance_api()
    
    print("\n" + "=" * 50)
    
    if config_ok and db_ok and api_ok:
        print("✅ All tests passed! Railway environment is properly configured.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Check Railway environment variables and configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main() 