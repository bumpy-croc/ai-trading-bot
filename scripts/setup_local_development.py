#!/usr/bin/env python3
"""
Local Development Setup Script
Helps developers set up their local development environment with either PostgreSQL or SQLite
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path


def print_header():
    """Print setup header"""
    print("🚀 AI Trading Bot - Local Development Setup")
    print("=" * 60)
    print()


def check_requirements():
    """Check if required tools are available"""
    print("🔍 Checking System Requirements...")
    
    requirements = {
        'python3': 'Python 3.9+ required',
        'docker': 'Docker required for PostgreSQL option',
        'docker-compose': 'Docker Compose required for PostgreSQL option'
    }
    
    missing = []
    
    for tool, description in requirements.items():
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"✅ {tool}: {version}")
            else:
                missing.append((tool, description))
        except FileNotFoundError:
            missing.append((tool, description))
    
    if missing:
        print("\n⚠️  Missing Requirements:")
        for tool, description in missing:
            print(f"   - {tool}: {description}")
        print()
    
    return len(missing) == 0


def choose_database_option():
    """Let user choose between PostgreSQL and SQLite"""
    print("🗄️  Database Configuration")
    print("Choose your local development database:")
    print()
    print("1. 🐘 PostgreSQL (Recommended)")
    print("   ✅ Environment parity with production")
    print("   ✅ Test all PostgreSQL features locally")
    print("   ✅ Realistic performance testing")
    print("   ❌ Requires Docker and more setup")
    print()
    print("2. 🗃️  SQLite (Simple)")
    print("   ✅ Zero setup, instant start")
    print("   ✅ Lightweight and fast")
    print("   ✅ Perfect for quick development")
    print("   ❌ Different from production environment")
    print()
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        print("Please enter 1 or 2")


def setup_environment_file(database_choice):
    """Set up the .env file based on user choices"""
    print("\n📝 Setting up Environment Configuration...")
    
    # Copy example file if .env doesn't exist
    env_file = Path('.env')
    example_file = Path('.env.example')
    
    if not env_file.exists():
        if example_file.exists():
            shutil.copy(example_file, env_file)
            print(f"✅ Created .env file from .env.example")
        else:
            # Create minimal .env file
            with open(env_file, 'w') as f:
                f.write("# AI Trading Bot Environment Configuration\n")
                f.write("TRADING_MODE=paper\n")
                f.write("INITIAL_BALANCE=10000\n")
                f.write("LOG_LEVEL=INFO\n")
            print(f"✅ Created basic .env file")
    
    # Update .env file based on database choice
    with open(env_file, 'r') as f:
        content = f.read()
    
    if database_choice == '1':  # PostgreSQL
        # Uncomment PostgreSQL DATABASE_URL
        content = content.replace(
            '# DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot',
            'DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot'
        )
        print("✅ Configured for PostgreSQL")
    else:  # SQLite
        # Ensure PostgreSQL DATABASE_URL is commented out
        content = content.replace(
            'DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot',
            '# DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot'
        )
        print("✅ Configured for SQLite")
    
    with open(env_file, 'w') as f:
        f.write(content)


def setup_postgresql():
    """Set up PostgreSQL using Docker Compose"""
    print("\n🐘 Setting up PostgreSQL with Docker...")
    
    try:
        # Start PostgreSQL service
        print("Starting PostgreSQL container...")
        result = subprocess.run([
            'docker-compose', 'up', '-d', 'postgres'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ PostgreSQL container started successfully")
            
            # Wait for PostgreSQL to be ready
            print("Waiting for PostgreSQL to be ready...")
            result = subprocess.run([
                'docker-compose', 'exec', '-T', 'postgres',
                'pg_isready', '-U', 'trading_bot', '-d', 'ai_trading_bot'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ PostgreSQL is ready for connections")
                return True
            else:
                print("⚠️  PostgreSQL started but not yet ready")
                print("   Run 'docker-compose logs postgres' to check status")
                return True
        else:
            print(f"❌ Failed to start PostgreSQL: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("❌ docker-compose not found. Please install Docker Compose.")
        return False


def install_python_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python Dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Python dependencies installed successfully")
            return True
        else:
            print(f"❌ Failed to install dependencies: {result.stderr}")
            print("\n💡 Try creating a virtual environment:")
            print("   python3 -m venv venv")
            print("   source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
            print("   pip install -r requirements.txt")
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def test_setup(database_choice):
    """Test the setup by running database verification"""
    print("\n🧪 Testing Setup...")
    
    try:
        # Test configuration
        result = subprocess.run([
            sys.executable, 'scripts/verify_database_connection.py'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Database connection test passed")
            return True
        else:
            print(f"❌ Database connection test failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error testing setup: {e}")
        return False


def print_next_steps(database_choice):
    """Print next steps for the user"""
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    
    if database_choice == '1':  # PostgreSQL
        print("\n🐘 PostgreSQL Development Environment Ready")
        print("\n📋 Useful Commands:")
        print("   # Start PostgreSQL")
        print("   docker-compose up -d postgres")
        print()
        print("   # Stop PostgreSQL")
        print("   docker-compose down")
        print()
        print("   # View PostgreSQL logs")
        print("   docker-compose logs postgres")
        print()
        print("   # Connect to PostgreSQL")
        print("   docker-compose exec postgres psql -U trading_bot -d ai_trading_bot")
        print()
    else:  # SQLite
        print("\n🗃️  SQLite Development Environment Ready")
        print("\n📋 Database Location:")
        print("   src/data/trading_bot.db")
        print()
    
    print("🚀 Run Your First Backtest:")
    print("   python scripts/run_backtest.py adaptive --days 30 --no-db")
    print()
    print("🔧 Verify Database Connection:")
    print("   python scripts/verify_database_connection.py")
    print()
    print("📊 Start Live Trading (Paper Mode):")
    print("   python scripts/run_live_trading.py adaptive")
    print()
    print("📈 Start Dashboard:")
    print("   python scripts/start_dashboard.py")
    print()
    
    if database_choice == '1':
        print("🔄 Switch to SQLite Later:")
        print("   1. Edit .env file")
        print("   2. Comment out DATABASE_URL line")
        print("   3. Restart your application")
        print()


def main():
    """Main setup function"""
    print_header()
    
    # Check system requirements
    if not check_requirements():
        print("Please install missing requirements and run setup again.")
        sys.exit(1)
    
    # Let user choose database option
    database_choice = choose_database_option()
    
    # Set up environment file
    setup_environment_file(database_choice)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("\n⚠️  Python dependencies installation failed.")
        print("Please install dependencies manually and re-run setup.")
        # Continue with setup even if pip install fails
    
    # Set up PostgreSQL if chosen
    if database_choice == '1':
        if not setup_postgresql():
            print("\n❌ PostgreSQL setup failed.")
            print("You can still use SQLite by editing your .env file.")
            sys.exit(1)
    
    # Test the setup
    print("\n⏳ Please wait while we test the setup...")
    if test_setup(database_choice):
        print_next_steps(database_choice)
    else:
        print("\n⚠️  Setup completed but tests failed.")
        print("Check the error messages above and try the manual commands.")


if __name__ == "__main__":
    main()