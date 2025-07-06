#!/usr/bin/env python3
"""
Local Development Setup Script
Helps developers set up their local development environment with PostgreSQL (default)
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
    """Check if required tools are available.

    `python3` is mandatory. Docker and Docker-Compose are recommended for the default
    containerised PostgreSQL setup, but their absence should not abort the entire
    process, as developers may run a local PostgreSQL instance instead.
    """
    print("🔍 Checking System Requirements...")

    requirements = {
        'python3': {
            'description': 'Python 3.9+ required',
            'required': True,
        },
        'docker': {
            'description': 'Docker recommended for containerised PostgreSQL',
            'required': False,
        },
        'docker-compose': {
            'description': 'Docker Compose recommended for containerised PostgreSQL',
            'required': False,
        },
    }

    missing_required = []
    missing_optional = []

    for tool, meta in requirements.items():
        description = meta['description']
        try:
            result = subprocess.run([tool, '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0]
                print(f"✅ {tool}: {version}")
            else:
                (missing_required if meta['required'] else missing_optional).append((tool, description))
        except FileNotFoundError:
            (missing_required if meta['required'] else missing_optional).append((tool, description))

    if missing_required or missing_optional:
        print("\n⚠️  Missing Requirements:")
        for tool, description in missing_required + missing_optional:
            print(f"   - {tool}: {description}")
        print()

    # Fail only if required dependencies are missing
    return len(missing_required) == 0


def choose_database_option():
    """Return PostgreSQL as the default database option.

    The project has standardized on PostgreSQL for local development. This helper exists
    for backward compatibility with the original flow, but it now simply returns "1"
    to indicate PostgreSQL without any interactive prompt.
    """
    print("🗄️  Database Configuration: Defaulting to PostgreSQL (🐘)")
    return '1'


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
        postgres_url_line = 'DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot'

        lines = [ln for ln in content.split('\n') if ln.strip()]
        # Remove any existing DATABASE_URL lines (commented or malformed)
        lines = [ln for ln in lines if not ln.strip().startswith('DATABASE_URL')]
        # Add the correct line
        lines.append(postgres_url_line)
        content = '\n'.join(lines) + '\n'
        print("✅ Configured for PostgreSQL")
    # PostgreSQL is the only option
    print("✅ Configured for PostgreSQL")
    
    with open(env_file, 'w') as f:
        f.write(content)


def setup_postgresql():
    """Ensure a PostgreSQL instance is available.

    Preference order:
    1. If `docker-compose` is available – spin up the container defined in the
       local `docker-compose.yml`.
    2. If Docker tooling is **not** available – assume the developer has a local
       PostgreSQL instance running (or will start it manually) and simply warn.
    """
    print("\n🐘 Setting up PostgreSQL...")

    # Quick check for docker-compose
    docker_compose_available = shutil.which('docker-compose') is not None

    if not docker_compose_available:
        print("⚠️  docker-compose not found – Skipping container startup.")
        print("   Ensure your local PostgreSQL instance is running and matches the credentials in the .env file.")
        return True  # Considered success; user handles DB themselves

    # Start container via docker-compose
    try:
        print("Starting PostgreSQL container via docker-compose...")
        result = subprocess.run([
            'docker-compose', 'up', '-d', 'postgres'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ Failed to start PostgreSQL container: {result.stderr}")
            return False

        print("✅ PostgreSQL container started successfully")

        # Wait for readiness (basic check)
        print("Waiting for PostgreSQL to be ready...")
        result = subprocess.run([
            'docker-compose', 'exec', '-T', 'postgres',
            'pg_isready', '-U', 'trading_bot', '-d', 'ai_trading_bot'
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ PostgreSQL is ready for connections")
        else:
            print("⚠️  PostgreSQL started but readiness check failed.")
            print("   You may need to check logs: docker-compose logs postgres")

        return True

    except Exception as e:
        print(f"❌ Error setting up PostgreSQL via docker-compose: {e}")
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
    # PostgreSQL is the only option
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
            print("You can still use PostgreSQL by editing your .env file.")
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