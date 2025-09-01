from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys

# Ensure project root and src are in sys.path for absolute imports
from src.utils.project_paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))


def _setup(ns: argparse.Namespace) -> int:
    """Set up local development environment"""
    
    print("🚀 AI Trading Bot - Local Development Setup")
    print("=" * 60)
    print()
    
    # Check system requirements
    if not _check_requirements():
        print("Please install missing requirements and run setup again.")
        return 1
    
    # Set up environment file
    _setup_environment_file()
    
    # Install Python dependencies
    if not _install_python_dependencies():
        print("\n⚠️  Python dependencies installation failed.")
        print("Please install dependencies manually and re-run setup.")
    
    # Set up PostgreSQL
    if not _setup_postgresql():
        print("\n❌ PostgreSQL setup failed.")
        print("You can still use PostgreSQL by editing your .env file.")
        return 1
    
    # Run database migrations
    if not _run_migrations():
        print("\n⚠️  Database migrations failed.")
        print("You can run migrations manually with: atb db migrate")
    
    # Test the setup
    print("\n⏳ Please wait while we test the setup...")
    if _test_setup():
        _print_next_steps()
        return 0
    else:
        print("\n⚠️  Setup completed but tests failed.")
        print("Check the error messages above and try the manual commands.")
        return 1


def _check_requirements() -> bool:
    """Check if required tools are available."""
    print("🔍 Checking System Requirements...")

    requirements = {
        "python3": {
            "description": "Python 3.9+ required",
            "required": True,
        },
        "docker": {
            "description": "Docker recommended for containerised PostgreSQL",
            "required": False,
        },
        "docker-compose": {
            "description": "Docker Compose recommended for containerised PostgreSQL",
            "required": False,
        },
    }

    missing_required = []
    missing_optional = []

    for tool, meta in requirements.items():
        description = meta["description"]
        try:
            result = subprocess.run([tool, "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split("\n")[0]
                print(f"✅ {tool}: {version}")
            else:
                (missing_required if meta["required"] else missing_optional).append(
                    (tool, description)
                )
        except FileNotFoundError:
            (missing_required if meta["required"] else missing_optional).append((tool, description))

    if missing_required or missing_optional:
        print("\n⚠️  Missing Requirements:")
        for tool, description in missing_required + missing_optional:
            print(f"   - {tool}: {description}")
        print()

    return len(missing_required) == 0


def _setup_environment_file() -> None:
    """Set up the .env file for PostgreSQL."""
    print("\n📝 Setting up Environment Configuration...")

    # Copy example file if .env doesn't exist
    env_file = PROJECT_ROOT / ".env"
    example_file = PROJECT_ROOT / ".env.example"

    if not env_file.exists():
        if example_file.exists():
            shutil.copy(example_file, env_file)
            print("✅ Created .env file from .env.example")
        else:
            # Create minimal .env file
            with open(env_file, "w") as f:
                f.write("# AI Trading Bot Environment Configuration\n")
                f.write("TRADING_MODE=paper\n")
                f.write("INITIAL_BALANCE=10000\n")
                f.write("LOG_LEVEL=INFO\n")
            print("✅ Created basic .env file")

    # Update .env file for PostgreSQL
    with open(env_file) as f:
        content = f.read()

    postgres_url_line = (
        "DATABASE_URL=postgresql://trading_bot:dev_password_123@localhost:5432/ai_trading_bot"
    )

    lines = [ln for ln in content.split("\n") if ln.strip()]
    # Remove any existing DATABASE_URL lines (commented or malformed)
    lines = [ln for ln in lines if not ln.strip().startswith("DATABASE_URL")]
    # Add the correct line
    lines.append(postgres_url_line)
    content = "\n".join(lines) + "\n"
    print("✅ Configured for PostgreSQL")

    with open(env_file, "w") as f:
        f.write(content)


def _setup_postgresql() -> bool:
    """Ensure a PostgreSQL instance is available."""
    print("\n🐘 Setting up PostgreSQL...")

    # Quick check for docker-compose
    docker_compose_available = shutil.which("docker-compose") is not None

    if not docker_compose_available:
        print("⚠️  docker-compose not found – Skipping container startup.")
        print(
            "   Ensure your local PostgreSQL instance is running and matches the credentials in the .env file."
        )
        return True  # Considered success; user handles DB themselves

    # Start container via docker-compose
    try:
        print("Starting PostgreSQL container via docker-compose...")
        result = subprocess.run(
            ["docker-compose", "up", "-d", "postgres"], 
            cwd=PROJECT_ROOT,
            capture_output=True, 
            text=True
        )

        if result.returncode != 0:
            print(f"❌ Failed to start PostgreSQL container: {result.stderr}")
            return False

        print("✅ PostgreSQL container started successfully")

        # Wait for readiness (basic check)
        print("Waiting for PostgreSQL to be ready...")
        result = subprocess.run(
            [
                "docker-compose",
                "exec",
                "-T",
                "postgres",
                "pg_isready",
                "-U",
                "trading_bot",
                "-d",
                "ai_trading_bot",
            ],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ PostgreSQL is ready for connections")
        else:
            print("⚠️  PostgreSQL started but readiness check failed.")
            print("   You may need to check logs: docker-compose logs postgres")

        return True

    except Exception as e:
        print(f"❌ Error setting up PostgreSQL via docker-compose: {e}")
        return False


def _install_python_dependencies() -> bool:
    """Install Python dependencies."""
    print("\n📦 Installing Python Dependencies...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

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


def _run_migrations() -> bool:
    """Run database migrations using Alembic."""
    print("\n🔄 Running Database Migrations...")
    
    try:
        # Check if DATABASE_URL is available
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL environment variable not found")
            print("   Please ensure your .env file is properly configured")
            return False
        
        print(f"✅ Database URL found: {database_url[:20]}...")
        
        # Run alembic upgrade
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Database migrations completed successfully")
            if result.stdout:
                print("📋 Migration output:")
                print(result.stdout)
            return True
        else:
            print("❌ Database migrations failed")
            print("📋 Error output:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False


def _test_setup() -> bool:
    """Test the setup by running database verification."""
    print("\n🧪 Testing Setup...")

    try:
        # Test configuration
        result = subprocess.run(
            [sys.executable, "scripts/verify_database_connection.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("✅ Database connection test passed")
            return True
        else:
            print("❌ Database connection test failed:")
            print(result.stdout)
            print(result.stderr)
            return False

    except Exception as e:
        print(f"❌ Error testing setup: {e}")
        return False


def _print_next_steps() -> None:
    """Print next steps for the user."""
    print("\n🎉 Setup Complete!")
    print("=" * 40)

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
    print("   atb backtest ml_basic --days 30 --no-db")
    print()
    print("🔧 Verify Database Connection:")
    print("   atb db verify")
    print()
    print("📊 Start Live Trading (Paper Mode):")
    print("   atb live ml_basic")
    print()
    print("📈 Start Dashboard:")
    print("   atb dev dashboard")


def _venv(ns: argparse.Namespace) -> int:
    """Set up virtual environment for development."""
    print("Setting up development environment...")
    
    venv_path = PROJECT_ROOT / ".venv"
    
    # Create virtual environment if it doesn't exist
    if not venv_path.exists():
        print(f"Creating virtual environment at {venv_path}")
        result = subprocess.run([sys.executable, "-m", "venv", str(venv_path)], cwd=PROJECT_ROOT)
        if result.returncode != 0:
            print("❌ Failed to create virtual environment")
            return 1
    else:
        print(f"Virtual environment already exists at {venv_path}")
    
    # Determine the correct pip path
    if os.name == "nt":  # Windows
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/macOS
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip
    print("Upgrading pip...")
    result = subprocess.run([str(pip_path), "install", "--upgrade", "pip"], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("❌ Failed to upgrade pip")
        return 1
    
    # Install the package in editable mode
    print("Installing package in editable mode...")
    result = subprocess.run([str(pip_path), "install", "-e", "."], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("❌ Failed to install package in editable mode")
        return 1
    
    # Install development dependencies
    print("Installing development dependencies...")
    result = subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], cwd=PROJECT_ROOT)
    if result.returncode != 0:
        print("❌ Failed to install development dependencies")
        return 1
    
    print("\n✅ Development environment setup complete!")
    print("\nTo activate the virtual environment:")
    if os.name == "nt":  # Windows
        print(f"  {venv_path}\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print(f"  source {venv_path}/bin/activate")
    
    print("\nOr use the CLI commands (they handle venv automatically):")
    print("  atb live          # Run live trading")
    print("  atb backtest      # Run backtest")
    print("  atb tests         # Run tests")
    
    return 0


def _dashboard(ns: argparse.Namespace) -> int:
    """Run the monitoring dashboard."""
    from src.dashboards.monitoring.dashboard import MonitoringDashboard
    from src.utils.logging_config import configure_logging
    
    # Configure logging
    configure_logging(use_json=True)
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", "8090"))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"Starting monitoring dashboard on {host}:{port}")
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"Received signal {signum}, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        # Create and run dashboard with timeout
        print("Initializing dashboard...")
        dashboard = MonitoringDashboard()
        print("Dashboard initialized successfully")
        
        print("Starting dashboard server...")
        dashboard.run(host=host, port=port, debug=False)
        return 0
    except KeyboardInterrupt:
        print("Dashboard stopped by user")
        return 0
    except Exception as e:
        print(f"Dashboard failed to start: {e}")
        return 1


def register(parser: argparse._SubParsersAction) -> None:
    """Register development commands."""
    dev_parser = parser.add_parser("dev", help="Local development utilities")
    dev_subparsers = dev_parser.add_subparsers(dest="dev_command", required=True)
    
    # Setup command
    setup_parser = dev_subparsers.add_parser("setup", help="Set up local development environment")
    setup_parser.set_defaults(func=_setup)
    
    # Virtual environment command
    venv_parser = dev_subparsers.add_parser("venv", help="Set up virtual environment")
    venv_parser.set_defaults(func=_venv)
    
    # Dashboard command
    dashboard_parser = dev_subparsers.add_parser("dashboard", help="Run monitoring dashboard")
    dashboard_parser.set_defaults(func=_dashboard)
