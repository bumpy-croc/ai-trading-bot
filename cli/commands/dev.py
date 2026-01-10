from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

# Ensure project root and src are in sys.path for absolute imports
from src.infrastructure.runtime.paths import get_project_root

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

    requirements: dict[str, dict[str, Any]] = {
        "python3": {
            "description": "Python 3.11+ required",
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

    missing_required: list[tuple[str, str]] = []
    missing_optional: list[tuple[str, str]] = []

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

    has_database_url = any(
        line.strip().startswith("DATABASE_URL=") for line in content.splitlines()
    )
    if has_database_url:
        print("ℹ️ Existing DATABASE_URL detected; leaving .env unchanged.")
        return

    new_content = content
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += postgres_url_line + "\n"
    print("✅ Added default PostgreSQL DATABASE_URL")

    with open(env_file, "w") as f:
        f.write(new_content)


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
            text=True,
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

    python311 = _find_python311()
    if not python311:
        print(
            "❌ python3.11 not found. Install it (e.g. `brew install python@3.11`) or set the PYTHON311 env var."
        )
        return False

    venv_python = _ensure_python311_venv(python311)
    if not venv_python:
        return False

    print(f"✅ Using virtualenv interpreter: {venv_python}")

    if not _run_cmd([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], "pip upgrade"):
        return False

    return _run_cmd(
        [str(venv_python), "-m", "pip", "install", "-r", str(PROJECT_ROOT / "requirements.txt")],
        "dependency install",
        cwd=None,
    )


def _run_cmd(cmd: list[str], description: str, cwd: Path | None = PROJECT_ROOT) -> bool:
    try:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    except Exception as exc:  # pragma: no cover - subprocess failures raise here
        print(f"❌ {description} failed: {exc}")
        return False

    if result.returncode == 0:
        print(f"✅ {description} completed")
        if result.stdout:
            print(result.stdout.strip())
        return True

    print(f"❌ {description} failed")
    if result.stderr:
        print(result.stderr.strip())
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
            ["alembic", "upgrade", "head"], cwd=PROJECT_ROOT, capture_output=True, text=True
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
        from src.config.config_manager import get_config
        from src.database.manager import DatabaseManager

        config = get_config()
        database_url = config.get("DATABASE_URL")
        if not database_url:
            print("❌ DATABASE_URL not set")
            return False

        db_manager = DatabaseManager()
        if not db_manager.test_connection():
            print("❌ Database connection test failed")
            return False

        session_id = db_manager.create_trading_session(
            strategy_name="SetupVerification",
            symbol="BTCUSDT",
            timeframe="1h",
            mode="PAPER",
            initial_balance=10_000.0,
            session_name="dev_setup_check",
        )
        db_manager.end_trading_session(session_id, final_balance=10_000.0)
        print("✅ Database connection test passed")
        return True
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


def _find_python311() -> Path | None:
    candidates: list[str] = []
    explicit = os.environ.get("PYTHON311")
    if explicit:
        candidates.append(explicit)

    which_candidate = shutil.which("python3.11")
    if which_candidate:
        candidates.append(which_candidate)

    default_homebrew = Path("/opt/homebrew/opt/python@3.11/bin/python3.11")
    candidates.append(str(default_homebrew))

    for candidate in candidates:
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    return None


def _ensure_python311_venv(python311: Path) -> Path | None:
    venv_dir = PROJECT_ROOT / ".venv"
    venv_python = _venv_python_path(venv_dir)

    def _create_venv() -> bool:
        if venv_dir.exists():
            shutil.rmtree(venv_dir)
        print(f"📁 Creating .venv with {python311}")
        result = subprocess.run(
            [str(python311), "-m", "venv", str(venv_dir)], capture_output=True, text=True
        )
        if result.returncode != 0:
            print("❌ Failed to create virtualenv:")
            print(result.stderr.strip())
            return False
        return True

    if not venv_python.exists():
        if not _create_venv():
            return None
    else:
        version_result = subprocess.run(
            [str(venv_python), "--version"], capture_output=True, text=True
        )
        version_str = version_result.stdout.strip()
        if "3.11" not in version_str:
            print("♻️ Existing .venv is not Python 3.11; recreating...")
            if not _create_venv():
                return None

    return _venv_python_path(venv_dir)


def _venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


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
    from src.infrastructure.logging.config import configure_logging

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


def _get_changed_files(branch: str | None = None) -> list[Path]:
    """Get list of changed Python files relative to git base or working directory.

    Args:
        branch: Compare against this branch (e.g., 'origin/develop').
                If None, returns all modified files in working directory.

    Returns:
        List of Paths for changed Python files.
    """
    import subprocess

    try:
        if branch:
            # Get files changed compared to branch
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=d", branch],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            # Get modified files in working directory (staged + unstaged)
            result = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=d", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        if result.returncode != 0:
            return []

        changed_files = []
        for line in result.stdout.splitlines():
            path = PROJECT_ROOT / line
            if path.suffix == ".py":
                changed_files.append(path)

        return changed_files

    except (subprocess.SubprocessError, FileNotFoundError):
        return []


def _quality(ns: argparse.Namespace) -> int:
    """Run code quality checks (black, ruff, mypy, bandit)."""
    changed_only = getattr(ns, "changed", False)
    branch = getattr(ns, "branch", None)

    if changed_only:
        changed_files = _get_changed_files(branch)
        if not changed_files:
            print("ℹ️ No Python files changed - skipping quality checks")
            return 0
        print(f"🔍 Running Code Quality Checks on {len(changed_files)} changed file(s)")
        print("=" * 60)
        for f in changed_files:
            print(f"  {f.relative_to(PROJECT_ROOT)}")
        print()
    else:
        print("🔍 Running Code Quality Checks")
        print("=" * 60)
        print()

    # Build target list based on mode
    if changed_only:
        file_strs = [str(f.relative_to(PROJECT_ROOT)) for f in changed_files]
        # Group files by directory for more efficient tool runs
        src_files = [f for f in file_strs if f.startswith("src/")]
        cli_files = [f for f in file_strs if f.startswith("cli/")]
        test_files = [f for f in file_strs if f.startswith("tests/")]
        # Files outside main dirs get checked individually
        other_files = [f for f in file_strs if not f.startswith(("src/", "cli/", "tests/"))]
    else:
        src_files = ["src"]
        cli_files = ["cli"]
        test_files = ["tests"]
        other_files = []

    # Pre-compute file groups for changed mode (used by multiple tools)
    all_targets: list[str] = (
        [t for group in [src_files, cli_files, test_files, other_files] for t in group]
        if changed_only
        else []
    )
    src_cli_targets: list[str] = (
        [t for group in [src_files, cli_files] for t in group] if changed_only else []
    )

    tools: list[dict[str, Any]] = []

    # Black - formatter
    if changed_only:
        if all_targets:
            tools.append(
                {
                    "name": "Black (code formatter)",
                    "cmd": ["black", "--check"] + all_targets,
                    "description": "Checking code formatting",
                }
            )
    else:
        tools.append(
            {
                "name": "Black (code formatter)",
                "cmd": ["black", "."],
                "description": "Checking code formatting",
            }
        )

    # Ruff - linter
    if changed_only:
        if all_targets:
            tools.append(
                {
                    "name": "Ruff (linter)",
                    "cmd": ["ruff", "check"] + all_targets,
                    "description": "Running linter checks",
                }
            )
    else:
        tools.append(
            {
                "name": "Ruff (linter)",
                "cmd": ["ruff", "check", "."],
                "description": "Running linter checks",
            }
        )

    # MyPy - type checker
    if changed_only:
        # Only run mypy on changed src/cli files (tests use different config)
        if src_cli_targets:
            tools.append(
                {
                    "name": "MyPy (type checker)",
                    "cmd": [sys.executable, "bin/run_mypy.py", "--targets"] + src_cli_targets,
                    "description": "Running type checks",
                }
            )
    else:
        tools.append(
            {
                "name": "MyPy (type checker)",
                "cmd": [sys.executable, "bin/run_mypy.py"],
                "description": "Running type checks",
            }
        )

    # Bandit - security (always runs on full src dir, as partial runs miss context)
    if not changed_only:
        tools.append(
            {
                "name": "Bandit (security scanner)",
                "cmd": ["bandit", "-c", "pyproject.toml", "-r", "src"],
                "description": "Running security checks",
            }
        )

    results: dict[str, bool | None] = {}

    for tool in tools:
        print(f"\n{tool['description']}...")
        print(f"Command: {' '.join(tool['cmd'])}")

        try:
            result = subprocess.run(
                tool["cmd"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0:
                print(f"✅ {tool['name']} passed")
                results[tool["name"]] = True
            else:
                print(f"❌ {tool['name']} failed")
                # Show stdout and stderr (black/ruff emit diagnostics on stdout)
                if result.stdout:
                    print(f"\n{tool['name']} output:")
                    print(result.stdout)
                if result.stderr:
                    print(f"\n{tool['name']} errors:")
                    print(result.stderr)
                results[tool["name"]] = False

        except FileNotFoundError:
            print(f"⚠️ {tool['name']} not found - skipping")
            results[tool["name"]] = None
        except Exception as e:
            print(f"❌ {tool['name']} failed with exception: {e}")
            results[tool["name"]] = False

    # Print summary
    print("\n" + "=" * 60)
    print("Code Quality Summary")
    print("=" * 60)

    for tool_name, status in results.items():
        if status is True:
            print(f"✅ {tool_name}")
        elif status is False:
            print(f"❌ {tool_name}")
        else:
            print(f"⚠️ {tool_name} (skipped)")

    # Return non-zero if any tool failed or was missing
    if any(status is False for status in results.values()):
        print("\n❌ Some quality checks failed")
        return 1

    if any(status is None for status in results.values()):
        print("\n❌ Some quality tools are missing - please install them")
        return 1

    print("\n✅ All quality checks passed!")
    return 0


def _clean(ns: argparse.Namespace) -> int:
    """Remove caches and build artifacts."""
    print("🧹 Cleaning Project")
    print("=" * 60)
    print()

    items_to_remove = [
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
    ]

    # Remove specific directories
    for item in items_to_remove:
        path = PROJECT_ROOT / item
        if path.exists():
            shutil.rmtree(path)
            print(f"✅ Removed {item}/")

    # Remove egg-info directories
    for egg_info in PROJECT_ROOT.glob("*.egg-info"):
        shutil.rmtree(egg_info)
        print(f"✅ Removed {egg_info.name}/")

    # Remove __pycache__ directories
    pycache_count = 0
    for pycache in PROJECT_ROOT.rglob("__pycache__"):
        shutil.rmtree(pycache)
        pycache_count += 1

    if pycache_count > 0:
        print(f"✅ Removed {pycache_count} __pycache__ directories")

    print("\n✅ Clean complete!")
    return 0


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

    # Quality command
    quality_parser = dev_subparsers.add_parser("quality", help="Run code quality checks")
    quality_parser.add_argument(
        "--changed",
        "-c",
        action="store_true",
        help="Only check files changed relative to HEAD (or branch with --branch)",
    )
    quality_parser.add_argument(
        "--branch",
        "-b",
        type=str,
        default=None,
        help="Compare against this branch (e.g., 'origin/develop') for --changed",
    )
    quality_parser.set_defaults(func=_quality)

    # Clean command
    clean_parser = dev_subparsers.add_parser("clean", help="Remove caches and build artifacts")
    clean_parser.set_defaults(func=_clean)
