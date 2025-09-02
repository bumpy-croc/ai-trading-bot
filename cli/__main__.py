from __future__ import annotations

import argparse
import sys
from pathlib import Path

# * Add project root to Python path for src imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# * Apply gevent monkey patching BEFORE any other imports if needed
def _apply_very_early_gevent_patching() -> None:
    """Apply gevent monkey patching at the very start, before any imports."""
    import os
    
    # Check if this will be a dashboard command and gevent is requested
    if "WEB_SERVER_USE_GEVENT" in os.environ and os.environ["WEB_SERVER_USE_GEVENT"] == "1":
        # Check if this looks like a dashboard command
        if len(sys.argv) >= 3 and sys.argv[1] == "dashboards" and sys.argv[2] == "run":
            try:
                import gevent.monkey
                gevent.monkey.patch_all()
                print("✅ Applied very early gevent monkey patching for dashboard command")
            except ImportError:
                print("⚠️ Gevent not available for very early monkey patching")

# Apply gevent patching immediately if needed
_apply_very_early_gevent_patching()

from src.utils.logging_config import configure_logging


def load_env_file_to_environment() -> None:
    """
    Load .env file and export variables to the current environment.

    This ensures that subprocess calls (like Railway CLI) can access
    the environment variables defined in .env.
    """
    import os
    from pathlib import Path

    env_file = Path(".env")
    if not env_file.exists():
        return

    try:
        with open(env_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    # Remove quotes if present
                    value = value.strip().strip('"').strip("'")
                    # Export to environment
                    os.environ[key] = value
        print(f"✅ Loaded environment variables from {env_file}")
    except Exception as e:
        print(f"⚠️  Warning: Failed to load {env_file}: {e}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Trading Bot unified CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Import and register command groups
    from cli.commands import (
        backtest,
        dashboards,
        data,
        db,
        dev,
        live,
        live_health,
        optimizer,
        tests,
        train,
    )
    from cli.commands import scripts as scripts_cmd

    dashboards.register(subparsers)
    live.register(subparsers)
    backtest.register(subparsers)
    optimizer.register(subparsers)
    scripts_cmd.register(subparsers)
    live_health.register(subparsers)
    data.register(subparsers)
    db.register(subparsers)
    dev.register(subparsers)
    train.register(subparsers)
    tests.register(subparsers)

    return parser





def main(argv: list[str] | None = None) -> int:
    # Load .env file to environment variables before anything else
    load_env_file_to_environment()

    configure_logging()
    parser = build_parser()
    ns = parser.parse_args(argv)

    func = getattr(ns, "func", None)
    if callable(func):
        return int(func(ns) or 0)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
