"""Alembic environment configuration.
Sets up context for autogeneration and migrations using the project's SQLAlchemy models.
"""

import os
import sys
from logging.config import fileConfig

from alembic import context  # type: ignore
from sqlalchemy import engine_from_config, pool  # type: ignore

# Ensure project root is on PYTHONPATH
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, BASE_DIR)

# Import metadata
from src.config.config_manager import get_config  # noqa: E402
from src.database.models import Base  # noqa: E402

# --------------------------------------------------
# Alembic Config
# --------------------------------------------------
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Metadata for 'autogenerate --compare-type' support
target_metadata = Base.metadata


def get_url() -> str:
    """Resolve the database URL from environment or config manager.

    Raises:
        RuntimeError: If the database URL cannot be determined.
    """
    cfg = get_config()
    url: str | None = cfg.get("DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "DATABASE_URL must be set in environment or configuration for Alembic migrations."
        )
    return url


# --------------------------------------------------
# Migration helpers
# --------------------------------------------------


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        {"sqlalchemy.url": get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
