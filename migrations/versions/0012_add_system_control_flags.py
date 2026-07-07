"""Add system_control_flags table (manual kill-switch, #922)

Revision ID: 0012_add_system_control_flags
Revises: 0011_drop_optimization_cycles
Create Date: 2026-07-07 12:00:00.000000

Durable operator-set control flags the live engine polls at runtime — one row
per flag name, upserted. The ``system_halt`` row is the manual kill-switch
written by ``atb live-control halt`` / ``resume``.

Guarded: the app's runtime ``Base.metadata.create_all`` may already have
created the table on environments that booted the new code (or ran the halt
CLI) before this migration, so upgrade is a no-op when the table exists.
"""

import sqlalchemy as sa
from alembic import op

revision = "0012_add_system_control_flags"
down_revision = "0011_drop_optimization_cycles"
branch_labels = None
depends_on = None

TABLE = "system_control_flags"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE in inspector.get_table_names():
        return
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=50), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    # Matches the model's `unique=True, index=True` (create_all emits a single
    # unique index named ix_<table>_name, not a separate unique constraint).
    op.create_index(f"ix_{TABLE}_name", TABLE, ["name"], unique=True)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if TABLE not in inspector.get_table_names():
        return
    existing_indexes = {idx["name"] for idx in inspector.get_indexes(TABLE)}
    if f"ix_{TABLE}_name" in existing_indexes:
        op.drop_index(f"ix_{TABLE}_name", table_name=TABLE)
    op.drop_table(TABLE)
