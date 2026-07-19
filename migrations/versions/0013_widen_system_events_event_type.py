"""Widen system_events.event_type varchar for CIRCUIT_BREAKER_DRY_RUN

Revision ID: 0013_widen_event_type
Revises: 0012_add_system_control_flags
Create Date: 2026-07-12 00:00:00.000000

EventType gains CIRCUIT_BREAKER_DRY_RUN (23 chars) so dry-run circuit-breaker
trips write DB-durable evidence (#964). The non-native enum column was created
as varchar(18) (sized for BALANCE_ADJUSTMENT); mirror of migration 0009's
orders.status widening. Width matches the model-compiled type exactly so
`atb db verify` type comparison stays clean.
"""

import sqlalchemy as sa
from alembic import op

revision = "0013_widen_event_type"
down_revision = "0012_add_system_control_flags"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column(
        "system_events",
        "event_type",
        existing_type=sa.String(18),
        type_=sa.String(23),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "system_events",
        "event_type",
        existing_type=sa.String(23),
        type_=sa.String(18),
        existing_nullable=False,
    )
