"""Widen orders.status varchar to accommodate new enum values

Revision ID: 0009_widen_order_status
Revises: 0008_reconciliation_schema
Create Date: 2026-03-28 16:00:00.000000

Migration 0008 added new OrderStatus values (PENDING_SUBMIT, SUBMITTED,
CONFIRMED, UNKNOWN, UNRESOLVED) but did not widen the varchar column.
The original column was varchar(9) (sized for CANCELLED). PENDING_SUBMIT
requires varchar(14). This caused pre-deploy verification to fail on all
Railway environments.
"""

import sqlalchemy as sa
from alembic import op

revision = "0009_widen_order_status"
down_revision = "0008_reconciliation_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Widen orders.status from varchar(9) to varchar(14) for PENDING_SUBMIT
    op.alter_column(
        "orders",
        "status",
        existing_type=sa.String(9),
        type_=sa.String(14),
        existing_nullable=False,
    )


def downgrade() -> None:
    op.alter_column(
        "orders",
        "status",
        existing_type=sa.String(14),
        type_=sa.String(9),
        existing_nullable=False,
    )
