"""Add reconciliation schema: audit events table, order journal columns, position client_order_id

Revision ID: 0008_reconciliation_schema
Revises: 0007_fix_balance_types
Create Date: 2026-03-20 19:00:00.000000

Adds infrastructure for Binance reconciliation:
- reconciliation_audit_events table for immutable correction audit trail
- client_order_id on orders and positions tables for idempotency key tracking
- actual_fill_price/quantity/commission on orders for authoritative exchange data
- replaced_order_id on orders for stop-loss replacement chain
- New OrderStatus values: PENDING_SUBMIT, SUBMITTED, CONFIRMED, UNKNOWN, UNRESOLVED
- New OrderType value: STOP_LOSS
- Makes orders.position_id nullable (entry orders journaled before position creation)

All changes are additive — no drops or renames — for zero-downtime deployment.
"""

import sqlalchemy as sa
from alembic import op

revision = "0008_reconciliation_schema"
down_revision = "0007_fix_balance_types"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- 1. New table: reconciliation_audit_events ---
    op.create_table(
        "reconciliation_audit_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("session_id", sa.Integer(), sa.ForeignKey("trading_sessions.id"), index=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("entity_type", sa.String(20), nullable=False),
        sa.Column("entity_id", sa.Integer()),
        sa.Column("field", sa.String(50), nullable=False),
        sa.Column("old_value", sa.Text()),
        sa.Column("new_value", sa.Text()),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("severity", sa.String(10), nullable=False),
    )
    op.create_index("idx_audit_session_time", "reconciliation_audit_events", ["session_id", "timestamp"])
    op.create_index("idx_audit_severity", "reconciliation_audit_events", ["severity", "timestamp"])

    # --- 2. Orders table: new columns ---
    op.add_column("orders", sa.Column("client_order_id", sa.String(100)))
    op.add_column("orders", sa.Column("actual_fill_price", sa.Numeric(18, 8)))
    op.add_column("orders", sa.Column("actual_fill_quantity", sa.Numeric(18, 8)))
    op.add_column("orders", sa.Column("actual_commission", sa.Numeric(18, 8)))
    op.add_column("orders", sa.Column("replaced_order_id", sa.Integer(), sa.ForeignKey("orders.id")))
    op.create_index("idx_order_client_id", "orders", ["client_order_id"])

    # Make position_id nullable (entry orders journaled before position creation)
    op.alter_column("orders", "position_id", existing_type=sa.Integer(), nullable=True)

    # --- 3. Positions table: client_order_id ---
    op.add_column("positions", sa.Column("client_order_id", sa.String(100)))
    op.create_index("idx_position_client_order_id", "positions", ["client_order_id"])


def downgrade() -> None:
    # Positions
    op.drop_index("idx_position_client_order_id", table_name="positions")
    op.drop_column("positions", "client_order_id")

    # Orders
    op.alter_column("orders", "position_id", existing_type=sa.Integer(), nullable=False)
    op.drop_index("idx_order_client_id", table_name="orders")
    op.drop_column("orders", "replaced_order_id")
    op.drop_column("orders", "actual_commission")
    op.drop_column("orders", "actual_fill_quantity")
    op.drop_column("orders", "actual_fill_price")
    op.drop_column("orders", "client_order_id")

    # Audit events table
    op.drop_index("idx_audit_severity", table_name="reconciliation_audit_events")
    op.drop_index("idx_audit_session_time", table_name="reconciliation_audit_events")
    op.drop_table("reconciliation_audit_events")
