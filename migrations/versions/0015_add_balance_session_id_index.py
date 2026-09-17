"""Add (session_id, id) index on account_balances

Revision ID: 0015_balance_session_id_idx
Revises: 0014_add_exit_category
Create Date: 2026-09-16 00:00:00.000000

GH #736/#1224 review. ``AccountBalance.get_current_balance`` now orders by ``id``
descending instead of ``last_updated`` (two writers can commit within the same
microsecond, making ``last_updated`` ties ambiguous — #735), but the existing
``idx_balance_session_updated`` index is on ``(session_id, last_updated)`` and
does not serve that query: it falls back to sorting every row for the session.
That query now also runs inside ``DatabaseManager._lock_balance_ledger``'s
advisory lock on every balance write, so its latency directly serializes every
writer. Add a composite index that actually matches the new ordering; keep the
old index since other queries still want ``last_updated`` ordering.
"""

from alembic import op

revision = "0015_balance_session_id_idx"
down_revision = "0014_add_exit_category"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "idx_balance_session_id",
        "account_balances",
        ["session_id", "id"],
    )


def downgrade() -> None:
    op.drop_index("idx_balance_session_id", table_name="account_balances")
