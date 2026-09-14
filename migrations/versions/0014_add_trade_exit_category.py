"""Add trades.exit_category and the legacy-mapping view

Revision ID: 0014_add_exit_category
Revises: 0013_widen_event_type
Create Date: 2026-08-25 00:00:00.000000

GH #1115. ``trades.exit_reason`` stays free text — it is historical record and its
values are embedded in ``account_balances.update_reason`` keys — and gains a typed
companion, ``exit_category``, carrying ``src.trading.exit_reason.ExitReason``.

Existing rows are deliberately NOT backfilled. The prose never distinguished a
protective stop from a trailing stop that took profit, so any UPDATE would encode a
guess as fact. ``v_trades_exit_category`` instead exposes the mapping at read time
with an ``exit_category_inferred`` flag, so an analyst can always see which rows are
engine-recorded and which are reconstructed. The per-row prod mapping (including the
rows whose true category is recoverable from ``positions.trailing_stop_activated``)
is recorded in ``agents/research/1115-exit-taxonomy.md``.
"""

import sqlalchemy as sa
from alembic import op

revision = "0014_add_exit_category"
down_revision = "0013_widen_event_type"
branch_labels = None
depends_on = None

# Legacy exit_reason -> category. Mirrors LEGACY_EXIT_REASON_CATEGORIES in
# src/trading/exit_reason.py; tests/unit/database/test_exit_category_migration.py
# asserts the two stay in step.
_LEGACY_VIEW = """
CREATE OR REPLACE VIEW v_trades_exit_category AS
SELECT
    t.id,
    t.session_id,
    t.position_id,
    t.symbol,
    t.side,
    t.source,
    t.strategy_name,
    t.entry_time,
    t.exit_time,
    t.entry_price,
    t.exit_price,
    t.quantity,
    t.size,
    t.pnl,
    t.pnl_percent,
    t.commission,
    t.margin_interest_cost,
    t.stop_loss,
    t.take_profit,
    t.mfe,
    t.mae,
    t.exit_reason,
    t.exit_category,
    COALESCE(
        t.exit_category,
        CASE
            WHEN t.exit_reason IN (
                'Stop loss', 'stop_loss', 'stop_loss_offline', 'stop_loss_filled_offline'
            ) THEN 'stop_loss'
            WHEN t.exit_reason IN ('Take profit', 'take_profit') THEN 'take_profit'
            WHEN t.exit_reason IN ('Signal reversal', 'Strategy signal') THEN 'signal_exit'
            WHEN t.exit_reason IN (
                'Time exit', 'time_exit', 'Max holding period', 'Weekend flat', 'End of day flat'
            ) THEN 'time_exit'
            WHEN t.exit_reason LIKE 'Early cut%' THEN 'early_cut'
            WHEN t.exit_reason LIKE 'Partial exits complete%' THEN 'partial_exit_complete'
            WHEN t.exit_reason IN (
                'Stop-loss placement failed - emergency close', 'Risk manager sync failure'
            ) THEN 'emergency_close'
            WHEN t.exit_reason = 'Engine shutdown' THEN 'engine_shutdown'
            WHEN t.exit_reason = 'Strategy change - close requested' THEN 'strategy_change'
            WHEN t.exit_reason IN ('external_close_recovery', 'manual_close') THEN 'external_close'
            WHEN t.exit_reason IN ('recovered_from_exchange', 'exit_order_recovery')
                THEN 'recovered'
            ELSE 'unknown'
        END
    ) AS exit_category_resolved,
    (t.exit_category IS NULL) AS exit_category_inferred
FROM trades t;
"""


def upgrade() -> None:
    op.add_column("trades", sa.Column("exit_category", sa.String(32), nullable=True))
    op.create_index("idx_trade_exit_category", "trades", ["exit_category"])
    op.execute(_LEGACY_VIEW)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS v_trades_exit_category")
    op.drop_index("idx_trade_exit_category", table_name="trades")
    op.drop_column("trades", "exit_category")
