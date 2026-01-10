"""Add enhanced performance metrics

Revision ID: 0005_enhanced_performance
Revises: 0004_preserve_entry_balance
Create Date: 2025-12-26 00:00:00.000000

Adds new performance tracking columns to support unified PerformanceTracker:
- VaR (Value at Risk) calculations
- Current drawdown tracking
- Consecutive win/loss streaks
- Average trade duration
- Fee and slippage costs
- Enhanced risk metrics for AccountHistory snapshots
"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0005_enhanced_performance"
down_revision = "0004_preserve_entry_balance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add new performance metric columns"""

    # Enhance AccountHistory table with risk metrics
    op.add_column(
        "account_history",
        sa.Column("sharpe_ratio", sa.Numeric(precision=18, scale=8), nullable=True),
    )
    op.add_column(
        "account_history",
        sa.Column("sortino_ratio", sa.Numeric(precision=18, scale=8), nullable=True),
    )
    op.add_column(
        "account_history",
        sa.Column("calmar_ratio", sa.Numeric(precision=18, scale=8), nullable=True),
    )
    op.add_column(
        "account_history",
        sa.Column("var_95", sa.Numeric(precision=18, scale=8), nullable=True),
    )

    # Enhance PerformanceMetrics table
    op.add_column(
        "performance_metrics",
        sa.Column("current_drawdown", sa.Numeric(precision=18, scale=8), server_default="0.0"),
    )
    op.add_column(
        "performance_metrics",
        sa.Column("var_95", sa.Numeric(precision=18, scale=8), nullable=True),
    )
    op.add_column(
        "performance_metrics",
        sa.Column(
            "avg_trade_duration_hours", sa.Numeric(precision=18, scale=8), server_default="0.0"
        ),
    )
    op.add_column(
        "performance_metrics",
        sa.Column("consecutive_wins_current", sa.Integer, server_default="0"),
    )
    op.add_column(
        "performance_metrics",
        sa.Column("consecutive_losses_current", sa.Integer, server_default="0"),
    )
    op.add_column(
        "performance_metrics",
        sa.Column("total_fees_paid", sa.Numeric(precision=18, scale=8), server_default="0.0"),
    )
    op.add_column(
        "performance_metrics",
        sa.Column("total_slippage_cost", sa.Numeric(precision=18, scale=8), server_default="0.0"),
    )


def downgrade() -> None:
    """Remove enhanced performance metric columns"""

    # Remove from AccountHistory
    op.drop_column("account_history", "var_95")
    op.drop_column("account_history", "calmar_ratio")
    op.drop_column("account_history", "sortino_ratio")
    op.drop_column("account_history", "sharpe_ratio")

    # Remove from PerformanceMetrics
    op.drop_column("performance_metrics", "total_slippage_cost")
    op.drop_column("performance_metrics", "total_fees_paid")
    op.drop_column("performance_metrics", "consecutive_losses_current")
    op.drop_column("performance_metrics", "consecutive_wins_current")
    op.drop_column("performance_metrics", "avg_trade_duration_hours")
    op.drop_column("performance_metrics", "var_95")
    op.drop_column("performance_metrics", "current_drawdown")
