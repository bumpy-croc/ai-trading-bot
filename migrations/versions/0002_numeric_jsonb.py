"""Convert Float money columns to Numeric(18,8) and JSON to JSONB

Revision ID: 0002_numeric_jsonb
Revises: 0001_initial
Create Date: 2025-07-06
"""

from alembic import op  # type: ignore
import sqlalchemy as sa  # type: ignore
from sqlalchemy.dialects import postgresql  # type: ignore

revision = '0002_numeric_jsonb'
down_revision = '0001_initial'
branch_labels = None
depends_on = None


_NUMERIC = sa.Numeric(18, 8)
_JSONB = postgresql.JSONB()


# Helper to alter multiple columns in a table ----------------------------

def _alter_numeric_columns(table_name: str, columns: list[str]):
    for col in columns:
        op.alter_column(
            table_name,
            col,
            type_=_NUMERIC,
            postgresql_using=f"{col}::numeric(18,8)",
            existing_nullable=True,
        )


def _alter_json_columns(table_name: str, columns: list[str]):
    for col in columns:
        op.alter_column(
            table_name,
            col,
            type_=_JSONB,
            postgresql_using=f"{col}::jsonb",
            existing_nullable=True,
        )


def upgrade() -> None:
    # Trades
    _alter_numeric_columns(
        "trades",
        [
            "entry_price",
            "exit_price",
            "size",
            "quantity",
            "pnl",
            "pnl_percent",
            "commission",
            "stop_loss",
            "take_profit",
            "confidence_score",
        ],
    )
    _alter_json_columns("trades", ["strategy_config"])

    # Positions
    _alter_numeric_columns(
        "positions",
        [
            "entry_price",
            "size",
            "quantity",
            "stop_loss",
            "take_profit",
            "current_price",
            "unrealized_pnl",
            "unrealized_pnl_percent",
            "confidence_score",
        ],
    )

    # Account history
    _alter_numeric_columns(
        "account_history",
        [
            "balance",
            "equity",
            "margin_used",
            "margin_available",
            "total_pnl",
            "daily_pnl",
            "drawdown",
            "total_exposure",
        ],
    )

    # Performance metrics
    _alter_numeric_columns(
        "performance_metrics",
        [
            "win_rate",
            "total_return",
            "total_return_percent",
            "max_drawdown",
            "sharpe_ratio",
            "sortino_ratio",
            "calmar_ratio",
            "avg_win",
            "avg_loss",
            "profit_factor",
            "expectancy",
            "best_trade_pnl",
            "worst_trade_pnl",
        ],
    )
    _alter_json_columns("performance_metrics", ["strategy_breakdown"])

    # Trading sessions
    _alter_numeric_columns(
        "trading_sessions",
        [
            "initial_balance",
            "final_balance",
            "total_pnl",
            "win_rate",
            "max_drawdown",
        ],
    )
    _alter_json_columns("trading_sessions", ["strategy_config"])

    # Strategy executions
    _alter_numeric_columns(
        "strategy_executions",
        [
            "signal_strength",
            "confidence_score",
            "position_size",
            "price",
            "volume",
            "volatility",
        ],
    )
    _alter_json_columns(
        "strategy_executions",
        ["indicators", "sentiment_data", "ml_predictions", "reasons"],
    )

    # System events
    _alter_json_columns("system_events", ["details"])



def downgrade() -> None:
    # Downgrade logic (revert to Float/JSON) would mirror the upgrade but is omitted for brevity.
    pass