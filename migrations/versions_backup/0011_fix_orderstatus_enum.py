"""Fix and normalize orderstatus enum to uppercase labels

Revision ID: 0011_fix_orderstatus_enum
Revises: 0010_add_prediction_cache_table
Create Date: 2025-09-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0011_fix_orderstatus_enum"
down_revision = "0010_add_prediction_cache_table"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()

    # Target enum and table/column
    enum_name = "orderstatus"
    table_name = "positions"
    column_name = "status"

    # Desired canonical labels
    desired_labels = ["PENDING", "OPEN", "FILLED", "CANCELLED", "FAILED"]

    # Detect backend dialect
    is_postgres = bind.dialect.name == "postgresql"

    if not is_postgres:
        # SQLite or others: SQLAlchemy handles enums as CHECK constraints/strings; nothing to do
        return

    # Helper: get current enum labels if enum exists
    def _get_current_labels() -> list[str] | None:
        try:
            result = bind.execute(
                sa.text(
                    """
                    SELECT e.enumlabel
                    FROM pg_type t
                    JOIN pg_enum e ON t.oid = e.enumtypid
                    JOIN pg_namespace n ON n.oid = t.typnamespace
                    WHERE t.typname = :name
                    ORDER BY e.enumsortorder
                    """
                ),
                {"name": enum_name},
            )
            rows = result.fetchall()
            return [r[0] for r in rows]
        except Exception:
            return None

    current_labels = _get_current_labels()

    if current_labels is None:
        # Enum type does not exist yet; create it and alter column
        # Construct explicit SQL for ENUM creation
        labels_sql = ", ".join([f"'{lbl}'" for lbl in desired_labels])
        op.execute(f"CREATE TYPE {enum_name} AS ENUM ({labels_sql})")
        op.execute(
            sa.text(
                f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {enum_name} USING {column_name}::text::{enum_name}"
            )
        )
        return

    # If enum exists, ensure all desired labels are present (add missing)
    for lbl in desired_labels:
        if lbl not in current_labels:
            op.execute(f"ALTER TYPE {enum_name} ADD VALUE IF NOT EXISTS '{lbl}'")

    # Attempt to coerce any lowercase/legacy values in the table to uppercase equivalents
    # This is safe because for any non-canonical value, cast to text and UPPER, then case expression
    try:
        op.execute(
            sa.text(
                f"""
                UPDATE {table_name}
                SET {column_name} = CAST(UPPER(CAST({column_name} AS text)) AS {enum_name})
                WHERE CAST({column_name} AS text) != UPPER(CAST({column_name} AS text))
                """
            )
        )
    except Exception:
        # Ignore if no such rows or if column already uses canonical labels
        pass

    # Finally, force column type to the named enum to avoid anonymous enums
    op.execute(
        sa.text(
            f"ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {enum_name} USING {column_name}::text::{enum_name}"
        )
    )


def downgrade() -> None:
    # Non-destructive downgrade: keep the normalized enum as data may depend on it
    # Optionally, we could recreate previous enum state, but we choose no-op for safety.
    pass
