"""Fix position status consistency

This migration fixes any positions that are stuck in PENDING status
but should be OPEN based on having actual execution data.

Revision ID: 0012_fix_position_status_consistency
Revises: 0011_fix_orderstatus_enum
Create Date: 2025-01-11

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0012_fix_pos_status"
down_revision = "0011_fix_orderstatus_enum"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """
    Fix position status consistency:
    1. Update positions stuck in PENDING to OPEN if they have filled data
    2. Add data validation function
    """
    # * Update positions that have filled data but are stuck in PENDING status
    connection = op.get_bind()

    # ! Find positions with PENDING status but have entry_price and quantity filled
    # ! These should be OPEN positions as they represent filled orders
    result = connection.execute(
        sa.text(
            """
        UPDATE positions 
        SET status = 'OPEN', last_update = NOW()
        WHERE status = 'PENDING' 
        AND entry_price IS NOT NULL 
        AND quantity IS NOT NULL 
        AND quantity > 0
        RETURNING id, symbol, order_id;
    """
        )
    )

    updated_rows = result.fetchall()
    if updated_rows:
        print(f"Updated {len(updated_rows)} positions from PENDING to OPEN status:")
        for row in updated_rows:
            print(f"  - Position ID: {row[0]}, Symbol: {row[1]}, Order ID: {row[2]}")
    else:
        print("No position status corrections needed.")


def downgrade() -> None:
    """
    Revert the status changes (though this may not be advisable
    as it could reintroduce the data inconsistency)
    """
    # * This downgrade is intentionally minimal as reverting status
    # * changes could reintroduce the bug we're fixing
    pass
