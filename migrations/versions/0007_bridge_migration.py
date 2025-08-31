"""Bridge migration to fix migration chain

Revision ID: 0007
Revises: 0006_partial_operations
Create Date: 2024-01-01 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '0007'
down_revision = '0006_partial_operations'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # This is a bridge migration to fix the broken migration chain
    # No schema changes needed, just maintaining the revision chain
    pass


def downgrade() -> None:
    # This is a bridge migration to fix the broken migration chain
    # No schema changes needed, just maintaining the revision chain
    pass
