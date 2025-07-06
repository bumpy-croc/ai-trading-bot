"""Initial schema

Revision ID: 0001_initial
Revises: 
Create Date: 2025-07-06
"""

from alembic import op  # type: ignore
import sqlalchemy as sa  # type: ignore
import src.database.models as models

# revision identifiers, used by Alembic.
revision = '0001_initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    """Create initial database schema based on SQLAlchemy models."""
    bind = op.get_bind()
    models.Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    """Drop all tables created in the initial schema."""
    bind = op.get_bind()
    models.Base.metadata.drop_all(bind=bind)