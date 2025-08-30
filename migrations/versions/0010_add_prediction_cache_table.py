"""add prediction cache table

Revision ID: 0010
Revises: 0009
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0010'
down_revision = '0009'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create prediction_cache table
    op.create_table('prediction_cache',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('cache_key', sa.String(length=255), nullable=False),
        sa.Column('model_name', sa.String(length=100), nullable=False),
        sa.Column('features_hash', sa.String(length=64), nullable=False),
        sa.Column('predicted_price', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('confidence', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('direction', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('access_count', sa.Integer(), nullable=True),
        sa.Column('last_accessed', sa.DateTime(), nullable=True),
        sa.Column('config_hash', sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('cache_key')
    )
    
    # Create indexes
    op.create_index('idx_pred_cache_expires', 'prediction_cache', ['expires_at'])
    op.create_index('idx_pred_cache_model_config', 'prediction_cache', ['model_name', 'config_hash'])
    op.create_index('idx_pred_cache_access', 'prediction_cache', ['last_accessed'])
    op.create_index('idx_pred_cache_model', 'prediction_cache', ['model_name'])
    op.create_index('idx_pred_cache_features', 'prediction_cache', ['features_hash'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_pred_cache_features', table_name='prediction_cache')
    op.drop_index('idx_pred_cache_model', table_name='prediction_cache')
    op.drop_index('idx_pred_cache_access', table_name='prediction_cache')
    op.drop_index('idx_pred_cache_model_config', table_name='prediction_cache')
    op.drop_index('idx_pred_cache_expires', table_name='prediction_cache')
    
    # Drop table
    op.drop_table('prediction_cache')