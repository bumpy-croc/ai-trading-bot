"""add prediction cache table

Revision ID: 0010_add_prediction_cache_table
Revises: 0009_correlation_tables
Create Date: 2024-01-01 00:00:00.000000

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = '0010_add_prediction_cache_table'
down_revision = '0009_correlation_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create prediction_cache table (idempotent)
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table('prediction_cache'):
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
    
    # Create indexes (idempotent)
    def _index_exists(name: str) -> bool:
        try:
            return any(i.get('name') == name for i in inspector.get_indexes('prediction_cache'))
        except Exception:
            return False

    if not _index_exists('idx_pred_cache_expires'):
        op.execute("CREATE INDEX IF NOT EXISTS idx_pred_cache_expires ON prediction_cache (expires_at)")
    if not _index_exists('idx_pred_cache_model_config'):
        op.execute("CREATE INDEX IF NOT EXISTS idx_pred_cache_model_config ON prediction_cache (model_name, config_hash)")
    if not _index_exists('idx_pred_cache_access'):
        op.execute("CREATE INDEX IF NOT EXISTS idx_pred_cache_access ON prediction_cache (last_accessed)")
    if not _index_exists('idx_pred_cache_model'):
        op.execute("CREATE INDEX IF NOT EXISTS idx_pred_cache_model ON prediction_cache (model_name)")
    if not _index_exists('idx_pred_cache_features'):
        op.execute("CREATE INDEX IF NOT EXISTS idx_pred_cache_features ON prediction_cache (features_hash)")


def downgrade() -> None:
    # Drop indexes (idempotent)
    op.execute('DROP INDEX IF EXISTS idx_pred_cache_features')
    op.execute('DROP INDEX IF EXISTS idx_pred_cache_model')
    op.execute('DROP INDEX IF EXISTS idx_pred_cache_access')
    op.execute('DROP INDEX IF EXISTS idx_pred_cache_model_config')
    op.execute('DROP INDEX IF EXISTS idx_pred_cache_expires')
    
    # Drop table if exists
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table('prediction_cache'):
        op.drop_table('prediction_cache')