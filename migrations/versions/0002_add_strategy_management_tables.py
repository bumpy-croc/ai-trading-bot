"""add_strategy_management_tables

Revision ID: 0002_strategy_management
Revises: 0001_initial_schema
Create Date: 2025-10-02 00:00:00.000000

"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy import JSON, text
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision = '0002_strategy_management'
down_revision = '0001_initial_schema'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create strategy management tables"""
    
    # Create StrategyStatus enum
    enum_name = 'strategystatus'
    enum_values = "'EXPERIMENTAL', 'TESTING', 'PRODUCTION', 'RETIRED', 'DEPRECATED'"
    
    check_sql = text(f"""
        SELECT 1 FROM pg_type t
        JOIN pg_namespace n ON t.typnamespace = n.oid
        WHERE t.typtype = 'e'
        AND t.typname = '{enum_name}'
        AND n.nspname = 'public'
    """)
    result = op.get_bind().execute(check_sql).fetchone()
    
    if result is None:
        create_sql = f"CREATE TYPE {enum_name} AS ENUM ({enum_values})"
        op.execute(create_sql)
        print(f"Created enum: {enum_name}")
    else:
        print(f"Enum {enum_name} already exists, skipping creation")
    
    # Create strategy_registry table
    op.create_table(
        'strategy_registry',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('parent_id', sa.String(length=100), nullable=True),
        sa.Column('lineage_path', JSONB, nullable=True),
        sa.Column('branch_name', sa.String(length=100), nullable=True),
        sa.Column('merge_source', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('created_by', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('tags', JSONB, nullable=True),
        sa.Column('status', sa.Enum('EXPERIMENTAL', 'TESTING', 'PRODUCTION', 'RETIRED', 'DEPRECATED', 
                                    name='strategystatus', create_type=False), nullable=False),
        sa.Column('signal_generator_config', JSONB, nullable=False),
        sa.Column('risk_manager_config', JSONB, nullable=False),
        sa.Column('position_sizer_config', JSONB, nullable=False),
        sa.Column('regime_detector_config', JSONB, nullable=False),
        sa.Column('parameters', JSONB, nullable=True),
        sa.Column('performance_summary', JSONB, nullable=True),
        sa.Column('validation_results', JSONB, nullable=True),
        sa.Column('config_hash', sa.String(length=64), nullable=False),
        sa.Column('component_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at_db', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('strategy_id'),
        sa.UniqueConstraint('name', 'version', name='uq_strategy_name_version'),
        sa.ForeignKeyConstraint(['parent_id'], ['strategy_registry.strategy_id'], ),
    )
    
    # Create indexes for strategy_registry
    op.create_index('idx_strategy_name_version', 'strategy_registry', ['name', 'version'])
    op.create_index('idx_strategy_status_created', 'strategy_registry', ['status', 'created_at'])
    op.create_index('idx_strategy_parent_created', 'strategy_registry', ['parent_id', 'created_at'])
    op.create_index(op.f('ix_strategy_registry_strategy_id'), 'strategy_registry', ['strategy_id'], unique=False)
    op.create_index(op.f('ix_strategy_registry_name'), 'strategy_registry', ['name'], unique=False)
    op.create_index(op.f('ix_strategy_registry_created_at'), 'strategy_registry', ['created_at'], unique=False)
    op.create_index(op.f('ix_strategy_registry_parent_id'), 'strategy_registry', ['parent_id'], unique=False)
    op.create_index(op.f('ix_strategy_registry_config_hash'), 'strategy_registry', ['config_hash'], unique=False)
    op.create_index(op.f('ix_strategy_registry_component_hash'), 'strategy_registry', ['component_hash'], unique=False)
    
    # Create strategy_versions table
    op.create_table(
        'strategy_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('changes', JSONB, nullable=False),
        sa.Column('performance_delta', JSONB, nullable=True),
        sa.Column('is_major', sa.Boolean(), nullable=True),
        sa.Column('component_changes', JSONB, nullable=True),
        sa.Column('parameter_changes', JSONB, nullable=True),
        sa.Column('created_at_db', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('strategy_id', 'version', name='uq_strategy_version'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategy_registry.strategy_id'], ),
    )
    
    # Create indexes for strategy_versions
    op.create_index('idx_version_strategy_created', 'strategy_versions', ['strategy_id', 'created_at'])
    op.create_index(op.f('ix_strategy_versions_strategy_id'), 'strategy_versions', ['strategy_id'], unique=False)
    op.create_index(op.f('ix_strategy_versions_created_at'), 'strategy_versions', ['created_at'], unique=False)
    
    # Create strategy_performance table
    op.create_table(
        'strategy_performance',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('strategy_id', sa.String(length=100), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('period_type', sa.String(length=20), nullable=False),
        sa.Column('total_return', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('total_return_pct', sa.Numeric(precision=18, scale=8), nullable=False),
        sa.Column('sharpe_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('max_drawdown', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('win_rate', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('total_trades', sa.Integer(), nullable=True),
        sa.Column('winning_trades', sa.Integer(), nullable=True),
        sa.Column('losing_trades', sa.Integer(), nullable=True),
        sa.Column('avg_trade_duration_hours', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('volatility', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('sortino_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('calmar_ratio', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('var_95', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('signal_generator_contribution', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('risk_manager_contribution', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('position_sizer_contribution', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('regime_performance', JSONB, nullable=True),
        sa.Column('additional_metrics', JSONB, nullable=True),
        sa.Column('test_symbol', sa.String(length=20), nullable=True),
        sa.Column('test_timeframe', sa.String(length=10), nullable=True),
        sa.Column('test_parameters', JSONB, nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['strategy_id'], ['strategy_registry.strategy_id'], ),
    )
    
    # Create indexes for strategy_performance
    op.create_index('idx_perf_strategy_period', 'strategy_performance', ['strategy_id', 'period_start', 'period_end'])
    op.create_index('idx_perf_return_sharpe', 'strategy_performance', ['total_return_pct', 'sharpe_ratio'])
    op.create_index('idx_perf_period_type', 'strategy_performance', ['period_type', 'period_start'])
    op.create_index(op.f('ix_strategy_performance_strategy_id'), 'strategy_performance', ['strategy_id'], unique=False)
    op.create_index(op.f('ix_strategy_performance_period_start'), 'strategy_performance', ['period_start'], unique=False)
    op.create_index(op.f('ix_strategy_performance_period_end'), 'strategy_performance', ['period_end'], unique=False)
    
    # Create strategy_lineage table
    op.create_table(
        'strategy_lineage',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('ancestor_id', sa.String(length=100), nullable=False),
        sa.Column('descendant_id', sa.String(length=100), nullable=False),
        sa.Column('relationship_type', sa.String(length=20), nullable=False),
        sa.Column('generation_distance', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('evolution_reason', sa.String(length=200), nullable=True),
        sa.Column('change_impact', JSONB, nullable=True),
        sa.Column('performance_improvement', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('risk_change', sa.Numeric(precision=18, scale=8), nullable=True),
        sa.Column('created_at_db', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('ancestor_id', 'descendant_id', name='uq_lineage_pair'),
        sa.ForeignKeyConstraint(['ancestor_id'], ['strategy_registry.strategy_id'], ),
        sa.ForeignKeyConstraint(['descendant_id'], ['strategy_registry.strategy_id'], ),
    )
    
    # Create indexes for strategy_lineage
    op.create_index('idx_lineage_ancestor', 'strategy_lineage', ['ancestor_id', 'generation_distance'])
    op.create_index('idx_lineage_descendant', 'strategy_lineage', ['descendant_id', 'generation_distance'])
    op.create_index(op.f('ix_strategy_lineage_ancestor_id'), 'strategy_lineage', ['ancestor_id'], unique=False)
    op.create_index(op.f('ix_strategy_lineage_descendant_id'), 'strategy_lineage', ['descendant_id'], unique=False)


def downgrade() -> None:
    """Drop strategy management tables"""
    
    # Drop tables in reverse order due to foreign key constraints
    op.drop_index(op.f('ix_strategy_lineage_descendant_id'), table_name='strategy_lineage')
    op.drop_index(op.f('ix_strategy_lineage_ancestor_id'), table_name='strategy_lineage')
    op.drop_index('idx_lineage_descendant', table_name='strategy_lineage')
    op.drop_index('idx_lineage_ancestor', table_name='strategy_lineage')
    op.drop_table('strategy_lineage')
    
    op.drop_index(op.f('ix_strategy_performance_period_end'), table_name='strategy_performance')
    op.drop_index(op.f('ix_strategy_performance_period_start'), table_name='strategy_performance')
    op.drop_index(op.f('ix_strategy_performance_strategy_id'), table_name='strategy_performance')
    op.drop_index('idx_perf_period_type', table_name='strategy_performance')
    op.drop_index('idx_perf_return_sharpe', table_name='strategy_performance')
    op.drop_index('idx_perf_strategy_period', table_name='strategy_performance')
    op.drop_table('strategy_performance')
    
    op.drop_index(op.f('ix_strategy_versions_created_at'), table_name='strategy_versions')
    op.drop_index(op.f('ix_strategy_versions_strategy_id'), table_name='strategy_versions')
    op.drop_index('idx_version_strategy_created', table_name='strategy_versions')
    op.drop_table('strategy_versions')
    
    op.drop_index(op.f('ix_strategy_registry_component_hash'), table_name='strategy_registry')
    op.drop_index(op.f('ix_strategy_registry_config_hash'), table_name='strategy_registry')
    op.drop_index(op.f('ix_strategy_registry_parent_id'), table_name='strategy_registry')
    op.drop_index(op.f('ix_strategy_registry_created_at'), table_name='strategy_registry')
    op.drop_index(op.f('ix_strategy_registry_name'), table_name='strategy_registry')
    op.drop_index(op.f('ix_strategy_registry_strategy_id'), table_name='strategy_registry')
    op.drop_index('idx_strategy_parent_created', table_name='strategy_registry')
    op.drop_index('idx_strategy_status_created', table_name='strategy_registry')
    op.drop_index('idx_strategy_name_version', table_name='strategy_registry')
    op.drop_table('strategy_registry')
    
    # Drop the StrategyStatus enum
    op.execute('DROP TYPE IF EXISTS strategystatus')
