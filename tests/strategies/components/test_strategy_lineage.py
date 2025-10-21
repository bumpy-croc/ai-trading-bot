"""
Tests for Strategy Lineage and Evolutionary Tracking

This module tests the StrategyLineageTracker implementation including
parent-child relationships, branching/merging, evolution visualization,
and change impact analysis.
"""

from datetime import datetime
from unittest.mock import Mock

import pytest

from src.strategies.components.strategy_lineage import (
    ChangeRecord,
    ChangeType,
    EvolutionBranch,
    ImpactLevel,
    RelationshipType,
    StrategyLineageTracker,
)


class TestChangeRecord:
    """Test ChangeRecord data class"""
    
    def test_change_record_creation(self):
        """Test ChangeRecord creation"""
        timestamp = datetime.now()
        change = ChangeRecord(
            change_id="change_123",
            change_type=ChangeType.PARAMETER_CHANGE,
            description="Updated risk parameters",
            impact_level=ImpactLevel.MEDIUM,
            changed_components=["risk_manager"],
            parameter_changes={"risk_per_trade": {"old": 0.02, "new": 0.025}},
            performance_impact={"sharpe_ratio": 0.1, "return": 0.05},
            created_at=timestamp,
            created_by="user123"
        )
        
        assert change.change_id == "change_123"
        assert change.change_type == ChangeType.PARAMETER_CHANGE
        assert change.impact_level == ImpactLevel.MEDIUM
        assert "risk_manager" in change.changed_components
        assert change.performance_impact["sharpe_ratio"] == 0.1
    
    def test_change_record_serialization(self):
        """Test ChangeRecord serialization and deserialization"""
        timestamp = datetime.now()
        change = ChangeRecord(
            change_id="change_456",
            change_type=ChangeType.BUG_FIX,
            description="Fixed signal generation bug",
            impact_level=ImpactLevel.HIGH,
            changed_components=["signal_generator"],
            parameter_changes={},
            performance_impact=None,
            created_at=timestamp,
            created_by="developer"
        )
        
        # Test serialization
        change_dict = change.to_dict()
        assert change_dict['change_type'] == 'bug_fix'
        assert change_dict['impact_level'] == 'high'
        assert change_dict['created_at'] == timestamp.isoformat()
        
        # Test deserialization
        restored_change = ChangeRecord.from_dict(change_dict)
        assert restored_change.change_id == change.change_id
        assert restored_change.change_type == change.change_type
        assert restored_change.created_at == change.created_at


class TestEvolutionBranch:
    """Test EvolutionBranch data class"""
    
    def test_evolution_branch_creation(self):
        """Test EvolutionBranch creation"""
        timestamp = datetime.now()
        branch = EvolutionBranch(
            branch_id="branch_123",
            branch_name="experimental_features",
            parent_strategy_id="strategy_001",
            created_at=timestamp,
            created_by="researcher",
            description="Testing new ML features",
            active=True,
            strategies=["strategy_001", "strategy_002"]
        )
        
        assert branch.branch_id == "branch_123"
        assert branch.branch_name == "experimental_features"
        assert branch.active == True
        assert len(branch.strategies) == 2
    
    def test_evolution_branch_serialization(self):
        """Test EvolutionBranch serialization"""
        timestamp = datetime.now()
        branch = EvolutionBranch(
            branch_id="branch_456",
            branch_name="performance_optimization",
            parent_strategy_id="strategy_base",
            created_at=timestamp,
            created_by="optimizer",
            description="Performance improvements",
            active=False,
            strategies=["strategy_base"]
        )
        
        # Test serialization
        branch_dict = branch.to_dict()
        assert branch_dict['branch_name'] == "performance_optimization"
        assert branch_dict['created_at'] == timestamp.isoformat()
        assert branch_dict['active'] == False
        
        # Test deserialization
        restored_branch = EvolutionBranch.from_dict(branch_dict)
        assert restored_branch.branch_id == branch.branch_id
        assert restored_branch.created_at == branch.created_at
        assert restored_branch.active == branch.active


class TestStrategyLineageTracker:
    """Test StrategyLineageTracker functionality"""
    
    @pytest.fixture
    def tracker(self):
        """Create a test lineage tracker"""
        return StrategyLineageTracker()
    
    def test_tracker_initialization(self, tracker):
        """Test tracker initialization"""
        assert isinstance(tracker, StrategyLineageTracker)
        assert len(tracker.strategies) == 0
        assert len(tracker.changes) == 0
        assert len(tracker.branches) == 0
        assert len(tracker.merges) == 0
    
    def test_register_strategy_without_parent(self, tracker):
        """Test registering a strategy without parent"""
        metadata = {
            'name': 'Base Strategy',
            'description': 'Initial strategy',
            'created_by': 'user1'
        }
        
        tracker.register_strategy("strategy_001", metadata=metadata)
        
        assert "strategy_001" in tracker.strategies
        assert tracker.strategies["strategy_001"]['parent_id'] is None
        assert tracker.strategies["strategy_001"]['generation'] == 0
        assert tracker.strategies["strategy_001"]['name'] == 'Base Strategy'
        
        # Should be in strategies dict (nodes are implicit in our simple graph)
        assert "strategy_001" in tracker.strategies
    
    def test_register_strategy_with_parent(self, tracker):
        """Test registering a strategy with parent"""
        # Register parent first
        tracker.register_strategy("parent_001", metadata={'name': 'Parent'})
        
        # Register child
        tracker.register_strategy("child_001", parent_id="parent_001", 
                                metadata={'name': 'Child'})
        
        assert tracker.strategies["child_001"]['parent_id'] == "parent_001"
        assert tracker.strategies["child_001"]['generation'] == 1
        
        # Should have edge in graph
        assert "child_001" in tracker.lineage_graph["parent_001"]
        
        # Check edge attributes
        edge_key = "parent_001->child_001"
        assert edge_key in tracker.graph_edges
        edge_data = tracker.graph_edges[edge_key]
        assert edge_data['relationship_type'] == RelationshipType.PARENT
    
    def test_register_strategy_lineage_chain(self, tracker):
        """Test registering a chain of strategies"""
        # Create a lineage chain: grandparent -> parent -> child
        tracker.register_strategy("grandparent", metadata={'name': 'Grandparent'})
        tracker.register_strategy("parent", parent_id="grandparent", 
                                metadata={'name': 'Parent'})
        tracker.register_strategy("child", parent_id="parent", 
                                metadata={'name': 'Child'})
        
        assert tracker.strategies["grandparent"]['generation'] == 0
        assert tracker.strategies["parent"]['generation'] == 1
        assert tracker.strategies["child"]['generation'] == 2
        
        # Check graph structure
        assert "parent" in tracker.lineage_graph["grandparent"]
        assert "child" in tracker.lineage_graph["parent"]
    
    def test_record_change(self, tracker):
        """Test recording changes to strategies"""
        # Register a strategy first
        tracker.register_strategy("strategy_001", metadata={'name': 'Test Strategy'})
        
        # Record a change
        change_id = tracker.record_change(
            strategy_id="strategy_001",
            change_type=ChangeType.PARAMETER_CHANGE,
            description="Updated risk parameters",
            impact_level=ImpactLevel.MEDIUM,
            changed_components=["risk_manager"],
            parameter_changes={"risk_per_trade": {"old": 0.02, "new": 0.025}},
            performance_impact={"sharpe_ratio": 0.1},
            created_by="user1"
        )
        
        assert change_id is not None
        assert "strategy_001" in tracker.changes
        assert len(tracker.changes["strategy_001"]) == 1
        
        change = tracker.changes["strategy_001"][0]
        assert change.change_type == ChangeType.PARAMETER_CHANGE
        assert change.description == "Updated risk parameters"
        assert change.impact_level == ImpactLevel.MEDIUM
        
        # Strategy metadata should be updated
        assert tracker.strategies["strategy_001"]['change_count'] == 1
        assert 'last_modified' in tracker.strategies["strategy_001"]
    
    def test_record_change_nonexistent_strategy(self, tracker):
        """Test recording change for non-existent strategy"""
        with pytest.raises(ValueError, match="Strategy nonexistent not found"):
            tracker.record_change(
                strategy_id="nonexistent",
                change_type=ChangeType.BUG_FIX,
                description="Fix bug"
            )
    
    def test_create_branch(self, tracker):
        """Test creating evolution branches"""
        # Register parent strategy
        tracker.register_strategy("parent_001", metadata={'name': 'Parent'})
        
        # Create branch
        branch_id = tracker.create_branch(
            parent_strategy_id="parent_001",
            branch_name="experimental",
            description="Experimental features branch",
            created_by="researcher"
        )
        
        assert branch_id is not None
        assert branch_id in tracker.branches
        
        branch = tracker.branches[branch_id]
        assert branch.branch_name == "experimental"
        assert branch.parent_strategy_id == "parent_001"
        assert branch.active == True
        assert "parent_001" in branch.strategies
        
        # Parent strategy should be updated with branch ID
        assert tracker.strategies["parent_001"]['branch_id'] == branch_id
    
    def test_create_branch_nonexistent_parent(self, tracker):
        """Test creating branch with non-existent parent"""
        with pytest.raises(ValueError, match="Parent strategy nonexistent not found"):
            tracker.create_branch(
                parent_strategy_id="nonexistent",
                branch_name="test_branch"
            )
    
    def test_add_strategy_to_branch(self, tracker):
        """Test adding strategy to existing branch"""
        # Setup: register strategies and create branch
        tracker.register_strategy("parent_001", metadata={'name': 'Parent'})
        tracker.register_strategy("child_001", parent_id="parent_001", 
                                metadata={'name': 'Child'})
        
        branch_id = tracker.create_branch("parent_001", "test_branch")
        
        # Add child to branch
        tracker.add_strategy_to_branch("child_001", branch_id)
        
        # Check branch contains both strategies
        branch = tracker.branches[branch_id]
        assert "parent_001" in branch.strategies
        assert "child_001" in branch.strategies
        
        # Child strategy should have branch ID
        assert tracker.strategies["child_001"]['branch_id'] == branch_id
        
        # Should have branch relationship in graph
        edge_key = "parent_001->child_001"
        assert edge_key in tracker.graph_edges
        edge_data = tracker.graph_edges[edge_key]
        assert edge_data['relationship_type'] == RelationshipType.BRANCH
        assert edge_data['branch_id'] == branch_id
    
    def test_merge_strategies(self, tracker):
        """Test merging multiple strategies"""
        # Setup: register multiple strategies
        tracker.register_strategy("base", metadata={'name': 'Base'})
        tracker.register_strategy("variant1", parent_id="base", metadata={'name': 'Variant1'})
        tracker.register_strategy("variant2", parent_id="base", metadata={'name': 'Variant2'})
        
        # Merge strategies
        merge_id = tracker.merge_strategies(
            target_strategy_id="variant1",
            source_strategy_ids=["variant2"],
            merge_strategy="best_performance",
            created_by="merger"
        )
        
        assert merge_id is not None
        assert merge_id in tracker.merges
        
        merge_record = tracker.merges[merge_id]
        assert merge_record.target_strategy_id == "variant1"
        assert "variant2" in merge_record.source_strategy_ids
        assert merge_record.merge_strategy == "best_performance"
        
        # Should create a new merged strategy
        merged_strategy_id = merge_record.merged_strategy_id
        assert merged_strategy_id in tracker.strategies
        
        # Should have merge relationships in graph
        assert merged_strategy_id in tracker.lineage_graph["variant1"]
        assert merged_strategy_id in tracker.lineage_graph["variant2"]
    
    def test_get_lineage_simple(self, tracker):
        """Test getting lineage for simple parent-child relationship"""
        # Create simple lineage: parent -> child
        tracker.register_strategy("parent", metadata={'name': 'Parent'})
        tracker.register_strategy("child", parent_id="parent", metadata={'name': 'Child'})
        
        # Get child's lineage
        lineage = tracker.get_lineage("child")
        
        assert lineage['strategy_id'] == "child"
        assert lineage['generation'] == 1
        assert len(lineage['ancestors']) == 1
        assert lineage['ancestors'][0]['id'] == "parent"
        assert len(lineage['descendants']) == 0
        assert lineage['lineage_depth'] == 2
        
        # Get parent's lineage
        parent_lineage = tracker.get_lineage("parent")
        assert len(parent_lineage['ancestors']) == 0
        assert len(parent_lineage['descendants']) == 1
        assert parent_lineage['descendants'][0]['id'] == "child"
    
    def test_get_lineage_complex(self, tracker):
        """Test getting lineage for complex family tree"""
        # Create complex lineage
        tracker.register_strategy("root", metadata={'name': 'Root'})
        tracker.register_strategy("parent1", parent_id="root", metadata={'name': 'Parent1'})
        tracker.register_strategy("parent2", parent_id="root", metadata={'name': 'Parent2'})
        tracker.register_strategy("child1", parent_id="parent1", metadata={'name': 'Child1'})
        tracker.register_strategy("child2", parent_id="parent1", metadata={'name': 'Child2'})
        tracker.register_strategy("grandchild", parent_id="child1", metadata={'name': 'Grandchild'})
        
        # Get grandchild's lineage
        lineage = tracker.get_lineage("grandchild")
        
        assert lineage['generation'] == 3
        assert len(lineage['ancestors']) == 3  # root, parent1, child1
        assert lineage['ancestors'][0]['id'] == "root"  # Root first
        assert lineage['ancestors'][1]['id'] == "parent1"
        assert lineage['ancestors'][2]['id'] == "child1"
        
        # Get parent1's lineage (should have siblings and descendants)
        parent1_lineage = tracker.get_lineage("parent1")
        assert len(parent1_lineage['siblings']) == 1  # parent2
        assert parent1_lineage['siblings'][0]['id'] == "parent2"
        assert len(parent1_lineage['descendants']) == 3  # child1, child2, grandchild
    
    def test_get_lineage_with_changes(self, tracker):
        """Test getting lineage with change history"""
        tracker.register_strategy("strategy_001", metadata={'name': 'Test'})
        
        # Add some changes
        tracker.record_change("strategy_001", ChangeType.PARAMETER_CHANGE, "Change 1")
        tracker.record_change("strategy_001", ChangeType.BUG_FIX, "Change 2")
        
        lineage = tracker.get_lineage("strategy_001")
        
        assert len(lineage['changes']) == 2
        assert lineage['changes'][0]['change_type'] == 'parameter_change'
        assert lineage['changes'][1]['change_type'] == 'bug_fix'
    
    def test_get_evolution_path(self, tracker):
        """Test getting evolution path between strategies"""
        # Create lineage chain
        tracker.register_strategy("v1", metadata={'name': 'V1'})
        tracker.register_strategy("v2", parent_id="v1", metadata={'name': 'V2'})
        tracker.register_strategy("v3", parent_id="v2", metadata={'name': 'V3'})
        
        # Add changes to track evolution
        tracker.record_change("v2", ChangeType.PARAMETER_CHANGE, "V1 to V2 changes")
        tracker.record_change("v3", ChangeType.FEATURE_ADDITION, "V2 to V3 changes")
        
        # Get evolution path
        path = tracker.get_evolution_path("v1", "v3")
        
        assert len(path) == 3
        assert path[0]['strategy_id'] == "v1"
        assert path[1]['strategy_id'] == "v2"
        assert path[2]['strategy_id'] == "v3"
        
        # Check step numbers
        assert path[0]['step'] == 0
        assert path[1]['step'] == 1
        assert path[2]['step'] == 2
        
        # Check relationships
        assert path[1]['relationship'] == 'parent'
        assert path[2]['relationship'] == 'parent'
        
        # Check changes are included
        assert len(path[1]['changes']) == 1
        assert len(path[2]['changes']) == 1
    
    def test_get_evolution_path_no_connection(self, tracker):
        """Test getting evolution path for unconnected strategies"""
        tracker.register_strategy("strategy1", metadata={'name': 'Strategy1'})
        tracker.register_strategy("strategy2", metadata={'name': 'Strategy2'})
        
        # Should return empty path (no connection)
        path = tracker.get_evolution_path("strategy1", "strategy2")
        assert len(path) == 0
    
    def test_analyze_change_impact(self, tracker):
        """Test analyzing impact of specific changes"""
        # Create lineage with changes
        tracker.register_strategy("parent", metadata={'name': 'Parent'})
        tracker.register_strategy("child1", parent_id="parent", metadata={'name': 'Child1'})
        tracker.register_strategy("child2", parent_id="parent", metadata={'name': 'Child2'})
        
        # Record change in parent
        change_id = tracker.record_change(
            "parent",
            ChangeType.COMPONENT_CHANGE,
            "Updated signal generator",
            changed_components=["signal_generator"],
            performance_impact={"sharpe_ratio": 0.2}
        )
        
        # Record related changes in children (after parent change)
        tracker.record_change(
            "child1",
            ChangeType.PARAMETER_CHANGE,
            "Adjusted parameters after parent change",
            changed_components=["signal_generator"]
        )
        
        # Analyze impact
        impact = tracker.analyze_change_impact("parent", change_id)
        
        assert impact['change_id'] == change_id
        assert impact['strategy_id'] == "parent"
        assert impact['change_type'] == 'component_change'
        assert impact['changed_components'] == ["signal_generator"]
        assert impact['performance_impact'] == {"sharpe_ratio": 0.2}
        
        # Should detect impact on descendants
        assert len(impact['descendant_impact']) >= 1
        assert impact['affected_strategies'] >= 1
    
    def test_get_evolution_metrics(self, tracker):
        """Test getting comprehensive evolution metrics"""
        # Create a complex strategy evolution
        tracker.register_strategy("root", metadata={'name': 'Root'})
        tracker.register_strategy("v1", parent_id="root", metadata={'name': 'V1'})
        tracker.register_strategy("v2", parent_id="v1", metadata={'name': 'V2'})
        
        # Create branches
        branch_id = tracker.create_branch("v1", "experimental")
        tracker.register_strategy("exp1", parent_id="v1", metadata={'name': 'Exp1'})
        tracker.add_strategy_to_branch("exp1", branch_id)
        
        # Add changes
        tracker.record_change("v1", ChangeType.PARAMETER_CHANGE, "Change 1", 
                            performance_impact={"return": 0.1})
        tracker.record_change("v2", ChangeType.BUG_FIX, "Change 2")
        tracker.record_change("exp1", ChangeType.FEATURE_ADDITION, "Change 3",
                            performance_impact={"return": 0.05})
        
        # Merge strategies
        tracker.merge_strategies("v2", ["exp1"])
        
        metrics = tracker.get_evolution_metrics()
        
        assert metrics.total_strategies >= 4
        assert metrics.total_branches >= 1
        assert metrics.total_merges >= 1
        assert metrics.max_generation_distance >= 2
        assert len(metrics.most_evolved_lineage) >= 3
        assert metrics.performance_improvement_rate > 0
        assert ChangeType.PARAMETER_CHANGE in metrics.change_frequency
    
    def test_visualize_lineage_dict_format(self, tracker):
        """Test lineage visualization in dict format"""
        # Create simple lineage
        tracker.register_strategy("parent", metadata={'name': 'Parent'})
        tracker.register_strategy("child", parent_id="parent", metadata={'name': 'Child'})
        
        viz_data = tracker.visualize_lineage("child", format="dict")
        
        assert 'nodes' in viz_data
        assert 'edges' in viz_data
        assert 'metadata' in viz_data
        
        # Check nodes
        nodes = viz_data['nodes']
        assert len(nodes) == 2
        node_ids = [n['id'] for n in nodes]
        assert "parent" in node_ids
        assert "child" in node_ids
        
        # Check edges
        edges = viz_data['edges']
        assert len(edges) == 1
        assert edges[0]['source'] == "parent"
        assert edges[0]['target'] == "child"
    
    def test_visualize_lineage_mermaid_format(self, tracker):
        """Test lineage visualization in Mermaid format"""
        tracker.register_strategy("parent", metadata={'name': 'Parent'})
        tracker.register_strategy("child", parent_id="parent", metadata={'name': 'Child'})
        
        mermaid = tracker.visualize_lineage("child", format="mermaid")
        
        assert isinstance(mermaid, str)
        assert "graph TD" in mermaid
        assert "parent" in mermaid
        assert "child" in mermaid
        assert "-->" in mermaid
    
    def test_visualize_lineage_dot_format(self, tracker):
        """Test lineage visualization in DOT format"""
        tracker.register_strategy("parent", metadata={'name': 'Parent'})
        tracker.register_strategy("child", parent_id="parent", metadata={'name': 'Child'})
        
        dot = tracker.visualize_lineage("child", format="dot")
        
        assert isinstance(dot, str)
        assert "digraph StrategyLineage" in dot
        assert "parent" in dot
        assert "child" in dot
        assert "->" in dot
    
    def test_visualize_lineage_invalid_format(self, tracker):
        """Test lineage visualization with invalid format"""
        tracker.register_strategy("strategy", metadata={'name': 'Strategy'})
        
        with pytest.raises(ValueError, match="Unsupported format"):
            tracker.visualize_lineage("strategy", format="invalid")
    
    def test_caching_behavior(self, tracker):
        """Test caching behavior for performance"""
        tracker.register_strategy("strategy1", metadata={'name': 'Strategy1'})
        
        # First call should populate cache
        lineage1 = tracker.get_lineage("strategy1")
        
        # Second call should use cache (same object)
        lineage2 = tracker.get_lineage("strategy1")
        
        # Should be the same data
        assert lineage1['strategy_id'] == lineage2['strategy_id']
        
        # Adding a change should invalidate cache
        tracker.record_change("strategy1", ChangeType.BUG_FIX, "Fix")
        
        # Next call should recalculate
        lineage3 = tracker.get_lineage("strategy1")
        assert len(lineage3['changes']) == 1
    
    def test_storage_backend_integration(self):
        """Test integration with storage backend"""
        mock_storage = Mock()
        tracker = StrategyLineageTracker(storage_backend=mock_storage)
        
        # Register strategy
        tracker.register_strategy("strategy1", metadata={'name': 'Test'})
        
        # Verify storage backend was called
        mock_storage.save_strategy_lineage.assert_called_once()
        
        # Record change
        tracker.record_change("strategy1", ChangeType.BUG_FIX, "Fix")
        
        # Verify change was saved
        mock_storage.save_change_record.assert_called_once()
        
        # Test storage failure handling
        mock_storage.save_strategy_lineage.side_effect = Exception("Storage error")
        
        # Should not raise exception, just log error
        tracker.register_strategy("strategy2", metadata={'name': 'Test2'})
        assert "strategy2" in tracker.strategies  # Should still be registered locally
    
    def test_complex_branching_and_merging(self, tracker):
        """Test complex branching and merging scenario"""
        # Create main line
        tracker.register_strategy("main_v1", metadata={'name': 'Main V1'})
        tracker.register_strategy("main_v2", parent_id="main_v1", metadata={'name': 'Main V2'})
        
        # Create experimental branch
        exp_branch = tracker.create_branch("main_v1", "experimental", "Experimental features")
        tracker.register_strategy("exp_v1", parent_id="main_v1", metadata={'name': 'Exp V1'})
        tracker.add_strategy_to_branch("exp_v1", exp_branch)
        
        tracker.register_strategy("exp_v2", parent_id="exp_v1", metadata={'name': 'Exp V2'})
        tracker.add_strategy_to_branch("exp_v2", exp_branch)
        
        # Create performance branch
        perf_branch = tracker.create_branch("main_v2", "performance", "Performance optimizations")
        tracker.register_strategy("perf_v1", parent_id="main_v2", metadata={'name': 'Perf V1'})
        tracker.add_strategy_to_branch("perf_v1", perf_branch)
        
        # Merge experimental into main
        merge_id = tracker.merge_strategies("main_v2", ["exp_v2"], "best_performance")
        
        # Verify complex structure
        assert len(tracker.strategies) >= 6  # 5 strategies + merged strategy
        assert len(tracker.branches) == 2
        assert len(tracker.merges) == 1
        
        # Check evolution metrics
        metrics = tracker.get_evolution_metrics()
        assert metrics.total_strategies >= 6
        assert metrics.total_branches == 2
        assert metrics.total_merges == 1
        
        # Verify lineage paths
        main_v2_lineage = tracker.get_lineage("main_v2")
        assert len(main_v2_lineage['ancestors']) == 1
        
        exp_v2_lineage = tracker.get_lineage("exp_v2")
        assert len(exp_v2_lineage['ancestors']) == 2  # main_v1, exp_v1
        assert exp_v2_lineage['branch']['name'] == "experimental"


if __name__ == "__main__":
    pytest.main([__file__])