"""
Strategy Lineage and Evolutionary Tracking

This module implements comprehensive lineage tracking for strategies,
including parent-child relationships, branching/merging capabilities,
evolution visualization, and change impact analysis.
"""

import logging
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

# Removed networkx dependency - using simple graph implementation


class RelationshipType(Enum):
    """Types of strategy relationships"""

    PARENT = "parent"
    CHILD = "child"
    BRANCH = "branch"
    MERGE = "merge"
    FORK = "fork"
    VARIANT = "variant"


class ChangeType(Enum):
    """Types of changes in strategy evolution"""

    PARAMETER_CHANGE = "parameter_change"
    COMPONENT_CHANGE = "component_change"
    ARCHITECTURE_CHANGE = "architecture_change"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    BUG_FIX = "bug_fix"
    FEATURE_ADDITION = "feature_addition"
    REFACTORING = "refactoring"


class ImpactLevel(Enum):
    """Impact levels for changes"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ChangeRecord:
    """Record of a specific change made to a strategy"""

    change_id: str
    change_type: ChangeType
    description: str
    impact_level: ImpactLevel
    changed_components: list[str]
    parameter_changes: dict[str, Any]
    performance_impact: dict[str, float] | None
    created_at: datetime
    created_by: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["change_type"] = self.change_type.value
        data["impact_level"] = self.impact_level.value
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ChangeRecord":
        """Create from dictionary"""
        data = data.copy()
        data["change_type"] = ChangeType(data["change_type"])
        data["impact_level"] = ImpactLevel(data["impact_level"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class EvolutionBranch:
    """Represents a branch in strategy evolution"""

    branch_id: str
    branch_name: str
    parent_strategy_id: str
    created_at: datetime
    created_by: str
    description: str
    active: bool
    strategies: list[str]  # Strategy IDs in this branch

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvolutionBranch":
        """Create from dictionary"""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class MergeRecord:
    """Record of a strategy merge operation"""

    merge_id: str
    target_strategy_id: str
    source_strategy_ids: list[str]
    merged_strategy_id: str
    merge_strategy: str  # 'best_performance', 'weighted_average', 'manual'
    conflict_resolution: dict[str, Any]
    created_at: datetime
    created_by: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MergeRecord":
        """Create from dictionary"""
        data = data.copy()
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class EvolutionMetrics:
    """Metrics for strategy evolution analysis"""

    total_strategies: int
    total_branches: int
    total_merges: int
    avg_generation_distance: float
    max_generation_distance: int
    most_evolved_lineage: list[str]
    performance_improvement_rate: float
    change_frequency: dict[ChangeType, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data["change_frequency"] = {k.value: v for k, v in self.change_frequency.items()}
        return data


class StrategyLineageTracker:
    """
    Strategy lineage and evolutionary tracking system

    This class manages strategy relationships, tracks evolution paths,
    handles branching and merging, and provides comprehensive analysis
    of strategy development over time.
    """

    def __init__(self, storage_backend: Any | None = None):
        """
        Initialize lineage tracker

        Args:
            storage_backend: Optional storage backend for persistence
        """
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(__name__)

        # Core data structures - simple graph implementation
        self.lineage_graph = defaultdict(list)  # Simple adjacency list for lineage relationships
        self.graph_edges = {}  # Store edge metadata
        self.strategies: dict[str, dict[str, Any]] = {}  # Strategy metadata
        self.changes: dict[str, list[ChangeRecord]] = defaultdict(list)  # Changes per strategy
        self.branches: dict[str, EvolutionBranch] = {}  # Evolution branches
        self.merges: dict[str, MergeRecord] = {}  # Merge records

        # Caching for performance
        self._lineage_cache: dict[str, dict[str, Any]] = {}
        self._evolution_cache: dict[str, Any] = {}
        self._cache_dirty = True

        self.logger.info("StrategyLineageTracker initialized")

    def register_strategy(
        self,
        strategy_id: str,
        parent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Register a strategy in the lineage system

        Args:
            strategy_id: Unique strategy identifier
            parent_id: Optional parent strategy ID
            metadata: Optional strategy metadata
        """
        # Add strategy to graph (nodes are implicitly created when referenced)

        # Store metadata
        self.strategies[strategy_id] = {
            "id": strategy_id,
            "parent_id": parent_id,
            "created_at": datetime.now(UTC),
            "generation": 0,
            "branch_id": None,
            **(metadata or {}),
        }

        # Add parent relationship if specified
        if parent_id and parent_id in self.strategies:
            self.lineage_graph[parent_id].append(strategy_id)
            edge_key = f"{parent_id}->{strategy_id}"
            self.graph_edges[edge_key] = {
                "relationship_type": RelationshipType.PARENT,
                "source": parent_id,
                "target": strategy_id,
            }

            # Update generation
            parent_generation = self.strategies[parent_id]["generation"]
            self.strategies[strategy_id]["generation"] = parent_generation + 1

        # Clear caches
        self._invalidate_caches()

        # Persist if backend available
        if self.storage_backend:
            try:
                self.storage_backend.save_strategy_lineage(
                    strategy_id, self.strategies[strategy_id]
                )
            except Exception:
                self.logger.exception("Failed to persist strategy lineage")

        self.logger.info(f"Registered strategy {strategy_id} with parent {parent_id}")

    def record_change(
        self,
        strategy_id: str,
        change_type: ChangeType,
        description: str,
        impact_level: ImpactLevel = ImpactLevel.MEDIUM,
        changed_components: list[str] | None = None,
        parameter_changes: dict[str, Any] | None = None,
        performance_impact: dict[str, float] | None = None,
        created_by: str = "system",
    ) -> str:
        """
        Record a change made to a strategy

        Args:
            strategy_id: Strategy that was changed
            change_type: Type of change made
            description: Description of the change
            impact_level: Impact level of the change
            changed_components: List of components that were changed
            parameter_changes: Dictionary of parameter changes
            performance_impact: Performance impact metrics
            created_by: Who made the change

        Returns:
            Change record ID
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        change_id = f"change_{uuid4().hex[:8]}"

        change_record = ChangeRecord(
            change_id=change_id,
            change_type=change_type,
            description=description,
            impact_level=impact_level,
            changed_components=changed_components or [],
            parameter_changes=parameter_changes or {},
            performance_impact=performance_impact,
            created_at=datetime.now(UTC),
            created_by=created_by,
        )

        self.changes[strategy_id].append(change_record)

        # Update strategy metadata
        self.strategies[strategy_id]["last_modified"] = datetime.now(UTC)
        self.strategies[strategy_id]["change_count"] = len(self.changes[strategy_id])

        # Clear caches
        self._invalidate_caches()

        # Persist if backend available
        if self.storage_backend:
            try:
                self.storage_backend.save_change_record(change_record)
            except Exception:
                self.logger.exception("Failed to persist change record")

        self.logger.info(f"Recorded {change_type.value} change for strategy {strategy_id}")
        return change_id

    def create_branch(
        self,
        parent_strategy_id: str,
        branch_name: str,
        description: str = "",
        created_by: str = "system",
    ) -> str:
        """
        Create a new evolution branch

        Args:
            parent_strategy_id: Strategy to branch from
            branch_name: Name for the new branch
            description: Branch description
            created_by: Who created the branch

        Returns:
            Branch ID
        """
        if parent_strategy_id not in self.strategies:
            raise ValueError(f"Parent strategy {parent_strategy_id} not found")

        branch_id = f"branch_{uuid4().hex[:8]}"

        branch = EvolutionBranch(
            branch_id=branch_id,
            branch_name=branch_name,
            parent_strategy_id=parent_strategy_id,
            created_at=datetime.now(UTC),
            created_by=created_by,
            description=description,
            active=True,
            strategies=[parent_strategy_id],
        )

        self.branches[branch_id] = branch

        # Update parent strategy's branch
        self.strategies[parent_strategy_id]["branch_id"] = branch_id

        # Clear caches
        self._invalidate_caches()

        self.logger.info(
            f"Created branch {branch_name} ({branch_id}) from strategy {parent_strategy_id}"
        )
        return branch_id

    def add_strategy_to_branch(self, strategy_id: str, branch_id: str) -> None:
        """
        Add a strategy to an existing branch

        Args:
            strategy_id: Strategy to add
            branch_id: Target branch
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        if branch_id not in self.branches:
            raise ValueError(f"Branch {branch_id} not found")

        # Add to branch
        if strategy_id not in self.branches[branch_id].strategies:
            self.branches[branch_id].strategies.append(strategy_id)

        # Update strategy metadata
        self.strategies[strategy_id]["branch_id"] = branch_id

        # Add branch relationship to graph
        parent_id = self.strategies[strategy_id]["parent_id"]
        if parent_id:
            edge_key = f"{parent_id}->{strategy_id}"
            if edge_key in self.graph_edges:
                self.graph_edges[edge_key]["relationship_type"] = RelationshipType.BRANCH
                self.graph_edges[edge_key]["branch_id"] = branch_id

        self._invalidate_caches()
        self.logger.info(f"Added strategy {strategy_id} to branch {branch_id}")

    def merge_strategies(
        self,
        target_strategy_id: str,
        source_strategy_ids: list[str],
        merge_strategy: str = "best_performance",
        conflict_resolution: dict[str, Any] | None = None,
        created_by: str = "system",
    ) -> str:
        """
        Merge multiple strategies into a new strategy

        Args:
            target_strategy_id: Target strategy for merge
            source_strategy_ids: Source strategies to merge
            merge_strategy: Strategy for merging ('best_performance', 'weighted_average', 'manual')
            conflict_resolution: Manual conflict resolution rules
            created_by: Who performed the merge

        Returns:
            Merge record ID
        """
        # Validate strategies exist
        all_strategies = [target_strategy_id, *source_strategy_ids]
        for sid in all_strategies:
            if sid not in self.strategies:
                raise ValueError(f"Strategy {sid} not found")

        merge_id = f"merge_{uuid4().hex[:8]}"
        merged_strategy_id = f"merged_{uuid4().hex[:8]}"

        merge_record = MergeRecord(
            merge_id=merge_id,
            target_strategy_id=target_strategy_id,
            source_strategy_ids=source_strategy_ids,
            merged_strategy_id=merged_strategy_id,
            merge_strategy=merge_strategy,
            conflict_resolution=conflict_resolution or {},
            created_at=datetime.now(UTC),
            created_by=created_by,
        )

        self.merges[merge_id] = merge_record

        # Add merge relationships to graph
        for source_id in source_strategy_ids:
            self.lineage_graph[source_id].append(merged_strategy_id)
            edge_key = f"{source_id}->{merged_strategy_id}"
            self.graph_edges[edge_key] = {
                "relationship_type": RelationshipType.MERGE,
                "merge_id": merge_id,
                "source": source_id,
                "target": merged_strategy_id,
            }

        self.lineage_graph[target_strategy_id].append(merged_strategy_id)
        edge_key = f"{target_strategy_id}->{merged_strategy_id}"
        self.graph_edges[edge_key] = {
            "relationship_type": RelationshipType.MERGE,
            "merge_id": merge_id,
            "source": target_strategy_id,
            "target": merged_strategy_id,
        }

        # Register merged strategy
        self.register_strategy(
            merged_strategy_id,
            parent_id=target_strategy_id,
            metadata={
                "merged_from": all_strategies,
                "merge_strategy": merge_strategy,
                "merge_id": merge_id,
            },
        )

        self.logger.info(f"Merged strategies {all_strategies} into {merged_strategy_id}")
        return merge_id

    def get_lineage(self, strategy_id: str) -> dict[str, Any]:
        """
        Get complete lineage information for a strategy

        Args:
            strategy_id: Strategy to analyze

        Returns:
            Lineage information dictionary
        """
        if strategy_id not in self.strategies:
            raise ValueError(f"Strategy {strategy_id} not found")

        # Check cache first
        if strategy_id in self._lineage_cache and not self._cache_dirty:
            return self._lineage_cache[strategy_id]

        # Get ancestors (parents, grandparents, etc.)
        ancestors = []
        current_id = self.strategies[strategy_id]["parent_id"]
        while current_id and current_id in self.strategies:
            ancestors.append(
                {
                    "id": current_id,
                    "generation": self.strategies[current_id]["generation"],
                    "created_at": self.strategies[current_id]["created_at"].isoformat(),
                    "branch_id": self.strategies[current_id].get("branch_id"),
                }
            )
            current_id = self.strategies[current_id]["parent_id"]

        # Get descendants (children, grandchildren, etc.)
        descendants = self._get_descendants(strategy_id)

        # Get siblings (strategies with same parent)
        siblings = []
        parent_id = self.strategies[strategy_id]["parent_id"]
        if parent_id:
            siblings = [
                {
                    "id": sid,
                    "generation": self.strategies[sid]["generation"],
                    "created_at": self.strategies[sid]["created_at"].isoformat(),
                }
                for sid in self.strategies
                if self.strategies[sid]["parent_id"] == parent_id and sid != strategy_id
            ]

        # Get branch information
        branch_info = None
        branch_id = self.strategies[strategy_id].get("branch_id")
        if branch_id and branch_id in self.branches:
            branch = self.branches[branch_id]
            branch_info = {
                "id": branch.branch_id,
                "name": branch.branch_name,
                "description": branch.description,
                "active": branch.active,
                "strategies": branch.strategies,
            }

        # Get change history
        change_history = [change.to_dict() for change in self.changes[strategy_id]]

        lineage_info = {
            "strategy_id": strategy_id,
            "generation": self.strategies[strategy_id]["generation"],
            "ancestors": list(reversed(ancestors)),  # Root first
            "descendants": descendants,
            "siblings": siblings,
            "branch": branch_info,
            "changes": change_history,
            "total_ancestors": len(ancestors),
            "total_descendants": len(descendants),
            "lineage_depth": len(ancestors) + 1,
        }

        # Cache result
        self._lineage_cache[strategy_id] = lineage_info

        return lineage_info

    def get_evolution_path(
        self, from_strategy_id: str, to_strategy_id: str
    ) -> list[dict[str, Any]]:
        """
        Get evolution path between two strategies

        Args:
            from_strategy_id: Starting strategy
            to_strategy_id: Target strategy

        Returns:
            List of strategies in evolution path
        """
        if from_strategy_id not in self.strategies or to_strategy_id not in self.strategies:
            raise ValueError("One or both strategies not found")

        # Simple BFS to find path
        path = self._find_path(from_strategy_id, to_strategy_id)

        if not path:
            return []

        evolution_path = []
        for i, strategy_id in enumerate(path):
            strategy_info = {
                "strategy_id": strategy_id,
                "generation": self.strategies[strategy_id]["generation"],
                "created_at": self.strategies[strategy_id]["created_at"].isoformat(),
                "step": i,
                "changes": [change.to_dict() for change in self.changes[strategy_id]],
            }

            # Add relationship info for non-first strategies
            if i > 0:
                prev_id = path[i - 1]
                edge_key = f"{prev_id}->{strategy_id}"
                if edge_key in self.graph_edges:
                    edge_data = self.graph_edges[edge_key]
                    strategy_info["relationship"] = edge_data.get(
                        "relationship_type", RelationshipType.PARENT
                    ).value

            evolution_path.append(strategy_info)

        return evolution_path

    def analyze_change_impact(self, strategy_id: str, change_id: str) -> dict[str, Any]:
        """
        Analyze the impact of a specific change

        Args:
            strategy_id: Strategy that was changed
            change_id: Specific change to analyze

        Returns:
            Change impact analysis
        """
        if strategy_id not in self.changes:
            raise ValueError(f"No changes found for strategy {strategy_id}")

        # Find the change record
        change_record = None
        for change in self.changes[strategy_id]:
            if change.change_id == change_id:
                change_record = change
                break

        if not change_record:
            raise ValueError(f"Change {change_id} not found")

        # Get descendants to analyze propagation
        descendants = self._get_descendants(strategy_id)

        # Analyze impact on descendants
        descendant_impact = []
        for desc in descendants:
            desc_id = desc["id"]
            desc_changes = self.changes.get(desc_id, [])

            # Look for related changes after this change
            related_changes = [
                c
                for c in desc_changes
                if c.created_at > change_record.created_at
                and any(comp in change_record.changed_components for comp in c.changed_components)
            ]

            if related_changes:
                descendant_impact.append(
                    {
                        "strategy_id": desc_id,
                        "generation_distance": desc["generation"]
                        - self.strategies[strategy_id]["generation"],
                        "related_changes": len(related_changes),
                        "impact_types": list({c.change_type.value for c in related_changes}),
                    }
                )

        return {
            "change_id": change_id,
            "strategy_id": strategy_id,
            "change_type": change_record.change_type.value,
            "impact_level": change_record.impact_level.value,
            "description": change_record.description,
            "changed_components": change_record.changed_components,
            "performance_impact": change_record.performance_impact,
            "descendant_impact": descendant_impact,
            "propagation_depth": (
                max([d["generation_distance"] for d in descendant_impact])
                if descendant_impact
                else 0
            ),
            "affected_strategies": len(descendant_impact),
            "created_at": change_record.created_at.isoformat(),
        }

    def get_evolution_metrics(self) -> EvolutionMetrics:
        """
        Get comprehensive evolution metrics

        Returns:
            Evolution metrics
        """
        if not self._cache_dirty and self._evolution_cache:
            return EvolutionMetrics(**self._evolution_cache)

        # Calculate metrics
        total_strategies = len(self.strategies)
        total_branches = len(self.branches)
        total_merges = len(self.merges)

        # Generation analysis
        generations = [s["generation"] for s in self.strategies.values()]
        avg_generation_distance = sum(generations) / len(generations) if generations else 0
        max_generation_distance = max(generations) if generations else 0

        # Find most evolved lineage (longest path from root)
        most_evolved_lineage = []
        max_depth = 0

        for strategy_id in self.strategies:
            lineage = self.get_lineage(strategy_id)
            if lineage["lineage_depth"] > max_depth:
                max_depth = lineage["lineage_depth"]
                most_evolved_lineage = [a["id"] for a in lineage["ancestors"]] + [strategy_id]

        # Performance improvement analysis
        performance_improvements = []
        for strategy_id in self.strategies:
            for change in self.changes[strategy_id]:
                if change.performance_impact:
                    improvements = [v for v in change.performance_impact.values() if v > 0]
                    performance_improvements.extend(improvements)

        performance_improvement_rate = (
            sum(performance_improvements) / len(performance_improvements)
            if performance_improvements
            else 0.0
        )

        # Change frequency analysis
        change_frequency = defaultdict(int)
        for changes_list in self.changes.values():
            for change in changes_list:
                change_frequency[change.change_type] += 1

        metrics = EvolutionMetrics(
            total_strategies=total_strategies,
            total_branches=total_branches,
            total_merges=total_merges,
            avg_generation_distance=avg_generation_distance,
            max_generation_distance=max_generation_distance,
            most_evolved_lineage=most_evolved_lineage,
            performance_improvement_rate=performance_improvement_rate,
            change_frequency=dict(change_frequency),
        )

        # Cache results
        self._evolution_cache = metrics.to_dict()
        self._cache_dirty = False

        return metrics

    def visualize_lineage(self, strategy_id: str, format: str = "dict") -> dict[str, Any] | str:
        """
        Create visualization data for strategy lineage

        Args:
            strategy_id: Strategy to visualize
            format: Output format ('dict', 'mermaid', 'dot')

        Returns:
            Visualization data in requested format
        """
        lineage = self.get_lineage(strategy_id)

        if format == "dict":
            return {
                "nodes": self._get_lineage_nodes(strategy_id),
                "edges": self._get_lineage_edges(strategy_id),
                "metadata": lineage,
            }
        if format == "mermaid":
            return self._generate_mermaid_diagram(strategy_id)
        if format == "dot":
            return self._generate_dot_diagram(strategy_id)
        raise ValueError(f"Unsupported format: {format}")

    def _get_descendants(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get all descendants of a strategy"""
        descendants = []

        # Use BFS to find all descendants
        queue = deque([strategy_id])
        visited = {strategy_id}

        while queue:
            current_id = queue.popleft()

            # Find direct children
            for sid, strategy_data in self.strategies.items():
                if strategy_data["parent_id"] == current_id and sid not in visited:
                    descendants.append(
                        {
                            "id": sid,
                            "generation": strategy_data["generation"],
                            "created_at": strategy_data["created_at"].isoformat(),
                            "branch_id": strategy_data.get("branch_id"),
                        }
                    )
                    queue.append(sid)
                    visited.add(sid)

        return descendants

    def _get_lineage_nodes(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get nodes for lineage visualization"""
        lineage = self.get_lineage(strategy_id)

        # Add ancestors using list comprehension
        nodes = [
            {
                "id": ancestor["id"],
                "type": "ancestor",
                "generation": ancestor["generation"],
                "created_at": ancestor["created_at"],
            }
            for ancestor in lineage["ancestors"]
        ]

        # Add current strategy
        nodes.append(
            {
                "id": strategy_id,
                "type": "current",
                "generation": lineage["generation"],
                "created_at": self.strategies[strategy_id]["created_at"].isoformat(),
            }
        )

        # Add descendants using extend
        nodes.extend(
            {
                "id": descendant["id"],
                "type": "descendant",
                "generation": descendant["generation"],
                "created_at": descendant["created_at"],
            }
            for descendant in lineage["descendants"]
        )

        return nodes

    def _get_lineage_edges(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get edges for lineage visualization"""
        edges = []

        # Get all related strategies
        lineage = self.get_lineage(strategy_id)
        all_strategies = (
            [a["id"] for a in lineage["ancestors"]]
            + [strategy_id]
            + [d["id"] for d in lineage["descendants"]]
        )

        # Get edges from graph using values() since we don't need the keys
        edges = [
            {
                "source": edge_data["source"],
                "target": edge_data["target"],
                "relationship": edge_data.get("relationship_type", RelationshipType.PARENT).value,
                "branch_id": edge_data.get("branch_id"),
                "merge_id": edge_data.get("merge_id"),
            }
            for edge_data in self.graph_edges.values()
            if edge_data["source"] in all_strategies and edge_data["target"] in all_strategies
        ]

        return edges

    def _generate_mermaid_diagram(self, strategy_id: str) -> str:
        """Generate Mermaid diagram for lineage"""
        nodes = self._get_lineage_nodes(strategy_id)
        edges = self._get_lineage_edges(strategy_id)

        mermaid = ["graph TD"]

        # Add nodes using extend
        for node in nodes:
            node_style = "fill:#e1f5fe" if node["type"] == "current" else "fill:#f3e5f5"
            mermaid.extend(
                [
                    f"    {node['id']}[{node['id']}]",
                    f"    {node['id']} --> {node_style}",
                ]
            )

        # Add edges using extend
        mermaid.extend(f"    {edge['source']} --> {edge['target']}" for edge in edges)

        return "\n".join(mermaid)

    def _generate_dot_diagram(self, strategy_id: str) -> str:
        """Generate DOT diagram for lineage"""
        nodes = self._get_lineage_nodes(strategy_id)
        edges = self._get_lineage_edges(strategy_id)

        dot = ["digraph StrategyLineage {", "    rankdir=TB;"]

        # Add nodes using extend
        dot.extend(
            f'    "{node["id"]}" [fillcolor={"lightblue" if node["type"] == "current" else "lightgray"}, style=filled];'
            for node in nodes
        )

        # Add edges using extend
        dot.extend(f'    "{edge["source"]}" -> "{edge["target"]}";' for edge in edges)

        dot.append("}")
        return "\n".join(dot)

    def _find_path(self, start: str, end: str) -> list[str]:
        """Find path between two strategies using BFS"""
        if start == end:
            return [start]

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            # Check all outgoing edges from current strategy
            for neighbor in self.lineage_graph.get(current, []):
                if neighbor == end:
                    return [*path, neighbor]

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, neighbor]))

        return []  # No path found

    def _invalidate_caches(self) -> None:
        """Invalidate all caches"""
        self._lineage_cache.clear()
        self._evolution_cache.clear()
        self._cache_dirty = True
