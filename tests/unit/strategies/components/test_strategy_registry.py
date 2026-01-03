"""
Tests for Strategy Registry and Version Control

This module tests the StrategyRegistry implementation including version management,
strategy metadata tracking, serialization/deserialization, and validation.
"""

from datetime import UTC, datetime
from unittest.mock import Mock, patch

import pytest

from src.strategies.components.position_sizer import FixedFractionSizer
from src.strategies.components.risk_manager import FixedRiskManager
from src.strategies.components.signal_generator import HoldSignalGenerator, RandomSignalGenerator
from src.strategies.components.strategy import Strategy
from src.strategies.components.strategy_registry import (
    ComponentConfig,
    StrategyMetadata,
    StrategyRegistry,
    StrategyStatus,
    StrategyValidationError,
    StrategyVersion,
)


class TestComponentConfig:
    """Test ComponentConfig data class"""

    def test_component_config_creation(self):
        """Test ComponentConfig creation and serialization"""
        config = ComponentConfig(
            type="signal_generator",
            class_name="HoldSignalGenerator",
            parameters={"test_param": "test_value"},
            version="1.0.0",
        )

        assert config.type == "signal_generator"
        assert config.class_name == "HoldSignalGenerator"
        assert config.parameters == {"test_param": "test_value"}
        assert config.version == "1.0.0"

    def test_component_config_serialization(self):
        """Test ComponentConfig to_dict and from_dict"""
        config = ComponentConfig(
            type="risk_manager",
            class_name="FixedRiskManager",
            parameters={"risk_per_trade": 0.02},
            version="1.0.0",
        )

        # Test serialization
        config_dict = config.to_dict()
        expected_dict = {
            "type": "risk_manager",
            "class_name": "FixedRiskManager",
            "parameters": {"risk_per_trade": 0.02},
            "version": "1.0.0",
        }
        assert config_dict == expected_dict

        # Test deserialization
        restored_config = ComponentConfig.from_dict(config_dict)
        assert restored_config.type == config.type
        assert restored_config.class_name == config.class_name
        assert restored_config.parameters == config.parameters
        assert restored_config.version == config.version


class TestStrategyMetadata:
    """Test StrategyMetadata data class"""

    def test_strategy_metadata_creation(self):
        """Test StrategyMetadata creation"""
        signal_config = ComponentConfig("signal_generator", "HoldSignalGenerator", {}, "1.0.0")
        risk_config = ComponentConfig(
            "risk_manager", "FixedRiskManager", {"risk_per_trade": 0.02}, "1.0.0"
        )
        sizer_config = ComponentConfig(
            "position_sizer", "FixedFractionSizer", {"fraction": 0.05}, "1.0.0"
        )
        regime_config = ComponentConfig("regime_detector", "EnhancedRegimeDetector", {}, "1.0.0")

        metadata = StrategyMetadata(
            id="test_strategy_123",
            name="Test Strategy",
            version="1.0.0",
            parent_id=None,
            created_at=datetime.now(UTC),
            created_by="test_user",
            description="Test strategy description",
            tags=["test", "experimental"],
            status=StrategyStatus.EXPERIMENTAL,
            signal_generator_config=signal_config,
            risk_manager_config=risk_config,
            position_sizer_config=sizer_config,
            regime_detector_config=regime_config,
            parameters={"test_param": "test_value"},
            performance_summary=None,
            validation_results=None,
            lineage_path=[],
            branch_name=None,
            merge_source=None,
            config_hash="test_hash",
            component_hash="test_component_hash",
        )

        assert metadata.id == "test_strategy_123"
        assert metadata.name == "Test Strategy"
        assert metadata.status == StrategyStatus.EXPERIMENTAL
        assert len(metadata.tags) == 2
        assert "test" in metadata.tags

    def test_strategy_metadata_serialization(self):
        """Test StrategyMetadata serialization and deserialization"""
        signal_config = ComponentConfig("signal_generator", "HoldSignalGenerator", {}, "1.0.0")
        risk_config = ComponentConfig(
            "risk_manager", "FixedRiskManager", {"risk_per_trade": 0.02}, "1.0.0"
        )
        sizer_config = ComponentConfig(
            "position_sizer", "FixedFractionSizer", {"fraction": 0.05}, "1.0.0"
        )
        regime_config = ComponentConfig("regime_detector", "EnhancedRegimeDetector", {}, "1.0.0")

        created_at = datetime.now(UTC)
        metadata = StrategyMetadata(
            id="test_strategy_123",
            name="Test Strategy",
            version="1.0.0",
            parent_id=None,
            created_at=created_at,
            created_by="test_user",
            description="Test strategy description",
            tags=["test"],
            status=StrategyStatus.EXPERIMENTAL,
            signal_generator_config=signal_config,
            risk_manager_config=risk_config,
            position_sizer_config=sizer_config,
            regime_detector_config=regime_config,
            parameters={},
            performance_summary=None,
            validation_results=None,
            lineage_path=[],
            branch_name=None,
            merge_source=None,
            config_hash="test_hash",
            component_hash="test_component_hash",
        )

        # Test serialization
        metadata_dict = metadata.to_dict()
        assert metadata_dict["id"] == "test_strategy_123"
        assert metadata_dict["status"] == "EXPERIMENTAL"
        assert metadata_dict["created_at"] == created_at.isoformat()

        # Test deserialization
        restored_metadata = StrategyMetadata.from_dict(metadata_dict)
        assert restored_metadata.id == metadata.id
        assert restored_metadata.status == metadata.status
        assert restored_metadata.created_at == metadata.created_at


class TestStrategyRegistry:
    """Test StrategyRegistry functionality"""

    @pytest.fixture
    def registry(self):
        """Create a test registry"""
        return StrategyRegistry()

    @pytest.fixture
    def test_strategy(self):
        """Create a test strategy"""
        signal_generator = HoldSignalGenerator()
        risk_manager = FixedRiskManager(risk_per_trade=0.02)
        position_sizer = FixedFractionSizer(fraction=0.05)

        return Strategy(
            name="Test Strategy",
            signal_generator=signal_generator,
            risk_manager=risk_manager,
            position_sizer=position_sizer,
        )

    def test_registry_initialization(self, registry):
        """Test registry initialization"""
        assert isinstance(registry, StrategyRegistry)
        assert len(registry._strategies) == 0
        assert len(registry._versions) == 0
        assert len(registry._lineage) == 0

    def test_register_strategy(self, registry, test_strategy):
        """Test strategy registration"""
        metadata = {
            "created_by": "test_user",
            "description": "Test strategy for unit testing",
            "tags": ["test", "experimental"],
            "status": "EXPERIMENTAL",
        }

        strategy_id = registry.register_strategy(test_strategy, metadata)

        assert strategy_id is not None
        assert strategy_id in registry._strategies
        assert strategy_id in registry._versions

        # Check metadata
        stored_metadata = registry._strategies[strategy_id]
        assert stored_metadata.name == "Test Strategy"
        assert stored_metadata.created_by == "test_user"
        assert stored_metadata.status == StrategyStatus.EXPERIMENTAL
        assert "test" in stored_metadata.tags

        # Check version
        versions = registry._versions[strategy_id]
        assert len(versions) == 1
        assert versions[0].version == "1.0.0"
        assert versions[0].is_major

    def test_register_strategy_with_parent(self, registry, test_strategy):
        """Test strategy registration with parent lineage"""
        # Register parent strategy
        parent_metadata = {
            "created_by": "test_user",
            "description": "Parent strategy",
            "tags": ["parent"],
            "status": "EXPERIMENTAL",
        }
        parent_id = registry.register_strategy(test_strategy, parent_metadata)

        # Register child strategy
        child_strategy = Strategy(
            name="Child Strategy",
            signal_generator=RandomSignalGenerator(buy_prob=0.3, sell_prob=0.3),
            risk_manager=FixedRiskManager(risk_per_trade=0.03),
            position_sizer=FixedFractionSizer(fraction=0.04),
        )

        child_metadata = {
            "created_by": "test_user",
            "description": "Child strategy",
            "tags": ["child"],
            "status": "EXPERIMENTAL",
        }

        child_id = registry.register_strategy(child_strategy, child_metadata, parent_id=parent_id)

        # Check lineage
        child_stored = registry._strategies[child_id]
        assert child_stored.parent_id == parent_id
        assert child_stored.lineage_path == [parent_id]

        # Check parent-child relationship
        assert parent_id in registry._lineage
        assert child_id in registry._lineage[parent_id]

    def test_update_strategy(self, registry, test_strategy):
        """Test strategy updates and versioning"""
        # Register initial strategy
        metadata = {
            "created_by": "test_user",
            "description": "Initial strategy",
            "status": "EXPERIMENTAL",
        }
        strategy_id = registry.register_strategy(test_strategy, metadata)

        # Update strategy
        updated_strategy = Strategy(
            name="Test Strategy",
            signal_generator=RandomSignalGenerator(buy_prob=0.4, sell_prob=0.3),
            risk_manager=FixedRiskManager(risk_per_trade=0.025),
            position_sizer=FixedFractionSizer(fraction=0.06),
        )

        changes = ["Updated signal generator probabilities", "Increased risk per trade"]
        new_version = registry.update_strategy(
            strategy_id, updated_strategy, changes, is_major=False
        )

        assert new_version == "1.0.1"

        # Check updated metadata
        updated_metadata = registry._strategies[strategy_id]
        assert updated_metadata.version == "1.0.1"

        # Check version history
        versions = registry._versions[strategy_id]
        assert len(versions) == 2
        assert versions[1].version == "1.0.1"
        assert versions[1].changes == changes
        assert not versions[1].is_major

    def test_update_strategy_major_version(self, registry, test_strategy):
        """Test major version updates"""
        # Register and update to major version
        metadata = {"created_by": "test_user", "status": "EXPERIMENTAL"}
        strategy_id = registry.register_strategy(test_strategy, metadata)

        new_version = registry.update_strategy(
            strategy_id, test_strategy, ["Major architectural change"], is_major=True
        )

        assert new_version == "2.0.0"

    def test_get_strategy_metadata(self, registry, test_strategy):
        """Test retrieving strategy metadata"""
        metadata = {"created_by": "test_user", "status": "EXPERIMENTAL"}
        strategy_id = registry.register_strategy(test_strategy, metadata)

        retrieved_metadata = registry.get_strategy_metadata(strategy_id)
        assert retrieved_metadata is not None
        assert retrieved_metadata.id == strategy_id
        assert retrieved_metadata.name == "Test Strategy"

        # Test non-existent strategy
        assert registry.get_strategy_metadata("non_existent") is None

    def test_get_strategy_versions(self, registry, test_strategy):
        """Test retrieving strategy versions"""
        metadata = {"created_by": "test_user", "status": "EXPERIMENTAL"}
        strategy_id = registry.register_strategy(test_strategy, metadata)

        # Add another version
        registry.update_strategy(strategy_id, test_strategy, ["Minor update"])

        versions = registry.get_strategy_versions(strategy_id)
        assert len(versions) == 2
        assert versions[0].version == "1.0.0"
        assert versions[1].version == "1.0.1"

        # Test non-existent strategy
        assert registry.get_strategy_versions("non_existent") == []

    def test_list_strategies(self, registry, test_strategy):
        """Test listing strategies with filters"""
        # Register multiple strategies
        metadata1 = {"created_by": "user1", "status": "EXPERIMENTAL", "tags": ["test", "ml"]}
        metadata2 = {"created_by": "user2", "status": "PRODUCTION", "tags": ["live", "ml"]}
        metadata3 = {"created_by": "user3", "status": "EXPERIMENTAL", "tags": ["test"]}

        strategy1 = Strategy(
            "Strategy1",
            test_strategy.signal_generator,
            test_strategy.risk_manager,
            test_strategy.position_sizer,
        )
        strategy2 = Strategy(
            "Strategy2",
            test_strategy.signal_generator,
            test_strategy.risk_manager,
            test_strategy.position_sizer,
        )
        strategy3 = Strategy(
            "Strategy3",
            test_strategy.signal_generator,
            test_strategy.risk_manager,
            test_strategy.position_sizer,
        )

        registry.register_strategy(strategy1, metadata1)
        registry.register_strategy(strategy2, metadata2)
        registry.register_strategy(strategy3, metadata3)

        # Test no filter
        all_strategies = registry.list_strategies()
        assert len(all_strategies) == 3

        # Test status filter
        experimental_strategies = registry.list_strategies(status=StrategyStatus.EXPERIMENTAL)
        assert len(experimental_strategies) == 2

        production_strategies = registry.list_strategies(status=StrategyStatus.PRODUCTION)
        assert len(production_strategies) == 1

        # Test tags filter
        ml_strategies = registry.list_strategies(tags=["ml"])
        assert len(ml_strategies) == 2

        test_strategies = registry.list_strategies(tags=["test"])
        assert len(test_strategies) == 2

        # Test combined filter
        experimental_ml = registry.list_strategies(status=StrategyStatus.EXPERIMENTAL, tags=["ml"])
        assert len(experimental_ml) == 1

    def test_get_strategy_lineage(self, registry, test_strategy):
        """Test strategy lineage tracking"""
        # Create a lineage: grandparent -> parent -> child
        grandparent_id = registry.register_strategy(
            test_strategy, {"created_by": "user", "status": "EXPERIMENTAL"}
        )

        parent_strategy = Strategy(
            "Parent",
            test_strategy.signal_generator,
            test_strategy.risk_manager,
            test_strategy.position_sizer,
        )
        parent_id = registry.register_strategy(
            parent_strategy,
            {"created_by": "user", "status": "EXPERIMENTAL"},
            parent_id=grandparent_id,
        )

        child_strategy = Strategy(
            "Child",
            test_strategy.signal_generator,
            test_strategy.risk_manager,
            test_strategy.position_sizer,
        )
        child_id = registry.register_strategy(
            child_strategy, {"created_by": "user", "status": "EXPERIMENTAL"}, parent_id=parent_id
        )

        # Test child lineage
        child_lineage = registry.get_strategy_lineage(child_id)
        assert child_lineage["strategy_id"] == child_id
        assert len(child_lineage["ancestors"]) == 2
        assert child_lineage["ancestors"][0]["id"] == grandparent_id  # Root first
        assert child_lineage["ancestors"][1]["id"] == parent_id

        # Test parent lineage
        parent_lineage = registry.get_strategy_lineage(parent_id)
        assert len(parent_lineage["ancestors"]) == 1
        assert len(parent_lineage["descendants"]) == 1
        assert parent_lineage["descendants"][0]["id"] == child_id

        # Test grandparent lineage
        grandparent_lineage = registry.get_strategy_lineage(grandparent_id)
        assert len(grandparent_lineage["ancestors"]) == 0
        assert len(grandparent_lineage["descendants"]) == 1
        assert grandparent_lineage["descendants"][0]["id"] == parent_id

    def test_serialize_deserialize_strategy(self, registry, test_strategy):
        """Test strategy serialization and deserialization"""
        # Register strategy
        metadata = {
            "created_by": "test_user",
            "description": "Test serialization",
            "tags": ["test"],
            "status": "EXPERIMENTAL",
        }
        strategy_id = registry.register_strategy(test_strategy, metadata)

        # Add a version
        registry.update_strategy(strategy_id, test_strategy, ["Test update"])

        # Serialize
        serialized_data = registry.serialize_strategy(strategy_id)

        assert "metadata" in serialized_data
        assert "versions" in serialized_data
        assert "lineage" in serialized_data
        assert len(serialized_data["versions"]) == 2

        # Create new registry and deserialize
        new_registry = StrategyRegistry()
        restored_id = new_registry.deserialize_strategy(serialized_data)

        assert restored_id == strategy_id
        assert strategy_id in new_registry._strategies
        assert len(new_registry._versions[strategy_id]) == 2

        # Verify metadata matches
        original_metadata = registry._strategies[strategy_id]
        restored_metadata = new_registry._strategies[strategy_id]
        assert original_metadata.name == restored_metadata.name
        assert original_metadata.version == restored_metadata.version
        assert original_metadata.created_by == restored_metadata.created_by

    def test_validate_strategy_integrity(self, registry, test_strategy):
        """Test strategy integrity validation"""
        # Register valid strategy
        metadata = {"created_by": "test_user", "status": "EXPERIMENTAL"}
        strategy_id = registry.register_strategy(test_strategy, metadata)

        # Test valid strategy
        validation_result = registry.validate_strategy_integrity(strategy_id)
        assert validation_result["valid"]
        assert len(validation_result["errors"]) == 0

        # Test non-existent strategy
        invalid_result = registry.validate_strategy_integrity("non_existent")
        assert not invalid_result["valid"]
        assert "Strategy not found" in invalid_result["errors"]

    def test_strategy_validation_errors(self, registry):
        """Test strategy validation error handling"""
        # Test invalid strategy type
        with pytest.raises(StrategyValidationError):
            registry.register_strategy("not_a_strategy", {})

        # Test strategy with empty name - Strategy validates during __init__
        with pytest.raises(ValueError, match="Strategy name must be a non-empty string"):
            Strategy(
                name="",
                signal_generator=HoldSignalGenerator(),
                risk_manager=FixedRiskManager(),
                position_sizer=FixedFractionSizer(),
            )

        # Test strategy with missing components
        incomplete_strategy = Mock()
        incomplete_strategy.name = "Test"
        incomplete_strategy.signal_generator = None

        with pytest.raises(StrategyValidationError):
            registry.register_strategy(incomplete_strategy, {})

    def test_update_nonexistent_strategy(self, registry, test_strategy):
        """Test updating non-existent strategy"""
        with pytest.raises(ValueError):
            registry.update_strategy("non_existent", test_strategy, ["Update"])

    def test_serialize_nonexistent_strategy(self, registry):
        """Test serializing non-existent strategy"""
        with pytest.raises(ValueError):
            registry.serialize_strategy("non_existent")

    def test_version_calculation(self, registry, test_strategy):
        """Test version number calculation"""
        metadata = {"created_by": "test_user", "status": "EXPERIMENTAL"}
        strategy_id = registry.register_strategy(test_strategy, metadata)

        # Test minor updates
        registry.update_strategy(strategy_id, test_strategy, ["Minor 1"], is_major=False)
        assert registry._strategies[strategy_id].version == "1.0.1"

        registry.update_strategy(strategy_id, test_strategy, ["Minor 2"], is_major=False)
        assert registry._strategies[strategy_id].version == "1.0.2"

        # Test major update
        registry.update_strategy(strategy_id, test_strategy, ["Major"], is_major=True)
        assert registry._strategies[strategy_id].version == "2.0.0"

        # Test minor after major
        registry.update_strategy(strategy_id, test_strategy, ["Minor after major"], is_major=False)
        assert registry._strategies[strategy_id].version == "2.0.1"

    def test_component_config_extraction(self, registry, test_strategy):
        """Test component configuration extraction"""
        configs = registry._extract_component_configs(test_strategy)

        assert "signal_generator" in configs
        assert "risk_manager" in configs
        assert "position_sizer" in configs
        assert "regime_detector" in configs

        # Check signal generator config
        sg_config = configs["signal_generator"]
        assert sg_config.type == "signal_generator"
        assert sg_config.class_name == "HoldSignalGenerator"
        assert isinstance(sg_config.parameters, dict)

        # Check risk manager config
        rm_config = configs["risk_manager"]
        assert rm_config.type == "risk_manager"
        assert rm_config.class_name == "FixedRiskManager"
        assert "risk_per_trade" in rm_config.parameters

    def test_hash_calculation(self, registry, test_strategy):
        """Test configuration and component hash calculation"""
        configs = registry._extract_component_configs(test_strategy)
        parameters = {"test_param": "test_value"}

        # Test config hash
        config_hash = registry._calculate_config_hash(configs, parameters)
        assert isinstance(config_hash, str)
        assert len(config_hash) == 64  # SHA256 hex digest length

        # Test component hash
        component_hash = registry._calculate_component_hash(test_strategy)
        assert isinstance(component_hash, str)
        assert len(component_hash) == 64

        # Test hash consistency
        config_hash2 = registry._calculate_config_hash(configs, parameters)
        assert config_hash == config_hash2

        component_hash2 = registry._calculate_component_hash(test_strategy)
        assert component_hash == component_hash2

    def test_lineage_path_building(self, registry):
        """Test lineage path building"""
        # Test with no parent
        path = registry._build_lineage_path(None)
        assert path == []

        # Test with non-existent parent
        path = registry._build_lineage_path("non_existent")
        assert path == []

        # Test with existing parent
        test_strategy = Strategy(
            "Test", HoldSignalGenerator(), FixedRiskManager(), FixedFractionSizer()
        )
        parent_id = registry.register_strategy(
            test_strategy, {"created_by": "user", "status": "EXPERIMENTAL"}
        )

        path = registry._build_lineage_path(parent_id)
        assert path == [parent_id]

    @patch("src.strategies.components.strategy_registry.uuid4")
    def test_strategy_id_generation(self, mock_uuid, registry):
        """Test strategy ID generation"""
        mock_uuid.return_value.hex = "abcd1234" * 4  # 32 chars

        strategy_id = registry._generate_strategy_id("Test Strategy")
        assert strategy_id == "test_strategy_abcd1234"

        strategy_id2 = registry._generate_strategy_id("Another Strategy")
        assert strategy_id2 == "another_strategy_abcd1234"


class TestStrategyVersion:
    """Test StrategyVersion data class"""

    def test_strategy_version_creation(self):
        """Test StrategyVersion creation"""
        version = StrategyVersion(
            version="1.0.1",
            strategy_id="test_strategy_123",
            created_at=datetime.now(UTC),
            changes=["Updated parameters", "Fixed bug"],
            performance_delta={"sharpe_ratio": 0.1, "return": 0.05},
            is_major=False,
        )

        assert version.version == "1.0.1"
        assert version.strategy_id == "test_strategy_123"
        assert len(version.changes) == 2
        assert not version.is_major
        assert version.performance_delta["sharpe_ratio"] == 0.1

    def test_strategy_version_serialization(self):
        """Test StrategyVersion serialization"""
        created_at = datetime.now(UTC)
        version = StrategyVersion(
            version="2.0.0",
            strategy_id="test_strategy_456",
            created_at=created_at,
            changes=["Major refactor"],
            performance_delta=None,
            is_major=True,
        )

        # Test serialization
        version_dict = version.to_dict()
        assert version_dict["version"] == "2.0.0"
        assert version_dict["created_at"] == created_at.isoformat()
        assert version_dict["is_major"]

        # Test deserialization
        restored_version = StrategyVersion.from_dict(version_dict)
        assert restored_version.version == version.version
        assert restored_version.created_at == version.created_at
        assert restored_version.is_major == version.is_major


if __name__ == "__main__":
    pytest.main([__file__])
