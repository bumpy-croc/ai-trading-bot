"""
Strategy Registry and Version Control

This module implements the StrategyRegistry with version management,
strategy metadata tracking, serialization/deserialization, and validation.
"""

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .position_sizer import PositionSizer
from .regime_context import EnhancedRegimeDetector
from .risk_manager import RiskManager
from .signal_generator import SignalGenerator
from .strategy import Strategy


class StrategyStatus(Enum):
    """Strategy status enumeration"""
    EXPERIMENTAL = "EXPERIMENTAL"
    TESTING = "TESTING"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"
    DEPRECATED = "DEPRECATED"


@dataclass
class ComponentConfig:
    """Configuration for a strategy component"""
    type: str
    class_name: str
    parameters: dict[str, Any]
    version: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ComponentConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class StrategyMetadata:
    """Comprehensive strategy metadata"""
    id: str
    name: str
    version: str
    parent_id: Optional[str]
    created_at: datetime
    created_by: str
    description: str
    tags: list[str]
    status: StrategyStatus

    # Component configurations
    signal_generator_config: ComponentConfig
    risk_manager_config: ComponentConfig
    position_sizer_config: ComponentConfig
    regime_detector_config: ComponentConfig

    # Additional metadata
    parameters: Dict[str, Any]
    performance_summary: Optional[Dict[str, Any]]
    validation_results: Optional[Dict[str, Any]]
    # Lineage tracking
    lineage_path: list[str]  # Path from root ancestor to this strategy
    branch_name: Optional[str]
    merge_source: Optional[str]

    # Checksums for integrity
    config_hash: str
    component_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StrategyMetadata':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['status'] = StrategyStatus(data['status'])
        data['signal_generator_config'] = ComponentConfig.from_dict(data['signal_generator_config'])
        data['risk_manager_config'] = ComponentConfig.from_dict(data['risk_manager_config'])
        data['position_sizer_config'] = ComponentConfig.from_dict(data['position_sizer_config'])
        data['regime_detector_config'] = ComponentConfig.from_dict(data['regime_detector_config'])
        return cls(**data)


@dataclass
class StrategyVersion:
    """Strategy version information"""
    version: str
    strategy_id: str
    created_at: datetime
    changes: list[str]
    performance_delta: Optional[dict[str, float]]
    is_major: bool

    # Configuration snapshot for this version
    signal_generator_config: Optional[dict[str, Any]] = None
    risk_manager_config: Optional[dict[str, Any]] = None
    position_sizer_config: Optional[dict[str, Any]] = None
    regime_detector_config: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None
    config_hash: Optional[str] = None
    component_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StrategyVersion':
        """Create from dictionary"""
        data = data.copy()
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class StrategyValidationError(Exception):
    """Exception raised when strategy validation fails"""
    pass


class StrategyRegistry:
    """
    Strategy registry with version control and metadata management
    
    This class manages strategy registration, versioning, serialization,
    and validation. It provides a centralized repository for all strategies
    with comprehensive metadata tracking.
    """

    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize strategy registry
        
        Args:
            storage_backend: Optional storage backend for persistence
        """
        self.logger = logging.getLogger(__name__)
        self.storage_backend = storage_backend

        # In-memory storage
        self._strategies: Dict[str, StrategyMetadata] = {}
        self._versions: Dict[str, List[StrategyVersion]] = {}
        self._lineage: Dict[str, List[str]] = {}  # parent_id -> [child_ids]
        # Component type registry for validation
        self._component_types = {
            'signal_generator': SignalGenerator,
            'risk_manager': RiskManager,
            'position_sizer': PositionSizer,
            'regime_detector': EnhancedRegimeDetector
        }

        self.logger.info("StrategyRegistry initialized")

    def register_strategy(self, strategy: Strategy, metadata: Dict[str, Any],
                         parent_id: Optional[str] = None) -> str:
        """
        Register a new strategy with metadata
        
        Args:
            strategy: Strategy instance to register
            metadata: Strategy metadata dictionary
            parent_id: Optional parent strategy ID for lineage tracking
            
        Returns:
            Strategy ID
            
        Raises:
            StrategyValidationError: If strategy validation fails
        """
        # Validate strategy first
        self._validate_strategy(strategy)

        # Generate unique ID
        strategy_id = self._generate_strategy_id(strategy.name)

        # Extract component configurations
        component_configs = self._extract_component_configs(strategy)

        # Calculate checksums
        config_hash = self._calculate_config_hash(component_configs, metadata.get('parameters', {}))
        component_hash = self._calculate_component_hash(strategy)

        # Build lineage path
        lineage_path = self._build_lineage_path(parent_id)

        # Create strategy metadata
        strategy_metadata = StrategyMetadata(
            id=strategy_id,
            name=strategy.name,
            version="1.0.0",
            parent_id=parent_id,
            created_at=datetime.now(),
            created_by=metadata.get('created_by', 'system'),
            description=metadata.get('description', ''),
            tags=metadata.get('tags', []),
            status=StrategyStatus(metadata.get('status', 'EXPERIMENTAL')),
            signal_generator_config=component_configs['signal_generator'],
            risk_manager_config=component_configs['risk_manager'],
            position_sizer_config=component_configs['position_sizer'],
            regime_detector_config=component_configs['regime_detector'],
            parameters=metadata.get('parameters', {}),
            performance_summary=None,
            validation_results=None,
            lineage_path=lineage_path,
            branch_name=metadata.get('branch_name'),
            merge_source=metadata.get('merge_source'),
            config_hash=config_hash,
            component_hash=component_hash
        )

        # Store strategy
        self._strategies[strategy_id] = strategy_metadata
        self._versions[strategy_id] = [
            StrategyVersion(
                version="1.0.0",
                strategy_id=strategy_id,
                created_at=datetime.now(),
                changes=["Initial version"],
                performance_delta=None,
                is_major=True,
                signal_generator_config=component_configs['signal_generator'].to_dict(),
                risk_manager_config=component_configs['risk_manager'].to_dict(),
                position_sizer_config=component_configs['position_sizer'].to_dict(),
                regime_detector_config=component_configs['regime_detector'].to_dict(),
                parameters=metadata.get('parameters', {}),
                config_hash=config_hash,
                component_hash=component_hash
            )
        ]

        # Update lineage tracking
        if parent_id:
            if parent_id not in self._lineage:
                self._lineage[parent_id] = []
            self._lineage[parent_id].append(strategy_id)

        # Persist if backend available
        if self.storage_backend:
            self._persist_strategy(strategy_metadata)

        self.logger.info(f"Registered strategy '{strategy.name}' with ID {strategy_id}")
        return strategy_id

    def update_strategy(self, strategy_id: str, strategy: Strategy,
                       changes: list[str], is_major: bool = False) -> str:
        """
        Update an existing strategy with a new version
        
        Args:
            strategy_id: Existing strategy ID
            strategy: Updated strategy instance
            changes: List of changes made
            is_major: Whether this is a major version update
            
        Returns:
            New version string
            
        Raises:
            ValueError: If strategy ID not found
            StrategyValidationError: If strategy validation fails
        """
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy ID {strategy_id} not found")

        # Validate updated strategy
        self._validate_strategy(strategy)

        # Get current metadata
        current_metadata = self._strategies[strategy_id]

        # Calculate new version
        current_version = current_metadata.version
        new_version = self._calculate_next_version(current_version, is_major)

        # Extract updated component configurations
        component_configs = self._extract_component_configs(strategy)

        # Calculate new checksums
        config_hash = self._calculate_config_hash(component_configs, current_metadata.parameters)
        component_hash = self._calculate_component_hash(strategy)

        # Update metadata
        updated_metadata = StrategyMetadata(
            id=strategy_id,
            name=strategy.name,
            version=new_version,
            parent_id=current_metadata.parent_id,
            created_at=datetime.now(),
            created_by=current_metadata.created_by,
            description=current_metadata.description,
            tags=current_metadata.tags,
            status=current_metadata.status,
            signal_generator_config=component_configs['signal_generator'],
            risk_manager_config=component_configs['risk_manager'],
            position_sizer_config=component_configs['position_sizer'],
            regime_detector_config=component_configs['regime_detector'],
            parameters=current_metadata.parameters,
            performance_summary=current_metadata.performance_summary,
            validation_results=None,
            lineage_path=current_metadata.lineage_path,
            branch_name=current_metadata.branch_name,
            merge_source=current_metadata.merge_source,
            config_hash=config_hash,
            component_hash=component_hash
        )

        # Store updated metadata
        self._strategies[strategy_id] = updated_metadata

        # Add version record
        # Add version record with configuration snapshot
        version_record = StrategyVersion(
            version=new_version,
            strategy_id=strategy_id,
            created_at=datetime.now(),
            changes=changes,
            performance_delta=None,
            is_major=is_major,
            signal_generator_config=component_configs['signal_generator'].to_dict(),
            risk_manager_config=component_configs['risk_manager'].to_dict(),
            position_sizer_config=component_configs['position_sizer'].to_dict(),
            regime_detector_config=component_configs['regime_detector'].to_dict(),
            parameters=current_metadata.parameters,
            config_hash=config_hash,
            component_hash=component_hash
        )
        self._versions[strategy_id].append(version_record)

        # Persist if backend available
        if self.storage_backend:
            self._persist_strategy(updated_metadata)

        self.logger.info(f"Updated strategy {strategy_id} to version {new_version}")
        return new_version

    def get_strategy_metadata(self, strategy_id: str) -> Optional[StrategyMetadata]:
        """
        Get strategy metadata by ID
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Strategy metadata or None if not found
        """
        return self._strategies.get(strategy_id)

    def get_strategy_versions(self, strategy_id: str) -> List[StrategyVersion]:
        """
        Get all versions for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            List of strategy versions
        """
        return self._versions.get(strategy_id, [])

    def update_strategy_status(self, strategy_id: str, new_status: StrategyStatus) -> None:
        """
        Update strategy status
        
        Args:
            strategy_id: Strategy ID
            new_status: New status to set
            
        Raises:
            ValueError: If strategy ID not found
        """
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy ID {strategy_id} not found")

        metadata = self._strategies[strategy_id]

        # Create updated metadata with new status
        updated_metadata = StrategyMetadata(
            id=metadata.id,
            name=metadata.name,
            version=metadata.version,
            parent_id=metadata.parent_id,
            created_at=metadata.created_at,
            created_by=metadata.created_by,
            description=metadata.description,
            tags=metadata.tags,
            status=new_status,
            signal_generator_config=metadata.signal_generator_config,
            risk_manager_config=metadata.risk_manager_config,
            position_sizer_config=metadata.position_sizer_config,
            regime_detector_config=metadata.regime_detector_config,
            parameters=metadata.parameters,
            performance_summary=metadata.performance_summary,
            validation_results=metadata.validation_results,
            lineage_path=metadata.lineage_path,
            branch_name=metadata.branch_name,
            merge_source=metadata.merge_source,
            config_hash=metadata.config_hash,
            component_hash=metadata.component_hash
        )

        # Update in-memory storage
        self._strategies[strategy_id] = updated_metadata

        # Persist if backend available
        if self.storage_backend:
            self._persist_strategy(updated_metadata)

        self.logger.info(f"Updated strategy {strategy_id} status to {new_status.value}")

    def revert_to_version(self, strategy_id: str, target_version: str) -> None:
        """
        Revert strategy to a previous version
        
        Args:
            strategy_id: Strategy ID
            target_version: Version to revert to
            
        Raises:
            ValueError: If strategy ID or version not found
        """
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy ID {strategy_id} not found")

        versions = self._versions.get(strategy_id, [])
        target_version_record = None

        for version_record in versions:
            if version_record.version == target_version:
                target_version_record = version_record
                break

        if not target_version_record:
            raise ValueError(f"Version {target_version} not found for strategy {strategy_id}")

        metadata = self._strategies[strategy_id]

        # Create updated metadata with reverted version
        # Verify the version record has configuration snapshot
        if not target_version_record.signal_generator_config:
            raise ValueError(
                f"Version {target_version} does not have configuration snapshot. "
                "Cannot revert to this version."
            )

        metadata = self._strategies[strategy_id]

        # Reconstruct ComponentConfig objects from the version snapshot
        signal_gen_config = ComponentConfig.from_dict(target_version_record.signal_generator_config)
        risk_mgr_config = ComponentConfig.from_dict(target_version_record.risk_manager_config)
        pos_sizer_config = ComponentConfig.from_dict(target_version_record.position_sizer_config)
        regime_det_config = ComponentConfig.from_dict(target_version_record.regime_detector_config)

        # Use parameters from target version if available, otherwise use current
        reverted_parameters = target_version_record.parameters if target_version_record.parameters is not None else metadata.parameters

        # Use hashes from target version, but fall back to current if target version has None
        # (for older version records created before hashes were persisted)
        reverted_config_hash = target_version_record.config_hash or metadata.config_hash
        reverted_component_hash = target_version_record.component_hash or metadata.component_hash

        # Create updated metadata with configuration from target version
        updated_metadata = StrategyMetadata(
            id=metadata.id,
            name=metadata.name,
            version=target_version,
            parent_id=metadata.parent_id,
            created_at=metadata.created_at,
            created_by=metadata.created_by,
            description=metadata.description,
            tags=metadata.tags,
            status=metadata.status,
            signal_generator_config=signal_gen_config,
            risk_manager_config=risk_mgr_config,
            position_sizer_config=pos_sizer_config,
            regime_detector_config=regime_det_config,
            parameters=reverted_parameters,
            performance_summary=None,  # Reset performance summary on revert
            validation_results=None,  # Reset validation on revert
            lineage_path=metadata.lineage_path,
            branch_name=metadata.branch_name,
            merge_source=metadata.merge_source,
            config_hash=reverted_config_hash,
            component_hash=reverted_component_hash
        )

        # Update in-memory storage
        self._strategies[strategy_id] = updated_metadata

        # Persist if backend available
        if self.storage_backend:
            self._persist_strategy(updated_metadata)

        self.logger.info(f"Reverted strategy {strategy_id} to version {target_version}")

    def list_strategies(self, status: Optional[StrategyStatus] = None,
                       tags: Optional[list[str]] = None) -> list[StrategyMetadata]:
        """
        List strategies with optional filtering
        
        Args:
            status: Optional status filter
            tags: Optional tags filter (strategies must have all tags)
            
        Returns:
            List of strategy metadata
        """
        strategies = list(self._strategies.values())

        if status:
            strategies = [s for s in strategies if s.status == status]

        if tags:
            strategies = [s for s in strategies if all(tag in s.tags for tag in tags)]

        return strategies

    def get_strategy_lineage(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get complete lineage information for a strategy
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Lineage information dictionary
        """
        if strategy_id not in self._strategies:
            return {}

        metadata = self._strategies[strategy_id]

        # Get ancestors
        ancestors = []
        current_id = metadata.parent_id
        while current_id and current_id in self._strategies:
            ancestor = self._strategies[current_id]
            ancestors.append({
                'id': ancestor.id,
                'name': ancestor.name,
                'version': ancestor.version,
                'created_at': ancestor.created_at.isoformat()
            })
            current_id = ancestor.parent_id

        # Get descendants
        descendants = self._get_descendants(strategy_id)

        return {
            'strategy_id': strategy_id,
            'lineage_path': metadata.lineage_path,
            'ancestors': list(reversed(ancestors)),  # Root first
            'descendants': descendants,
            'branch_name': metadata.branch_name,
            'merge_source': metadata.merge_source
        }

    def serialize_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Serialize strategy to dictionary
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Serialized strategy data
            
        Raises:
            ValueError: If strategy ID not found
        """
        if strategy_id not in self._strategies:
            raise ValueError(f"Strategy ID {strategy_id} not found")

        metadata = self._strategies[strategy_id]
        versions = self._versions[strategy_id]

        return {
            'metadata': metadata.to_dict(),
            'versions': [v.to_dict() for v in versions],
            'lineage': self.get_strategy_lineage(strategy_id)
        }

    def deserialize_strategy(self, data: Dict[str, Any]) -> str:
        """
        Deserialize strategy from dictionary
        
        Args:
            data: Serialized strategy data
            
        Returns:
            Strategy ID
            
        Raises:
            StrategyValidationError: If deserialization fails
        """
        try:
            # Deserialize metadata
            metadata = StrategyMetadata.from_dict(data['metadata'])

            # Deserialize versions
            versions = [StrategyVersion.from_dict(v) for v in data['versions']]

            # Validate integrity
            self._validate_serialized_data(metadata, versions)

            # Store in registry
            strategy_id = metadata.id
            self._strategies[strategy_id] = metadata
            self._versions[strategy_id] = versions

            # Update lineage if parent exists
            if metadata.parent_id and metadata.parent_id in self._strategies:
                if metadata.parent_id not in self._lineage:
                    self._lineage[metadata.parent_id] = []
                if strategy_id not in self._lineage[metadata.parent_id]:
                    self._lineage[metadata.parent_id].append(strategy_id)

            self.logger.info(f"Deserialized strategy {strategy_id}")
            return strategy_id

        except Exception as e:
            raise StrategyValidationError(f"Failed to deserialize strategy: {e}")

    def validate_strategy_integrity(self, strategy_id: str) -> Dict[str, Any]:
        """
        Validate strategy integrity and consistency
        
        Args:
            strategy_id: Strategy ID
            
        Returns:
            Validation results dictionary
        """
        if strategy_id not in self._strategies:
            return {'valid': False, 'errors': ['Strategy not found']}

        metadata = self._strategies[strategy_id]
        errors = []
        warnings = []

        # Validate component configurations
        try:
            self._validate_component_config(metadata.signal_generator_config, 'signal_generator')
            self._validate_component_config(metadata.risk_manager_config, 'risk_manager')
            self._validate_component_config(metadata.position_sizer_config, 'position_sizer')
            self._validate_component_config(metadata.regime_detector_config, 'regime_detector')
        except Exception as e:
            errors.append(f"Component validation failed: {e}")

        # Validate lineage consistency
        if metadata.parent_id:
            if metadata.parent_id not in self._strategies:
                errors.append(f"Parent strategy {metadata.parent_id} not found")
            else:
                parent_lineage = self._strategies[metadata.parent_id].lineage_path
                expected_lineage = parent_lineage + [metadata.parent_id]
                if metadata.lineage_path != expected_lineage:
                    warnings.append("Lineage path inconsistency detected")

        # Validate version consistency
        versions = self._versions.get(strategy_id, [])
        if not versions:
            errors.append("No versions found for strategy")
        else:
            current_version = versions[-1].version
            if current_version != metadata.version:
                errors.append(f"Version mismatch: metadata={metadata.version}, latest={current_version}")

        # Validate checksums if possible
        # Note: This would require reconstructing the strategy, which we can't do without the actual components

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'validated_at': datetime.now().isoformat()
        }

    def _generate_strategy_id(self, name: str) -> str:
        """Generate unique strategy ID"""
        base_id = f"{name.lower().replace(' ', '_')}_{uuid4().hex[:8]}"
        return base_id

    def _validate_strategy(self, strategy: Strategy) -> None:
        """Validate strategy instance"""
        if not isinstance(strategy, Strategy):
            raise StrategyValidationError("Invalid strategy type")

        if not strategy.name or not strategy.name.strip():
            raise StrategyValidationError("Strategy name cannot be empty")

        # Validate components exist and are correct types
        if not hasattr(strategy, 'signal_generator') or strategy.signal_generator is None:
            raise StrategyValidationError("Strategy missing signal_generator")

        if not hasattr(strategy, 'risk_manager') or strategy.risk_manager is None:
            raise StrategyValidationError("Strategy missing risk_manager")

        if not hasattr(strategy, 'position_sizer') or strategy.position_sizer is None:
            raise StrategyValidationError("Strategy missing position_sizer")

    def _extract_component_configs(self, strategy: Strategy) -> Dict[str, ComponentConfig]:
        """Extract component configurations from strategy"""
        return {
            'signal_generator': ComponentConfig(
                type='signal_generator',
                class_name=strategy.signal_generator.__class__.__name__,
                parameters=strategy.signal_generator.get_parameters(),
                version="1.0.0"
            ),
            'risk_manager': ComponentConfig(
                type='risk_manager',
                class_name=strategy.risk_manager.__class__.__name__,
                parameters=strategy.risk_manager.get_parameters(),
                version="1.0.0"
            ),
            'position_sizer': ComponentConfig(
                type='position_sizer',
                class_name=strategy.position_sizer.__class__.__name__,
                parameters=strategy.position_sizer.get_parameters(),
                version="1.0.0"
            ),
            'regime_detector': ComponentConfig(
                type='regime_detector',
                class_name=strategy.regime_detector.__class__.__name__,
                parameters={},
                version="1.0.0"
            )
        }

    def _calculate_config_hash(self, component_configs: Dict[str, ComponentConfig],
                              parameters: Dict[str, Any]) -> str:
        """Calculate configuration hash for integrity checking"""
        config_data = {
            'components': {k: v.to_dict() for k, v in component_configs.items()},
            'parameters': parameters
        }
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()

    def _calculate_component_hash(self, strategy: Strategy) -> str:
        """Calculate component hash based on component types and parameters"""
        component_data = {
            'signal_generator': {
                'class': strategy.signal_generator.__class__.__name__,
                'params': strategy.signal_generator.get_parameters()
            },
            'risk_manager': {
                'class': strategy.risk_manager.__class__.__name__,
                'params': strategy.risk_manager.get_parameters()
            },
            'position_sizer': {
                'class': strategy.position_sizer.__class__.__name__,
                'params': strategy.position_sizer.get_parameters()
            }
        }
        component_str = json.dumps(component_data, sort_keys=True)
        return hashlib.sha256(component_str.encode()).hexdigest()

    def _build_lineage_path(self, parent_id: Optional[str]) -> List[str]:
        """Build lineage path from root to current strategy"""
        if not parent_id or parent_id not in self._strategies:
            return []

        parent_metadata = self._strategies[parent_id]
        return parent_metadata.lineage_path + [parent_id]

    def _calculate_next_version(self, current_version: str, is_major: bool) -> str:
        """Calculate next version number"""
        parts = current_version.split('.')
        if len(parts) != 3:
            return "1.0.0"

        major, minor, patch = map(int, parts)

        if is_major:
            return f"{major + 1}.0.0"
        else:
            return f"{major}.{minor}.{patch + 1}"

    def _get_descendants(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get all descendants of a strategy"""
        descendants = []

        if strategy_id in self._lineage:
            for child_id in self._lineage[strategy_id]:
                if child_id in self._strategies:
                    child = self._strategies[child_id]
                    descendants.append({
                        'id': child.id,
                        'name': child.name,
                        'version': child.version,
                        'created_at': child.created_at.isoformat(),
                        'descendants': self._get_descendants(child_id)
                    })

        return descendants

    def _validate_component_config(self, config: ComponentConfig, component_type: str) -> None:
        """Validate component configuration"""
        if config.type != component_type:
            raise StrategyValidationError(f"Component type mismatch: expected {component_type}, got {config.type}")

        if not config.class_name:
            raise StrategyValidationError(f"Missing class name for {component_type}")

        if not isinstance(config.parameters, dict):
            raise StrategyValidationError(f"Invalid parameters for {component_type}")

    def _validate_serialized_data(self, metadata: StrategyMetadata, versions: List[StrategyVersion]) -> None:
        """Validate serialized data consistency"""
        if not versions:
            raise StrategyValidationError("No versions provided")

        # Check version consistency
        latest_version = versions[-1]
        if latest_version.version != metadata.version:
            raise StrategyValidationError("Version mismatch between metadata and versions")

        if latest_version.strategy_id != metadata.id:
            raise StrategyValidationError("Strategy ID mismatch between metadata and versions")

    def _persist_strategy(self, metadata: StrategyMetadata) -> None:
        """Persist strategy to storage backend"""
        if self.storage_backend:
            try:
                self.storage_backend.save_strategy(metadata)
            except Exception as e:
                self.logger.error(f"Failed to persist strategy {metadata.id}: {e}")
