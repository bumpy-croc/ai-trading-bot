"""
Migration Rollback Manager

This module provides comprehensive rollback capabilities for strategy migration,
including safe rollback to legacy systems, validation procedures, impact analysis,
and emergency rollback procedures for production issues.
"""

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.strategies.base import BaseStrategy
from src.strategies.adapters.legacy_adapter import LegacyStrategyAdapter


@dataclass
class RollbackPoint:
    """
    Snapshot of system state for rollback purposes
    
    Attributes:
        rollback_id: Unique identifier for the rollback point
        timestamp: When the rollback point was created
        strategy_name: Name of the strategy
        legacy_strategy_config: Configuration of the legacy strategy
        converted_strategy_config: Configuration of the converted strategy
        migration_metadata: Metadata about the migration
        system_state: System state at the time of rollback point creation
        validation_results: Validation results at the time of creation
        file_backups: Paths to backed up files
        database_state: Database state information
    """
    rollback_id: str
    timestamp: datetime
    strategy_name: str
    legacy_strategy_config: Dict[str, Any]
    converted_strategy_config: Dict[str, Any]
    migration_metadata: Dict[str, Any]
    system_state: Dict[str, Any]
    validation_results: Dict[str, Any]
    file_backups: Dict[str, str]
    database_state: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "rollback_id": self.rollback_id,
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "legacy_strategy_config": self.legacy_strategy_config,
            "converted_strategy_config": self.converted_strategy_config,
            "migration_metadata": self.migration_metadata,
            "system_state": self.system_state,
            "validation_results": self.validation_results,
            "file_backups": self.file_backups,
            "database_state": self.database_state
        }


@dataclass
class RollbackResult:
    """
    Result of a rollback operation
    
    Attributes:
        rollback_id: ID of the rollback point used
        success: Whether the rollback was successful
        timestamp: When the rollback was performed
        strategy_name: Name of the strategy rolled back
        actions_performed: List of actions performed during rollback
        files_restored: List of files that were restored
        validation_results: Results of post-rollback validation
        warnings: List of warnings encountered
        errors: List of errors encountered
        impact_analysis: Analysis of rollback impact
    """
    rollback_id: str
    success: bool
    timestamp: datetime
    strategy_name: str
    actions_performed: List[str]
    files_restored: List[str]
    validation_results: Dict[str, Any]
    warnings: List[str]
    errors: List[str]
    impact_analysis: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "rollback_id": self.rollback_id,
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "strategy_name": self.strategy_name,
            "actions_performed": self.actions_performed,
            "files_restored": self.files_restored,
            "validation_results": self.validation_results,
            "warnings": self.warnings,
            "errors": self.errors,
            "impact_analysis": self.impact_analysis
        }


class RollbackManager:
    """
    Comprehensive rollback manager for strategy migration
    
    This class provides safe rollback capabilities including system state snapshots,
    validation procedures, impact analysis, and emergency rollback procedures.
    """

    def __init__(self, backup_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the rollback manager
        
        Args:
            backup_dir: Directory for storing rollback data (default: ./rollback_backups)
        """
        self.logger = logging.getLogger("RollbackManager")

        # Set up backup directory
        self.backup_dir = Path(backup_dir) if backup_dir else Path("rollback_backups")
        self.backup_dir.mkdir(exist_ok=True)

        # Rollback points storage
        self.rollback_points: Dict[str, RollbackPoint] = {}
        self.rollback_history: List[RollbackResult] = []

        # Load existing rollback points
        self._load_rollback_points()

        self.logger.info(f"RollbackManager initialized with backup directory: {self.backup_dir}")

    def create_rollback_point(self, strategy_name: str,
                            legacy_strategy: BaseStrategy,
                            converted_strategy: Optional[LegacyStrategyAdapter] = None,
                            migration_metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a rollback point before migration
        
        Args:
            strategy_name: Name of the strategy
            legacy_strategy: Legacy strategy instance
            converted_strategy: Converted strategy instance (optional)
            migration_metadata: Additional migration metadata
            
        Returns:
            Rollback point ID
        """
        timestamp = datetime.now()
        rollback_id = f"rollback_{strategy_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Creating rollback point: {rollback_id}")

        try:
            # Collect legacy strategy configuration
            legacy_config = self._collect_strategy_config(legacy_strategy)

            # Collect converted strategy configuration if available
            converted_config = {}
            if converted_strategy:
                converted_config = self._collect_strategy_config(converted_strategy)

            # Collect system state
            system_state = self._collect_system_state()

            # Perform validation
            validation_results = self._validate_current_state(legacy_strategy, converted_strategy)

            # Create file backups
            file_backups = self._create_file_backups(rollback_id, strategy_name)

            # Collect database state
            database_state = self._collect_database_state(strategy_name)

            # Create rollback point
            rollback_point = RollbackPoint(
                rollback_id=rollback_id,
                timestamp=timestamp,
                strategy_name=strategy_name,
                legacy_strategy_config=legacy_config,
                converted_strategy_config=converted_config,
                migration_metadata=migration_metadata or {},
                system_state=system_state,
                validation_results=validation_results,
                file_backups=file_backups,
                database_state=database_state
            )

            # Store rollback point
            self.rollback_points[rollback_id] = rollback_point

            # Save to persistent storage
            self._save_rollback_point(rollback_point)

            self.logger.info(f"Rollback point created successfully: {rollback_id}")

            return rollback_id

        except Exception as e:
            self.logger.error(f"Failed to create rollback point: {e}")
            raise

    def perform_rollback(self, rollback_id: str,
                        validate_before_rollback: bool = True,
                        force_rollback: bool = False) -> RollbackResult:
        """
        Perform rollback to a specific rollback point
        
        Args:
            rollback_id: ID of the rollback point
            validate_before_rollback: Whether to validate before performing rollback
            force_rollback: Whether to force rollback even if validation fails
            
        Returns:
            RollbackResult with detailed results
        """
        start_time = datetime.now()

        self.logger.info(f"Starting rollback to: {rollback_id}")

        # Initialize result
        result = RollbackResult(
            rollback_id=rollback_id,
            success=False,
            timestamp=start_time,
            strategy_name="",
            actions_performed=[],
            files_restored=[],
            validation_results={},
            warnings=[],
            errors=[],
            impact_analysis={}
        )

        try:
            # Check if rollback point exists
            if rollback_id not in self.rollback_points:
                error_msg = f"Rollback point not found: {rollback_id}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)
                return result

            rollback_point = self.rollback_points[rollback_id]
            result.strategy_name = rollback_point.strategy_name

            # Pre-rollback validation
            if validate_before_rollback:
                validation_results = self._validate_rollback_feasibility(rollback_point)
                result.validation_results.update(validation_results)

                if not validation_results.get("feasible", False) and not force_rollback:
                    error_msg = "Rollback validation failed and force_rollback is False"
                    result.errors.append(error_msg)
                    self.logger.error(error_msg)
                    return result

                if validation_results.get("warnings"):
                    result.warnings.extend(validation_results["warnings"])

            # Perform rollback actions
            self._perform_rollback_actions(rollback_point, result)

            # Post-rollback validation
            post_validation = self._validate_post_rollback(rollback_point)
            result.validation_results.update(post_validation)

            # Impact analysis
            result.impact_analysis = self._analyze_rollback_impact(rollback_point, result)

            # Determine success
            result.success = len(result.errors) == 0

            # Store in history
            self.rollback_history.append(result)

            if result.success:
                self.logger.info(f"Rollback completed successfully: {rollback_id}")
            else:
                self.logger.error(f"Rollback completed with errors: {rollback_id}")

            return result

        except Exception as e:
            error_msg = f"Rollback failed with exception: {e}"
            result.errors.append(error_msg)
            self.logger.error(error_msg, exc_info=True)
            return result

    def emergency_rollback(self, strategy_name: str) -> RollbackResult:
        """
        Perform emergency rollback using the most recent rollback point
        
        Args:
            strategy_name: Name of the strategy to rollback
            
        Returns:
            RollbackResult with detailed results
        """
        self.logger.warning(f"Performing emergency rollback for strategy: {strategy_name}")

        # Find most recent rollback point for the strategy
        strategy_rollbacks = [
            rp for rp in self.rollback_points.values()
            if rp.strategy_name == strategy_name
        ]

        if not strategy_rollbacks:
            # Create error result
            return RollbackResult(
                rollback_id="",
                success=False,
                timestamp=datetime.now(),
                strategy_name=strategy_name,
                actions_performed=[],
                files_restored=[],
                validation_results={},
                warnings=[],
                errors=[f"No rollback points found for strategy: {strategy_name}"],
                impact_analysis={}
            )

        # Get most recent rollback point
        most_recent = max(strategy_rollbacks, key=lambda rp: rp.timestamp)

        # Perform rollback with minimal validation (emergency mode)
        return self.perform_rollback(
            most_recent.rollback_id,
            validate_before_rollback=False,
            force_rollback=True
        )

    def list_rollback_points(self, strategy_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available rollback points
        
        Args:
            strategy_name: Filter by strategy name (optional)
            
        Returns:
            List of rollback point summaries
        """
        rollback_points = list(self.rollback_points.values())

        if strategy_name:
            rollback_points = [rp for rp in rollback_points if rp.strategy_name == strategy_name]

        # Sort by timestamp (most recent first)
        rollback_points.sort(key=lambda rp: rp.timestamp, reverse=True)

        return [
            {
                "rollback_id": rp.rollback_id,
                "timestamp": rp.timestamp.isoformat(),
                "strategy_name": rp.strategy_name,
                "has_converted_strategy": bool(rp.converted_strategy_config),
                "file_backup_count": len(rp.file_backups),
                "validation_status": rp.validation_results.get("overall_status", "unknown")
            }
            for rp in rollback_points
        ]

    def delete_rollback_point(self, rollback_id: str) -> bool:
        """
        Delete a rollback point and its associated backups
        
        Args:
            rollback_id: ID of the rollback point to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if rollback_id not in self.rollback_points:
                self.logger.warning(f"Rollback point not found: {rollback_id}")
                return False

            rollback_point = self.rollback_points[rollback_id]

            # Delete file backups
            for backup_path in rollback_point.file_backups.values():
                try:
                    backup_file = Path(backup_path)
                    if backup_file.exists():
                        backup_file.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete backup file {backup_path}: {e}")

            # Delete rollback point directory
            rollback_dir = self.backup_dir / rollback_id
            if rollback_dir.exists():
                shutil.rmtree(rollback_dir)

            # Remove from memory
            del self.rollback_points[rollback_id]

            self.logger.info(f"Rollback point deleted: {rollback_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete rollback point {rollback_id}: {e}")
            return False

    def cleanup_old_rollback_points(self, days_to_keep: int = 30) -> int:
        """
        Clean up rollback points older than specified days
        
        Args:
            days_to_keep: Number of days to keep rollback points
            
        Returns:
            Number of rollback points deleted
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=days_to_keep)

        old_rollback_ids = [
            rp.rollback_id for rp in self.rollback_points.values()
            if rp.timestamp < cutoff_date
        ]

        deleted_count = 0
        for rollback_id in old_rollback_ids:
            if self.delete_rollback_point(rollback_id):
                deleted_count += 1

        self.logger.info(f"Cleaned up {deleted_count} old rollback points")
        return deleted_count

    def _collect_strategy_config(self, strategy: Union[BaseStrategy, LegacyStrategyAdapter]) -> Dict[str, Any]:
        """Collect strategy configuration"""
        config = {}

        try:
            # Basic strategy information
            config["class_name"] = strategy.__class__.__name__
            config["name"] = strategy.name
            config["trading_pair"] = strategy.get_trading_pair()

            # Strategy parameters
            if hasattr(strategy, "get_parameters"):
                config["parameters"] = strategy.get_parameters()

            # Component information for adapted strategies
            if isinstance(strategy, LegacyStrategyAdapter):
                config["component_status"] = strategy.get_component_status()
                config["performance_metrics"] = strategy.get_performance_metrics()

            # Strategy-specific attributes
            common_attrs = [
                "model_path", "sequence_length", "stop_loss_pct", "take_profit_pct",
                "use_prediction_engine", "model_name"
            ]

            for attr in common_attrs:
                if hasattr(strategy, attr):
                    config[attr] = getattr(strategy, attr)

        except Exception as e:
            config["error"] = f"Failed to collect strategy config: {e}"
            self.logger.warning(f"Error collecting strategy config: {e}")

        return config

    def _collect_system_state(self) -> Dict[str, Any]:
        """Collect current system state"""
        return {
            "timestamp": datetime.now().isoformat(),
            "python_version": f"{pd.__version__}",  # Using pandas version as proxy
            "working_directory": str(Path.cwd()),
            "environment_variables": {
                # Only collect non-sensitive environment variables
                "PATH": str(Path.cwd()),  # Simplified for security
            }
        }

    def _validate_current_state(self, legacy_strategy: BaseStrategy,
                              converted_strategy: Optional[LegacyStrategyAdapter]) -> Dict[str, Any]:
        """Validate current system state"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "legacy_strategy_valid": False,
            "converted_strategy_valid": False,
            "overall_status": "unknown"
        }

        try:
            # Validate legacy strategy
            if hasattr(legacy_strategy, "get_parameters"):
                legacy_params = legacy_strategy.get_parameters()
                validation["legacy_strategy_valid"] = isinstance(legacy_params, dict)
            else:
                validation["legacy_strategy_valid"] = True  # Basic validation

            # Validate converted strategy
            if converted_strategy:
                try:
                    component_status = converted_strategy.get_component_status()
                    validation["converted_strategy_valid"] = isinstance(component_status, dict)
                except Exception:
                    validation["converted_strategy_valid"] = False
            else:
                validation["converted_strategy_valid"] = True  # No converted strategy yet

            # Overall status
            if validation["legacy_strategy_valid"] and validation["converted_strategy_valid"]:
                validation["overall_status"] = "valid"
            else:
                validation["overall_status"] = "invalid"

        except Exception as e:
            validation["error"] = str(e)
            validation["overall_status"] = "error"

        return validation

    def _create_file_backups(self, rollback_id: str, strategy_name: str) -> Dict[str, str]:
        """Create backups of important files"""
        backups = {}

        # Create rollback-specific directory
        rollback_dir = self.backup_dir / rollback_id
        rollback_dir.mkdir(exist_ok=True)

        # Files to backup (if they exist)
        files_to_backup = [
            f"src/strategies/{strategy_name.lower()}.py",
            f"src/strategies/{strategy_name.lower()}_adaptive.py",
            f"src/strategies/{strategy_name.lower()}_basic.py",
            "src/strategies/base.py",
            "src/config/config_manager.py"
        ]

        for file_path in files_to_backup:
            source_path = Path(file_path)
            if source_path.exists():
                try:
                    backup_path = rollback_dir / source_path.name
                    shutil.copy2(source_path, backup_path)
                    backups[str(source_path)] = str(backup_path)
                except Exception as e:
                    self.logger.warning(f"Failed to backup {file_path}: {e}")

        return backups

    def _collect_database_state(self, strategy_name: str) -> Dict[str, Any]:
        """Collect database state information"""
        # This is a simplified implementation
        # In a real system, you might want to backup database schemas,
        # configuration tables, etc.
        return {
            "timestamp": datetime.now().isoformat(),
            "strategy_name": strategy_name,
            "note": "Database state collection not implemented in this example"
        }

    def _validate_rollback_feasibility(self, rollback_point: RollbackPoint) -> Dict[str, Any]:
        """Validate if rollback is feasible"""
        validation = {
            "feasible": True,
            "warnings": [],
            "errors": [],
            "checks_performed": []
        }

        # Check if backup files exist
        missing_backups = []
        for original_path, backup_path in rollback_point.file_backups.items():
            if not Path(backup_path).exists():
                missing_backups.append(backup_path)

        if missing_backups:
            validation["warnings"].append(f"Missing backup files: {missing_backups}")

        validation["checks_performed"].append("backup_file_existence")

        # Check system compatibility
        current_time = datetime.now()
        rollback_age = current_time - rollback_point.timestamp

        if rollback_age.days > 30:
            validation["warnings"].append(f"Rollback point is {rollback_age.days} days old")

        validation["checks_performed"].append("rollback_age")

        # Check if any critical errors exist
        if validation["errors"]:
            validation["feasible"] = False

        return validation

    def _perform_rollback_actions(self, rollback_point: RollbackPoint, result: RollbackResult) -> None:
        """Perform the actual rollback actions"""

        # Restore files
        for original_path, backup_path in rollback_point.file_backups.items():
            try:
                if Path(backup_path).exists():
                    shutil.copy2(backup_path, original_path)
                    result.files_restored.append(original_path)
                    result.actions_performed.append(f"Restored file: {original_path}")
                else:
                    result.warnings.append(f"Backup file not found: {backup_path}")
            except Exception as e:
                error_msg = f"Failed to restore {original_path}: {e}"
                result.errors.append(error_msg)
                self.logger.error(error_msg)

        # Additional rollback actions could be added here
        # For example: database rollbacks, configuration resets, etc.

        result.actions_performed.append("File restoration completed")

    def _validate_post_rollback(self, rollback_point: RollbackPoint) -> Dict[str, Any]:
        """Validate system state after rollback"""
        validation = {
            "timestamp": datetime.now().isoformat(),
            "files_validated": [],
            "validation_errors": [],
            "overall_status": "unknown"
        }

        # Validate restored files
        for original_path in rollback_point.file_backups.keys():
            if Path(original_path).exists():
                validation["files_validated"].append(original_path)
            else:
                validation["validation_errors"].append(f"File not found after rollback: {original_path}")

        # Overall status
        if validation["validation_errors"]:
            validation["overall_status"] = "errors"
        else:
            validation["overall_status"] = "success"

        return validation

    def _analyze_rollback_impact(self, rollback_point: RollbackPoint, result: RollbackResult) -> Dict[str, Any]:
        """Analyze the impact of the rollback"""
        return {
            "files_affected": len(result.files_restored),
            "actions_performed": len(result.actions_performed),
            "warnings_count": len(result.warnings),
            "errors_count": len(result.errors),
            "rollback_duration": (datetime.now() - result.timestamp).total_seconds(),
            "strategy_reverted_to": rollback_point.legacy_strategy_config.get("class_name", "unknown"),
            "migration_metadata": rollback_point.migration_metadata
        }

    def _save_rollback_point(self, rollback_point: RollbackPoint) -> None:
        """Save rollback point to persistent storage"""
        try:
            rollback_file = self.backup_dir / f"{rollback_point.rollback_id}.json"

            with open(rollback_file, "w") as f:
                json.dump(rollback_point.to_dict(), f, indent=2, default=str)

        except Exception as e:
            self.logger.error(f"Failed to save rollback point {rollback_point.rollback_id}: {e}")

    def _load_rollback_points(self) -> None:
        """Load rollback points from persistent storage"""
        try:
            for rollback_file in self.backup_dir.glob("rollback_*.json"):
                try:
                    with open(rollback_file) as f:
                        data = json.load(f)

                    # Convert timestamp back to datetime
                    data["timestamp"] = datetime.fromisoformat(data["timestamp"])

                    rollback_point = RollbackPoint(**data)
                    self.rollback_points[rollback_point.rollback_id] = rollback_point

                except Exception as e:
                    self.logger.warning(f"Failed to load rollback point from {rollback_file}: {e}")

            self.logger.info(f"Loaded {len(self.rollback_points)} rollback points")

        except Exception as e:
            self.logger.warning(f"Failed to load rollback points: {e}")

    def get_rollback_history(self) -> List[RollbackResult]:
        """Get history of rollback operations"""
        return self.rollback_history.copy()

    def clear_rollback_history(self) -> None:
        """Clear rollback operation history"""
        self.rollback_history.clear()
        self.logger.info("Rollback history cleared")

    def generate_rollback_report(self) -> Dict[str, Any]:
        """Generate comprehensive rollback report"""
        return {
            "total_rollback_points": len(self.rollback_points),
            "total_rollback_operations": len(self.rollback_history),
            "successful_rollbacks": sum(1 for r in self.rollback_history if r.success),
            "failed_rollbacks": sum(1 for r in self.rollback_history if not r.success),
            "rollback_points_by_strategy": self._group_rollback_points_by_strategy(),
            "recent_rollback_operations": [
                r.to_dict() for r in sorted(self.rollback_history, key=lambda x: x.timestamp, reverse=True)[:10]
            ],
            "backup_directory_size": self._calculate_backup_directory_size(),
            "oldest_rollback_point": min(
                (rp.timestamp for rp in self.rollback_points.values()),
                default=None
            ).isoformat() if self.rollback_points else None,
            "newest_rollback_point": max(
                (rp.timestamp for rp in self.rollback_points.values()),
                default=None
            ).isoformat() if self.rollback_points else None
        }

    def _group_rollback_points_by_strategy(self) -> Dict[str, int]:
        """Group rollback points by strategy name"""
        strategy_counts = {}
        for rollback_point in self.rollback_points.values():
            strategy_name = rollback_point.strategy_name
            strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
        return strategy_counts

    def _calculate_backup_directory_size(self) -> Dict[str, Any]:
        """Calculate backup directory size"""
        try:
            total_size = 0
            file_count = 0

            for file_path in self.backup_dir.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1

            return {
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "file_count": file_count
            }
        except Exception as e:
            return {"error": str(e)}
