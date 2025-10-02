"""
Strategy Manager with Versioning

This module defines the StrategyManager class that orchestrates the component-based
strategy architecture with versioning capabilities for A/B testing and rollbacks.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import pandas as pd

from .signal_generator import Signal, SignalGenerator
from .risk_manager import RiskManager, Position, MarketData
from .position_sizer import PositionSizer
from .regime_context import RegimeContext, EnhancedRegimeDetector


class ValidationGate(Enum):
    """Validation gates for strategy promotion"""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    MINIMUM_TRADES = "minimum_trades"
    DRAWDOWN_LIMIT = "drawdown_limit"
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    MAX_DRAWDOWN = "max_drawdown"


class PromotionStatus(Enum):
    """Status of strategy promotion requests"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    FAILED = "failed"


class RollbackTrigger(Enum):
    """Triggers for automatic strategy rollback"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    EXCESSIVE_DRAWDOWN = "excessive_drawdown"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"
    EMERGENCY = "emergency"


@dataclass
class ValidationResult:
    """Result of strategy validation"""
    gate: ValidationGate
    passed: bool
    value: float
    threshold: float
    message: str


@dataclass
class PromotionRequest:
    """Strategy promotion request"""
    request_id: str
    strategy_id: str
    version_id: str
    requested_by: str
    requested_at: datetime
    from_status: str
    to_status: str
    reason: str
    status: PromotionStatus = PromotionStatus.PENDING
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['from_status'] = self.from_status
        data['to_status'] = self.to_status
        data['requested_at'] = self.requested_at.isoformat()
        data['status'] = self.status.value
        data['approved_at'] = self.approved_at.isoformat() if self.approved_at else None
        data['deployed_at'] = self.deployed_at.isoformat() if self.deployed_at else None
        return data


@dataclass
class RollbackRecord:
    """Record of strategy rollback"""
    rollback_id: str
    strategy_id: str
    from_version: str
    to_version: str
    trigger: RollbackTrigger
    triggered_at: datetime
    triggered_by: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['trigger'] = self.trigger.value
        data['triggered_at'] = self.triggered_at.isoformat()
        return data


@dataclass
class StrategyVersion:
    """
    Strategy version information
    
    Attributes:
        version_id: Unique identifier for this version
        name: Human-readable version name
        description: Description of changes in this version
        created_at: When this version was created
        components: Dictionary of component configurations
        parameters: Strategy-level parameters
        is_active: Whether this version is currently active
        performance_metrics: Performance data for this version
    """
    version_id: str
    name: str
    description: str
    created_at: datetime
    components: dict[str, dict[str, Any]]
    parameters: dict[str, Any]
    is_active: bool = False
    performance_metrics: Optional[dict[str, float]] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'StrategyVersion':
        """Create from dictionary"""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class StrategyExecution:
    """
    Record of strategy execution
    
    Attributes:
        timestamp: When the execution occurred
        signal: Generated signal
        regime: Market regime context
        position_size: Calculated position size
        risk_metrics: Risk-related metrics
        execution_time_ms: Time taken for execution
        version_id: Strategy version used
    """
    timestamp: datetime
    signal: Signal
    regime: Optional[RegimeContext]
    position_size: float
    risk_metrics: dict[str, float]
    execution_time_ms: float
    version_id: str


class StrategyManager:
    """
    Component-based strategy manager with versioning
    
    Orchestrates signal generation, risk management, and position sizing
    components while maintaining version history for A/B testing and rollbacks.
    """
    
    def __init__(self, name: str, signal_generator: SignalGenerator,
                 risk_manager: RiskManager, position_sizer: PositionSizer,
                 regime_detector: Optional[EnhancedRegimeDetector] = None):
        """
        Initialize strategy manager
        
        Args:
            name: Strategy name
            signal_generator: Component for generating trading signals
            risk_manager: Component for risk management
            position_sizer: Component for position sizing
            regime_detector: Optional regime detection component
        """
        self.name = name
        self.logger = logging.getLogger(f"StrategyManager.{name}")
        
        # Core components
        self.signal_generator = signal_generator
        self.risk_manager = risk_manager
        self.position_sizer = position_sizer
        self.regime_detector = regime_detector or EnhancedRegimeDetector()
        
        # Version management
        self.versions: dict[str, StrategyVersion] = {}
        self.current_version_id: Optional[str] = None
        self.execution_history: list[StrategyExecution] = []
        
        # Performance tracking
        self.performance_metrics: dict[str, float] = {}
        
        # Create initial version
        self._create_initial_version()
    
    def execute_strategy(self, df: pd.DataFrame, index: int, balance: float,
                        current_positions: Optional[list[Position]] = None) -> tuple[Signal, float, dict[str, Any]]:
        """
        Execute complete strategy pipeline
        
        Args:
            df: DataFrame with OHLCV data and indicators
            index: Current index position
            balance: Available account balance
            current_positions: List of current positions
            
        Returns:
            Tuple of (signal, position_size, execution_metadata)
        """
        start_time = datetime.now()
        
        try:
            # Detect current market regime
            regime = self.regime_detector.detect_regime(df, index)
            
            # Generate trading signal
            signal = self.signal_generator.generate_signal(df, index, regime)
            
            # Calculate position size using risk manager
            risk_position_size = self.risk_manager.calculate_position_size(signal, balance, regime)
            
            # Allow position sizer to further adjust the risk manager's position size
            # Position sizer gets the risk manager's position as the "risk amount"
            position_size = self.position_sizer.calculate_size(signal, balance, risk_position_size, regime)
            
            # Final validation to ensure position size is within reasonable bounds
            position_size = self._validate_position_size(position_size, signal, balance, regime)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Create execution metadata
            metadata = {
                'regime': regime,
                'risk_position_size': risk_position_size,
                'execution_time_ms': execution_time,
                'version_id': self.current_version_id,
                'signal_confidence': signal.confidence,
                'signal_strength': signal.strength,
                'regime_confidence': regime.confidence if regime else 0.0,
                'components': {
                    'signal_generator': self.signal_generator.name,
                    'risk_manager': self.risk_manager.name,
                    'position_sizer': self.position_sizer.name
                }
            }
            
            # Record execution
            execution = StrategyExecution(
                timestamp=start_time,
                signal=signal,
                regime=regime,
                position_size=position_size,
                risk_metrics={'risk_position_size': risk_position_size},
                execution_time_ms=execution_time,
                version_id=self.current_version_id or 'unknown'
            )
            self.execution_history.append(execution)
            
            # Limit execution history
            if len(self.execution_history) > 10000:
                self.execution_history = self.execution_history[-5000:]
            
            self.logger.info(f"Strategy executed: {signal.direction.value} signal with size {position_size:.2f}")
            
            return signal, position_size, metadata
            
        except Exception as e:
            self.logger.error(f"Strategy execution failed: {e}")
            # Return safe defaults
            from .signal_generator import SignalDirection
            safe_signal = Signal(
                direction=SignalDirection.HOLD,
                strength=0.0,
                confidence=0.0,
                metadata={'error': str(e)}
            )
            return safe_signal, 0.0, {'error': str(e)}
    
    def create_version(self, name: str, description: str, 
                      signal_generator: Optional[SignalGenerator] = None,
                      risk_manager: Optional[RiskManager] = None,
                      position_sizer: Optional[PositionSizer] = None,
                      parameters: Optional[dict[str, Any]] = None) -> str:
        """
        Create a new strategy version
        
        Args:
            name: Version name
            description: Description of changes
            signal_generator: New signal generator (optional)
            risk_manager: New risk manager (optional)
            position_sizer: New position sizer (optional)
            parameters: Strategy parameters (optional)
            
        Returns:
            Version ID of the created version
        """
        version_id = str(uuid4())
        
        # Use provided components or keep current ones
        components = {
            'signal_generator': (signal_generator or self.signal_generator).get_parameters(),
            'risk_manager': (risk_manager or self.risk_manager).get_parameters(),
            'position_sizer': (position_sizer or self.position_sizer).get_parameters(),
            'regime_detector': {'type': 'EnhancedRegimeDetector'}
        }
        
        version = StrategyVersion(
            version_id=version_id,
            name=name,
            description=description,
            created_at=datetime.now(),
            components=components,
            parameters=parameters or {},
            is_active=False
        )
        
        self.versions[version_id] = version
        
        self.logger.info(f"Created strategy version {name} ({version_id})")
        
        return version_id
    
    def activate_version(self, version_id: str, 
                        signal_generator: Optional[SignalGenerator] = None,
                        risk_manager: Optional[RiskManager] = None,
                        position_sizer: Optional[PositionSizer] = None) -> bool:
        """
        Activate a specific strategy version
        
        Args:
            version_id: Version to activate
            signal_generator: Signal generator instance for this version
            risk_manager: Risk manager instance for this version
            position_sizer: Position sizer instance for this version
            
        Returns:
            True if activation successful, False otherwise
        """
        if version_id not in self.versions:
            self.logger.error(f"Version {version_id} not found")
            return False
        
        try:
            # Deactivate current version
            if self.current_version_id:
                self.versions[self.current_version_id].is_active = False
            
            # Update components if provided
            if signal_generator:
                self.signal_generator = signal_generator
            if risk_manager:
                self.risk_manager = risk_manager
            if position_sizer:
                self.position_sizer = position_sizer
            
            # Activate new version
            self.versions[version_id].is_active = True
            self.current_version_id = version_id
            
            self.logger.info(f"Activated strategy version {version_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to activate version {version_id}: {e}")
            return False
    
    def rollback_to_version(self, version_id: str) -> bool:
        """
        Rollback to a previous strategy version
        
        Args:
            version_id: Version to rollback to
            
        Returns:
            True if rollback successful, False otherwise
        """
        if version_id not in self.versions:
            self.logger.error(f"Cannot rollback: version {version_id} not found")
            return False
        
        # Note: This is a simplified rollback that only changes the active version
        # In a full implementation, you would need to recreate component instances
        # based on the stored configuration
        
        success = self.activate_version(version_id)
        
        if success:
            self.logger.info(f"Rolled back to strategy version {version_id}")
        
        return success
    
    def get_version_performance(self, version_id: str, 
                              lookback_executions: int = 100) -> dict[str, float]:
        """
        Get performance metrics for a specific version
        
        Args:
            version_id: Version to analyze
            lookback_executions: Number of recent executions to analyze
            
        Returns:
            Dictionary of performance metrics
        """
        if version_id not in self.versions:
            return {}
        
        # Filter executions for this version
        version_executions = [
            exec for exec in self.execution_history[-lookback_executions:]
            if exec.version_id == version_id
        ]
        
        if not version_executions:
            return {}
        
        # Calculate basic metrics
        total_executions = len(version_executions)
        avg_execution_time = sum(exec.execution_time_ms for exec in version_executions) / total_executions
        
        # Signal distribution
        signal_counts = {}
        confidence_sum = 0.0
        strength_sum = 0.0
        
        for exec in version_executions:
            direction = exec.signal.direction.value
            signal_counts[direction] = signal_counts.get(direction, 0) + 1
            confidence_sum += exec.signal.confidence
            strength_sum += exec.signal.strength
        
        metrics = {
            'total_executions': total_executions,
            'avg_execution_time_ms': avg_execution_time,
            'avg_signal_confidence': confidence_sum / total_executions,
            'avg_signal_strength': strength_sum / total_executions,
        }
        
        # Add signal distribution
        for direction, count in signal_counts.items():
            metrics[f'{direction}_signals_pct'] = (count / total_executions) * 100
        
        # Update version performance metrics
        self.versions[version_id].performance_metrics = metrics
        
        return metrics
    
    def compare_versions(self, version_ids: list[str], 
                        metric: str = 'avg_signal_confidence') -> dict[str, float]:
        """
        Compare performance metrics across versions
        
        Args:
            version_ids: List of version IDs to compare
            metric: Metric to compare
            
        Returns:
            Dictionary mapping version IDs to metric values
        """
        comparison = {}
        
        for version_id in version_ids:
            if version_id in self.versions:
                performance = self.get_version_performance(version_id)
                comparison[version_id] = performance.get(metric, 0.0)
        
        return comparison
    
    def get_execution_statistics(self, lookback_hours: int = 24) -> dict[str, Any]:
        """
        Get execution statistics for recent period
        
        Args:
            lookback_hours: Hours to look back
            
        Returns:
            Dictionary of execution statistics
        """
        cutoff_time = datetime.now() - pd.Timedelta(hours=lookback_hours)
        
        recent_executions = [
            exec for exec in self.execution_history
            if exec.timestamp >= cutoff_time
        ]
        
        if not recent_executions:
            return {}
        
        # Calculate statistics
        total_executions = len(recent_executions)
        avg_execution_time = sum(exec.execution_time_ms for exec in recent_executions) / total_executions
        
        # Signal analysis
        signals_by_direction = {}
        position_sizes = []
        
        for exec in recent_executions:
            direction = exec.signal.direction.value
            signals_by_direction[direction] = signals_by_direction.get(direction, 0) + 1
            position_sizes.append(exec.position_size)
        
        stats = {
            'period_hours': lookback_hours,
            'total_executions': total_executions,
            'executions_per_hour': total_executions / lookback_hours,
            'avg_execution_time_ms': avg_execution_time,
            'avg_position_size': sum(position_sizes) / len(position_sizes) if position_sizes else 0,
            'max_position_size': max(position_sizes) if position_sizes else 0,
            'signal_distribution': signals_by_direction,
            'current_version': self.current_version_id
        }
        
        return stats
    
    def _create_initial_version(self) -> None:
        """Create initial strategy version"""
        version_id = self.create_version(
            name="Initial Version",
            description="Initial strategy configuration",
            parameters={'created_by': 'StrategyManager'}
        )
        self.activate_version(version_id)
    
    def _calculate_risk_amount(self, balance: float, signal: Signal, 
                             regime: Optional[RegimeContext]) -> float:
        """
        Calculate risk amount based on signal and regime
        
        DEPRECATED: This method is deprecated in favor of using RiskManager.calculate_position_size()
        directly. It's kept for backward compatibility but should not be used in new code.
        """
        # Base risk percentage (could be configurable)
        base_risk_pct = 0.02  # 2%
        
        # Adjust for signal confidence
        confidence_adj = max(0.5, signal.confidence)
        
        # Adjust for regime if available
        regime_adj = 1.0
        if regime:
            regime_adj = regime.get_risk_multiplier()
        
        # Calculate final risk amount
        risk_amount = balance * base_risk_pct * confidence_adj * regime_adj
        
        return max(balance * 0.001, min(balance * 0.1, risk_amount))  # 0.1% to 10%
    
    def _validate_position_size(self, position_size: float, signal: Signal, 
                              balance: float, regime: Optional[RegimeContext]) -> float:
        """Validate and adjust position size using risk manager"""
        # For now, just ensure position size is reasonable
        max_position = balance * 0.2  # Maximum 20% of balance
        min_position = balance * 0.001  # Minimum 0.1% of balance
        
        if signal.direction.value == 'hold':
            return 0.0
        
        # Respect position sizer's decision to return 0.0 (no trade)
        if position_size == 0.0:
            return 0.0
        
        # Only apply minimum bound when position sizer produced a positive size
        return max(min_position, min(max_position, position_size))
    
    def get_current_version(self) -> Optional[StrategyVersion]:
        """Get currently active version"""
        if self.current_version_id:
            return self.versions.get(self.current_version_id)
        return None
    
    def list_versions(self) -> list[StrategyVersion]:
        """Get list of all versions"""
        return list(self.versions.values())
    
    def export_version(self, version_id: str) -> Optional[dict[str, Any]]:
        """Export version configuration for backup/sharing"""
        if version_id not in self.versions:
            return None
        
        return self.versions[version_id].to_dict()
    
    def import_version(self, version_data: dict[str, Any]) -> Optional[str]:
        """Import version configuration from backup"""
        try:
            version = StrategyVersion.from_dict(version_data)
            # Generate new ID to avoid conflicts
            version.version_id = str(uuid4())
            version.is_active = False
            
            self.versions[version.version_id] = version
            
            self.logger.info(f"Imported strategy version {version.name} ({version.version_id})")
            
            return version.version_id
            
        except Exception as e:
            self.logger.error(f"Failed to import version: {e}")
            return None