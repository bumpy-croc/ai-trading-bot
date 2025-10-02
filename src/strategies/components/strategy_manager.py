"""
Strategy Manager with Versioning and Performance Tracking

This module implements the StrategyManager that orchestrates strategy registry,
performance tracking, lineage management, and provides promotion/rollback
capabilities with comprehensive validation and safety mechanisms.
"""

import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .performance_tracker import PerformanceTracker, TradeResult
from .strategy import Strategy
from .strategy_lineage import ChangeType, ImpactLevel, StrategyLineageTracker
from .strategy_registry import StrategyRegistry, StrategyStatus


class PromotionStatus(Enum):
    """Strategy promotion status"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    DEPLOYED = "deployed"
    ROLLED_BACK = "rolled_back"


class ValidationGate(Enum):
    """Validation gates for strategy promotion"""
    PERFORMANCE_THRESHOLD = "performance_threshold"
    MINIMUM_TRADES = "minimum_trades"
    DRAWDOWN_LIMIT = "drawdown_limit"
    SHARPE_RATIO = "sharpe_ratio"
    WIN_RATE = "win_rate"
    STABILITY_CHECK = "stability_check"
    RISK_ASSESSMENT = "risk_assessment"


class RollbackTrigger(Enum):
    """Triggers for automatic rollback"""
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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['gate'] = self.gate.value
        return data


@dataclass
class PromotionRequest:
    """Strategy promotion request"""
    request_id: str
    strategy_id: str
    from_status: StrategyStatus
    to_status: StrategyStatus
    requested_by: str
    requested_at: datetime
    reason: str
    validation_results: List[ValidationResult]
    status: PromotionStatus
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    deployed_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['from_status'] = self.from_status.value
        data['to_status'] = self.to_status.value
        data['requested_at'] = self.requested_at.isoformat()
        data['status'] = self.status.value
        data['approved_at'] = self.approved_at.isoformat() if self.approved_at else None
        data['deployed_at'] = self.deployed_at.isoformat() if self.deployed_at else None
        data['validation_results'] = [vr.to_dict() for vr in self.validation_results]
        return data


@dataclass
class RollbackRecord:
    """Record of strategy rollback"""
    rollback_id: str
    strategy_id: str
    trigger: RollbackTrigger
    from_version: str
    to_version: str
    reason: str
    triggered_at: datetime
    triggered_by: str
    performance_before: Optional[Dict[str, float]]
    performance_after: Optional[Dict[str, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['trigger'] = self.trigger.value
        data['triggered_at'] = self.triggered_at.isoformat()
        return data


class StrategyManager:
    """
    Comprehensive strategy management system
    
    This class orchestrates strategy registry, performance tracking, lineage management,
    and provides safe promotion/rollback capabilities with validation gates and
    automatic monitoring.
    """
    
    def __init__(self, storage_backend: Optional[Any] = None):
        """
        Initialize strategy manager
        
        Args:
            storage_backend: Optional storage backend for persistence
        """
        self.storage_backend = storage_backend
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.registry = StrategyRegistry(storage_backend)
        self.lineage_tracker = StrategyLineageTracker(storage_backend)
        self.performance_trackers: Dict[str, PerformanceTracker] = {}
        
        # Promotion and rollback management
        self.promotion_requests: Dict[str, PromotionRequest] = {}
        self.rollback_records: Dict[str, RollbackRecord] = {}
        self.active_strategies: Dict[StrategyStatus, List[str]] = defaultdict(list)
        
        # Configuration
        self.validation_thresholds = {
            ValidationGate.PERFORMANCE_THRESHOLD: 0.05,  # 5% minimum return
            ValidationGate.MINIMUM_TRADES: 50,  # Minimum trades for validation
            ValidationGate.DRAWDOWN_LIMIT: 0.15,  # Maximum 15% drawdown
            ValidationGate.SHARPE_RATIO: 1.0,  # Minimum Sharpe ratio
            ValidationGate.WIN_RATE: 0.45,  # Minimum 45% win rate
        }
        
        self.rollback_thresholds = {
            RollbackTrigger.PERFORMANCE_DEGRADATION: -0.10,  # 10% performance drop
            RollbackTrigger.EXCESSIVE_DRAWDOWN: 0.20,  # 20% drawdown triggers rollback
            RollbackTrigger.ERROR_RATE: 0.05,  # 5% error rate
        }
        
        # Monitoring
        self.monitoring_enabled = True
        self.monitoring_interval = timedelta(hours=1)
        self.last_monitoring_check = datetime.now()
        
        self.logger.info("StrategyManager initialized")
    
    def register_strategy(self, strategy: Strategy, metadata: Dict[str, Any],
                         parent_id: Optional[str] = None) -> str:
        """
        Register a new strategy with comprehensive tracking
        
        Args:
            strategy: Strategy instance to register
            metadata: Strategy metadata
            parent_id: Optional parent strategy ID
            
        Returns:
            Strategy ID
        """
        # Register in registry
        strategy_id = self.registry.register_strategy(strategy, metadata, parent_id)
        
        # Register in lineage tracker
        self.lineage_tracker.register_strategy(strategy_id, parent_id, metadata)
        
        # Create performance tracker
        self.performance_trackers[strategy_id] = PerformanceTracker(
            strategy_id, storage_backend=self.storage_backend
        )
        
        # Add to active strategies list
        strategy_metadata = self.registry.get_strategy_metadata(strategy_id)
        if strategy_metadata:
            self.active_strategies[strategy_metadata.status].append(strategy_id)
        
        self.logger.info(f"Registered strategy {strategy_id} with comprehensive tracking")
        return strategy_id
    
    def record_trade_result(self, strategy_id: str, trade_result: TradeResult) -> None:
        """
        Record a trade result for performance tracking
        
        Args:
            strategy_id: Strategy that executed the trade
            trade_result: Trade result to record
        """
        if strategy_id not in self.performance_trackers:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Record in performance tracker
        self.performance_trackers[strategy_id].record_trade(trade_result)
        
        # Check for automatic rollback triggers
        if self.monitoring_enabled:
            self._check_rollback_triggers(strategy_id)
        
        self.logger.debug(f"Recorded trade result for strategy {strategy_id}")
    
    def request_promotion(self, strategy_id: str, to_status: StrategyStatus,
                         reason: str, requested_by: str) -> str:
        """
        Request strategy promotion with validation
        
        Args:
            strategy_id: Strategy to promote
            to_status: Target status
            reason: Reason for promotion
            requested_by: Who requested the promotion
            
        Returns:
            Promotion request ID
        """
        strategy_metadata = self.registry.get_strategy_metadata(strategy_id)
        if not strategy_metadata:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Validate promotion path
        if not self._is_valid_promotion_path(strategy_metadata.status, to_status):
            raise ValueError(f"Invalid promotion path: {strategy_metadata.status} -> {to_status}")
        
        # Run validation gates
        validation_results = self._run_validation_gates(strategy_id, to_status)
        
        # Create promotion request
        request_id = f"promotion_{uuid4().hex[:8]}"
        
        promotion_request = PromotionRequest(
            request_id=request_id,
            strategy_id=strategy_id,
            from_status=strategy_metadata.status,
            to_status=to_status,
            requested_by=requested_by,
            requested_at=datetime.now(),
            reason=reason,
            validation_results=validation_results,
            status=PromotionStatus.PENDING
        )
        
        # Auto-approve if all validations pass
        all_passed = all(vr.passed for vr in validation_results)
        if all_passed and to_status != StrategyStatus.PRODUCTION:
            promotion_request.status = PromotionStatus.APPROVED
            promotion_request.approved_by = "system"
            promotion_request.approved_at = datetime.now()
        
        self.promotion_requests[request_id] = promotion_request
        
        # Record change in lineage
        self.lineage_tracker.record_change(
            strategy_id,
            ChangeType.PERFORMANCE_OPTIMIZATION,
            f"Promotion requested: {strategy_metadata.status.value} -> {to_status.value}",
            ImpactLevel.HIGH if to_status == StrategyStatus.PRODUCTION else ImpactLevel.MEDIUM,
            created_by=requested_by
        )
        
        self.logger.info(f"Created promotion request {request_id} for strategy {strategy_id}")
        return request_id
    
    def approve_promotion(self, request_id: str, approved_by: str) -> bool:
        """
        Approve a promotion request
        
        Args:
            request_id: Promotion request ID
            approved_by: Who approved the promotion
            
        Returns:
            True if approved successfully
        """
        if request_id not in self.promotion_requests:
            raise ValueError(f"Promotion request {request_id} not found")
        
        request = self.promotion_requests[request_id]
        
        if request.status != PromotionStatus.PENDING:
            raise ValueError(f"Request {request_id} is not pending (status: {request.status})")
        
        # Update request
        request.status = PromotionStatus.APPROVED
        request.approved_by = approved_by
        request.approved_at = datetime.now()
        
        self.logger.info(f"Approved promotion request {request_id} by {approved_by}")
        return True
    
    def deploy_strategy(self, request_id: str) -> bool:
        """
        Deploy an approved strategy promotion
        
        Args:
            request_id: Approved promotion request ID
            
        Returns:
            True if deployed successfully
        """
        if request_id not in self.promotion_requests:
            raise ValueError(f"Promotion request {request_id} not found")
        
        request = self.promotion_requests[request_id]
        
        if request.status != PromotionStatus.APPROVED:
            raise ValueError(f"Request {request_id} is not approved (status: {request.status})")
        
        # Get strategy metadata before update
        strategy_metadata = self.registry.get_strategy_metadata(request.strategy_id)
        if not strategy_metadata:
            raise ValueError(f"Strategy {request.strategy_id} not found")
        
        # Update strategy status in registry
        self.registry.update_strategy_status(request.strategy_id, request.to_status)
        
        # Move strategy between active lists
        self.active_strategies[strategy_metadata.status].remove(request.strategy_id)
        self.active_strategies[request.to_status].append(request.strategy_id)
        
        # Update request
        request.status = PromotionStatus.DEPLOYED
        request.deployed_at = datetime.now()
        
        # Record deployment in lineage
        self.lineage_tracker.record_change(
            request.strategy_id,
            ChangeType.PERFORMANCE_OPTIMIZATION,
            f"Deployed to {request.to_status.value}",
            ImpactLevel.HIGH,
            created_by=request.approved_by or "system"
        )
        
        self.logger.info(f"Deployed strategy {request.strategy_id} to {request.to_status}")
        return True
    
    def rollback_strategy(self, strategy_id: str, trigger: RollbackTrigger,
                         reason: str, triggered_by: str = "system") -> str:
        """
        Rollback a strategy to previous version
        
        Args:
            strategy_id: Strategy to rollback
            trigger: What triggered the rollback
            reason: Reason for rollback
            triggered_by: Who triggered the rollback
            
        Returns:
            Rollback record ID
        """
        strategy_metadata = self.registry.get_strategy_metadata(strategy_id)
        if not strategy_metadata:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Find previous version
        versions = self.registry.get_strategy_versions(strategy_id)
        if len(versions) < 2:
            raise ValueError("No previous version available for rollback")
        
        current_version = versions[-1].version
        previous_version = versions[-2].version
        
        # Get performance before rollback
        performance_tracker = self.performance_trackers.get(strategy_id)
        performance_before = None
        if performance_tracker:
            metrics = performance_tracker.get_performance_metrics()
            performance_before = {
                'total_return': metrics.total_return_pct,
                'sharpe_ratio': metrics.sharpe_ratio,
                'max_drawdown': metrics.max_drawdown,
                'win_rate': metrics.win_rate
            }
        
        # Revert to previous version in registry
        self.registry.revert_to_version(strategy_id, previous_version)
        
        # Create rollback record
        rollback_id = f"rollback_{uuid4().hex[:8]}"
        
        rollback_record = RollbackRecord(
            rollback_id=rollback_id,
            strategy_id=strategy_id,
            trigger=trigger,
            from_version=current_version,
            to_version=previous_version,
            reason=reason,
            triggered_at=datetime.now(),
            triggered_by=triggered_by,
            performance_before=performance_before,
            performance_after=None  # Will be updated later
        )
        
        self.rollback_records[rollback_id] = rollback_record
        
        # Update strategy status if it was in production
        if strategy_metadata.status == StrategyStatus.PRODUCTION:
            # Update status in registry to testing
            self.registry.update_strategy_status(strategy_id, StrategyStatus.TESTING)
            
            # Update active strategies lists
            if strategy_id in self.active_strategies[StrategyStatus.PRODUCTION]:
                self.active_strategies[StrategyStatus.PRODUCTION].remove(strategy_id)
            if strategy_id not in self.active_strategies[StrategyStatus.TESTING]:
                self.active_strategies[StrategyStatus.TESTING].append(strategy_id)
        
        # Record rollback in lineage
        self.lineage_tracker.record_change(
            strategy_id,
            ChangeType.BUG_FIX,
            f"Rolled back from {current_version} to {previous_version}: {reason}",
            ImpactLevel.CRITICAL,
            created_by=triggered_by
        )
        
        self.logger.warning(f"Rolled back strategy {strategy_id} due to {trigger.value}: {reason}")
        return rollback_id
    
    def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """
        Get comprehensive status for a strategy
        
        Args:
            strategy_id: Strategy to analyze
            
        Returns:
            Comprehensive status dictionary
        """
        # Get metadata
        metadata = self.registry.get_strategy_metadata(strategy_id)
        if not metadata:
            raise ValueError(f"Strategy {strategy_id} not found")
        
        # Get performance metrics
        performance_summary = None
        if strategy_id in self.performance_trackers:
            performance_summary = self.performance_trackers[strategy_id].get_performance_summary()
        
        # Get lineage information
        lineage = self.lineage_tracker.get_lineage(strategy_id)
        
        # Get recent promotion requests
        recent_promotions = [
            req.to_dict() for req in self.promotion_requests.values()
            if req.strategy_id == strategy_id
        ]
        
        # Get rollback history
        rollback_history = [
            record.to_dict() for record in self.rollback_records.values()
            if record.strategy_id == strategy_id
        ]
        
        return {
            'strategy_id': strategy_id,
            'metadata': metadata.to_dict(),
            'performance': performance_summary,
            'lineage': lineage,
            'promotion_requests': recent_promotions,
            'rollback_history': rollback_history,
            'last_updated': datetime.now().isoformat()
        }
    
    def get_active_strategies(self, status: Optional[StrategyStatus] = None) -> Dict[str, List[str]]:
        """
        Get active strategies by status
        
        Args:
            status: Optional status filter
            
        Returns:
            Dictionary of strategies by status
        """
        if status:
            return {status.value: self.active_strategies[status]}
        
        return {
            status.value: strategies
            for status, strategies in self.active_strategies.items()
            if strategies
        }
    
    def compare_strategies(self, strategy_ids: List[str]) -> Dict[str, Any]:
        """
        Compare performance of multiple strategies
        
        Args:
            strategy_ids: List of strategy IDs to compare
            
        Returns:
            Comparison results
        """
        if len(strategy_ids) < 2:
            raise ValueError("At least 2 strategies required for comparison")
        
        # Validate all strategies exist
        for sid in strategy_ids:
            if sid not in self.performance_trackers:
                raise ValueError(f"Strategy {sid} not found or has no performance data")
        
        # Get performance metrics for all strategies
        strategy_metrics = {}
        for sid in strategy_ids:
            tracker = self.performance_trackers[sid]
            metrics = tracker.get_performance_metrics()
            strategy_metrics[sid] = metrics.to_dict()
        
        # Calculate rankings
        rankings = self._calculate_strategy_rankings(strategy_metrics)
        
        # Get metadata for context
        strategy_info = {}
        for sid in strategy_ids:
            metadata = self.registry.get_strategy_metadata(sid)
            if metadata:
                strategy_info[sid] = {
                    'name': metadata.name,
                    'status': metadata.status.value,
                    'version': metadata.version,
                    'created_at': metadata.created_at.isoformat()
                }
        
        return {
            'strategies': strategy_info,
            'metrics': strategy_metrics,
            'rankings': rankings,
            'comparison_date': datetime.now().isoformat()
        }
    
    def _is_valid_promotion_path(self, from_status: StrategyStatus, to_status: StrategyStatus) -> bool:
        """Check if promotion path is valid"""
        valid_paths = {
            StrategyStatus.EXPERIMENTAL: [StrategyStatus.TESTING],
            StrategyStatus.TESTING: [StrategyStatus.PRODUCTION],
            StrategyStatus.PRODUCTION: [StrategyStatus.RETIRED],
            StrategyStatus.RETIRED: [],
            StrategyStatus.DEPRECATED: []
        }
        
        return to_status in valid_paths.get(from_status, [])
    
    def _run_validation_gates(self, strategy_id: str, target_status: StrategyStatus) -> List[ValidationResult]:
        """Run validation gates for strategy promotion"""
        results = []
        
        # Get performance tracker
        if strategy_id not in self.performance_trackers:
            # Return all validation gates as failed
            return [
                ValidationResult(ValidationGate.PERFORMANCE_THRESHOLD, False, 0.0, 0.05, "No performance tracker"),
                ValidationResult(ValidationGate.MINIMUM_TRADES, False, 0, 50, "No performance tracker"),
                ValidationResult(ValidationGate.DRAWDOWN_LIMIT, False, 0.0, 0.15, "No performance tracker"),
                ValidationResult(ValidationGate.SHARPE_RATIO, False, 0.0, 1.0, "No performance tracker"),
                ValidationResult(ValidationGate.WIN_RATE, False, 0.0, 0.45, "No performance tracker")
            ]
        
        tracker = self.performance_trackers[strategy_id]
        metrics = tracker.get_performance_metrics()
        
        # Performance threshold check
        threshold = self.validation_thresholds[ValidationGate.PERFORMANCE_THRESHOLD]
        passed = metrics.total_return_pct >= threshold
        results.append(ValidationResult(
            ValidationGate.PERFORMANCE_THRESHOLD,
            passed,
            metrics.total_return_pct,
            threshold,
            f"Return: {metrics.total_return_pct:.2%} vs {threshold:.2%} threshold"
        ))
        
        # Minimum trades check
        min_trades = self.validation_thresholds[ValidationGate.MINIMUM_TRADES]
        passed = metrics.total_trades >= min_trades
        results.append(ValidationResult(
            ValidationGate.MINIMUM_TRADES,
            passed,
            metrics.total_trades,
            min_trades,
            f"Trades: {metrics.total_trades} vs {min_trades} minimum"
        ))
        
        # Drawdown limit check
        drawdown_limit = self.validation_thresholds[ValidationGate.DRAWDOWN_LIMIT]
        passed = metrics.max_drawdown <= drawdown_limit
        results.append(ValidationResult(
            ValidationGate.DRAWDOWN_LIMIT,
            passed,
            metrics.max_drawdown,
            drawdown_limit,
            f"Max drawdown: {metrics.max_drawdown:.2%} vs {drawdown_limit:.2%} limit"
        ))
        
        # Sharpe ratio check
        sharpe_threshold = self.validation_thresholds[ValidationGate.SHARPE_RATIO]
        passed = metrics.sharpe_ratio >= sharpe_threshold
        results.append(ValidationResult(
            ValidationGate.SHARPE_RATIO,
            passed,
            metrics.sharpe_ratio,
            sharpe_threshold,
            f"Sharpe ratio: {metrics.sharpe_ratio:.2f} vs {sharpe_threshold:.2f} threshold"
        ))
        
        # Win rate check
        win_rate_threshold = self.validation_thresholds[ValidationGate.WIN_RATE]
        passed = metrics.win_rate >= win_rate_threshold
        results.append(ValidationResult(
            ValidationGate.WIN_RATE,
            passed,
            metrics.win_rate,
            win_rate_threshold,
            f"Win rate: {metrics.win_rate:.2%} vs {win_rate_threshold:.2%} threshold"
        ))
        
        return results
    
    def _check_rollback_triggers(self, strategy_id: str) -> None:
        """Check if strategy should be automatically rolled back"""
        strategy_metadata = self.registry.get_strategy_metadata(strategy_id)
        if not strategy_metadata or strategy_metadata.status != StrategyStatus.PRODUCTION:
            return  # Only monitor production strategies
        
        tracker = self.performance_trackers[strategy_id]
        metrics = tracker.get_performance_metrics()
        
        # Check performance degradation
        perf_threshold = self.rollback_thresholds[RollbackTrigger.PERFORMANCE_DEGRADATION]
        if metrics.total_return_pct <= perf_threshold:
            self.rollback_strategy(
                strategy_id,
                RollbackTrigger.PERFORMANCE_DEGRADATION,
                f"Performance dropped to {metrics.total_return_pct:.2%}"
            )
            return
        
        # Check excessive drawdown
        drawdown_threshold = self.rollback_thresholds[RollbackTrigger.EXCESSIVE_DRAWDOWN]
        if metrics.max_drawdown >= drawdown_threshold:
            self.rollback_strategy(
                strategy_id,
                RollbackTrigger.EXCESSIVE_DRAWDOWN,
                f"Drawdown reached {metrics.max_drawdown:.2%}"
            )
            return
    
    def _calculate_strategy_rankings(self, strategy_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        """Calculate rankings for strategy comparison"""
        rankings = {}
        
        # Metrics to rank (higher is better)
        positive_metrics = ['total_return_pct', 'sharpe_ratio', 'win_rate', 'profit_factor']
        # Metrics to rank (lower is better)
        negative_metrics = ['max_drawdown', 'volatility']
        
        for metric in positive_metrics + negative_metrics:
            # Get values for this metric
            values = []
            for sid, metrics in strategy_metrics.items():
                if metric in metrics:
                    values.append((sid, metrics[metric]))
            
            # Sort and rank
            reverse = metric in positive_metrics
            values.sort(key=lambda x: x[1], reverse=reverse)
            
            for rank, (sid, value) in enumerate(values, 1):
                if sid not in rankings:
                    rankings[sid] = {}
                rankings[sid][metric] = rank
        
        return rankings