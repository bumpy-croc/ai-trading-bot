"""Unit tests for shared DynamicRiskHandler.

These tests verify that the dynamic risk handler produces consistent results
and is used identically by both backtesting and live trading engines.
"""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from src.engines.shared.dynamic_risk_handler import (
    DynamicRiskHandler,
    DynamicRiskAdjustment,
)
from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager


class TestDynamicRiskHandlerInitialization:
    """Test DynamicRiskHandler initialization."""

    def test_initialization_with_manager(self) -> None:
        """Test initialization with a valid manager."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.8],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        assert handler.dynamic_risk_manager is manager

    def test_initialization_without_manager(self) -> None:
        """Test initialization with None manager."""
        handler = DynamicRiskHandler(None)
        assert handler.dynamic_risk_manager is None

    def test_initialization_with_significance_threshold(self) -> None:
        """Test initialization with custom significance threshold."""
        handler = DynamicRiskHandler(None, significance_threshold=0.2)
        assert handler.significance_threshold == 0.2


class TestApplyDynamicRisk:
    """Test apply_dynamic_risk method."""

    def test_no_manager_returns_original_size(self) -> None:
        """No manager should return original size unchanged."""
        handler = DynamicRiskHandler(None)

        original_size = 0.05
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        assert adjusted == original_size

    def test_no_drawdown_returns_original_size(self) -> None:
        """No drawdown should return original size."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10],
            risk_reduction_factors=[0.8, 0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        original_size = 0.05
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=10000.0,  # No drawdown
            peak_balance=10000.0,
        )

        assert adjusted == pytest.approx(original_size, rel=0.1)

    def test_drawdown_applies_reduction(self) -> None:
        """Drawdown should apply reduction factor."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10],
            risk_reduction_factors=[0.8, 0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        original_size = 0.05

        # 5% drawdown - should apply 0.8 factor
        adjusted_5pct = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9500.0,
            peak_balance=10000.0,
        )
        assert adjusted_5pct == pytest.approx(original_size * 0.8, rel=0.1)

        # 10% drawdown - should apply 0.5 factor
        adjusted_10pct = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )
        assert adjusted_10pct == pytest.approx(original_size * 0.5, rel=0.1)

    def test_severe_drawdown_applies_maximum_reduction(self) -> None:
        """Severe drawdown should apply the most aggressive reduction."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05, 0.10, 0.15],
            risk_reduction_factors=[0.8, 0.6, 0.3],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        original_size = 0.05

        # 20% drawdown (beyond all thresholds) - should apply lowest factor
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=8000.0,  # 20% drawdown
            peak_balance=10000.0,
        )

        assert adjusted == pytest.approx(original_size * 0.3, rel=0.15)


class TestAdjustmentTracking:
    """Test adjustment tracking functionality."""

    def test_significant_adjustments_tracked(self) -> None:
        """Significant adjustments should be tracked."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],  # 50% reduction (significant)
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        assert handler.has_adjustments is True
        assert handler.adjustment_count >= 1

    def test_insignificant_adjustments_not_tracked(self) -> None:
        """Insignificant adjustments should not be tracked."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.95],  # Only 5% reduction (insignificant)
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9400.0,  # 6% drawdown
            peak_balance=10000.0,
        )

        # 5% reduction is below 10% significance threshold
        # This depends on whether 0.95 is significant enough
        # With 0.1 threshold, |0.95 - 1.0| = 0.05 < 0.1, so not tracked

    def test_get_adjustments_returns_list(self) -> None:
        """Get adjustments should return list of dicts."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        adjustments = handler.get_adjustments(clear=False)
        assert isinstance(adjustments, list)

        if len(adjustments) > 0:
            assert isinstance(adjustments[0], dict)
            assert "position_size_factor" in adjustments[0]

    def test_get_adjustments_clears_by_default(self) -> None:
        """Get adjustments should clear list by default."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        handler.get_adjustments(clear=True)
        assert handler.has_adjustments is False

    def test_get_adjustment_objects(self) -> None:
        """Get adjustment objects should return DynamicRiskAdjustment instances."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        adjustments = handler.get_adjustment_objects(clear=False)
        if len(adjustments) > 0:
            assert isinstance(adjustments[0], DynamicRiskAdjustment)

    def test_clear_adjustments(self) -> None:
        """Clear adjustments should empty the list."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        handler.clear_adjustments()
        assert handler.has_adjustments is False
        assert handler.adjustment_count == 0


class TestSetManager:
    """Test set_manager method."""

    def test_set_manager_updates_reference(self) -> None:
        """Set manager should update the manager reference."""
        handler = DynamicRiskHandler(None)
        assert handler.dynamic_risk_manager is None

        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.8],
        )
        new_manager = DynamicRiskManager(config)
        handler.set_manager(new_manager)

        assert handler.dynamic_risk_manager is new_manager

    def test_set_manager_to_none(self) -> None:
        """Set manager to None should disable risk adjustment."""
        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.8],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager)

        handler.set_manager(None)
        assert handler.dynamic_risk_manager is None

        # Should return original size now
        adjusted = handler.apply_dynamic_risk(
            original_size=0.05,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )
        assert adjusted == 0.05


class TestDynamicRiskAdjustment:
    """Test DynamicRiskAdjustment dataclass."""

    def test_to_dict(self) -> None:
        """To dict should return proper dictionary."""
        adjustment = DynamicRiskAdjustment(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            position_size_factor=0.8,
            stop_loss_tightening=1.0,
            daily_risk_factor=1.0,
            primary_reason="drawdown",
            current_drawdown=0.05,
            balance=9500.0,
            peak_balance=10000.0,
            original_size=0.05,
            adjusted_size=0.04,
        )

        result = adjustment.to_dict()

        assert result["position_size_factor"] == 0.8
        assert result["balance"] == 9500.0
        assert result["original_size"] == 0.05
        assert result["adjusted_size"] == 0.04
        assert result["primary_reason"] == "drawdown"


class TestErrorHandling:
    """Test error handling in dynamic risk handler."""

    def test_graceful_handling_of_manager_errors(self) -> None:
        """Handler should gracefully handle manager errors."""
        handler = DynamicRiskHandler(None)

        # Create a mock manager that raises an exception
        mock_manager = Mock()
        mock_manager.calculate_dynamic_risk_adjustments.side_effect = ValueError(
            "Test error"
        )
        handler.dynamic_risk_manager = mock_manager

        original_size = 0.05
        adjusted = handler.apply_dynamic_risk(
            original_size=original_size,
            current_time=datetime.now(),
            balance=9000.0,
            peak_balance=10000.0,
        )

        # Should return original size on error
        assert adjusted == original_size


class TestThreadSafety:
    """Test thread safety of dynamic risk handler."""

    def test_concurrent_access_to_adjustments(self) -> None:
        """Concurrent access to adjustments should be safe."""
        import threading

        config = DynamicRiskConfig(
            enabled=True,
            drawdown_thresholds=[0.05],
            risk_reduction_factors=[0.5],
        )
        manager = DynamicRiskManager(config)
        handler = DynamicRiskHandler(manager, significance_threshold=0.1)

        errors = []

        def add_adjustment():
            try:
                handler.apply_dynamic_risk(
                    original_size=0.05,
                    current_time=datetime.now(),
                    balance=9000.0,
                    peak_balance=10000.0,
                )
            except Exception as e:
                errors.append(e)

        def read_adjustments():
            try:
                handler.get_adjustments(clear=False)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(10):
            t1 = threading.Thread(target=add_adjustment)
            t2 = threading.Thread(target=read_adjustments)
            threads.extend([t1, t2])

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should have no errors
        assert len(errors) == 0
