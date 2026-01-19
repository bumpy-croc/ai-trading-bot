"""Tests for PortfolioRiskManager class rename and backward compatibility."""

import pytest

from src.risk import PortfolioRiskManager, RiskManager, RiskParameters


class TestPortfolioRiskManagerRename:
    """Test suite for PortfolioRiskManager rename refactoring."""

    def test_new_class_name_imports(self):
        """Test that PortfolioRiskManager can be imported directly."""
        assert PortfolioRiskManager is not None
        assert hasattr(PortfolioRiskManager, "__init__")

    def test_backward_compatibility_alias(self):
        """Test that RiskManager alias still works for backward compatibility."""
        # The old name should still be importable
        assert RiskManager is not None

        # The alias should point to the same class
        assert RiskManager is PortfolioRiskManager

    def test_instantiation_with_new_name(self):
        """Test instantiating with the new PortfolioRiskManager name."""
        params = RiskParameters(
            base_risk_per_trade=0.02,
            max_daily_risk=0.06,
        )
        mgr = PortfolioRiskManager(parameters=params, max_concurrent_positions=3)

        assert mgr is not None
        assert mgr.params == params
        assert mgr.max_concurrent_positions == 3

    def test_instantiation_with_old_name(self):
        """Test that old RiskManager name still works (backward compatibility)."""
        params = RiskParameters()
        mgr = RiskManager(parameters=params)

        assert mgr is not None
        assert isinstance(mgr, PortfolioRiskManager)

    def test_type_checking_compatibility(self):
        """Test that type checking works with both names."""
        params = RiskParameters()

        # Create with new name
        mgr1 = PortfolioRiskManager(parameters=params)
        assert isinstance(mgr1, PortfolioRiskManager)
        assert isinstance(mgr1, RiskManager)  # Should also pass with alias

        # Create with old name
        mgr2 = RiskManager(parameters=params)
        assert isinstance(mgr2, RiskManager)
        assert isinstance(mgr2, PortfolioRiskManager)  # Should also pass

    def test_class_name_attribute(self):
        """Test that the class __name__ attribute is correct."""
        assert PortfolioRiskManager.__name__ == "PortfolioRiskManager"

        # Alias should point to same class, so same __name__
        assert RiskManager.__name__ == "PortfolioRiskManager"

    def test_exports_in_init(self):
        """Test that __all__ exports both names."""
        from src.risk import __all__

        assert "PortfolioRiskManager" in __all__
        assert "RiskManager" in __all__
        assert "RiskParameters" in __all__

    def test_docstring_updated(self):
        """Test that class docstring mentions portfolio management."""
        doc = PortfolioRiskManager.__doc__
        assert doc is not None
        assert "portfolio" in doc.lower() or "Portfolio" in doc

    def test_functionality_unchanged(self):
        """Test that core functionality still works after rename."""
        params = RiskParameters(max_daily_risk=0.06)
        mgr = PortfolioRiskManager(parameters=params)

        # Test basic operations work
        assert mgr.daily_risk_used == 0.0
        assert mgr.get_total_exposure() == 0.0
        assert mgr.get_max_concurrent_positions() == 3  # default

        # Test position tracking still works
        mgr.update_position("BTCUSDT", "long", 0.05, 50000.0)
        assert "BTCUSDT" in mgr.positions
        assert mgr.daily_risk_used == 0.05

        mgr.close_position("BTCUSDT")
        assert "BTCUSDT" not in mgr.positions
        assert mgr.daily_risk_used == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
