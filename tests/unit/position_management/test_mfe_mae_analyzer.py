"""Unit tests for MFE/MAE analyzer functionality."""

from datetime import datetime

import pytest

from src.position_management.mfe_mae_analyzer import MFEMAEAnalyzer, TradeMFERecord

pytestmark = pytest.mark.unit


class TestMFEMAEAnalyzer:
    """Test MFE/MAE analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = MFEMAEAnalyzer()

    def test_calculate_avg_mfe_mae_by_strategy_basic(self):
        """Test basic MFE/MAE calculation by strategy."""
        trades = [
            {"strategy": "ml_basic", "mfe": 0.05, "mae": -0.02},
            {"strategy": "ml_basic", "mfe": 0.08, "mae": -0.03},
            {"strategy": "ml_adaptive", "mfe": 0.06, "mae": -0.01},
        ]

        # Test specific strategy
        result = self.analyzer.calculate_avg_mfe_mae_by_strategy(trades, "ml_basic")
        assert result["avg_mfe"] == pytest.approx(0.065)  # (0.05 + 0.08) / 2
        assert result["avg_mae"] == pytest.approx(-0.025)  # (-0.02 + -0.03) / 2

        # Test all strategies
        result = self.analyzer.calculate_avg_mfe_mae_by_strategy(trades, None)
        assert result["avg_mfe"] == pytest.approx(0.063333, rel=1e-5)  # (0.05 + 0.08 + 0.06) / 3
        assert result["avg_mae"] == pytest.approx(-0.02, rel=1e-5)  # (-0.02 + -0.03 + -0.01) / 3

    def test_calculate_avg_mfe_mae_empty_data(self):
        """Test MFE/MAE calculation with empty data."""
        result = self.analyzer.calculate_avg_mfe_mae_by_strategy([], "any_strategy")
        assert result["avg_mfe"] == 0.0
        assert result["avg_mae"] == 0.0

    def test_calculate_avg_mfe_mae_missing_strategy(self):
        """Test MFE/MAE calculation with non-existent strategy."""
        trades = [
            {"strategy": "ml_basic", "mfe": 0.05, "mae": -0.02},
        ]
        result = self.analyzer.calculate_avg_mfe_mae_by_strategy(trades, "nonexistent")
        assert result["avg_mfe"] == 0.0
        assert result["avg_mae"] == 0.0

    def test_calculate_avg_mfe_mae_none_values(self):
        """Test MFE/MAE calculation with None values."""
        trades = [
            {"strategy": "ml_basic", "mfe": None, "mae": None},
            {"strategy": "ml_basic", "mfe": 0.05, "mae": -0.02},
            {"strategy": "ml_basic", "mfe": 0.08, "mae": None},
        ]
        result = self.analyzer.calculate_avg_mfe_mae_by_strategy(trades, "ml_basic")
        assert result["avg_mfe"] == pytest.approx(0.043333, rel=1e-4)  # (0 + 0.05 + 0.08) / 3
        assert result["avg_mae"] == pytest.approx(-0.006667, rel=1e-4)  # (0 + -0.02 + 0) / 3

    def test_calculate_avg_mfe_mae_missing_keys(self):
        """Test MFE/MAE calculation with missing keys."""
        trades = [
            {"strategy": "ml_basic"},  # Missing mfe/mae
            {"strategy": "ml_basic", "mfe": 0.05},  # Missing mae
            {"strategy": "ml_basic", "mae": -0.02},  # Missing mfe
        ]
        result = self.analyzer.calculate_avg_mfe_mae_by_strategy(trades, "ml_basic")
        assert result["avg_mfe"] == pytest.approx(0.016667, rel=1e-4)  # (0 + 0.05 + 0) / 3
        assert result["avg_mae"] == pytest.approx(-0.006667, rel=1e-4)  # (0 + 0 + -0.02) / 3

    def test_calculate_mfe_mae_ratios_basic(self):
        """Test MFE/MAE ratio calculation."""
        trades = [
            {"mfe": 0.08, "mae": -0.04},  # ratio = 0.08 / 0.04 = 2.0
            {"mfe": 0.06, "mae": -0.02},  # ratio = 0.06 / 0.02 = 3.0
            {"mfe": 0.04, "mae": -0.08},  # ratio = 0.04 / 0.08 = 0.5
        ]
        result = self.analyzer.calculate_mfe_mae_ratios(trades)
        assert result["avg_ratio"] == pytest.approx(1.833333, rel=1e-5)  # (2.0 + 3.0 + 0.5) / 3

    def test_calculate_mfe_mae_ratios_zero_mae(self):
        """Test MFE/MAE ratio calculation with zero MAE."""
        trades = [
            {"mfe": 0.08, "mae": 0.0},  # Should be ignored (division by zero)
            {"mfe": 0.06, "mae": -0.02},  # ratio = 0.06 / 0.02 = 3.0
        ]
        result = self.analyzer.calculate_mfe_mae_ratios(trades)
        assert result["avg_ratio"] == pytest.approx(3.0)

    def test_calculate_mfe_mae_ratios_empty_data(self):
        """Test MFE/MAE ratio calculation with empty data."""
        result = self.analyzer.calculate_mfe_mae_ratios([])
        assert result["avg_ratio"] == 0.0

    def test_calculate_mfe_mae_ratios_all_zero_mae(self):
        """Test MFE/MAE ratio calculation with all zero MAE."""
        trades = [
            {"mfe": 0.08, "mae": 0.0},
            {"mfe": 0.06, "mae": 0.0},
        ]
        result = self.analyzer.calculate_mfe_mae_ratios(trades)
        assert result["avg_ratio"] == 0.0

    def test_calculate_mfe_mae_ratios_none_values(self):
        """Test MFE/MAE ratio calculation with None values."""
        trades = [
            {"mfe": None, "mae": None},  # Should be treated as 0
            {"mfe": 0.06, "mae": -0.02},  # ratio = 0.06 / 0.02 = 3.0
        ]
        result = self.analyzer.calculate_mfe_mae_ratios(trades)
        assert result["avg_ratio"] == pytest.approx(3.0)

    def test_analyze_exit_timing_efficiency_basic(self):
        """Test exit timing efficiency analysis."""
        trades = [
            {"pnl_percent": 4.0, "mfe": 0.05},  # efficiency = 0.04 / 0.05 = 0.8
            {"pnl_percent": 3.0, "mfe": 0.06},  # efficiency = 0.03 / 0.06 = 0.5
            {"pnl_percent": 2.0, "mfe": 0.04},  # efficiency = 0.02 / 0.04 = 0.5
        ]
        result = self.analyzer.analyze_exit_timing_efficiency(trades)
        assert result["avg_exit_efficiency"] == pytest.approx(0.6)  # (0.8 + 0.5 + 0.5) / 3

    def test_analyze_exit_timing_efficiency_zero_mfe(self):
        """Test exit timing efficiency with zero MFE."""
        trades = [
            {"pnl_percent": 2.0, "mfe": 0.0},  # Should be ignored
            {"pnl_percent": 3.0, "mfe": 0.06},  # efficiency = 0.03 / 0.06 = 0.5
        ]
        result = self.analyzer.analyze_exit_timing_efficiency(trades)
        assert result["avg_exit_efficiency"] == pytest.approx(0.5)

    def test_analyze_exit_timing_efficiency_negative_pnl(self):
        """Test exit timing efficiency with negative PnL."""
        trades = [
            {"pnl_percent": -2.0, "mfe": 0.05},  # efficiency = max(0, min(1, -0.02 / 0.05)) = 0
            {"pnl_percent": 3.0, "mfe": 0.06},  # efficiency = 0.03 / 0.06 = 0.5
        ]
        result = self.analyzer.analyze_exit_timing_efficiency(trades)
        assert result["avg_exit_efficiency"] == pytest.approx(0.25)  # (0 + 0.5) / 2

    def test_analyze_exit_timing_efficiency_over_100_percent(self):
        """Test exit timing efficiency with PnL > MFE (clamped to 1.0)."""
        trades = [
            {"pnl_percent": 8.0, "mfe": 0.05},  # efficiency = min(1, 0.08 / 0.05) = 1.0
            {"pnl_percent": 3.0, "mfe": 0.06},  # efficiency = 0.03 / 0.06 = 0.5
        ]
        result = self.analyzer.analyze_exit_timing_efficiency(trades)
        assert result["avg_exit_efficiency"] == pytest.approx(0.75)  # (1.0 + 0.5) / 2

    def test_analyze_exit_timing_efficiency_empty_data(self):
        """Test exit timing efficiency with empty data."""
        result = self.analyzer.analyze_exit_timing_efficiency([])
        assert result["avg_exit_efficiency"] == 0.0

    def test_analyze_exit_timing_efficiency_none_values(self):
        """Test exit timing efficiency with None values."""
        trades = [
            {"pnl_percent": None, "mfe": None},  # Should be treated as 0
            {"pnl_percent": 3.0, "mfe": 0.06},  # efficiency = 0.03 / 0.06 = 0.5
        ]
        result = self.analyzer.analyze_exit_timing_efficiency(trades)
        assert result["avg_exit_efficiency"] == pytest.approx(0.5)

    def test_identify_optimal_exit_points_basic(self):
        """Test identification of optimal exit points."""
        now = datetime.utcnow()
        trades = [
            {
                "strategy": "ml_basic",
                "mfe": 0.05,
                "mae": -0.02,
                "mfe_time": now,
                "mae_time": now,
            },
            {
                "strategy": "ml_adaptive",
                "mfe": 0.08,
                "mae": -0.03,
                "mfe_time": None,
                "mae_time": None,
            },
        ]

        result = self.analyzer.identify_optimal_exit_points(trades)
        assert len(result) == 2
        
        assert isinstance(result[0], TradeMFERecord)
        assert result[0].strategy_name == "ml_basic"
        assert result[0].mfe == pytest.approx(0.05)
        assert result[0].mae == pytest.approx(-0.02)
        assert result[0].mfe_time == now
        assert result[0].mae_time == now

        assert result[1].strategy_name == "ml_adaptive"
        assert result[1].mfe == pytest.approx(0.08)
        assert result[1].mae == pytest.approx(-0.03)
        assert result[1].mfe_time is None
        assert result[1].mae_time is None

    def test_identify_optimal_exit_points_empty_data(self):
        """Test identification of optimal exit points with empty data."""
        result = self.analyzer.identify_optimal_exit_points([])
        assert result == []

    def test_identify_optimal_exit_points_missing_fields(self):
        """Test identification of optimal exit points with missing fields."""
        trades = [
            {},  # Missing all fields
            {"strategy": "ml_basic"},  # Missing mfe/mae
        ]

        result = self.analyzer.identify_optimal_exit_points(trades)
        assert len(result) == 2
        
        assert result[0].strategy_name == ""
        assert result[0].mfe == pytest.approx(0.0)
        assert result[0].mae == pytest.approx(0.0)
        assert result[0].mfe_time is None
        assert result[0].mae_time is None

        assert result[1].strategy_name == "ml_basic"
        assert result[1].mfe == pytest.approx(0.0)
        assert result[1].mae == pytest.approx(0.0)

    def test_identify_optimal_exit_points_none_values(self):
        """Test identification of optimal exit points with None values."""
        trades = [
            {
                "strategy": None,
                "mfe": None,
                "mae": None,
                "mfe_time": None,
                "mae_time": None,
            },
        ]

        result = self.analyzer.identify_optimal_exit_points(trades)
        assert len(result) == 1
        assert result[0].strategy_name == "None"  # str(None) = "None"
        assert result[0].mfe == pytest.approx(0.0)
        assert result[0].mae == pytest.approx(0.0)


class TestTradeMFERecord:
    """Test TradeMFERecord dataclass."""

    def test_trade_mfe_record_creation(self):
        """Test TradeMFERecord creation and attributes."""
        now = datetime.utcnow()
        record = TradeMFERecord(
            strategy_name="ml_basic",
            mfe=0.05,
            mae=-0.02,
            mfe_time=now,
            mae_time=now,
        )

        assert record.strategy_name == "ml_basic"
        assert record.mfe == pytest.approx(0.05)
        assert record.mae == pytest.approx(-0.02)
        assert record.mfe_time == now
        assert record.mae_time == now

    def test_trade_mfe_record_none_values(self):
        """Test TradeMFERecord with None values."""
        record = TradeMFERecord(
            strategy_name="test",
            mfe=0.0,
            mae=0.0,
            mfe_time=None,
            mae_time=None,
        )

        assert record.strategy_name == "test"
        assert record.mfe == pytest.approx(0.0)
        assert record.mae == pytest.approx(0.0)
        assert record.mfe_time is None
        assert record.mae_time is None