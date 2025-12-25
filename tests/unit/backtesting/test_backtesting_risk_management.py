"""Risk management specific backtesting tests."""

from datetime import datetime

from src.engines.backtest.engine import Backtester
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy


class TestRiskManagementIntegration:
    """Confirm that risk settings are respected during runs."""

    def test_risk_parameters_integration(self, mock_data_provider, sample_ohlcv_data):
        """Running with explicit risk limits should succeed."""

        strategy = create_ml_basic_strategy()
        risk_params = RiskParameters(
            base_risk_per_trade=0.01,
            max_position_size=0.05,
            max_daily_risk=0.03,
        )

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
        assert "total_trades" in results

    def test_position_size_limits(self, mock_data_provider, sample_ohlcv_data):
        """Very small position limits should still run successfully."""

        strategy = create_ml_basic_strategy()
        risk_params = RiskParameters(max_position_size=0.01)

        mock_data_provider.get_historical_data.return_value = sample_ohlcv_data

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        results = backtester.run("BTCUSDT", "1h", datetime(2024, 1, 1))

        assert isinstance(results, dict)
