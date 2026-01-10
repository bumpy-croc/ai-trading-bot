"""Tests covering Backtester initialization scenarios."""

from src.engines.backtest.engine import Backtester
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import create_ml_basic_strategy


class TestBacktesterInitialization:
    """Validate Backtester setup paths."""

    def test_backtester_initialization(self, mock_data_provider):
        """Ensure explicit constructor arguments are wired correctly."""

        strategy = create_ml_basic_strategy()
        risk_params = RiskParameters()

        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            risk_parameters=risk_params,
            initial_balance=10000,
        )

        assert backtester.strategy == strategy
        assert backtester.data_provider == mock_data_provider
        assert backtester.risk_parameters == risk_params
        assert backtester.initial_balance == 10000
        assert backtester.balance == 10000
        assert len(backtester.trades) == 0
        assert backtester.current_trade is None

    def test_backtester_with_default_parameters(self, mock_data_provider):
        """Backtester should populate defaults when optional args omitted."""

        strategy = create_ml_basic_strategy()
        backtester = Backtester(strategy=strategy, data_provider=mock_data_provider)

        assert backtester.strategy == strategy
        assert backtester.data_provider == mock_data_provider
        assert backtester.initial_balance > 0
        assert backtester.balance == backtester.initial_balance

    def test_backtester_with_sentiment_provider(self, mock_data_provider, mock_sentiment_provider):
        """Sentiment providers should be captured when supplied."""

        strategy = create_ml_basic_strategy()
        backtester = Backtester(
            strategy=strategy,
            data_provider=mock_data_provider,
            sentiment_provider=mock_sentiment_provider,
        )

        assert backtester.sentiment_provider == mock_sentiment_provider
