"""Unit tests for HyperGrowth strategy components."""

from unittest.mock import MagicMock, patch

import pytest

from src.strategies.components import Signal, SignalDirection
from src.strategies.components.ml_signal_generator import MLBasicSignalGenerator
from src.strategies.hyper_growth import FlatRiskManager, create_hyper_growth_strategy

pytestmark = pytest.mark.unit


class TestFlatRiskManager:
    """Test FlatRiskManager returns fixed risk without confidence scaling."""

    def test_flat_risk_initialization(self):
        """Test FlatRiskManager initialization with valid parameters."""
        risk_manager = FlatRiskManager(
            risk_fraction=0.10,
            stop_loss_pct=0.10,
            min_confidence=0.03,
        )

        assert risk_manager.risk_fraction == 0.10
        assert risk_manager.stop_loss_pct == 0.10
        assert risk_manager.min_confidence == 0.03

    def test_flat_risk_initialization_invalid_risk_fraction(self):
        """Test FlatRiskManager rejects invalid risk_fraction."""
        with pytest.raises(ValueError, match="risk_fraction must be between"):
            FlatRiskManager(risk_fraction=0.005)  # Too low

        with pytest.raises(ValueError, match="risk_fraction must be between"):
            FlatRiskManager(risk_fraction=0.60)  # Too high

    def test_calculate_position_size_fixed_risk(self):
        """Test position size is balance * risk_fraction without scaling."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.30,  # Above min_confidence threshold of 0.20
            strength=0.3,
            metadata={},
        )

        balance = 10000.0
        position_size = risk_manager.calculate_position_size(signal=signal, balance=balance)

        # Fixed 10% of balance regardless of confidence
        assert position_size == 1000.0

    def test_calculate_position_size_filters_low_confidence(self):
        """Test positions below min_confidence are filtered out."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10, min_confidence=0.05)

        # Signal below min_confidence threshold
        low_conf_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.03,  # Below 0.05 threshold
            strength=0.3,
            metadata={},
        )

        balance = 10000.0
        position_size = risk_manager.calculate_position_size(
            signal=low_conf_signal, balance=balance
        )

        # Should return 0 for signals below min_confidence
        assert position_size == 0.0

    def test_calculate_position_size_hold_signal(self):
        """Test HOLD signals return 0 position size."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        hold_signal = Signal(
            direction=SignalDirection.HOLD,
            confidence=0.10,
            strength=0.0,
            metadata={},
        )

        balance = 10000.0
        position_size = risk_manager.calculate_position_size(signal=hold_signal, balance=balance)

        assert position_size == 0.0

    def test_calculate_position_size_sell_signal(self):
        """Test SELL signals get full risk allocation."""
        risk_manager = FlatRiskManager(risk_fraction=0.20, stop_loss_pct=0.15)

        sell_signal = Signal(
            direction=SignalDirection.SELL,
            confidence=0.30,  # Above min_confidence threshold of 0.20
            strength=0.4,
            metadata={},
        )

        balance = 5000.0
        position_size = risk_manager.calculate_position_size(signal=sell_signal, balance=balance)

        # 20% of balance for short positions too
        assert position_size == 1000.0

    def test_should_exit_on_stop_loss(self):
        """Test exit when unrealized loss exceeds stop_loss_pct."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        # Mock position with 15% loss (exceeds 10% stop)
        class MockPosition:
            unrealized_pnl_pct = -0.15

        position = MockPosition()
        should_exit = risk_manager.should_exit(position=position, current_data=None)

        assert should_exit is True

    def test_should_not_exit_within_stop_loss(self):
        """Test no exit when unrealized loss is within stop_loss_pct."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        # Mock position with 5% loss (within 10% stop)
        class MockPosition:
            unrealized_pnl_pct = -0.05

        position = MockPosition()
        should_exit = risk_manager.should_exit(position=position, current_data=None)

        assert should_exit is False

    def test_should_exit_on_profit(self):
        """Test no exit when position is profitable."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        # Mock position with 5% profit
        class MockPosition:
            unrealized_pnl_pct = 0.05

        position = MockPosition()
        should_exit = risk_manager.should_exit(position=position, current_data=None)

        assert should_exit is False

    def test_should_exit_handles_missing_pnl_attr(self):
        """Test should_exit handles positions without unrealized_pnl_pct."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        # Position without unrealized_pnl_pct attribute
        class MockPosition:
            entry_price = 50000.0

        position = MockPosition()
        should_exit = risk_manager.should_exit(position=position, current_data=None)

        assert should_exit is False

    def test_get_stop_loss_buy_direction(self):
        """Test stop loss calculation for BUY (long) positions."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        buy_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.10,
            strength=0.5,
            metadata={},
        )

        entry_price = 50000.0
        stop_loss = risk_manager.get_stop_loss(entry_price=entry_price, signal=buy_signal)

        # 10% below entry for long
        assert stop_loss == 45000.0

    def test_get_stop_loss_sell_direction(self):
        """Test stop loss calculation for SELL (short) positions."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        sell_signal = Signal(
            direction=SignalDirection.SELL,
            confidence=0.10,
            strength=0.5,
            metadata={},
        )

        entry_price = 50000.0
        stop_loss = risk_manager.get_stop_loss(entry_price=entry_price, signal=sell_signal)

        # 10% above entry for short
        assert stop_loss == pytest.approx(55000.0)

    def test_get_stop_loss_invalid_entry_price(self):
        """Test stop calculation rejects non-positive entry prices."""
        risk_manager = FlatRiskManager(risk_fraction=0.10, stop_loss_pct=0.10)

        buy_signal = Signal(
            direction=SignalDirection.BUY,
            confidence=0.10,
            strength=0.5,
            metadata={},
        )

        with pytest.raises(ValueError, match="entry_price must be positive"):
            risk_manager.get_stop_loss(entry_price=0.0, signal=buy_signal)

        with pytest.raises(ValueError, match="entry_price must be positive"):
            risk_manager.get_stop_loss(entry_price=-100.0, signal=buy_signal)

    def test_get_parameters(self):
        """Test get_parameters returns risk manager configuration."""
        risk_manager = FlatRiskManager(risk_fraction=0.15, stop_loss_pct=0.12, min_confidence=0.04)

        params = risk_manager.get_parameters()

        assert params == {
            "risk_fraction": 0.15,
            "stop_loss_pct": 0.12,
        }


class TestHyperGrowthStrategy:
    """Test HyperGrowth strategy creation and configuration."""

    def test_create_hyper_growth_strategy_default_params(self):
        """Test strategy creation with default parameters."""
        strategy = create_hyper_growth_strategy()

        assert strategy.name == "HyperGrowth"
        assert hasattr(strategy, "leverage_manager")
        assert getattr(strategy, "base_position_size", None) == 0.25
        assert getattr(strategy, "take_profit_pct", None) == 0.30

    def test_default_signal_generator_uses_basic_model(self):
        """Regression guard: the factory must wire the basic ML model.

        Prior to the signal-fix commit the factory installed
        ``MLBasicSignalGenerator(model_type="sentiment")``. The sentiment
        bundle expects 10 features, but MLBasicSignalGenerator only feeds
        5 (OHLCV via PriceOnlyFeatureExtractor), so the model silently
        returned 0.0 on every bar — which the generator converted to a
        constant SELL sentinel. Asserting the default here prevents a
        silent reversion.
        """
        strategy = create_hyper_growth_strategy()

        sg = strategy.signal_generator
        assert isinstance(sg, MLBasicSignalGenerator)
        assert sg.model_type == "basic"

    def test_create_hyper_growth_strategy_custom_params(self):
        """Test strategy creation with custom parameters."""
        strategy = create_hyper_growth_strategy(
            name="CustomHyperGrowth",
            risk_fraction=0.15,
            base_fraction=0.15,
            max_leverage=2.5,
            stop_loss_pct=0.15,
        )

        assert strategy.name == "CustomHyperGrowth"
        assert getattr(strategy, "base_position_size", None) == 0.15
        assert getattr(strategy, "take_profit_pct", None) == 0.30

    def test_strategy_has_ignore_signal_reversal_metadata(self):
        """Test strategy sets ignore_signal_reversal metadata flag."""
        strategy = create_hyper_growth_strategy()

        # This metadata flag tells the engine to hold positions through signal flips
        assert getattr(strategy, "_extra_metadata", {}).get("ignore_signal_reversal") is True

    def test_strategy_risk_overrides_configured(self):
        """Test strategy has proper risk overrides configured."""
        strategy = create_hyper_growth_strategy()

        overrides = strategy._risk_overrides

        assert overrides["position_sizer"] == "leveraged_fixed_fraction"
        assert overrides["base_fraction"] == 0.25
        assert overrides["max_fraction"] == 0.25  # 0.25 * 1.0 (leverage disabled by default)
        assert overrides["stop_loss_pct"] == 0.10
        assert overrides["take_profit_pct"] == 0.30

    def test_strategy_leverage_configured(self):
        """Test strategy has leverage disabled by default (max_leverage=1.0)."""
        strategy = create_hyper_growth_strategy()

        assert strategy._risk_overrides["leverage"]["enabled"] is True
        assert strategy._risk_overrides["leverage"]["max_leverage"] == 1.0

    def test_strategy_dynamic_risk_configured(self):
        """Test strategy has dynamic risk management enabled."""
        strategy = create_hyper_growth_strategy()

        assert strategy._risk_overrides["dynamic_risk"]["enabled"] is True
        assert strategy._risk_overrides["dynamic_risk"]["drawdown_thresholds"] == [
            0.15,
            0.30,
            0.45,
        ]

    def test_strategy_partial_operations_configured(self):
        """Test strategy has partial exit operations configured."""
        strategy = create_hyper_growth_strategy()

        assert strategy._risk_overrides["partial_operations"]["exit_targets"] == [
            0.08,
            0.15,
            0.30,
        ]
        assert strategy._risk_overrides["partial_operations"]["exit_sizes"] == [
            0.20,
            0.30,
            0.50,
        ]

    def test_strategy_trailing_stop_configured(self):
        """Test strategy has trailing stop configured."""
        strategy = create_hyper_growth_strategy()

        trailing = strategy._risk_overrides["trailing_stop"]
        assert trailing["activation_threshold"] == 0.03
        assert trailing["trailing_distance_pct"] == 0.015
        assert trailing["breakeven_threshold"] == 0.05

    def test_create_strategy_with_momentum_signals(self):
        """Test strategy can be created with momentum signal source."""
        strategy = create_hyper_growth_strategy(signal_source="momentum")

        assert strategy.name == "HyperGrowth"
        # Momentum source should work without ML models


class TestHyperGrowthExitPolicyExpressibility:
    """#971: trailing/breakeven/early-cut must be factory-configurable."""

    def test_default_overrides_dict_is_bit_identical_to_pre_971(self):
        """Zero behavior change: the default factory must emit EXACTLY the
        overrides dict it emitted before the expressibility kwargs existed."""
        strategy = create_hyper_growth_strategy()

        assert strategy._risk_overrides["trailing_stop"] == {
            "activation_threshold": 0.03,
            "trailing_distance_pct": 0.015,
            "breakeven_threshold": 0.05,
            "breakeven_buffer": 0.008,
        }
        assert "early_cut" not in strategy._risk_overrides

    def test_trailing_kwargs_flow_into_overrides(self):
        strategy = create_hyper_growth_strategy(
            trailing_activation_threshold=0.05,
            trailing_distance_pct=0.025,
            breakeven_threshold=0.08,
            breakeven_buffer=0.01,
        )

        assert strategy._risk_overrides["trailing_stop"] == {
            "activation_threshold": 0.05,
            "trailing_distance_pct": 0.025,
            "breakeven_threshold": 0.08,
            "breakeven_buffer": 0.01,
        }

    def test_trailing_can_be_disabled(self):
        strategy = create_hyper_growth_strategy(enable_trailing_stop=False)

        assert strategy._risk_overrides["trailing_stop"] == {"enabled": False}

    def test_disabled_trailing_builds_no_policy(self):
        """End to end: the shared builder honors the disabled config."""
        from src.engines.shared.risk_configuration import build_trailing_stop_policy
        from src.risk.risk_manager import RiskManager, RiskParameters

        strategy = create_hyper_growth_strategy(enable_trailing_stop=False)

        assert build_trailing_stop_policy(strategy, RiskManager(RiskParameters())) is None

    def test_early_cut_kwargs_flow_into_overrides(self):
        strategy = create_hyper_growth_strategy(
            early_cut_mfe_threshold_pct=0.015,
            early_cut_evaluation_window_hours=18,
        )

        assert strategy._risk_overrides["early_cut"] == {
            "mfe_threshold_pct": 0.015,
            "evaluation_window_hours": 18,
        }

    def test_early_cut_builds_policy_via_shared_builder(self):
        from src.engines.shared.risk_configuration import build_early_cut_policy
        from src.position_management.early_cut import EarlyCutPolicy

        strategy = create_hyper_growth_strategy(
            early_cut_mfe_threshold_pct=0.015,
            early_cut_evaluation_window_hours=18,
        )
        policy = build_early_cut_policy(strategy)

        assert isinstance(policy, EarlyCutPolicy)
        assert policy.mfe_threshold_pct == 0.015
        assert policy.evaluation_window_hours == 18

    def test_early_cut_default_off(self):
        from src.engines.shared.risk_configuration import build_early_cut_policy

        assert build_early_cut_policy(create_hyper_growth_strategy()) is None

    def test_partial_early_cut_kwargs_raise(self):
        """Both early-cut kwargs are required together — a silent partial
        config would run with an unintended default."""
        import pytest

        with pytest.raises(ValueError):
            create_hyper_growth_strategy(early_cut_mfe_threshold_pct=0.015)
        with pytest.raises(ValueError):
            create_hyper_growth_strategy(early_cut_evaluation_window_hours=18)

    def test_default_trailing_policy_unchanged_via_builder(self):
        """The built TrailingStopPolicy for the default factory + default
        RiskParameters must keep the exact pre-971 field values."""
        from src.engines.shared.risk_configuration import build_trailing_stop_policy
        from src.risk.risk_manager import RiskManager, RiskParameters

        policy = build_trailing_stop_policy(
            create_hyper_growth_strategy(), RiskManager(RiskParameters())
        )

        assert policy is not None
        assert policy.activation_threshold == 0.03
        assert policy.trailing_distance_pct == 0.015
        assert policy.atr_multiplier == 1.5  # params fallback leak, pinned
        assert policy.breakeven_threshold == 0.05
        assert policy.breakeven_buffer == 0.008


class TestHyperGrowthModelVersionPin:
    """Factory threads the point-in-time model pin to the signal generator (GH #988)."""

    PINNED_VERSION = "2026-01-01_1h_v1"

    def _mock_engine(self, mock_engine_class):
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        mock_registry = MagicMock()
        pinned_bundle = MagicMock()
        pinned_bundle.key = f"ETHUSDT:1h:basic:{self.PINNED_VERSION}"
        pinned_bundle.symbol = "ETHUSDT"
        mock_registry.get_bundle_by_key.return_value = pinned_bundle
        mock_engine.model_registry = mock_registry
        mock_engine_class.return_value = mock_engine

    @patch("src.strategies.components.ml_signal_generator.PredictionEngine")
    @patch("src.strategies.components.ml_signal_generator.PredictionConfig")
    def test_factory_threads_model_version_to_signal_generator(
        self, mock_config_class, mock_engine_class
    ):
        self._mock_engine(mock_engine_class)

        strategy = create_hyper_growth_strategy(symbol="ETHUSDT", model_version=self.PINNED_VERSION)

        sg = strategy.signal_generator
        assert isinstance(sg, MLBasicSignalGenerator)
        assert sg.pinned_model_key == f"ETHUSDT:1h:basic:{self.PINNED_VERSION}"

    def test_momentum_signal_source_rejects_model_version(self):
        with pytest.raises(ValueError, match="model_version"):
            create_hyper_growth_strategy(
                signal_source="momentum", model_version=self.PINNED_VERSION
            )
