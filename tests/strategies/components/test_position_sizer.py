"""
Unit tests for PositionSizer components
"""

import pytest

from src.strategies.components.position_sizer import (
    ConfidenceWeightedSizer,
    FixedFractionSizer,
    KellySizer,
    PositionSizer,
    calculate_position_from_risk,
    calculate_risk_from_position,
    validate_position_size,
)
from src.strategies.components.signal_generator import Signal, SignalDirection


class MockPositionSizer(PositionSizer):
    """Mock position sizer for testing abstract base class"""
    
    def calculate_size(self, signal, balance, risk_amount, regime=None):
        return balance * 0.02


class TestPositionSizer:
    """Test PositionSizer abstract base class"""
    
    def create_test_signal(self, direction=SignalDirection.BUY, strength=0.8, confidence=0.9):
        """Create test signal"""
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={}
        )
    
    def test_position_sizer_initialization(self):
        """Test position sizer initialization"""
        sizer = MockPositionSizer("test_sizer")
        assert sizer.name == "test_sizer"
    
    def test_validate_inputs_valid(self):
        """Test input validation with valid inputs"""
        sizer = MockPositionSizer("test")
        
        # Should not raise any exception
        sizer.validate_inputs(10000.0, 200.0)
    
    def test_validate_inputs_invalid_balance(self):
        """Test input validation with invalid balance"""
        sizer = MockPositionSizer("test")
        
        with pytest.raises(ValueError, match="balance must be positive"):
            sizer.validate_inputs(-1000.0, 200.0)
        
        with pytest.raises(ValueError, match="balance must be positive"):
            sizer.validate_inputs(0.0, 200.0)
    
    def test_validate_inputs_invalid_risk_amount(self):
        """Test input validation with invalid risk amount"""
        sizer = MockPositionSizer("test")
        
        with pytest.raises(ValueError, match="risk_amount must be non-negative"):
            sizer.validate_inputs(10000.0, -200.0)
    
    def test_validate_inputs_risk_exceeds_balance(self):
        """Test input validation when risk exceeds balance"""
        sizer = MockPositionSizer("test")
        
        with pytest.raises(ValueError, match="risk_amount .* cannot exceed balance"):
            sizer.validate_inputs(1000.0, 2000.0)
    
    def test_apply_bounds_checking_within_bounds(self):
        """Test bounds checking with size within bounds"""
        sizer = MockPositionSizer("test")
        balance = 10000.0
        size = 500.0  # 5% of balance
        
        bounded_size = sizer.apply_bounds_checking(size, balance)
        
        assert bounded_size == size
    
    def test_apply_bounds_checking_below_minimum(self):
        """Test bounds checking with size below minimum"""
        sizer = MockPositionSizer("test")
        balance = 10000.0
        size = 5.0  # 0.05% of balance, below 0.1% minimum
        
        bounded_size = sizer.apply_bounds_checking(size, balance)
        
        assert bounded_size == 10.0  # 0.1% of balance
    
    def test_apply_bounds_checking_above_maximum(self):
        """Test bounds checking with size above maximum"""
        sizer = MockPositionSizer("test")
        balance = 10000.0
        size = 3000.0  # 30% of balance, above 20% maximum
        
        bounded_size = sizer.apply_bounds_checking(size, balance)
        
        assert bounded_size == 2000.0  # 20% of balance
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        sizer = MockPositionSizer("test_sizer")
        params = sizer.get_parameters()
        
        assert params['name'] == "test_sizer"
        assert params['type'] == "MockPositionSizer"


class TestFixedFractionSizer:
    """Test FixedFractionSizer implementation"""
    
    def create_test_signal(self, direction=SignalDirection.BUY, strength=0.8, confidence=0.9):
        """Create test signal"""
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={}
        )
    
    def test_fixed_fraction_sizer_initialization_default(self):
        """Test FixedFractionSizer initialization with defaults"""
        sizer = FixedFractionSizer()
        assert sizer.name == "fixed_fraction_sizer"
        assert sizer.fraction == 0.02
        assert sizer.adjust_for_confidence is True
        assert sizer.adjust_for_strength is True
    
    def test_fixed_fraction_sizer_initialization_custom(self):
        """Test FixedFractionSizer initialization with custom parameters"""
        sizer = FixedFractionSizer(
            fraction=0.05,
            adjust_for_confidence=False,
            adjust_for_strength=False
        )
        assert sizer.fraction == 0.05
        assert sizer.adjust_for_confidence is False
        assert sizer.adjust_for_strength is False
    
    def test_fixed_fraction_sizer_validation_fraction(self):
        """Test fraction validation"""
        with pytest.raises(ValueError, match="fraction must be between 0.001 and 0.5"):
            FixedFractionSizer(fraction=0.0005)
        
        with pytest.raises(ValueError, match="fraction must be between 0.001 and 0.5"):
            FixedFractionSizer(fraction=0.6)
    
    def test_calculate_size_buy_signal_with_adjustments(self):
        """Test position size calculation for BUY signal with adjustments"""
        sizer = FixedFractionSizer(fraction=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 200.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Base: 10000 * 0.02 = 200
        # Confidence adj: 200 * 0.9 = 180
        # Strength adj: 180 * 0.8 = 144
        assert abs(position_size - 144.0) < 0.01
    
    def test_calculate_size_buy_signal_without_adjustments(self):
        """Test position size calculation without adjustments"""
        sizer = FixedFractionSizer(
            fraction=0.02,
            adjust_for_confidence=False,
            adjust_for_strength=False
        )
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 200.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Base: 10000 * 0.02 = 200 (no adjustments)
        assert position_size == 200.0
    
    def test_calculate_size_hold_signal(self):
        """Test position size calculation for HOLD signal"""
        sizer = FixedFractionSizer()
        signal = self.create_test_signal(SignalDirection.HOLD, 0.0, 1.0)
        balance = 10000.0
        risk_amount = 200.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        assert position_size == 0.0
    
    def test_calculate_size_low_confidence_strength(self):
        """Test position size calculation with low confidence and strength"""
        sizer = FixedFractionSizer(fraction=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.1, 0.1)
        balance = 10000.0
        risk_amount = 200.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Base: 10000 * 0.02 = 200
        # Confidence adj: 200 * 0.2 = 40 (minimum 20%)
        # Strength adj: 40 * 0.2 = 8 (minimum 20%)
        # But bounds checking applies minimum of 0.1% = 10
        assert position_size == 10.0
    
    def test_calculate_size_risk_limit(self):
        """Test position size calculation with risk amount limit"""
        sizer = FixedFractionSizer(fraction=0.1)  # 10% fraction
        signal = self.create_test_signal(SignalDirection.BUY, 1.0, 1.0)
        balance = 10000.0
        risk_amount = 500.0  # Lower than calculated size
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Calculated would be 1000, but limited by risk_amount
        assert position_size == 500.0
    
    def test_calculate_size_zero_risk_amount(self):
        """Test position size calculation with zero risk amount (risk manager veto)"""
        sizer = FixedFractionSizer(fraction=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 0.0  # Risk manager vetoed the trade
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Should return 0.0 immediately, respecting risk manager decision
        assert position_size == 0.0
    
    def test_calculate_size_with_regime(self):
        """Test position size calculation with regime context"""
        from src.regime.detector import TrendLabel, VolLabel
        from src.strategies.components.regime_context import RegimeContext
        
        sizer = FixedFractionSizer(fraction=0.02)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 200.0
        
        # High volatility regime should reduce position size
        regime = RegimeContext(
            trend=TrendLabel.TREND_UP,
            volatility=VolLabel.HIGH,
            confidence=0.8,
            duration=10,
            strength=0.7
        )
        
        position_size = sizer.calculate_size(signal, balance, risk_amount, regime)
        
        # Should be reduced due to high volatility
        base_size = 144.0  # From previous test
        expected_size = base_size * 0.8  # High vol multiplier
        assert abs(position_size - expected_size) < 0.1
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        sizer = FixedFractionSizer(
            fraction=0.03,
            adjust_for_confidence=False,
            adjust_for_strength=True
        )
        params = sizer.get_parameters()
        
        assert params['name'] == "fixed_fraction_sizer"
        assert params['type'] == "FixedFractionSizer"
        assert params['fraction'] == 0.03
        assert params['adjust_for_confidence'] is False
        assert params['adjust_for_strength'] is True


class TestConfidenceWeightedSizer:
    """Test ConfidenceWeightedSizer implementation"""
    
    def create_test_signal(self, direction=SignalDirection.BUY, strength=0.8, confidence=0.9):
        """Create test signal"""
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={}
        )
    
    def test_confidence_weighted_sizer_initialization_default(self):
        """Test ConfidenceWeightedSizer initialization with defaults"""
        sizer = ConfidenceWeightedSizer()
        assert sizer.name == "confidence_weighted_sizer"
        assert sizer.base_fraction == 0.05
        assert sizer.min_confidence == 0.3
    
    def test_confidence_weighted_sizer_initialization_custom(self):
        """Test ConfidenceWeightedSizer initialization with custom parameters"""
        sizer = ConfidenceWeightedSizer(base_fraction=0.08, min_confidence=0.5)
        assert sizer.base_fraction == 0.08
        assert sizer.min_confidence == 0.5
    
    def test_confidence_weighted_sizer_validation_base_fraction(self):
        """Test base_fraction validation"""
        with pytest.raises(ValueError, match="base_fraction must be between 0.001 and 0.5"):
            ConfidenceWeightedSizer(base_fraction=0.0005)
        
        with pytest.raises(ValueError, match="base_fraction must be between 0.001 and 0.5"):
            ConfidenceWeightedSizer(base_fraction=0.6)
    
    def test_confidence_weighted_sizer_validation_min_confidence(self):
        """Test min_confidence validation"""
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            ConfidenceWeightedSizer(min_confidence=-0.1)
        
        with pytest.raises(ValueError, match="min_confidence must be between 0.0 and 1.0"):
            ConfidenceWeightedSizer(min_confidence=1.1)
    
    def test_calculate_size_high_confidence(self):
        """Test position size calculation with high confidence"""
        sizer = ConfidenceWeightedSizer(base_fraction=0.05, min_confidence=0.3)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 1000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Base: 10000 * 0.05 * 0.9 = 450
        # Strength adj: 450 * 0.8 = 360
        assert position_size == 360.0
    
    def test_calculate_size_below_min_confidence(self):
        """Test position size calculation below minimum confidence"""
        sizer = ConfidenceWeightedSizer(base_fraction=0.05, min_confidence=0.5)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.4)  # Below min
        balance = 10000.0
        risk_amount = 1000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        assert position_size == 0.0
    
    def test_calculate_size_hold_signal(self):
        """Test position size calculation for HOLD signal"""
        sizer = ConfidenceWeightedSizer()
        signal = self.create_test_signal(SignalDirection.HOLD, 0.0, 1.0)
        balance = 10000.0
        risk_amount = 1000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        assert position_size == 0.0
    
    def test_calculate_size_zero_risk_amount(self):
        """Test position size calculation with zero risk amount (risk manager veto)"""
        sizer = ConfidenceWeightedSizer()
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 0.0  # Risk manager vetoed the trade
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Should return 0.0 immediately, respecting risk manager decision
        assert position_size == 0.0
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        sizer = ConfidenceWeightedSizer(base_fraction=0.08, min_confidence=0.4)
        params = sizer.get_parameters()
        
        assert params['name'] == "confidence_weighted_sizer"
        assert params['type'] == "ConfidenceWeightedSizer"
        assert params['base_fraction'] == 0.08
        assert params['min_confidence'] == 0.4


class TestKellySizer:
    """Test KellySizer implementation"""
    
    def create_test_signal(self, direction=SignalDirection.BUY, strength=0.8, confidence=0.9):
        """Create test signal"""
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={}
        )
    
    def test_kelly_sizer_initialization_default(self):
        """Test KellySizer initialization with defaults"""
        sizer = KellySizer()
        assert sizer.name == "kelly_sizer"
        assert sizer.win_rate == 0.55
        assert sizer.avg_win == 0.02
        assert sizer.avg_loss == 0.015
        assert sizer.kelly_fraction == 0.25
        assert sizer.lookback_period == 100
    
    def test_kelly_sizer_initialization_custom(self):
        """Test KellySizer initialization with custom parameters"""
        sizer = KellySizer(
            win_rate=0.6,
            avg_win=0.03,
            avg_loss=0.02,
            kelly_fraction=0.5,
            lookback_period=200
        )
        assert sizer.win_rate == 0.6
        assert sizer.avg_win == 0.03
        assert sizer.avg_loss == 0.02
        assert sizer.kelly_fraction == 0.5
        assert sizer.lookback_period == 200
    
    def test_kelly_sizer_validation_win_rate(self):
        """Test win_rate validation"""
        with pytest.raises(ValueError, match="win_rate must be between 0.1 and 0.9"):
            KellySizer(win_rate=0.05)
        
        with pytest.raises(ValueError, match="win_rate must be between 0.1 and 0.9"):
            KellySizer(win_rate=0.95)
    
    def test_kelly_sizer_validation_avg_win(self):
        """Test avg_win validation"""
        with pytest.raises(ValueError, match="avg_win must be positive"):
            KellySizer(avg_win=-0.01)
    
    def test_kelly_sizer_validation_avg_loss(self):
        """Test avg_loss validation"""
        with pytest.raises(ValueError, match="avg_loss must be positive"):
            KellySizer(avg_loss=-0.01)
    
    def test_kelly_sizer_validation_kelly_fraction(self):
        """Test kelly_fraction validation"""
        with pytest.raises(ValueError, match="kelly_fraction must be between 0.01 and 1.0"):
            KellySizer(kelly_fraction=0.005)
        
        with pytest.raises(ValueError, match="kelly_fraction must be between 0.01 and 1.0"):
            KellySizer(kelly_fraction=1.1)
    
    def test_calculate_kelly_percentage(self):
        """Test Kelly percentage calculation"""
        sizer = KellySizer(win_rate=0.6, avg_win=0.02, avg_loss=0.015)
        
        kelly_pct = sizer._calculate_kelly_percentage()
        
        # Kelly formula: f = (bp - q) / b
        # b = 0.02 / 0.015 = 1.333
        # p = 0.6, q = 0.4
        # f = (1.333 * 0.6 - 0.4) / 1.333 = 0.3
        expected_kelly = 0.3
        assert abs(kelly_pct - expected_kelly) < 0.01
    
    def test_calculate_size_buy_signal(self):
        """Test position size calculation for BUY signal"""
        sizer = KellySizer(
            win_rate=0.6,
            avg_win=0.02,
            avg_loss=0.015,
            kelly_fraction=0.5
        )
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 2000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Kelly ~0.3, fractional Kelly = 0.3 * 0.5 = 0.15
        # Adjusted: 0.15 * 0.9 * 0.8 = 0.108
        # Position: 10000 * 0.108 = 1080
        # But capped at 15% = 1500
        assert 1000 <= position_size <= 1500
    
    def test_calculate_size_hold_signal(self):
        """Test position size calculation for HOLD signal"""
        sizer = KellySizer()
        signal = self.create_test_signal(SignalDirection.HOLD, 0.0, 1.0)
        balance = 10000.0
        risk_amount = 1000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        assert position_size == 0.0
    
    def test_calculate_size_zero_risk_amount(self):
        """Test position size calculation with zero risk amount (risk manager veto)"""
        sizer = KellySizer()
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 0.0  # Risk manager vetoed the trade
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Should return 0.0 immediately, respecting risk manager decision
        assert position_size == 0.0
    
    def test_update_trade_result_win(self):
        """Test updating trade result with win"""
        sizer = KellySizer()
        initial_win_rate = sizer.win_rate
        
        # Add some wins
        for _ in range(25):
            sizer.update_trade_result(True, 0.02)
        
        # Win rate should increase (but need enough trades to trigger update)
        assert len(sizer.trade_history) == 25
        assert sizer.win_rate >= initial_win_rate  # May not change until 20+ trades
    
    def test_update_trade_result_loss(self):
        """Test updating trade result with loss"""
        sizer = KellySizer()
        
        # Add some losses
        for _ in range(25):
            sizer.update_trade_result(False, 0.015)
        
        assert len(sizer.trade_history) == 25
    
    def test_trade_history_limit(self):
        """Test trade history is limited to lookback period"""
        sizer = KellySizer(lookback_period=50)
        
        # Add more trades than lookback period
        for i in range(100):
            sizer.update_trade_result(i % 2 == 0, 0.02)
        
        assert len(sizer.trade_history) == 50
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        sizer = KellySizer(win_rate=0.6, avg_win=0.03, kelly_fraction=0.3)
        params = sizer.get_parameters()
        
        assert params['name'] == "kelly_sizer"
        assert params['type'] == "KellySizer"
        assert params['win_rate'] == 0.6
        assert params['avg_win'] == 0.03
        assert params['kelly_fraction'] == 0.3
        assert params['trade_count'] == 0


class TestRegimeAdaptiveSizer:
    """Test RegimeAdaptiveSizer implementation"""
    
    def create_test_signal(self, direction=SignalDirection.BUY, strength=0.8, confidence=0.9):
        """Create test signal"""
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={}
        )
    
    def create_test_regime(self, trend='trend_up', volatility='low_vol', confidence=0.8, 
                          duration=10, strength=0.7):
        """Create test regime context"""
        from src.regime.detector import TrendLabel, VolLabel
        from src.strategies.components.regime_context import RegimeContext
        
        trend_map = {
            'trend_up': TrendLabel.TREND_UP,
            'trend_down': TrendLabel.TREND_DOWN,
            'range': TrendLabel.RANGE
        }
        
        vol_map = {
            'low_vol': VolLabel.LOW,
            'high_vol': VolLabel.HIGH
        }
        
        return RegimeContext(
            trend=trend_map.get(trend, TrendLabel.RANGE),
            volatility=vol_map.get(volatility, VolLabel.LOW),
            confidence=confidence,
            duration=duration,
            strength=strength
        )
    
    def test_regime_adaptive_sizer_initialization_default(self):
        """Test RegimeAdaptiveSizer initialization with defaults"""
        sizer = RegimeAdaptiveSizer()
        assert sizer.name == "regime_adaptive_sizer"
        assert sizer.base_fraction == 0.03
        assert sizer.volatility_adjustment is True
        assert 'bull_low_vol' in sizer.regime_multipliers
        assert sizer.regime_multipliers['bull_low_vol'] == 1.8
    
    def test_regime_adaptive_sizer_initialization_custom(self):
        """Test RegimeAdaptiveSizer initialization with custom parameters"""
        custom_multipliers = {
            'bull_low_vol': 2.0,
            'bear_high_vol': 0.1
        }
        sizer = RegimeAdaptiveSizer(
            base_fraction=0.05,
            regime_multipliers=custom_multipliers,
            volatility_adjustment=False
        )
        assert sizer.base_fraction == 0.05
        assert sizer.volatility_adjustment is False
        assert sizer.regime_multipliers['bull_low_vol'] == 2.0
        assert sizer.regime_multipliers['bear_high_vol'] == 0.1
    
    def test_regime_adaptive_sizer_validation_base_fraction(self):
        """Test base_fraction validation"""
        with pytest.raises(ValueError, match="base_fraction must be between 0.001 and 0.2"):
            RegimeAdaptiveSizer(base_fraction=0.0005)
        
        with pytest.raises(ValueError, match="base_fraction must be between 0.001 and 0.2"):
            RegimeAdaptiveSizer(base_fraction=0.25)
    
    def test_regime_adaptive_sizer_validation_multipliers(self):
        """Test regime multiplier validation"""
        with pytest.raises(ValueError, match="regime multiplier.*must be between 0.1 and 3.0"):
            RegimeAdaptiveSizer(regime_multipliers={'test': 5.0})
    
    def test_calculate_size_bull_low_vol(self):
        """Test position size calculation in bull low volatility regime"""
        sizer = RegimeAdaptiveSizer(base_fraction=0.03)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        regime = self.create_test_regime('trend_up', 'low_vol', 0.8, 10, 0.8)
        balance = 10000.0
        risk_amount = 2000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount, regime)
        
        # Base: 10000 * 0.03 = 300
        # Regime multiplier: 300 * 1.8 = 540
        # Confidence adj: 540 * 0.9 = 486
        # Strength adj: 486 * 0.8 = 388.8
        # Regime confidence: 388.8 * 0.8 = 311.04
        # Volatility adj: 311.04 * 1.1 = 342.144 (strength >= 0.8)
        assert abs(position_size - 342.144) < 1.0
    
    def test_calculate_size_bear_high_vol(self):
        """Test position size calculation in bear high volatility regime"""
        sizer = RegimeAdaptiveSizer(base_fraction=0.03)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        regime = self.create_test_regime('trend_down', 'high_vol', 0.8, 10, 0.3)
        balance = 10000.0
        risk_amount = 2000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount, regime)
        
        # Base: 10000 * 0.03 = 300
        # Regime multiplier: 300 * 0.2 = 60
        # Confidence adj: 60 * 0.9 = 54
        # Strength adj: 54 * 0.8 = 43.2
        # Regime confidence: 43.2 * 0.8 = 34.56
        # Volatility adj: 34.56 * 0.7 = 24.192 (strength < 0.4)
        # But minimum bounds checking applies
        min_size = balance * 0.001  # 10
        assert position_size >= min_size
    
    def test_calculate_size_no_regime(self):
        """Test position size calculation without regime context"""
        sizer = RegimeAdaptiveSizer(base_fraction=0.03)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        balance = 10000.0
        risk_amount = 2000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount)
        
        # Base: 10000 * 0.03 = 300
        # Regime multiplier: 300 * 0.5 = 150 (unknown regime)
        # Confidence adj: 150 * 0.9 = 135
        # Strength adj: 135 * 0.8 = 108
        assert position_size == 108.0
    
    def test_calculate_size_hold_signal(self):
        """Test position size calculation for HOLD signal"""
        sizer = RegimeAdaptiveSizer()
        signal = self.create_test_signal(SignalDirection.HOLD, 0.0, 1.0)
        regime = self.create_test_regime('trend_up', 'low_vol', 0.8, 10, 0.8)
        balance = 10000.0
        risk_amount = 2000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount, regime)
        
        assert position_size == 0.0
    
    def test_get_regime_allocation_bull_low_vol(self):
        """Test regime allocation breakdown for bull low vol"""
        sizer = RegimeAdaptiveSizer()
        regime = self.create_test_regime('trend_up', 'low_vol', 0.8, 10, 0.8)
        
        allocation = sizer.get_regime_allocation(regime)
        
        assert allocation['regime_key'] == 'bull_low_vol'
        assert allocation['regime_multiplier'] == 1.8
        assert allocation['volatility_adjustment'] == 1.1  # strength >= 0.8
        assert allocation['max_fraction'] == 0.25
    
    def test_get_regime_allocation_bear_high_vol(self):
        """Test regime allocation breakdown for bear high vol"""
        sizer = RegimeAdaptiveSizer()
        regime = self.create_test_regime('trend_down', 'high_vol', 0.6, 5, 0.3)
        
        allocation = sizer.get_regime_allocation(regime)
        
        assert allocation['regime_key'] == 'bear_high_vol'
        assert allocation['regime_multiplier'] == 0.2
        assert allocation['volatility_adjustment'] == 0.7  # strength < 0.4
        assert allocation['max_fraction'] == 0.08
    
    def test_get_regime_allocation_no_regime(self):
        """Test regime allocation breakdown without regime"""
        sizer = RegimeAdaptiveSizer()
        
        allocation = sizer.get_regime_allocation(None)
        
        assert allocation['regime_key'] == 'unknown'
        assert allocation['regime_multiplier'] == 0.5
        assert allocation['volatility_adjustment'] == 1.0
        assert allocation['max_fraction'] == 0.1
    
    def test_update_regime_multipliers(self):
        """Test updating regime multipliers"""
        sizer = RegimeAdaptiveSizer()
        
        new_multipliers = {
            'bull_low_vol': 2.5,
            'bear_high_vol': 0.15
        }
        
        sizer.update_regime_multipliers(new_multipliers)
        
        assert sizer.regime_multipliers['bull_low_vol'] == 2.5
        assert sizer.regime_multipliers['bear_high_vol'] == 0.15
    
    def test_update_regime_multipliers_validation(self):
        """Test regime multiplier update validation"""
        sizer = RegimeAdaptiveSizer()
        
        with pytest.raises(ValueError, match="regime multiplier.*must be between 0.1 and 3.0"):
            sizer.update_regime_multipliers({'test': 5.0})
    
    def test_volatility_adjustment_disabled(self):
        """Test position sizing with volatility adjustment disabled"""
        sizer = RegimeAdaptiveSizer(base_fraction=0.03, volatility_adjustment=False)
        signal = self.create_test_signal(SignalDirection.BUY, 0.8, 0.9)
        regime = self.create_test_regime('trend_up', 'low_vol', 0.8, 10, 0.3)  # Low strength
        balance = 10000.0
        risk_amount = 2000.0
        
        position_size = sizer.calculate_size(signal, balance, risk_amount, regime)
        
        # Should not apply volatility adjustment (strength-based)
        # Base: 10000 * 0.03 = 300
        # Regime multiplier: 300 * 1.8 = 540
        # Confidence adj: 540 * 0.9 = 486
        # Strength adj: 486 * 0.8 = 388.8
        # Regime confidence: 388.8 * 0.8 = 311.04
        # No volatility adjustment
        assert abs(position_size - 311.04) < 1.0
    
    def test_get_parameters(self):
        """Test get_parameters method"""
        custom_multipliers = {'bull_low_vol': 2.0}
        sizer = RegimeAdaptiveSizer(
            base_fraction=0.05,
            regime_multipliers=custom_multipliers,
            volatility_adjustment=False
        )
        params = sizer.get_parameters()
        
        assert params['name'] == "regime_adaptive_sizer"
        assert params['type'] == "RegimeAdaptiveSizer"
        assert params['base_fraction'] == 0.05
        assert params['volatility_adjustment'] is False
        assert params['regime_multipliers']['bull_low_vol'] == 2.0


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_calculate_position_from_risk_valid(self):
        """Test position calculation from risk amount"""
        risk_amount = 1000.0
        entry_price = 50000.0
        stop_loss_price = 47500.0  # 5% stop loss
        
        position_size = calculate_position_from_risk(risk_amount, entry_price, stop_loss_price)
        
        # Risk per unit = 50000 - 47500 = 2500
        # Position size = 1000 / 2500 = 0.4
        assert position_size == 0.4
    
    def test_calculate_position_from_risk_zero_risk(self):
        """Test position calculation with zero risk"""
        position_size = calculate_position_from_risk(0.0, 50000.0, 47500.0)
        assert position_size == 0.0
    
    def test_calculate_position_from_risk_invalid_prices(self):
        """Test position calculation with invalid prices"""
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_position_from_risk(1000.0, -50000.0, 47500.0)
        
        with pytest.raises(ValueError, match="Prices must be positive"):
            calculate_position_from_risk(1000.0, 50000.0, -47500.0)
    
    def test_calculate_risk_from_position_valid(self):
        """Test risk calculation from position size"""
        position_size = 0.4
        entry_price = 50000.0
        stop_loss_price = 47500.0
        
        risk_amount = calculate_risk_from_position(position_size, entry_price, stop_loss_price)
        
        # Risk per unit = 50000 - 47500 = 2500
        # Total risk = 0.4 * 2500 = 1000
        assert risk_amount == 1000.0
    
    def test_calculate_risk_from_position_zero_position(self):
        """Test risk calculation with zero position"""
        risk_amount = calculate_risk_from_position(0.0, 50000.0, 47500.0)
        assert risk_amount == 0.0
    
    def test_validate_position_size_valid(self):
        """Test position size validation with valid size"""
        assert validate_position_size(500.0, 10000.0) is True  # 5%
    
    def test_validate_position_size_too_small(self):
        """Test position size validation with too small size"""
        assert validate_position_size(5.0, 10000.0) is False  # 0.05%
    
    def test_validate_position_size_too_large(self):
        """Test position size validation with too large size"""
        assert validate_position_size(3000.0, 10000.0) is False  # 30%
    
    def test_validate_position_size_negative(self):
        """Test position size validation with negative size"""
        assert validate_position_size(-100.0, 10000.0) is False
    
    def test_validate_position_size_invalid_balance(self):
        """Test position size validation with invalid balance"""
        assert validate_position_size(100.0, -10000.0) is False