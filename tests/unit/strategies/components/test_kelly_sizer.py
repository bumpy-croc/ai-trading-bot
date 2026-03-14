"""
Unit tests for KellyCriterionSizer component.
"""

from __future__ import annotations

import pytest

from src.strategies.components.position_sizer import KellyCriterionSizer
from src.strategies.components.regime_context import RegimeContext, TrendLabel, VolLabel
from src.strategies.components.signal_generator import Signal, SignalDirection

pytestmark = pytest.mark.unit


def _make_signal(
    direction: SignalDirection = SignalDirection.BUY,
    strength: float = 0.8,
    confidence: float = 0.9,
) -> Signal:
    """Create a test signal."""
    return Signal(direction=direction, strength=strength, confidence=confidence, metadata={})


def _make_regime(
    trend: TrendLabel = TrendLabel.TREND_UP,
    vol: VolLabel = VolLabel.LOW,
    confidence: float = 0.8,
    strength: float = 0.7,
) -> RegimeContext:
    """Create a test regime context."""
    return RegimeContext(
        trend=trend, volatility=vol, confidence=confidence, duration=10, strength=strength
    )


def _seed_trades(sizer: KellyCriterionSizer, wins: int, losses: int) -> None:
    """Record a batch of trades with a 2:1 reward-to-risk for wins."""
    for _ in range(wins):
        sizer.record_trade(win=True, profit_pct=0.04, loss_risk_pct=0.02)
    for _ in range(losses):
        sizer.record_trade(win=False, profit_pct=0.02, loss_risk_pct=0.02)


class TestKellyCriterionSizerInit:
    """Test initialization and parameter validation."""

    def test_default_initialization(self):
        """Test default parameter values."""
        sizer = KellyCriterionSizer()
        assert sizer.kelly_fraction == 0.5
        assert sizer.min_trades == 30
        assert sizer.lookback_trades == 100
        assert sizer.fallback_fraction == 0.02
        assert sizer.trade_count == 0
        assert not sizer.has_sufficient_history

    def test_custom_initialization(self):
        """Test custom parameter values."""
        sizer = KellyCriterionSizer(
            kelly_fraction=0.25,
            min_trades=20,
            lookback_trades=50,
            fallback_fraction=0.03,
        )
        assert sizer.kelly_fraction == 0.25
        assert sizer.min_trades == 20
        assert sizer.lookback_trades == 50
        assert sizer.fallback_fraction == 0.03

    def test_invalid_kelly_fraction_raises(self):
        """Test that invalid kelly_fraction raises ValueError."""
        with pytest.raises(ValueError, match="kelly_fraction"):
            KellyCriterionSizer(kelly_fraction=0.0)
        with pytest.raises(ValueError, match="kelly_fraction"):
            KellyCriterionSizer(kelly_fraction=1.5)

    def test_invalid_min_trades_raises(self):
        """Test that invalid min_trades raises ValueError."""
        with pytest.raises(ValueError, match="min_trades"):
            KellyCriterionSizer(min_trades=0)

    def test_lookback_less_than_min_trades_raises(self):
        """Test that lookback_trades < min_trades raises ValueError."""
        with pytest.raises(ValueError, match="lookback_trades"):
            KellyCriterionSizer(min_trades=50, lookback_trades=30)

    def test_invalid_fallback_fraction_raises(self):
        """Test that invalid fallback_fraction raises ValueError."""
        with pytest.raises(ValueError, match="fallback_fraction"):
            KellyCriterionSizer(fallback_fraction=0.0)

    def test_invalid_expected_win_rate_raises(self):
        """Test that invalid expected_win_rate raises ValueError."""
        with pytest.raises(ValueError, match="expected_win_rate"):
            KellyCriterionSizer(expected_win_rate=0.0)

    def test_invalid_expected_reward_risk_raises(self):
        """Test that invalid expected_reward_risk raises ValueError."""
        with pytest.raises(ValueError, match="expected_reward_risk"):
            KellyCriterionSizer(expected_reward_risk=-1.0)

    def test_invalid_overfitting_threshold_raises(self):
        """Test that invalid overfitting_threshold raises ValueError."""
        with pytest.raises(ValueError, match="overfitting_threshold"):
            KellyCriterionSizer(overfitting_threshold=1.5)

    def test_invalid_max_fraction_raises(self):
        """Test that invalid max_fraction raises ValueError."""
        with pytest.raises(ValueError, match="max_fraction"):
            KellyCriterionSizer(max_fraction=0.0)
        with pytest.raises(ValueError, match="max_fraction"):
            KellyCriterionSizer(max_fraction=1.5)

    def test_valid_max_fraction_accepted(self):
        """Test valid max_fraction values are accepted."""
        sizer = KellyCriterionSizer(max_fraction=0.50)
        assert sizer.max_fraction == 0.50


class TestKellyCalculation:
    """Test the core Kelly formula and statistics computation."""

    def test_kelly_percentage_positive_edge(self):
        """Test Kelly formula with a positive edge (profitable strategy)."""
        sizer = KellyCriterionSizer()
        # p=0.6, b=2.0 -> f* = (2*0.6 - 0.4)/2 = 0.4
        result = sizer._kelly_percentage(win_rate=0.6, reward_risk=2.0)
        assert result == pytest.approx(0.4, abs=1e-6)

    def test_kelly_percentage_no_edge(self):
        """Test Kelly formula with no edge returns zero."""
        sizer = KellyCriterionSizer()
        # p=0.5, b=1.0 -> f* = (1*0.5 - 0.5)/1 = 0.0
        result = sizer._kelly_percentage(win_rate=0.5, reward_risk=1.0)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_kelly_percentage_negative_edge(self):
        """Test Kelly formula with negative edge returns zero."""
        sizer = KellyCriterionSizer()
        # p=0.3, b=1.0 -> f* = (1*0.3 - 0.7)/1 = -0.4 -> clamped to 0.0
        result = sizer._kelly_percentage(win_rate=0.3, reward_risk=1.0)
        assert result == 0.0

    def test_kelly_percentage_capped_at_half(self):
        """Test Kelly formula caps at 50%."""
        sizer = KellyCriterionSizer()
        # p=0.9, b=10.0 -> f* = (10*0.9 - 0.1)/10 = 0.89 -> capped at 0.5
        result = sizer._kelly_percentage(win_rate=0.9, reward_risk=10.0)
        assert result == 0.5

    def test_kelly_percentage_zero_reward_risk(self):
        """Test Kelly returns 0 when reward_risk is zero."""
        sizer = KellyCriterionSizer()
        result = sizer._kelly_percentage(win_rate=0.6, reward_risk=0.0)
        assert result == 0.0

    def test_compute_statistics_from_buffer(self):
        """Test live statistics computation from trade buffer."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        _seed_trades(sizer, wins=7, losses=3)

        win_rate, avg_rr = sizer._compute_statistics()
        assert win_rate == pytest.approx(0.7, abs=1e-6)
        # b = mean(win_amounts) / mean(loss_amounts) = 0.04 / 0.02 = 2.0
        assert avg_rr == pytest.approx(2.0, abs=1e-6)

    def test_compute_statistics_includes_loss_magnitude(self):
        """Test b calculation uses both win and loss magnitudes."""
        sizer = KellyCriterionSizer(min_trades=3, lookback_trades=20)
        # 2 wins at 6%, 1 loss at 3%
        sizer.record_trade(win=True, profit_pct=0.06, loss_risk_pct=0.02)
        sizer.record_trade(win=True, profit_pct=0.06, loss_risk_pct=0.02)
        sizer.record_trade(win=False, profit_pct=0.03, loss_risk_pct=0.02)

        win_rate, avg_rr = sizer._compute_statistics()
        assert win_rate == pytest.approx(2.0 / 3.0, abs=1e-6)
        # b = mean([0.06, 0.06]) / mean([0.03]) = 0.06 / 0.03 = 2.0
        assert avg_rr == pytest.approx(2.0, abs=1e-6)

    def test_compute_statistics_no_losses_fallback(self):
        """Test b falls back to expected_reward_risk when no losses exist."""
        sizer = KellyCriterionSizer(
            min_trades=2, lookback_trades=20, expected_reward_risk=1.5
        )
        sizer.record_trade(win=True, profit_pct=0.04, loss_risk_pct=0.02)
        sizer.record_trade(win=True, profit_pct=0.04, loss_risk_pct=0.02)

        win_rate, avg_rr = sizer._compute_statistics()
        assert win_rate == 1.0
        assert avg_rr == 1.5  # Falls back to expected

    def test_compute_statistics_empty_buffer(self):
        """Test statistics default to expected values when buffer is empty."""
        sizer = KellyCriterionSizer(expected_win_rate=0.55, expected_reward_risk=1.5)
        win_rate, avg_rr = sizer._compute_statistics()
        assert win_rate == 0.55
        assert avg_rr == 1.5


class TestFractionalKelly:
    """Test fractional Kelly modes (quarter, half, full)."""

    def test_quarter_kelly(self):
        """Test quarter-Kelly produces ~25% of full Kelly size."""
        sizer_quarter = KellyCriterionSizer(
            kelly_fraction=0.25, min_trades=5, lookback_trades=20, max_fraction=0.50
        )
        sizer_full = KellyCriterionSizer(
            kelly_fraction=1.0, min_trades=5, lookback_trades=20, max_fraction=0.50
        )

        _seed_trades(sizer_quarter, wins=12, losses=8)
        _seed_trades(sizer_full, wins=12, losses=8)

        signal = _make_signal()
        balance = 10000.0
        risk = 10000.0  # High risk cap to avoid capping

        size_quarter = sizer_quarter.calculate_size(signal, balance, risk)
        size_full = sizer_full.calculate_size(signal, balance, risk)

        # Quarter Kelly should be roughly 25% of full Kelly
        assert size_quarter < size_full
        if size_full > 0:
            ratio = size_quarter / size_full
            assert ratio == pytest.approx(0.25, abs=0.05)

    def test_half_kelly(self):
        """Test half-Kelly produces ~50% of full Kelly size."""
        sizer_half = KellyCriterionSizer(
            kelly_fraction=0.5, min_trades=5, lookback_trades=20, max_fraction=0.50
        )
        sizer_full = KellyCriterionSizer(
            kelly_fraction=1.0, min_trades=5, lookback_trades=20, max_fraction=0.50
        )

        _seed_trades(sizer_half, wins=12, losses=8)
        _seed_trades(sizer_full, wins=12, losses=8)

        signal = _make_signal()
        balance = 10000.0
        risk = 10000.0  # High risk cap to avoid capping

        size_half = sizer_half.calculate_size(signal, balance, risk)
        size_full = sizer_full.calculate_size(signal, balance, risk)

        assert size_half < size_full
        if size_full > 0:
            ratio = size_half / size_full
            assert ratio == pytest.approx(0.5, abs=0.05)


class TestColdStartFallback:
    """Test behavior during the cold start period."""

    def test_cold_start_uses_fallback_fraction(self):
        """Test that cold start uses fallback fraction."""
        sizer = KellyCriterionSizer(
            fallback_fraction=0.03,
            min_trades=30,
            lookback_trades=100,
        )
        signal = _make_signal(confidence=1.0, strength=1.0)
        balance = 10000.0
        risk = 5000.0

        # No trades recorded → cold start
        size = sizer.calculate_size(signal, balance, risk)
        # Should be based on fallback_fraction * balance * confidence
        assert size == pytest.approx(300.0, rel=0.01)

    def test_cold_start_transitions_to_kelly(self):
        """Test transition from cold start to Kelly once min_trades reached."""
        sizer = KellyCriterionSizer(
            kelly_fraction=0.5,
            min_trades=5,
            lookback_trades=20,
            fallback_fraction=0.02,
        )
        signal = _make_signal(confidence=0.9, strength=0.8)
        balance = 10000.0
        risk = 2000.0

        # Cold start
        assert not sizer.has_sufficient_history
        cold_size = sizer.calculate_size(signal, balance, risk)
        assert cold_size > 0

        # Seed trades to activate Kelly
        _seed_trades(sizer, wins=4, losses=1)
        assert sizer.has_sufficient_history

        kelly_size = sizer.calculate_size(signal, balance, risk)
        # Kelly with 80% win rate and 2:1 reward should give different sizing
        assert kelly_size > 0
        assert kelly_size != cold_size

    def test_cold_start_respects_risk_limit(self):
        """Test cold start respects risk amount cap."""
        sizer = KellyCriterionSizer(fallback_fraction=0.10, min_trades=30, lookback_trades=100)
        signal = _make_signal(confidence=1.0)
        balance = 10000.0
        risk = 50.0  # Very small risk cap

        size = sizer.calculate_size(signal, balance, risk)
        assert size <= risk


class TestRegimeAdjustments:
    """Test regime-aware position scaling."""

    def test_bear_regime_reduces_size(self):
        """Test that bear regime reduces position size."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        _seed_trades(sizer, wins=8, losses=2)

        signal = _make_signal()
        balance = 10000.0
        risk = 5000.0

        size_no_regime = sizer.calculate_size(signal, balance, risk, regime=None)
        bear_regime = _make_regime(trend=TrendLabel.TREND_DOWN, vol=VolLabel.HIGH)
        size_bear = sizer.calculate_size(signal, balance, risk, regime=bear_regime)

        assert size_bear < size_no_regime

    def test_bull_regime_no_boost(self):
        """Test that bull regime does not boost beyond base size."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        _seed_trades(sizer, wins=8, losses=2)

        signal = _make_signal()
        balance = 10000.0
        risk = 5000.0

        size_no_regime = sizer.calculate_size(signal, balance, risk, regime=None)
        bull_regime = _make_regime(trend=TrendLabel.TREND_UP, vol=VolLabel.LOW)
        size_bull = sizer.calculate_size(signal, balance, risk, regime=bull_regime)

        # Bull with conservative config should not exceed no-regime size
        assert size_bull <= size_no_regime * 1.01  # Allow tiny floating point margin

    def test_high_vol_regime_reduces_size(self):
        """Test that high volatility regime reduces position size."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        _seed_trades(sizer, wins=8, losses=2)

        signal = _make_signal()
        balance = 10000.0
        risk = 5000.0

        low_vol = _make_regime(trend=TrendLabel.TREND_UP, vol=VolLabel.LOW)
        high_vol = _make_regime(trend=TrendLabel.TREND_UP, vol=VolLabel.HIGH)

        size_low_vol = sizer.calculate_size(signal, balance, risk, regime=low_vol)
        size_high_vol = sizer.calculate_size(signal, balance, risk, regime=high_vol)

        assert size_high_vol < size_low_vol


class TestOverfittingAdjustment:
    """Test overfitting detection and position reduction."""

    def test_no_adjustment_when_performance_matches(self):
        """Test no reduction when live matches expected performance."""
        sizer = KellyCriterionSizer(
            expected_win_rate=0.55,
            expected_reward_risk=1.5,
            overfitting_threshold=0.15,
        )
        # Live matches expectations → multiplier = 1.0
        adj = sizer._overfitting_adjustment(live_win_rate=0.55, live_rr=1.5)
        assert adj == 1.0

    def test_no_adjustment_within_threshold(self):
        """Test no reduction when deviation is within threshold."""
        sizer = KellyCriterionSizer(
            expected_win_rate=0.60,
            expected_reward_risk=2.0,
            overfitting_threshold=0.15,
        )
        # Slight underperformance within threshold
        adj = sizer._overfitting_adjustment(live_win_rate=0.55, live_rr=1.8)
        assert adj == 1.0

    def test_reduction_when_performance_deviates(self):
        """Test position reduction when live performance is much worse."""
        sizer = KellyCriterionSizer(
            expected_win_rate=0.65,
            expected_reward_risk=2.0,
            overfitting_threshold=0.10,
        )
        # Large deviation → should reduce
        adj = sizer._overfitting_adjustment(live_win_rate=0.40, live_rr=1.0)
        assert adj < 1.0
        assert adj >= 0.5  # Floor at 50%

    def test_reduction_floored_at_half(self):
        """Test overfitting adjustment never goes below 0.5."""
        sizer = KellyCriterionSizer(
            expected_win_rate=0.70,
            expected_reward_risk=3.0,
            overfitting_threshold=0.05,
        )
        # Extreme deviation
        adj = sizer._overfitting_adjustment(live_win_rate=0.20, live_rr=0.5)
        assert adj >= 0.5

    def test_no_penalty_when_outperforming(self):
        """Test no reduction when live performance exceeds expectations."""
        sizer = KellyCriterionSizer(
            expected_win_rate=0.50,
            expected_reward_risk=1.0,
            overfitting_threshold=0.10,
        )
        adj = sizer._overfitting_adjustment(live_win_rate=0.70, live_rr=2.0)
        assert adj == 1.0


class TestBoundsEnforcement:
    """Test position size bounds checking."""

    def test_respects_max_fraction(self):
        """Test position size is capped at max_fraction of balance."""
        sizer = KellyCriterionSizer(
            kelly_fraction=1.0,
            min_trades=5,
            lookback_trades=20,
        )
        # Seed very high win rate to get large Kelly
        _seed_trades(sizer, wins=9, losses=1)

        signal = _make_signal(confidence=1.0, strength=1.0)
        balance = 10000.0
        risk = 10000.0

        size = sizer.calculate_size(signal, balance, risk)
        # Max fraction is 0.20 → max size = 2000
        assert size <= balance * 0.20 + 1e-6

    def test_kelly_zero_position_not_inflated_by_min_fraction(self):
        """Test that Kelly zero/negative position returns 0.0, not min_fraction floor."""
        sizer = KellyCriterionSizer(min_trades=3, lookback_trades=10)
        # Seed trades with negative edge: 20% win rate, small wins, big losses
        for _ in range(2):
            sizer.record_trade(win=True, profit_pct=0.01, loss_risk_pct=0.02)
        for _ in range(8):
            sizer.record_trade(win=False, profit_pct=0.03, loss_risk_pct=0.02)

        signal = _make_signal(confidence=0.9, strength=0.8)
        size = sizer.calculate_size(signal, balance=10000.0, risk_amount=5000.0)
        # Kelly should recommend zero with negative edge
        assert size == 0.0

    def test_hold_signal_returns_zero(self):
        """Test hold signal returns zero position size."""
        sizer = KellyCriterionSizer()
        signal = _make_signal(direction=SignalDirection.HOLD)
        size = sizer.calculate_size(signal, balance=10000.0, risk_amount=200.0)
        assert size == 0.0

    def test_zero_risk_amount_returns_zero(self):
        """Test zero risk amount returns zero (risk manager veto)."""
        sizer = KellyCriterionSizer()
        signal = _make_signal()
        size = sizer.calculate_size(signal, balance=10000.0, risk_amount=0.0)
        assert size == 0.0

    def test_invalid_balance_raises(self):
        """Test negative balance raises ValueError."""
        sizer = KellyCriterionSizer()
        signal = _make_signal()
        with pytest.raises(ValueError, match="balance must be positive"):
            sizer.calculate_size(signal, balance=-1000.0, risk_amount=200.0)

    def test_risk_exceeding_balance_raises(self):
        """Test risk exceeding balance raises ValueError."""
        sizer = KellyCriterionSizer()
        signal = _make_signal()
        with pytest.raises(ValueError, match="cannot exceed balance"):
            sizer.calculate_size(signal, balance=1000.0, risk_amount=2000.0)


class TestRingBuffer:
    """Test trade recording ring buffer behavior."""

    def test_trade_count_increments(self):
        """Test trade count increments with each recorded trade."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=10)
        assert sizer.trade_count == 0

        sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.02)
        assert sizer.trade_count == 1

        sizer.record_trade(win=False, profit_pct=0.01, loss_risk_pct=0.02)
        assert sizer.trade_count == 2

    def test_ring_buffer_caps_at_lookback(self):
        """Test ring buffer does not exceed lookback_trades."""
        lookback = 10
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=lookback)

        for i in range(20):
            sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.02)

        assert sizer.trade_count == lookback

    def test_zero_loss_risk_ignored(self):
        """Test trades with zero loss_risk_pct are ignored."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=10)
        sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.0)
        assert sizer.trade_count == 0

    def test_nan_values_ignored(self):
        """Test trades with NaN profit or risk are silently dropped."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=10)
        sizer.record_trade(win=True, profit_pct=float("nan"), loss_risk_pct=0.02)
        sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=float("nan"))
        assert sizer.trade_count == 0

    def test_infinity_values_ignored(self):
        """Test trades with Infinity profit or risk are silently dropped."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=10)
        sizer.record_trade(win=True, profit_pct=float("inf"), loss_risk_pct=0.02)
        sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=float("inf"))
        assert sizer.trade_count == 0

    def test_has_sufficient_history_threshold(self):
        """Test has_sufficient_history flag at exact threshold."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=10)

        for i in range(4):
            sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.02)
        assert not sizer.has_sufficient_history

        sizer.record_trade(win=True, profit_pct=0.03, loss_risk_pct=0.02)
        assert sizer.has_sufficient_history


class TestGetParameters:
    """Test parameter serialization."""

    def test_get_parameters_includes_all_fields(self):
        """Test get_parameters returns all expected keys."""
        sizer = KellyCriterionSizer()
        params = sizer.get_parameters()

        assert params["type"] == "KellyCriterionSizer"
        assert params["name"] == "kelly_criterion_sizer"
        assert "kelly_fraction" in params
        assert "min_trades" in params
        assert "lookback_trades" in params
        assert "fallback_fraction" in params
        assert "trade_count" in params
        assert "has_sufficient_history" in params
        assert "live_win_rate" in params
        assert "live_avg_reward_risk" in params
        assert "expected_win_rate" in params
        assert "expected_reward_risk" in params

    def test_get_parameters_reflects_live_stats(self):
        """Test get_parameters updates with live statistics."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        _seed_trades(sizer, wins=7, losses=3)

        params = sizer.get_parameters()
        assert params["trade_count"] == 10
        assert params["has_sufficient_history"] is True
        assert params["live_win_rate"] == pytest.approx(0.7, abs=0.01)


class TestSellSignal:
    """Test that sell signals also produce valid sizing."""

    def test_sell_signal_produces_nonzero_size(self):
        """Test sell direction produces a position size."""
        sizer = KellyCriterionSizer(min_trades=5, lookback_trades=20)
        _seed_trades(sizer, wins=8, losses=2)

        signal = _make_signal(direction=SignalDirection.SELL)
        size = sizer.calculate_size(signal, balance=10000.0, risk_amount=2000.0)
        assert size > 0
