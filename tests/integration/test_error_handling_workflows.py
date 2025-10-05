"""
Integration tests for error handling and recovery workflows

Tests how the component system handles various error conditions
and recovers gracefully during trading operations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from src.strategies.components.strategy import Strategy, TradingDecision
from src.strategies.components.signal_generator import (
    SignalGenerator, Signal, SignalDirection, MLBasicSignalGenerator
)
from src.strategies.components.risk_manager import FixedRiskManager, Position, MarketData
from src.strategies.components.position_sizer import ConfidenceWeightedSizer
from src.strategies.components.regime_context import EnhancedRegimeDetector


pytestmark = pytest.mark.integration


class TestErrorHandlingWorkflows:
    """Test error handling and recovery in complete workflows"""
    
    def create_test_data(self):
        """Create basic test data"""
        return pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'onnx_pred': [101, 102, 103, 104, 105],
            'atr': [1.0, 1.0, 1.0, 1.0, 1.0]
        })
    
    def test_signal_generator_failure_recovery(self):
        """Test recovery when signal generator fails"""
        class FailingSignalGenerator(SignalGenerator):
            def __init__(self):
                super().__init__("failing_generator")
                self.call_count = 0
            
            def generate_signal(self, df, index, regime=None):
                self.call_count += 1
                if self.call_count <= 2:
                    raise Exception("Signal generation failed")
                # Recover after 2 failures
                return Signal(
                    direction=SignalDirection.BUY,
                    strength=0.8,
                    confidence=0.7,
                    metadata={'recovered': True}
                )
            
            def get_confidence(self, df, index):
                return 0.5
        
        strategy = Strategy(
            name="signal_failure_test",
            signal_generator=FailingSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        decisions = []
        
        # Process multiple candles
        for i in range(len(df)):
            decision = strategy.process_candle(df, i, balance)
            decisions.append(decision)
        
        # Should have decisions for all candles
        assert len(decisions) == len(df)
        
        # First two should be error recovery (HOLD)
        assert decisions[0].signal.direction == SignalDirection.HOLD
        assert decisions[1].signal.direction == SignalDirection.HOLD
        assert 'error' in decisions[0].metadata
        assert 'error' in decisions[1].metadata
        
        # Later decisions should recover
        assert decisions[2].signal.direction == SignalDirection.BUY
        assert decisions[2].signal.metadata.get('recovered') is True
    
    def test_risk_manager_failure_recovery(self):
        """Test recovery when risk manager fails"""
        class FailingRiskManager(FixedRiskManager):
            def __init__(self):
                super().__init__()
                self.call_count = 0
            
            def calculate_position_size(self, signal, balance, regime=None):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Risk calculation failed")
                return super().calculate_position_size(signal, balance, regime)
        
        strategy = Strategy(
            name="risk_failure_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FailingRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        
        # First call should handle risk manager failure
        decision1 = strategy.process_candle(df, 0, balance)
        assert decision1.position_size == 0.0  # Safe fallback
        
        # Second call should work normally
        decision2 = strategy.process_candle(df, 1, balance)
        # May have position size if signal is not HOLD
        assert decision2.position_size >= 0.0
    
    def test_position_sizer_failure_recovery(self):
        """Test recovery when position sizer fails"""
        class FailingPositionSizer(ConfidenceWeightedSizer):
            def __init__(self):
                super().__init__()
                self.call_count = 0
            
            def calculate_size(self, signal, balance, risk_amount, regime=None):
                self.call_count += 1
                if self.call_count == 1:
                    raise Exception("Position sizing failed")
                return super().calculate_size(signal, balance, risk_amount, regime)
        
        strategy = Strategy(
            name="sizer_failure_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=FailingPositionSizer()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        
        # First call should handle position sizer failure
        decision1 = strategy.process_candle(df, 0, balance)
        # Should fallback to risk manager's calculation
        assert decision1.position_size >= 0.0
        
        # Second call should work normally
        decision2 = strategy.process_candle(df, 1, balance)
        assert decision2.position_size >= 0.0
    
    def test_regime_detector_failure_recovery(self):
        """Test recovery when regime detector fails"""
        class FailingRegimeDetector(EnhancedRegimeDetector):
            def __init__(self):
                super().__init__()
                self.call_count = 0
            
            def detect_regime(self, df, index):
                self.call_count += 1
                if self.call_count <= 2:
                    raise Exception("Regime detection failed")
                return super().detect_regime(df, index)
        
        strategy = Strategy(
            name="regime_failure_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer(),
            regime_detector=FailingRegimeDetector()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        
        # First two calls should handle regime detector failure
        decision1 = strategy.process_candle(df, 0, balance)
        decision2 = strategy.process_candle(df, 1, balance)
        
        # Should still produce valid decisions without regime
        assert decision1.regime is None
        assert decision2.regime is None
        assert isinstance(decision1.signal.direction, SignalDirection)
        assert isinstance(decision2.signal.direction, SignalDirection)
        
        # Third call should work with regime
        decision3 = strategy.process_candle(df, 2, balance)
        # May have regime detected
        assert decision3 is not None
    
    def test_multiple_component_failures(self):
        """Test handling multiple simultaneous component failures"""
        class MultiFailingSignalGenerator(SignalGenerator):
            def generate_signal(self, df, index, regime=None):
                raise Exception("Signal failed")
            def get_confidence(self, df, index):
                raise Exception("Confidence failed")
        
        class MultiFailingRiskManager(FixedRiskManager):
            def calculate_position_size(self, signal, balance, regime=None):
                raise Exception("Risk failed")
        
        strategy = Strategy(
            name="multi_failure_test",
            signal_generator=MultiFailingSignalGenerator("multi_failing"),
            risk_manager=MultiFailingRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        
        # Should handle multiple failures gracefully
        decision = strategy.process_candle(df, 0, balance)
        
        # Should return safe decision
        assert decision.signal.direction == SignalDirection.HOLD
        assert decision.position_size == 0.0
        assert 'error' in decision.metadata
        assert decision.metadata.get('safe_mode') is True
    
    def test_invalid_data_handling(self):
        """Test handling of invalid or corrupted data"""
        strategy = Strategy(
            name="invalid_data_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        # Test with missing columns
        invalid_df = pd.DataFrame({
            'close': [100, 101, 102]
            # Missing required columns
        })
        
        balance = 10000.0
        
        # Should handle invalid data gracefully
        decision = strategy.process_candle(invalid_df, 1, balance)
        
        # Should return safe decision
        assert decision.signal.direction == SignalDirection.HOLD
        assert decision.position_size == 0.0
        assert 'error' in decision.metadata
    
    def test_extreme_market_conditions(self):
        """Test handling of extreme market conditions"""
        # Create extreme data (huge price movements, zero volume, etc.)
        extreme_df = pd.DataFrame({
            'open': [100, 1000000, 0.001, 100, 100],  # Extreme price jumps
            'high': [101, 1000001, 0.002, 101, 101],
            'low': [99, 999999, 0.0005, 99, 99],
            'close': [100.5, 1000000.5, 0.0015, 100.5, 100.5],
            'volume': [1000, 0, 1000000000, 1000, 1000],  # Zero and extreme volume
            'onnx_pred': [101, 1000001, 0.002, 101, 101],
            'atr': [1.0, 100000, 0.001, 1.0, 1.0]
        })
        
        strategy = Strategy(
            name="extreme_conditions_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        balance = 10000.0
        decisions = []
        
        # Process extreme data
        for i in range(len(extreme_df)):
            decision = strategy.process_candle(extreme_df, i, balance)
            decisions.append(decision)
        
        # Should handle all extreme conditions
        assert len(decisions) == len(extreme_df)
        
        # All decisions should be valid
        for decision in decisions:
            assert isinstance(decision.signal.direction, SignalDirection)
            assert 0 <= decision.signal.confidence <= 1
            assert 0 <= decision.signal.strength <= 1
            assert decision.position_size >= 0
            # Position sizes should be bounded even with extreme data
            assert decision.position_size <= balance
    
    def test_memory_and_state_recovery(self):
        """Test recovery from memory/state corruption"""
        strategy = Strategy(
            name="state_recovery_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        
        # Process some decisions to build state
        for i in range(3):
            strategy.process_candle(df, i, balance)
        
        # Simulate state corruption by clearing history
        original_history_length = len(strategy.decision_history)
        strategy.decision_history.clear()
        
        # Should continue working after state loss
        decision = strategy.process_candle(df, 3, balance)
        
        assert decision is not None
        assert isinstance(decision.signal.direction, SignalDirection)
        
        # Metrics should handle missing history gracefully
        metrics = strategy.get_performance_metrics()
        assert metrics['total_decisions'] == 1  # Only current decision
    
    def test_concurrent_access_safety(self):
        """Test thread safety and concurrent access handling"""
        import threading
        import time
        
        strategy = Strategy(
            name="concurrent_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = self.create_test_data()
        balance = 10000.0
        results = []
        errors = []
        
        def process_candle_thread(index):
            try:
                decision = strategy.process_candle(df, index % len(df), balance)
                results.append(decision)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_candle_thread, args=(i,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5.0)
        
        # Should handle concurrent access without major errors
        assert len(errors) == 0 or len(errors) < len(threads)  # Some errors acceptable
        assert len(results) > 0  # At least some should succeed
        
        # All results should be valid
        for result in results:
            assert isinstance(result.signal.direction, SignalDirection)


class TestRecoveryMechanisms:
    """Test specific recovery mechanisms and fallback strategies"""
    
    def test_component_fallback_chain(self):
        """Test fallback chain when components fail sequentially"""
        class ChainFailingSignalGenerator(SignalGenerator):
            def __init__(self):
                super().__init__("chain_failing")
                self.attempt = 0
            
            def generate_signal(self, df, index, regime=None):
                self.attempt += 1
                if self.attempt <= 3:
                    raise Exception(f"Attempt {self.attempt} failed")
                # Eventually succeed
                return Signal(
                    direction=SignalDirection.BUY,
                    strength=0.5,
                    confidence=0.6,
                    metadata={'attempts': self.attempt}
                )
            
            def get_confidence(self, df, index):
                return 0.5
        
        strategy = Strategy(
            name="fallback_chain_test",
            signal_generator=ChainFailingSignalGenerator(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [101, 102, 103, 104, 105],
            'low': [99, 100, 101, 102, 103],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        balance = 10000.0
        decisions = []
        
        # Process through failure and recovery
        for i in range(len(df)):
            decision = strategy.process_candle(df, i, balance)
            decisions.append(decision)
        
        # First 3 should be error recovery
        for i in range(3):
            assert decisions[i].signal.direction == SignalDirection.HOLD
            assert 'error' in decisions[i].metadata
        
        # 4th should succeed
        assert decisions[3].signal.direction == SignalDirection.BUY
        assert decisions[3].signal.metadata.get('attempts') == 4
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components partially fail"""
        class PartiallyFailingRiskManager(FixedRiskManager):
            def calculate_position_size(self, signal, balance, regime=None):
                # Fail for BUY signals, work for others
                if signal.direction == SignalDirection.BUY:
                    raise Exception("Cannot calculate size for BUY")
                return super().calculate_position_size(signal, balance, regime)
        
        strategy = Strategy(
            name="degradation_test",
            signal_generator=MLBasicSignalGenerator(),
            risk_manager=PartiallyFailingRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        # Create data that would generate BUY signals
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200],
            'onnx_pred': [105, 106, 107]  # High predictions to trigger BUY
        })
        
        balance = 10000.0
        decisions = []
        
        for i in range(len(df)):
            decision = strategy.process_candle(df, i, balance)
            decisions.append(decision)
        
        # Should handle partial failures gracefully
        for decision in decisions:
            assert isinstance(decision.signal.direction, SignalDirection)
            # Position size should be 0 when risk manager fails
            if 'error' in decision.metadata:
                assert decision.position_size == 0.0
    
    def test_circuit_breaker_mechanism(self):
        """Test circuit breaker for repeated failures"""
        class RepeatedlyFailingComponent(MLBasicSignalGenerator):
            def __init__(self):
                super().__init__()
                self.failure_count = 0
            
            def generate_signal(self, df, index, regime=None):
                self.failure_count += 1
                raise Exception(f"Failure #{self.failure_count}")
        
        strategy = Strategy(
            name="circuit_breaker_test",
            signal_generator=RepeatedlyFailingComponent(),
            risk_manager=FixedRiskManager(),
            position_sizer=ConfidenceWeightedSizer()
        )
        
        df = pd.DataFrame({
            'open': [100] * 10,
            'high': [101] * 10,
            'low': [99] * 10,
            'close': [100.5] * 10,
            'volume': [1000] * 10
        })
        
        balance = 10000.0
        decisions = []
        
        # Process multiple failures
        for i in range(10):
            decision = strategy.process_candle(df, i, balance)
            decisions.append(decision)
        
        # All should be safe HOLD decisions
        for decision in decisions:
            assert decision.signal.direction == SignalDirection.HOLD
            assert decision.position_size == 0.0
            assert 'error' in decision.metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])