"""
Signal Generator Components

This module defines the abstract SignalGenerator interface and related data models
for generating trading signals in the component-based strategy architecture.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import pandas as pd


class SignalDirection(Enum):
    """Enumeration for signal directions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Signal:
    """
    Data class representing a trading signal
    
    Attributes:
        direction: The signal direction (BUY, SELL, HOLD)
        strength: Signal strength from 0.0 to 1.0
        confidence: Confidence in the signal from 0.0 to 1.0
        metadata: Additional signal information and context
    """
    direction: SignalDirection
    strength: float
    confidence: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate signal parameters after initialization"""
        self._validate_signal()
    
    def _validate_signal(self):
        """Validate signal parameters are within acceptable bounds"""
        if not isinstance(self.direction, SignalDirection):
            raise ValueError(f"direction must be a SignalDirection enum, got {type(self.direction)}")
        
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be between 0.0 and 1.0, got {self.strength}")
        
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if not isinstance(self.metadata, dict):
            raise ValueError(f"metadata must be a dictionary, got {type(self.metadata)}")


class SignalGenerator(ABC):
    """
    Abstract base class for signal generators
    
    Signal generators are responsible for analyzing market data and generating
    trading signals with associated confidence scores. They can be regime-aware
    and adapt their behavior based on market conditions.
    """
    
    def __init__(self, name: str):
        """
        Initialize the signal generator
        
        Args:
            name: Unique name for this signal generator
        """
        self.name = name
    
    @abstractmethod
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional['RegimeContext'] = None) -> Signal:
        """
        Generate a trading signal based on market data
        
        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame
            regime: Optional regime context for regime-aware signal generation
            
        Returns:
            Signal object containing direction, strength, confidence, and metadata
            
        Raises:
            ValueError: If input parameters are invalid
            IndexError: If index is out of bounds
        """
        pass
    
    @abstractmethod
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """
        Get confidence score for signal generation at the given index
        
        Args:
            df: DataFrame containing OHLCV data with calculated indicators
            index: Current index position in the DataFrame
            
        Returns:
            Confidence score between 0.0 and 1.0
            
        Raises:
            ValueError: If input parameters are invalid
            IndexError: If index is out of bounds
        """
        pass
    
    def validate_inputs(self, df: pd.DataFrame, index: int) -> None:
        """
        Validate input parameters for signal generation
        
        Args:
            df: DataFrame to validate
            index: Index to validate
            
        Raises:
            ValueError: If DataFrame is empty or missing required columns
            IndexError: If index is out of bounds
        """
        if df.empty:
            raise ValueError("DataFrame cannot be empty")
        
        if index < 0 or index >= len(df):
            raise IndexError(f"Index {index} is out of bounds for DataFrame of length {len(df)}")
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"DataFrame missing required columns: {missing_columns}")
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get signal generator parameters for logging and serialization
        
        Returns:
            Dictionary of parameter names and values
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__
        }


class HoldSignalGenerator(SignalGenerator):
    """
    Simple signal generator that always returns HOLD signals
    
    Useful for testing and as a conservative fallback strategy
    """
    
    def __init__(self):
        super().__init__("hold_signal_generator")
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional['RegimeContext'] = None) -> Signal:
        """Generate a HOLD signal with neutral strength and high confidence"""
        self.validate_inputs(df, index)
        
        return Signal(
            direction=SignalDirection.HOLD,
            strength=0.0,
            confidence=1.0,
            metadata={
                'generator': self.name,
                'index': index,
                'timestamp': df.index[index] if hasattr(df.index, '__getitem__') else None,
                'regime': regime.trend.value if regime else None
            }
        )
    
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Always return high confidence for HOLD signals"""
        self.validate_inputs(df, index)
        return 1.0


class RandomSignalGenerator(SignalGenerator):
    """
    Random signal generator for testing purposes
    
    Generates random signals with configurable probabilities
    """
    
    def __init__(self, buy_prob: float = 0.3, sell_prob: float = 0.3, seed: Optional[int] = None):
        """
        Initialize random signal generator
        
        Args:
            buy_prob: Probability of generating BUY signal (0.0 to 1.0)
            sell_prob: Probability of generating SELL signal (0.0 to 1.0)
            seed: Random seed for reproducible results
        """
        super().__init__("random_signal_generator")
        
        if not 0.0 <= buy_prob <= 1.0:
            raise ValueError(f"buy_prob must be between 0.0 and 1.0, got {buy_prob}")
        if not 0.0 <= sell_prob <= 1.0:
            raise ValueError(f"sell_prob must be between 0.0 and 1.0, got {sell_prob}")
        if buy_prob + sell_prob > 1.0:
            raise ValueError(f"buy_prob + sell_prob cannot exceed 1.0, got {buy_prob + sell_prob}")
        
        self.buy_prob = buy_prob
        self.sell_prob = sell_prob
        self.hold_prob = 1.0 - buy_prob - sell_prob
        
        if seed is not None:
            import numpy as np
            np.random.seed(seed)
    
    def generate_signal(self, df: pd.DataFrame, index: int, regime: Optional['RegimeContext'] = None) -> Signal:
        """Generate a random signal based on configured probabilities"""
        import numpy as np
        
        self.validate_inputs(df, index)
        
        # Generate random signal direction
        rand = np.random.random()
        if rand < self.buy_prob:
            direction = SignalDirection.BUY
        elif rand < self.buy_prob + self.sell_prob:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD
        
        # Generate random strength and confidence
        strength = np.random.random() if direction != SignalDirection.HOLD else 0.0
        confidence = np.random.uniform(0.3, 0.9)  # Avoid very low confidence
        
        return Signal(
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                'generator': self.name,
                'index': index,
                'timestamp': df.index[index] if hasattr(df.index, '__getitem__') else None,
                'regime': regime.trend.value if regime else None,
                'random_seed': rand
            }
        )
    
    def get_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Return random confidence score"""
        import numpy as np
        
        self.validate_inputs(df, index)
        return np.random.uniform(0.3, 0.9)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get random signal generator parameters"""
        params = super().get_parameters()
        params.update({
            'buy_prob': self.buy_prob,
            'sell_prob': self.sell_prob,
            'hold_prob': self.hold_prob
        })
        return params