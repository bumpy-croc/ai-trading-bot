"""
Standardized Test Datasets

This module provides comprehensive test datasets for strategy component testing,
including historical market data, synthetic scenarios, and edge case datasets.
"""

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MarketScenario:
    """Definition of a market scenario for testing"""
    name: str
    description: str
    duration_days: int

    # Market characteristics
    trend_direction: str  # 'up', 'down', 'sideways'
    volatility_level: str  # 'low', 'medium', 'high'
    trend_strength: float  # 0.0 to 1.0
    volatility_value: float  # Annualized volatility

    # Price movement parameters
    initial_price: float
    final_price: Optional[float] = None
    max_drawdown: Optional[float] = None
    max_runup: Optional[float] = None

    # Market regime labels
    regime_labels: Optional[Dict[str, Any]] = None

    # Special characteristics
    has_gaps: bool = False
    has_flash_crash: bool = False
    has_bubble: bool = False
    has_consolidation: bool = False


class TestDatasetGenerator:
    """
    Generator for standardized test datasets
    
    Provides comprehensive historical market data, synthetic scenarios,
    and edge case datasets for thorough component testing.
    """

    def __init__(self, data_dir: Optional[str] = None, cache_dir: Optional[str] = None, 
                 max_cache_size_mb: int = 500):
        """
        Initialize test dataset generator
        
        Args:
            data_dir: Directory containing historical market data
            cache_dir: Directory for caching generated datasets
            max_cache_size_mb: Maximum cache size in MB (default 500MB)
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data")
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache/test_datasets")
        self.max_cache_size_mb = max_cache_size_mb
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Evict old cache files if size exceeds limit
        self._evict_cache_if_needed()

        # Initialize scenario definitions
        self.market_scenarios = self._define_market_scenarios()

        # Initialize synthetic data generator
        self.synthetic_generator = SyntheticDataGenerator()
    
    def _evict_cache_if_needed(self) -> None:
        """Evict oldest cache files if total size exceeds limit"""
        try:
            cache_files = list(self.cache_dir.glob("*.feather"))
            if not cache_files:
                return
            
            # Calculate total cache size
            total_size_bytes = sum(f.stat().st_size for f in cache_files)
            total_size_mb = total_size_bytes / (1024 * 1024)
            
            if total_size_mb > self.max_cache_size_mb:
                # Sort by modification time (oldest first)
                cache_files.sort(key=lambda f: f.stat().st_mtime)
                
                # Remove oldest files until under limit
                for cache_file in cache_files:
                    file_size_mb = cache_file.stat().st_size / (1024 * 1024)
                    cache_file.unlink()
                    total_size_mb -= file_size_mb
                    if total_size_mb <= self.max_cache_size_mb * 0.8:  # 80% threshold
                        break
                
                logger.info(f"Cache evicted to {total_size_mb:.1f}MB")
        except Exception as e:
            logger.warning(f"Cache eviction failed: {e}")

    def _define_market_scenarios(self) -> List[MarketScenario]:
        """Define standard market scenarios for testing"""
        scenarios = [
            # Bull market scenarios
            MarketScenario(
                name="strong_bull_low_vol",
                description="Strong bull market with low volatility",
                duration_days=252,
                trend_direction="up",
                volatility_level="low",
                trend_strength=0.8,
                volatility_value=0.15,
                initial_price=100.0,
                final_price=150.0,
                max_drawdown=0.05
            ),

            MarketScenario(
                name="moderate_bull_high_vol",
                description="Moderate bull market with high volatility",
                duration_days=252,
                trend_direction="up",
                volatility_level="high",
                trend_strength=0.6,
                volatility_value=0.35,
                initial_price=100.0,
                final_price=130.0,
                max_drawdown=0.15
            ),

            # Bear market scenarios
            MarketScenario(
                name="strong_bear_low_vol",
                description="Strong bear market with low volatility",
                duration_days=252,
                trend_direction="down",
                volatility_level="low",
                trend_strength=0.8,
                volatility_value=0.18,
                initial_price=100.0,
                final_price=70.0,
                max_drawdown=0.30
            ),

            MarketScenario(
                name="volatile_bear_crash",
                description="Volatile bear market with crash",
                duration_days=126,
                trend_direction="down",
                volatility_level="high",
                trend_strength=0.9,
                volatility_value=0.50,
                initial_price=100.0,
                final_price=60.0,
                max_drawdown=0.45,
                has_flash_crash=True
            ),

            # Sideways market scenarios
            MarketScenario(
                name="tight_range_low_vol",
                description="Tight sideways range with low volatility",
                duration_days=252,
                trend_direction="sideways",
                volatility_level="low",
                trend_strength=0.1,
                volatility_value=0.12,
                initial_price=100.0,
                final_price=102.0,
                max_drawdown=0.03,
                has_consolidation=True
            ),

            MarketScenario(
                name="wide_range_high_vol",
                description="Wide sideways range with high volatility",
                duration_days=252,
                trend_direction="sideways",
                volatility_level="high",
                trend_strength=0.2,
                volatility_value=0.30,
                initial_price=100.0,
                final_price=98.0,
                max_drawdown=0.12
            ),

            # Special scenarios
            MarketScenario(
                name="bubble_and_crash",
                description="Bubble formation followed by crash",
                duration_days=504,
                trend_direction="up",
                volatility_level="high",
                trend_strength=0.9,
                volatility_value=0.40,
                initial_price=100.0,
                final_price=80.0,
                max_runup=2.5,
                max_drawdown=0.60,
                has_bubble=True,
                has_flash_crash=True
            ),

            MarketScenario(
                name="gap_heavy_market",
                description="Market with frequent gaps",
                duration_days=252,
                trend_direction="up",
                volatility_level="medium",
                trend_strength=0.5,
                volatility_value=0.25,
                initial_price=100.0,
                final_price=120.0,
                has_gaps=True
            ),

            # Regime transition scenarios
            MarketScenario(
                name="bull_to_bear_transition",
                description="Transition from bull to bear market",
                duration_days=504,
                trend_direction="down",
                volatility_level="medium",
                trend_strength=0.6,
                volatility_value=0.28,
                initial_price=100.0,
                final_price=75.0,
                regime_labels={
                    'transition_point': 252,
                    'first_regime': 'bull_low_vol',
                    'second_regime': 'bear_high_vol'
                }
            ),

            MarketScenario(
                name="multiple_regime_changes",
                description="Multiple regime changes throughout period",
                duration_days=756,
                trend_direction="sideways",
                volatility_level="medium",
                trend_strength=0.4,
                volatility_value=0.22,
                initial_price=100.0,
                final_price=105.0,
                regime_labels={
                    'regime_changes': [126, 252, 378, 504, 630],
                    'regimes': ['bull_low_vol', 'bear_high_vol', 'sideways_low_vol', 'bull_high_vol', 'bear_low_vol', 'sideways_medium_vol']
                }
            )
        ]

        return scenarios

    def get_historical_dataset(self, symbol: str = "BTCUSDT",
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             timeframe: str = "1d") -> pd.DataFrame:
        """
        Get historical market data for testing
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            timeframe: Data timeframe ('1d', '1h', etc.)
            
        Returns:
            DataFrame with OHLCV data and technical indicators
        """
        # Try to load from data directory
        data_file = self.data_dir / f"{symbol}_{timeframe}_{start_date}_{end_date}.feather"

        if data_file.exists():
            try:
                data = pd.read_feather(data_file)
                data.set_index('timestamp', inplace=True)
                return self._prepare_historical_data(data)
            except Exception as e:
                warnings.warn(f"Error loading historical data: {e}")

        # If historical data not available, generate synthetic data
        warnings.warn(f"Historical data not found for {symbol}, generating synthetic data")
        return self.generate_synthetic_dataset("moderate_bull_low_vol")

    def _prepare_historical_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare historical data with technical indicators"""
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Historical data missing required column: {col}")

        # Add technical indicators
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()

        # RSI
        data['rsi'] = self._calculate_rsi(data['close'])

        # ATR
        data['atr'] = self._calculate_atr(data)

        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        data['bb_std'] = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * bb_std)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * bb_std)

        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Volatility
        data['volatility'] = data['returns'].rolling(20).std()

        return data.dropna()

    def _prepare_edge_case_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare edge case data with technical indicators, preserving intentional NaNs
        
        This method is used for edge case datasets where NaNs and missing data
        are intentional test conditions and should not be dropped.
        """
        # Ensure required columns exist
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Historical data missing required column: {col}")

        # Add technical indicators (NaNs will naturally occur but won't be dropped)
        data['sma_20'] = data['close'].rolling(20).mean()
        data['sma_50'] = data['close'].rolling(50).mean()
        data['ema_12'] = data['close'].ewm(span=12).mean()
        data['ema_26'] = data['close'].ewm(span=26).mean()

        # RSI
        data['rsi'] = self._calculate_rsi(data['close'])

        # ATR
        data['atr'] = self._calculate_atr(data)

        # MACD
        data['macd'] = data['ema_12'] - data['ema_26']
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        data['macd_histogram'] = data['macd'] - data['macd_signal']

        # Bollinger Bands
        bb_period = 20
        bb_std = 2
        data['bb_middle'] = data['close'].rolling(bb_period).mean()
        data['bb_std'] = data['close'].rolling(bb_period).std()
        data['bb_upper'] = data['bb_middle'] + (data['bb_std'] * bb_std)
        data['bb_lower'] = data['bb_middle'] - (data['bb_std'] * bb_std)

        # Returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # Volatility
        data['volatility'] = data['returns'].rolling(20).std()

        # Drop only rows where ALL required OHLCV columns are NaN to preserve edge cases
        data = data.dropna(subset=required_columns, how='all')

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Avoid division by zero: when loss is 0, RSI is 100
        rs = np.where(loss == 0, np.inf, gain / loss)
        rsi = np.where(np.isinf(rs), 100, 100 - (100 / (1 + rs)))
        return pd.Series(rsi, index=prices.index)

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()

    def generate_synthetic_dataset(self, scenario_name: str,
                                 seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic dataset for a specific market scenario
        
        Args:
            scenario_name: Name of the market scenario
            seed: Random seed for reproducible results
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        scenario = self._get_scenario_by_name(scenario_name)
        if not scenario:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        # Check cache first
        cache_file = self.cache_dir / f"synthetic_{scenario_name}_{seed}.feather"
        if cache_file.exists():
            try:
                data = pd.read_feather(cache_file)
                data.set_index('timestamp', inplace=True)
                return data
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}. Regenerating...")

        # Generate synthetic data
        data = self.synthetic_generator.generate_scenario_data(scenario, seed)

        # Add technical indicators
        data = self._prepare_historical_data(data)

        # Cache the result
        try:
            data_to_cache = data.reset_index()
            data_to_cache.to_feather(cache_file)
        except Exception as e:
            warnings.warn(f"Could not cache synthetic data: {e}")

        return data

    def _get_scenario_by_name(self, name: str) -> Optional[MarketScenario]:
        """Get scenario definition by name"""
        for scenario in self.market_scenarios:
            if scenario.name == name:
                return scenario
        return None

    def get_all_scenarios(self) -> List[str]:
        """Get list of all available scenario names"""
        return [scenario.name for scenario in self.market_scenarios]

    def get_scenario_description(self, scenario_name: str) -> str:
        """Get description of a specific scenario"""
        scenario = self._get_scenario_by_name(scenario_name)
        return scenario.description if scenario else "Unknown scenario"

    def generate_edge_case_dataset(self, case_type: str,
                                 duration_days: int = 100,
                                 seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate edge case datasets for stress testing
        
        Args:
            case_type: Type of edge case ('missing_data', 'extreme_volatility', 'zero_volume', etc.)
            duration_days: Duration of the dataset
            seed: Random seed for reproducible results
            
        Returns:
            DataFrame with edge case data
        """
        # Use isolated random number generator to avoid polluting global state
        rng = np.random.default_rng(seed)

        # Generate base dataset
        base_scenario = MarketScenario(
            name=f"edge_case_{case_type}",
            description=f"Edge case: {case_type}",
            duration_days=duration_days,
            trend_direction="sideways",
            volatility_level="medium",
            trend_strength=0.3,
            volatility_value=0.20,
            initial_price=100.0
        )

        data = self.synthetic_generator.generate_scenario_data(base_scenario, seed)

        # Apply edge case modifications with isolated rng
        if case_type == "missing_data":
            data = self._create_missing_data_case(data, rng)
        elif case_type == "extreme_volatility":
            data = self._create_extreme_volatility_case(data, rng)
        elif case_type == "zero_volume":
            data = self._create_zero_volume_case(data, rng)
        elif case_type == "price_gaps":
            data = self._create_price_gaps_case(data, rng)
        elif case_type == "flat_prices":
            data = self._create_flat_prices_case(data, rng)
        elif case_type == "extreme_outliers":
            data = self._create_extreme_outliers_case(data, rng)
        else:
            raise ValueError(f"Unknown edge case type: {case_type}")

        return self._prepare_edge_case_data(data)

    def _ensure_ohlc_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure OHLC consistency: high >= open, close, low and low <= open, close, high
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            DataFrame with consistent OHLC values
        """
        data['high'] = data[['open', 'high', 'low', 'close']].max(axis=1)
        data['low'] = data[['open', 'high', 'low', 'close']].min(axis=1)
        return data

    def _create_missing_data_case(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Create dataset with missing data points"""
        # Randomly remove 10% of data points
        missing_indices = rng.choice(data.index, size=int(len(data) * 0.1), replace=False)
        data.loc[missing_indices, ['open', 'high', 'low', 'close']] = np.nan

        # Create some consecutive missing periods
        if len(data) >= 10:
            start_idx = rng.integers(0, len(data) - 10)
            data.iloc[start_idx:start_idx+5, :4] = np.nan

        return data

    def _create_extreme_volatility_case(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Create dataset with extreme volatility spikes"""
        # Add random extreme moves
        extreme_moves = rng.normal(0, 0.1, len(data))
        extreme_mask = rng.random(len(data)) < 0.05  # 5% of days have extreme moves

        data['close'] *= (1 + extreme_moves * extreme_mask)

        # Ensure OHLC consistency
        data = self._ensure_ohlc_consistency(data)

        return data

    def _create_zero_volume_case(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Create dataset with zero volume periods"""
        # Set random periods to zero volume
        zero_volume_mask = rng.random(len(data)) < 0.15  # 15% of days
        data.loc[zero_volume_mask, 'volume'] = 0

        # Create consecutive zero volume periods
        if len(data) >= 20:
            start_idx = rng.integers(0, len(data) - 20)
            data.iloc[start_idx:start_idx+10, data.columns.get_loc('volume')] = 0

        return data

    def _create_price_gaps_case(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Create dataset with price gaps"""
        # Add random gaps
        gap_indices = rng.choice(data.index[1:], size=int(len(data) * 0.05), replace=False)

        for idx in gap_indices:
            gap_size = rng.uniform(0.02, 0.08)  # 2-8% gaps
            gap_direction = rng.choice([-1, 1])

            prev_close = data.loc[data.index[data.index.get_loc(idx)-1], 'close']
            gap_open = prev_close * (1 + gap_size * gap_direction)

            data.loc[idx, 'open'] = gap_open
            data.loc[idx, 'high'] = max(gap_open, data.loc[idx, 'close'])
            data.loc[idx, 'low'] = min(gap_open, data.loc[idx, 'close'])

        return data

    def _create_flat_prices_case(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Create dataset with flat price periods"""
        # Create periods where all OHLC are the same
        flat_periods = 3
        for _ in range(flat_periods):
            if len(data) >= 15:
                start_idx = rng.integers(0, len(data) - 15)
                flat_price = data.iloc[start_idx]['close']

                data.iloc[start_idx:start_idx+10, :4] = flat_price

        return data


    def _create_extreme_outliers_case(self, data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        """Create dataset with extreme price outliers"""
        # Add extreme outliers to high/low prices
        outlier_indices = rng.choice(data.index, size=int(len(data) * 0.02), replace=False)

        for idx in outlier_indices:
            if rng.random() > 0.5:
                # Extreme high
                data.loc[idx, 'high'] *= rng.uniform(2.0, 5.0)
            else:
                # Extreme low
                data.loc[idx, 'low'] *= rng.uniform(0.1, 0.5)

        # Ensure OHLC consistency
        data = self._ensure_ohlc_consistency(data)

        return data

    def create_regime_labeled_dataset(self, scenario_name: str,
                                    regime_detection_method: str = "simple",
                                    seed: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create dataset with regime labels for regime-specific testing
        
        Args:
            scenario_name: Name of the market scenario
            regime_detection_method: Method for regime detection
            seed: Random seed for reproducible results
            
        Returns:
            Tuple of (market_data, regime_labels)
        """
        # Generate market data
        market_data = self.generate_synthetic_dataset(scenario_name, seed)

        # Generate regime labels
        regime_labels = self._generate_regime_labels(market_data, regime_detection_method)

        return market_data, regime_labels

    def _generate_regime_labels(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Generate regime labels for market data"""
        regime_data = pd.DataFrame(index=data.index)

        if method == "simple":
            # Simple regime detection based on moving averages and volatility
            data['sma_20'] = data['close'].rolling(20).mean()
            data['sma_50'] = data['close'].rolling(50).mean()
            data['volatility'] = data['returns'].rolling(20).std()

            # Trend classification
            trend_condition = data['sma_20'] > data['sma_50']
            price_trend = data['close'] > data['close'].shift(20)

            regime_data['trend'] = 'range'
            regime_data.loc[trend_condition & price_trend, 'trend'] = 'trend_up'
            regime_data.loc[~trend_condition & ~price_trend, 'trend'] = 'trend_down'

            # Volatility classification
            vol_median = data['volatility'].median()
            regime_data['volatility'] = 'medium_vol'
            regime_data.loc[data['volatility'] > vol_median * 1.5, 'volatility'] = 'high_vol'
            regime_data.loc[data['volatility'] < vol_median * 0.7, 'volatility'] = 'low_vol'

            # Combined regime
            regime_data['regime_type'] = regime_data['trend'] + '_' + regime_data['volatility']

            # Calculate confidence based on trend strength (how far from MA crossover)
            ma_diff = abs(data['sma_20'] - data['sma_50']) / data['close']
            regime_data['confidence'] = ma_diff.fillna(0.7).clip(0.5, 0.95)  # Normalize to 0.5-0.95 range
            
            regime_data['duration'] = self._calculate_regime_duration(regime_data)
            
            # Calculate strength based on price momentum and volatility consistency
            price_momentum = abs(data['close'].pct_change(20)).fillna(0.05)
            vol_consistency = 1.0 / (1.0 + data['volatility'].rolling(10).std().fillna(0.1))
            regime_data['strength'] = (price_momentum * 0.6 + vol_consistency * 0.4).clip(0.3, 0.9)

        return regime_data.dropna()

    def _calculate_regime_duration(self, regime_data: pd.DataFrame) -> pd.Series:
        """Calculate regime duration for each period using vectorized operations"""
        # Identify regime changes
        regime_changes = (regime_data['regime_type'] != regime_data['regime_type'].shift()).fillna(True)
        
        # Create regime group IDs
        regime_groups = regime_changes.cumsum()
        
        # Calculate duration within each group
        duration = regime_groups.groupby(regime_groups).cumcount() + 1
        
        return pd.Series(duration.values, index=regime_data.index, dtype=int)

    def get_comprehensive_test_suite(self, seed: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Get comprehensive test suite with all scenarios and edge cases
        
        Args:
            seed: Random seed for reproducible results
            
        Returns:
            Dictionary mapping test names to datasets
        """
        test_suite = {}

        # Add all market scenarios
        for scenario in self.market_scenarios:
            test_suite[f"scenario_{scenario.name}"] = self.generate_synthetic_dataset(scenario.name, seed)

        # Add edge cases
        edge_cases = [
            "missing_data", "extreme_volatility", "zero_volume",
            "price_gaps", "flat_prices", "extreme_outliers"
        ]

        for edge_case in edge_cases:
            test_suite[f"edge_case_{edge_case}"] = self.generate_edge_case_dataset(edge_case, seed=seed)

        # Add historical data if available
        try:
            historical = self.get_historical_dataset()
            test_suite["historical_btcusdt"] = historical
        except Exception as e:
            logger.debug(f"Historical data not available: {e}. Skipping...")

        return test_suite


class SyntheticDataGenerator:
    """
    Generator for synthetic market data
    
    Creates realistic OHLCV data based on market scenario specifications
    using various stochastic processes and market microstructure models.
    """

    def __init__(self):
        """Initialize synthetic data generator"""
        pass

    def generate_scenario_data(self, scenario: MarketScenario, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic data for a market scenario
        
        Args:
            scenario: Market scenario specification
            seed: Random seed for reproducible results
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # Generate timestamps
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(days=scenario.duration_days),
            periods=scenario.duration_days,
            freq='D'
        )

        # Generate price series based on scenario
        prices = self._generate_price_series(scenario)

        # Generate OHLCV data from price series
        ohlcv_data = self._generate_ohlcv_from_prices(prices, scenario)

        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'open': ohlcv_data['open'],
            'high': ohlcv_data['high'],
            'low': ohlcv_data['low'],
            'close': ohlcv_data['close'],
            'volume': ohlcv_data['volume']
        })

        data.set_index('timestamp', inplace=True)

        # Apply special characteristics
        if scenario.has_gaps:
            data = self._add_price_gaps(data)

        if scenario.has_flash_crash:
            data = self._add_flash_crash(data)

        if scenario.has_bubble:
            data = self._add_bubble_pattern(data)

        if scenario.has_consolidation:
            data = self._add_consolidation_periods(data)

        return data

    def _generate_price_series(self, scenario: MarketScenario) -> np.ndarray:
        """Generate price series using geometric Brownian motion with drift"""
        n_periods = scenario.duration_days

        # Calculate drift and volatility parameters
        if scenario.final_price:
            total_return = (scenario.final_price / scenario.initial_price) - 1
            drift = total_return / n_periods
        else:
            # Estimate drift from trend direction and strength
            if scenario.trend_direction == "up":
                drift = 0.0005 * scenario.trend_strength  # ~0.05% daily for strong uptrend
            elif scenario.trend_direction == "down":
                drift = -0.0005 * scenario.trend_strength
            else:  # sideways
                drift = 0.0

        # Daily volatility
        daily_vol = scenario.volatility_value / np.sqrt(252)

        # Generate random returns
        random_returns = np.random.normal(drift, daily_vol, n_periods)

        # Add trend strength effect
        trend_component = np.linspace(0, drift * scenario.trend_strength, n_periods)
        random_returns += trend_component

        # Generate price series
        prices = np.zeros(n_periods + 1)
        prices[0] = scenario.initial_price

        for i in range(n_periods):
            prices[i + 1] = prices[i] * (1 + random_returns[i])

        return prices[1:]  # Return without initial price

    def _generate_ohlcv_from_prices(self, prices: np.ndarray, scenario: MarketScenario) -> Dict[str, np.ndarray]:
        """Generate OHLCV data from price series"""
        n_periods = len(prices)

        # Initialize arrays
        opens = np.zeros(n_periods)
        highs = np.zeros(n_periods)
        lows = np.zeros(n_periods)
        closes = prices.copy()
        volumes = np.zeros(n_periods)

        # Generate opens (previous close + small gap)
        opens[0] = scenario.initial_price
        for i in range(1, n_periods):
            gap = np.random.normal(0, 0.001)  # Small overnight gap
            opens[i] = closes[i-1] * (1 + gap)

        # Generate highs and lows
        for i in range(n_periods):
            # Intraday volatility (fraction of daily volatility)
            intraday_vol = scenario.volatility_value / np.sqrt(252) * 0.5

            # Generate intraday range
            high_move = abs(np.random.normal(0, intraday_vol))
            low_move = abs(np.random.normal(0, intraday_vol))

            # Calculate high and low
            highs[i] = max(opens[i], closes[i]) * (1 + high_move)
            lows[i] = min(opens[i], closes[i]) * (1 - low_move)

            # Ensure OHLC consistency
            highs[i] = max(highs[i], opens[i], closes[i])
            lows[i] = min(lows[i], opens[i], closes[i])

        # Generate volumes (correlated with volatility and price moves)
        base_volume = 1000000  # Base daily volume
        for i in range(n_periods):
            # Volume increases with volatility and large price moves
            price_change = abs(closes[i] - opens[i]) / opens[i]
            volume_multiplier = 1 + price_change * 5  # Higher volume on big moves

            # Add random component
            volume_noise = np.random.lognormal(0, 0.3)
            volumes[i] = base_volume * volume_multiplier * volume_noise

        return {
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }

    def _add_price_gaps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price gaps to the data"""
        n_gaps = max(1, int(len(data) * 0.02))  # 2% of days have gaps
        gap_indices = np.random.choice(data.index[1:], size=n_gaps, replace=False)

        for idx in gap_indices:
            prev_idx = data.index[data.index.get_loc(idx) - 1]
            prev_close = data.loc[prev_idx, 'close']

            gap_size = np.random.uniform(0.01, 0.05)  # 1-5% gaps
            gap_direction = np.random.choice([-1, 1])

            gap_open = prev_close * (1 + gap_size * gap_direction)
            data.loc[idx, 'open'] = gap_open

            # Adjust high/low to maintain consistency
            data.loc[idx, 'high'] = max(data.loc[idx, 'high'], gap_open)
            data.loc[idx, 'low'] = min(data.loc[idx, 'low'], gap_open)

        return data

    def _add_flash_crash(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add flash crash event to the data"""
        crash_day = np.random.randint(len(data) // 4, 3 * len(data) // 4)
        crash_idx = data.index[crash_day]

        # Create flash crash (sudden drop and partial recovery)
        crash_magnitude = np.random.uniform(0.15, 0.30)  # 15-30% crash

        original_open = data.loc[crash_idx, 'open']
        crash_low = original_open * (1 - crash_magnitude)
        recovery_close = original_open * (1 - crash_magnitude * 0.6)  # Partial recovery

        data.loc[crash_idx, 'low'] = crash_low
        data.loc[crash_idx, 'close'] = recovery_close
        data.loc[crash_idx, 'volume'] *= 5  # High volume during crash

        return data

    def _add_bubble_pattern(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add bubble formation and burst pattern"""
        bubble_start = len(data) // 4
        bubble_peak = 3 * len(data) // 4

        # Vectorized accelerating growth phase
        growth_indices = np.arange(bubble_start, bubble_peak)
        acceleration = (growth_indices - bubble_start) / (bubble_peak - bubble_start)
        extra_returns = 0.002 * acceleration ** 2
        data.loc[data.index[bubble_start:bubble_peak], 'close'] *= (1 + extra_returns)

        # Vectorized bubble burst
        burst_magnitude = 0.4  # 40% crash from peak
        burst_end = min(bubble_peak + 20, len(data))
        burst_indices = np.arange(bubble_peak, burst_end)
        crash_factors = 1 - (burst_magnitude * (burst_indices - bubble_peak) / 20)
        data.loc[data.index[bubble_peak:burst_end], 'close'] *= crash_factors

        return data

    def _add_consolidation_periods(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add consolidation periods with reduced volatility"""
        n_consolidations = 2

        for _ in range(n_consolidations):
            if len(data) < 30:
                continue
            start_idx = np.random.randint(0, len(data) - 30)
            end_idx = start_idx + np.random.randint(15, 30)

            # Reduce price movement during consolidation
            consolidation_center = data.iloc[start_idx]['close']

            # Vectorized consolidation price adjustment
            actual_end = min(end_idx, len(data))
            consolidation_slice = slice(start_idx, actual_end)
            pull_factor = 0.1  # 10% pull toward center

            current_prices = data.iloc[consolidation_slice]['close'].values
            new_prices = current_prices * (1 - pull_factor) + consolidation_center * pull_factor
            data.loc[data.index[consolidation_slice], 'close'] = new_prices

            # Reduce volume during consolidation
            data.loc[data.index[consolidation_slice], 'volume'] *= 0.7

        return data
