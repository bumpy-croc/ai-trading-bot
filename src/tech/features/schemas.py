"""
Feature schemas and definitions.

This module defines the schema for features and feature extractors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class FeatureType(Enum):
    """Types of features supported by the prediction engine."""

    TECHNICAL = "technical"
    SENTIMENT = "sentiment"
    MARKET_MICROSTRUCTURE = "market_microstructure"
    NORMALIZED_PRICE = "normalized_price"
    DERIVED = "derived"


class NormalizationMethod(Enum):
    """Normalization methods for feature scaling."""

    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROLLING_MIN_MAX = "rolling_min_max"
    ROLLING_Z_SCORE = "rolling_z_score"
    NONE = "none"


@dataclass
class FeatureDefinition:
    """
    Definition of a single feature including its metadata.
    """

    name: str
    feature_type: FeatureType
    description: str
    normalization: NormalizationMethod = NormalizationMethod.NONE
    required: bool = True
    default_value: Optional[float] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    dependencies: Optional[list[str]] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class FeatureSchema:
    """
    Schema defining a collection of features for a specific purpose.
    """

    name: str
    version: str
    features: list[FeatureDefinition]
    description: str
    sequence_length: int = 120

    def get_feature_names(self) -> list[str]:
        """Get list of feature names in this schema."""
        return [f.name for f in self.features]

    def get_required_features(self) -> list[str]:
        """Get list of required feature names."""
        return [f.name for f in self.features if f.required]

    def get_features_by_type(self, feature_type: FeatureType) -> list[FeatureDefinition]:
        """Get features of a specific type."""
        return [f for f in self.features if f.feature_type == feature_type]


# Define standard technical features schema
TECHNICAL_FEATURES_SCHEMA = FeatureSchema(
    name="technical_features_v1",
    version="1.0.0",
    description="Standard technical indicators and price features",
    sequence_length=120,
    features=[
        # Normalized price features
        FeatureDefinition(
            name="close_normalized",
            feature_type=FeatureType.NORMALIZED_PRICE,
            description="Min-max normalized close price over sequence window",
            normalization=NormalizationMethod.ROLLING_MIN_MAX,
            required=True,
        ),
        FeatureDefinition(
            name="volume_normalized",
            feature_type=FeatureType.NORMALIZED_PRICE,
            description="Min-max normalized volume over sequence window",
            normalization=NormalizationMethod.ROLLING_MIN_MAX,
            required=True,
        ),
        FeatureDefinition(
            name="high_normalized",
            feature_type=FeatureType.NORMALIZED_PRICE,
            description="Min-max normalized high price over sequence window",
            normalization=NormalizationMethod.ROLLING_MIN_MAX,
            required=True,
        ),
        FeatureDefinition(
            name="low_normalized",
            feature_type=FeatureType.NORMALIZED_PRICE,
            description="Min-max normalized low price over sequence window",
            normalization=NormalizationMethod.ROLLING_MIN_MAX,
            required=True,
        ),
        FeatureDefinition(
            name="open_normalized",
            feature_type=FeatureType.NORMALIZED_PRICE,
            description="Min-max normalized open price over sequence window",
            normalization=NormalizationMethod.ROLLING_MIN_MAX,
            required=True,
        ),
        # Technical indicators
        FeatureDefinition(
            name="rsi",
            feature_type=FeatureType.TECHNICAL,
            description="Relative Strength Index (14 period)",
            min_value=0.0,
            max_value=100.0,
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="atr",
            feature_type=FeatureType.TECHNICAL,
            description="Average True Range (14 period)",
            min_value=0.0,
            required=True,
            dependencies=["high", "low", "close"],
        ),
        FeatureDefinition(
            name="atr_pct",
            feature_type=FeatureType.TECHNICAL,
            description="ATR as percentage of close price",
            min_value=0.0,
            required=True,
            dependencies=["atr", "close"],
        ),
        # Moving averages
        FeatureDefinition(
            name="ma_20",
            feature_type=FeatureType.TECHNICAL,
            description="20-period simple moving average",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="ma_50",
            feature_type=FeatureType.TECHNICAL,
            description="50-period simple moving average",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="ma_200",
            feature_type=FeatureType.TECHNICAL,
            description="200-period simple moving average",
            required=True,
            dependencies=["close"],
        ),
        # Bollinger Bands
        FeatureDefinition(
            name="bb_upper",
            feature_type=FeatureType.TECHNICAL,
            description="Bollinger Bands upper band",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="bb_lower",
            feature_type=FeatureType.TECHNICAL,
            description="Bollinger Bands lower band",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="bb_middle",
            feature_type=FeatureType.TECHNICAL,
            description="Bollinger Bands middle band (SMA)",
            required=True,
            dependencies=["close"],
        ),
        # MACD
        FeatureDefinition(
            name="macd",
            feature_type=FeatureType.TECHNICAL,
            description="MACD line",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="macd_signal",
            feature_type=FeatureType.TECHNICAL,
            description="MACD signal line",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="macd_hist",
            feature_type=FeatureType.TECHNICAL,
            description="MACD histogram",
            required=True,
            dependencies=["close"],
        ),
        # Derived features
        FeatureDefinition(
            name="returns",
            feature_type=FeatureType.DERIVED,
            description="Price returns (close.pct_change())",
            required=True,
            dependencies=["close"],
        ),
        FeatureDefinition(
            name="volatility_20",
            feature_type=FeatureType.DERIVED,
            description="20-period rolling volatility of returns",
            min_value=0.0,
            required=True,
            dependencies=["returns"],
        ),
        FeatureDefinition(
            name="volatility_50",
            feature_type=FeatureType.DERIVED,
            description="50-period rolling volatility of returns",
            min_value=0.0,
            required=True,
            dependencies=["returns"],
        ),
        FeatureDefinition(
            name="trend_strength",
            feature_type=FeatureType.DERIVED,
            description="Trend strength relative to MA50",
            required=True,
            dependencies=["close", "ma_50"],
        ),
        FeatureDefinition(
            name="trend_direction",
            feature_type=FeatureType.DERIVED,
            description="Trend direction (1 for up, -1 for down)",
            min_value=-1.0,
            max_value=1.0,
            required=True,
            dependencies=["ma_20", "ma_50"],
        ),
    ],
)

# Define sentiment features schema (disabled for MVP)
SENTIMENT_FEATURES_SCHEMA = FeatureSchema(
    name="sentiment_features_v1",
    version="1.0.0",
    description="Sentiment analysis features (MVP: disabled)",
    features=[
        FeatureDefinition(
            name="sentiment_primary",
            feature_type=FeatureType.SENTIMENT,
            description="Primary sentiment score",
            min_value=0.0,
            max_value=1.0,
            required=False,
            default_value=0.5,
        ),
        FeatureDefinition(
            name="sentiment_momentum",
            feature_type=FeatureType.SENTIMENT,
            description="Sentiment momentum",
            required=False,
            default_value=0.0,
        ),
        FeatureDefinition(
            name="sentiment_volatility",
            feature_type=FeatureType.SENTIMENT,
            description="Sentiment volatility",
            min_value=0.0,
            required=False,
            default_value=0.3,
        ),
        FeatureDefinition(
            name="sentiment_confidence",
            feature_type=FeatureType.SENTIMENT,
            description="Sentiment data confidence",
            min_value=0.0,
            max_value=1.0,
            required=False,
            default_value=0.7,
        ),
    ],
)
