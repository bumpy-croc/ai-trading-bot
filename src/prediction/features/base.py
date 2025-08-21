"""
Base Feature Extractor

This module provides the abstract base class for all feature extractors
in the prediction engine. All feature extractors must implement the
extract method and provide feature names.
"""

from abc import ABC, abstractmethod

import pandas as pd


class FeatureExtractor(ABC):
    """
    Abstract base class for feature extractors.

    All feature extractors must inherit from this class and implement
    the extract method to transform raw data into ML-ready features.
    """

    def __init__(self, name: str):
        """
        Initialize the feature extractor.

        Args:
            name: Unique name for this feature extractor
        """
        self.name = name
        self._feature_names: list[str] = []

    @abstractmethod
    def extract(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from raw data.

        Args:
            data: Raw market data (OHLCV format)

        Returns:
            DataFrame with original data plus extracted features

        Raises:
            ValueError: If input data is invalid
            RuntimeError: If feature extraction fails
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        """
        Return list of feature names this extractor produces.

        Returns:
            List of feature names
        """
        pass

    def validate_input(self, data: pd.DataFrame) -> bool:
        """
        Validate input data format.

        Args:
            data: Input DataFrame to validate

        Returns:
            True if data is valid, False otherwise
        """
        if data is None or data.empty:
            return False

        required_columns = ["open", "high", "low", "close", "volume"]
        return all(col in data.columns for col in required_columns)

    def handle_missing_values(
        self, data: pd.DataFrame, method: str = "forward_fill"
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.

        Args:
            data: DataFrame with potential missing values
            method: Method to handle missing values ('forward_fill', 'backward_fill', 'drop')

        Returns:
            DataFrame with missing values handled
        """
        if method == "forward_fill":
            return data.ffill()
        elif method == "backward_fill":
            return data.bfill()
        elif method == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown missing value method: {method}")

    def get_config(self) -> dict:
        """
        Get configuration parameters for this extractor.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            "name": self.name,
            "feature_names": self.get_feature_names(),
            "type": self.__class__.__name__,
        }
