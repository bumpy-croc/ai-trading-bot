"""LightGBM baseline model for cryptocurrency price prediction.

This module implements gradient boosting with LightGBM as a fast baseline
for comparison with deep learning models. Requires manual feature engineering
but offers very fast training and built-in feature importance.

Characteristics:
    - 10-100x faster training than LSTM
    - Built-in feature importance analysis
    - Handles missing data well
    - Requires lag features and technical indicators
    - Best for directional prediction and baselines

Performance expectations:
    - Competitive with deep learning on some datasets
    - Excellent for feature importance analysis
    - Fast experimentation and hyperparameter tuning
    - Good ensemble component

Limitations:
    - Requires manual feature engineering (lags, rolling stats)
    - No built-in temporal modeling
    - Limited on high-frequency sequential patterns
    - Not suitable for multi-horizon prediction

References:
    - Research showing LightGBM competitive with LSTM on some cryptocurrencies
    - "High-Frequency Cryptocurrency Price Forecasting" (MDPI, 2024)
    - Hybrid LSTM+XGBoost outperforms standalone models
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb

    _LIGHTGBM_AVAILABLE = True
except ImportError:
    _LIGHTGBM_AVAILABLE = False
    lgb = None  # type: ignore

logger = logging.getLogger(__name__)


def _ensure_lightgbm_available() -> None:
    """Ensure lightgbm is available, raising ImportError with helpful message if not."""
    if not _LIGHTGBM_AVAILABLE:
        raise ImportError(
            "lightgbm is required for gradient boosting models but is not installed. "
            "Install it with: pip install lightgbm"
        )


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = "close",
    lags: list[int] = [1, 2, 3, 5, 10, 20],
) -> pd.DataFrame:
    """Create lag features for time series prediction.

    Lag features allow gradient boosting models to capture temporal patterns
    by using previous values as features.

    Args:
        df: DataFrame with time series data
        target_col: Column to create lags for (typically 'close')
        lags: List of lag periods (e.g., [1, 2, 5] for t-1, t-2, t-5)

    Returns:
        DataFrame with added lag features

    Example:
        >>> df_with_lags = create_lag_features(df, lags=[1, 2, 5, 10])
        >>> # Creates: close_lag_1, close_lag_2, close_lag_5, close_lag_10
    """
    result = df.copy()

    for lag in lags:
        result[f"{target_col}_lag_{lag}"] = result[target_col].shift(lag)

    return result


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = "close",
    windows: list[int] = [5, 10, 20, 50],
    features: list[str] = ["mean", "std", "min", "max"],
) -> pd.DataFrame:
    """Create rolling window features.

    Rolling statistics capture trends and volatility patterns.

    Args:
        df: DataFrame with time series data
        target_col: Column to compute rolling stats for
        windows: List of window sizes (e.g., [5, 10, 20])
        features: List of statistics to compute (mean, std, min, max)

    Returns:
        DataFrame with added rolling features

    Example:
        >>> df_with_rolling = create_rolling_features(df, windows=[5, 10, 20])
        >>> # Creates: close_rolling_mean_5, close_rolling_std_5, etc.
    """
    result = df.copy()

    for window in windows:
        rolling = result[target_col].rolling(window=window)

        if "mean" in features:
            result[f"{target_col}_rolling_mean_{window}"] = rolling.mean()
        if "std" in features:
            result[f"{target_col}_rolling_std_{window}"] = rolling.std()
        if "min" in features:
            result[f"{target_col}_rolling_min_{window}"] = rolling.min()
        if "max" in features:
            result[f"{target_col}_rolling_max_{window}"] = rolling.max()

    return result


def create_momentum_features(df: pd.DataFrame, target_col: str = "close") -> pd.DataFrame:
    """Create momentum and change features.

    Captures price momentum and percentage changes.

    Args:
        df: DataFrame with time series data
        target_col: Column to compute momentum for

    Returns:
        DataFrame with added momentum features
    """
    result = df.copy()

    # Price changes (absolute and percentage)
    result[f"{target_col}_change_1"] = result[target_col].diff(1)
    result[f"{target_col}_change_5"] = result[target_col].diff(5)
    result[f"{target_col}_pct_change_1"] = result[target_col].pct_change(1)
    result[f"{target_col}_pct_change_5"] = result[target_col].pct_change(5)

    # Momentum indicators
    result[f"{target_col}_momentum_5"] = result[target_col] / result[target_col].shift(5) - 1
    result[f"{target_col}_momentum_10"] = result[target_col] / result[target_col].shift(10) - 1
    result[f"{target_col}_momentum_20"] = result[target_col] / result[target_col].shift(20) - 1

    return result


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features from datetime index.

    Captures cyclical patterns (hour, day, month).

    Args:
        df: DataFrame with DatetimeIndex

    Returns:
        DataFrame with added time features
    """
    result = df.copy()

    # Extract time components
    result["hour"] = result.index.hour
    result["day_of_week"] = result.index.dayofweek
    result["day_of_month"] = result.index.day
    result["month"] = result.index.month

    # Cyclical encoding (sine/cosine) for better gradient boosting
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["day_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["day_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)

    return result


def engineer_features_for_lgb(
    df: pd.DataFrame,
    target_col: str = "close",
    include_technical: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create comprehensive feature set for LightGBM.

    Combines lag features, rolling statistics, momentum, and time features.

    Args:
        df: DataFrame with OHLCV data and technical indicators
        target_col: Target column (typically 'close')
        include_technical: Whether to include existing technical indicators

    Returns:
        Tuple of (feature_dataframe, feature_names)

    Example:
        >>> features_df, feature_names = engineer_features_for_lgb(price_df)
        >>> X = features_df[feature_names].dropna()
        >>> y = features_df[target_col].loc[X.index]
    """
    result = df.copy()

    # Lag features (recent history)
    result = create_lag_features(result, target_col, lags=[1, 2, 3, 5, 10, 20])

    # Rolling statistics (trends and volatility)
    result = create_rolling_features(result, target_col, windows=[5, 10, 20, 50])

    # Momentum features (price changes)
    result = create_momentum_features(result, target_col)

    # Time features (cyclical patterns)
    result = create_time_features(result)

    # Volume features (if available)
    if "volume" in result.columns:
        result = create_rolling_features(result, "volume", windows=[5, 10, 20])

    # Identify feature columns (exclude target and original OHLCV)
    exclude_cols = [target_col, "open", "high", "low", "volume"]
    if include_technical:
        # Include existing technical indicators (RSI, MACD, etc.)
        feature_cols = [col for col in result.columns if col not in exclude_cols]
    else:
        # Only use engineered features
        feature_cols = [
            col
            for col in result.columns
            if col not in exclude_cols
            and (
                col.startswith(f"{target_col}_")
                or col in ["hour", "day_of_week", "hour_sin", "hour_cos", "day_sin", "day_cos"]
            )
        ]

    return result, feature_cols


def create_lightgbm_model(
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    max_depth: int = 7,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
    random_state: int = 42,
    verbose: int = -1,
) -> Any:
    """Create LightGBM regressor for price prediction.

    Args:
        n_estimators: Number of boosting iterations
        learning_rate: Learning rate (0.01-0.1 typical)
        max_depth: Maximum tree depth (5-10 typical)
        num_leaves: Maximum number of leaves (31-127 typical)
        min_child_samples: Minimum samples in leaf (prevents overfitting)
        subsample: Row sampling ratio (0.7-1.0)
        colsample_bytree: Column sampling ratio (0.7-1.0)
        reg_alpha: L1 regularization
        reg_lambda: L2 regularization
        random_state: Random seed
        verbose: Verbosity level (-1 = silent)

    Returns:
        LightGBM regressor model

    Example:
        >>> model = create_lightgbm_model(n_estimators=1000, learning_rate=0.05)
        >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    """
    _ensure_lightgbm_available()

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        verbose=verbose,
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
    )

    return model


def train_lightgbm_with_early_stopping(
    model: Any,
    X_train: np.ndarray | pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: np.ndarray | pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    early_stopping_rounds: int = 50,
    verbose: int = 100,
) -> Any:
    """Train LightGBM model with early stopping.

    Args:
        model: LightGBM regressor
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        early_stopping_rounds: Rounds to wait for improvement
        verbose: Print metrics every N iterations

    Returns:
        Trained model

    Example:
        >>> model = create_lightgbm_model()
        >>> model = train_lightgbm_with_early_stopping(
        ...     model, X_train, y_train, X_val, y_val
        ... )
        >>> predictions = model.predict(X_test)
    """
    _ensure_lightgbm_available()

    # Configure callbacks
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose > 0),
        lgb.log_evaluation(period=verbose),
    ]

    # Train with validation set
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)], eval_metric="rmse", callbacks=callbacks
    )

    logger.info(f"Best iteration: {model.best_iteration_}")
    logger.info(f"Best score: {model.best_score_}")

    return model


def get_feature_importance(
    model: Any, feature_names: list[str], importance_type: str = "gain", top_n: int = 20
) -> pd.DataFrame:
    """Get feature importance from trained LightGBM model.

    Args:
        model: Trained LightGBM model
        feature_names: List of feature names
        importance_type: Type of importance ('gain', 'split', 'weight')
        top_n: Number of top features to return

    Returns:
        DataFrame with feature importance sorted descending

    Example:
        >>> importance = get_feature_importance(model, feature_names)
        >>> print(importance.head(10))  # Top 10 features
    """
    _ensure_lightgbm_available()

    importance = model.feature_importances_
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values(by="importance", ascending=False)

    return importance_df.head(top_n)


def create_directional_classifier(
    n_estimators: int = 1000,
    learning_rate: float = 0.05,
    max_depth: int = 7,
    num_leaves: int = 31,
) -> Any:
    """Create LightGBM classifier for directional prediction (up/down).

    Often more profitable than regression in trading. Predicts whether
    price will go up or down rather than exact price.

    Args:
        n_estimators: Number of boosting iterations
        learning_rate: Learning rate
        max_depth: Maximum tree depth
        num_leaves: Maximum leaves per tree

    Returns:
        LightGBM classifier model

    Example:
        >>> # Create binary target (1 = up, 0 = down)
        >>> y_direction = (df['close'].shift(-1) > df['close']).astype(int)
        >>> model = create_directional_classifier()
        >>> model.fit(X_train, y_direction)
    """
    _ensure_lightgbm_available()

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        objective="binary",
        metric="binary_logloss",
        boosting_type="gbdt",
        random_state=42,
        verbose=-1,
    )

    return model


# Recommended hyperparameters for cryptocurrency prediction
RECOMMENDED_HYPERPARAMETERS = {
    "default": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 7,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "early_stopping_rounds": 50,
    },
    "fast": {
        "n_estimators": 500,
        "learning_rate": 0.1,
        "max_depth": 5,
        "num_leaves": 15,
        "min_child_samples": 30,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 0.05,
        "early_stopping_rounds": 30,
    },
    "accurate": {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "max_depth": 9,
        "num_leaves": 63,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
        "early_stopping_rounds": 75,
    },
}


# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG = {
    "lags": [1, 2, 3, 5, 10, 20],
    "rolling_windows": [5, 10, 20, 50],
    "rolling_features": ["mean", "std", "min", "max"],
    "include_technical": True,  # Use existing technical indicators
    "include_time": True,  # Add time-based features
}
