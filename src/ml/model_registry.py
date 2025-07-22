from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Tuple, List

import numpy as np
import pandas as pd
import onnxruntime as ort

__all__ = [
    "ModelInfo",
    "ModelRegistry",
]

VOLUME_ROLLING_WINDOW = 1000  # * Rolling window for volume z-score
EPSILON = 1e-9  # * Small value to prevent division by zero

# ---------------------------------------------------------------------------
# Normalisation helper functions
# ---------------------------------------------------------------------------

def _minmax_price_normalize(
    df: pd.DataFrame, seq_len: int
) -> Tuple[pd.DataFrame, List[str]]:
    """Replicate the rolling min-max normalisation used by legacy models.

    Adds *_normalized columns for close, volume, high, low, open and returns
    the updated dataframe plus the list of feature column names *in order*.
    """
    df = df.copy()
    price_feats = ["close", "volume", "high", "low", "open"]
    feature_cols: List[str] = []

    for feat in price_feats:
        if feat not in df.columns:
            # If the caller forgot to supply this column just create zeros to
            # keep feature shape consistent.
            df[feat] = 0.0
        norm_col = f"{feat}_normalized"
        if norm_col not in df.columns:
            df[norm_col] = df[feat].rolling(seq_len, min_periods=1).apply(
                lambda x: (
                    (x[-1] - (min_val := np.min(x))) / ((max_val := np.max(x)) - min_val)
                    if (max_val := np.max(x)) != (min_val := np.min(x)) else 0.5
                ),
                raw=True,
            )
        feature_cols.append(norm_col)
    return df, feature_cols


def _log_return_normalize(
    df: pd.DataFrame, _seq_len: int
) -> Tuple[pd.DataFrame, List[str]]:
    """Feature engineering for the v2 GRU price model.

    Produces log return, high-low range and volume z-score features.
    No window-based normalisation is needed because these transforms are
    scale-invariant.
    """
    df = df.copy()
    # Basic safety: ensure numeric types
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["log_return"] = np.log(df["close"]).diff().fillna(0.0)
    df["hl_range"] = ((df["high"] - df["low"]) / df["close"]).fillna(0.0)
    rolling_mean = df["volume"].rolling(VOLUME_ROLLING_WINDOW).mean()
    rolling_std = df["volume"].rolling(VOLUME_ROLLING_WINDOW).std().replace(0, np.nan)
    df["volume_z"] = ((df["volume"] - rolling_mean) / (rolling_std + EPSILON)).fillna(0.0)

    return df, ["log_return", "hl_range", "volume_z"]

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

class ModelInfo:
    """Lightweight container for model metadata and ONNX runtime session."""

    def __init__(
        self,
        name: str,
        path: str | Path,
        normalise_fn: Callable[[pd.DataFrame, int], Tuple[pd.DataFrame, List[str]]],
        expected_features: int,
    ) -> None:
        self.name = name
        self.path = str(path)
        self.normalise_fn = normalise_fn
        self.expected_features = expected_features

        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Model file not found: {self.path}")

        # Lazy session initialisation – keep memory footprint low
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def session(self) -> ort.InferenceSession:
        if self._session is None:
            self._session = ort.InferenceSession(self.path, providers=["CPUExecutionProvider"])
            self._input_name = self._session.get_inputs()[0].name
        return self._session

    @property
    def input_name(self) -> str:
        if self._input_name is None:
            _ = self.session  # Forces session creation which sets input name
        return self._input_name  # type: ignore

    # Normalise returns (updated_df, feature_cols)
    def normalise(self, df: pd.DataFrame, seq_len: int) -> Tuple[pd.DataFrame, List[str]]:
        return self.normalise_fn(df, seq_len)


class ModelRegistry:
    """Singleton registry holding all ML models and their metadata."""

    _instance: "ModelRegistry" | None = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_models"):
            self._models: Dict[str, ModelInfo] = {}
            self._init_registry()

    def _init_registry(self) -> None:
        if self._models:  # already initialised
            return
        root = Path(__file__).resolve().parents[2]  # project root
        self._register(
            ModelInfo(
                "btc_price_minmax",
                root / "src/ml" / "btcusdt_price.onnx",
                _minmax_price_normalize,
                expected_features=5,
            )
        )
        # Register optional v2 model – skip if file missing
        try:
            self._register(
                ModelInfo(
                    "btc_price_v2",
                    root / "src/ml" / "btcusdt_price_v2.onnx",
                    _log_return_normalize,
                    expected_features=3,
                )
            )
        except FileNotFoundError:
            # Model not present in repo – ignore. It can be added later.
            pass

    def _register(self, model: ModelInfo) -> None:
        self._models[model.name] = model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @classmethod
    def load_model(cls, name: str) -> ModelInfo:
        instance = cls()
        if name not in instance._models:
            raise ValueError(f"Model '{name}' is not registered.")
        return instance._models[name]