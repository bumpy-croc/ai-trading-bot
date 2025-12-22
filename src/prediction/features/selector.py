"""Feature selection and normalization aligned with model training schema."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeatureSelector:
    """Selects and normalizes features according to a schema.

    The schema is expected to contain a list of features with ordering, e.g.:
    {
      "sequence_length": 120,
      "features": [
        {"name": "close_normalized", "required": true,
         "normalization": {"mean": 0.0, "std": 1.0}},
        ...
      ]
    }
    """

    def __init__(self, schema: dict[str, Any], sequence_length: int | None = None) -> None:
        self.schema = schema or {}
        self.sequence_length = int(sequence_length or self.schema.get("sequence_length", 120))
        feats = self.schema.get("features", [])
        self.ordered_features: list[dict[str, Any]] = list(feats)

    def select(self, features_df: pd.DataFrame) -> np.ndarray:
        """Select required features in order and return (seq, num_features) array.

        Applies normalization if provided in schema. Uses the last `sequence_length`
        rows from the DataFrame.
        """
        if not isinstance(features_df, pd.DataFrame):
            raise ValueError("FeatureSelector expects a pandas DataFrame")

        cols: list[str] = []
        normalizers: list[dict[str, float] | None] = []
        for f in self.ordered_features:
            name = f.get("name")
            if not name:
                raise ValueError("Feature schema entry missing 'name'")
            required = bool(f.get("required", True))
            if required and name not in features_df.columns:
                raise ValueError(f"Required feature '{name}' missing in pipeline output")
            if name in features_df.columns:
                cols.append(name)
                normalizers.append(f.get("normalization"))

        if not cols:
            raise ValueError("No matching features found for selection")

        window = features_df.tail(self.sequence_length)
        arr = window[cols].to_numpy(dtype=np.float32, copy=False)

        # Apply per-feature normalization when provided
        for j, norm in enumerate(normalizers):
            if not norm:
                continue
            mean = float(norm.get("mean", 0.0))
            std = float(norm.get("std", 1.0))
            if std == 0.0:
                std = 1e-8
            arr[:, j] = (arr[:, j] - mean) / std

        return arr
