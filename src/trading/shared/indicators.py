"""Backwards-compatible exports for indicator extraction helpers."""

from __future__ import annotations

from src.tech.adapters import row_extractors

extract_indicators = row_extractors.extract_indicators
extract_sentiment_data = row_extractors.extract_sentiment_data
extract_ml_predictions = row_extractors.extract_ml_predictions

__all__ = [
    "extract_indicators",
    "extract_sentiment_data",
    "extract_ml_predictions",
]
