"""
Model Ensemble Package

This package provides model ensemble and aggregation capabilities
for the prediction engine (Post-MVP feature).
"""

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np

from prediction.models.onnx_runner import ModelPrediction


@dataclass
class EnsembleResult:
    price: float
    confidence: float
    direction: int
    member_predictions: List[ModelPrediction]


class SimpleEnsembleAggregator:
    """Lightweight ensemble aggregator supporting mean/median/weighted methods.

    Weighted method uses confidence as weights.
    """

    def __init__(self, method: str = "mean"):
        self.method = method

    def aggregate(self, preds: Iterable[ModelPrediction]) -> EnsembleResult:
        members = list(preds)
        if not members:
            return EnsembleResult(price=0.0, confidence=0.0, direction=0, member_predictions=[])

        prices = np.array([p.price for p in members], dtype=float)
        confs = np.array([max(0.0, min(1.0, p.confidence)) for p in members], dtype=float)
        dirs = np.array([p.direction for p in members], dtype=int)

        if self.method == "median":
            agg_price = float(np.median(prices))
            agg_conf = float(np.median(confs))
        elif self.method == "weighted":
            w = confs
            if np.all(w == 0):
                w = np.ones_like(w)
            norm_w = w / w.sum()
            agg_price = float((prices * norm_w).sum())
            agg_conf = float((confs * norm_w).sum())
        else:  # mean
            agg_price = float(prices.mean())
            agg_conf = float(confs.mean())

        # Direction by majority vote; tie → 0
        votes = dirs
        pos = (votes == 1).sum()
        neg = (votes == -1).sum()
        agg_dir = 0
        if pos > neg:
            agg_dir = 1
        elif neg > pos:
            agg_dir = -1

        return EnsembleResult(
            price=agg_price, confidence=agg_conf, direction=agg_dir, member_predictions=members
        )
