import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List
import logging

from .sentiment_provider import SentimentDataProvider

logger = logging.getLogger(__name__)


class FearGreedProvider(SentimentDataProvider):
    """
    Free sentiment provider using Alternative.me Fear & Greed Index.

    - Historical daily data (multi-year) via `https://api.alternative.me/fng/?limit=0`.
    - Maps index value [0..100] to [0..1] as `sentiment_primary`.
    - Derives momentum, volatility, and simple moving averages.
    - Provides resampling to target timeframe with forward-fill and freshness checks.
    """

    BASE_URL = "https://api.alternative.me/fng/"

    def __init__(self, freshness_days: int = 7):
        super().__init__()
        self.freshness_days = freshness_days
        self.data: pd.DataFrame = pd.DataFrame()
        self._load_data()

    def calculate_sentiment_score(self, sentiment_data: list[dict]) -> float:
        """
        Compute a normalized score in [-1, 1] from fear/greed values.

        If multiple points are provided, use the mean of available numeric values.
        """
        if not sentiment_data:
            return 0.0
        values: list[float] = []
        for item in sentiment_data:
            try:
                v_raw = item.get("value")
                if v_raw is None:
                    continue
                v = float(v_raw)
                # Map 0..100 to 0..1 then to -1..1
                n01 = max(0.0, min(100.0, v)) / 100.0
                values.append(n01 * 2 - 1)
            except Exception:
                continue
        if not values:
            return 0.0
        return float(np.mean(values))

    def _load_data(self) -> None:
        try:
            params = {"limit": 0, "format": "json"}
            resp = requests.get(self.BASE_URL, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            records = payload.get("data", [])
            if not records:
                logger.warning("FearGreedProvider: empty dataset received")
                self.data = pd.DataFrame()
                return

            df = pd.DataFrame([
                {
                    "timestamp": datetime.fromtimestamp(int(r.get("timestamp", 0)), tz=timezone.utc),
                    "value": float(r.get("value", 0.0)),
                    "classification": r.get("value_classification", "Unknown"),
                }
                for r in records
                if r.get("timestamp") is not None
            ])
            if df.empty:
                self.data = df
                return

            df = df.sort_values("timestamp").set_index("timestamp")

            # Map 0..100 -> 0..1
            df["sentiment_primary"] = df["value"].clip(lower=0, upper=100) / 100.0

            # Momentum: rate of change
            df["sentiment_momentum"] = df["sentiment_primary"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Volatility: rolling std (7-day window)
            df["sentiment_volatility"] = (
                df["sentiment_primary"].rolling(window=7, min_periods=2).std().fillna(0.0)
            )

            # Moving averages
            for window in [3, 7, 14]:
                df[f"sentiment_ma_{window}"] = (
                    df["sentiment_primary"].rolling(window=window, min_periods=1).mean()
                )

            # Binary extremes at fixed thresholds (mapped to 0..1 scale)
            threshold = 0.8  # very greedy
            df["sentiment_extreme_positive"] = (df["sentiment_primary"] > threshold).astype(float)
            threshold_neg = 0.2  # very fearful
            df["sentiment_extreme_negative"] = (df["sentiment_primary"] < threshold_neg).astype(float)

            self.data = df
            logger.info(
                "FearGreedProvider: loaded %d records (%s to %s)",
                len(df), df.index.min(), df.index.max()
            )
        except Exception as e:
            logger.error("FearGreedProvider: failed to load data: %s", e)
            self.data = pd.DataFrame()

    def _is_fresh(self, now_ts: datetime) -> bool:
        if self.data.empty:
            return False
        last = self.data.index.max()
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        # Normalize now_ts to timezone-aware UTC for safe subtraction
        if now_ts.tzinfo is None:
            now_ts = now_ts.replace(tzinfo=timezone.utc)
        return (now_ts - last) <= timedelta(days=self.freshness_days)

    def get_historical_sentiment(
        self,
        symbol: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        # Fear & Greed is market-level; symbol ignored except for logging
        if end is None:
            end = datetime.now(timezone.utc)
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)

        if self.data.empty:
            logger.warning("FearGreedProvider: no data loaded")
            return pd.DataFrame()

        mask = (self.data.index >= start) & (self.data.index <= end)
        df = self.data.loc[mask].copy()
        if df.empty:
            logger.warning("FearGreedProvider: no records in range %s to %s", start, end)
            return pd.DataFrame()

        # Only return sentiment_* columns expected by downstream
        cols = [c for c in df.columns if c.startswith("sentiment_")]
        return df[cols]

    def aggregate_sentiment(self, df: pd.DataFrame, window: str = '1d') -> pd.DataFrame:
        if df.empty:
            return df
        agg = df.resample(window).agg({
            "sentiment_primary": "mean",
            "sentiment_momentum": "mean",
            "sentiment_volatility": "mean",
            "sentiment_extreme_positive": "max",
            "sentiment_extreme_negative": "max",
            "sentiment_ma_3": "mean",
            "sentiment_ma_7": "mean",
            "sentiment_ma_14": "mean",
        }).ffill()
        return agg

    def get_sentiment_for_date(self, date: datetime) -> Dict[str, float]:
        if self.data.empty:
            return self._neutral()
        if date.tzinfo is None:
            date = date.replace(tzinfo=timezone.utc)
        idx = self.data.index.get_indexer([date], method='ffill')
        if idx.size == 0 or idx[0] < 0:
            return self._neutral()
        row = self.data.iloc[idx[0]]
        return {c: float(row[c]) for c in self.data.columns if c.startswith("sentiment_")}

    def _neutral(self) -> Dict[str, float]:
        return {
            "sentiment_primary": 0.5,
            "sentiment_momentum": 0.0,
            "sentiment_volatility": 0.3,
            "sentiment_extreme_positive": 0.0,
            "sentiment_extreme_negative": 0.0,
            "sentiment_ma_3": 0.5,
            "sentiment_ma_7": 0.5,
            "sentiment_ma_14": 0.5,
        }