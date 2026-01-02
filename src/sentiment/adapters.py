from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def merge_historical_sentiment(
    df: pd.DataFrame,
    sentiment_provider,
    symbol: str,
    timeframe: str,
    start,
    end,
) -> pd.DataFrame:
    if sentiment_provider is None:
        return df
    try:
        s_df = sentiment_provider.get_historical_sentiment(symbol, start, end)
        if s_df is None or s_df.empty:
            return df
        s_df = sentiment_provider.aggregate_sentiment(s_df, window=timeframe)
        out = df.join(s_df, how="left")
        # Forward fill and default
        for col in [
            "sentiment_score",
            "sentiment_primary",
            "sentiment_momentum",
            "sentiment_volatility",
        ]:
            if col in out.columns:
                out[col] = out[col].ffill()
        if "sentiment_score" in out.columns:
            out["sentiment_score"] = out["sentiment_score"].fillna(0)
        return out
    except Exception as e:
        logger.warning(
            "Failed to merge historical sentiment for %s: %s. Returning data without sentiment.",
            symbol,
            e,
            exc_info=True,
        )
        return df


def apply_live_sentiment(
    df: pd.DataFrame,
    sentiment_provider,
    recent_hours: int = 4,
) -> pd.DataFrame:
    if sentiment_provider is None or df is None or df.empty:
        return df
    try:
        live = sentiment_provider.get_live_sentiment()
        if not isinstance(live, dict) or not live:
            return df
        recent_mask = df.index >= (df.index.max() - pd.Timedelta(hours=recent_hours))
        for feature, value in live.items():
            if feature not in df.columns:
                df[feature] = 0.0
            df.loc[recent_mask, feature] = value
        df["sentiment_freshness"] = 0
        df.loc[recent_mask, "sentiment_freshness"] = 1
    except Exception as e:
        logger.warning(
            "Failed to apply live sentiment: %s. Returning data without recent sentiment updates.",
            e,
            exc_info=True,
        )
        return df
    return df
