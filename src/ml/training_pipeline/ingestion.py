"""Data ingestion utilities for the training pipeline."""

from __future__ import annotations

import logging
from argparse import Namespace
from pathlib import Path
from typing import Optional

import pandas as pd

from cli.commands import data as data_commands
from src.data_providers.feargreed_provider import FearGreedProvider
from src.ml.training_pipeline.config import TrainingContext

logger = logging.getLogger(__name__)


def _resolve_latest_file(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern {pattern} in {directory}")
    return matches[0]


def download_price_data(ctx: TrainingContext) -> pd.DataFrame:
    """Download OHLCV data for the requested symbol and date range."""

    ns = Namespace(
        symbol=ctx.symbol_exchange,
        timeframe=ctx.config.timeframe,
        start_date=ctx.start_iso,
        end_date=ctx.end_iso,
        output_dir=str(ctx.paths.data_dir),
        format="csv",
    )
    logger.info(
        "Downloading price data for %s (%s %s-%s)",
        ctx.config.symbol,
        ctx.config.timeframe,
        ctx.config.start_date.date(),
        ctx.config.end_date.date(),
    )
    status = data_commands._download(ns)
    if status != 0:
        raise RuntimeError("Price data download failed")

    latest_file = _resolve_latest_file(ctx.price_data_glob, ctx.paths.data_dir)
    logger.debug("Using price data file %s", latest_file)
    if latest_file.suffix == ".feather":
        df = pd.read_feather(latest_file)
    else:
        df = pd.read_csv(latest_file, encoding="utf-8")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df.sort_index()


def load_sentiment_data(ctx: TrainingContext) -> Optional[pd.DataFrame]:
    """Retrieve sentiment data or return None if unavailable."""

    if ctx.config.force_price_only:
        logger.info("Price-only mode forced; skipping sentiment download")
        return None

    try:
        provider = FearGreedProvider()
        df = provider.get_historical_sentiment(
            ctx.config.symbol, ctx.config.start_date, ctx.config.end_date
        )
        logger.info("Loaded %d sentiment points", len(df))
        return df
    except (
        Exception
    ) as exc:  # noqa: BLE001 - Catch all provider errors (network, API changes, parsing)
        # Sentiment data is optional - allow training to continue with price-only features
        # if sentiment download fails for any reason
        logger.warning("Sentiment download failed: %s", exc)
        return None
