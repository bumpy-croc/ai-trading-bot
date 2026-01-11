"""Data ingestion utilities for the training pipeline."""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from pathlib import Path

import pandas as pd

from cli.commands import data as data_commands
from src.data_providers.feargreed_provider import FearGreedProvider
from src.ml.training_pipeline.config import TrainingContext

logger = logging.getLogger(__name__)

# SageMaker S3 data channel path
SAGEMAKER_INPUT_DATA_PATH = Path("/opt/ml/input/data/training")


def _download_from_s3(s3_uri: str, local_path: Path) -> Path:
    """Download training data from S3.

    Args:
        s3_uri: S3 URI (e.g., s3://bucket/path/data.csv)
        local_path: Local directory to download to

    Returns:
        Path to downloaded file

    Raises:
        RuntimeError: If download fails
    """
    try:
        import boto3
        from botocore.config import Config

        config = Config(
            connect_timeout=10,
            read_timeout=60,
            retries={"max_attempts": 3, "mode": "standard"},
        )
        s3_client = boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1"), config=config)
    except ImportError as exc:
        raise RuntimeError("boto3 required for S3 download") from exc

    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri[5:]
    bucket, _, key = path.partition("/")

    local_path.mkdir(parents=True, exist_ok=True)
    local_file = local_path / Path(key).name

    logger.info(f"Downloading training data from s3://{bucket}/{key}")
    s3_client.download_file(bucket, key, str(local_file))
    logger.info(f"Downloaded to {local_file}")
    return local_file


def _resolve_latest_file(pattern: str, directory: Path) -> Path:
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No files matched pattern {pattern} in {directory}")
    return matches[0]


def download_price_data(ctx: TrainingContext, s3_data_uri: str | None = None) -> pd.DataFrame:
    """Download OHLCV data for the requested symbol and date range.

    Args:
        ctx: Training context with config and paths
        s3_data_uri: Optional S3 URI for pre-downloaded training data

    Returns:
        DataFrame with price data indexed by timestamp
    """
    # Priority 1: SageMaker input channel (S3 data mounted by SageMaker)
    if SAGEMAKER_INPUT_DATA_PATH.exists():
        logger.info(f"Using SageMaker input data from {SAGEMAKER_INPUT_DATA_PATH}")
        data_files = list(SAGEMAKER_INPUT_DATA_PATH.glob("*.csv")) + list(
            SAGEMAKER_INPUT_DATA_PATH.glob("*.feather")
        )
        if data_files:
            latest_file = data_files[0]  # SageMaker mounts single file
            logger.info(f"Found input data file: {latest_file}")
            return _load_price_data_file(latest_file)
        logger.warning("SageMaker input path exists but contains no data files")

    # Priority 2: S3 URI passed directly
    if s3_data_uri:
        logger.info(f"Downloading training data from S3: {s3_data_uri}")
        downloaded_file = _download_from_s3(s3_data_uri, ctx.paths.data_dir)
        return _load_price_data_file(downloaded_file)

    # Priority 3: Download from exchange API
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
    # Note: Uses internal CLI command directly to avoid subprocess overhead
    # This is intentional coupling as training pipeline needs programmatic access
    status = data_commands._download(ns)
    if status != 0:
        raise RuntimeError("Price data download failed")

    latest_file = _resolve_latest_file(ctx.price_data_glob, ctx.paths.data_dir)
    logger.debug("Using price data file %s", latest_file)
    return _load_price_data_file(latest_file)


def _load_price_data_file(file_path: Path) -> pd.DataFrame:
    """Load price data from CSV or feather file.

    Args:
        file_path: Path to data file

    Returns:
        DataFrame with price data indexed by timestamp
    """
    logger.debug(f"Loading price data from {file_path}")
    if file_path.suffix == ".feather":
        df = pd.read_feather(file_path)
    else:
        df = pd.read_csv(file_path, encoding="utf-8")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
    return df.sort_index()


def load_sentiment_data(ctx: TrainingContext) -> pd.DataFrame | None:
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
