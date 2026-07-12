"""Data ingestion utilities for the training pipeline."""

from __future__ import annotations

import logging
import os
from datetime import UTC, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_providers.feargreed_provider import FearGreedProvider
from src.ml.training_pipeline.config import TrainingContext

logger = logging.getLogger(__name__)

# SageMaker S3 data channel path
SAGEMAKER_INPUT_DATA_PATH = Path("/opt/ml/input/data/training")

# Minimum fraction of expected fixed-frequency bars a corpus must contain.
# Below this the range has real holes (missing rows), not just a late listing date.
MIN_CORPUS_COVERAGE_RATIO = 0.99


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
        s3_client = boto3.client(
            "s3", region_name=os.getenv("AWS_REGION", "us-east-1"), config=config
        )
    except ImportError as exc:
        raise RuntimeError("boto3 required for S3 download") from exc

    # Parse S3 URI
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    path = s3_uri[5:]
    bucket, _, key = path.partition("/")

    # Validate bucket and key are non-empty to prevent cryptic boto3 errors
    if not bucket:
        raise ValueError(f"S3 URI missing bucket name: {s3_uri}")
    if not key:
        raise ValueError(f"S3 URI missing object key: {s3_uri}")

    local_path.mkdir(parents=True, exist_ok=True)
    local_file = local_path / Path(key).name

    logger.info(f"Downloading training data from s3://{bucket}/{key}")
    s3_client.download_file(bucket, key, str(local_file))
    logger.info(f"Downloaded to {local_file}")
    return local_file


def _validate_data_coverage(df: pd.DataFrame, ctx: TrainingContext, file_path: Path) -> None:
    """Validate that loaded data covers the expected date range.

    Args:
        df: Loaded price DataFrame with timestamp index
        ctx: Training context with config
        file_path: Path to data file (for logging)

    Raises:
        ValueError: If data does not cover the requested date range
    """
    if df.empty:
        raise ValueError(f"Data file is empty: {file_path}")

    # Cast to Timestamp for type safety - index should be DatetimeIndex
    # Normalize timezone awareness to prevent comparison errors
    data_start = pd.Timestamp(df.index.min()).tz_localize(None)
    data_end = pd.Timestamp(df.index.max()).tz_localize(None)
    expected_start = pd.Timestamp(ctx.config.start_date).tz_localize(None)
    expected_end = pd.Timestamp(ctx.config.end_date).tz_localize(None)

    # A candle's index is its *open* time, and a symbol's first-ever candle can open
    # hours after midnight on its listing day (e.g. ETHUSDT opens 04:00 on 2017-08-17),
    # so compare calendar days on the start side; data starting on a later day still
    # fails. Same day-boundary semantics as _validate_corpus_coverage.
    if data_start.normalize() > expected_start.normalize():
        raise ValueError(
            f"Data starts at {data_start.date()} but training expects data from "
            f"{expected_start.date()}. Check S3 input channel configuration."
        )
    # For --days runs end_date is "now", but the staged channel can only contain bars
    # up to the most recent closed candle at staging time, so clamp to now and require
    # coverage through the last bar opening before the end day's midnight. Bar-level
    # completeness was already enforced at staging time by _validate_corpus_coverage.
    bar = _timeframe_delta(ctx.config.timeframe)
    effective_end = min(expected_end, pd.Timestamp.now(tz="UTC").tz_localize(None))
    if data_end < effective_end.normalize() - bar:
        raise ValueError(
            f"Data ends at {data_end.date()} but training expects data through "
            f"{expected_end.date()}. Check S3 input channel configuration."
        )

    # Warn if filename doesn't match expected symbol (best-effort check)
    filename = file_path.name.upper()
    expected_symbol = ctx.config.symbol.upper()
    if expected_symbol not in filename:
        logger.warning(
            f"Data file '{file_path.name}' may not match expected symbol '{ctx.config.symbol}'. "
            "Verify S3 input channel contains correct data."
        )

    logger.info(
        f"Data validation passed: covers {data_start.date()} to {data_end.date()} "
        f"({len(df)} rows)"
    )


def _timeframe_delta(timeframe: str) -> timedelta:
    """Convert a timeframe string (e.g. ``1h``) to its candle interval.

    Raises:
        RuntimeError: If the timeframe cannot be parsed (coverage validation
            would be meaningless with an unknown bar interval).
    """
    units = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}
    try:
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1]) if len(timeframe) > 1 else 1
        if unit in units and value > 0:
            return timedelta(**{units[unit]: value})
    except (ValueError, TypeError, IndexError):
        pass
    raise RuntimeError(f"Cannot validate corpus coverage for unknown timeframe: {timeframe!r}")


def _corpus_failure(ctx: TrainingContext, reason: str) -> RuntimeError:
    """Build the loud, actionable error raised when a training corpus is unusable."""
    return RuntimeError(
        f"Training corpus for {ctx.config.symbol} {ctx.config.timeframe} "
        f"({ctx.config.start_date.date()} to {ctx.config.end_date.date()}) "
        f"could not be loaded: {reason}. Training never falls back to a third-party "
        "data source (silent source-switching mid-corpus is a data-integrity hazard, "
        "see #909). Prefill the local cache with "
        f"`atb data prefill-cache --symbols {ctx.config.symbol} "
        f"--timeframes {ctx.config.timeframe}`, provide the data explicitly via "
        "--input-data-s3, or adjust the training date range."
    )


def _validate_corpus_coverage(
    df: pd.DataFrame,
    ctx: TrainingContext,
    query_start: pd.Timestamp,
    query_end: pd.Timestamp,
) -> None:
    """Validate that a corpus covers the requested window without internal gaps.

    Raises:
        RuntimeError: If the corpus misses the window boundaries or has holes.
    """
    if df is None or df.empty:
        raise _corpus_failure(ctx, "no candles were returned")

    bar = _timeframe_delta(ctx.config.timeframe)
    data_start = pd.Timestamp(df.index.min())
    data_end = pd.Timestamp(df.index.max())

    # A candle's index is its *open* time, so the last bar of the window opens up to one
    # interval before the end boundary. Windows that extend to now (or the future) can
    # only contain bars up to the most recent closed candle, so clamp before comparing.
    effective_end = min(query_end, pd.Timestamp.now(tz="UTC"))
    # The start side only requires the same calendar day: a symbol's first-ever candle can
    # open hours after midnight on its listing day (e.g. ETHUSDT opens 04:00 on 2017-08-17).
    # That is market history, not a cache gap; real holes are caught by the ratio check
    # below, which is computed from the corpus's own span rather than the query window.
    if data_start.normalize() > query_start.normalize() or data_end < (effective_end - bar):
        raise _corpus_failure(
            ctx,
            f"available data ({data_start} to {data_end}) does not cover "
            f"the requested window ({query_start} to {query_end})",
        )

    expected_bars = int((data_end - data_start) / bar) + 1
    coverage_ratio = len(df) / expected_bars if expected_bars > 0 else 0.0
    if coverage_ratio < MIN_CORPUS_COVERAGE_RATIO:
        raise _corpus_failure(
            ctx,
            f"only {len(df)} of {expected_bars} expected bars are present "
            f"({coverage_ratio:.1%} coverage) between {data_start} and {data_end}",
        )


def load_training_corpus(ctx: TrainingContext) -> pd.DataFrame:
    """Load the OHLCV training corpus via the parquet cache backed by Binance.

    Consults the year-based parquet cache (populated by ``atb data prefill-cache``)
    before any network fetch; missing years are fetched from Binance and cached, so
    repeated runs (e.g. multi-entrant model tournaments) see identical rows without
    re-downloading. There is deliberately no third-party fallback: a training corpus
    must come from a single source, so failures and coverage gaps raise loudly
    instead of silently switching providers mid-corpus (#909).

    Args:
        ctx: Training context with config and paths

    Returns:
        DataFrame with OHLCV data covering [start_date, end_date], indexed by
        UTC timestamp

    Raises:
        RuntimeError: If the corpus cannot be fetched or fails coverage validation
    """
    # End-of-day boundaries (ctx.start_iso/end_iso) so the corpus covers the full
    # requested last day instead of truncating it at midnight.
    query_start = pd.Timestamp(ctx.start_iso)
    query_end = pd.Timestamp(ctx.end_iso)

    if ctx.config.use_mock_data:
        # Test-only escape hatch (mirrors `atb live --mock-data`,
        # src/engines/live/runner.py): deterministic synthetic OHLCV, zero
        # network, zero disk cache -- lets acceptance/integration tests
        # exercise the REAL training pipeline end-to-end. Never set by any
        # production CLI path.
        from src.data_providers.mock_data_provider import MockDataProvider

        provider: Any = MockDataProvider(seed=42)
    else:
        from src.config.paths import get_cache_dir
        from src.data_providers.binance_provider import BinanceProvider
        from src.data_providers.cached_data_provider import CachedDataProvider

        provider = CachedDataProvider(BinanceProvider(), cache_dir=str(get_cache_dir()))

    try:
        df = provider.get_historical_data(
            ctx.symbol_exchange,
            ctx.config.timeframe,
            query_start.to_pydatetime(),
            query_end.to_pydatetime(),
        )
    except (ConnectionError, TimeoutError, ValueError, OSError) as exc:
        raise _corpus_failure(ctx, f"provider error: {exc}") from exc

    if df is not None and not df.empty and df.index.tz is None:
        df = df.tz_localize(UTC)

    _validate_corpus_coverage(df, ctx, query_start, query_end)
    logger.info(
        "Training corpus for %s %s: %d candles (%s to %s)",
        ctx.config.symbol,
        ctx.config.timeframe,
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df[(df.index >= query_start) & (df.index <= query_end)]


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
            df = _load_price_data_file(latest_file)
            _validate_data_coverage(df, ctx, latest_file)
            return df
        logger.warning("SageMaker input path exists but contains no data files")

    # Priority 2: S3 URI passed directly
    if s3_data_uri:
        logger.info(f"Downloading training data from S3: {s3_data_uri}")
        downloaded_file = _download_from_s3(s3_data_uri, ctx.paths.data_dir)
        return _load_price_data_file(downloaded_file)

    # Priority 3: prefilled parquet cache backed by Binance (single-source, fail-loud)
    return load_training_corpus(ctx)


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
    # format="mixed": Binance's earliest kline archives mix on-the-hour and sub-second
    # timestamps within one file (e.g. ETHUSDT 2018-02 has a run offset by ":28:14.8").
    # A plain pd.to_datetime() infers one fixed format from the first rows and raises
    # ValueError the moment a later row's format differs.
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
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
        ConnectionError,
        TimeoutError,
        ValueError,
        KeyError,
        pd.errors.EmptyDataError,
        RuntimeError,
    ) as exc:
        # Sentiment data is optional - allow training to continue with price-only features
        # if sentiment download fails for expected reasons (network, parsing, API changes)
        logger.warning("Sentiment download failed: %s", exc)
        return None
