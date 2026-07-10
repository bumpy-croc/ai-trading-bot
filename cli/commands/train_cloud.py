"""CLI command for cloud-based model training.

Provides the `atb train cloud` command for training models on
AWS SageMaker or other cloud providers.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ml.cloud.providers.base import CloudTrainingProvider

# Training window length used when neither --days nor --start-date is given
DEFAULT_TRAINING_DAYS = 365


def _parse_utc_date(value: str) -> datetime:
    """Parse an ISO date/datetime string into a UTC-aware datetime."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}: expected ISO format (YYYY-MM-DD)") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _resolve_date_range(
    days: int | None,
    start_date_str: str | None,
    end_date_str: str | None,
) -> tuple[datetime, datetime]:
    """Resolve the training data window from CLI arguments.

    Supported combinations:
    - nothing: last DEFAULT_TRAINING_DAYS days ending now
    - --days N: last N days ending now
    - --start-date [--end-date]: explicit window (end defaults to now)
    - --end-date --days N: fixed-cutoff window of N days ending at --end-date

    Raises:
        ValueError: If --days is combined with --start-date, a date string is
            invalid, or the resolved start is not before the end.
    """
    if days is not None and start_date_str:
        raise ValueError("--days and --start-date are mutually exclusive")

    end_date = _parse_utc_date(end_date_str) if end_date_str else datetime.now(UTC)
    if start_date_str:
        start_date = _parse_utc_date(start_date_str)
    else:
        start_date = end_date - timedelta(days=days if days is not None else DEFAULT_TRAINING_DAYS)

    if start_date >= end_date:
        raise ValueError(f"start date must be before end date ({start_date} >= {end_date})")

    return start_date, end_date


def _parse_job_name(job_id: str) -> tuple[str, str] | None:
    """Recover (SYMBOL, timeframe) from a job name or ARN.

    Job names are generated as ``atb-{symbol}-{timeframe}-{YYYYmmdd-HHMMSS}``.

    Returns:
        (symbol, timeframe) tuple, or None if the name doesn't match.
    """
    job_name = job_id.split("/")[-1]
    parts = job_name.split("-")
    if len(parts) >= 4 and parts[0] == "atb" and parts[1] and parts[2]:
        return parts[1].upper(), parts[2]
    return None


def _handle_cloud(ns: argparse.Namespace) -> int:
    """Handle cloud training command.

    Launches a training job on the configured cloud provider (SageMaker by default).
    """
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train models on cloud infrastructure (AWS SageMaker)",
    )
    parser.add_argument("symbol", help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help=f"Days of training data ending at --end-date or now "
        f"(default: {DEFAULT_TRAINING_DAYS}; mutually exclusive with --start-date)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Training data start date (UTC). Mutually exclusive with --days.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        metavar="YYYY-MM-DD",
        help="Training data end date (UTC, default: now). "
        "Enables fixed-cutoff experimental protocols.",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Data timeframe (default: 1h)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs (default: 300)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=120,
        help="Sequence length (default: 120)",
    )
    parser.add_argument(
        "--force-sentiment",
        action="store_true",
        help="Force sentiment feature inclusion",
    )
    parser.add_argument(
        "--force-price-only",
        action="store_true",
        help="Force price-only model (no sentiment)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_lstm",
        choices=[
            "lstm",
            "cnn_lstm",
            "attention_lstm",
            "tcn",
            "tcn_attention",
            "tft",
            "tft_ternary",
            "lightgbm",
        ],
        help="Model architecture (default: cnn_lstm). tft_ternary is the 3-class "
        "TARGET-REDESIGN tournament entrant (c) head -- pair it with "
        "--target-type triple_barrier. lightgbm requires the optional lightgbm "
        "dependency (not installed by default) and is unrelated to --target-type "
        "meta_label, which always trains a sklearn LogisticRegression.",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="default",
        choices=["default", "lightweight", "deep"],
        help="Architecture variant (default: default)",
    )
    parser.add_argument(
        "--target-type",
        type=str,
        default="regression",
        choices=[
            "regression",
            "binary_direction",
            "triple_barrier",
            "smoothed_return",
            "meta_label",
        ],
        help="Training target (default: regression, the incumbent next-bar price target). "
        "TARGET-REDESIGN tournament entrants: binary_direction (b), triple_barrier (c), "
        "smoothed_return (d), meta_label (a). meta_label requires --primary-model-type.",
    )
    parser.add_argument(
        "--target-horizon",
        type=int,
        default=1,
        help="Forward horizon in bars for binary_direction/smoothed_return targets "
        "(default: 1; ignored by regression/triple_barrier).",
    )
    parser.add_argument(
        "--primary-model-type",
        type=str,
        default=None,
        help="Registry model_type of the primary signal to run forward when "
        "--target-type meta_label is used (e.g. 'basic'). Required for meta_label, "
        "ignored otherwise.",
    )
    parser.add_argument(
        "--instance-type",
        type=str,
        default="ml.g4dn.xlarge",
        help="Cloud instance type (default: ml.g4dn.xlarge with T4 GPU)",
    )
    parser.add_argument(
        "--no-spot",
        action="store_true",
        help="Disable spot instances (use on-demand, more expensive)",
    )
    parser.add_argument(
        "--max-runtime-hours",
        type=int,
        default=4,
        help="Maximum runtime in hours (default: 4)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="sagemaker",
        choices=["sagemaker", "local"],
        help="Cloud provider (default: sagemaker)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit job and exit immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Don't sync artifacts to local registry after completion",
    )
    parser.add_argument(
        "--input-data-s3",
        type=str,
        metavar="S3_URI",
        help="S3 URI of pre-downloaded training data (e.g., s3://bucket/training-data/BTCUSDT_1h.csv). "
        "Required when Binance API blocks cloud IPs.",
    )

    args = parser.parse_args(ns.args or [])

    # Resolve dates first: cheap validation should fail before any AWS call
    try:
        start_date, end_date = _resolve_date_range(args.days, args.start_date, args.end_date)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    # Import cloud training modules (may fail if boto3 not installed)
    try:
        from src.ml.cloud.config import CloudInstanceConfig, CloudTrainingConfig
        from src.ml.cloud.orchestrator import CloudTrainingOrchestrator
        from src.ml.cloud.providers import get_provider
        from src.ml.training_pipeline.config import DiagnosticsOptions, TrainingConfig
    except ImportError as exc:
        print(f"Error: Cloud training dependencies not available: {exc}")
        print("Install with: pip install '.[cloud]'")
        return 1

    # Validate provider configuration
    provider = get_provider(args.provider)
    if not provider.is_available():
        print(f"Error: {provider.provider_name} is not configured.")
        print()
        print("For SageMaker, set these environment variables:")
        print("  SAGEMAKER_ROLE_ARN=arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole")
        print("  SAGEMAKER_S3_BUCKET=your-training-bucket")
        print("  AWS_REGION=us-east-1")
        print()
        print("For local testing, use: --provider local")
        return 1

    training_config = TrainingConfig(
        symbol=args.symbol.upper(),
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        force_sentiment=args.force_sentiment,
        force_price_only=args.force_price_only,
        model_type=args.model_type,
        model_variant=args.model_variant,
        target_type=args.target_type,
        target_horizon=args.target_horizon,
        primary_model_type=args.primary_model_type,
        diagnostics=DiagnosticsOptions(
            generate_plots=False,  # Skip plots in cloud (no display)
            evaluate_robustness=True,
            convert_to_onnx=True,
        ),
    )

    instance_config = CloudInstanceConfig(
        instance_type=args.instance_type,
        use_spot_instances=not args.no_spot,
        max_runtime_hours=args.max_runtime_hours,
    )

    # Get storage config from environment
    cloud_config = CloudTrainingConfig.from_env(training_config)
    cloud_config.instance_config = instance_config
    cloud_config.auto_sync_artifacts = not args.no_sync
    cloud_config.input_data_s3_uri = args.input_data_s3

    # Print job summary
    print("=" * 60)
    print("Cloud Training Configuration")
    print("=" * 60)
    print(f"  Symbol:          {args.symbol.upper()}")
    print(f"  Timeframe:       {args.timeframe}")
    print(
        f"  Data Range:      {start_date.date()} to {end_date.date()} "
        f"({(end_date - start_date).days} days)"
    )
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Sequence Length: {args.sequence_length}")
    print(f"  Architecture:    {args.model_type} ({args.model_variant})")
    print(f"  Target:          {args.target_type} (horizon={args.target_horizon})")
    print()
    print(f"  Provider:        {provider.provider_name}")
    print(f"  Instance:        {args.instance_type}")
    print(f"  Spot Instances:  {not args.no_spot}")
    print(f"  Max Runtime:     {args.max_runtime_hours} hours")
    print(f"  S3 Bucket:       {cloud_config.storage_config.s3_bucket}")
    print("=" * 60)
    print()

    # Run training
    orchestrator = CloudTrainingOrchestrator(cloud_config, provider)

    if args.no_wait:
        print("Uploading training data and submitting job (not waiting for completion)...")
        try:
            job_id = orchestrator.submit_job()
            print(f"Job submitted: {job_id}")
            print()
            print("To check status:")
            print(f"  atb train cloud-status {job_id}")
            print("To download and sync artifacts once completed:")
            print(f"  atb train cloud-status {job_id} --sync")
            return 0
        except Exception as exc:
            print(f"Error: Failed to submit job: {exc}")
            return 1
    else:
        print("Starting cloud training (this may take 1-4 hours)...")
        print()

        result = orchestrator.run_training(wait=True)

        if result.success:
            print()
            print("=" * 60)
            print("Training Completed Successfully!")
            print("=" * 60)
            print(f"  Job ID:          {result.job_id}")
            print(f"  Duration:        {result.duration_seconds:.1f} seconds")
            print(f"  Artifacts:       {result.artifact_path}")
            if result.metrics:
                print()
                print("  Metrics:")
                for key, value in result.metrics.items():
                    print(f"    {key}: {value:.4f}")
            print("=" * 60)
            return 0
        else:
            print()
            print("=" * 60)
            print("Training Failed")
            print("=" * 60)
            print(f"  Error: {result.error}")
            print("=" * 60)
            return 1


def _handle_cloud_status(ns: argparse.Namespace) -> int:
    """Check status of a cloud training job."""
    parser = argparse.ArgumentParser(description="Check cloud training job status")
    parser.add_argument("job_id", help="Job ID from 'atb train cloud --no-wait'")
    parser.add_argument(
        "--provider",
        type=str,
        default="sagemaker",
        choices=["sagemaker", "local"],
        help="Cloud provider (default: sagemaker)",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Download artifacts and sync to the local registry if job is complete",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Trading symbol for --sync when it cannot be derived from the job name",
    )

    args = parser.parse_args(ns.args or [])

    try:
        from src.ml.cloud.providers import get_provider
    except ImportError as exc:
        print(f"Error: Cloud training dependencies not available: {exc}")
        print("Install with: pip install '.[cloud]'")
        return 1

    provider = get_provider(args.provider)
    if not provider.is_available():
        print(f"Error: {provider.provider_name} is not configured.")
        return 1

    try:
        status = provider.get_job_status(args.job_id)

        print("=" * 60)
        print("Job Status")
        print("=" * 60)
        print(f"  Job Name:    {status.job_name}")
        print(f"  Status:      {status.status}")
        print(f"  Start Time:  {status.start_time or 'N/A'}")
        print(f"  End Time:    {status.end_time or 'N/A'}")
        if status.duration_seconds:
            print(f"  Duration:    {status.duration_seconds:.1f} seconds")
        if status.failure_reason:
            print(f"  Error:       {status.failure_reason}")
        if status.output_s3_path:
            print(f"  Output:      {status.output_s3_path}")
        if status.metrics:
            print()
            print("  Metrics:")
            for key, value in status.metrics.items():
                print(f"    {key}: {value:.4f}")
        print("=" * 60)

        if args.sync:
            if not status.is_successful:
                print()
                print(f"Cannot sync: job is in state {status.status}.")
                return 1
            return _sync_job_artifacts(args.job_id, status.job_name, args.symbol, provider)

        return 0 if status.is_successful or not status.is_terminal else 1

    except Exception as exc:
        print(f"Error: Failed to get job status: {exc}")
        return 1


def _sync_job_artifacts(
    job_id: str,
    job_name: str,
    symbol_override: str | None,
    provider: CloudTrainingProvider,
) -> int:
    """Download a completed job's artifacts and sync them into the registry.

    The symbol/timeframe recovered from the job name only seed a placeholder
    config; the synced bundle's own metadata.json determines the final
    registry location.
    """
    from src.ml.cloud.config import CloudTrainingConfig
    from src.ml.cloud.orchestrator import CloudTrainingOrchestrator
    from src.ml.training_pipeline.config import TrainingConfig

    parsed = _parse_job_name(job_name)
    if parsed:
        symbol, timeframe = parsed
    elif symbol_override:
        symbol, timeframe = symbol_override.upper(), "1h"
    else:
        print("Error: Cannot derive symbol from job name; pass --symbol explicitly.")
        return 1

    # Placeholder window: sync only reads symbol/timeframe, never the dates
    placeholder_config = TrainingConfig(
        symbol=symbol,
        timeframe=timeframe,
        start_date=datetime(2020, 1, 1, tzinfo=UTC),
        end_date=datetime.now(UTC),
    )

    try:
        cloud_config = CloudTrainingConfig.from_env(placeholder_config)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print()
    print("Syncing artifacts to local registry...")
    try:
        orchestrator = CloudTrainingOrchestrator(cloud_config, provider)
        artifact_path = orchestrator.sync_artifacts(job_id)
        print(f"Artifacts synced to: {artifact_path}")
        return 0
    except Exception as exc:
        print(f"Error: Failed to sync artifacts: {exc}")
        return 1


def _handle_cloud_list(ns: argparse.Namespace) -> int:
    """List cloud training job outputs stored in S3."""
    parser = argparse.ArgumentParser(description="List cloud-trained model outputs in S3")
    parser.add_argument(
        "symbol",
        nargs="?",
        default=None,
        help="Optional trading symbol filter (e.g., BTCUSDT)",
    )

    args = parser.parse_args(ns.args or [])

    try:
        import os

        from src.ml.cloud.artifacts.s3_manager import S3ArtifactManager
    except ImportError as exc:
        print(f"Error: Cloud training dependencies not available: {exc}")
        print("Install with: pip install '.[cloud]'")
        return 1

    bucket = os.getenv("SAGEMAKER_S3_BUCKET")
    if not bucket:
        print("Error: SAGEMAKER_S3_BUCKET not set")
        return 1

    try:
        s3_manager = S3ArtifactManager(bucket)
        jobs = s3_manager.list_training_jobs(symbol=args.symbol)

        scope = args.symbol.upper() if args.symbol else "all symbols"
        if not jobs:
            print(f"No cloud training outputs found for {scope}")
            return 0

        print(f"Cloud training outputs for {scope} (newest first):")
        print()
        for job in jobs:
            print(f"  - {job}")
        print()
        print("Sync one into the local registry with:")
        print("  atb train cloud-status <JOB_NAME> --sync")

        return 0

    except Exception as exc:
        print(f"Error: {exc}")
        return 1


def _handle_cloud_promote(ns: argparse.Namespace) -> int:
    """Promote a synced cloud model from price/ into another namespace."""
    parser = argparse.ArgumentParser(
        description="Promote a cloud-trained model bundle between registry namespaces "
        "(never touches the target's 'latest' symlink unless --set-latest is passed)",
    )
    parser.add_argument("symbol", help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument("version", help="Version directory name (e.g., 2026-07-05_10h30m00s_v1)")
    parser.add_argument(
        "--from",
        dest="source_type",
        type=str,
        default="price",
        help="Source namespace (default: price, where cloud syncs land)",
    )
    parser.add_argument(
        "--to",
        dest="target_type",
        type=str,
        default="basic",
        help="Target namespace (default: basic, loaded by live strategies)",
    )
    parser.add_argument(
        "--set-latest",
        action="store_true",
        help="Also point the target namespace's 'latest' symlink at this version. "
        "For basic/ this changes which model live strategies load.",
    )

    args = parser.parse_args(ns.args or [])

    from src.ml.cloud.exceptions import ModelPromotionError
    from src.ml.cloud.promotion import promote_model_version

    try:
        target = promote_model_version(
            symbol=args.symbol,
            version_id=args.version,
            source_type=args.source_type,
            target_type=args.target_type,
            set_latest=args.set_latest,
        )
    except ModelPromotionError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Promoted {args.symbol.upper()}/{args.source_type}/{args.version}")
    print(f"  -> {target}")
    if args.set_latest:
        print(f"  {args.target_type}/latest now points to {args.version}")
        if args.target_type == "basic":
            print("  WARNING: live strategies load basic/latest — verify before deploying.")
    else:
        print(f"  {args.target_type}/latest was NOT changed (pass --set-latest to update it)")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Register cloud training subcommands under 'atb train'."""
    # Main cloud training command
    p_cloud = subparsers.add_parser(
        "cloud",
        help="Train models on cloud infrastructure (AWS SageMaker)",
    )
    p_cloud.add_argument("args", nargs=argparse.REMAINDER)
    p_cloud.set_defaults(func=_handle_cloud)

    # Status check command
    p_status = subparsers.add_parser(
        "cloud-status",
        help="Check status of a cloud training job",
    )
    p_status.add_argument("args", nargs=argparse.REMAINDER)
    p_status.set_defaults(func=_handle_cloud_status)

    # List versions command
    p_list = subparsers.add_parser(
        "cloud-list",
        help="List cloud-trained model outputs in S3",
    )
    p_list.add_argument("args", nargs=argparse.REMAINDER)
    p_list.set_defaults(func=_handle_cloud_list)

    # Promotion command (price/ -> basic/ is always explicit, never automatic)
    p_promote = subparsers.add_parser(
        "cloud-promote",
        help="Promote a synced cloud model bundle (default: price/ -> basic/)",
    )
    p_promote.add_argument("args", nargs=argparse.REMAINDER)
    p_promote.set_defaults(func=_handle_cloud_promote)
