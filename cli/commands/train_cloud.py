"""CLI command for cloud-based model training.

Provides the `atb train cloud` command for training models on
AWS SageMaker or other cloud providers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta


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
        default=365,
        help="Days of training data (default: 365)",
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

    args = parser.parse_args(ns.args or [])

    # Import cloud training modules (may fail if boto3 not installed)
    try:
        from src.ml.cloud.config import CloudInstanceConfig, CloudTrainingConfig
        from src.ml.cloud.orchestrator import CloudTrainingOrchestrator
        from src.ml.cloud.providers import get_provider
        from src.ml.training_pipeline.config import DiagnosticsOptions, TrainingConfig
    except ImportError as exc:
        print(f"Error: Cloud training dependencies not available: {exc}")
        print("Install with: pip install boto3")
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

    # Build training configuration
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)

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

    # Print job summary
    print("=" * 60)
    print("Cloud Training Configuration")
    print("=" * 60)
    print(f"  Symbol:          {args.symbol.upper()}")
    print(f"  Timeframe:       {args.timeframe}")
    print(f"  Data Range:      {start_date.date()} to {end_date.date()} ({args.days} days)")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch Size:      {args.batch_size}")
    print(f"  Sequence Length: {args.sequence_length}")
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
        print("Submitting training job (not waiting for completion)...")
        try:
            job_id = orchestrator.submit_job()
            print(f"Job submitted: {job_id}")
            print()
            print("To check status:")
            print(f"  atb train cloud-status {job_id}")
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
        help="Sync artifacts if job is complete",
    )

    args = parser.parse_args(ns.args or [])

    try:
        from src.ml.cloud.providers import get_provider
    except ImportError as exc:
        print(f"Error: Cloud training dependencies not available: {exc}")
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

        if args.sync and status.is_successful:
            print()
            print("Syncing artifacts to local registry...")
            # Would need full config to sync, simplified for status check
            print("Use 'atb train cloud' with --wait to auto-sync on completion.")

        return 0 if status.is_successful or not status.is_terminal else 1

    except Exception as exc:
        print(f"Error: Failed to get job status: {exc}")
        return 1


def _handle_cloud_list(ns: argparse.Namespace) -> int:
    """List S3 model versions."""
    parser = argparse.ArgumentParser(description="List model versions in S3")
    parser.add_argument("symbol", help="Trading symbol (e.g., BTCUSDT)")
    parser.add_argument(
        "--model-type",
        type=str,
        default="basic",
        help="Model type (default: basic)",
    )

    args = parser.parse_args(ns.args or [])

    try:
        import os

        from src.ml.cloud.artifacts.s3_manager import S3ArtifactManager
    except ImportError as exc:
        print(f"Error: Cloud training dependencies not available: {exc}")
        return 1

    bucket = os.getenv("SAGEMAKER_S3_BUCKET")
    if not bucket:
        print("Error: SAGEMAKER_S3_BUCKET not set")
        return 1

    try:
        s3_manager = S3ArtifactManager(bucket)
        versions = s3_manager.list_model_versions(args.symbol.upper(), args.model_type)

        if not versions:
            print(f"No model versions found for {args.symbol.upper()}/{args.model_type}")
            return 0

        print(f"Model versions for {args.symbol.upper()}/{args.model_type}:")
        print()
        for version in versions:
            print(f"  - {version}")

        return 0

    except Exception as exc:
        print(f"Error: {exc}")
        return 1


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
        help="List model versions in S3",
    )
    p_list.add_argument("args", nargs=argparse.REMAINDER)
    p_list.set_defaults(func=_handle_cloud_list)
