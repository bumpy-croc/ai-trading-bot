"""CLI command for cloud-based model training.

Provides the `atb train cloud` command for training models on
AWS SageMaker or other cloud providers.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path


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
    parser.add_argument(
        "--input-data-s3",
        type=str,
        metavar="S3_URI",
        help="S3 URI of pre-downloaded training data (e.g., s3://bucket/training-data/BTCUSDT_1h.csv). "
        "Required when Binance API blocks cloud IPs.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="cnn_lstm",
        choices=["cnn_lstm", "attention_lstm", "tcn", "tcn_attention", "lstm"],
        help="Model architecture (default: cnn_lstm). Options: cnn_lstm (baseline), "
        "attention_lstm (12-15%% improvement), tcn (fast), tcn_attention (hybrid)",
    )
    parser.add_argument(
        "--model-variant",
        type=str,
        default="default",
        choices=["default", "lightweight", "deep"],
        help="Model variant (default: default). lightweight=faster, deep=more accurate",
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

    # Build training configuration
    end_date = datetime.now(UTC)
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
        model_type=args.model_type,
        model_variant=args.model_variant,
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
    print(f"  Data Range:      {start_date.date()} to {end_date.date()} ({args.days} days)")
    print(f"  Model Type:      {args.model_type}")
    print(f"  Model Variant:   {args.model_variant}")
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
        print("Install with: pip install '.[cloud]'")
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


def _handle_cloud_benchmark(ns: argparse.Namespace) -> int:
    """Orchestrate multi-model benchmark training across symbols and architectures.

    Manages concurrent job submission, handles spot instance interruptions,
    downloads models to local registry, and generates benchmark reports.
    """
    parser = argparse.ArgumentParser(
        description="Train and benchmark multiple model architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all models for BTC and ETH
  atb train cloud-benchmark --symbols BTCUSDT,ETHUSDT

  # Train specific architectures only
  atb train cloud-benchmark --symbols BTCUSDT --models attention_lstm,tcn

  # Use pre-uploaded S3 data
  atb train cloud-benchmark --symbols BTCUSDT,ETHUSDT \\
      --s3-data-path s3://bucket/training-data/
        """,
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default="BTCUSDT,ETHUSDT",
        help="Comma-separated symbols to train (default: BTCUSDT,ETHUSDT)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="cnn_lstm,attention_lstm,tcn",
        help="Comma-separated model architectures (default: cnn_lstm,attention_lstm,tcn)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent training jobs (default: 2, SageMaker spot quota)",
    )
    parser.add_argument(
        "--s3-data-path",
        type=str,
        help="S3 path containing pre-uploaded training data (e.g., s3://bucket/training-data/)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=3000,
        help="Days of training data (default: 3000, ~8 years)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Training epochs (default: 300)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Data timeframe (default: 1h)",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=120,
        help="Seconds between status checks (default: 120)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/benchmark_report.json",
        help="Output path for benchmark report (default: logs/benchmark_report.json)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Skip downloading models to local registry",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=os.getenv("AWS_PROFILE", ""),
        help="AWS profile to use (default: from AWS_PROFILE env var)",
    )

    args = parser.parse_args(ns.args or [])

    # Parse symbols and models
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    model_types = [m.strip() for m in args.models.split(",")]

    # Validate model types
    valid_models = {"cnn_lstm", "attention_lstm", "tcn", "tcn_attention", "lstm"}
    for m in model_types:
        if m not in valid_models:
            print(f"Error: Invalid model type '{m}'. Valid options: {valid_models}")
            return 1

    # Build job configurations
    jobs_config = []
    for symbol in symbols:
        for model_type in model_types:
            data_file = f"{symbol}_{args.timeframe}.csv"
            jobs_config.append({
                "symbol": symbol,
                "model_type": model_type,
                "data_file": data_file,
            })

    print("=" * 70)
    print("Cloud Benchmark Training Orchestrator")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print(f"Symbols:        {', '.join(symbols)}")
    print(f"Models:         {', '.join(model_types)}")
    print(f"Total jobs:     {len(jobs_config)}")
    print(f"Max concurrent: {args.max_concurrent}")
    print(f"Epochs:         {args.epochs}")
    print(f"Days:           {args.days}")
    if args.s3_data_path:
        print(f"S3 Data:        {args.s3_data_path}")
    print("=" * 70)
    print()

    # Create orchestrator and run
    orchestrator = _BenchmarkOrchestrator(
        jobs_config=jobs_config,
        max_concurrent=args.max_concurrent,
        check_interval=args.check_interval,
        s3_data_path=args.s3_data_path,
        days=args.days,
        epochs=args.epochs,
        timeframe=args.timeframe,
        aws_profile=args.aws_profile,
        download_models=not args.no_download,
        output_path=args.output,
    )

    return orchestrator.run()


class _BenchmarkOrchestrator:
    """Orchestrates multi-model benchmark training."""

    def __init__(
        self,
        jobs_config: list[dict],
        max_concurrent: int = 2,
        check_interval: int = 120,
        s3_data_path: str | None = None,
        days: int = 3000,
        epochs: int = 300,
        timeframe: str = "1h",
        aws_profile: str = "",
        download_models: bool = True,
        output_path: str = "logs/benchmark_report.json",
    ):
        self.jobs_config = jobs_config
        self.max_concurrent = max_concurrent
        self.check_interval = check_interval
        self.s3_data_path = s3_data_path
        self.days = days
        self.epochs = epochs
        self.timeframe = timeframe
        self.aws_profile = aws_profile
        self.download_models = download_models
        self.output_path = Path(output_path)
        self.cutoff_time = datetime.now(UTC)

        # Ensure unbuffered output for real-time status updates
        try:
            sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        except AttributeError:
            pass  # Not available on all Python versions/platforms

    def _run_aws(self, args: list[str]) -> dict:
        """Run AWS CLI command and return JSON result."""
        cmd = ["aws"]
        if self.aws_profile:
            cmd.extend(["--profile", self.aws_profile])
        cmd.extend(args + ["--output", "json"])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}
        return json.loads(result.stdout) if result.stdout.strip() else {}

    def _get_jobs_after_cutoff(self) -> dict[tuple[str, str], dict]:
        """Get all training jobs started after cutoff time."""
        data = self._run_aws([
            "sagemaker", "list-training-jobs",
            "--max-results", "50",
            "--sort-by", "CreationTime",
            "--sort-order", "Descending",
        ])

        jobs: dict[tuple[str, str], dict] = {}
        for job in data.get("TrainingJobSummaries", []):
            created_str = job["CreationTime"]
            if isinstance(created_str, str):
                created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))
            else:
                created = created_str

            if created.astimezone(UTC) < self.cutoff_time:
                continue

            # Get job details for hyperparameters
            details = self._run_aws([
                "sagemaker", "describe-training-job",
                "--training-job-name", job["TrainingJobName"],
            ])
            hp = details.get("HyperParameters", {})
            symbol = hp.get("symbol", "").upper()
            model_type = hp.get("model_type", "cnn_lstm")

            if not symbol:
                continue

            key = (symbol, model_type)
            # Keep newest job for each (symbol, model_type) pair
            if key not in jobs or created > jobs[key]["created"]:
                job_info: dict = {
                    "name": job["TrainingJobName"],
                    "status": job["TrainingJobStatus"],
                    "created": created,
                    "symbol": symbol,
                    "model_type": model_type,
                }

                # Get metrics and artifacts for completed jobs
                if job["TrainingJobStatus"] == "Completed":
                    metrics = {}
                    for m in details.get("FinalMetricDataList", []):
                        metrics[m["MetricName"]] = m["Value"]
                    job_info["metrics"] = metrics
                    job_info["artifacts"] = details.get("ModelArtifacts", {}).get(
                        "S3ModelArtifacts"
                    )

                    # Calculate duration
                    end_time = details.get("TrainingEndTime")
                    start_time = details.get("TrainingStartTime")
                    if end_time and start_time:
                        try:
                            if isinstance(end_time, str):
                                end_time = datetime.fromisoformat(
                                    end_time.replace("Z", "+00:00")
                                )
                            if isinstance(start_time, str):
                                start_time = datetime.fromisoformat(
                                    start_time.replace("Z", "+00:00")
                                )
                            job_info["duration_sec"] = (
                                end_time - start_time
                            ).total_seconds()
                        except (ValueError, TypeError):
                            pass

                jobs[key] = job_info

        return jobs

    def _submit_job(self, symbol: str, model_type: str, data_file: str) -> str | None:
        """Submit a training job and return job name."""
        cmd = [
            "atb", "train", "cloud", symbol,
            "--days", str(self.days),
            "--model-type", model_type,
            "--epochs", str(self.epochs),
            "--no-wait",
        ]

        if self.s3_data_path:
            s3_uri = f"{self.s3_data_path.rstrip('/')}/{data_file}"
            cmd.extend(["--input-data-s3", s3_uri])

        print(f"  Submitting {symbol} {model_type}...")

        env = os.environ.copy()
        if self.aws_profile:
            env["AWS_PROFILE"] = self.aws_profile

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        # Parse job name from output
        for line in (result.stderr + "\n" + result.stdout).split("\n"):
            if "Job submitted:" in line:
                arn = line.split("Job submitted:")[-1].strip()
                job_name = arn.split("/")[-1]
                print(f"  ✅ Submitted: {job_name}")
                return job_name

        print(f"  ❌ Failed to submit: {result.stderr[:200]}")
        return None

    def _download_model(self, symbol: str, model_type: str, s3_path: str) -> bool:
        """Download model artifacts from S3 to local registry."""
        if not s3_path:
            return False

        local_path = Path("src/ml/models") / symbol / model_type
        local_path.mkdir(parents=True, exist_ok=True)

        print(f"  Downloading {symbol}/{model_type} to {local_path}...")

        with tempfile.TemporaryDirectory() as tmpdir:
            tar_path = Path(tmpdir) / "model.tar.gz"
            extract_path = Path(tmpdir) / "extracted"

            # Download from S3
            cmd = ["aws"]
            if self.aws_profile:
                cmd.extend(["--profile", self.aws_profile])
            cmd.extend(["s3", "cp", s3_path, str(tar_path)])

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ❌ Download failed: {result.stderr[:100]}")
                return False

            # Extract tar.gz
            extract_path.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path, "r:gz") as tar:
                tar.extractall(extract_path)

            # Find metadata.json to locate model files
            metadata_files = list(extract_path.rglob("metadata.json"))
            if not metadata_files:
                print("  ❌ No metadata.json found in archive")
                return False

            model_dir = metadata_files[0].parent

            # Copy model files to local registry
            for src_file in model_dir.iterdir():
                if src_file.is_file():
                    dst_file = local_path / src_file.name
                    dst_file.write_bytes(src_file.read_bytes())

            print(f"  ✅ Downloaded to {local_path}")
            return True

    def _generate_benchmark_report(
        self, jobs: dict[tuple[str, str], dict]
    ) -> dict:
        """Generate benchmark report comparing all models."""
        report: dict = {
            "timestamp": datetime.now(UTC).isoformat(),
            "training_params": {
                "epochs": self.epochs,
                "days": self.days,
                "timeframe": self.timeframe,
            },
        }

        # Group by symbol
        symbols_data: dict[str, dict] = {}
        for (symbol, model_type), job in jobs.items():
            if job["status"] != "Completed":
                continue

            if symbol not in symbols_data:
                symbols_data[symbol] = {
                    "baseline": None,
                    "models": [],
                    "best_model": None,
                    "best_rmse": None,
                    "improvement_pct": 0,
                }

            # Try to get metrics from job or load from downloaded metadata
            metrics = job.get("metrics", {})
            test_rmse = metrics.get("test_rmse")

            # If no metrics from SageMaker, try to load from local metadata
            if test_rmse is None:
                metadata_path = (
                    Path("src/ml/models") / symbol / model_type / "metadata.json"
                )
                if metadata_path.exists():
                    try:
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        eval_results = metadata.get("evaluation_results", {})
                        test_rmse = eval_results.get("test_rmse")
                        metrics = {
                            "test_rmse": test_rmse,
                            "train_rmse": eval_results.get("train_rmse"),
                            "mape": eval_results.get("mape"),
                        }
                    except (json.JSONDecodeError, OSError):
                        pass

            entry = {
                "model_type": model_type,
                "test_rmse": test_rmse,
                "train_rmse": metrics.get("train_rmse"),
                "mape": metrics.get("mape"),
                "job_name": job["name"],
                "duration_min": (
                    job.get("duration_sec", 0) / 60
                    if job.get("duration_sec")
                    else None
                ),
            }

            symbols_data[symbol]["models"].append(entry)
            if model_type == "cnn_lstm":
                symbols_data[symbol]["baseline"] = entry

        # Calculate improvements for each symbol
        for symbol, data in symbols_data.items():
            baseline_rmse = (
                data["baseline"]["test_rmse"] if data["baseline"] else None
            )

            if data["models"]:
                # Filter models with valid RMSE
                valid_models = [m for m in data["models"] if m["test_rmse"] is not None]
                if valid_models:
                    best = min(valid_models, key=lambda x: x["test_rmse"])
                    data["best_model"] = best["model_type"]
                    data["best_rmse"] = best["test_rmse"]

                    if baseline_rmse and best["test_rmse"]:
                        improvement = (
                            (baseline_rmse - best["test_rmse"]) / baseline_rmse * 100
                        )
                        data["improvement_pct"] = improvement
                        data["threshold_met"] = improvement >= 10

            report[symbol.lower()] = data

        return report

    def run(self) -> int:
        """Run the benchmark orchestration loop."""
        while True:
            jobs = self._get_jobs_after_cutoff()

            # Categorize jobs (treat "Stopped" as failed for retry)
            running = {k: v for k, v in jobs.items() if v["status"] == "InProgress"}
            completed = {k: v for k, v in jobs.items() if v["status"] == "Completed"}
            failed = {
                k: v for k, v in jobs.items() if v["status"] in ["Failed", "Stopped"]
            }

            # Find pending jobs (not submitted or need retry)
            pending = []
            for cfg in self.jobs_config:
                key = (cfg["symbol"], cfg["model_type"])
                if key not in jobs:
                    pending.append(cfg)
                elif jobs[key]["status"] in ["Failed", "Stopped"]:
                    pending.append(cfg)

            # Print status
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Status Check")
            print(
                f"  Completed: {len(completed)} | Running: {len(running)} | "
                f"Pending: {len(pending)} | Failed: {len(failed)}"
            )

            for job in running.values():
                elapsed = datetime.now(UTC) - job["created"]
                mins = int(elapsed.total_seconds() / 60)
                print(f"  🔄 {job['symbol']} {job['model_type']}: Running ({mins}min)")

            for job in completed.values():
                rmse = job.get("metrics", {}).get("test_rmse", "N/A")
                if rmse != "N/A":
                    rmse = f"${rmse:.2f}"
                print(f"  ✅ {job['symbol']} {job['model_type']}: RMSE={rmse}")

            for job in failed.values():
                status = job["status"]
                print(f"  ❌ {job['symbol']} {job['model_type']}: {status}")

            # Submit new jobs if slots available
            slots = self.max_concurrent - len(running)
            if slots > 0 and pending:
                print(f"\n  Submitting {min(slots, len(pending))} new job(s)...")
                for cfg in pending[:slots]:
                    self._submit_job(cfg["symbol"], cfg["model_type"], cfg["data_file"])
                    time.sleep(2)  # Small delay between submissions

            # Check if all done
            if len(completed) == len(self.jobs_config):
                print("\n" + "=" * 70)
                print("ALL TRAINING JOBS COMPLETED!")
                print("=" * 70)

                # Download models if requested
                if self.download_models:
                    print("\nDownloading models to local registry...")
                    for job in completed.values():
                        if job.get("artifacts"):
                            self._download_model(
                                job["symbol"], job["model_type"], job["artifacts"]
                            )

                # Generate benchmark report
                report = self._generate_benchmark_report(jobs)

                # Save report
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_path, "w") as f:
                    json.dump(report, f, indent=2, default=str)
                print(f"\nBenchmark report saved to: {self.output_path}")

                # Print summary
                self._print_summary(report)

                return 0

            print(f"\n  Next check in {self.check_interval}s...")
            time.sleep(self.check_interval)

    def _print_summary(self, report: dict) -> None:
        """Print benchmark summary to console."""
        print("\n" + "=" * 70)
        print("BENCHMARK SUMMARY")
        print("=" * 70)

        for symbol in [cfg["symbol"] for cfg in self.jobs_config]:
            symbol_key = symbol.lower()
            if symbol_key not in report:
                continue

            data = report[symbol_key]
            print(f"\n{symbol} Results:")

            if data.get("baseline") and data["baseline"].get("test_rmse"):
                print(f"  Baseline (CNN-LSTM) RMSE: ${data['baseline']['test_rmse']:.2f}")
            else:
                print("  Baseline: N/A")

            if data.get("best_model"):
                print(f"  Best Model: {data['best_model']}")
                if data.get("best_rmse"):
                    print(f"  Best RMSE: ${data['best_rmse']:.2f}")

            improvement = data.get("improvement_pct", 0)
            print(f"  Improvement: {improvement:.2f}%")

            if improvement >= 10:
                print("  ✅ 10% THRESHOLD MET - Ensemble stacking recommended!")
            else:
                print("  ❌ 10% threshold not met")

        print("\n" + "=" * 70)

        # Overall recommendation
        any_threshold_met = any(
            report.get(cfg["symbol"].lower(), {}).get("threshold_met", False)
            for cfg in self.jobs_config
        )

        if any_threshold_met:
            print("\nRECOMMENDATION: Implement ensemble stacking for improved models")
        else:
            print("\nRECOMMENDATION: Continue with CNN-LSTM baseline")


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

    # Benchmark orchestration command
    p_benchmark = subparsers.add_parser(
        "cloud-benchmark",
        help="Train and benchmark multiple model architectures",
    )
    p_benchmark.add_argument("args", nargs=argparse.REMAINDER)
    p_benchmark.set_defaults(func=_handle_cloud_benchmark)
