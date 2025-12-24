"""SageMaker training entrypoint.

This module is invoked by SageMaker Training Jobs to run the model training pipeline.
It reads hyperparameters from SageMaker's standard locations and outputs artifacts
to the expected paths.

SageMaker Paths:
- /opt/ml/input/config/hyperparameters.json - Training parameters
- /opt/ml/model/ - Output directory for trained model
- /opt/ml/output/ - Output directory for failure artifacts
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Set up logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# SageMaker standard paths
HYPERPARAMS_PATH = Path("/opt/ml/input/config/hyperparameters.json")
MODEL_OUTPUT_PATH = Path("/opt/ml/model")
FAILURE_OUTPUT_PATH = Path("/opt/ml/output/failure")


def load_hyperparameters() -> dict:
    """Load hyperparameters from SageMaker config.

    SageMaker passes hyperparameters as strings in a JSON file.

    Returns:
        Dictionary of hyperparameters with types converted
    """
    if not HYPERPARAMS_PATH.exists():
        logger.warning("No hyperparameters file found, using defaults")
        return {}

    with open(HYPERPARAMS_PATH) as f:
        params = json.load(f)

    logger.info(f"Loaded hyperparameters: {params}")
    return params


def parse_hyperparameters(params: dict) -> dict:
    """Parse and validate hyperparameters.

    Converts string values to appropriate types.

    Args:
        params: Raw hyperparameters (all strings)

    Returns:
        Parsed hyperparameters with correct types
    """
    return {
        "symbol": params.get("symbol", "BTCUSDT"),
        "timeframe": params.get("timeframe", "1h"),
        "start_date": params.get("start_date"),
        "end_date": params.get("end_date"),
        "epochs": int(params.get("epochs", "300")),
        "batch_size": int(params.get("batch_size", "32")),
        "sequence_length": int(params.get("sequence_length", "120")),
        "force_sentiment": params.get("force_sentiment", "false").lower() == "true",
        "force_price_only": params.get("force_price_only", "false").lower() == "true",
        "mixed_precision": params.get("mixed_precision", "true").lower() == "true",
    }


def run_training(parsed_params: dict) -> int:
    """Run the training pipeline.

    Args:
        parsed_params: Parsed hyperparameters

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        # Import training modules (may not be available in local testing)
        from src.ml.training_pipeline.config import (
            DiagnosticsOptions,
            TrainingConfig,
            TrainingContext,
            TrainingPaths,
        )
        from src.ml.training_pipeline.pipeline import run_training_pipeline

        # Parse dates
        start_date = datetime.fromisoformat(parsed_params["start_date"])
        end_date = datetime.fromisoformat(parsed_params["end_date"])

        # Create training config
        config = TrainingConfig(
            symbol=parsed_params["symbol"],
            timeframe=parsed_params["timeframe"],
            start_date=start_date,
            end_date=end_date,
            epochs=parsed_params["epochs"],
            batch_size=parsed_params["batch_size"],
            sequence_length=parsed_params["sequence_length"],
            force_sentiment=parsed_params["force_sentiment"],
            force_price_only=parsed_params["force_price_only"],
            mixed_precision=parsed_params["mixed_precision"],
            diagnostics=DiagnosticsOptions(
                generate_plots=False,  # No display in container
                evaluate_robustness=True,
                convert_to_onnx=True,
            ),
        )

        # Override paths for SageMaker
        paths = TrainingPaths(
            project_root=Path("/opt/ml/code"),
            data_dir=Path("/opt/ml/input/data"),
            models_dir=MODEL_OUTPUT_PATH,
        )

        ctx = TrainingContext(config=config, paths=paths)

        logger.info(f"Starting training for {config.symbol}")
        logger.info(f"Data range: {start_date} to {end_date}")
        logger.info(f"Epochs: {config.epochs}, Batch size: {config.batch_size}")

        # Run training
        result = run_training_pipeline(ctx)

        if result.success:
            logger.info(f"Training completed successfully in {result.duration_seconds:.1f}s")
            logger.info(f"Artifacts saved to {result.artifact_paths}")
            return 0
        else:
            logger.error(f"Training failed: {result.metadata.get('error')}")
            write_failure_file(result.metadata.get("error", "Unknown error"))
            return 1

    except Exception as exc:
        logger.exception("Training failed with exception")
        write_failure_file(str(exc))
        return 1


def write_failure_file(error_message: str) -> None:
    """Write failure information for SageMaker.

    SageMaker reads /opt/ml/output/failure for error details.

    Args:
        error_message: Error message to write
    """
    FAILURE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    failure_file = FAILURE_OUTPUT_PATH / "failure"
    failure_file.write_text(error_message)
    logger.info(f"Wrote failure file: {failure_file}")


def main() -> int:
    """Main entrypoint for SageMaker training.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    logger.info("SageMaker training entrypoint starting")

    # Ensure output directory exists
    MODEL_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Load and parse hyperparameters
    raw_params = load_hyperparameters()
    parsed_params = parse_hyperparameters(raw_params)

    # Validate required parameters
    if not parsed_params["start_date"] or not parsed_params["end_date"]:
        error_msg = "start_date and end_date hyperparameters are required"
        logger.error(error_msg)
        write_failure_file(error_msg)
        return 1

    # Run training
    return run_training(parsed_params)


if __name__ == "__main__":
    sys.exit(main())
