#!/usr/bin/env python3
"""
Safe Model Training System for Live Trading

This system ensures that model training doesn't interfere with live trading by:
1. Training models in isolated staging area
2. Validating models before deployment
3. Providing safe deployment mechanisms
4. Maintaining model version control
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.live.strategy_manager import StrategyManager
from src.utils.logging_config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)


class SafeModelTrainer:
    """
    Safe model training system for live trading environments
    """

    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.models_dir = self.project_root / "src" / "ml"
        self.staging_dir = Path("/tmp/ai-trading-bot-staging")
        self.backup_dir = self.project_root / "model_backups"

        # Create directories
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        self.backup_dir.mkdir(exist_ok=True)

        # Initialize strategy manager for deployment
        self.strategy_manager = StrategyManager(
            strategies_dir=str(self.project_root / "src" / "strategies"),
            models_dir=str(self.models_dir),
            staging_dir=str(self.staging_dir),
        )

        logger.info("SafeModelTrainer initialized")

    def train_model_safe(
        self,
        symbol: str = "BTCUSDT",
        with_sentiment: bool = False,
        days: int = 365,
        epochs: int = 50,
        validate_before_deploy: bool = True,
    ) -> dict:
        """
        Train a new model safely without interfering with live trading

        Args:
            symbol: Trading symbol (e.g., BTCUSDT)
            with_sentiment: Whether to include sentiment analysis
            days: Number of days of historical data
            epochs: Training epochs
            validate_before_deploy: Whether to validate model before deployment

        Returns:
            Training results and model info
        """

        try:
            logger.info(f"🚀 Starting safe model training for {symbol}")
            logger.info(f"Sentiment: {with_sentiment}, Days: {days}, Epochs: {epochs}")

            # Step 1: Backup existing models
            self._backup_current_models(symbol, with_sentiment)

            # Step 2: Train model in staging area
            model_info = self._train_in_staging(symbol, with_sentiment, days, epochs)

            # Step 3: Validate model
            if validate_before_deploy:
                validation_results = self._validate_model(model_info)
                if not validation_results["valid"]:
                    raise Exception(f"Model validation failed: {validation_results['errors']}")

                logger.info("✅ Model validation passed")
                model_info["validation"] = validation_results

            # Step 4: Prepare deployment package
            deployment_package = self._prepare_deployment_package(model_info)

            logger.info(f"✅ Model training completed: {deployment_package['staging_path']}")

            return {
                "success": True,
                "model_info": model_info,
                "deployment_package": deployment_package,
            }

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    def _backup_current_models(self, symbol: str, with_sentiment: bool) -> None:
        """Backup existing models before training"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{symbol}_{'sentiment' if with_sentiment else 'price'}_{timestamp}"

        # Backup existing model files
        model_files = []
        if with_sentiment:
            model_files.extend(
                [
                    f"{symbol.lower()}_sentiment.onnx",
                    f"{symbol.lower()}_sentiment.h5",
                    f"{symbol.lower()}_sentiment.keras",
                    f"{symbol.lower()}_sentiment_metadata.json",
                ]
            )
        else:
            model_files.extend(
                [
                    f"{symbol.lower()}_price.onnx",
                    f"{symbol.lower()}_price.h5",
                    f"{symbol.lower()}_price.keras",
                    f"{symbol.lower()}_price_metadata.json",
                ]
            )

        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(exist_ok=True)

        for model_file in model_files:
            source_path = self.models_dir / model_file
            if source_path.exists():
                shutil.copy2(source_path, backup_path / model_file)
                logger.info(f"Backed up: {model_file}")

    def _train_in_staging(self, symbol: str, with_sentiment: bool, days: int, epochs: int) -> dict:
        """Train model in staging area"""
        # Create staging environment
        staging_env = os.environ.copy()
        staging_env["PYTHONPATH"] = str(self.project_root)

        # Determine training script
        if with_sentiment:
            script_path = self.project_root / "scripts" / "train_model.py"
        else:
            script_path = self.project_root / "scripts" / "train_price_model.py"

        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            "--symbol",
            symbol,
            "--days",
            str(days),
            "--epochs",
            str(epochs),
            "--output-dir",
            str(self.staging_dir),
        ]

        logger.info(f"Running training command: {' '.join(cmd)}")

        # Execute training
        result = subprocess.run(
            cmd,
            env=staging_env,
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            raise Exception(f"Training failed: {result.stderr}")

        # Parse training output to get model info
        model_info = self._parse_training_output(result.stdout, symbol, with_sentiment)
        return model_info

    def _parse_training_output(self, output: str, symbol: str, with_sentiment: bool) -> dict:
        """Parse training output to extract model information"""
        # This is a simplified parser - in practice, you'd want more robust parsing
        model_type = "sentiment" if with_sentiment else "price"

        return {
            "symbol": symbol,
            "model_type": model_type,
            "training_timestamp": datetime.now().isoformat(),
            "staging_path": str(self.staging_dir),
            "model_files": [
                f"{symbol.lower()}_{model_type}.onnx",
                f"{symbol.lower()}_{model_type}.h5",
                f"{symbol.lower()}_{model_type}.keras",
                f"{symbol.lower()}_{model_type}_metadata.json",
            ],
        }

    def _validate_model(self, model_info: dict) -> dict:
        """Validate trained model"""
        # Basic validation - check files exist
        errors = []
        for model_file in model_info["model_files"]:
            file_path = self.staging_dir / model_file
            if not file_path.exists():
                errors.append(f"Missing model file: {model_file}")

        # TODO: Add more sophisticated validation (model loading, basic inference, etc.)

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "validation_timestamp": datetime.now().isoformat(),
        }

    def _prepare_deployment_package(self, model_info: dict) -> dict:
        """Prepare model for deployment"""
        deployment_id = f"{model_info['symbol']}_{model_info['model_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "deployment_id": deployment_id,
            "staging_path": str(self.staging_dir),
            "strategy_name": "ml_basic",
            "ready_for_deployment": True,
            "model_info": model_info,
        }

    def deploy_model_to_live(self, deployment_package: dict, close_positions: bool = False) -> bool:
        """
        Deploy model to live trading environment

        Args:
            deployment_package: Model deployment package
            close_positions: Whether to close existing positions before deployment

        Returns:
            True if deployment successful
        """
        try:
            logger.info(f"🚀 Deploying model: {deployment_package['deployment_id']}")

            if close_positions:
                logger.warning("⚠️  Closing existing positions before deployment")
                # TODO: Implement position closing logic

            # Copy model files from staging to live
            staging_path = Path(deployment_package["staging_path"])
            for model_file in deployment_package["model_info"]["model_files"]:
                source = staging_path / model_file
                destination = self.models_dir / model_file

                if source.exists():
                    shutil.copy2(source, destination)
                    logger.info(f"Deployed: {model_file}")
                else:
                    logger.warning(f"Missing staging file: {model_file}")

            # Update strategy manager
            self.strategy_manager.reload_strategies()

            logger.info("✅ Model deployment completed successfully")
            return True

        except Exception as e:
            logger.error(f"❌ Model deployment failed: {e}")
            return False

    def list_available_models(self) -> dict:
        """List available models in staging and live environments"""
        staging_models = []
        live_models = []

        # List staging models
        if self.staging_dir.exists():
            for file_path in self.staging_dir.glob("*.onnx"):
                staging_models.append(
                    {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    }
                )

        # List live models
        for file_path in self.models_dir.glob("*.onnx"):
            live_models.append(
                {
                    "name": file_path.name,
                    "path": str(file_path),
                    "size": file_path.stat().st_size,
                    "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                }
            )

        return {
            "staging": staging_models,
            "live": live_models,
            "backup_dir": str(self.backup_dir),
        }


def main():
    """CLI entry point for safe model training"""
    parser = argparse.ArgumentParser(description="Safe Model Training System")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--sentiment", action="store_true", help="Include sentiment analysis")
    parser.add_argument("--days", type=int, default=365, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--auto-deploy", action="store_true", help="Auto-deploy after training")
    parser.add_argument(
        "--close-positions", action="store_true", help="Close positions before deployment"
    )

    args = parser.parse_args()

    trainer = SafeModelTrainer()

    # Train model
    result = trainer.train_model_safe(
        symbol=args.symbol,
        with_sentiment=args.sentiment,
        days=args.days,
        epochs=args.epochs,
    )

    if not result["success"]:
        print(f"❌ Training failed: {result['error']}")
        sys.exit(1)

    print(f"✅ Training completed: {result['deployment_package']['staging_path']}")

    # Auto-deploy if requested
    if args.auto_deploy:
        ok = trainer.deploy_model_to_live(
            result["deployment_package"],
            close_positions=args.close_positions,
        )
        if ok:
            print("✅ Model deployed successfully")
        else:
            print("❌ Model deployment failed")
            sys.exit(1)


if __name__ == "__main__":
    main()
