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
import json
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
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / "ml"
        self.staging_dir = Path("/tmp/ai-trading-bot-staging")
        self.backup_dir = self.project_root / "model_backups"

        # Create directories
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        self.backup_dir.mkdir(exist_ok=True)

        # Initialize strategy manager for deployment
        self.strategy_manager = StrategyManager(
            strategies_dir=str(self.project_root / "strategies"),
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
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"❌ Model training failed: {e}")
            return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

    def deploy_model_to_live(self, deployment_package: dict, close_positions: bool = False) -> bool:
        """
        Deploy a trained model to live trading

        Args:
            deployment_package: Package from train_model_safe
            close_positions: Whether to close existing positions before switching

        Returns:
            True if deployment was successful
        """

        try:
            logger.info("🚀 Deploying model to live trading")

            model_path = deployment_package["staging_path"]
            strategy_name = deployment_package["strategy_name"]

            # Use strategy manager to update model
            success = self.strategy_manager.update_model(
                strategy_name=strategy_name, new_model_path=model_path, validate_model=True
            )

            if success:
                logger.info("✅ Model deployed successfully")

                # Log deployment
                self._log_deployment(deployment_package)

                return True
            else:
                logger.error("❌ Model deployment failed")
                return False

        except Exception as e:
            logger.error(f"❌ Model deployment error: {e}")
            return False

    def _backup_current_models(self, symbol: str, with_sentiment: bool):
        """Backup current production models"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine model filename
            if with_sentiment:
                model_filename = f"model_{symbol.lower()}_sentiment.onnx"
                metadata_filename = f"model_{symbol.lower()}_sentiment_metadata.json"
            else:
                model_filename = f"model_{symbol.lower()}.onnx"
                metadata_filename = f"model_{symbol.lower()}_metadata.json"

            # Backup model file
            current_model = self.models_dir / model_filename
            if current_model.exists():
                backup_model = self.backup_dir / f"{timestamp}_{model_filename}"
                shutil.copy2(current_model, backup_model)
                logger.info(f"Backed up model: {backup_model}")

            # Backup metadata file
            current_metadata = self.models_dir / metadata_filename
            if current_metadata.exists():
                backup_metadata = self.backup_dir / f"{timestamp}_{metadata_filename}"
                shutil.copy2(current_metadata, backup_metadata)
                logger.info(f"Backed up metadata: {backup_metadata}")

        except Exception as e:
            logger.warning(f"Backup failed (non-critical): {e}")

    def _train_in_staging(self, symbol: str, with_sentiment: bool, days: int, epochs: int) -> dict:
        """Train model in staging area using the new train_model.py script"""

        # Create staging timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if with_sentiment:
            # Sentiment variant is no longer a separate strategy; we still allow training a sentiment model
            strategy_name = "ml_basic"
            model_type = "sentiment"
        else:
            strategy_name = "ml_basic"
            model_type = "price"

        try:
            logger.info(f"📊 Training {model_type} model for {symbol}")

            # Calculate date range (days back from today)
            from datetime import timedelta

            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days)

            # Prepare training command
            cmd = [
                "python",
                "scripts/train_model.py",
                symbol,
                "--start-date",
                start_date.strftime("%Y-%m-%d"),
                "--end-date",
                end_date.strftime("%Y-%m-%d"),
            ]

            # Add sentiment flag if needed
            if with_sentiment:
                cmd.append("--force-sentiment")
            else:
                cmd.append("--force-price-only")

            # Run training in project root directory
            logger.info(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                raise Exception(f"Training failed: {result.stderr}")

            logger.info("✅ Training completed successfully")

            # Find the generated model files
            model_pattern = f"{symbol.lower()}_{model_type}"
            onnx_model = self.project_root / "ml" / f"{model_pattern}.onnx"
            metadata_file = self.project_root / "ml" / f"{model_pattern}_metadata.json"

            # Move files to staging with timestamp
            staging_model = self.staging_dir / f"{model_pattern}_{timestamp}.onnx"
            staging_metadata = self.staging_dir / f"{model_pattern}_{timestamp}_metadata.json"

            if onnx_model.exists():
                shutil.copy2(onnx_model, staging_model)
                logger.info(f"📁 Moved ONNX model to staging: {staging_model}")
            else:
                raise Exception(f"ONNX model not found: {onnx_model}")

            if metadata_file.exists():
                shutil.copy2(metadata_file, staging_metadata)
                logger.info(f"📁 Moved metadata to staging: {staging_metadata}")

            # Create compatible metadata for safe trainer
            metadata = {
                "symbol": symbol,
                "with_sentiment": with_sentiment,
                "training_days": days,
                "epochs": epochs,
                "timestamp": timestamp,
                "strategy_name": strategy_name,
                "model_type": model_type,
                "training_output": result.stdout,
            }

            # Save metadata to staging
            with open(staging_metadata, "w") as f:
                json.dump(metadata, f, indent=2, default=str)

            return {
                "symbol": symbol,
                "with_sentiment": with_sentiment,
                "staging_model_path": str(staging_model),
                "staging_metadata_path": str(staging_metadata),
                "strategy_name": strategy_name,
                "timestamp": timestamp,
                "training_output": result.stdout,
                "model_type": model_type,
            }

        except subprocess.TimeoutExpired as e:
            raise Exception("Training timed out after 1 hour") from e
        except Exception as e:  # noqa: BLE001
            logger.error(f"❌ Training failed: {e}")
            raise

    def _validate_model(self, model_info: dict) -> dict:
        """Validate trained model"""

        validation_results = {"valid": True, "errors": [], "warnings": [], "metrics": {}}

        try:
            model_path = model_info["staging_model_path"]

            # Test 1: Can load ONNX model
            try:
                import onnx
                import onnxruntime as ort

                model = onnx.load(model_path)
                onnx.checker.check_model(model)

                session = ort.InferenceSession(model_path)
                input_shape = session.get_inputs()[0].shape

                validation_results["metrics"]["input_shape"] = input_shape

            except Exception as e:
                validation_results["valid"] = False
                validation_results["errors"].append(f"ONNX loading failed: {e}")

            # Test 2: Model file size check
            model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            validation_results["metrics"]["model_size_mb"] = model_size_mb

            if model_size_mb > 100:  # Warn if model is very large
                validation_results["warnings"].append(f"Large model size: {model_size_mb:.1f}MB")

            # Test 3: Check training results
            if model_info.get("training_results"):
                results = model_info["training_results"]
                if isinstance(results, dict):
                    if "val_loss" in results:
                        val_loss = results["val_loss"]
                        if val_loss > 0.1:  # Arbitrary threshold
                            validation_results["warnings"].append(
                                f"High validation loss: {val_loss}"
                            )
                        validation_results["metrics"]["validation_loss"] = val_loss

            # Test 4: Metadata validation
            metadata_path = model_info.get("staging_metadata_path")
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    metadata = json.load(f)
                    validation_results["metrics"]["metadata"] = metadata
            else:
                validation_results["warnings"].append("No metadata file found")

            logger.info(
                f"Model validation completed: {len(validation_results['errors'])} errors, {len(validation_results['warnings'])} warnings"
            )

        except Exception as e:
            validation_results["valid"] = False
            validation_results["errors"].append(f"Validation error: {e}")

        return validation_results

    def _prepare_deployment_package(self, model_info: dict) -> dict:
        """Prepare model for deployment"""

        return {
            "staging_path": model_info["staging_model_path"],
            "metadata_path": model_info.get("staging_metadata_path"),
            "strategy_name": model_info["strategy_name"],
            "symbol": model_info["symbol"],
            "timestamp": model_info["timestamp"],
            "ready_for_deployment": True,
        }

    def _log_deployment(self, deployment_package: dict):
        """Log model deployment"""

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": "model_deployment",
            "model_path": deployment_package["staging_path"],
            "strategy": deployment_package["strategy_name"],
            "symbol": deployment_package["symbol"],
        }

        # Write to deployment log
        log_file = self.project_root / "deployment_log.json"

        if log_file.exists():
            with open(log_file) as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append(log_entry)

        with open(log_file, "w") as f:
            json.dump(logs, f, indent=2)

        logger.info(f"Deployment logged: {log_file}")

    def list_available_models(self) -> dict:
        """List available models in staging area"""

        staging_models = []
        for model_file in self.staging_dir.glob("*.onnx"):
            metadata_file = model_file.with_suffix("") / "_metadata.json"

            model_info = {
                "name": model_file.name,
                "path": str(model_file),
                "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            }

            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                        model_info["metadata"] = metadata
                except Exception:
                    pass

            staging_models.append(model_info)

        return {
            "staging_models": staging_models,
            "production_models": self._list_production_models(),
        }

    def _list_production_models(self) -> list:
        """List current production models"""
        production_models = []
        for model_file in self.models_dir.glob("*.onnx"):
            production_models.append(
                {
                    "name": model_file.name,
                    "path": str(model_file),
                    "size_mb": round(model_file.stat().st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
                }
            )
        return production_models


def main():
    parser = argparse.ArgumentParser(description="Safe Model Training for Live Trading")
    parser.add_argument("action", choices=["train", "deploy", "list"], help="Action to perform")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--sentiment", action="store_true", help="Include sentiment analysis")
    parser.add_argument("--days", type=int, default=365, help="Training days")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--model-path", help="Path to model for deployment")
    parser.add_argument(
        "--close-positions", action="store_true", help="Close positions before deployment"
    )

    args = parser.parse_args()

    trainer = SafeModelTrainer()

    if args.action == "train":
        result = trainer.train_model_safe(
            symbol=args.symbol, with_sentiment=args.sentiment, days=args.days, epochs=args.epochs
        )

        print(json.dumps(result, indent=2))

        if result["success"]:
            print("\n✅ Model trained successfully!")
            print(f"📁 Staged at: {result['deployment_package']['staging_path']}")
            print(
                f"🚀 To deploy: python scripts/safe_model_trainer.py deploy --model-path {result['deployment_package']['staging_path']}"
            )

    elif args.action == "deploy":
        if not args.model_path:
            print("❌ Model path required for deployment")
            sys.exit(1)

        # Create deployment package from model path
        deployment_package = {
            "staging_path": args.model_path,
            "strategy_name": "ml_basic",
            "ready_for_deployment": True,
        }

        success = trainer.deploy_model_to_live(
            deployment_package=deployment_package, close_positions=args.close_positions
        )

        if success:
            print("✅ Model deployed successfully!")
        else:
            print("❌ Model deployment failed!")

    elif args.action == "list":
        models = trainer.list_available_models()
        print(json.dumps(models, indent=2))


if __name__ == "__main__":
    main()
