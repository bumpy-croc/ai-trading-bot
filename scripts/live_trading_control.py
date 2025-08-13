#!/usr/bin/env python3
"""
Live Trading Control Script

This script provides easy control over live trading operations including:
- Strategy hot-swapping
- Model updates
- System monitoring
- Emergency controls
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from scripts.safe_model_trainer import SafeModelTrainer  # noqa: E402


class LiveTradingController:
    """
    Controller for managing live trading operations
    """

    def __init__(self):
        self.trainer = SafeModelTrainer()
        # In a real implementation, you'd connect to the running trading engine
        # via IPC, socket, or shared memory
        self.trading_engine_connected = False

    def train_and_deploy_model(
        self,
        symbol: str = "BTCUSDT",
        with_sentiment: bool = False,
        days: int = 365,
        epochs: int = 50,
        auto_deploy: bool = False,
    ) -> bool:
        """
        Train a new model and optionally deploy it to live trading
        """
        print(f"🚀 Training new model for {symbol}...")
        print(f"   Sentiment: {with_sentiment}")
        print(f"   Training days: {days}")
        print(f"   Epochs: {epochs}")
        print()

        # Step 1: Train model safely
        result = self.trainer.train_model_safe(
            symbol=symbol, with_sentiment=with_sentiment, days=days, epochs=epochs
        )

        if not result["success"]:
            print(f"❌ Model training failed: {result['error']}")
            return False

        print("✅ Model training completed successfully!")
        deployment_package = result["deployment_package"]
        print(f"📁 Model staged at: {deployment_package['staging_path']}")

        # Step 2: Deploy if requested
        if auto_deploy:
            print("\n🚀 Auto-deploying model to live trading...")
            success = self.trainer.deploy_model_to_live(deployment_package)
            if success:
                print("✅ Model deployed successfully!")
                return True
            else:
                print("❌ Model deployment failed!")
                return False
        else:
            print("\n📋 Model ready for deployment. To deploy manually run:")
            print(
                f"   python live_trading_control.py deploy-model --model-path {deployment_package['staging_path']}"
            )
            return True

    def deploy_model(self, model_path: str, close_positions: bool = False) -> bool:
        """
        Deploy a staged model to live trading
        """
        print(f"🚀 Deploying model: {model_path}")

        # Create deployment package
        deployment_package = {
            "staging_path": model_path,
            "strategy_name": "ml_with_sentiment" if "sentiment" in model_path else "ml_basic",
            "ready_for_deployment": True,
        }

        success = self.trainer.deploy_model_to_live(
            deployment_package=deployment_package, close_positions=close_positions
        )

        if success:
            print("✅ Model deployment successful!")
        else:
            print("❌ Model deployment failed!")

        return success

    def swap_strategy(self, new_strategy: str, close_positions: bool = False) -> bool:
        """
        Hot-swap to a different strategy
        """
        print(f"🔄 Swapping to strategy: {new_strategy}")
        print(f"   Close existing positions: {close_positions}")

        # In a real implementation, this would communicate with the live trading engine
        # For now, we'll simulate the operation
        print("⚠️  Strategy swap simulation - not connected to live trading engine")
        print("   In production, this would:")
        print(f"   1. Load new strategy: {new_strategy}")
        if close_positions:
            print("   2. Close all existing positions")
        print("   3. Switch to new strategy on next trading cycle")
        print("   4. Send alerts about the change")

        return True

    def emergency_stop(self) -> bool:
        """
        Emergency stop all trading
        """
        print("🚨 EMERGENCY STOP INITIATED")
        print("   Stopping all trading operations...")
        print("   Closing all positions...")
        print("   Sending emergency alerts...")

        # In production, this would send stop signal to trading engine
        print("⚠️  Emergency stop simulation - not connected to live trading engine")

        return True

    def get_status(self) -> dict:
        """
        Get current trading status
        """
        # In production, this would query the live trading engine
        status = {
            "connected": self.trading_engine_connected,
            "running": False,
            "current_strategy": "Unknown",
            "active_positions": 0,
            "current_balance": 0.0,
            "total_trades": 0,
            "win_rate": 0.0,
        }

        return status

    def list_models(self) -> dict:
        """
        List available models
        """
        return self.trainer.list_available_models()


def main():
    parser = argparse.ArgumentParser(description="Live Trading Control")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train new model")
    train_parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    train_parser.add_argument("--sentiment", action="store_true", help="Include sentiment")
    train_parser.add_argument("--days", type=int, default=365, help="Training days")
    train_parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    train_parser.add_argument(
        "--auto-deploy", action="store_true", help="Auto-deploy after training"
    )

    # Deploy model command
    deploy_parser = subparsers.add_parser("deploy-model", help="Deploy trained model")
    deploy_parser.add_argument("--model-path", required=True, help="Path to model file")
    deploy_parser.add_argument(
        "--close-positions", action="store_true", help="Close positions before deploy"
    )

    # Strategy swap command
    swap_parser = subparsers.add_parser("swap-strategy", help="Hot-swap strategy")
    swap_parser.add_argument("--strategy", required=True, help="New strategy name")
    swap_parser.add_argument(
        "--close-positions", action="store_true", help="Close positions before swap"
    )

    # Status command
    subparsers.add_parser("status", help="Get current status")

    # List models command
    subparsers.add_parser("list-models", help="List available models")

    # Emergency stop command
    subparsers.add_parser("emergency-stop", help="Emergency stop all trading")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    controller = LiveTradingController()

    try:
        if args.command == "train":
            success = controller.train_and_deploy_model(
                symbol=args.symbol,
                with_sentiment=args.sentiment,
                days=args.days,
                epochs=args.epochs,
                auto_deploy=args.auto_deploy,
            )
            sys.exit(0 if success else 1)

        elif args.command == "deploy-model":
            success = controller.deploy_model(
                model_path=args.model_path, close_positions=args.close_positions
            )
            sys.exit(0 if success else 1)

        elif args.command == "swap-strategy":
            success = controller.swap_strategy(
                new_strategy=args.strategy, close_positions=args.close_positions
            )
            sys.exit(0 if success else 1)

        elif args.command == "status":
            status = controller.get_status()
            print(json.dumps(status, indent=2))

        elif args.command == "list-models":
            models = controller.list_models()
            print(json.dumps(models, indent=2))

        elif args.command == "emergency-stop":
            success = controller.emergency_stop()
            sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
