import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from src.strategies.components import Strategy
from src.strategies.versioning import StrategyVersionRecord

logger = logging.getLogger(__name__)


# Backwards compatibility alias so existing imports continue to work.
StrategyVersion = StrategyVersionRecord


class StrategyManager:
    """
    Manages strategy hot-swapping, model updates, and version control for live trading.

    Key Features:
    - Hot-swap strategies without stopping trading
    - Safe model updates with rollback capability
    - Version control for strategies and models
    - Graceful position handling during switches
    - Performance comparison between versions
    """

    def __init__(
        self,
        strategies_dir: str = "strategies",
        models_dir: str = "src/ml",
        staging_dir: str | None = None,
    ):
        self.strategies_dir = Path(strategies_dir)
        self.models_dir = Path(models_dir)
        default_staging = Path(
            os.environ.get(
                "ATB_STAGING_DIR", os.path.join(tempfile.gettempdir(), "ai-trading-bot-staging")
            )
        )
        self.staging_dir = Path(staging_dir) if staging_dir else default_staging

        # Create staging directory for safe updates in /tmp
        self.staging_dir.mkdir(exist_ok=True, parents=True)

        # Current active strategy (component-based)
        self.current_strategy: Strategy | None = None
        self.current_version: StrategyVersion | None = None

        # Strategy registry with factory functions
        self.strategy_registry = {}

        # Register strategy factory functions
        try:
            from src.strategies.ml_basic import create_ml_basic_strategy

            self.strategy_registry["ml_basic"] = create_ml_basic_strategy
        except Exception as e:
            logger.debug(f"ML Basic strategy not available: {e}")

        try:
            from src.strategies.ml_adaptive import create_ml_adaptive_strategy

            self.strategy_registry["ml_adaptive"] = create_ml_adaptive_strategy
        except Exception as e:
            logger.debug(f"ML Adaptive strategy not available: {e}")

        try:
            from src.strategies.ensemble_weighted import create_ensemble_weighted_strategy

            self.strategy_registry["ensemble_weighted"] = create_ensemble_weighted_strategy
        except Exception as e:
            logger.debug(f"Ensemble Weighted strategy not available: {e}")

        try:
            from src.strategies.momentum_leverage import create_momentum_leverage_strategy

            self.strategy_registry["momentum_leverage"] = create_momentum_leverage_strategy
        except Exception as e:
            logger.debug(f"Momentum Leverage strategy not available: {e}")

        try:
            from src.strategies.ml_sentiment import create_ml_sentiment_strategy

            self.strategy_registry["ml_sentiment"] = create_ml_sentiment_strategy
        except Exception as e:
            logger.debug(f"ML Sentiment strategy not available: {e}")

        # Version history
        self.version_history: dict[str, StrategyVersion] = {}

        # Threading for safe updates
        self.update_lock = threading.RLock()
        self.pending_update: dict[str, Any] | None = None

        # Callbacks for live trading engine
        self.on_strategy_change: Callable[[dict[str, Any]], None] | None = None
        self.on_model_update: Callable[[dict[str, Any]], None] | None = None

        logger.info("StrategyManager initialized")

    @staticmethod
    def _display_name(strategy: Strategy | None) -> str:
        """Return a human readable name for a strategy instance."""

        if strategy is None:
            return "None"

        return getattr(strategy, "name", strategy.__class__.__name__)

    def _instantiate_strategy(
        self, strategy_name: str, version: str, config: dict[str, Any] | None = None
    ) -> tuple[Strategy, StrategyVersionRecord]:
        """Create a strategy instance and version record without mutating state.

        Uses factory functions to create component-based Strategy instances.
        """

        if strategy_name not in self.strategy_registry:
            raise ValueError(f"Unknown strategy: {strategy_name}")

        factory_function = self.strategy_registry[strategy_name]

        # Call factory function with config
        if config:
            strategy = factory_function(**config)
        else:
            strategy = factory_function()

        version_id = f"{strategy_name}_{version}"
        strategy_version = StrategyVersionRecord(
            version_id=version_id,
            strategy_name=strategy_name,
            version=version,
            created_at=datetime.now(),
            config=config,
        )

        return strategy, strategy_version

    def load_strategy(
        self, strategy_name: str, version: str = "latest", config: dict[str, Any] | None = None
    ) -> Strategy:
        """Load a strategy with version control.

        Uses factory functions to create component-based Strategy instances.
        """

        with self.update_lock:
            try:
                strategy, strategy_version = self._instantiate_strategy(
                    strategy_name, version, config
                )

                self.current_strategy = strategy
                self.current_version = strategy_version
                self.version_history[strategy_version.version_id] = strategy_version

                logger.info(f"Loaded component-based strategy: {strategy_name} v{version}")
                return strategy

            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_name}: {e}")
                raise

    def hot_swap_strategy(
        self,
        new_strategy_name: str,
        new_config: dict[str, Any] | None = None,
        close_existing_positions: bool = False,
    ) -> bool:
        """
        Hot-swap to a new strategy without stopping the trading engine

        Args:
            new_strategy_name: Name of new strategy to load
            new_config: Configuration for new strategy
            close_existing_positions: Whether to close current positions

        Returns:
            True if swap was successful
        """

        with self.update_lock:
            try:
                # Check if there's already a pending update
                if self.pending_update is not None:
                    logger.warning("❌ Hot-swap blocked: There's already a pending update")
                    return False

                logger.info(f"🔄 Starting hot-swap to strategy: {new_strategy_name}")

                # Load new strategy in staging without mutating current state
                old_strategy = self.current_strategy
                new_version = f"{int(time.time())}"

                new_strategy, staged_version = self._instantiate_strategy(
                    new_strategy_name, new_version, new_config
                )

                # Prepare swap data
                swap_data = {
                    "old_strategy": old_strategy,
                    "new_strategy": new_strategy,
                    "new_version": staged_version,
                    "close_positions": close_existing_positions,
                    "timestamp": datetime.now(),
                }

                # Set pending update (trading engine will pick this up)
                self.pending_update = {"type": "strategy_swap", "data": swap_data}

                # Notify trading engine if callback is set
                if self.on_strategy_change:
                    self.on_strategy_change(swap_data)

                logger.info(
                    f"✅ Hot-swap prepared: {self._display_name(old_strategy)} "
                    f"→ {self._display_name(new_strategy)}"
                )
                return True

            except Exception as e:
                logger.error(f"❌ Hot-swap failed: {e}")
                return False

    def update_model(
        self, strategy_name: str, new_model_path: str, validate_model: bool = True
    ) -> bool:
        """
        Safely update ML model for a strategy

        Args:
            strategy_name: Name of strategy using the model
            new_model_path: Path to new model file
            validate_model: Whether to validate model before deployment

        Returns:
            True if update was successful
        """

        with self.update_lock:
            try:
                # Check if there's already a pending update
                if self.pending_update is not None:
                    logger.warning("❌ Model update blocked: There's already a pending update")
                    return False

                logger.info(f"🔄 Updating model for strategy: {strategy_name}")

                # Validate new model exists
                if not os.path.exists(new_model_path):
                    raise FileNotFoundError(f"Model file not found: {new_model_path}")

                # Get current strategy
                if (
                    not self.current_strategy
                    or self.current_strategy.name.lower() != strategy_name.lower()
                ):
                    raise ValueError(
                        f"Current strategy {self.current_strategy.name} doesn't match {strategy_name}"
                    )

                # Validate model if requested
                if validate_model:
                    self._validate_model(new_model_path)

                # Create staging copy
                staging_path = self.staging_dir / f"model_{strategy_name}_{int(time.time())}.onnx"
                shutil.copy2(new_model_path, staging_path)

                # Prepare model update
                update_data = {
                    "strategy_name": strategy_name,
                    "old_model_path": getattr(self.current_strategy, "model_path", None),
                    "new_model_path": str(staging_path),
                    "timestamp": datetime.now(),
                }

                # Set pending update
                self.pending_update = {"type": "model_update", "data": update_data}

                # Notify trading engine
                if self.on_model_update:
                    self.on_model_update(update_data)

                logger.info(f"✅ Model update prepared for {strategy_name}")
                return True

            except Exception as e:
                logger.error(f"❌ Model update failed: {e}")
                return False

    def apply_pending_update(self) -> bool:
        """Apply any pending updates (called by trading engine)"""

        if not self.pending_update:
            return False

        with self.update_lock:
            try:
                update_type = self.pending_update["type"]
                update_data = self.pending_update["data"]

                if update_type == "strategy_swap":
                    return self._apply_strategy_swap(update_data)
                elif update_type == "model_update":
                    return self._apply_model_update(update_data)
                else:
                    logger.error(f"Unknown update type: {update_type}")
                    return False

            except Exception as e:
                logger.error(f"Failed to apply update: {e}")
                return False
            finally:
                self.pending_update = None

    def _apply_strategy_swap(self, swap_data: dict) -> bool:
        """Apply strategy swap"""
        try:
            old_strategy = swap_data["old_strategy"]
            new_strategy = swap_data["new_strategy"]
            new_version = swap_data.get("new_version")

            # Update current strategy
            self.current_strategy = new_strategy

            if isinstance(new_version, StrategyVersionRecord):
                self.current_version = new_version
                self.version_history[new_version.version_id] = new_version

            logger.info(
                f"✅ Strategy swapped: {self._display_name(old_strategy)} "
                f"→ {self._display_name(new_strategy)}"
            )
            return True

        except Exception as e:
            logger.error(f"Strategy swap failed: {e}")
            return False

    def _apply_model_update(self, update_data: dict) -> bool:
        """Apply model update"""
        try:
            strategy_name = update_data["strategy_name"]

            logger.warning(
                f"Strategy {strategy_name} doesn't support model updates in component-based architecture"
            )
            return False

        except Exception as e:
            logger.error(f"Model update failed: {e}")
            return False

    def _validate_model(self, model_path: str) -> bool:
        """Validate model can be loaded and used"""
        try:
            import onnx
            import onnxruntime as ort

            # Load ONNX model
            model = onnx.load(model_path)
            onnx.checker.check_model(model)

            # Test inference session
            session = ort.InferenceSession(model_path)
            input_shape = session.get_inputs()[0].shape

            logger.info(f"Model validation passed: {model_path} (input shape: {input_shape})")
            return True

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return False

    def rollback_to_previous_version(self) -> bool:
        """Rollback to previous strategy version"""
        # Implementation for rollback functionality
        logger.info("Rollback functionality - to be implemented")
        return False

    def get_performance_comparison(self) -> dict[str, Any]:
        """Compare performance between strategy versions"""
        # Implementation for performance comparison
        return {}

    def list_available_strategies(self) -> dict[str, Any]:
        """List all available strategies and their versions"""
        return {
            "available_strategies": list(self.strategy_registry.keys()),
            "current_strategy": self.current_strategy.name if self.current_strategy else None,
            "current_version": self.current_version.version if self.current_version else None,
            "version_history": {
                k: {
                    "name": v.strategy_name,
                    "version": v.version,
                    "timestamp": v.created_at.isoformat(),
                }
                for k, v in self.version_history.items()
            },
        }

    def has_pending_update(self) -> bool:
        """Check if there's a pending update"""
        return self.pending_update is not None

    def get_pending_update_info(self) -> dict[str, Any] | None:
        """Get information about pending update"""
        return self.pending_update
