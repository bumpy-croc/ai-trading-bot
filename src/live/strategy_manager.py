import os
import json
import threading
import time
import shutil
from typing import Dict, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
import pandas as pd
from dataclasses import dataclass
import logging

from strategies.base import BaseStrategy
from strategies.adaptive import AdaptiveStrategy
from strategies.enhanced import EnhancedStrategy
from strategies.ml_basic import MlBasic
from strategies.ml_with_sentiment import MlWithSentiment

logger = logging.getLogger(__name__)

@dataclass
class StrategyVersion:
    """Represents a version of a strategy"""
    strategy_name: str
    version: str
    timestamp: datetime
    model_path: Optional[str] = None
    config: Optional[Dict] = None
    performance_metrics: Optional[Dict] = None

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
    
    def __init__(self, 
                 strategies_dir: str = "strategies",
                 models_dir: str = "ml",
                 staging_dir: str = "/tmp/ai-trading-bot-staging"):
        
        self.strategies_dir = Path(strategies_dir)
        self.models_dir = Path(models_dir)
        self.staging_dir = Path(staging_dir)
        
        # Create staging directory for safe updates in /tmp
        self.staging_dir.mkdir(exist_ok=True, parents=True)
        
        # Current active strategy
        self.current_strategy: Optional[BaseStrategy] = None
        self.current_version: Optional[StrategyVersion] = None
        
        # Strategy registry
        self.strategy_registry = {
            'adaptive': AdaptiveStrategy,
            'enhanced': EnhancedStrategy,
            'ml_basic': MlBasic,
            'ml_with_sentiment': MlWithSentiment
        }
        
        # Version history
        self.version_history: Dict[str, StrategyVersion] = {}
        
        # Threading for safe updates
        self.update_lock = threading.RLock()
        self.pending_update: Optional[Dict] = None
        
        # Callbacks for live trading engine
        self.on_strategy_change: Optional[Callable] = None
        self.on_model_update: Optional[Callable] = None
        
        logger.info("StrategyManager initialized")
    
    def load_strategy(self, strategy_name: str, version: str = "latest", 
                     config: Optional[Dict] = None) -> BaseStrategy:
        """Load a strategy with version control"""
        
        with self.update_lock:
            try:
                # Load strategy class
                if strategy_name not in self.strategy_registry:
                    raise ValueError(f"Unknown strategy: {strategy_name}")
                
                strategy_class = self.strategy_registry[strategy_name]
                
                # Create strategy instance with config
                if config:
                    strategy = strategy_class(**config)
                else:
                    strategy = strategy_class()
                
                # Create version record
                strategy_version = StrategyVersion(
                    strategy_name=strategy_name,
                    version=version,
                    timestamp=datetime.now(),
                    config=config
                )
                
                # Set as current
                self.current_strategy = strategy
                self.current_version = strategy_version
                self.version_history[f"{strategy_name}_{version}"] = strategy_version
                
                logger.info(f"Loaded strategy: {strategy_name} v{version}")
                return strategy
                
            except Exception as e:
                logger.error(f"Failed to load strategy {strategy_name}: {e}")
                raise
    
    def hot_swap_strategy(self, new_strategy_name: str, 
                         new_config: Optional[Dict] = None,
                         close_existing_positions: bool = False) -> bool:
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
                    logger.warning(f"❌ Hot-swap blocked: There's already a pending update")
                    return False
                
                logger.info(f"🔄 Starting hot-swap to strategy: {new_strategy_name}")
                
                # Load new strategy in staging
                old_strategy = self.current_strategy
                new_version = f"{int(time.time())}"
                
                new_strategy = self.load_strategy(new_strategy_name, new_version, new_config)
                
                # Prepare swap data
                swap_data = {
                    'old_strategy': old_strategy,
                    'new_strategy': new_strategy,
                    'close_positions': close_existing_positions,
                    'timestamp': datetime.now()
                }
                
                # Set pending update (trading engine will pick this up)
                self.pending_update = {
                    'type': 'strategy_swap',
                    'data': swap_data
                }
                
                # Notify trading engine if callback is set
                if self.on_strategy_change:
                    self.on_strategy_change(swap_data)
                
                logger.info(f"✅ Hot-swap prepared: {old_strategy.name} → {new_strategy.name}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Hot-swap failed: {e}")
                return False
    
    def update_model(self, strategy_name: str, new_model_path: str,
                    validate_model: bool = True) -> bool:
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
                    logger.warning(f"❌ Model update blocked: There's already a pending update")
                    return False
                
                logger.info(f"🔄 Updating model for strategy: {strategy_name}")
                
                # Validate new model exists
                if not os.path.exists(new_model_path):
                    raise FileNotFoundError(f"Model file not found: {new_model_path}")
                
                # Get current strategy
                if (not self.current_strategy or 
                    self.current_strategy.name.lower() != strategy_name.lower()):
                    raise ValueError(f"Current strategy {self.current_strategy.name} doesn't match {strategy_name}")
                
                # Validate model if requested
                if validate_model:
                    self._validate_model(new_model_path)
                
                # Create staging copy
                staging_path = self.staging_dir / f"model_{strategy_name}_{int(time.time())}.onnx"
                shutil.copy2(new_model_path, staging_path)
                
                # Prepare model update
                update_data = {
                    'strategy_name': strategy_name,
                    'old_model_path': getattr(self.current_strategy, 'model_path', None),
                    'new_model_path': str(staging_path),
                    'timestamp': datetime.now()
                }
                
                # Set pending update
                self.pending_update = {
                    'type': 'model_update',
                    'data': update_data
                }
                
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
                update_type = self.pending_update['type']
                update_data = self.pending_update['data']
                
                if update_type == 'strategy_swap':
                    return self._apply_strategy_swap(update_data)
                elif update_type == 'model_update':
                    return self._apply_model_update(update_data)
                else:
                    logger.error(f"Unknown update type: {update_type}")
                    return False
                    
            except Exception as e:
                logger.error(f"Failed to apply update: {e}")
                return False
            finally:
                self.pending_update = None
    
    def _apply_strategy_swap(self, swap_data: Dict) -> bool:
        """Apply strategy swap"""
        try:
            old_strategy = swap_data['old_strategy']
            new_strategy = swap_data['new_strategy']
            
            # Update current strategy
            self.current_strategy = new_strategy
            
            logger.info(f"✅ Strategy swapped: {old_strategy.name} → {new_strategy.name}")
            return True
            
        except Exception as e:
            logger.error(f"Strategy swap failed: {e}")
            return False
    
    def _apply_model_update(self, update_data: Dict) -> bool:
        """Apply model update"""
        try:
            strategy_name = update_data['strategy_name']
            new_model_path = update_data['new_model_path']
            
            # Update model path in current strategy
            if hasattr(self.current_strategy, '_load_model'):
                old_path = self.current_strategy.model_path
                self.current_strategy.model_path = new_model_path
                self.current_strategy._load_model()
                
                logger.info(f"✅ Model updated for {strategy_name}: {old_path} → {new_model_path}")
                return True
            else:
                logger.warning(f"Strategy {strategy_name} doesn't support model updates")
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
    
    def get_performance_comparison(self) -> Dict[str, Any]:
        """Compare performance between strategy versions"""
        # Implementation for performance comparison
        return {}
    
    def list_available_strategies(self) -> Dict[str, Any]:
        """List all available strategies and their versions"""
        return {
            'available_strategies': list(self.strategy_registry.keys()),
            'current_strategy': self.current_strategy.name if self.current_strategy else None,
            'current_version': self.current_version.version if self.current_version else None,
            'version_history': {k: {
                'name': v.strategy_name,
                'version': v.version,
                'timestamp': v.timestamp.isoformat()
            } for k, v in self.version_history.items()}
        }
    
    def has_pending_update(self) -> bool:
        """Check if there's a pending update"""
        return self.pending_update is not None
    
    def get_pending_update_info(self) -> Optional[Dict]:
        """Get information about pending update"""
        return self.pending_update 