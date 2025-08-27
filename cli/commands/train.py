from __future__ import annotations

import argparse
import os
import sys

# * Conditionally import training commands only in non-Railway environments
def is_railway_environment() -> bool:
    """
    Check if running in any Railway environment (development, staging, or production).
    
    Returns:
        True if running in any Railway environment
    """
    try:
        # Use the existing Railway provider for robust environment detection
        from src.config.providers.railway_provider import RailwayProvider
        
        railway_provider = RailwayProvider()
        
        # If Railway provider is available, we're in a Railway environment
        return railway_provider.is_available()
        
    except ImportError:
        # If Railway provider is not available, fall back to environment variables
        # This should rarely happen but provides a safety net
        railway_indicators = [
            "RAILWAY_DEPLOYMENT_ID",
            "RAILWAY_PROJECT_ID", 
            "RAILWAY_SERVICE_ID",
            "RAILWAY_ENVIRONMENT_ID",
        ]
        
        return any(key in os.environ for key in railway_indicators)


# * Only import training commands if not in Railway environment
if not is_railway_environment():
    try:
        from cli.commands.train_commands import (
            train_model_main,
            train_price_model_main,
            train_price_only_model_main,
            simple_model_validator_main,
        )
        from cli.core.forward import forward_to_module_main
        _TRAINING_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: Training commands not available: {e}")
        _TRAINING_AVAILABLE = False
else:
    _TRAINING_AVAILABLE = False


def _handle_safe(ns: argparse.Namespace) -> int:
    if not _TRAINING_AVAILABLE:
        return _handle_railway_error("safe model trainer")
    return forward_to_module_main("src.ml.safe_model_trainer", ns.args or [])


def _handle_model(ns: argparse.Namespace) -> int:
    if not _TRAINING_AVAILABLE:
        return _handle_railway_error("combined model training")
    
    # Parse arguments for the training command
    parser = argparse.ArgumentParser(description="Train combined model")
    parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--start-date", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    parser.add_argument("--force-sentiment", action="store_true", help="Force sentiment inclusion")
    parser.add_argument("--force-price-only", action="store_true", help="Train price-only model")
    
    # Parse the arguments from ns.args
    args = parser.parse_args(ns.args or [])
    return train_model_main(args)


def _handle_price(ns: argparse.Namespace) -> int:
    if not _TRAINING_AVAILABLE:
        return _handle_railway_error("price model training")
    
    # Parse arguments for the price model training command
    parser = argparse.ArgumentParser(description="Train price model")
    parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--start-date", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    
    # Parse the arguments from ns.args
    args = parser.parse_args(ns.args or [])
    return train_price_model_main(args)


def _handle_price_only(ns: argparse.Namespace) -> int:
    if not _TRAINING_AVAILABLE:
        return _handle_railway_error("price-only model training")
    
    # Parse arguments for the price-only model training command
    parser = argparse.ArgumentParser(description="Train price-only model")
    parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    parser.add_argument("--start-date", type=str, default="2019-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default="2024-12-01", help="End date (YYYY-MM-DD)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Timeframe")
    
    # Parse the arguments from ns.args
    args = parser.parse_args(ns.args or [])
    return train_price_only_model_main(args)


def _handle_simple_validator(ns: argparse.Namespace) -> int:
    if not _TRAINING_AVAILABLE:
        return _handle_railway_error("model validation")
    
    # Parse arguments for the simple validator command
    parser = argparse.ArgumentParser(description="Simple model validator")
    parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    
    # Parse the arguments from ns.args
    args = parser.parse_args(ns.args or [])
    return simple_model_validator_main(args)


def _handle_railway_error(command_name: str) -> int:
    """Handle training commands in Railway environment"""
    print(f"Error: {command_name} is not available in Railway environments.")
    print("Training requires heavy dependencies (matplotlib, tensorflow) that are excluded from Railway builds.")
    print("Use a local development environment for model training.")
    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("train", help="Model training and validation")
    sub = p.add_subparsers(dest="train_cmd", required=True)

    p_safe = sub.add_parser("safe", help="Safe model trainer")
    p_safe.add_argument("args", nargs=argparse.REMAINDER)
    p_safe.set_defaults(func=_handle_safe)

    p_model = sub.add_parser("model", help="Train combined model")
    p_model.add_argument("args", nargs=argparse.REMAINDER)
    p_model.set_defaults(func=_handle_model)

    p_price = sub.add_parser("price", help="Train price model")
    p_price.add_argument("args", nargs=argparse.REMAINDER)
    p_price.set_defaults(func=_handle_price)

    p_price_only = sub.add_parser("price-only", help="Train price-only model")
    p_price_only.add_argument("args", nargs=argparse.REMAINDER)
    p_price_only.set_defaults(func=_handle_price_only)

    p_validate = sub.add_parser("validate", help="Simple model validator")
    p_validate.add_argument("args", nargs=argparse.REMAINDER)
    p_validate.set_defaults(func=_handle_simple_validator)
