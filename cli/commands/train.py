from __future__ import annotations

import argparse

from cli.commands.train_commands import (
    simple_model_validator_main,
    train_model_main,
    train_price_model_main,
    train_price_only_model_main,
)
from cli.core.forward import forward_to_module_main


def _handle_safe(ns: argparse.Namespace) -> int:
    return forward_to_module_main("src.ml.safe_model_trainer", ns.args or [])


def _handle_model(ns: argparse.Namespace) -> int:
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
    # Parse arguments for the simple validator command
    parser = argparse.ArgumentParser(description="Simple model validator")
    parser.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    
    # Parse the arguments from ns.args
    args = parser.parse_args(ns.args or [])
    return simple_model_validator_main(args)


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
