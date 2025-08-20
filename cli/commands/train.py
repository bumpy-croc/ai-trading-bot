from __future__ import annotations

import argparse

from cli.core.forward import forward_to_module_main


def _handle_safe(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.safe_model_trainer", ns.args or [])


def _handle_model(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.train_model", ns.args or [])


def _handle_price(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.train_price_model", ns.args or [])


def _handle_price_only(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.train_price_only_model", ns.args or [])


def _handle_simple_validator(ns: argparse.Namespace) -> int:
    return forward_to_module_main("scripts.simple_model_validator", ns.args or [])


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
