from __future__ import annotations

import argparse

from cli.core.forward import forward_to_module_main


def _handle(ns: argparse.Namespace) -> int:
    tail = ns.args or []
    return forward_to_module_main("scripts.run_live_trading", tail)


def _control(ns: argparse.Namespace) -> int:
    # Minimal inline controller using SafeModelTrainer
    from scripts.safe_model_trainer import SafeModelTrainer
    import json

    trainer = SafeModelTrainer()

    if ns.control_cmd == "train":
        result = trainer.train_model_safe(symbol=ns.symbol, with_sentiment=ns.sentiment, days=ns.days, epochs=ns.epochs)
        if not result.get("success"):
            print(f"❌ Model training failed: {result.get('error')}")
            return 1
        pkg = result["deployment_package"]
        print(f"✅ Model training completed. Staged at: {pkg['staging_path']}")
        if ns.auto_deploy:
            ok = trainer.deploy_model_to_live(pkg)
            print("✅ Model deployed" if ok else "❌ Model deployment failed")
            return 0 if ok else 1
        return 0

    if ns.control_cmd == "deploy-model":
        ok = trainer.deploy_model_to_live({"staging_path": ns.model_path, "strategy_name": "ml_basic", "ready_for_deployment": True}, close_positions=ns.close_positions)
        print("✅ Model deployment successful!" if ok else "❌ Model deployment failed!")
        return 0 if ok else 1

    if ns.control_cmd == "list-models":
        print(json.dumps(trainer.list_available_models(), indent=2))
        return 0

    if ns.control_cmd == "status":
        # Placeholder until wired into running engine
        status = {"connected": False, "running": False, "current_strategy": "Unknown", "active_positions": 0}
        print(json.dumps(status, indent=2))
        return 0

    if ns.control_cmd == "emergency-stop":
        print("🚨 EMERGENCY STOP INITIATED (simulated)")
        return 0

    if ns.control_cmd == "swap-strategy":
        print(f"🔄 Strategy swap requested to {ns.strategy} (close_positions={ns.close_positions}) (simulated)")
        return 0

    return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("live", help="Run live trading (proxies to scripts.run_live_trading)")
    p.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed through to runner")
    p.set_defaults(func=_handle)

    # Control group
    pc = subparsers.add_parser("live-control", help="Control live trading (train/deploy/swap/status)")
    sub = pc.add_subparsers(dest="control_cmd", required=True)

    p_train = sub.add_parser("train", help="Train new model")
    p_train.add_argument("--symbol", default="BTCUSDT")
    p_train.add_argument("--sentiment", action="store_true")
    p_train.add_argument("--days", type=int, default=365)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--auto-deploy", action="store_true")
    p_train.set_defaults(func=_control)

    p_deploy = sub.add_parser("deploy-model", help="Deploy a staged model")
    p_deploy.add_argument("--model-path", required=True)
    p_deploy.add_argument("--close-positions", action="store_true")
    p_deploy.set_defaults(func=_control)

    p_models = sub.add_parser("list-models", help="List available models")
    p_models.set_defaults(func=_control)

    p_status = sub.add_parser("status", help="Show engine status")
    p_status.set_defaults(func=_control)

    p_stop = sub.add_parser("emergency-stop", help="Emergency stop trading")
    p_stop.set_defaults(func=_control)

    p_swap = sub.add_parser("swap-strategy", help="Hot-swap strategy (simulated)")
    p_swap.add_argument("--strategy", required=True)
    p_swap.add_argument("--close-positions", action="store_true")
    p_swap.set_defaults(func=_control)


