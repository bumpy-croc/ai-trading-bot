#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root and its 'src' directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
src_path = project_root / 'src'
if src_path.exists():
    sys.path.append(str(src_path))

from src.optimizer.runner import ExperimentRunner
from src.optimizer.schemas import ExperimentConfig, ParameterSet
from src.optimizer.analyzer import PerformanceAnalyzer
from src.optimizer.validator import StatisticalValidator, ValidationConfig
from src.config.constants import DEFAULT_INITIAL_BALANCE
from src.database.manager import DatabaseManager


def parse_args():
    p = argparse.ArgumentParser(description="Run optimization cycle (Phase 2)")
    p.add_argument("--strategy", default="ml_basic")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    p.add_argument("--provider", choices=["binance", "coinbase", "mock", "fixture"], default="mock")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output", default="artifacts/optimizer_report.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-validate", action="store_true")
    p.add_argument("--persist", action="store_true", help="Persist cycle to database when available")
    return p.parse_args()


def main():
    args = parse_args()

    end = datetime.now()
    start = end - timedelta(days=args.days)

    baseline_cfg = ExperimentConfig(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start=start,
        end=end,
        initial_balance=args.initial_balance,
        risk_parameters={},
        feature_flags={},
        use_cache=not args.no_cache,
        provider=args.provider,
        random_seed=args.seed,
    )

    runner = ExperimentRunner()
    baseline_result = runner.run(baseline_cfg)

    analyzer = PerformanceAnalyzer()
    suggestions = analyzer.analyze([baseline_result])

    report = {
        "timestamp": datetime.now().isoformat(),
        "experiment": {
            "strategy": args.strategy,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
        },
        "results": {
            "total_trades": baseline_result.total_trades,
            "win_rate": baseline_result.win_rate,
            "total_return": baseline_result.total_return,
            "annualized_return": baseline_result.annualized_return,
            "max_drawdown": baseline_result.max_drawdown,
            "sharpe_ratio": baseline_result.sharpe_ratio,
            "final_balance": baseline_result.final_balance,
        },
        "suggestions": [
            {
                "target": s.target,
                "change": s.change,
                "rationale": s.rationale,
                "expected_delta": s.expected_delta,
                "confidence": s.confidence,
            }
            for s in suggestions
        ],
    }

    validation_section = None
    candidate_params = None
    candidate_metrics = None
    if suggestions and not args.no_validate:
        # Build a simple candidate config from first suggestion (strategy-level only for MVP)
        s0 = suggestions[0]
        param_values = {}
        for k, v in s0.change.items():
            if k.startswith("MlBasic."):
                param_values[k] = v
        candidate_cfg = ExperimentConfig(
            strategy_name=args.strategy,
            symbol=args.symbol,
            timeframe=args.timeframe,
            start=start,
            end=end,
            initial_balance=args.initial_balance,
            risk_parameters=baseline_cfg.risk_parameters,
            feature_flags=baseline_cfg.feature_flags,
            parameters=ParameterSet(name="candidate", values=param_values),
            use_cache=not args.no_cache,
            provider=args.provider,
            random_seed=args.seed,
        )
        candidate_result = runner.run(candidate_cfg)
        candidate_params = param_values
        candidate_metrics = {
            "total_trades": candidate_result.total_trades,
            "win_rate": candidate_result.win_rate,
            "total_return": candidate_result.total_return,
            "annualized_return": candidate_result.annualized_return,
            "max_drawdown": candidate_result.max_drawdown,
            "sharpe_ratio": candidate_result.sharpe_ratio,
            "final_balance": candidate_result.final_balance,
        }

        validator = StatisticalValidator(ValidationConfig())
        val_report = validator.validate([baseline_result], [candidate_result])
        validation_section = {
            "passed": val_report.passed,
            "p_value": val_report.p_value,
            "effect_size": val_report.effect_size,
            "baseline_metrics": val_report.baseline_metrics,
            "candidate_metrics": val_report.candidate_metrics,
        }
        report["validation"] = validation_section

    if args.persist:
        try:
            db = DatabaseManager()
            base_metrics = report["results"]
            decision = "propose"
            if validation_section:
                decision = "apply" if validation_section.get("passed") else "reject"
            db.record_optimization_cycle(
                strategy_name=args.strategy,
                symbol=args.symbol,
                timeframe=args.timeframe,
                baseline_metrics=base_metrics,
                candidate_params=candidate_params or {},
                candidate_metrics=candidate_metrics or {},
                validator_report=validation_section or {},
                decision=decision,
                session_id=None,
            )
        except Exception as e:
            # Do not fail the run if persistence is unavailable
            report.setdefault("persistence", {})["error"] = str(e)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()