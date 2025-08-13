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
from src.optimizer.schemas import ExperimentConfig
from src.optimizer.analyzer import PerformanceAnalyzer
from src.config.constants import DEFAULT_INITIAL_BALANCE


def parse_args():
    p = argparse.ArgumentParser(description="Run optimization cycle (MVP)")
    p.add_argument("--strategy", default="ml_basic")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE)
    p.add_argument("--provider", choices=["binance", "coinbase", "mock", "fixture"], default="mock")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--output", default="artifacts/optimizer_report.json")
    return p.parse_args()


def main():
    args = parse_args()

    end = datetime.now()
    start = end - timedelta(days=args.days)

    cfg = ExperimentConfig(
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
    )

    runner = ExperimentRunner()
    result = runner.run(cfg)

    analyzer = PerformanceAnalyzer()
    suggestions = analyzer.analyze([result])

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
            "total_trades": result.total_trades,
            "win_rate": result.win_rate,
            "total_return": result.total_return,
            "annualized_return": result.annualized_return,
            "max_drawdown": result.max_drawdown,
            "sharpe_ratio": result.sharpe_ratio,
            "final_balance": result.final_balance,
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

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()