#!/usr/bin/env python3
"""Benchmark script for training pipeline with detailed timing."""

import cProfile
import pstats
import io
import time
import argparse
from datetime import datetime, timedelta
from pathlib import Path

from src.ml.training_pipeline import DiagnosticsOptions, TrainingConfig, TrainingContext
from src.ml.training_pipeline.pipeline import run_training_pipeline


def profile_training(args):
    """Run training with profiling and detailed timing."""

    # Calculate dates
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)

    # Configure training
    config = TrainingConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=start_date,
        end_date=end_date,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        force_price_only=True,  # Skip sentiment for faster baseline
        mixed_precision=not args.disable_mixed_precision,
        diagnostics=DiagnosticsOptions(
            generate_plots=not args.skip_plots,
            evaluate_robustness=not args.skip_robustness,
            convert_to_onnx=not args.skip_onnx,
        ),
    )

    ctx = TrainingContext(config=config)

    print("=" * 80)
    print("TRAINING PIPELINE BENCHMARK")
    print("=" * 80)
    print(f"Symbol: {config.symbol}")
    print(f"Timeframe: {config.timeframe}")
    print(f"Date range: {config.start_date.date()} to {config.end_date.date()}")
    print(f"Days: {args.days}")
    print(f"Epochs: {config.epochs}")
    print(f"Batch size: {config.batch_size}")
    print(f"Sequence length: {config.sequence_length}")
    print(f"Mixed precision: {config.mixed_precision}")
    print(f"Generate plots: {config.diagnostics.generate_plots}")
    print(f"Robustness eval: {config.diagnostics.evaluate_robustness}")
    print(f"ONNX export: {config.diagnostics.convert_to_onnx}")
    print("=" * 80)

    # Run with profiling
    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()

    start_time = time.perf_counter()
    result = run_training_pipeline(ctx)
    total_time = time.perf_counter() - start_time

    if args.profile:
        profiler.disable()

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Success: {result.success}")
    print(f"Total time: {total_time:.2f}s ({total_time / 60:.2f}m)")
    print(f"Pipeline duration: {result.duration_seconds:.2f}s")

    if result.success:
        eval_results = result.metadata.get("evaluation_results", {})
        print(f"Test RMSE: {eval_results.get('test_rmse', 0.0):.6f}")
        print(f"MAPE: {eval_results.get('mape', 0.0):.2f}%")

        training_params = result.metadata.get("training_params", {})
        actual_epochs = training_params.get("epochs", 0)
        if actual_epochs > 0:
            time_per_epoch = total_time / actual_epochs
            print(f"Time per epoch: {time_per_epoch:.2f}s")

    # Print profiling stats
    if args.profile and result.success:
        print("\n" + "=" * 80)
        print("TOP 20 TIME-CONSUMING FUNCTIONS")
        print("=" * 80)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())

    print("=" * 80)

    return 0 if result.success else 1


def main():
    parser = argparse.ArgumentParser(description="Benchmark training pipeline")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--timeframe", default="1h", help="Timeframe")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=120, help="Sequence length")
    parser.add_argument("--skip-plots", action="store_true", help="Skip plots")
    parser.add_argument("--skip-robustness", action="store_true", help="Skip robustness")
    parser.add_argument("--skip-onnx", action="store_true", help="Skip ONNX")
    parser.add_argument(
        "--disable-mixed-precision", action="store_true", help="Disable mixed precision"
    )
    parser.add_argument("--profile", action="store_true", help="Enable cProfile profiling")

    args = parser.parse_args()
    return profile_training(args)


if __name__ == "__main__":
    exit(main())
