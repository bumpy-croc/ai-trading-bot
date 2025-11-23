#!/usr/bin/env python3
"""
Automated training and benchmarking script for Phase 1 model validation.

This script automates the complete Phase 1 testing workflow:
1. Quick validation tests (7 days, 5 epochs)
2. Full training runs (30 days, 20 epochs)
3. Performance comparison
4. Results analysis

Usage:
    # Quick validation only
    python scripts/phase1_benchmark.py --quick

    # Full benchmark
    python scripts/phase1_benchmark.py --full

    # Skip training, just analyze existing models
    python scripts/phase1_benchmark.py --analyze-only
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Color codes for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def run_training(symbol: str, model_type: str, days: int, epochs: int, variant: str = "default") -> Tuple[bool, float, Dict]:
    """
    Run a training job.

    Returns:
        Tuple of (success, duration, metrics)
    """
    print(f"\n{BOLD}Training {model_type} ({variant}) on {symbol}...{RESET}")
    print(f"  Dataset: {days} days, {epochs} epochs")

    cmd = [
        "atb", "train", "model", symbol,
        "--model-type", model_type,
        "--model-variant", variant,
        "--days", str(days),
        "--epochs", str(epochs),
        "--timeframe", "1h",
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200  # 2 hour timeout
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print_success(f"Training completed in {duration:.1f}s")

            # Extract metrics from output
            metrics = _parse_training_output(result.stdout)
            return True, duration, metrics
        else:
            print_error(f"Training failed: {result.stderr}")
            return False, duration, {}

    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print_error(f"Training timed out after {duration:.1f}s")
        return False, duration, {}
    except Exception as e:
        duration = time.time() - start_time
        print_error(f"Training error: {e}")
        return False, duration, {}


def _parse_training_output(output: str) -> Dict:
    """Parse metrics from training output."""
    metrics = {}

    # Look for metrics in output
    for line in output.split('\n'):
        if 'Test RMSE:' in line:
            parts = line.split('|')
            if len(parts) >= 2:
                # Extract RMSE
                rmse_part = parts[0].split(':')[-1].strip()
                try:
                    metrics['rmse'] = float(rmse_part)
                except ValueError:
                    pass

                # Extract MAPE
                mape_part = parts[1].split(':')[-1].strip().rstrip('%')
                try:
                    metrics['mape'] = float(mape_part)
                except ValueError:
                    pass

    return metrics


def load_model_metadata(symbol: str, model_type: str) -> Dict:
    """Load metadata from latest model."""
    metadata_path = project_root / "src" / "ml" / "models" / symbol / "price" / "latest" / "metadata.json"

    if not metadata_path.exists():
        return {}

    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print_warning(f"Could not load metadata: {e}")
        return {}


def quick_validation(symbol: str = "BTCUSDT"):
    """
    Run quick validation tests (7 days, 5 epochs).

    Fast test to verify all models can train without errors.
    """
    print_header("PHASE 1: QUICK VALIDATION")

    models_to_test = [
        ("cnn_lstm", "default", "CNN-LSTM Baseline"),
        ("attention_lstm", "default", "Attention-LSTM"),
        ("tcn", "default", "TCN"),
        ("tcn_attention", "default", "TCN+Attention"),
    ]

    results = []

    for model_type, variant, name in models_to_test:
        print(f"\n{BOLD}Testing {name}...{RESET}")
        success, duration, metrics = run_training(symbol, model_type, days=7, epochs=5, variant=variant)

        results.append({
            'model': name,
            'model_type': model_type,
            'variant': variant,
            'success': success,
            'duration': duration,
            'metrics': metrics
        })

        if success:
            print_success(f"{name} validation passed ({duration:.1f}s)")
        else:
            print_error(f"{name} validation failed")

    # Summary
    print_header("QUICK VALIDATION SUMMARY")

    passed = sum(1 for r in results if r['success'])
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Passed: {GREEN}{passed} ✓{RESET}")
    print(f"Failed: {RED}{total - passed} ✗{RESET}\n")

    if passed == total:
        print_success("All models validated successfully!")
        print("\nReady to proceed with full training runs.")
        return True
    else:
        print_error("Some models failed validation. Fix errors before proceeding.")
        return False


def full_benchmark(symbol: str = "BTCUSDT"):
    """
    Run full benchmark (30 days, 20 epochs).

    Real training runs for meaningful comparison.
    """
    print_header("PHASE 1: FULL TRAINING BENCHMARK")

    models_to_test = [
        ("cnn_lstm", "default", "CNN-LSTM Baseline"),
        ("attention_lstm", "default", "Attention-LSTM"),
        ("tcn", "default", "TCN"),
    ]

    results = []

    for model_type, variant, name in models_to_test:
        print(f"\n{BOLD}{'='*80}{RESET}")
        print(f"{BOLD}Training {name} (Full Benchmark){RESET}")
        print(f"{BOLD}{'='*80}{RESET}")

        success, duration, metrics = run_training(symbol, model_type, days=30, epochs=20, variant=variant)

        # Load full metadata
        metadata = load_model_metadata(symbol, model_type)

        results.append({
            'model': name,
            'model_type': model_type,
            'variant': variant,
            'success': success,
            'duration': duration,
            'metrics': metrics,
            'metadata': metadata
        })

        if success:
            print_success(f"{name} training completed")
            if metrics:
                print(f"  RMSE: {metrics.get('rmse', 'N/A')}")
                print(f"  MAPE: {metrics.get('mape', 'N/A')}%")
        else:
            print_error(f"{name} training failed")

    # Analysis
    analyze_results(results)

    return results


def analyze_results(results: List[Dict]):
    """Analyze and compare training results."""
    print_header("RESULTS ANALYSIS")

    # Find baseline
    baseline = next((r for r in results if r['model_type'] == 'cnn_lstm'), None)

    if not baseline or not baseline['success']:
        print_warning("Baseline (CNN-LSTM) not available for comparison")
        return

    baseline_rmse = baseline['metrics'].get('rmse', 0)
    baseline_mape = baseline['metrics'].get('mape', 0)
    baseline_time = baseline['duration']

    print(f"\n{BOLD}Baseline (CNN-LSTM):{RESET}")
    print(f"  RMSE: {baseline_rmse:.6f}")
    print(f"  MAPE: {baseline_mape:.2f}%")
    print(f"  Training Time: {baseline_time:.1f}s")

    print(f"\n{BOLD}Comparison vs Baseline:{RESET}\n")
    print(f"{'Model':<20} {'RMSE Change':<15} {'MAPE Change':<15} {'Time Ratio':<15} {'Status':<10}")
    print("-" * 75)

    for result in results:
        if result['model_type'] == 'cnn_lstm':
            continue

        if not result['success']:
            print(f"{result['model']:<20} {'FAILED':<15} {'FAILED':<15} {'FAILED':<15} {'✗':<10}")
            continue

        rmse = result['metrics'].get('rmse', 0)
        mape = result['metrics'].get('mape', 0)
        duration = result['duration']

        rmse_change = ((rmse - baseline_rmse) / baseline_rmse) * 100 if baseline_rmse > 0 else 0
        mape_change = ((mape - baseline_mape) / baseline_mape) * 100 if baseline_mape > 0 else 0
        time_ratio = duration / baseline_time if baseline_time > 0 else 0

        # Color code improvements
        rmse_str = f"{rmse_change:+.1f}%"
        if rmse_change < -5:
            rmse_str = f"{GREEN}{rmse_str}{RESET}"
        elif rmse_change > 5:
            rmse_str = f"{RED}{rmse_str}{RESET}"

        mape_str = f"{mape_change:+.1f}%"
        if mape_change < -5:
            mape_str = f"{GREEN}{mape_str}{RESET}"
        elif mape_change > 5:
            mape_str = f"{RED}{mape_str}{RESET}"

        time_str = f"{time_ratio:.2f}x"
        if time_ratio < 0.5:
            time_str = f"{GREEN}{time_str}{RESET}"
        elif time_ratio > 1.5:
            time_str = f"{RED}{time_str}{RESET}"

        status = "✓" if rmse_change < 0 else "✗"

        print(f"{result['model']:<20} {rmse_str:<24} {mape_str:<24} {time_str:<24} {status:<10}")

    # Recommendations
    print_header("RECOMMENDATIONS")

    best_accuracy = min((r for r in results if r['success'] and 'rmse' in r['metrics']),
                       key=lambda r: r['metrics']['rmse'], default=None)

    fastest = min((r for r in results if r['success']),
                 key=lambda r: r['duration'], default=None)

    if best_accuracy:
        rmse_improvement = ((baseline_rmse - best_accuracy['metrics']['rmse']) / baseline_rmse) * 100
        print(f"\n{BOLD}Best Accuracy:{RESET} {best_accuracy['model']}")
        print(f"  RMSE Improvement: {GREEN}{rmse_improvement:.1f}%{RESET}")

        if rmse_improvement >= 10:
            print_success("EXCELLENT! Proceed to Phase 2: Ensemble Implementation")
        elif rmse_improvement >= 5:
            print_warning("GOOD! Consider Phase 2 or try longer training")
        else:
            print_warning("MARGINAL. Try longer training or different hyperparameters")

    if fastest:
        speedup = baseline_time / fastest['duration']
        print(f"\n{BOLD}Fastest Training:{RESET} {fastest['model']}")
        print(f"  Speedup: {GREEN}{speedup:.1f}x faster{RESET}")


def save_results(results: List[Dict], filename: str = "phase1_results.json"):
    """Save results to JSON file."""
    output_path = project_root / "artifacts" / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, f, indent=2)

    print_success(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Model Benchmarking")
    parser.add_argument('--quick', action='store_true', help="Run quick validation only (7 days, 5 epochs)")
    parser.add_argument('--full', action='store_true', help="Run full benchmark (30 days, 20 epochs)")
    parser.add_argument('--analyze-only', action='store_true', help="Analyze existing models without training")
    parser.add_argument('--symbol', default='BTCUSDT', help="Trading symbol (default: BTCUSDT)")

    args = parser.parse_args()

    print_header("PHASE 1: ML MODEL ARCHITECTURE BENCHMARKING")
    print(f"Symbol: {args.symbol}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        if args.analyze_only:
            print_warning("Analyze-only mode not yet implemented")
            print("Load existing model metadata and compare")
            return 0

        elif args.quick:
            success = quick_validation(args.symbol)
            return 0 if success else 1

        elif args.full:
            results = full_benchmark(args.symbol)
            save_results(results)
            return 0

        else:
            # Default: run quick validation
            print("No mode specified. Running quick validation...")
            print("Use --quick, --full, or --analyze-only")
            print()
            success = quick_validation(args.symbol)
            return 0 if success else 1

    except KeyboardInterrupt:
        print_warning("\n\nBenchmark interrupted by user")
        return 130
    except Exception as e:
        print_error(f"\n\nBenchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
