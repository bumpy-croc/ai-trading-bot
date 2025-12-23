from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd

from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

from src.data_providers.binance_provider import BinanceProvider  # noqa: E402
from src.regime.assessment import RegimeAssessment, RegimeAssessmentConfig  # noqa: E402
from src.regime.assessment_visualizer import RegimeAssessmentVisualizer  # noqa: E402
from src.regime.detector import RegimeDetector  # noqa: E402
from src.regime.enhanced_detector import EnhancedRegimeDetector  # noqa: E402

LOGGER = logging.getLogger("atb.regime")


def _fetch_price_data(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    LOGGER.info("Fetching %s days of %s data for %s", days, timeframe, symbol)
    provider = BinanceProvider()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    df = provider.get_historical_data(symbol, timeframe, start_date, end_date)
    if df.empty:
        raise ValueError(f"No data returned for {symbol} {timeframe}")
    LOGGER.info(
        "Fetched %s candles from %s to %s",
        len(df),
        df.index.min(),
        df.index.max(),
    )
    return df


def _apply_regime_detection(df: pd.DataFrame) -> pd.DataFrame:
    LOGGER.info("Applying regime detection")
    detector = RegimeDetector()
    return detector.annotate(df)


def _create_visualization(
    df: pd.DataFrame, symbol: str, timeframe: str, days: int, output: Path
) -> None:
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 1, 1], hspace=0.3)
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(df.index, df["close"], "k-", linewidth=1, alpha=0.8, label="Price")
    regime_colors = {
        "trend_up:low_vol": "lightgreen",
        "trend_up:high_vol": "green",
        "trend_down:low_vol": "lightcoral",
        "trend_down:high_vol": "red",
        "range:low_vol": "lightblue",
        "range:high_vol": "blue",
    }
    current_regime = None
    regime_start = None
    for timestamp, row in df.iterrows():
        regime = row["regime_label"]
        if regime != current_regime:
            if current_regime is not None and regime_start is not None:
                color = regime_colors.get(current_regime, "lightgray")
                ax1.axvspan(regime_start, timestamp, alpha=0.3, color=color)
            current_regime = regime
            regime_start = timestamp
    if current_regime is not None and regime_start is not None:
        color = regime_colors.get(current_regime, "lightgray")
        ax1.axvspan(regime_start, df.index[-1], alpha=0.3, color=color)
    ax1.set_title(
        f"{symbol} Price with Market Regime Detection\n{timeframe} timeframe, last {days} days",
        fontsize=14,
        fontweight="bold",
    )
    ax1.set_ylabel("Price (USDT)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    regime_changes = df[df["regime_label"] != df["regime_label"].shift(1)]
    for timestamp, row in regime_changes.iterrows():
        ax1.axvline(x=timestamp, color="black", linestyle="--", alpha=0.5)
        ax1.text(
            timestamp,
            ax1.get_ylim()[1] * 0.95,
            row["regime_label"].replace(":", "\n"),
            rotation=90,
            fontsize=8,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(df.index, df["trend_score"], "purple", linewidth=1, label="Trend Score")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax2.fill_between(
        df.index,
        df["trend_score"],
        0,
        where=(df["trend_score"] >= 0),
        color="green",
        alpha=0.3,
        label="Positive Trend",
    )
    ax2.fill_between(
        df.index,
        df["trend_score"],
        0,
        where=(df["trend_score"] < 0),
        color="red",
        alpha=0.3,
        label="Negative Trend",
    )
    ax2.set_ylabel("Trend Score", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(df.index, df["atr_percentile"], "orange", linewidth=1, label="ATR Percentile")
    ax3.axhline(y=0.7, color="red", linestyle="--", alpha=0.7, label="High Vol Threshold")
    ax3.fill_between(
        df.index,
        0,
        1,
        where=(df["vol_label"] == "high_vol"),
        color="red",
        alpha=0.2,
        label="High Vol",
    )
    ax3.fill_between(
        df.index,
        0,
        1,
        where=(df["vol_label"] == "low_vol"),
        color="green",
        alpha=0.2,
        label="Low Vol",
    )
    ax3.set_ylabel("Volatility\n(ATR %ile)", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax4.plot(df.index, df["regime_confidence"], "blue", linewidth=1, label="Regime Confidence")
    ax4.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="Medium Confidence")
    ax4.fill_between(df.index, 0, df["regime_confidence"], alpha=0.3, color="blue")
    ax4.set_ylabel("Confidence", fontsize=10)
    ax4.set_ylim(0, 1)
    ax4.set_xlabel("Date", fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    legend_elements = [patches.Patch(color=c, alpha=0.3, label=r) for r, c in regime_colors.items()]
    ax1.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=8,
        title="Market Regimes",
        title_fontsize=9,
    )
    stats_text = (
        "Regime Statistics:\n"
        f"• Total periods: {len(df)}\n"
        f"• Trend Up: {len(df[df['trend_label'] == 'trend_up'])}"
        f" ({len(df[df['trend_label'] == 'trend_up'])/len(df)*100:.1f}%)\n"
        f"• Trend Down: {len(df[df['trend_label'] == 'trend_down'])}"
        f" ({len(df[df['trend_label'] == 'trend_down'])/len(df)*100:.1f}%)\n"
        f"• Range: {len(df[df['trend_label'] == 'range'])}"
        f" ({len(df[df['trend_label'] == 'range'])/len(df)*100:.1f}%)\n"
        f"• High Vol: {len(df[df['vol_label'] == 'high_vol'])}"
        f" ({len(df[df['vol_label'] == 'high_vol'])/len(df)*100:.1f}%)\n"
        f"• Avg Confidence: {df['regime_confidence'].mean():.3f}"
    )
    ax1.text(
        0.02,
        0.02,
        stats_text,
        transform=ax1.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    LOGGER.info("Visualization saved to %s", output)
    if not plt.isinteractive():
        plt.close(fig)


def _handle_visualize(ns: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        df = _fetch_price_data(ns.symbol, ns.timeframe, ns.days)
        df = _apply_regime_detection(df)
        output_path = Path(ns.output)
        _create_visualization(df, ns.symbol, ns.timeframe, ns.days, output_path)
        print(f"✅ Regime visualization written to {output_path}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"❌ Failed to generate visualization: {exc}")
        return 1


def _handle_assess(ns: argparse.Namespace) -> int:
    """
    Handle the regime assess subcommand.

    Fetches historical data, runs regime detection, computes assessment metrics,
    generates visualizations, and outputs a comprehensive report.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        output_dir = Path(ns.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Fetch data
        print(f"Fetching {ns.days} days of {ns.timeframe} data for {ns.symbol}...")
        df = _fetch_price_data(ns.symbol, ns.timeframe, ns.days)
        print(f"Fetched {len(df):,} candles")

        # Run base regime detector
        print("\nRunning RegimeDetector...")
        base_detector = RegimeDetector()
        df_annotated = base_detector.annotate(df.copy())

        # Configure and run assessment
        config = RegimeAssessmentConfig(
            lookahead=ns.lookahead,
            range_threshold=0.02,
            confidence_bins=10,
        )

        print("Computing assessment metrics...")
        assessment = RegimeAssessment(df_annotated, config)
        metrics = assessment.compute_all_metrics()

        # Get durations for histogram
        persistence = assessment.compute_persistence_metrics()
        durations = persistence.get("durations", [])

        # Generate visualizations
        print("\nGenerating visualizations...")
        visualizer = RegimeAssessmentVisualizer(metrics, output_dir)
        visualizer.save_all_charts(durations)

        # Run enhanced detector comparison if requested
        enhanced_metrics = None
        if ns.compare_enhanced:
            print("\nRunning EnhancedRegimeDetector for comparison...")
            enhanced_detector = EnhancedRegimeDetector()

            # Use base detector's annotate to get regime labels, then enhance
            df_for_enhanced = base_detector.annotate(df.copy())

            # Build enhanced regime labels by iterating through the data
            trend_labels = []
            vol_labels = []
            confidences = []

            for i in range(len(df_for_enhanced)):
                try:
                    regime_ctx = enhanced_detector.detect_regime(df_for_enhanced, i)
                    trend_labels.append(regime_ctx.trend.value)
                    vol_labels.append(regime_ctx.volatility.value)
                    confidences.append(regime_ctx.confidence)
                except (IndexError, ValueError, KeyError):
                    # Fallback for edge cases - use base detector values
                    row = df_for_enhanced.iloc[i]
                    trend_labels.append(str(row.get("trend_label", "range")))
                    vol_labels.append(str(row.get("vol_label", "low_vol")))
                    confidences.append(float(row.get("regime_confidence", 0.5)))

            df_enhanced = df_for_enhanced.copy()
            df_enhanced["trend_label"] = trend_labels
            df_enhanced["vol_label"] = vol_labels
            df_enhanced["regime_confidence"] = confidences

            enhanced_assessment = RegimeAssessment(df_enhanced, config)
            enhanced_metrics = enhanced_assessment.compute_all_metrics()

            # Generate comparison chart
            visualizer.plot_detector_comparison(
                metrics,
                enhanced_metrics,
                labels=("RegimeDetector", "EnhancedDetector"),
            )

        # Save metrics to JSON
        metrics_path = output_dir / "metrics.json"
        assessment.save_metrics(metrics_path)

        if enhanced_metrics:
            enhanced_path = output_dir / "enhanced_metrics.json"
            with open(enhanced_path, "w", encoding="utf-8") as f:
                json.dump(enhanced_metrics.to_dict(), f, indent=2)
            print(f"Enhanced metrics saved to {enhanced_path}")

        # Print console report
        print("\n" + assessment.generate_report())

        # Print comparison summary if available
        if enhanced_metrics:
            print("\n" + "=" * 60)
            print("DETECTOR COMPARISON SUMMARY")
            print("=" * 60)
            print(f"{'Metric':<30} {'RegimeDetector':>15} {'Enhanced':>15}")
            print("-" * 60)
            print(
                f"{'Overall Accuracy':<30} "
                f"{metrics.overall_accuracy:>14.1%} "
                f"{enhanced_metrics.overall_accuracy:>14.1%}"
            )
            print(
                f"{'Avg Regime Duration':<30} "
                f"{metrics.avg_regime_duration:>14.1f} "
                f"{enhanced_metrics.avg_regime_duration:>14.1f}"
            )
            print(
                f"{'Transition Frequency':<30} "
                f"{metrics.transition_frequency:>14.2%} "
                f"{enhanced_metrics.transition_frequency:>14.2%}"
            )
            print(
                f"{'ECE':<30} "
                f"{metrics.expected_calibration_error:>14.3f} "
                f"{enhanced_metrics.expected_calibration_error:>14.3f}"
            )
            print("=" * 60)

        print(f"\n✅ Assessment complete. Results saved to {output_dir}")
        return 0

    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Assessment failed")
        print(f"❌ Assessment failed: {exc}")
        plt.close("all")  # Clean up any open figures
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("regime", help="Market regime analysis utilities")
    sub = parser.add_subparsers(dest="regime_cmd", required=True)
    p_viz = sub.add_parser("visualize", help="Generate regime visualization for a symbol")
    p_viz.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    p_viz.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    p_viz.add_argument(
        "--days", type=int, default=30, help="Number of days to analyse (default: 30)"
    )
    p_viz.add_argument(
        "--output",
        default="artifacts/regime_visualizations/regime_analysis.png",
        help="Path to save the generated figure",
    )
    p_viz.set_defaults(func=_handle_visualize)

    # Assess subcommand
    p_assess = sub.add_parser("assess", help="Run regime detector assessment with historical data")
    p_assess.add_argument("symbol", help="Trading pair symbol (e.g., BTCUSDT)")
    p_assess.add_argument(
        "--days", type=int, default=730, help="Number of days to analyse (default: 730)"
    )
    p_assess.add_argument("--timeframe", default="1h", help="Candle timeframe (default: 1h)")
    p_assess.add_argument(
        "--lookahead",
        type=int,
        default=20,
        help="Lookahead bars for forward accuracy (default: 20)",
    )
    p_assess.add_argument(
        "--compare-enhanced",
        action="store_true",
        help="Compare with EnhancedRegimeDetector",
    )
    p_assess.add_argument(
        "--output-dir",
        default="artifacts/regime_assessment",
        help="Directory to save outputs (default: artifacts/regime_assessment)",
    )
    p_assess.set_defaults(func=_handle_assess)
