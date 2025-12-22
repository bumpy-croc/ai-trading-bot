"""
Regime Assessment Visualizer

Generates matplotlib charts for regime detector assessment results.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.regime.assessment import AssessmentMetrics

logger = logging.getLogger(__name__)


class RegimeAssessmentVisualizer:
    """
    Generates visualizations for regime assessment metrics.

    Creates charts for accuracy analysis, calibration curves,
    regime distribution, and transition heatmaps.
    """

    def __init__(self, metrics: AssessmentMetrics, output_dir: Optional[Path] = None):
        """
        Initialize visualizer.

        Args:
            metrics: Assessment metrics to visualize
            output_dir: Directory to save charts (optional)
        """
        self.metrics = metrics
        self.output_dir = output_dir or Path("artifacts/regime_assessment")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Style configuration
        plt.style.use("seaborn-v0_8-whitegrid")
        self.colors = {
            "trend_up": "#2ecc71",
            "trend_down": "#e74c3c",
            "range": "#3498db",
            "high_vol": "#e67e22",
            "low_vol": "#9b59b6",
        }

    def plot_accuracy_by_regime(self, save: bool = True) -> plt.Figure:
        """
        Create bar chart showing accuracy by regime type.

        Args:
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # By trend regime
        ax1 = axes[0]
        regimes = list(self.metrics.accuracy_by_regime.keys())
        accuracies = [self.metrics.accuracy_by_regime[r] for r in regimes]
        colors = [self.colors.get(r, "#95a5a6") for r in regimes]

        bars = ax1.bar(regimes, accuracies, color=colors, edgecolor="black", linewidth=1.2)
        ax1.axhline(y=0.33, color="red", linestyle="--", label="Random baseline (33%)")
        ax1.axhline(y=0.6, color="green", linestyle="--", label="Target (60%)")
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_xlabel("Trend Regime", fontsize=12)
        ax1.set_title("Forward-Looking Accuracy by Trend Regime", fontsize=14, fontweight="bold")
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper right")

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        # By volatility
        ax2 = axes[1]
        vol_types = list(self.metrics.accuracy_by_volatility.keys())
        vol_accuracies = [self.metrics.accuracy_by_volatility[v] for v in vol_types]
        vol_colors = [self.colors.get(v, "#95a5a6") for v in vol_types]

        bars2 = ax2.bar(vol_types, vol_accuracies, color=vol_colors, edgecolor="black", linewidth=1.2)
        ax2.axhline(y=0.5, color="red", linestyle="--", label="Random baseline (50%)")
        ax2.set_ylabel("Accuracy", fontsize=12)
        ax2.set_xlabel("Volatility Regime", fontsize=12)
        ax2.set_title("Accuracy by Volatility Regime", fontsize=14, fontweight="bold")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper right")

        for bar, acc in zip(bars2, vol_accuracies):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{acc:.1%}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

        plt.tight_layout()

        if save:
            path = self.output_dir / "accuracy_by_regime.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved accuracy chart to %s", path)

        return fig

    def plot_confidence_calibration(self, save: bool = True) -> plt.Figure:
        """
        Create calibration curve plot.

        Shows predicted confidence vs actual accuracy to assess calibration.

        Args:
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))

        curve = self.metrics.calibration_curve
        if not curve:
            ax.text(0.5, 0.5, "No calibration data available", ha="center", va="center", fontsize=14)
            return fig

        confidences = [p["avg_confidence"] for p in curve]
        accuracies = [p["accuracy"] for p in curve]
        sizes = [p["sample_size"] for p in curve]

        # Normalize sizes for scatter plot
        max_size = max(sizes) if sizes else 1
        normalized_sizes = [100 + 400 * (s / max_size) for s in sizes]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Perfect calibration")

        # Calibration curve
        ax.scatter(confidences, accuracies, s=normalized_sizes, alpha=0.7, c="#3498db", edgecolors="black")
        ax.plot(confidences, accuracies, "b-", alpha=0.5, linewidth=2)

        # ECE annotation
        ece = self.metrics.expected_calibration_error
        ax.text(
            0.05,
            0.95,
            f"ECE = {ece:.3f}",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
        ax.set_ylabel("Observed Accuracy", fontsize=12)
        ax.set_title("Confidence Calibration Curve", fontsize=14, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc="lower right")
        ax.set_aspect("equal")

        plt.tight_layout()

        if save:
            path = self.output_dir / "calibration_curve.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved calibration curve to %s", path)

        return fig

    def plot_regime_distribution(self, save: bool = True) -> plt.Figure:
        """
        Create pie charts showing regime and volatility distribution.

        Args:
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Trend regime distribution
        ax1 = axes[0]
        regime_dist = self.metrics.regime_distribution
        if regime_dist:
            labels = list(regime_dist.keys())
            sizes = list(regime_dist.values())
            colors = [self.colors.get(l, "#95a5a6") for l in labels]

            wedges, texts, autotexts = ax1.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                explode=[0.02] * len(labels),
            )
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight("bold")
            ax1.set_title("Trend Regime Distribution", fontsize=14, fontweight="bold")
        else:
            ax1.text(0.5, 0.5, "No data", ha="center", va="center")

        # Volatility distribution
        ax2 = axes[1]
        vol_dist = self.metrics.volatility_distribution
        if vol_dist:
            labels = list(vol_dist.keys())
            sizes = list(vol_dist.values())
            colors = [self.colors.get(l, "#95a5a6") for l in labels]

            wedges, texts, autotexts = ax2.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=90,
                explode=[0.02] * len(labels),
            )
            for autotext in autotexts:
                autotext.set_fontsize(11)
                autotext.set_fontweight("bold")
            ax2.set_title("Volatility Distribution", fontsize=14, fontweight="bold")
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center")

        plt.tight_layout()

        if save:
            path = self.output_dir / "regime_distribution.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved distribution chart to %s", path)

        return fig

    def plot_transition_heatmap(self, save: bool = True) -> plt.Figure:
        """
        Create heatmap of regime transition probabilities.

        Args:
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        trans_probs = self.metrics.transition_probabilities
        if not trans_probs:
            ax.text(0.5, 0.5, "No transition data available", ha="center", va="center", fontsize=14)
            return fig

        regimes = list(trans_probs.keys())
        n = len(regimes)

        # Build matrix
        matrix = np.zeros((n, n))
        for i, from_regime in enumerate(regimes):
            for j, to_regime in enumerate(regimes):
                matrix[i, j] = trans_probs[from_regime].get(to_regime, 0)

        # Create heatmap
        im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Transition Probability", fontsize=11)

        # Set ticks
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(regimes, rotation=45, ha="right")
        ax.set_yticklabels(regimes)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                text_color = "white" if matrix[i, j] > 0.5 else "black"
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color=text_color, fontsize=10)

        ax.set_xlabel("To Regime", fontsize=12)
        ax.set_ylabel("From Regime", fontsize=12)
        ax.set_title("Regime Transition Probabilities", fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save:
            path = self.output_dir / "transition_heatmap.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved transition heatmap to %s", path)

        return fig

    def plot_persistence_histogram(self, durations: list[int], save: bool = True) -> plt.Figure:
        """
        Create histogram of regime durations.

        Args:
            durations: List of regime duration values
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if not durations:
            ax.text(0.5, 0.5, "No duration data available", ha="center", va="center", fontsize=14)
            return fig

        # Create histogram
        n_bins = min(50, len(set(durations)))
        ax.hist(durations, bins=n_bins, color="#3498db", edgecolor="black", alpha=0.7)

        # Add statistics lines
        avg = np.mean(durations)
        median = np.median(durations)

        ax.axvline(avg, color="red", linestyle="--", linewidth=2, label=f"Mean: {avg:.1f}")
        ax.axvline(median, color="green", linestyle="-.", linewidth=2, label=f"Median: {median:.1f}")
        ax.axvline(20, color="orange", linestyle=":", linewidth=2, label="Target: 20")

        ax.set_xlabel("Regime Duration (bars)", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Regime Durations", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right")

        plt.tight_layout()

        if save:
            path = self.output_dir / "persistence_histogram.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved persistence histogram to %s", path)

        return fig

    def plot_detector_comparison(
        self,
        metrics1: AssessmentMetrics,
        metrics2: AssessmentMetrics,
        labels: tuple[str, str] = ("RegimeDetector", "EnhancedDetector"),
        save: bool = True,
    ) -> plt.Figure:
        """
        Create side-by-side comparison of two detectors.

        Args:
            metrics1: Metrics from first detector
            metrics2: Metrics from second detector
            labels: Names for each detector
            save: Whether to save the figure

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        x = np.arange(2)
        width = 0.35

        # Overall accuracy comparison
        ax1 = axes[0]
        accuracies = [metrics1.overall_accuracy, metrics2.overall_accuracy]
        bars = ax1.bar(x, accuracies, width, color=["#3498db", "#e74c3c"], edgecolor="black")
        ax1.axhline(y=0.6, color="green", linestyle="--", label="Target (60%)")
        ax1.set_ylabel("Accuracy", fontsize=12)
        ax1.set_title("Overall Forward Accuracy", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.set_ylim(0, 1)
        ax1.legend()

        for bar, acc in zip(bars, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.1%}", ha="center", fontweight="bold")

        # ECE comparison
        ax2 = axes[1]
        eces = [metrics1.expected_calibration_error, metrics2.expected_calibration_error]
        bars = ax2.bar(x, eces, width, color=["#3498db", "#e74c3c"], edgecolor="black")
        ax2.axhline(y=0.1, color="green", linestyle="--", label="Target (< 0.1)")
        ax2.set_ylabel("ECE", fontsize=12)
        ax2.set_title("Expected Calibration Error", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels)
        ax2.legend()

        for bar, ece in zip(bars, eces):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{ece:.3f}", ha="center", fontweight="bold")

        # Persistence comparison
        ax3 = axes[2]
        durations = [metrics1.avg_regime_duration, metrics2.avg_regime_duration]
        bars = ax3.bar(x, durations, width, color=["#3498db", "#e74c3c"], edgecolor="black")
        ax3.axhline(y=20, color="green", linestyle="--", label="Target (> 20)")
        ax3.set_ylabel("Average Duration (bars)", fontsize=12)
        ax3.set_title("Regime Persistence", fontsize=14, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels)
        ax3.legend()

        for bar, dur in zip(bars, durations):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f"{dur:.1f}", ha="center", fontweight="bold")

        plt.tight_layout()

        if save:
            path = self.output_dir / "detector_comparison.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved detector comparison to %s", path)

        return fig

    def save_all_charts(self, durations: Optional[list[int]] = None) -> list[Path]:
        """
        Generate and save all charts.

        Args:
            durations: Optional list of regime durations for histogram

        Returns:
            List of saved file paths
        """
        saved_paths = []

        logger.info("Generating accuracy chart...")
        self.plot_accuracy_by_regime()
        saved_paths.append(self.output_dir / "accuracy_by_regime.png")

        logger.info("Generating calibration curve...")
        self.plot_confidence_calibration()
        saved_paths.append(self.output_dir / "calibration_curve.png")

        logger.info("Generating distribution charts...")
        self.plot_regime_distribution()
        saved_paths.append(self.output_dir / "regime_distribution.png")

        logger.info("Generating transition heatmap...")
        self.plot_transition_heatmap()
        saved_paths.append(self.output_dir / "transition_heatmap.png")

        if durations:
            logger.info("Generating persistence histogram...")
            self.plot_persistence_histogram(durations)
            saved_paths.append(self.output_dir / "persistence_histogram.png")

        plt.close("all")
        logger.info("All charts saved to %s", self.output_dir)

        return saved_paths
