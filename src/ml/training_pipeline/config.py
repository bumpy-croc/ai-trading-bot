"""Configuration objects for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.infrastructure.runtime.paths import get_project_root


@dataclass
class DiagnosticsOptions:
    """Flags that control optional diagnostics and exports."""

    generate_plots: bool = True
    evaluate_robustness: bool = True
    convert_to_onnx: bool = True


@dataclass
class TrainingPaths:
    """Filesystem locations used by the training pipeline."""

    project_root: Path
    data_dir: Path
    models_dir: Path

    @classmethod
    def default(cls) -> "TrainingPaths":
        root = get_project_root()
        data_dir = root / "data"
        models_dir = root / "src" / "ml"
        data_dir.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)
        return cls(project_root=root, data_dir=data_dir, models_dir=models_dir)


@dataclass
class TrainingConfig:
    """High-level parameters for a training run."""

    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    epochs: int = 300
    batch_size: int = 32
    sequence_length: int = 120
    force_sentiment: bool = False
    force_price_only: bool = False
    mixed_precision: bool = True
    diagnostics: DiagnosticsOptions = field(default_factory=DiagnosticsOptions)

    def days_requested(self) -> int:
        return (self.end_date - self.start_date).days


@dataclass
class TrainingContext:
    """Container object that binds config and paths for the pipeline."""

    config: TrainingConfig
    paths: TrainingPaths = field(default_factory=TrainingPaths.default)

    @property
    def symbol_exchange(self) -> str:
        from src.utils.symbol_factory import SymbolFactory

        return SymbolFactory.to_exchange_symbol(self.config.symbol, "binance")

    @property
    def start_iso(self) -> str:
        return self.config.start_date.strftime("%Y-%m-%dT00:00:00Z")

    @property
    def end_iso(self) -> str:
        return self.config.end_date.strftime("%Y-%m-%dT23:59:59Z")

    @property
    def price_data_glob(self) -> str:
        return f"{self.symbol_exchange}_{self.config.timeframe}_{self.start_iso}_{self.end_iso}.*"
