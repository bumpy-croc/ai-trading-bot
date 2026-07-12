"""Configuration objects for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.infrastructure.runtime.paths import get_model_registry_root, get_project_root


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
    def default(cls) -> TrainingPaths:
        """Create default training paths based on project root.

        ``models_dir`` comes from ``get_model_registry_root()``, which
        honors the ``MODEL_REGISTRY_PATH`` environment variable when set --
        the same key ``PredictionConfig.model_registry_path`` reads on the
        backtest/prediction READ side, so one env var redirects both sides
        consistently (e.g. for acceptance-test hermeticity).

        Returns:
            TrainingPaths configured for local development
        """
        root = get_project_root()
        data_dir = root / "data"
        models_dir = get_model_registry_root()
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
    early_stopping_patience: int = 15  # Patience for early stopping callback
    diagnostics: DiagnosticsOptions = field(default_factory=DiagnosticsOptions)

    # Model architecture selection (new)
    model_type: str = "cnn_lstm"  # cnn_lstm, attention_lstm, tcn, tcn_attention, lstm
    model_variant: str = "default"  # default, lightweight, deep

    # Training target selection (TARGET-REDESIGN tournament, GH #933 Phase 2).
    # "regression" preserves the incumbent next-bar price-regression target
    # exactly. Cross-validating model_type against target_type (the #947
    # guard) happens at run_training_pipeline() entry, NOT here -- see
    # validate_target_head_compatibility() in task_types.py. Keeping that
    # check out of __post_init__ means a TrainingConfig stays freely
    # constructible (e.g. `atb train cloud --model-type tft`, which has no
    # --target-type flag yet) and only fails when a run is actually
    # attempted with an incompatible pairing.
    target_type: str = "regression"
    target_horizon: int = 1  # forward bars; used by binary_direction/smoothed_return

    # Registry model_type of the primary signal to run forward when
    # target_type="meta_label" (entrant (a) -- meta_label's label depends on
    # simulating an ALREADY-TRAINED primary signal's fired trades, not a
    # transform of the close-price series). Required when target_type ==
    # "meta_label" (validated below), ignored otherwise.
    primary_model_type: str | None = None

    # Fire-generation checkpoint directory for target_type="meta_label"
    # (#955 defect 2: the ~60-min forward pass over the corpus was killed
    # twice by task-lifetime caps with all progress lost). "auto" (default)
    # checkpoints under <data_dir>/meta_label_fire_checkpoints; any other
    # string is used as the directory; None disables checkpointing.
    meta_label_fire_checkpoint_dir: str | None = "auto"

    # Test-only escape hatch: routes load_training_corpus (ingestion.py) to
    # MockDataProvider (deterministic synthetic OHLCV, no network) instead of
    # the real Binance-backed cache. Mirrors `atb live --mock-data`'s
    # existing pattern (src/engines/live/runner.py). Never set in production
    # training -- exists so acceptance/integration tests can exercise the
    # REAL training pipeline end-to-end without a network dependency.
    use_mock_data: bool = False

    # Valid model types and variants for validation
    _VALID_MODEL_TYPES = frozenset(
        {
            "cnn_lstm",
            "adaptive",
            "default",
            "attention_lstm",
            "tcn",
            "tcn_attention",
            "tft",
            "tft_ternary",
            "lstm",
            "lightgbm",
        }
    )
    _VALID_MODEL_VARIANTS = frozenset({"default", "lightweight", "deep"})
    _VALID_TARGET_TYPES = frozenset(
        {"regression", "binary_direction", "triple_barrier", "smoothed_return", "meta_label"}
    )

    def __post_init__(self):
        """Validate training configuration parameters for fail-fast behavior."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {self.sequence_length}")
        if self.start_date >= self.end_date:
            raise ValueError(
                f"start_date must be before end_date, got {self.start_date} >= {self.end_date}"
            )
        if self.early_stopping_patience <= 0:
            raise ValueError(
                f"early_stopping_patience must be positive, got {self.early_stopping_patience}"
            )
        # Validate model_type
        if self.model_type not in self._VALID_MODEL_TYPES:
            raise ValueError(
                f"model_type must be one of {sorted(self._VALID_MODEL_TYPES)}, "
                f"got '{self.model_type}'"
            )
        # Validate model_variant
        if self.model_variant not in self._VALID_MODEL_VARIANTS:
            raise ValueError(
                f"model_variant must be one of {sorted(self._VALID_MODEL_VARIANTS)}, "
                f"got '{self.model_variant}'"
            )
        # Validate target_type (membership only -- model_type/target_type
        # cross-compatibility is checked at run_training_pipeline() entry)
        if self.target_type not in self._VALID_TARGET_TYPES:
            raise ValueError(
                f"target_type must be one of {sorted(self._VALID_TARGET_TYPES)}, "
                f"got '{self.target_type}'"
            )
        if self.target_horizon <= 0:
            raise ValueError(f"target_horizon must be positive, got {self.target_horizon}")
        if self.target_type == "meta_label" and not self.primary_model_type:
            raise ValueError(
                "primary_model_type is required when target_type='meta_label' -- "
                "it names the registry model_type of the primary signal to run "
                "forward and meta-label (e.g. 'basic')."
            )

    def days_requested(self) -> int:
        """Calculate number of days in the training date range.

        Returns:
            Number of days between start_date and end_date.
            Returns negative if start_date > end_date (caught by __post_init__ validation).
        """
        return (self.end_date - self.start_date).days


@dataclass
class TrainingContext:
    """Container object that binds config and paths for the pipeline."""

    config: TrainingConfig
    paths: TrainingPaths = field(default_factory=TrainingPaths.default)

    @property
    def symbol_exchange(self) -> str:
        """Convert symbol to exchange format (e.g., BTCUSDT → BTC-USD).

        Returns:
            Symbol in exchange-specific format for data providers
        """
        from src.trading.symbols.factory import SymbolFactory

        return SymbolFactory.to_exchange_symbol(self.config.symbol, "binance")

    @property
    def start_iso(self) -> str:
        """Get start date in ISO format.

        Returns:
            Start date as ISO 8601 string (YYYY-MM-DD)
        """
        return self.config.start_date.strftime("%Y-%m-%dT00:00:00Z")

    @property
    def end_iso(self) -> str:
        """Get end date in ISO format.

        Returns:
            End date as ISO 8601 string (YYYY-MM-DD)
        """
        return self.config.end_date.strftime("%Y-%m-%dT23:59:59Z")
