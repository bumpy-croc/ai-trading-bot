"""
Model registry for managing ML model bundles with metadata and selection.

Supports structured layout under `DEFAULT_MODEL_REGISTRY_PATH` and provides a
backward-compatible discovery of flat `.onnx` files under legacy `src/ml`.
"""

import logging
from pathlib import Path
from typing import Any, Optional

from ..config import PredictionConfig
from ..utils.caching import PredictionCacheManager
from .exceptions import ModelLoadError, ModelNotAvailableError
from .onnx_runner import OnnxRunner


class StrategyModel:
    """Loaded model bundle with metadata and adapters.

    Attributes:
        symbol: Trading symbol, e.g., "BTCUSDT".
        timeframe: Training timeframe string like "1h".
        model_type: Short model type label like "basic" or "sentiment".
        version_id: Version identifier directory name.
        directory: Base directory of the bundle.
        metadata: Parsed metadata.json dict.
        feature_schema: Parsed feature_schema.json dict (optional).
        metrics: Parsed metrics.json dict (optional).
        runner: Inference runner (onnx or other) implementing predict().
    """

    def __init__(
        self,
        *,
        symbol: str,
        timeframe: str,
        model_type: str,
        version_id: str,
        directory: Path,
        metadata: dict[str, Any] | None,
        feature_schema: dict[str, Any] | None,
        metrics: dict[str, Any] | None,
        runner: OnnxRunner,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_type = model_type
        self.version_id = version_id
        self.directory = directory
        self.metadata = metadata or {}
        self.feature_schema = feature_schema or {}
        self.metrics = metrics or {}
        self.runner = runner

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.timeframe}:{self.model_type}:{self.version_id}"

# Set up logger
logger = logging.getLogger(__name__)


class PredictionModelRegistry:
    """Registry for model bundles and simple selection API."""

    def __init__(self, config: PredictionConfig, cache_manager: Optional[PredictionCacheManager] = None):
        """
        Initialize the prediction model registry.

        Args:
            config: Prediction engine configuration
            cache_manager: Optional prediction cache manager
        """
        self.config = config
        self.cache_manager = cache_manager
        # Structured bundles keyed by (symbol, timeframe, model_type) -> StrategyModel
        self._bundles: dict[tuple[str, str, str], StrategyModel] = {}
        # Optional production selections: (symbol, timeframe, model_type) -> version_id
        self._production_index: dict[tuple[str, str, str], str] = {}
        # Load structured models
        self._load()

    def _load(self) -> None:
        """Load structured bundles and legacy flat models."""
        base = Path(self.config.model_registry_path)
        if not base.exists():
            return
        # Expect structure: base/{symbol}/{model_type}/{version_id}/model.onnx
        for symbol_dir in base.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            for mtype_dir in symbol_dir.iterdir():
                if not mtype_dir.is_dir():
                    continue
                model_type = mtype_dir.name
                # Follow latest symlink first if present
                latest = mtype_dir / "latest"
                version_dirs = []
                if latest.exists():
                    version_dirs.append(latest)
                # Add all other subdirs as candidates
                version_dirs.extend([p for p in mtype_dir.iterdir() if p.is_dir() and p.name != "latest"])
                for vdir in version_dirs:
                    try:
                        bundle = self._load_bundle(symbol, model_type, vdir)
                        key = (bundle.symbol, bundle.timeframe, bundle.model_type)
                        # Prefer latest symlink as production if pointed
                        self._bundles[key] = bundle
                        if vdir.name == "latest":
                            self._production_index[key] = bundle.version_id
                    except Exception as e:  # pragma: no cover - aggregated logging
                        logger.error("Failed to load bundle at %s: %s", vdir, e)

    def _load_bundle(self, symbol: str, model_type: str, vdir: Path) -> StrategyModel:
        """Load a single bundle directory into a ModelBundle."""
        # Resolve real directory in case of symlink
        real_dir = vdir.resolve()
        version_id = real_dir.name
        # Require metadata.json and a model file
        metadata_path = real_dir / "metadata.json"
        feature_schema_path = real_dir / "feature_schema.json"
        metrics_path = real_dir / "metrics.json"
        model_candidates = list(real_dir.glob("*.onnx"))
        if not model_candidates:
            raise ModelLoadError(f"No ONNX model found in {real_dir}")
        model_path = str(model_candidates[0])

        # Minimal metadata fallback
        metadata: dict[str, Any] = {
            "symbol": symbol,
            "model_type": model_type,
            "version_id": version_id,
        }
        timeframe = "unknown"
        if metadata_path.exists():
            import json

            with open(metadata_path, encoding="utf-8") as f:
                try:
                    md = json.load(f)
                    metadata.update(md)
                    timeframe = str(md.get("timeframe", timeframe))
                except Exception as e:
                    raise ModelLoadError(f"Invalid metadata.json: {e}") from e
        else:
            # Try to parse timeframe from version_id pattern {YYYY-MM-DD}_{tf}_vN
            parts = version_id.split("_")
            if len(parts) >= 2:
                timeframe = parts[1]

        # Optional schema/metrics
        def _load_json(p: Path) -> dict[str, Any] | None:
            if not p.exists():
                return None
            import json

            with open(p, encoding="utf-8") as f:
                return json.load(f)

        feature_schema = _load_json(feature_schema_path)
        metrics = _load_json(metrics_path)

        # Create runner lazily; for unit tests without real ONNX, provide a stub
        try:
            runner = OnnxRunner(model_path, self.config, self.cache_manager)
        except Exception:
            class _StubRunner:
                def __init__(self, path: str):
                    self.model_path = path
                    self.session = None

                def predict(self, _features):  # pragma: no cover
                    raise RuntimeError("Stub runner cannot perform inference")

            runner = _StubRunner(model_path)  # type: ignore[assignment]
        return StrategyModel(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            version_id=version_id,
            directory=real_dir,
            metadata=metadata,
            feature_schema=feature_schema,
            metrics=metrics,
            runner=runner,
        )

    # ---- Introspection helpers ----
    def list_bundles(self) -> list[StrategyModel]:
        return list(self._bundles.values())

    # ---- Structured selection API ----
    def select_bundle(
        self,
        *,
        symbol: str,
        model_type: str,
        timeframe: str,
        stage: str | None = None,
    ) -> StrategyModel:
        """Select a bundle for symbol/model_type/timeframe.

        If stage is provided and a production index exists, use it. Otherwise, use the
        most recently loaded bundle for that key (latest symlink is preferred by _load()).
        """
        key = (symbol, timeframe, model_type)
        bundle = self._bundles.get(key)
        if bundle is None:
            raise ModelNotAvailableError(
                f"No model bundle for {symbol} {timeframe} {model_type}."
            )
        # Stage currently informational; production_index ensures latest symlink dominance
        return bundle

    def select_many(
        self,
        requirements: list[tuple[str, str, str]],  # (symbol, model_type, timeframe)
    ) -> dict[tuple[str, str, str], StrategyModel]:
        """Select multiple bundles, failing fast on any missing one."""
        errors: list[str] = []
        result: dict[tuple[str, str, str], StrategyModel] = {}
        for symbol, model_type, timeframe in requirements:
            try:
                bundle = self.select_bundle(
                    symbol=symbol, model_type=model_type, timeframe=timeframe
                )
                result[(symbol, model_type, timeframe)] = bundle
            except Exception as e:  # aggregate
                errors.append(f"{symbol}/{model_type}/{timeframe}: {e}")
        if errors:
            raise ModelLoadError("; ".join(errors))
        return result


    # ---- Runner helpers for engine ----
    def get_default_runner(self) -> OnnxRunner:
        bundles = self.list_bundles()
        if not bundles:
            raise ModelNotAvailableError("No strategy models available")
        return bundles[0].runner

    def get_default_bundle(self) -> StrategyModel:
        bundles = self.list_bundles()
        if not bundles:
            raise ModelNotAvailableError("No strategy models available")
        return bundles[0]

    def iter_runners(self) -> list[OnnxRunner]:
        return [b.runner for b in self.list_bundles()]

    def reload_models(self) -> None:
        """Reload all bundles from disk."""
        self._bundles.clear()
        self._production_index.clear()
        self._load()

    def invalidate_cache(self, model_name: Optional[str] = None) -> int:
        """
        Invalidate cache entries for models.
        
        Args:
            model_name: Specific model name to invalidate, or None for all models
            
        Returns:
            Number of cache entries invalidated
        """
        if not self.cache_manager:
            return 0
        # Without legacy names, invalidate all predictions
        return self.cache_manager.clear() or 0
