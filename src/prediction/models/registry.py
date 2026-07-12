"""Model registry for managing ML model bundles with metadata and selection."""

import logging
import threading
from pathlib import Path
from typing import Any

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

    def __init__(
        self, config: PredictionConfig, cache_manager: PredictionCacheManager | None = None
    ):
        """
        Initialize the prediction model registry.

        Args:
            config: Prediction engine configuration
            cache_manager: Optional prediction cache manager
        """
        self.config = config
        self.cache_manager = cache_manager
        # Structured bundles keyed by (symbol, timeframe, model_type) -> StrategyModel
        # ("latest wins" -- see _scan_registry)
        self._bundles: dict[tuple[str, str, str], StrategyModel] = {}
        # Optional production selections: (symbol, timeframe, model_type) -> version_id
        self._production_index: dict[tuple[str, str, str], str] = {}
        # LOADED bundles only, keyed by their EXACT
        # (symbol, timeframe, model_type, version_id). Seeded from _bundles
        # (the "latest"/fallback bundle per key -- already loaded, zero
        # extra cost) and grown lazily by get_bundle_by_key() the first time
        # a caller pins a specific non-latest version. Deliberately NOT
        # populated for every version on disk at scan time -- opening an
        # ONNX InferenceSession per version would make the live 24/7
        # process's memory grow unbounded as versions accumulate (PR #950
        # review finding). See _versioned_bundle_paths for the lightweight
        # (no-session) index of what's available to lazily load.
        self._versioned_bundles: dict[tuple[str, str, str, str], StrategyModel] = {}
        # Every on-disk version's key -> directory, indexed cheaply (metadata
        # JSON parse only, no OnnxRunner/InferenceSession) at scan time.
        # get_bundle_by_key() consults this to lazily load a version nobody
        # has pinned yet.
        self._versioned_bundle_paths: dict[tuple[str, str, str, str], Path] = {}
        # RLock for thread-safe atomic swaps during reload. RLock (re-entrant) is used
        # rather than Lock because _load and reload_models share traversal logic, and a
        # future refactoring to consolidate them could call locked methods internally.
        self._lock = threading.RLock()
        # Load structured models
        self._load()

    def _scan_registry(
        self,
    ) -> tuple[
        dict[tuple[str, str, str], StrategyModel],
        dict[tuple[str, str, str], str],
        dict[tuple[str, str, str, str], Path],
    ]:
        """Scan the registry directory, eagerly loading only "latest" bundles.

        Every concrete version directory is indexed by its
        (symbol, timeframe, model_type, version_id) key -> Path (a cheap
        metadata.json parse, no ONNX session opened) so a non-latest
        version stays discoverable for get_bundle_by_key() without paying
        the InferenceSession-per-version cost at scan time -- steady-state
        live memory must stay O(latest), not O(every version ever trained).

        Returns:
            Tuple of (bundles dict, production index dict, versioned bundle
            PATHS dict -- not loaded StrategyModel instances).
        """
        bundles: dict[tuple[str, str, str], StrategyModel] = {}
        production_index: dict[tuple[str, str, str], str] = {}
        versioned_paths: dict[tuple[str, str, str, str], Path] = {}
        base = Path(self.config.model_registry_path)
        if not base.exists():
            return bundles, production_index, versioned_paths
        # Expect structure: base/{symbol}/{model_type}/{version_id}/model.onnx
        for symbol_dir in base.iterdir():
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name
            for mtype_dir in symbol_dir.iterdir():
                if not mtype_dir.is_dir():
                    continue
                model_type = mtype_dir.name
                latest = mtype_dir / "latest"
                version_dirs = [p for p in mtype_dir.iterdir() if p.is_dir() and p.name != "latest"]
                # Deterministic order keeps logging/tests stable
                version_dirs.sort()

                # Index every concrete version's key -> path WITHOUT opening
                # its ONNX InferenceSession -- see get_bundle_by_key for the
                # lazy load this enables.
                for vdir in version_dirs:
                    try:
                        key4 = self._peek_version_key(symbol, model_type, vdir)
                    except Exception as e:  # pragma: no cover - aggregated logging
                        logger.error("Failed to index version at %s: %s", vdir, e)
                        continue
                    if key4 is not None:
                        versioned_paths[key4] = vdir

                # Eagerly load a fallback bundle (the highest-sorted version)
                # only when no `latest` symlink exists yet -- preserves prior
                # behavior for a mid-training/symlink-missing model_type dir
                # without eagerly loading every OTHER version too.
                if not latest.exists() and version_dirs:
                    fallback_dir = version_dirs[-1]
                    try:
                        bundle = self._load_bundle(symbol, model_type, fallback_dir)
                        key = (bundle.symbol, bundle.timeframe, bundle.model_type)
                        bundles[key] = bundle
                    except Exception as e:  # pragma: no cover - aggregated logging
                        logger.error("Failed to load bundle at %s: %s", fallback_dir, e)
                if latest.exists():
                    try:
                        bundle = self._load_bundle(symbol, model_type, latest)
                        key = (bundle.symbol, bundle.timeframe, bundle.model_type)
                        bundles[key] = bundle
                        production_index[key] = bundle.version_id
                    except Exception as e:  # pragma: no cover - aggregated logging
                        logger.error("Failed to load bundle at %s: %s", latest, e)
        return bundles, production_index, versioned_paths

    def _peek_version_key(
        self, symbol: str, model_type: str, vdir: Path
    ) -> tuple[str, str, str, str] | None:
        """Cheaply resolve a version directory's full StrategyModel key
        (symbol, timeframe, model_type, version_id) WITHOUT constructing an
        OnnxRunner -- indexing every on-disk version must not open an ONNX
        InferenceSession for versions nobody has pinned yet.

        Mirrors _load_bundle's timeframe-resolution logic (metadata.json's
        "timeframe" field, falling back to parsing the version_id) so the
        key matches exactly what a real (lazy) load would later produce.
        Best-effort: returns None on any error (path traversal, missing/
        corrupt directory) and lets get_bundle_by_key's actual load raise a
        proper error if that specific version is ever requested.
        """
        try:
            real_dir = vdir.resolve()
            registry_base = Path(self.config.model_registry_path).resolve()
            real_dir.relative_to(registry_base)
        except (OSError, ValueError):
            return None

        version_id = real_dir.name
        timeframe = "unknown"
        metadata_path = real_dir / "metadata.json"
        if metadata_path.exists():
            import json

            try:
                with open(metadata_path, encoding="utf-8") as f:
                    md = json.load(f)
                if isinstance(md, dict):
                    timeframe = str(md.get("timeframe", timeframe))
            except (json.JSONDecodeError, OSError):
                pass
        else:
            parts = version_id.split("_")
            if len(parts) >= 2:
                timeframe = parts[1]
        return (symbol, timeframe, model_type, version_id)

    def _load(self) -> None:
        """Load structured bundles from the configured registry path."""
        self._bundles, self._production_index, self._versioned_bundle_paths = self._scan_registry()
        # Seed _versioned_bundles from the already-loaded "latest"/fallback
        # bundles only -- zero extra cost, no new sessions opened.
        self._versioned_bundles = {
            (b.symbol, b.timeframe, b.model_type, b.version_id): b for b in self._bundles.values()
        }

    def _load_bundle(self, symbol: str, model_type: str, vdir: Path) -> StrategyModel:
        """Load a single bundle directory into a ModelBundle."""
        # Resolve real directory in case of symlink
        real_dir = vdir.resolve()

        # Validate resolved path is still within registry to prevent path traversal
        # This prevents malicious symlinks from escaping the model registry directory
        registry_base = Path(self.config.model_registry_path).resolve()
        try:
            # Raises ValueError if real_dir is not relative to registry_base
            real_dir.relative_to(registry_base)
        except ValueError as e:
            raise ModelLoadError(
                f"Path traversal detected: Model path {real_dir} is outside registry {registry_base}. "
                f"Symlinks must point to locations within the model registry."
            ) from e

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
                    # Validate metadata is a dictionary before using
                    if not isinstance(md, dict):
                        raise ModelLoadError(
                            f"Invalid metadata.json: expected dict, got {type(md).__name__}. "
                            f"File may be corrupted."
                        )
                    metadata.update(md)
                    timeframe = str(md.get("timeframe", timeframe))
                except json.JSONDecodeError as e:
                    raise ModelLoadError(f"Malformed metadata.json: {e}") from e
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
                try:
                    data = json.load(f)
                    # Validate is dictionary - silently skip if corrupted
                    if not isinstance(data, dict):
                        logger.warning(
                            "Skipping %s: expected dict, got %s", p.name, type(data).__name__
                        )
                        return None
                    return data
                except json.JSONDecodeError as e:
                    logger.warning("Skipping %s: malformed JSON - %s", p.name, e)
                    return None

        feature_schema = _load_json(feature_schema_path)
        metrics = _load_json(metrics_path)

        # Create runner lazily; for unit tests without real ONNX, provide a stub
        try:
            runner = OnnxRunner(model_path, self.config, self.cache_manager)
        except Exception as e:
            # Log the original error to aid debugging when stub runner is used
            logger.warning(
                "Failed to create OnnxRunner for %s, using stub runner: %s",
                model_path,
                e,
            )

            class _StubRunner:
                def __init__(self, path: str, error_message: str):
                    self.model_path = path
                    self.session = None
                    self._load_error_message = error_message

                def predict(self, _features):  # pragma: no cover
                    raise RuntimeError(
                        f"Model {self.model_path} failed to load - cannot perform inference. "
                        f"Original error: {self._load_error_message}"
                    )

                def close(self):  # pragma: no cover
                    pass  # No resources to release

            # Store string representation to avoid retaining traceback frames in memory
            runner = _StubRunner(model_path, str(e))  # type: ignore[assignment]
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
        """Return a snapshot of all loaded bundles.

        Thread-safe: Acquires lock so the returned list is consistent with
        any concurrent ``reload_models`` swap.
        """
        with self._lock:
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

        Thread-safe: Acquires lock so the returned bundle is from the current
        generation, preventing use-after-close if a reload happens concurrently.

        If stage is provided and a production index exists, use it. Otherwise, use the
        most recently loaded bundle for that key (latest symlink is preferred by _load()).

        Deliberately NOT version-pinnable: this method is also called
        unconditionally on the LIVE trading path (ml_signal_generator.py),
        so it always resolves "latest" with no override. Version pinning
        (e.g. the exam harness pinning a specific fold's model) goes
        through ``get_bundle_by_key`` instead, which only activates when a
        caller explicitly passes a full bundle key -- exam-only code never
        imported by the live runner.
        """
        with self._lock:
            key = (symbol, timeframe, model_type)
            bundle = self._bundles.get(key)
            if bundle is None:
                raise ModelNotAvailableError(
                    f"No model bundle for {symbol} {timeframe} {model_type}."
                )
            # Stage currently informational; production_index ensures latest symlink dominance
            return bundle

    def get_bundle_by_key(self, key: str) -> StrategyModel | None:
        """Look up any bundle -- including a non-latest version -- by its
        exact ``StrategyModel.key`` string, loading it on first use.

        Unlike ``select_bundle``/``list_bundles`` (latest-wins, one entry
        per symbol/timeframe/model_type), this can find every version ever
        trained. Used by ``PredictionEngine._resolve_bundle`` as a fallback
        when a ``model_name`` doesn't match any currently-latest bundle --
        e.g. the TARGET-REDESIGN exam harness's fold-runner pinning a
        specific version via ``f"{symbol}:{timeframe}:{model_type}:{version_id}"``.

        A non-latest version's ONNX InferenceSession is opened LAZILY, the
        first time it's actually requested here -- not at registry-scan
        time. This method is exam-only (never called on the live trading
        path, which only ever calls ``select_bundle``), so live steady-state
        memory stays O(latest) regardless of how many versions accumulate
        on disk; the lazily-loaded session is cached in ``_versioned_bundles``
        so a repeated pin for the same version doesn't reopen it.

        Thread-safe: Acquires lock so the result is from the current
        generation.
        """
        with self._lock:
            for bundle in self._versioned_bundles.values():
                if bundle.key == key:
                    return bundle
            for key4, vdir in self._versioned_bundle_paths.items():
                symbol, timeframe, model_type, version_id = key4
                if f"{symbol}:{timeframe}:{model_type}:{version_id}" != key:
                    continue
                try:
                    bundle = self._load_bundle(symbol, model_type, vdir)
                except Exception as e:
                    logger.error("Failed to lazily load pinned version at %s: %s", vdir, e)
                    return None
                self._versioned_bundles[key4] = bundle
                return bundle
            return None

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
        """Get the default model runner.

        Thread-safe: Delegates to ``list_bundles`` which acquires lock.
        """
        bundles = self.list_bundles()
        if not bundles:
            raise ModelNotAvailableError("No strategy models available")
        return bundles[0].runner

    def get_default_bundle(self) -> StrategyModel:
        """Get the default model bundle.

        Thread-safe: Delegates to ``list_bundles`` which acquires lock.
        """
        bundles = self.list_bundles()
        if not bundles:
            raise ModelNotAvailableError("No strategy models available")
        return bundles[0]

    def iter_runners(self) -> list[OnnxRunner]:
        """Return runners for all loaded bundles.

        Thread-safe: Delegates to ``list_bundles`` which acquires lock.
        """
        return [b.runner for b in self.list_bundles()]

    def reload_models(self) -> None:
        """Reload all bundles from disk with copy-on-write pattern.

        Loads new bundles into temporary dicts first, then atomically swaps
        them in. If loading fails, the existing bundles remain available so
        predictions continue working with the previous model versions.
        """
        # Load new bundles in background (outside lock to avoid blocking)
        new_bundles, new_production_index, new_versioned_paths = self._scan_registry()

        # Preserve existing bundles when reload produces empty results (e.g., transient
        # filesystem issue, NFS timeout, Docker volume unmount). This prevents a full
        # prediction outage from a temporary registry path unavailability.
        with self._lock:
            old_bundles = self._bundles
            # Everything actually LOADED under the old generation: the
            # latest/fallback bundles plus any version a caller had pinned
            # via get_bundle_by_key during this generation's lifetime (both
            # must be closed below; versions never pinned were never opened
            # in the first place, so there's nothing to close for them).
            old_versioned_bundles = self._versioned_bundles
            if not new_bundles and old_bundles:
                logger.warning(
                    "reload_models produced 0 bundles (had %d) — keeping existing bundles. "
                    "Check model_registry_path: %s",
                    len(old_bundles),
                    self.config.model_registry_path,
                )
                return
            self._bundles = new_bundles
            self._production_index = new_production_index
            self._versioned_bundle_paths = new_versioned_paths
            # Re-seed from the newly-loaded latest/fallback bundles only --
            # any previously-pinned non-latest version must be re-requested
            # (and re-loaded) through get_bundle_by_key against the new
            # generation; it is not carried forward.
            self._versioned_bundles = {
                (b.symbol, b.timeframe, b.model_type, b.version_id): b for b in new_bundles.values()
            }

        # Invalidate prediction caches after model swap to prevent stale
        # features from being served with the new model weights.
        if self.cache_manager:
            try:
                cleared = self.cache_manager.clear()
                logger.info(
                    "Cleared %d prediction cache entries after model reload",
                    cleared or 0,
                )
            except Exception as e:
                logger.warning("Failed to clear prediction cache after reload: %s", e)
        else:
            logger.warning(
                "No cache_manager available — external caches should be cleared "
                "after model reload to avoid stale predictions"
            )

        # Close old runners outside lock to avoid blocking other threads.
        # old_versioned_bundles may hold MORE StrategyModel instances than
        # old_bundles (the latest/fallback bundles plus any version a caller
        # lazily pinned via get_bundle_by_key during this generation) -- the
        # latest-resolved bundle is the SAME object in both dicts, so dedup
        # by identity to close each distinct runner exactly once (never
        # double-close, never leak a version-pinned session).
        seen_ids: set[int] = set()
        old_all_bundles: list[StrategyModel] = []
        for bundle in (*old_bundles.values(), *old_versioned_bundles.values()):
            if id(bundle) not in seen_ids:
                seen_ids.add(id(bundle))
                old_all_bundles.append(bundle)

        for bundle in old_all_bundles:
            if hasattr(bundle.runner, "close"):
                try:
                    bundle.runner.close()
                except Exception as e:
                    logger.warning(
                        "Failed to close runner for %s during reload: %s",
                        bundle.key,
                        e,
                    )

    def invalidate_cache(self, model_name: str | None = None) -> int:
        """
        Invalidate cache entries for the provided model or all models.

        If a model name is supplied, only the matching entries are removed using
        PredictionCacheManager.invalidate_model(). When *model_name* is None,
        the entire cache is cleared. The underlying cache manager returns the
        number of entries it removed, which we propagate back to callers so
        they can act on the actual number of invalidations performed.
        """

        if not self.cache_manager:
            return 0

        # Invalidate entire cache when no specific model is provided
        if model_name is None:
            cleared = self.cache_manager.clear()
            return cleared or 0

        # Attempt direct invalidation first (flat cache keys)
        invalidated = self.cache_manager.invalidate_model(model_name) or 0
        if invalidated:
            return invalidated

        # Map structured identifiers or aliases to underlying runner filenames
        target_runner_names: set[str] = set()

        for bundle in self.list_bundles():
            candidate_names: set[str] = {
                bundle.key,
                f"{bundle.symbol}:{bundle.timeframe}:{bundle.model_type}",
                bundle.version_id,
            }

            # Metadata may expose an explicit model_name
            metadata_name = bundle.metadata.get("model_name")
            if isinstance(metadata_name, str):
                candidate_names.add(metadata_name)

            # Runner path / filename also acts as an alias
            runner_path = getattr(bundle.runner, "model_path", None)
            runner_name: str | None = None
            if runner_path:
                runner_path_str = str(runner_path)
                candidate_names.add(runner_path_str)
                runner_name = Path(runner_path_str).name
                candidate_names.add(runner_name)

            if model_name in candidate_names and runner_name:
                target_runner_names.add(runner_name)

        total_invalidated = 0
        for runner_name in target_runner_names:
            total_invalidated += self.cache_manager.invalidate_model(runner_name) or 0

        return total_invalidated
