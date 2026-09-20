"""Symbol-keyed model selection and cross-symbol guards for ML signal generators.

Without these guards the prediction engine silently falls back to its default
bundle (whatever symbol happens to load first), so a strategy trading ETHUSDT
can be scored by a BTCUSDT model. Shared by every registry-backed generator so
backtest and live get identical behavior.
"""

import logging
import time
from typing import TYPE_CHECKING, Any

from src.config.feature_flags import is_enabled
from src.prediction.models.exceptions import ModelNotAvailableError

if TYPE_CHECKING:
    from src.prediction.models.registry import PredictionModelRegistry, StrategyModel

logger = logging.getLogger(__name__)


class SymbolModelGuardMixin:
    """Registry selection by symbol/type/timeframe plus mismatch guards.

    Host classes must set ``symbol``, ``model_type``, ``model_timeframe`` and
    ``_registry`` before calling :meth:`_validate_model_availability`, and call
    :meth:`_init_symbol_guard_state` during construction.
    """

    DEFAULT_SYMBOL = "BTCUSDT"

    # Feature flag that explicitly opts into scoring one symbol with another
    # symbol's model when no model exists for the trading symbol.
    CROSS_SYMBOL_FLAG = "allow_cross_symbol_model"

    # Guard logs (mismatch/substitution) are emitted at most once per interval
    # per generator instance so a 1m loop cannot flood the logs.
    SYMBOL_GUARD_LOG_INTERVAL_SECONDS = 300.0

    symbol: str
    model_type: str
    model_timeframe: str
    _registry: "PredictionModelRegistry | None"

    def _init_symbol_guard_state(self) -> None:
        # Which model symbol actually scored the last prediction, the pinned
        # substitute bundle (flag opt-in only), and per-condition rate-limit
        # clocks keyed by kind so one condition's log cannot swallow another's.
        self._model_symbol: str | None = None
        self._cross_symbol_bundle_key: str | None = None
        self._symbol_guard_log_ts: dict[str, float] = {}

    def _validate_model_availability(self) -> None:
        """Fail fast when the registry has no model for the trading symbol.

        FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true explicitly opts into scoring
        with another symbol's model as a transition path while the symbol's
        own model has not shipped yet.
        """
        if self._registry is None:
            # Engine degraded/unavailable — the prediction path already fails
            # safe (returns None -> HOLD), so nothing to validate against.
            return
        try:
            self._registry.select_bundle(
                symbol=self.symbol,
                model_type=self.model_type,
                timeframe=self.model_timeframe,
            )
            return
        except ModelNotAvailableError:
            pass
        except Exception:
            # Registry probing failure (not a missing model) — leave it to
            # the prediction path, which degrades to HOLD.
            return

        available = self._available_bundle_keys()
        if not is_enabled(self.CROSS_SYMBOL_FLAG, default=False):
            raise ModelNotAvailableError(
                f"No {self.model_type}/{self.model_timeframe} model exists for trading "
                f"symbol {self.symbol}. Available models: {', '.join(available) or 'none'}. "
                f"Train and deploy one via `atb live-control train --symbol {self.symbol}`, "
                f"or set FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true to explicitly accept scoring "
                f"{self.symbol} with another symbol's model."
            )

        fallback = self._select_cross_symbol_fallback()
        if fallback is None:
            raise ModelNotAvailableError(
                f"FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true but no "
                f"{self.model_type}/{self.model_timeframe} model exists for any symbol "
                f"to substitute for {self.symbol}. "
                f"Available models: {', '.join(available) or 'none'}."
            )
        self._cross_symbol_bundle_key = fallback.key
        self._model_symbol = fallback.symbol
        logger.critical(
            "CROSS-SYMBOL MODEL SUBSTITUTION ACTIVE: trading %s but scoring with %s "
            "model %s (FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true). Predictions are not "
            "trained on %s — no %s/%s/%s model exists in the registry. Train one via "
            "`atb live-control train --symbol %s`, deploy it, then unset the flag to "
            "remove this substitution.",
            self.symbol,
            fallback.symbol,
            fallback.key,
            self.symbol,
            self.symbol,
            self.model_type,
            self.model_timeframe,
            self.symbol,
        )

    def _available_bundle_keys(self) -> list[str]:
        """List loaded bundle keys for error messages; empty on failure."""
        if self._registry is None:
            return []
        try:
            return sorted(str(bundle.key) for bundle in self._registry.list_bundles())
        except Exception:
            return []

    def _select_cross_symbol_fallback(self) -> "StrategyModel | None":
        """Pick a deterministic same-type/timeframe bundle from another symbol.

        Prefers DEFAULT_SYMBOL, then lexicographic order, so restarts always
        substitute the same model.
        """
        if self._registry is None:
            return None
        try:
            candidates = [
                bundle
                for bundle in self._registry.list_bundles()
                if bundle.model_type == self.model_type and bundle.timeframe == self.model_timeframe
            ]
        except Exception:
            return None
        if not candidates:
            return None
        candidates.sort(key=lambda bundle: (bundle.symbol != self.DEFAULT_SYMBOL, bundle.symbol))
        return candidates[0]

    def _log_symbol_guard(self, kind: str, level: int, msg: str, *args: Any) -> None:
        """Emit a guard log, rate-limited per instance and per condition kind.

        Separate clocks per ``kind`` so distinct conditions (substitution
        warning, mismatch, bundle vanished) each announce at least once per
        window instead of suppressing one another.
        """
        now = time.monotonic()
        last = self._symbol_guard_log_ts.get(kind)
        if last is not None and now - last < self.SYMBOL_GUARD_LOG_INTERVAL_SECONDS:
            return
        self._symbol_guard_log_ts[kind] = now
        logger.log(level, msg, *args)

    @staticmethod
    def _parse_model_symbol(model_name: Any) -> str | None:
        """Extract the symbol from a bundle key like ``BTCUSDT:1h:basic:v1``."""
        if isinstance(model_name, str) and ":" in model_name:
            return model_name.split(":", 1)[0]
        return None

    def _symbol_guard_stamps(self) -> dict[str, str | None]:
        """Cross-symbol guard stamps included in every Signal's metadata.

        ``model_symbol`` is the symbol whose model scored the most recent
        prediction (None when no prediction has resolved, e.g. HOLD paths).
        """
        return {"trading_symbol": self.symbol, "model_symbol": self._model_symbol}

    def _select_registry_bundle(self) -> tuple[str | None, str | None]:
        """Resolve ``(bundle_key, bundle_symbol)`` for the current prediction.

        Raises ModelNotAvailableError (after logging) when the bundle vanished
        since startup validation, so callers fail the prediction (HOLD) instead
        of scoring another symbol's model. Returns ``(None, None)`` only when
        there is no registry at all (degraded engine).
        """
        if self._cross_symbol_bundle_key is not None:
            # Startup explicitly opted into substitution via
            # FEATURE_ALLOW_CROSS_SYMBOL_MODEL — score with the pinned bundle
            # and keep reminding the operator.
            self._log_symbol_guard(
                "cross_symbol_substitution",
                logging.WARNING,
                "Cross-symbol model substitution: scoring %s with %s model %s "
                "(FEATURE_ALLOW_CROSS_SYMBOL_MODEL=true)",
                self.symbol,
                self._model_symbol,
                self._cross_symbol_bundle_key,
            )
            return self._cross_symbol_bundle_key, self._model_symbol
        if self._registry is None:
            return None, None
        try:
            bundle = self._registry.select_bundle(
                symbol=self.symbol,
                model_type=self.model_type,
                timeframe=self.model_timeframe,
            )
        except ModelNotAvailableError:
            self._log_symbol_guard(
                "model_unavailable",
                logging.ERROR,
                "No %s/%s model available for %s at prediction time — holding "
                "(refusing cross-symbol fallback)",
                self.model_type,
                self.model_timeframe,
                self.symbol,
            )
            self._model_symbol = None
            raise
        except (KeyError, ValueError, AttributeError):
            # Deferring to the engine's default bundle could score another
            # symbol's model — the exact failure this guard exists to stop.
            self._model_symbol = None
            raise ModelNotAvailableError(
                f"Registry selection failed for {self.symbol} "
                f"{self.model_type}/{self.model_timeframe}"
            ) from None
        bundle_symbol = getattr(bundle, "symbol", None)
        return bundle.key, bundle_symbol if isinstance(bundle_symbol, str) else None

    def _record_scoring_symbol(
        self, resolved_model_symbol: str | None, result_model_name: Any, engine_model_name: Any
    ) -> None:
        """Record which symbol's model scored and log loudly on a mismatch."""
        if resolved_model_symbol is None:
            resolved_model_symbol = self._parse_model_symbol(result_model_name)
        self._model_symbol = resolved_model_symbol
        if (
            self._cross_symbol_bundle_key is None
            and resolved_model_symbol is not None
            and resolved_model_symbol != self.symbol
        ):
            self._log_symbol_guard(
                "symbol_mismatch",
                logging.ERROR,
                "MODEL/SYMBOL MISMATCH: trading %s but the resolved model bundle is "
                "for %s (model=%s). Signals are scored by a model trained on a "
                "different symbol.",
                self.symbol,
                resolved_model_symbol,
                engine_model_name,
            )
