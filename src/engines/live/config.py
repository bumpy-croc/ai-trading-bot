"""Construction-time settings resolution for the live trading engine.

``LiveEngineSettings`` captures the feature-flag / environment / app-config
reads the engine performs exactly once at construction, so the resolution
logic lives outside ``LiveTradingEngine`` and the runner can build and inject
settings explicitly (#486 step d).

Deliberately NOT captured here: flags the engine re-reads at runtime because
they may change mid-process and tests pin that dynamism —
``ws_user_hard_reconnect`` (read per reconnect decision),
``live_partial_operations`` re-checks on strategy hot-swap, and
``ENGINE_HEARTBEAT_STEPS`` (read at loop start).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass

from src.config import get_config
from src.config.constants import DEFAULT_EXECUTION_FILL_POLICY
from src.config.feature_flags import is_enabled
from src.engines.live.config_source import ConfigSource
from src.engines.shared.execution.fill_policy import FillPolicy, resolve_fill_policy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveEngineSettings:
    """Resolved construction-time settings for ``LiveTradingEngine``.

    Attributes:
        partial_operations_allowed: ``live_partial_operations`` feature flag.
            Hard gate for the #734 bookkeeping-only partial-ops hazard; the
            engine disables partial exits/scale-ins when False.
        regime_detection_enabled: ``FEATURE_ENABLE_REGIME_DETECTION`` env flag
            gating the optional ``RegimeDetector``.
        execution_fill_policy: Fill policy resolved from the
            ``EXECUTION_FILL_POLICY`` app-config key.
    """

    partial_operations_allowed: bool
    regime_detection_enabled: bool
    execution_fill_policy: FillPolicy

    def __post_init__(self) -> None:
        """Reject invalid settings at construction (CODE.md Input Validation)."""
        if self.execution_fill_policy is None:
            raise ValueError("execution_fill_policy must not be None")

    @classmethod
    def resolve(
        cls,
        *,
        flag_lookup: Callable[[str, bool], bool] = is_enabled,
        env_lookup: Callable[[str, str], str] = os.getenv,
        config_lookup: Callable[[], ConfigSource] = get_config,
    ) -> LiveEngineSettings:
        """Resolve settings from feature flags, environment, and app config.

        The source callables default to the real lookups; the engine passes
        its own module-level names so existing test patch points
        (``trading_engine.is_enabled``, ``trading_engine.get_config``) keep
        intercepting resolution.
        """
        policy_name: str | None = DEFAULT_EXECUTION_FILL_POLICY
        try:
            cfg = config_lookup()
            policy_name = cfg.get("EXECUTION_FILL_POLICY", DEFAULT_EXECUTION_FILL_POLICY)
        except Exception as exc:
            logger.warning("Failed to read execution fill policy config: %s", exc)

        return cls(
            partial_operations_allowed=flag_lookup("live_partial_operations", False),
            regime_detection_enabled=env_lookup("FEATURE_ENABLE_REGIME_DETECTION", "").lower()
            == "true",
            execution_fill_policy=resolve_fill_policy(policy_name),
        )
