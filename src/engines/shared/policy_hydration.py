"""Unified policy hydration logic for trading engines.

This module provides consistent policy hydration from runtime strategy
decisions for both backtesting and live trading engines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.position_management.dynamic_risk import DynamicRiskConfig, DynamicRiskManager
    from src.position_management.partial_manager import PartialExitPolicy
    from src.position_management.trailing_stops import TrailingStopPolicy

logger = logging.getLogger(__name__)


@runtime_checkable
class PolicyBundle(Protocol):
    """Protocol for a bundle of policy descriptors from a runtime decision."""

    @property
    def partial_exit(self) -> Any | None:
        """Partial exit policy descriptor."""
        ...

    @property
    def trailing_stop(self) -> Any | None:
        """Trailing stop policy descriptor."""
        ...

    @property
    def dynamic_risk(self) -> Any | None:
        """Dynamic risk config descriptor."""
        ...


@dataclass
class HydratedPolicies:
    """Container for hydrated policies from a decision.

    Attributes:
        partial_exit_policy: Hydrated partial exit policy, if any.
        trailing_stop_policy: Hydrated trailing stop policy, if any.
        dynamic_risk_config: Hydrated dynamic risk config, if any.
        partial_exit_updated: Whether partial exit was updated.
        trailing_stop_updated: Whether trailing stop was updated.
        dynamic_risk_updated: Whether dynamic risk was updated.
    """

    partial_exit_policy: PartialExitPolicy | None = None
    trailing_stop_policy: TrailingStopPolicy | None = None
    dynamic_risk_config: DynamicRiskConfig | None = None
    partial_exit_updated: bool = False
    trailing_stop_updated: bool = False
    dynamic_risk_updated: bool = False


class PolicyHydrator:
    """Unified policy hydration from runtime decisions.

    This class provides consistent policy hydration logic that is used
    by both backtesting and live trading engines.

    Attributes:
        partial_operations_opt_in: Whether partial operations are enabled.
        trailing_stop_opt_in: Whether trailing stops are enabled.
        dynamic_risk_opt_in: Whether dynamic risk is enabled.
        log_skipped: Whether to log when policies are skipped.
    """

    def __init__(
        self,
        partial_operations_opt_in: bool = False,
        trailing_stop_opt_in: bool = False,
        dynamic_risk_opt_in: bool = False,
        log_skipped: bool = False,
    ) -> None:
        """Initialize the policy hydrator.

        Args:
            partial_operations_opt_in: Enable partial operations.
            trailing_stop_opt_in: Enable trailing stops.
            dynamic_risk_opt_in: Enable dynamic risk.
            log_skipped: Log when policies are skipped.
        """
        self.partial_operations_opt_in = partial_operations_opt_in
        self.trailing_stop_opt_in = trailing_stop_opt_in
        self.dynamic_risk_opt_in = dynamic_risk_opt_in
        self.log_skipped = log_skipped

    def hydrate_policies(
        self,
        decision: Any,
        current_partial_policy: Any | None = None,
        current_trailing_policy: Any | None = None,
        current_dynamic_config: Any | None = None,
    ) -> HydratedPolicies:
        """Hydrate policies from a runtime decision.

        Args:
            decision: Runtime decision with policies bundle.
            current_partial_policy: Current partial exit policy (for opt-in check).
            current_trailing_policy: Current trailing stop policy (for opt-in check).
            current_dynamic_config: Current dynamic risk config (for comparison).

        Returns:
            HydratedPolicies with any updated policies.
        """
        result = HydratedPolicies()

        if decision is None:
            return result

        bundle = getattr(decision, "policies", None)
        if not bundle:
            return result

        # Hydrate partial exit policy
        result.partial_exit_policy, result.partial_exit_updated = self._hydrate_partial_exit(
            bundle, current_partial_policy
        )
        if result.partial_exit_updated:
            self.partial_operations_opt_in = True

        # Hydrate trailing stop policy
        result.trailing_stop_policy, result.trailing_stop_updated = self._hydrate_trailing_stop(
            bundle, current_trailing_policy
        )
        if result.trailing_stop_updated:
            self.trailing_stop_opt_in = True

        # Hydrate dynamic risk config
        result.dynamic_risk_config, result.dynamic_risk_updated = self._hydrate_dynamic_risk(
            bundle, current_dynamic_config
        )
        if result.dynamic_risk_updated:
            self.dynamic_risk_opt_in = True

        return result

    def _hydrate_partial_exit(
        self,
        bundle: Any,
        current_policy: Any | None,
    ) -> tuple[Any | None, bool]:
        """Hydrate partial exit policy from bundle.

        Args:
            bundle: Policy bundle from decision.
            current_policy: Current policy for opt-in check.

        Returns:
            Tuple of (policy, was_updated).
        """
        try:
            descriptor = getattr(bundle, "partial_exit", None)
            if descriptor is not None:
                if self.partial_operations_opt_in or current_policy is not None:
                    return descriptor.to_policy(), True
                elif self.log_skipped:
                    logger.debug(
                        "Skipping partial-exit policy: partial operations disabled"
                    )
        except Exception as e:
            logger.debug("Failed to hydrate partial-exit policy: %s", e)

        return None, False

    def _hydrate_trailing_stop(
        self,
        bundle: Any,
        current_policy: Any | None,
    ) -> tuple[Any | None, bool]:
        """Hydrate trailing stop policy from bundle.

        Args:
            bundle: Policy bundle from decision.
            current_policy: Current policy for opt-in check.

        Returns:
            Tuple of (policy, was_updated).
        """
        try:
            descriptor = getattr(bundle, "trailing_stop", None)
            if descriptor is not None:
                if self.trailing_stop_opt_in or current_policy is not None:
                    return descriptor.to_policy(), True
                elif self.log_skipped:
                    logger.debug(
                        "Skipping trailing-stop policy: trailing stops disabled"
                    )
        except Exception as e:
            logger.debug("Failed to hydrate trailing-stop policy: %s", e)

        return None, False

    def _hydrate_dynamic_risk(
        self,
        bundle: Any,
        current_config: Any | None,
    ) -> tuple[Any | None, bool]:
        """Hydrate dynamic risk config from bundle.

        Args:
            bundle: Policy bundle from decision.
            current_config: Current config for comparison.

        Returns:
            Tuple of (config, was_updated).
        """
        try:
            descriptor = getattr(bundle, "dynamic_risk", None)
            if descriptor is not None:
                config = descriptor.to_config()
                # Always return the config, let caller decide if manager needs update
                if current_config is None or current_config != config:
                    return config, True
                return config, False
        except Exception as e:
            logger.debug("Failed to hydrate dynamic risk config: %s", e)

        return None, False


def apply_policies_to_engine(
    decision: Any,
    engine: Any,
    db_manager: Any | None = None,
) -> None:
    """Apply hydrated policies to an engine instance.

    This is a convenience function that applies policies directly
    to engine attributes.

    Args:
        decision: Runtime decision with policies bundle.
        engine: Engine instance (Backtester or LiveTradingEngine).
        db_manager: Database manager for dynamic risk manager creation.
    """
    if decision is None:
        return

    bundle = getattr(decision, "policies", None)
    if not bundle:
        return

    # Partial exit policy
    try:
        partial_descriptor = getattr(bundle, "partial_exit", None)
        if partial_descriptor is not None:
            partial_opt_in = getattr(engine, "_partial_operations_opt_in", False)
            partial_manager = getattr(engine, "partial_manager", None)
            if partial_opt_in or partial_manager is not None:
                engine.partial_manager = partial_descriptor.to_policy()
                engine._partial_operations_opt_in = True
                # Update exit handler if present
                exit_handler = getattr(engine, "exit_handler", None)
                if exit_handler:
                    exit_handler.partial_manager = engine.partial_manager
    except Exception as e:
        logger.debug("Failed to hydrate partial-exit policy: %s", e)

    # Trailing stop policy
    try:
        trailing_descriptor = getattr(bundle, "trailing_stop", None)
        if trailing_descriptor is not None:
            trailing_opt_in = getattr(engine, "_trailing_stop_opt_in", False)
            trailing_policy = getattr(engine, "trailing_stop_policy", None)
            if trailing_opt_in or trailing_policy is not None:
                engine.trailing_stop_policy = trailing_descriptor.to_policy()
                engine._trailing_stop_opt_in = True
                # Update exit handler if present
                exit_handler = getattr(engine, "exit_handler", None)
                if exit_handler:
                    exit_handler.trailing_stop_policy = engine.trailing_stop_policy
    except Exception as e:
        logger.debug("Failed to hydrate trailing-stop policy: %s", e)

    # Dynamic risk config
    try:
        dynamic_descriptor = getattr(bundle, "dynamic_risk", None)
        if dynamic_descriptor is not None:
            from src.position_management.dynamic_risk import DynamicRiskManager

            config = dynamic_descriptor.to_config()
            if not getattr(engine, "enable_dynamic_risk", False):
                engine.enable_dynamic_risk = True

            manager = getattr(engine, "dynamic_risk_manager", None)
            should_create = manager is None or getattr(manager, "config", None) != config

            if should_create and db_manager is not None:
                engine.dynamic_risk_manager = DynamicRiskManager(
                    config=config, db_manager=db_manager
                )
                # Update entry handler if present
                entry_handler = getattr(engine, "entry_handler", None)
                if entry_handler:
                    entry_handler.dynamic_risk_manager = engine.dynamic_risk_manager
    except Exception as e:
        logger.debug("Failed to hydrate dynamic risk config: %s", e)


__all__ = [
    "PolicyHydrator",
    "HydratedPolicies",
    "apply_policies_to_engine",
]
