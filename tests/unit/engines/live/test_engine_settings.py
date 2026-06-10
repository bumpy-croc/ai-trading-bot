"""Unit tests for LiveEngineSettings (#486 step d).

The dataclass owns the engine's construction-time feature-flag / env /
app-config resolution; the engine accepts it injected (runner path) or
resolves it with its own module-level lookups (test patch points).
"""

from unittest.mock import MagicMock, patch

import pytest

from src.engines.live.config import LiveEngineSettings
from src.engines.shared.execution.fill_policy import resolve_fill_policy

pytestmark = pytest.mark.fast


class TestResolve:
    def test_defaults_with_quiet_sources(self):
        settings = LiveEngineSettings.resolve(
            flag_lookup=lambda name, default: default,
            env_lookup=lambda key, default: default,
            config_lookup=lambda: {},  # dict satisfies the ConfigSource protocol
        )

        assert settings.partial_operations_allowed is False
        assert settings.regime_detection_enabled is False
        assert settings.execution_fill_policy == resolve_fill_policy(None)

    def test_flag_and_env_sources_are_consulted(self):
        flags = {"live_partial_operations": True}
        env = {"FEATURE_ENABLE_REGIME_DETECTION": "TRUE"}

        settings = LiveEngineSettings.resolve(
            flag_lookup=lambda name, default: flags.get(name, default),
            env_lookup=lambda key, default: env.get(key, default),
            config_lookup=lambda: {},
        )

        assert settings.partial_operations_allowed is True
        assert settings.regime_detection_enabled is True

    def test_config_read_failure_falls_back_to_default_policy(self):
        def boom():
            raise RuntimeError("config unavailable")

        settings = LiveEngineSettings.resolve(
            flag_lookup=lambda name, default: default,
            env_lookup=lambda key, default: default,
            config_lookup=boom,
        )

        assert settings.execution_fill_policy == resolve_fill_policy(None)

    def test_fill_policy_read_from_config(self):
        cfg = MagicMock()
        cfg.get.return_value = "next_open"

        settings = LiveEngineSettings.resolve(
            flag_lookup=lambda name, default: default,
            env_lookup=lambda key, default: default,
            config_lookup=lambda: cfg,
        )

        assert settings.execution_fill_policy == resolve_fill_policy("next_open")
        assert cfg.get.call_args[0][0] == "EXECUTION_FILL_POLICY"


class TestEngineInjection:
    def _make_engine(self, **kwargs):
        from src.engines.live.trading_engine import LiveTradingEngine

        with patch("src.engines.live.trading_engine.DatabaseManager"):
            return LiveTradingEngine(
                strategy=MagicMock(),
                data_provider=MagicMock(),
                initial_balance=1000.0,
                **kwargs,
            )

    def test_injected_settings_are_used_verbatim(self):
        injected = LiveEngineSettings(
            partial_operations_allowed=False,
            regime_detection_enabled=False,
            execution_fill_policy=resolve_fill_policy(None),
        )

        engine = self._make_engine(settings=injected)

        assert engine.settings is injected
        assert engine.execution_fill_policy is injected.execution_fill_policy

    def test_self_resolution_respects_trading_engine_patch_point(self):
        # The #734 gate tests patch trading_engine.is_enabled around
        # construction; self-resolution must keep honoring that.
        with patch("src.engines.live.trading_engine.is_enabled", return_value=True):
            engine = self._make_engine()

        assert engine.settings.partial_operations_allowed is True


class TestPostInitValidation:
    def test_none_fill_policy_rejected_at_construction(self):
        with pytest.raises(ValueError, match="execution_fill_policy"):
            LiveEngineSettings(
                partial_operations_allowed=False,
                regime_detection_enabled=False,
                execution_fill_policy=None,  # type: ignore[arg-type]
            )
