"""Tests for atb backtest command."""

import argparse
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from cli.commands.backtest import (
    _get_date_range,
    _handle,
    _load_strategy,
    _resolve_model_pin,
    register,
)


class TestLoadStrategy:
    """Tests for the _load_strategy function."""

    def test_loads_ml_basic_strategy(self):
        """Test that ml_basic strategy is loaded successfully."""
        # Arrange & Act
        with patch("cli.commands.backtest.create_ml_basic_strategy") as mock_create:
            mock_strategy = Mock(name="ml_basic")
            mock_create.return_value = mock_strategy

            result = _load_strategy("ml_basic")

            # Assert
            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_loads_ml_sentiment_strategy(self):
        """Test that ml_sentiment strategy is loaded successfully."""
        # Arrange & Act
        with patch("cli.commands.backtest.create_ml_sentiment_strategy") as mock_create:
            mock_strategy = Mock(name="ml_sentiment")
            mock_create.return_value = mock_strategy

            result = _load_strategy("ml_sentiment")

            # Assert
            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_raises_system_exit_for_unknown_strategy(self):
        """Test that SystemExit is raised for unknown strategy."""
        # Arrange & Act & Assert
        with pytest.raises(SystemExit):
            _load_strategy("unknown_strategy")

    def test_raises_exception_when_strategy_creation_fails(self):
        """Test that exception is raised when strategy creation fails."""
        # Arrange & Act & Assert
        with patch("cli.commands.backtest.create_ml_basic_strategy") as mock_create:
            mock_create.side_effect = Exception("Model not found")

            with pytest.raises(Exception, match="Model not found"):
                _load_strategy("ml_basic")

    @pytest.mark.parametrize(
        ("strategy_name", "patch_target"),
        [
            (
                "exam_binary_direction",
                "cli.commands.backtest.create_exam_binary_direction_strategy",
            ),
            ("exam_triple_barrier", "cli.commands.backtest.create_exam_triple_barrier_strategy"),
            ("exam_smoothed_return", "cli.commands.backtest.create_exam_smoothed_return_strategy"),
            ("exam_meta_label", "cli.commands.backtest.create_exam_meta_label_strategy"),
        ],
    )
    def test_loads_exam_target_redesign_strategies(self, strategy_name, patch_target):
        """TARGET-REDESIGN tournament exam-only strategies (Phase 2b item 4)
        must be selectable via the backtest CLI -- this is the fold-runner's
        only entry point for entrants (b)/(c)/(d)."""
        with patch(patch_target) as mock_create:
            mock_strategy = Mock(name=strategy_name)
            mock_create.return_value = mock_strategy

            result = _load_strategy(strategy_name)

            assert result == mock_strategy
            mock_create.assert_called_once()

    def test_exam_target_redesign_strategies_are_not_selectable_by_live_runner(self):
        """Arch-review requirement: the exam harness (ConfidenceWeightedSizer
        + ratified risk-limits defaults, deliberately different from
        HyperGrowth's live wiring) must never be reachable from a live/paper
        trading session. src/engines/live/runner.py's strategy dict must not
        import or reference any exam_target_redesign factory."""
        import inspect

        from src.engines.live import runner as live_runner

        source = inspect.getsource(live_runner)
        assert "exam_target_redesign" not in source
        assert "create_exam_" not in source


class TestGetDateRange:
    """Tests for the _get_date_range function."""

    def test_uses_start_and_end_dates_when_provided(self):
        """Test that start and end dates are used when both provided."""
        # Arrange
        args = argparse.Namespace(start="2024-01-01", end="2024-12-31", days=None)

        # Act
        start_date, end_date = _get_date_range(args)

        # Assert
        assert start_date == datetime(2024, 1, 1, tzinfo=UTC)
        assert end_date == datetime(2024, 12, 31, tzinfo=UTC)

    def test_uses_start_and_now_when_only_start_provided(self):
        """Test that start date and current date are used when only start provided."""
        # Arrange
        args = argparse.Namespace(start="2024-01-01", end=None, days=None)

        # Act
        with patch("cli.commands.backtest.datetime") as mock_datetime:
            mock_now = datetime(2024, 10, 29, tzinfo=UTC)
            mock_datetime.now.return_value = mock_now
            mock_datetime.strptime = datetime.strptime

            start_date, end_date = _get_date_range(args)

            # Assert
            assert start_date == datetime(2024, 1, 1, tzinfo=UTC)
            assert end_date == mock_now

    def test_uses_days_parameter_when_provided(self):
        """Test that days parameter calculates correct date range."""
        # Arrange
        args = argparse.Namespace(start=None, end=None, days=7)

        # Act
        with patch("cli.commands.backtest.datetime") as mock_datetime:
            mock_now = datetime(2024, 10, 29)
            mock_datetime.now.return_value = mock_now
            mock_datetime_class = MagicMock()
            mock_datetime_class.now.return_value = mock_now

            start_date, end_date = _get_date_range(args)

            # Assert
            assert end_date == mock_now
            assert (end_date - start_date).days == 7

    def test_defaults_to_30_days_when_no_params_provided(self):
        """Test that defaults to 30 days when no parameters provided."""
        # Arrange
        args = argparse.Namespace(start=None, end=None, days=None)

        # Act
        with patch("cli.commands.backtest.datetime") as mock_datetime:
            mock_now = datetime(2024, 10, 29)
            mock_datetime.now.return_value = mock_now

            start_date, end_date = _get_date_range(args)

            # Assert
            assert end_date == mock_now
            assert (end_date - start_date).days == 30


class TestHandleBacktest:
    """Tests for the _handle function."""

    @pytest.fixture
    def default_args(self):
        """Provides default backtest arguments."""
        return argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            timeframe="1h",
            days=30,
            start=None,
            end=None,
            initial_balance=10000,
            risk_per_trade=0.01,
            max_risk_per_trade=0.02,
            max_drawdown=0.5,
            max_position_size=None,
            disable_engine_sl=False,
            use_sentiment=False,
            no_cache=False,
            cache_ttl=24,
            log_to_db=False,
            provider="binance",
            mock_data=False,
        )

    @pytest.fixture
    def mock_backtester(self):
        """Provides a mocked Backtester instance."""
        mock = Mock()
        mock.run.return_value = {
            "total_trades": 10,
            "win_rate": 60.0,
            "total_return": 15.5,
            "annualized_return": 25.0,
            "max_drawdown": 8.5,
            "sharpe_ratio": 1.5,
            "final_balance": 11550.0,
            "hold_return": 12.0,
            "trading_vs_hold_difference": 3.5,
            "yearly_returns": {2024: 15.5},
        }
        return mock

    def test_successful_backtest_execution(self, default_args, mock_backtester):
        """Test successful backtest execution with default parameters."""
        # Arrange
        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch(
                "src.infrastructure.runtime.cache.get_cache_ttl_for_provider",
                return_value=24,
            ),
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_load_strategy.assert_called_once_with(
                "ml_basic", symbol="BTCUSDT", model_version=None
            )
            mock_backtester.run.assert_called_once()

    def test_backtest_with_sentiment_enabled(self, default_args, mock_backtester):
        """Test backtest execution with sentiment analysis enabled."""
        # Arrange
        default_args.use_sentiment = True

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch(
                "src.data_providers.feargreed_provider.FearGreedProvider"
            ) as mock_sentiment_provider,
            patch(
                "src.infrastructure.runtime.cache.get_cache_ttl_for_provider",
                return_value=24,
            ),
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_data_provider.get_historical_data.return_value = Mock()
            mock_cached_provider.return_value = mock_data_provider

            mock_sentiment = Mock()
            mock_sentiment.get_historical_sentiment.return_value = Mock(empty=True)
            mock_sentiment_provider.return_value = mock_sentiment

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_sentiment_provider.assert_called_once()

    def test_backtest_with_cache_disabled(self, default_args, mock_backtester):
        """Test backtest execution with cache disabled."""
        # Arrange
        default_args.no_cache = True

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            # Verify CachedDataProvider was not used when no_cache is True
            mock_cached_provider.assert_not_called()

    def test_backtest_with_coinbase_provider(self, default_args, mock_backtester):
        """Test backtest execution with Coinbase provider."""
        # Arrange
        default_args.provider = "coinbase"

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch(
                "src.infrastructure.runtime.cache.get_cache_ttl_for_provider",
                return_value=24,
            ),
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTC-USD"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTC-USD"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            mock_create_provider.assert_called_once_with(provider_type="coinbase")

    def test_backtest_with_mock_data_bypasses_real_providers_entirely(
        self, default_args, mock_backtester
    ):
        """--mock-data (mirrors `atb live --mock-data`) must never construct
        a real network-backed provider or the disk cache -- this is what
        acceptance/integration tests rely on to exercise the REAL backtest
        CLI path with zero network dependency (Phase 2b item 6)."""
        default_args.mock_data = True

        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("builtins.open", create=True),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):
            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            result = _handle(default_args)

            assert result == 0
            mock_create_provider.assert_not_called()
            mock_cached_provider.assert_not_called()

    def test_returns_error_on_invalid_strategy(self, default_args):
        """Test that error is returned when strategy loading fails."""
        # Arrange
        default_args.strategy = "invalid_strategy"

        with (
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
        ):

            mock_load_strategy.side_effect = SystemExit(1)

            # Act & Assert
            with pytest.raises(SystemExit):
                _handle(default_args)

    def test_returns_error_on_backtest_exception(self, default_args):
        """Test that error is returned when backtest execution fails."""
        # Arrange
        with (
            patch("src.engines.backtest.engine.Backtester") as mock_backtester_class,
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch(
                "src.infrastructure.runtime.cache.get_cache_ttl_for_provider",
                return_value=24,
            ),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            mock_backtester_instance = Mock()
            mock_backtester_instance.run.side_effect = Exception("Backtest failed")
            mock_backtester_class.return_value = mock_backtester_instance

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 1

    def test_saves_backtest_results_to_file(self, default_args, mock_backtester):
        """Test that backtest completes successfully and attempts to save results."""
        # Arrange
        with (
            patch("src.engines.backtest.engine.Backtester", return_value=mock_backtester),
            patch("cli.commands.backtest._load_strategy") as mock_load_strategy,
            patch("cli.commands.backtest.configure_logging"),
            patch(
                "src.data_providers.provider_factory.create_data_provider"
            ) as mock_create_provider,
            patch(
                "src.data_providers.cached_data_provider.CachedDataProvider"
            ) as mock_cached_provider,
            patch(
                "src.infrastructure.runtime.cache.get_cache_ttl_for_provider",
                return_value=24,
            ),
            patch("cli.commands.backtest.SymbolFactory.to_exchange_symbol", return_value="BTCUSDT"),
            patch("cli.commands.backtest.PROJECT_ROOT", Path("/tmp/test")),
        ):

            mock_strategy = Mock(name="ml_basic")
            mock_strategy.get_trading_pair.return_value = "BTCUSDT"
            mock_load_strategy.return_value = mock_strategy

            mock_provider = Mock()
            mock_create_provider.return_value = mock_provider

            mock_data_provider = Mock()
            mock_data_provider.get_cache_info.return_value = {
                "total_files": 5,
                "total_size_mb": 10.5,
            }
            mock_cached_provider.return_value = mock_data_provider

            # Act
            result = _handle(default_args)

            # Assert
            assert result == 0
            # File saving is tested in integration tests


def _make_pin_registry_bundle(
    registry: Path, symbol: str, model_type: str, version: str, created_at: str
) -> Path:
    """Write a minimal synthetic bundle dir (dummy ONNX -> stub runner)."""
    import json

    bundle_dir = registry / symbol / model_type / version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    (bundle_dir / "model.onnx").write_bytes(b"dummy")
    (bundle_dir / "metadata.json").write_text(
        json.dumps(
            {
                "symbol": symbol,
                "model_type": model_type,
                "timeframe": "1h",
                "version_id": version,
                "created_at": created_at,
            }
        )
    )
    return bundle_dir


def _make_pin_registry(tmp_path: Path) -> tuple[Path, str, str]:
    """Synthetic BTCUSDT/basic registry: old version + newer 'latest'."""
    registry = tmp_path / "models"
    old_version = "2026-01-01_1h_v1"
    new_version = "2026-05-01_1h_v1"
    _make_pin_registry_bundle(
        registry, "BTCUSDT", "basic", old_version, "2026-01-01T00:00:00+00:00"
    )
    new_dir = _make_pin_registry_bundle(
        registry, "BTCUSDT", "basic", new_version, "2026-05-01T00:00:00+00:00"
    )
    latest = new_dir.parent / "latest"
    latest.symlink_to(new_dir, target_is_directory=True)
    return registry, old_version, new_version


class TestModelPinningArgs:
    """Argparse surface for --model-version / --model-as-of (GH #988)."""

    def _parse(self, argv):
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        register(subparsers)
        return parser.parse_args(["backtest", *argv])

    def test_model_version_parsed(self):
        ns = self._parse(["ml_basic", "--model-version", "2026-01-01_1h_v1"])
        assert ns.model_version == "2026-01-01_1h_v1"
        assert ns.model_as_of is None

    def test_model_as_of_parsed_as_utc_datetime(self):
        ns = self._parse(["ml_basic", "--model-as-of", "2026-06-02"])
        assert ns.model_as_of == datetime(2026, 6, 2, tzinfo=UTC)

    def test_defaults_are_none(self):
        ns = self._parse(["ml_basic"])
        assert ns.model_version is None
        assert ns.model_as_of is None

    def test_model_version_and_as_of_are_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            self._parse(
                [
                    "ml_basic",
                    "--model-version",
                    "2026-01-01_1h_v1",
                    "--model-as-of",
                    "2026-06-02",
                ]
            )

    def test_invalid_model_as_of_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["ml_basic", "--model-as-of", "not-a-date"])


class TestResolveModelPin:
    """Tests for _resolve_model_pin (as-of resolution + boundary warning)."""

    def _ns(self, **overrides):
        ns = argparse.Namespace(
            strategy="ml_basic",
            symbol="BTCUSDT",
            model_version=None,
            model_as_of=None,
        )
        for key, value in overrides.items():
            setattr(ns, key, value)
        return ns

    def test_returns_none_without_pin_args(self):
        result = _resolve_model_pin(
            self._ns(),
            datetime(2026, 1, 1, tzinfo=UTC),
            datetime(2026, 2, 1, tzinfo=UTC),
        )
        assert result is None

    def test_legacy_namespace_without_pin_attrs_returns_none(self):
        ns = argparse.Namespace(strategy="ml_basic", symbol="BTCUSDT")
        result = _resolve_model_pin(
            ns, datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC)
        )
        assert result is None

    def test_model_as_of_resolves_version_that_was_latest(self, tmp_path, monkeypatch):
        registry, old_version, _ = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        result = _resolve_model_pin(
            self._ns(model_as_of=datetime(2026, 3, 1, tzinfo=UTC)),
            datetime(2026, 2, 1, tzinfo=UTC),
            datetime(2026, 3, 1, tzinfo=UTC),
        )

        assert result == old_version

    def test_model_as_of_before_first_version_raises(self, tmp_path, monkeypatch):
        from src.prediction.models.exceptions import ModelNotAvailableError

        registry, _, _ = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        with pytest.raises(ModelNotAvailableError):
            _resolve_model_pin(
                self._ns(model_as_of=datetime(2025, 6, 1, tzinfo=UTC)),
                datetime(2025, 6, 1, tzinfo=UTC),
                datetime(2025, 7, 1, tzinfo=UTC),
            )

    def test_model_as_of_unsupported_strategy_exits(self, tmp_path, monkeypatch):
        registry, _, _ = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        with pytest.raises(SystemExit):
            _resolve_model_pin(
                self._ns(
                    strategy="adaptive_trend",
                    model_as_of=datetime(2026, 3, 1, tzinfo=UTC),
                ),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 3, 1, tzinfo=UTC),
            )

    def test_boundary_spanning_window_warns_with_segments(self, tmp_path, monkeypatch, caplog):
        registry, old_version, new_version = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        with caplog.at_level("WARNING", logger="atb.backtest"):
            result = _resolve_model_pin(
                self._ns(model_as_of=datetime(2026, 2, 1, tzinfo=UTC)),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 6, 1, tzinfo=UTC),
            )

        assert result == old_version
        warning = "\n".join(
            record.getMessage()
            for record in caplog.records
            if "MODEL PROMOTION BOUNDARY" in record.getMessage()
        )
        assert warning, "expected a promotion-boundary warning"
        assert old_version in warning
        assert new_version in warning

    def test_window_within_single_version_does_not_warn(self, tmp_path, monkeypatch, caplog):
        registry, old_version, _ = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        with caplog.at_level("WARNING", logger="atb.backtest"):
            result = _resolve_model_pin(
                self._ns(model_version=old_version),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 3, 1, tzinfo=UTC),
            )

        assert result == old_version
        assert not any(
            "MODEL PROMOTION BOUNDARY" in record.getMessage() for record in caplog.records
        )

    def test_explicit_model_version_also_gets_boundary_warning(self, tmp_path, monkeypatch, caplog):
        registry, old_version, new_version = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        with caplog.at_level("WARNING", logger="atb.backtest"):
            result = _resolve_model_pin(
                self._ns(model_version=old_version),
                datetime(2026, 2, 1, tzinfo=UTC),
                datetime(2026, 6, 1, tzinfo=UTC),
            )

        assert result == old_version
        assert any("MODEL PROMOTION BOUNDARY" in record.getMessage() for record in caplog.records)


class TestLoadStrategyModelVersionThreading:
    """_load_strategy must thread the pin — or refuse — never drop it."""

    def test_mainline_factory_receives_model_version(self):
        with patch("cli.commands.backtest.create_ml_basic_strategy", autospec=True) as mock_create:
            mock_create.return_value = Mock(name="pinned_strategy")

            _load_strategy("ml_basic", symbol="BTCUSDT", model_version="2026-01-01_1h_v1")

            assert mock_create.call_args.kwargs["model_version"] == "2026-01-01_1h_v1"

    def test_unpinned_call_does_not_pass_model_version(self):
        """Zero behavior change: without a pin the factory kwargs are unchanged."""
        with patch("cli.commands.backtest.create_ml_basic_strategy", autospec=True) as mock_create:
            mock_create.return_value = Mock(name="strategy")

            _load_strategy("ml_basic", symbol="BTCUSDT")

            assert "model_version" not in mock_create.call_args.kwargs

    def test_pin_with_unsupported_strategy_raises(self):
        """A factory without model_version support must refuse the pin
        rather than silently resolve 'latest'."""
        with pytest.raises(ValueError, match="model_version"):
            _load_strategy("adaptive_trend", symbol="BTCUSDT", model_version="2026-01-01_1h_v1")

    def test_exam_strategy_pin_sets_env_var_only_during_construction(self, monkeypatch):
        import os

        from src.strategies.exam_target_redesign import MODEL_VERSION_OVERRIDE_ENV_VAR

        monkeypatch.delenv(MODEL_VERSION_OVERRIDE_ENV_VAR, raising=False)
        seen: dict[str, str | None] = {}

        def capture_env(**kwargs):
            seen["override"] = os.environ.get(MODEL_VERSION_OVERRIDE_ENV_VAR)
            return Mock(name="exam_strategy")

        with patch(
            "cli.commands.backtest.create_exam_binary_direction_strategy",
            side_effect=capture_env,
        ):
            _load_strategy(
                "exam_binary_direction", symbol="BTCUSDT", model_version="2026-01-01_1h_v1"
            )

        assert seen["override"] == "2026-01-01_1h_v1"
        assert MODEL_VERSION_OVERRIDE_ENV_VAR not in os.environ

    def test_exam_strategy_pin_restores_preexisting_env_var(self, monkeypatch):
        import os

        from src.strategies.exam_target_redesign import MODEL_VERSION_OVERRIDE_ENV_VAR

        monkeypatch.setenv(MODEL_VERSION_OVERRIDE_ENV_VAR, "preexisting_v0")

        with patch(
            "cli.commands.backtest.create_exam_binary_direction_strategy",
            side_effect=lambda **kwargs: Mock(name="exam_strategy"),
        ):
            _load_strategy(
                "exam_binary_direction", symbol="BTCUSDT", model_version="2026-01-01_1h_v1"
            )

        assert os.environ[MODEL_VERSION_OVERRIDE_ENV_VAR] == "preexisting_v0"


class TestPinnedBacktestEndToEnd:
    """A pinned backtest must actually load and score with the pinned
    bundle — asserted via PredictionEngine.get_model_info (GH #988)."""

    def test_pinned_backtest_loads_pinned_bundle(self, tmp_path, monkeypatch):
        from src.data_providers.mock_data_provider import MockDataProvider
        from src.engines.backtest.engine import Backtester

        registry, old_version, _ = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))
        pinned_key = f"BTCUSDT:1h:basic:{old_version}"

        strategy = _load_strategy("ml_basic", symbol="BTCUSDT", model_version=old_version)

        assert strategy.signal_generator.pinned_model_key == pinned_key

        # Spy on the engine so the run provably scores bars with the pin —
        # get_model_info alone cannot distinguish "loaded" from "used".
        engine = strategy.signal_generator.prediction_engine
        scored_model_names: list[str | None] = []
        original_predict = engine.predict

        def recording_predict(data, model_name=None):
            scored_model_names.append(model_name)
            return original_predict(data, model_name=model_name)

        monkeypatch.setattr(engine, "predict", recording_predict)

        backtester = Backtester(
            strategy=strategy,
            data_provider=MockDataProvider(seed=42),
            initial_balance=10000,
            log_to_database=False,
        )
        backtester.run(
            symbol="BTCUSDT",
            timeframe="1h",
            start=datetime(2026, 2, 1, tzinfo=UTC),
            end=datetime(2026, 2, 11, tzinfo=UTC),
        )

        assert scored_model_names, "backtest never reached the prediction engine"
        assert set(scored_model_names) == {pinned_key}

        info = engine.get_model_info(pinned_key)
        assert info.get("name") == pinned_key
        assert info.get("loaded") is True
        assert info.get("metadata", {}).get("version_id") == old_version

    def test_unpinned_strategy_resolves_latest(self, tmp_path, monkeypatch):
        """Zero behavior change: without a pin the latest symlink wins."""
        registry, _, new_version = _make_pin_registry(tmp_path)
        monkeypatch.setenv("MODEL_REGISTRY_PATH", str(registry))

        strategy = _load_strategy("ml_basic", symbol="BTCUSDT")

        sg = strategy.signal_generator
        assert sg.pinned_model_key is None
        latest_key = f"BTCUSDT:1h:basic:{new_version}"
        info = sg.prediction_engine.get_model_info(latest_key)
        assert info.get("name") == latest_key
