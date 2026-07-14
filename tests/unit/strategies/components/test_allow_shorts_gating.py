"""Entry-only short gating in MLBasicSignalGenerator (GH #1020).

``allow_shorts=False`` must suppress SHORT *entries* at signal generation by
withholding the ``enter_short`` opt-in — and nothing else:

- the SELL direction itself survives, so signal-reversal exits of longs are
  ungated (risk-review condition C2);
- BUY signals (which close existing shorts via reversal) are untouched, so an
  open short (e.g. live position #22) can always be covered;
- the shared ``extract_entry_plan`` chokepoint used by BOTH engines converts
  the withheld opt-in into "no entry", giving backtest-live parity for free.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.engines.shared.entry_utils import extract_entry_plan
from src.engines.shared.models import PositionSide
from src.engines.shared.strategy_exit_checker import StrategyExitChecker
from src.prediction import PredictionResult
from src.strategies.components import SignalDirection
from src.strategies.components.ml_signal_generator import (
    SHORT_ENTRY_SUPPRESSED_KEY,
    MLBasicSignalGenerator,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast, pytest.mark.mock_only]

ENGINE_PATH = "src.strategies.components.ml_signal_generator.PredictionEngine"
CONFIG_PATH = "src.strategies.components.ml_signal_generator.PredictionConfig"


def _test_dataframe(length: int = 150) -> pd.DataFrame:
    """Deterministic OHLCV frame long enough for the 120-bar sequence window."""
    dates = pd.date_range("2023-01-01", periods=length, freq="1h")
    rng = np.random.default_rng(42)
    prices = 50000.0 * np.cumprod(1 + rng.normal(0, 0.002, length))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices,
            "volume": rng.uniform(1000, 10000, length),
        },
        index=dates,
    )


def _generator_with_prediction(
    mock_engine_class: MagicMock,
    df: pd.DataFrame,
    *,
    price_factor: float,
    allow_shorts: bool | None = None,
) -> MLBasicSignalGenerator:
    """Build a generator whose mocked engine predicts close * price_factor."""

    def _predict(window_df: pd.DataFrame, model_name: Any = None) -> Mock:
        result = Mock(spec=PredictionResult)
        result.error = None
        result.metadata = {}
        result.price = float(window_df["close"].iloc[-1]) * price_factor
        return result

    mock_engine = MagicMock()
    mock_engine.predict.side_effect = _predict
    mock_engine.health_check.return_value = {"status": "healthy"}
    mock_engine_class.return_value = mock_engine

    kwargs: dict[str, Any] = {"sequence_length": 120}
    if allow_shorts is not None:
        kwargs["allow_shorts"] = allow_shorts
    return MLBasicSignalGenerator(**kwargs)


class TestSignalLevelGating:
    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_sell_signal_sets_enter_short_by_default(self, _cfg, engine_cls):
        """Default allow_shorts=True keeps the pre-#1020 behavior exactly."""
        df = _test_dataframe()
        generator = _generator_with_prediction(engine_cls, df, price_factor=0.99)

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.SELL
        assert signal.metadata["enter_short"] is True
        assert SHORT_ENTRY_SUPPRESSED_KEY not in signal.metadata

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_allow_shorts_false_withholds_enter_short(self, _cfg, engine_cls):
        """Suppression withholds the entry opt-in but keeps the SELL direction."""
        df = _test_dataframe()
        generator = _generator_with_prediction(
            engine_cls, df, price_factor=0.99, allow_shorts=False
        )

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.SELL
        assert "enter_short" not in signal.metadata
        assert signal.metadata[SHORT_ENTRY_SUPPRESSED_KEY] is True

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_allow_shorts_false_leaves_buy_signals_untouched(self, _cfg, engine_cls):
        """Longs are unaffected: no opt-in, no suppression marker on BUY."""
        df = _test_dataframe()
        generator = _generator_with_prediction(
            engine_cls, df, price_factor=1.01, allow_shorts=False
        )

        signal = generator.generate_signal(df, 130)

        assert signal.direction == SignalDirection.BUY
        assert "enter_short" not in signal.metadata
        assert SHORT_ENTRY_SUPPRESSED_KEY not in signal.metadata

    @patch(ENGINE_PATH)
    @patch(CONFIG_PATH)
    def test_get_parameters_reports_allow_shorts(self, _cfg, engine_cls):
        """The flag is traceable in serialized generator parameters."""
        mock_engine = MagicMock()
        mock_engine.health_check.return_value = {"status": "healthy"}
        engine_cls.return_value = mock_engine

        generator = MLBasicSignalGenerator(sequence_length=120, allow_shorts=False)

        assert generator.get_parameters()["allow_shorts"] is False


@dataclass
class _FakeSignal:
    direction: SignalDirection
    strength: float = 1.0
    confidence: float = 0.9
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FakeDecision:
    signal: _FakeSignal
    position_size: float
    metadata: dict[str, Any] | None
    regime: Any = None


class TestSharedEntryPlanChokepoint:
    """Both engines resolve entries through extract_entry_plan — the parity gate."""

    def test_sell_without_enter_short_yields_no_entry_plan(self):
        decision = _FakeDecision(
            signal=_FakeSignal(direction=SignalDirection.SELL),
            position_size=100.0,
            metadata={"enter_short": False},
        )

        assert extract_entry_plan(decision, balance=1000.0) is None

    def test_sell_with_enter_short_yields_short_plan(self):
        decision = _FakeDecision(
            signal=_FakeSignal(direction=SignalDirection.SELL),
            position_size=100.0,
            metadata={"enter_short": True},
        )

        plan = extract_entry_plan(decision, balance=1000.0)

        assert plan is not None
        assert plan.side == PositionSide.SHORT


@dataclass
class _FakePosition:
    symbol: str
    side: str
    entry_price: float
    entry_time: datetime
    entry_balance: float = 1000.0
    current_size: float = 0.1
    size: float = 0.1


class TestExitsUngated:
    """C2: exits, stops, and BUY-to-cover are never gated by allow_shorts."""

    def _decision(self, direction: SignalDirection, signal_metadata: dict[str, Any]):
        return _FakeDecision(
            signal=_FakeSignal(direction=direction, metadata=signal_metadata),
            position_size=0.0,
            metadata={},  # no ignore_signal_reversal -> reversal exits active
        )

    def test_buy_signal_still_exits_short_position(self):
        """BUY-to-cover of an existing short must remain unconditional."""
        checker = StrategyExitChecker()
        position = _FakePosition(
            symbol="ETHUSDT", side="short", entry_price=2000.0, entry_time=datetime.now(UTC)
        )
        decision = self._decision(SignalDirection.BUY, {})

        result = checker.check_exit(
            position=position,
            current_price=1900.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"

    def test_suppressed_sell_still_exits_long_position(self):
        """The retained SELL direction keeps long reversal exits working."""
        checker = StrategyExitChecker()
        position = _FakePosition(
            symbol="ETHUSDT", side="long", entry_price=2000.0, entry_time=datetime.now(UTC)
        )
        decision = self._decision(SignalDirection.SELL, {SHORT_ENTRY_SUPPRESSED_KEY: True})

        result = checker.check_exit(
            position=position,
            current_price=2100.0,
            runtime_decision=decision,
            component_strategy=None,
        )

        assert result.should_exit is True
        assert result.exit_reason == "Signal reversal"
