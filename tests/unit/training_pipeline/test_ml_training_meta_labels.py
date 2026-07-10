"""Unit tests for meta-labeling entrant (a) scaffolding.

Preregistration §2a: label = 1 if simulating the primary signal's fired
trade through the exam-harness exit geometry closes net-profitable after
fees, else 0. Feature set: 48-bar trailing realized volatility, rolling
hit-rate of the primary signal's trailing 20 fires, session/time-of-day
cyclical encoding, EnhancedRegimeDetector's regime label (reused, not
reimplemented), and the primary model's own predicted_return magnitude as
ONE feature among the above.

Label-generation correctness gets special paranoia (per the #838 units-bug
lesson): simulate_fired_trade_profitability has a hand-computable fixture,
and every feature-builder test asserts explicit leak boundaries (a feature
at fire index t must not be affected by bars after t).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.training_pipeline.meta_labels import (
    PrimarySignalRecord,
    TradeResolution,
    build_meta_label_features,
    resolve_fired_trade,
    run_primary_signal_forward,
    simulate_fired_trade_profitability,
)
from src.strategies.components.signal_generator import Signal, SignalDirection, SignalGenerator


class _StubSignalGenerator(SignalGenerator):
    """Deterministic canned-signal generator for testing run_primary_signal_forward."""

    def __init__(self, signals_by_index: dict[int, Signal], warmup: int = 5):
        super().__init__("stub_signal_generator")
        self._signals_by_index = signals_by_index
        self._warmup = warmup

    def generate_signal(self, df, index, regime=None) -> Signal:
        return self._signals_by_index.get(
            index,
            Signal(direction=SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={}),
        )

    def get_confidence(self, df, index) -> float:
        return 0.0

    @property
    def warmup_period(self) -> int:
        return self._warmup


class TestRunPrimarySignalForward:
    def test_records_only_non_hold_fires(self):
        df = pd.DataFrame({"close": range(20)})
        signals = {
            6: Signal(
                direction=SignalDirection.BUY,
                strength=0.5,
                confidence=0.5,
                metadata={"predicted_return": 0.01},
            ),
            9: Signal(
                direction=SignalDirection.SELL,
                strength=0.5,
                confidence=0.5,
                metadata={"predicted_return": -0.02},
            ),
        }
        generator = _StubSignalGenerator(signals, warmup=5)

        records = run_primary_signal_forward(generator, df)

        assert [r.index for r in records] == [6, 9]
        assert records[0].direction == 1
        assert records[0].predicted_return == pytest.approx(0.01)
        assert records[1].direction == -1
        assert records[1].predicted_return == pytest.approx(-0.02)

    def test_starts_at_warmup_period_by_default(self):
        df = pd.DataFrame({"close": range(10)})
        signals = {
            2: Signal(
                direction=SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={}
            ),  # before warmup, must be skipped
            6: Signal(direction=SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={}),
        }
        generator = _StubSignalGenerator(signals, warmup=5)

        records = run_primary_signal_forward(generator, df)

        assert [r.index for r in records] == [6]

    def test_explicit_start_index_overrides_warmup(self):
        df = pd.DataFrame({"close": range(10)})
        signals = {
            2: Signal(direction=SignalDirection.BUY, strength=1.0, confidence=1.0, metadata={}),
        }
        generator = _StubSignalGenerator(signals, warmup=5)

        records = run_primary_signal_forward(generator, df, start_index=0)

        assert [r.index for r in records] == [2]

    def test_no_fires_returns_empty_list(self):
        df = pd.DataFrame({"close": range(10)})
        generator = _StubSignalGenerator({}, warmup=0)

        assert run_primary_signal_forward(generator, df) == []


class TestSimulateFiredTradeProfitability:
    """Hand-computable fixture: take_profit_pct=0.05, stop_loss_pct=0.03,
    max_holding_bars=3.

    idx:    0     1     2     3     4     5
    close: 100   101   103   106   104    97
    high:  100   102   104   107   105    98
    low:    99   100   102   105   103    96
    """

    @staticmethod
    def _fixture_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "close": [100.0, 101.0, 103.0, 106.0, 104.0, 97.0],
                "high": [100.0, 102.0, 104.0, 107.0, 105.0, 98.0],
                "low": [99.0, 100.0, 102.0, 105.0, 103.0, 96.0],
            }
        )

    def test_long_fire_hits_take_profit_is_profitable_net_of_fees(self):
        df = self._fixture_df()
        # fire at t=0, entry=100, TP=105, SL=97: bar3 high=107>=105 -> TP hit,
        # exit_price=105. raw_pct_return = (105-100)/100 = 0.05. Fee-rate
        # default 0.001 round-trip 0.002 -- still net profitable.
        result = simulate_fired_trade_profitability(
            df,
            fire_index=0,
            direction=1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert result is True

    def test_long_fire_hits_stop_loss_is_unprofitable(self):
        df = self._fixture_df()
        # fire at t=3, entry=106, TP=111.3, SL=102.82: bars 4,5 (truncated
        # window, only 2 bars) -> bar5 low=96<=102.82 -> SL hit.
        result = simulate_fired_trade_profitability(
            df,
            fire_index=3,
            direction=1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert result is False

    def test_short_fire_profits_when_price_falls(self):
        df = self._fixture_df()
        # fire SHORT at t=3, entry=106. Short TP = 106*(1-0.05)=100.7,
        # short SL = 106*(1+0.03)=109.18. bar4: high=105,low=103 -> no
        # touch. bar5: high=98,low=96 -> TP (low<=100.7) -> exit=100.7.
        # raw_pct_return = direction(-1) * (100.7-106)/106 = 0.05 -> profitable.
        result = simulate_fired_trade_profitability(
            df,
            fire_index=3,
            direction=-1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert result is True

    def test_vertical_barrier_exit_uses_close_of_max_holding_bar(self):
        # Flat price -> neither barrier touched -> vertical exit at close.
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
            }
        )
        result = simulate_fired_trade_profitability(
            df,
            fire_index=0,
            direction=1,
            take_profit_pct=0.5,
            stop_loss_pct=0.5,
            max_holding_bars=1,
        )
        # raw_pct_return = 0 (flat), minus fees -> unprofitable.
        assert result is False

    def test_truncated_window_without_resolution_is_unresolved(self):
        """Fire point near series end with no forward data to resolve the
        trade -- must return None (unresolved), never False or True."""
        df = self._fixture_df()
        result = simulate_fired_trade_profitability(
            df,
            fire_index=5,
            direction=1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert result is None

    def test_fee_awareness_flips_marginal_trade_to_unprofitable(self):
        """A raw-positive move smaller than the round-trip fee cost must be
        labeled unprofitable -- this is the entire point of simulating
        "net of fees," not just sign(raw return)."""
        df = pd.DataFrame(
            {
                # +0.05% move over 1 bar -- smaller than a 0.2% round-trip fee
                # (2 * DEFAULT_FEE_RATE=0.001).
                "close": [100.0, 100.05],
                "high": [100.0, 100.05],
                "low": [100.0, 100.05],
            }
        )
        result = simulate_fired_trade_profitability(
            df,
            fire_index=0,
            direction=1,
            take_profit_pct=0.5,
            stop_loss_pct=0.5,
            max_holding_bars=1,
        )
        assert result is False

    def test_rejects_invalid_direction(self):
        df = self._fixture_df()
        with pytest.raises(ValueError, match="direction"):
            simulate_fired_trade_profitability(
                df,
                fire_index=0,
                direction=0,
                take_profit_pct=0.05,
                stop_loss_pct=0.03,
                max_holding_bars=3,
            )


class TestResolveFiredTrade:
    """resolve_fired_trade() is simulate_fired_trade_profitability()'s
    superset -- same hand-computed fixture, plus the resolution bar index
    that build_meta_label_features needs to stay causal."""

    @staticmethod
    def _fixture_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "close": [100.0, 101.0, 103.0, 106.0, 104.0, 97.0],
                "high": [100.0, 102.0, 104.0, 107.0, 105.0, 98.0],
                "low": [99.0, 100.0, 102.0, 105.0, 103.0, 96.0],
            }
        )

    def test_exit_index_is_the_barrier_touch_bar(self):
        df = self._fixture_df()
        # fire at t=0: resolves at bar3 (TP touch) per the hand-computed
        # fixture in TestSimulateFiredTradeProfitability.
        resolution = resolve_fired_trade(
            df,
            fire_index=0,
            direction=1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert resolution == TradeResolution(profitable=True, exit_index=3)

    def test_exit_index_is_the_vertical_exit_bar_when_no_barrier_touched(self):
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
            }
        )
        resolution = resolve_fired_trade(
            df,
            fire_index=0,
            direction=1,
            take_profit_pct=0.5,
            stop_loss_pct=0.5,
            max_holding_bars=1,
        )
        assert resolution == TradeResolution(profitable=False, exit_index=1)

    def test_unresolved_trade_returns_none(self):
        df = self._fixture_df()
        resolution = resolve_fired_trade(
            df,
            fire_index=5,
            direction=1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert resolution is None

    def test_simulate_fired_trade_profitability_matches_resolve_fired_trade(self):
        """The thin-wrapper contract: same profitable/None outcome as the
        richer function, for every hand-computed fixture case."""
        df = self._fixture_df()
        for fire_index, direction in [(0, 1), (3, 1), (3, -1), (5, 1)]:
            resolution = resolve_fired_trade(
                df,
                fire_index=fire_index,
                direction=direction,
                take_profit_pct=0.05,
                stop_loss_pct=0.03,
                max_holding_bars=3,
            )
            simple = simulate_fired_trade_profitability(
                df,
                fire_index=fire_index,
                direction=direction,
                take_profit_pct=0.05,
                stop_loss_pct=0.03,
                max_holding_bars=3,
            )
            expected = resolution.profitable if resolution is not None else None
            assert simple == expected


class TestBuildMetaLabelFeatures:
    @staticmethod
    def _synthetic_df(periods=300):
        rng = np.random.default_rng(7)
        closes = 100.0 + np.cumsum(rng.normal(0, 0.3, periods))
        return pd.DataFrame(
            {
                "open": closes - 0.1,
                "high": closes + 0.5,
                "low": closes - 0.5,
                "close": closes,
                "volume": 1000.0 + rng.uniform(0, 50, periods),
            },
            index=pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC"),
        )

    @staticmethod
    def _resolved_next_bar(
        fires: list[PrimarySignalRecord], labels: list[bool]
    ) -> list[TradeResolution]:
        """Resolutions that resolve the bar right after each fire -- the
        simplest case, used by tests that don't care about resolution
        timing specifically (every fire here is >=2 bars apart, so "resolved
        next bar" is always eligible for any later fire's hit-rate)."""
        return [
            TradeResolution(profitable=label, exit_index=fire.index + 1)
            for fire, label in zip(fires, labels, strict=True)
        ]

    def test_returns_one_row_per_fire_with_expected_columns(self):
        df = self._synthetic_df()
        fires = [
            PrimarySignalRecord(index=100, direction=1, predicted_return=0.01),
            PrimarySignalRecord(index=120, direction=-1, predicted_return=-0.02),
            PrimarySignalRecord(index=150, direction=1, predicted_return=0.005),
        ]
        resolutions = self._resolved_next_bar(fires, [True, False, True])

        features = build_meta_label_features(df, fires, resolutions)

        assert len(features) == 3
        expected_columns = {
            "index",
            "realized_vol_48",
            "rolling_hit_rate_20",
            "session_sin",
            "session_cos",
            "regime_trend",
            "regime_volatility",
            "regime_confidence",
            "predicted_return_magnitude",
            "label",
        }
        assert expected_columns.issubset(set(features.columns))
        assert list(features["index"]) == [100, 120, 150]
        assert list(features["label"]) == [1, 0, 1]

    def test_predicted_return_magnitude_is_absolute_value(self):
        df = self._synthetic_df()
        fires = [PrimarySignalRecord(index=100, direction=-1, predicted_return=-0.03)]
        features = build_meta_label_features(df, fires, self._resolved_next_bar(fires, [True]))
        assert features["predicted_return_magnitude"].iloc[0] == pytest.approx(0.03)

    def test_cyclical_session_encoding_is_on_unit_circle(self):
        df = self._synthetic_df()
        fires = [PrimarySignalRecord(index=100, direction=1, predicted_return=0.01)]
        features = build_meta_label_features(df, fires, self._resolved_next_bar(fires, [True]))
        sin_val = features["session_sin"].iloc[0]
        cos_val = features["session_cos"].iloc[0]
        assert sin_val**2 + cos_val**2 == pytest.approx(1.0, abs=1e-9)

    def test_rolling_hit_rate_uses_only_strictly_prior_fires(self):
        """A LATER fire's label must never affect an EARLIER fire's
        hit-rate feature (list-ordering causality -- necessary but not
        sufficient; see test_rolling_hit_rate_excludes_unresolved_prior_fires
        below for the resolution-TIME causality this alone does not cover)."""
        df = self._synthetic_df()
        fires = [
            PrimarySignalRecord(index=100, direction=1, predicted_return=0.01),
            PrimarySignalRecord(index=110, direction=1, predicted_return=0.01),
            PrimarySignalRecord(index=120, direction=1, predicted_return=0.01),
        ]
        resolutions_a = self._resolved_next_bar(fires, [True, True, True])
        resolutions_b = self._resolved_next_bar(fires, [True, True, False])  # only LAST differs

        features_a = build_meta_label_features(df, fires, resolutions_a)
        features_b = build_meta_label_features(df, fires, resolutions_b)

        # First fire has zero prior fires -- hit rate must be NaN in both cases.
        assert np.isnan(features_a["rolling_hit_rate_20"].iloc[0])
        assert np.isnan(features_b["rolling_hit_rate_20"].iloc[0])
        # Second fire's hit-rate depends only on fire[0]'s label (same in
        # both) -- must be identical regardless of fire[2]'s label.
        assert features_a["rolling_hit_rate_20"].iloc[1] == pytest.approx(
            features_b["rolling_hit_rate_20"].iloc[1]
        )

    def test_rolling_hit_rate_excludes_unresolved_prior_fires(self):
        """P1 regression test: a prior fire's label must be excluded from a
        later fire's rolling_hit_rate_20 unless the prior fire had actually
        RESOLVED (exit_index <= current fire's index) by then. With
        max_holding_bars=336 and dense fires, a prior fire commonly resolves
        LONG after a nearby later fire -- using its label anyway leaks that
        prior fire's own future price path into the earlier bar's feature.
        This test fails on the pre-fix code (which used ALL strictly-prior
        fires regardless of resolution time)."""
        df = self._synthetic_df()
        fire_a = PrimarySignalRecord(index=50, direction=1, predicted_return=0.01)
        fire_b = PrimarySignalRecord(
            index=55, direction=1, predicted_return=0.01
        )  # fires before A resolves
        fire_c = PrimarySignalRecord(
            index=250, direction=1, predicted_return=0.01
        )  # fires after A resolves
        fires = [fire_a, fire_b, fire_c]

        resolution_a = TradeResolution(profitable=True, exit_index=200)  # resolves late
        resolution_b = TradeResolution(profitable=False, exit_index=56)
        resolution_c = TradeResolution(profitable=False, exit_index=251)
        resolutions = [resolution_a, resolution_b, resolution_c]

        features = build_meta_label_features(df, fires, resolutions)

        # B fires at index 55, BEFORE A resolves (exit_index=200 > 55) --
        # A must be excluded. B has no other eligible prior fires -> NaN.
        assert np.isnan(features["rolling_hit_rate_20"].iloc[1])

        # C fires at index 250, AFTER A resolves (200 <= 250) -- A IS
        # eligible, and B (exit_index=56 <= 250) is too.
        # mean([A.profitable=True, B.profitable=False]) = 0.5.
        assert features["rolling_hit_rate_20"].iloc[2] == pytest.approx(0.5)

    def test_rolling_hit_rate_hand_computed(self):
        df = self._synthetic_df()
        fires = [
            PrimarySignalRecord(index=100 + 5 * i, direction=1, predicted_return=0.01)
            for i in range(4)
        ]
        # Prior-fire labels for fire[3]'s hit-rate: fires[0..2] = [True, True, False],
        # each resolved the bar right after it fires (well before fire[3]'s
        # own index) -- all three are eligible.
        resolutions = self._resolved_next_bar(fires, [True, True, False, True])

        features = build_meta_label_features(df, fires, resolutions)

        # fire[3]'s rolling_hit_rate_20 = hit rate over its (up to 20)
        # eligible prior fires = mean([True, True, False]) = 2/3.
        assert features["rolling_hit_rate_20"].iloc[3] == pytest.approx(2.0 / 3.0)

    def test_realized_vol_feature_does_not_use_bars_after_fire_index(self):
        """Leak-boundary test: mutating bars strictly AFTER a fire's index
        must not change that fire's realized_vol_48 feature."""
        df = self._synthetic_df()
        mutated = df.copy()
        mutated.iloc[105:, mutated.columns.get_loc("close")] = 99999.0

        fires = [PrimarySignalRecord(index=100, direction=1, predicted_return=0.01)]
        resolutions = self._resolved_next_bar(fires, [True])
        features_orig = build_meta_label_features(df, fires, resolutions)
        features_mut = build_meta_label_features(mutated, fires, resolutions)

        assert features_orig["realized_vol_48"].iloc[0] == pytest.approx(
            features_mut["realized_vol_48"].iloc[0]
        )

    def test_regime_label_is_reused_from_enhanced_regime_detector(self):
        """The regime feature must come from EnhancedRegimeDetector.detect_regime
        (reused), not a hand-rolled regime computation."""
        from src.regime.detector import TrendLabel, VolLabel

        df = self._synthetic_df()
        fires = [PrimarySignalRecord(index=150, direction=1, predicted_return=0.01)]

        features = build_meta_label_features(df, fires, self._resolved_next_bar(fires, [True]))

        assert features["regime_trend"].iloc[0] in {t.value for t in TrendLabel}
        assert features["regime_volatility"].iloc[0] in {v.value for v in VolLabel}
        assert 0.0 <= features["regime_confidence"].iloc[0] <= 1.0

    def test_mismatched_fires_and_resolutions_length_raises(self):
        df = self._synthetic_df()
        fires = [PrimarySignalRecord(index=100, direction=1, predicted_return=0.01)]
        resolutions = [
            TradeResolution(profitable=True, exit_index=101),
            TradeResolution(profitable=False, exit_index=201),
        ]
        with pytest.raises(ValueError, match="length"):
            build_meta_label_features(df, fires, resolutions)
