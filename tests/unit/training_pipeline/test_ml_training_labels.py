"""Unit tests for alternative training-target label generation.

Covers the TARGET-REDESIGN tournament entrants (b) binary direction,
(c) triple-barrier ternary, (d) smoothed forward return — per
docs/research/experiments/2026-07-10_target-redesign-tournament-prereg.md §2/§8.

Label-generation correctness gets special paranoia here (per the #838
partial-exit units-bug lesson): every function has a hand-computable fixture,
and lookahead boundaries are explicitly tested — a label at bar t must never
be consumable as a feature at bar t, and the trailing bars that lack a full
forward horizon must be marked invalid, not silently zero-filled.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.training_pipeline.labels import (
    LabelResult,
    binary_direction_labels,
    smoothed_forward_return_labels,
    target_horizon_bars,
    triple_barrier_labels,
)


class TestBinaryDirectionLabels:
    def test_hand_computable_horizon_1(self):
        close = pd.Series([100.0, 101.0, 99.0, 99.0, 105.0])
        result = binary_direction_labels(close, horizon=1)

        assert isinstance(result, LabelResult)
        assert result.horizon_bars == 1
        # t=0: close[1]=101 > close[0]=100 -> up (1)
        # t=1: close[2]=99 > close[1]=101? no -> down (0)
        # t=2: close[3]=99 > close[2]=99? no (equal) -> down (0)
        # t=3: close[4]=105 > close[3]=99 -> up (1)
        # t=4: no forward bar -> invalid
        np.testing.assert_array_equal(result.values[:4], [1, 0, 0, 1])
        np.testing.assert_array_equal(result.valid_mask, [True, True, True, True, False])

    def test_hand_computable_horizon_2(self):
        close = pd.Series([100.0, 101.0, 99.0, 105.0, 90.0])
        result = binary_direction_labels(close, horizon=2)

        assert result.horizon_bars == 2
        # t=0: close[2]=99 vs close[0]=100 -> down (0)
        # t=1: close[3]=105 vs close[1]=101 -> up (1)
        # t=2, t=3, t=4: no bar at t+2 within range for t=2 (close[4]=90 vs close[2]=99 -> down(0))
        # t=3, t=4: invalid (t+2 out of range)
        np.testing.assert_array_equal(result.values[:3], [0, 1, 0])
        np.testing.assert_array_equal(result.valid_mask, [True, True, True, False, False])

    def test_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            binary_direction_labels(pd.Series([100.0, 101.0]), horizon=0)

    def test_no_lookahead_label_at_t_unaffected_by_future_beyond_horizon(self):
        """Changing a bar strictly beyond t+horizon must not change label[t]."""
        base = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        mutated = base.copy()
        mutated.iloc[4] = 1.0  # crash the far-future bar
        r1 = binary_direction_labels(base, horizon=1)
        r2 = binary_direction_labels(mutated, horizon=1)
        # label[0] only depends on close[0], close[1] -- unaffected by index 4
        assert r1.values[0] == r2.values[0]
        assert r1.valid_mask[0] == r2.valid_mask[0]


class TestSmoothedForwardReturnLabels:
    def test_hand_computable_horizon_2(self):
        close = pd.Series([100.0, 102.0, 104.0, 103.0, 105.0])
        result = smoothed_forward_return_labels(close, horizon=2)

        assert result.horizon_bars == 2
        # t=0: mean(close[1],close[2]) = 103 -> (103/100 - 1) = 0.03
        # t=1: mean(close[2],close[3]) = 103.5 -> (103.5/102 - 1) = 0.0147058823...
        # t=2: mean(close[3],close[4]) = 104 -> (104/104 - 1) = 0.0
        assert result.values[0] == pytest.approx(0.03, abs=1e-9)
        assert result.values[1] == pytest.approx(0.014705882352941176, abs=1e-9)
        assert result.values[2] == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_array_equal(result.valid_mask, [True, True, True, False, False])

    def test_horizon_1_matches_single_bar_return(self):
        close = pd.Series([100.0, 110.0, 90.0])
        result = smoothed_forward_return_labels(close, horizon=1)
        assert result.values[0] == pytest.approx(0.10, abs=1e-9)
        assert result.values[1] == pytest.approx(-0.181818181818, abs=1e-9)
        np.testing.assert_array_equal(result.valid_mask, [True, True, False])

    def test_rejects_non_positive_horizon(self):
        with pytest.raises(ValueError, match="horizon"):
            smoothed_forward_return_labels(pd.Series([100.0, 101.0]), horizon=0)

    def test_invalid_rows_are_nan(self):
        close = pd.Series([100.0, 101.0, 102.0])
        result = smoothed_forward_return_labels(close, horizon=2)
        assert np.isnan(result.values[1])
        assert np.isnan(result.values[2])

    def test_zero_divisor_marks_row_invalid_not_zero_or_inf(self):
        """PR #948 review finding (claude[bot]): close_arr[idx]==0 at the
        divisor must not silently produce inf/nan written into the label
        array as if it were a valid, fully-realized row -- the row must be
        marked invalid (valid_mask=False), matching this module's own
        'unresolved is not the same as zero' convention used everywhere else."""
        close = pd.Series([0.0, 101.0, 103.0])
        result = smoothed_forward_return_labels(close, horizon=1)
        assert result.valid_mask[0] == False  # noqa: E712 -- explicit bool check
        assert np.isnan(result.values[0])

    def test_negative_divisor_marks_row_invalid(self):
        close = pd.Series([-5.0, 101.0, 103.0])
        result = smoothed_forward_return_labels(close, horizon=1)
        assert result.valid_mask[0] == False  # noqa: E712
        assert np.isnan(result.values[0])

    def test_nan_divisor_marks_row_invalid(self):
        close = pd.Series([float("nan"), 101.0, 103.0])
        result = smoothed_forward_return_labels(close, horizon=1)
        assert result.valid_mask[0] == False  # noqa: E712
        assert np.isnan(result.values[0])

    def test_bad_divisor_does_not_affect_other_rows(self):
        """A single corrupted close must not poison neighboring rows -- only
        the row whose OWN entry price is the bad divisor is marked invalid."""
        close = pd.Series([100.0, 0.0, 103.0, 105.0])
        result = smoothed_forward_return_labels(close, horizon=1)
        # t=0: divisor is close[0]=100 (fine) -- valid even though the
        # FUTURE close it reads happens to be zero (a legitimate, if
        # unusual, forward value -- only the divisor, close[t] itself, is
        # guarded).
        assert result.valid_mask[0]
        # t=1: divisor is close[1]=0.0 -- invalid.
        assert result.valid_mask[1] == False  # noqa: E712
        # t=2: divisor is close[2]=103.0 -- valid, unaffected by t=1's bad row.
        assert result.valid_mask[2]
        assert result.values[2] == pytest.approx(105.0 / 103.0 - 1.0, abs=1e-9)

    def test_no_runtime_warning_on_zero_divisor(self):
        """Guards against a numpy RuntimeWarning (divide by zero / invalid
        value) leaking from the vectorized computation -- the fix must
        avoid dividing by the bad values at all, not divide-then-discard."""
        import warnings

        close = pd.Series([0.0, 101.0, 103.0])
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            smoothed_forward_return_labels(close, horizon=1)


class TestTripleBarrierLabels:
    """Hand-computable fixture (per task instructions): a tiny synthetic OHLC
    series where the correct label at every bar is derivable on paper.

    take_profit_pct=0.05, stop_loss_pct=0.03, max_holding_bars=3.

    idx:    0     1     2     3     4     5
    close: 100   101   103   106   104   100
    high:  100   102   104   107   105   101
    low:    99   100   102   105   103    99
    """

    @staticmethod
    def _fixture_df() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "close": [100.0, 101.0, 103.0, 106.0, 104.0, 100.0],
                "high": [100.0, 102.0, 104.0, 107.0, 105.0, 101.0],
                "low": [99.0, 100.0, 102.0, 105.0, 103.0, 99.0],
            }
        )

    def test_hand_computed_labels(self):
        df = self._fixture_df()
        result = triple_barrier_labels(
            df, take_profit_pct=0.05, stop_loss_pct=0.03, max_holding_bars=3
        )

        assert result.horizon_bars == 3
        # t=0: entry=100, TP=105, SL=97. bars 1,2,3 -> bar3 high=107>=105 -> TP (+1)
        # t=1: entry=101, TP=106.05, SL=97.97. bars 2,3,4 -> bar3 high=107>=106.05 -> TP (+1)
        # t=2: entry=103, TP=108.15, SL=99.91. bars 3,4,5 -> bar5 low=99<=99.91 -> SL (-1)
        # t=3: entry=106, TP=111.3, SL=102.82. bars 4,5 (only 2, truncated window)
        #      -> bar5 low=99<=102.82 -> SL (-1), still VALID (resolved within window)
        # t=4: entry=104, TP=109.2, SL=100.88. bar 5 only -> low=99<=100.88 -> SL (-1), valid
        # t=5: last bar, no forward data at all -> invalid
        np.testing.assert_array_equal(result.values[:5], [1, 1, -1, -1, -1])
        np.testing.assert_array_equal(result.valid_mask, [True, True, True, True, True, False])

    def test_vertical_barrier_when_neither_touched(self):
        """Flat price path within the full window -> label 0 (vertical/time barrier)."""
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0, 100.0],
                "high": [100.0, 100.0, 100.0],
                "low": [100.0, 100.0, 100.0],
            }
        )
        result = triple_barrier_labels(
            df, take_profit_pct=0.5, stop_loss_pct=0.5, max_holding_bars=1
        )
        assert result.values[0] == 0
        assert result.valid_mask[0] is np.True_ or result.valid_mask[0] is True

    def test_stop_loss_priority_when_both_touched_same_bar(self):
        """Same-bar SL+TP touch resolves to SL (matches the ExitHandler /
        check_barrier_touch money-path convention: conservative tie-break)."""
        df = pd.DataFrame(
            {
                "close": [100.0, 100.0],
                "high": [200.0, 200.0],  # far above any take-profit
                "low": [1.0, 1.0],  # far below any stop-loss
            }
        )
        result = triple_barrier_labels(
            df, take_profit_pct=0.05, stop_loss_pct=0.05, max_holding_bars=1
        )
        assert result.values[0] == -1
        assert result.valid_mask[0]

    def test_no_lookahead_beyond_resolution_bar(self):
        """Once a barrier resolves the label, mutating bars strictly after the
        resolution bar must not change the label (a label at t must not be
        sensitive to bars beyond what actually determined the outcome)."""
        df = self._fixture_df()
        mutated = df.copy()
        mutated.loc[5, "high"] = 1000.0  # crash a bar that resolves nothing new for t=0/t=1
        r1 = triple_barrier_labels(df, take_profit_pct=0.05, stop_loss_pct=0.03, max_holding_bars=3)
        r2 = triple_barrier_labels(
            mutated, take_profit_pct=0.05, stop_loss_pct=0.03, max_holding_bars=3
        )
        # t=0 and t=1 both resolve at bar 3 (TP), strictly before bar 5 -- unaffected.
        assert r1.values[0] == r2.values[0]
        assert r1.values[1] == r2.values[1]

    def test_rejects_missing_columns(self):
        df = pd.DataFrame({"close": [100.0, 101.0]})
        with pytest.raises(ValueError, match="high|low"):
            triple_barrier_labels(df, take_profit_pct=0.05, stop_loss_pct=0.03, max_holding_bars=1)

    @pytest.mark.parametrize(
        ("take_profit_pct", "stop_loss_pct", "max_holding_bars"),
        [(0.0, 0.05, 1), (0.05, 0.0, 1), (0.05, 0.05, 0)],
    )
    def test_rejects_non_positive_params(self, take_profit_pct, stop_loss_pct, max_holding_bars):
        df = pd.DataFrame({"close": [100.0], "high": [100.0], "low": [100.0]})
        with pytest.raises(ValueError):
            triple_barrier_labels(
                df,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                max_holding_bars=max_holding_bars,
            )


class TestTargetHorizonBars:
    def test_binary_direction(self):
        assert target_horizon_bars("binary_direction", horizon=6) == 6

    def test_smoothed_return(self):
        assert target_horizon_bars("smoothed_return", horizon=6) == 6

    def test_triple_barrier(self):
        assert target_horizon_bars("triple_barrier", max_holding_bars=336) == 336

    def test_missing_required_param_raises(self):
        with pytest.raises(ValueError):
            target_horizon_bars("binary_direction")

    def test_unknown_target_type_raises(self):
        with pytest.raises(ValueError, match="target_type"):
            target_horizon_bars("not_a_real_target")
