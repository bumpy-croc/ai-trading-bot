"""Tests for the MFE-conditioned early-cut policy (#971 follow-up).

Every scenario uses a tiny hand-computable synthetic series so the expected
cut bar and window-MFE are derivable on paper (the #838 lesson: never trust a
number a test can't reproduce by hand).

Fixture geometry (long, entry 100.0 at 2024-01-01 00:00 UTC, 1h bars):

    bar time  high    low     favorable excursion (long)
    00:00     100.8    99.5   entry bar — excluded from the window
    01:00     101.5    99.8   +1.5%
    02:00     101.9   100.1   +1.9%
    03:00     105.0   100.0   at/after window end (3h) — excluded
    04:00     106.0   100.0   excluded

With mfe_threshold_pct=0.02 and evaluation_window_hours=3 the window MFE is
max(1.5%, 1.9%) = 1.9% < 2.0%, so the position must be cut at the first
evaluation with now >= entry + 3h.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from src.position_management.early_cut import (
    EarlyCutPolicy,
    compute_window_raw_mfe,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]

ENTRY_TIME = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
ENTRY_PRICE = 100.0


def _make_df(
    highs: list[float],
    lows: list[float],
    *,
    start: datetime = ENTRY_TIME,
    tz_aware: bool = False,
) -> pd.DataFrame:
    index = pd.date_range(
        start=start.replace(tzinfo=None) if not tz_aware else start,
        periods=len(highs),
        freq="1h",
        tz="UTC" if tz_aware else None,
    )
    closes = [(h + lo) / 2 for h, lo in zip(highs, lows)]
    return pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": 1000.0},
        index=index,
    )


def _df_below_threshold() -> pd.DataFrame:
    # Window MFE = 1.9% (< 2% threshold); bars at/after 03:00 must not count.
    return _make_df(
        highs=[100.8, 101.5, 101.9, 105.0, 106.0],
        lows=[99.5, 99.8, 100.1, 100.0, 100.0],
    )


def _df_above_threshold() -> pd.DataFrame:
    # Window MFE = 2.1% (>= 2% threshold) reached inside the window.
    return _make_df(
        highs=[100.8, 101.5, 102.1, 100.5, 100.5],
        lows=[99.5, 99.8, 100.1, 100.0, 100.0],
    )


def _policy() -> EarlyCutPolicy:
    return EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=3)


class TestEarlyCutPolicyValidation:
    def test_rejects_non_positive_threshold(self) -> None:
        with pytest.raises(ValueError):
            EarlyCutPolicy(mfe_threshold_pct=0.0, evaluation_window_hours=3)
        with pytest.raises(ValueError):
            EarlyCutPolicy(mfe_threshold_pct=-0.01, evaluation_window_hours=3)

    def test_rejects_non_positive_window(self) -> None:
        with pytest.raises(ValueError):
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=0)

    def test_rejects_non_finite_values(self) -> None:
        with pytest.raises(ValueError):
            EarlyCutPolicy(mfe_threshold_pct=float("nan"), evaluation_window_hours=3)
        with pytest.raises(ValueError):
            EarlyCutPolicy(mfe_threshold_pct=0.02, evaluation_window_hours=float("inf"))


class TestComputeWindowRawMfe:
    def test_long_mfe_from_window_highs_only(self) -> None:
        # Hand-computed: bars 01:00 (101.5) and 02:00 (101.9) are in the
        # window; the entry bar (00:00) and the 03:00/04:00 bars are not.
        mfe = compute_window_raw_mfe(
            _df_below_threshold(),
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            window_end=ENTRY_TIME + timedelta(hours=3),
        )
        assert mfe == pytest.approx((101.9 - 100.0) / 100.0)

    def test_short_mfe_from_window_lows(self) -> None:
        # Short favorable excursion uses lows: min(99.8, 100.1) = 99.8
        # => MFE = (100 - 99.8) / 100 = 0.2%.
        mfe = compute_window_raw_mfe(
            _df_below_threshold(),
            side="short",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            window_end=ENTRY_TIME + timedelta(hours=3),
        )
        assert mfe == pytest.approx((100.0 - 99.8) / 100.0)

    def test_mfe_floored_at_zero_when_never_favorable(self) -> None:
        df = _make_df(highs=[100.0, 99.5, 99.0], lows=[99.0, 98.5, 98.0])
        mfe = compute_window_raw_mfe(
            df,
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            window_end=ENTRY_TIME + timedelta(hours=3),
        )
        assert mfe == 0.0

    def test_returns_none_when_history_does_not_cover_entry(self) -> None:
        # Earliest bar is AFTER entry (e.g. live restart with a short buffer):
        # the true window MFE is unknowable, so the policy must not guess.
        df = _make_df(
            highs=[101.5, 101.9],
            lows=[99.8, 100.1],
            start=ENTRY_TIME + timedelta(hours=1),
        )
        mfe = compute_window_raw_mfe(
            df,
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            window_end=ENTRY_TIME + timedelta(hours=3),
        )
        assert mfe is None

    def test_empty_window_is_zero_mfe(self) -> None:
        # Data covers entry but no bar falls strictly inside the window.
        df = _make_df(highs=[100.8], lows=[99.5])
        mfe = compute_window_raw_mfe(
            df,
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            window_end=ENTRY_TIME + timedelta(hours=3),
        )
        assert mfe == 0.0

    def test_invalid_entry_price_returns_none(self) -> None:
        for bad in (0.0, -1.0, float("nan")):
            assert (
                compute_window_raw_mfe(
                    _df_below_threshold(),
                    side="long",
                    entry_price=bad,
                    entry_time=ENTRY_TIME,
                    window_end=ENTRY_TIME + timedelta(hours=3),
                )
                is None
            )

    def test_invalid_side_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_window_raw_mfe(
                _df_below_threshold(),
                side="sideways",
                entry_price=ENTRY_PRICE,
                entry_time=ENTRY_TIME,
                window_end=ENTRY_TIME + timedelta(hours=3),
            )

    def test_tz_aware_index_with_naive_entry_time(self) -> None:
        df = _make_df(
            highs=[100.8, 101.5, 101.9, 105.0],
            lows=[99.5, 99.8, 100.1, 100.0],
            tz_aware=True,
        )
        mfe = compute_window_raw_mfe(
            df,
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME.replace(tzinfo=None),
            window_end=ENTRY_TIME.replace(tzinfo=None) + timedelta(hours=3),
        )
        assert mfe == pytest.approx(0.019)

    def test_naive_index_with_aware_entry_time(self) -> None:
        mfe = compute_window_raw_mfe(
            _df_below_threshold(),
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            window_end=ENTRY_TIME + timedelta(hours=3),
        )
        assert mfe == pytest.approx(0.019)


class TestCheckEarlyCutConditions:
    def test_no_cut_before_window_elapses(self) -> None:
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=2),
            df=_df_below_threshold(),
        )
        assert decision.should_exit is False

    def test_cuts_at_window_end_when_mfe_below_threshold(self) -> None:
        # First evaluation at entry + 3h: window MFE 1.9% < 2.0% => cut.
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=3),
            df=_df_below_threshold(),
        )
        assert decision.should_exit is True
        assert decision.reason is not None and decision.reason.startswith("Early cut")
        assert decision.window_mfe_pct == pytest.approx(0.019)

    def test_no_cut_when_mfe_reached_threshold_inside_window(self) -> None:
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=3),
            df=_df_above_threshold(),
        )
        assert decision.should_exit is False
        assert decision.window_mfe_pct == pytest.approx(0.021)

    def test_mfe_exactly_at_threshold_is_not_cut(self) -> None:
        # Threshold semantics: cut only when MFE is strictly BELOW threshold.
        df = _make_df(
            highs=[100.8, 102.0, 101.0, 100.5],
            lows=[99.5, 99.8, 100.1, 100.0],
        )
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=3),
            df=df,
        )
        assert decision.should_exit is False

    def test_still_cuts_after_window_when_evaluated_late(self) -> None:
        # An evaluation later than window end (e.g. brief downtime) still
        # cuts — the window MFE it judges stays frozen at the first 3 hours,
        # so the 105/106 highs after the window must not rescue the trade.
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=4),
            df=_df_below_threshold(),
        )
        assert decision.should_exit is True

    def test_no_cut_when_history_insufficient(self) -> None:
        df = _make_df(
            highs=[101.5, 101.9],
            lows=[99.8, 100.1],
            start=ENTRY_TIME + timedelta(hours=1),
        )
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=3),
            df=df,
        )
        assert decision.should_exit is False
        assert decision.window_mfe_pct is None

    def test_no_cut_with_missing_df(self) -> None:
        decision = _policy().check_early_cut_conditions(
            side="long",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=3),
            df=None,
        )
        assert decision.should_exit is False

    def test_short_side_cut(self) -> None:
        # Short entry 100: window lows 99.8/100.1 => MFE 0.2% < 2% => cut.
        decision = _policy().check_early_cut_conditions(
            side="short",
            entry_price=ENTRY_PRICE,
            entry_time=ENTRY_TIME,
            now_time=ENTRY_TIME + timedelta(hours=3),
            df=_df_below_threshold(),
        )
        assert decision.should_exit is True
