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

import json

import numpy as np
import pandas as pd
import pytest

from src.engines.shared.cost_calculator import CostCalculator
from src.ml.training_pipeline.meta_labels import (
    META_LABEL_FEATURE_ORDER,
    PrimarySignalRecord,
    TradeResolution,
    build_meta_label_features,
    encode_meta_label_feature_row,
    encode_meta_label_features_for_training,
    resolution_ordered_hit_rate,
    resolve_fired_trade,
    run_primary_signal_forward,
    simulate_fired_trade_profitability,
)
from src.regime.enhanced_detector import EnhancedRegimeDetector
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


class TestRunPrimarySignalForwardCheckpointing:
    """Defect 2 of #955: a ~47k-bar fire-generation pass ran 60+ min in one
    uninterruptible pass. run_primary_signal_forward now supports a
    caller-supplied checkpoint file: progress (fires so far + last completed
    bar) is persisted every N bars via an atomic write, and a rerun with the
    same checkpoint resumes after the last completed bar instead of
    reprocessing from the start."""

    @staticmethod
    def _make_df(periods: int = 40) -> pd.DataFrame:
        return pd.DataFrame({"close": np.linspace(100.0, 110.0, periods)})

    class _ScriptedSignalGenerator(SignalGenerator):
        """Deterministic fires; records every bar index it is asked to
        evaluate (the counting stub the resume tests assert against), and
        optionally simulates an interruption at a given bar."""

        def __init__(self, warmup: int = 5, fail_at: int | None = None):
            super().__init__("scripted_signal_generator")
            self.evaluated: list[int] = []
            self._warmup = warmup
            self._fail_at = fail_at

        def generate_signal(self, df, index, regime=None) -> Signal:
            if self._fail_at is not None and index >= self._fail_at:
                raise RuntimeError("simulated interruption")
            self.evaluated.append(index)
            if index % 3 == 0:
                direction = SignalDirection.BUY if index % 2 == 0 else SignalDirection.SELL
                return Signal(
                    direction=direction,
                    strength=0.5,
                    confidence=0.5,
                    metadata={"predicted_return": index / 1000.0},
                )
            return Signal(direction=SignalDirection.HOLD, strength=0.0, confidence=0.0, metadata={})

        def get_confidence(self, df, index) -> float:
            return 0.0

        @property
        def warmup_period(self) -> int:
            return self._warmup

    def test_checkpointed_run_matches_single_pass_output(self, tmp_path):
        df = self._make_df()
        single_pass = run_primary_signal_forward(self._ScriptedSignalGenerator(), df)
        assert single_pass  # fixture sanity: the comparison below is not vacuous

        checkpoint = tmp_path / "fires.checkpoint.json"
        chunked = run_primary_signal_forward(
            self._ScriptedSignalGenerator(),
            df,
            checkpoint_path=checkpoint,
            checkpoint_every_bars=7,
        )

        assert chunked == single_pass

    def test_interrupted_run_resumes_from_checkpoint_and_matches_single_pass(self, tmp_path):
        df = self._make_df()
        single_pass = run_primary_signal_forward(self._ScriptedSignalGenerator(), df)

        checkpoint = tmp_path / "fires.checkpoint.json"
        failing = self._ScriptedSignalGenerator(fail_at=25)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run_primary_signal_forward(
                failing, df, checkpoint_path=checkpoint, checkpoint_every_bars=10
            )
        assert checkpoint.exists()

        resumed_generator = self._ScriptedSignalGenerator()
        resumed = run_primary_signal_forward(
            resumed_generator, df, checkpoint_path=checkpoint, checkpoint_every_bars=10
        )

        # warmup=5 + checkpoint every 10 bars -> the last durable checkpoint
        # before the bar-25 crash covers bars 5..24. The resume must evaluate
        # ONLY bars after it -- no bar at-or-before 24 is reprocessed.
        assert resumed_generator.evaluated == list(range(25, len(df)))
        assert resumed == single_pass

    def test_completed_checkpoint_skips_all_bars_on_rerun(self, tmp_path):
        df = self._make_df()
        checkpoint = tmp_path / "fires.checkpoint.json"
        first = run_primary_signal_forward(
            self._ScriptedSignalGenerator(),
            df,
            checkpoint_path=checkpoint,
            checkpoint_every_bars=10,
        )

        rerun_generator = self._ScriptedSignalGenerator()
        rerun = run_primary_signal_forward(
            rerun_generator, df, checkpoint_path=checkpoint, checkpoint_every_bars=10
        )

        assert rerun_generator.evaluated == []
        assert rerun == first

    def test_no_temp_file_left_behind(self, tmp_path):
        df = self._make_df()
        checkpoint = tmp_path / "fires.checkpoint.json"
        run_primary_signal_forward(
            self._ScriptedSignalGenerator(),
            df,
            checkpoint_path=checkpoint,
            checkpoint_every_bars=10,
        )
        assert [p.name for p in tmp_path.iterdir()] == ["fires.checkpoint.json"]

    def test_checkpoint_for_a_different_corpus_is_rejected(self, tmp_path):
        """Silently reusing a checkpoint recorded against a different window
        would splice fires from two corpora into one training set -- must
        raise, never guess."""
        checkpoint = tmp_path / "fires.checkpoint.json"
        run_primary_signal_forward(
            self._ScriptedSignalGenerator(), self._make_df(periods=40), checkpoint_path=checkpoint
        )
        with pytest.raises(ValueError, match="checkpoint"):
            run_primary_signal_forward(
                self._ScriptedSignalGenerator(),
                self._make_df(periods=60),
                checkpoint_path=checkpoint,
            )

    def test_invalid_checkpoint_interval_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="checkpoint_every_bars"):
            run_primary_signal_forward(
                self._ScriptedSignalGenerator(),
                self._make_df(),
                checkpoint_path=tmp_path / "fires.checkpoint.json",
                checkpoint_every_bars=0,
            )

    def test_same_length_different_corpus_checkpoint_is_rejected(self, tmp_path):
        """PR #981 arch-review finding: guarding only (start_index,
        total_bars) lets two equal-length but different corpora (another
        window or symbol) resume-splice silently -- the exact failure the
        guard exists to prevent. The checkpoint carries a content
        fingerprint (index endpoints + close-column checksum + generator
        identity), so an equal-length different corpus must be rejected."""
        checkpoint = tmp_path / "fires.checkpoint.json"
        run_primary_signal_forward(
            self._ScriptedSignalGenerator(), self._make_df(periods=40), checkpoint_path=checkpoint
        )

        other_window = pd.DataFrame({"close": np.linspace(200.0, 210.0, 40)})
        with pytest.raises(ValueError, match="checkpoint"):
            run_primary_signal_forward(
                self._ScriptedSignalGenerator(), other_window, checkpoint_path=checkpoint
            )

    def test_start_index_mismatch_checkpoint_is_rejected(self, tmp_path):
        df = self._make_df()
        checkpoint = tmp_path / "fires.checkpoint.json"
        run_primary_signal_forward(self._ScriptedSignalGenerator(), df, checkpoint_path=checkpoint)

        with pytest.raises(ValueError, match="checkpoint"):
            run_primary_signal_forward(
                self._ScriptedSignalGenerator(), df, start_index=3, checkpoint_path=checkpoint
            )

    def test_corrupt_checkpoint_file_is_rejected_loudly(self, tmp_path):
        """A checkpoint that isn't valid JSON must raise a clear ValueError
        naming the checkpoint file (policy: never guess, never overwrite a
        file we can't positively identify as our own checkpoint), not leak
        a raw json decoding error."""
        checkpoint = tmp_path / "fires.checkpoint.json"
        checkpoint.write_text("not json {{{", encoding="utf-8")

        with pytest.raises(ValueError, match="checkpoint"):
            run_primary_signal_forward(
                self._ScriptedSignalGenerator(), self._make_df(), checkpoint_path=checkpoint
            )

    def test_partial_checkpoint_missing_keys_is_rejected_loudly(self, tmp_path):
        """Valid JSON that isn't a complete checkpoint (e.g. truncated or
        foreign file) must raise the same clear ValueError -- never a raw
        KeyError, and never a silent resume from garbage."""
        checkpoint = tmp_path / "fires.checkpoint.json"
        checkpoint.write_text(
            json.dumps({"start_index": 5, "total_bars": 40, "last_completed_index": 10}),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="checkpoint"):
            run_primary_signal_forward(
                self._ScriptedSignalGenerator(), self._make_df(), checkpoint_path=checkpoint
            )


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


class TestResolveFiredTradeEntryPriceValidation:
    """PR #948 review finding (claude[bot]): entry_price (close[fire_index],
    raw market data) was used as a P&L divisor with no zero/positive/finite
    check -- CODE.md#L133 'Validate entry_price > 0 before any P&L or
    stop-loss calculation.' A zero/NaN close at fire_index must raise, not
    silently compute NaN and mislabel the row as unprofitable."""

    def test_zero_entry_price_raises(self):
        df = pd.DataFrame(
            {"close": [0.0, 101.0, 103.0], "high": [0.0, 102.0, 104.0], "low": [0.0, 100.0, 102.0]}
        )
        with pytest.raises(ValueError, match="entry_price"):
            resolve_fired_trade(
                df,
                fire_index=0,
                direction=1,
                take_profit_pct=0.05,
                stop_loss_pct=0.03,
                max_holding_bars=1,
            )

    def test_negative_entry_price_raises(self):
        df = pd.DataFrame(
            {
                "close": [-5.0, 101.0, 103.0],
                "high": [-5.0, 102.0, 104.0],
                "low": [-5.0, 100.0, 102.0],
            }
        )
        with pytest.raises(ValueError, match="entry_price"):
            resolve_fired_trade(
                df,
                fire_index=0,
                direction=1,
                take_profit_pct=0.05,
                stop_loss_pct=0.03,
                max_holding_bars=1,
            )

    def test_nan_entry_price_raises(self):
        df = pd.DataFrame(
            {
                "close": [float("nan"), 101.0, 103.0],
                "high": [float("nan"), 102.0, 104.0],
                "low": [float("nan"), 100.0, 102.0],
            }
        )
        with pytest.raises(ValueError, match="entry_price"):
            resolve_fired_trade(
                df,
                fire_index=0,
                direction=1,
                take_profit_pct=0.05,
                stop_loss_pct=0.03,
                max_holding_bars=1,
            )

    def test_pnl_calculation_matches_shared_pnl_percent(self):
        """The P&L math itself must be src.performance.metrics.pnl_percent,
        not a hand-rolled duplicate (CODE.md#L331 'never duplicate financial
        logic') -- verified by cross-checking a hand-computed fixture
        against pnl_percent's own output directly."""
        from src.performance.metrics import Side, pnl_percent

        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 103.0, 106.0],
                "high": [100.0, 102.0, 104.0, 107.0],
                "low": [99.0, 100.0, 102.0, 105.0],
            }
        )
        resolution = resolve_fired_trade(
            df,
            fire_index=0,
            direction=1,
            take_profit_pct=0.05,
            stop_loss_pct=0.03,
            max_holding_bars=3,
        )
        assert resolution is not None
        # bar3 high=107 >= entry(100)*1.05=105 -> TP hit, exit_price=105 (exact level).
        expected_raw_return = pnl_percent(100.0, 105.0, Side.LONG)
        round_trip_cost = 2.0 * CostCalculator().fee_rate
        assert resolution.profitable == ((expected_raw_return - round_trip_cost) > 0.0)


class TestResolutionOrderedHitRate:
    """resolution_ordered_hit_rate is the single shared windowing function
    both build_meta_label_features (training) and
    MetaLabelExamSignalGenerator._rolling_hit_rate (serving) now call --
    the fix for the fire-order-vs-resolution-order train/serve skew found
    in PR #950 review."""

    def test_empty_history_is_nan(self):
        assert np.isnan(resolution_ordered_hit_rate([], 20))

    def test_windows_by_resolution_order_not_input_order(self):
        """Passing entries in FIRE order (not resolution order) must not
        change the result -- the function re-sorts internally."""
        # fire_index 0 resolves LAST (exit_index=100); fire_index 1 resolves
        # FIRST (exit_index=10). "Last 1 by resolution order" must be
        # fire_index 0 (True), not fire_index 1 (False), even though
        # fire_index 1 comes second in fire/input order.
        fire_order_input = [(100, 0, True), (10, 1, False)]
        assert resolution_ordered_hit_rate(fire_order_input, 1) == pytest.approx(1.0)

    def test_ties_on_exit_index_break_by_fire_index(self):
        """Same-bar resolutions must order by fire_index ascending (matching
        the online path: _resolve_open_fires iterates still-open fires in
        their original registration order within one bar)."""
        tied = [(50, 2, False), (50, 1, True)]
        # Sorted by (exit_index, fire_index): (50,1,True) then (50,2,False).
        # Last 1 -> fire_index 2 -> False.
        assert resolution_ordered_hit_rate(tied, 1) == pytest.approx(0.0)

    def test_overlapping_trades_past_lookback_selects_resolution_ordered_subset(self):
        """With >20 resolved fires whose fire-order and resolution-order
        diverge, the correct (resolution-ordered) trailing-20 window must
        differ from the old, buggy fire-ordered trailing-20 window -- this
        is the regression this fix closes."""
        # 25 fires, alternating: even fire_index positions resolve almost
        # immediately (fast, profitable=True); odd positions resolve much
        # later (slow, profitable=False) -- classic overlapping-trades
        # pattern (an early slow fire resolves after several later fast
        # fires have already resolved).
        history = []
        for i in range(25):
            fire_index = i
            if i % 2 == 0:
                exit_index = fire_index + 1  # fast
                profitable = True
            else:
                exit_index = fire_index + 40  # slow, overlaps many fast fires
                profitable = False
            history.append((exit_index, fire_index, profitable))

        correct = resolution_ordered_hit_rate(history, 20)

        # The old (buggy) fire-order windowing: sort by fire_index only,
        # take the last 20 by FIRE order (i.e. fires 5..24).
        fire_ordered = sorted(history, key=lambda item: item[1])
        buggy_last_20 = [profitable for _, _, profitable in fire_ordered[-20:]]
        buggy = float(np.mean(buggy_last_20))

        # The two windowing conventions must select genuinely different
        # subsets for this scenario (proving the test is a meaningful
        # regression guard, not a no-op).
        assert correct != pytest.approx(buggy)


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


class TestBuildMetaLabelFeaturesRegimeAnnotationSingleTraversal:
    """Defect 1 of #955: EnhancedRegimeDetector.detect_regime re-annotates
    the ENTIRE DataFrame on every per-fire call when the regime columns are
    absent -- O(n_fires * n_bars), which made real (~46k-fire x ~47k-bar)
    corpora infeasible. build_meta_label_features must annotate exactly once
    upfront, with output identical to the per-call semantics."""

    _synthetic_df = staticmethod(TestBuildMetaLabelFeatures._synthetic_df)
    _resolved_next_bar = staticmethod(TestBuildMetaLabelFeatures._resolved_next_bar)

    def _fires(self) -> list[PrimarySignalRecord]:
        return [
            PrimarySignalRecord(index=index, direction=1, predicted_return=0.01)
            for index in (60, 100, 150, 220)
        ]

    def test_regime_annotation_runs_exactly_once_for_multi_fire_run(self):
        df = self._synthetic_df()
        fires = self._fires()
        resolutions = self._resolved_next_bar(fires, [True, False, True, False])

        detector = EnhancedRegimeDetector()
        annotate_calls = []
        original_annotate = detector.base_detector.annotate

        def counting_annotate(frame):
            annotate_calls.append(len(frame))
            return original_annotate(frame)

        detector.base_detector.annotate = counting_annotate

        features = build_meta_label_features(df, fires, resolutions, regime_detector=detector)

        assert len(features) == len(fires)
        assert len(annotate_calls) == 1

    def test_regime_features_match_per_fire_annotation_reference(self):
        """Each fire's regime feature must equal a fresh-state full-frame
        annotation read at that fire's index -- the CORRECT semantics, and
        what annotate-once produces. This is deliberately NOT a pure
        old-vs-new equivalence: the old shared-detector path re-annotated
        per fire with hysteresis state carried over from the previous
        call's pass, so fires evaluated after the first call could get
        stale-state labels near the frame start. Where old and new differ,
        the new fresh-state output is the fix (a behavior correction, not
        drift); where the old path was already fresh (its first call, and
        empirically most fires), they are identical."""
        df = self._synthetic_df()
        fires = self._fires()
        resolutions = self._resolved_next_bar(fires, [True, False, True, False])

        features = build_meta_label_features(df, fires, resolutions)

        reference = pd.DataFrame(
            [
                {
                    "regime_trend": regime.trend.value,
                    "regime_volatility": regime.volatility.value,
                    "regime_confidence": float(regime.confidence),
                }
                for regime in (
                    EnhancedRegimeDetector().detect_regime(df.copy(), fire.index) for fire in fires
                )
            ]
        )

        pd.testing.assert_frame_equal(
            features[["regime_trend", "regime_volatility", "regime_confidence"]].reset_index(
                drop=True
            ),
            reference,
        )

    def test_pre_annotated_frame_is_used_as_is_without_reannotation(self):
        """The explicit annotate-once seam: a caller that already annotated
        the frame (e.g. once for a whole training run) must get identical
        features with ZERO further annotation passes."""
        from src.regime.detector import RegimeDetector

        df = self._synthetic_df()
        fires = self._fires()
        resolutions = self._resolved_next_bar(fires, [True, False, True, False])
        annotated = RegimeDetector().annotate(df.copy())

        detector = EnhancedRegimeDetector()
        annotate_calls = []
        original_annotate = detector.base_detector.annotate

        def counting_annotate(frame):
            annotate_calls.append(len(frame))
            return original_annotate(frame)

        detector.base_detector.annotate = counting_annotate

        features_pre_annotated = build_meta_label_features(
            annotated, fires, resolutions, regime_detector=detector
        )
        features_plain = build_meta_label_features(df, fires, resolutions)

        assert annotate_calls == []
        pd.testing.assert_frame_equal(features_pre_annotated, features_plain)


class TestEncodeMetaLabelFeaturesForTraining:
    """encode_meta_label_features_for_training turns build_meta_label_features'
    DataFrame (with string regime enum columns) into numeric (X, y) arrays a
    sklearn LogisticRegression can fit on."""

    @staticmethod
    def _synthetic_df(periods=300):
        rng = np.random.default_rng(11)
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

    def test_shape_matches_feature_order(self):
        df = self._synthetic_df()
        fires = [
            PrimarySignalRecord(index=100, direction=1, predicted_return=0.01),
            PrimarySignalRecord(index=120, direction=-1, predicted_return=-0.02),
        ]
        resolutions = [
            TradeResolution(profitable=True, exit_index=101),
            TradeResolution(profitable=False, exit_index=121),
        ]
        features_df = build_meta_label_features(df, fires, resolutions)

        X, y = encode_meta_label_features_for_training(features_df)

        assert X.shape == (2, len(META_LABEL_FEATURE_ORDER))
        assert y.shape == (2,)
        assert X.dtype == np.float32
        assert list(y) == [1, 0]

    def test_nan_rolling_hit_rate_filled_with_neutral_prior(self):
        """The first fire has no eligible prior resolved fires -- its
        rolling_hit_rate_20 is NaN in build_meta_label_features' output and
        must be filled with a documented neutral prior (0.5), not left as
        NaN (sklearn cannot fit on NaN)."""
        df = self._synthetic_df()
        fires = [PrimarySignalRecord(index=100, direction=1, predicted_return=0.01)]
        resolutions = [TradeResolution(profitable=True, exit_index=101)]
        features_df = build_meta_label_features(df, fires, resolutions)
        assert np.isnan(features_df["rolling_hit_rate_20"].iloc[0])  # sanity check

        X, _y = encode_meta_label_features_for_training(features_df)

        assert np.all(np.isfinite(X))
        hit_rate_idx = META_LABEL_FEATURE_ORDER.index("rolling_hit_rate_20")
        assert X[0, hit_rate_idx] == pytest.approx(0.5)

    def test_regime_columns_are_numeric_encoded(self):
        df = self._synthetic_df()
        fires = [PrimarySignalRecord(index=150, direction=1, predicted_return=0.01)]
        resolutions = [TradeResolution(profitable=True, exit_index=151)]
        features_df = build_meta_label_features(df, fires, resolutions)

        X, _y = encode_meta_label_features_for_training(features_df)

        # Every value must be finite and numeric -- no string enum values leaked through.
        assert np.all(np.isfinite(X))

    def test_empty_features_raises(self):
        with pytest.raises(ValueError, match="empty"):
            encode_meta_label_features_for_training(pd.DataFrame())


class TestEncodeMetaLabelFeatureRow:
    """encode_meta_label_feature_row is the INFERENCE-time counterpart --
    one row at a time (the exam signal generator computes features
    incrementally, not as a batch DataFrame) -- sharing the exact same
    encoding as encode_meta_label_features_for_training so training and
    inference can never drift apart."""

    def test_returns_array_in_feature_order(self):
        row = encode_meta_label_feature_row(
            realized_vol_48=0.01,
            rolling_hit_rate_20=0.6,
            session_sin=0.5,
            session_cos=0.86,
            regime_trend="trend_up",
            regime_volatility="high_vol",
            regime_confidence=0.7,
            predicted_return_magnitude=0.02,
        )

        assert row.shape == (len(META_LABEL_FEATURE_ORDER),)
        assert row.dtype == np.float32
        assert np.all(np.isfinite(row))

    def test_nan_rolling_hit_rate_filled_with_neutral_prior(self):
        row = encode_meta_label_feature_row(
            realized_vol_48=0.01,
            rolling_hit_rate_20=float("nan"),
            session_sin=0.5,
            session_cos=0.86,
            regime_trend="range",
            regime_volatility="low_vol",
            regime_confidence=0.5,
            predicted_return_magnitude=0.01,
        )

        hit_rate_idx = META_LABEL_FEATURE_ORDER.index("rolling_hit_rate_20")
        assert row[hit_rate_idx] == pytest.approx(0.5)

    def test_matches_batch_encoding_for_the_same_logical_row(self):
        """Training-time (batch) and inference-time (row) encoders must
        agree bit-for-bit on the same input -- this is the leak/drift guard
        the whole split exists to prevent."""
        df = self._synthetic_df_helper()
        fires = [PrimarySignalRecord(index=150, direction=1, predicted_return=0.02)]
        resolutions = [TradeResolution(profitable=True, exit_index=151)]
        features_df = build_meta_label_features(df, fires, resolutions)
        batch_X, _y = encode_meta_label_features_for_training(features_df)

        row = features_df.iloc[0]
        row_X = encode_meta_label_feature_row(
            realized_vol_48=row["realized_vol_48"],
            rolling_hit_rate_20=row["rolling_hit_rate_20"],
            session_sin=row["session_sin"],
            session_cos=row["session_cos"],
            regime_trend=row["regime_trend"],
            regime_volatility=row["regime_volatility"],
            regime_confidence=row["regime_confidence"],
            predicted_return_magnitude=row["predicted_return_magnitude"],
        )

        np.testing.assert_allclose(batch_X[0], row_X)

    @staticmethod
    def _synthetic_df_helper(periods=300):
        rng = np.random.default_rng(13)
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
