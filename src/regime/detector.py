from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


class TrendLabel(str, Enum):
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"


class VolLabel(str, Enum):
    HIGH = "high_vol"
    LOW = "low_vol"


@dataclass
class RegimeConfig:
    slope_window: int = 50
    band_window: int = 20
    atr_window: int = 14
    atr_percentile_lookback: int = 252
    trend_threshold: float = 0.0  # threshold applied to slope*R2 (log-price)
    r2_min: float = 0.2
    atr_high_percentile: float = 0.7
    hysteresis_k: int = 3  # consecutive confirmations required to switch
    min_dwell: int = 12  # minimum bars to stay in a regime before switching


class RegimeDetector:
    """
    Minimal regime detector using:
    - Trend: sign of rolling OLS slope on log-price, weighted by R^2
    - Volatility: ATR percentile over a rolling history
    - Hysteresis: require K consecutive confirmations and minimum dwell time

    annotate(df) adds columns: 'trend_score', 'trend_label', 'vol_label', 'regime_label', 'regime_confidence'
    """

    def __init__(self, config: RegimeConfig | None = None):
        self.config = config or RegimeConfig()
        self._last_label: str | None = None
        self._consecutive: int = 0
        self._dwell: int = 0

    @staticmethod
    def _rolling_ols_slope_and_r2(x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        """Compute rolling OLS slope and R^2 for log-price vs time using vectorized sums."""

        y = np.log(x.clip(lower=1e-8).astype(float))
        n = len(y)
        slopes = np.full(n, np.nan, dtype=float)
        r2s = np.full(n, np.nan, dtype=float)
        if window <= 0 or n == 0:
            return pd.Series(slopes, index=x.index), pd.Series(r2s, index=x.index)

        if window > n:
            return pd.Series(slopes, index=y.index), pd.Series(r2s, index=y.index)

        t = np.arange(n, dtype=float)
        y_vals = y.to_numpy(dtype=float)

        # Prefix cumulative sums to efficiently compute rolling window sums
        # Use nancumsum to avoid NaN propagation - windows without NaN will still be valid
        def _nancumsum_with_zero(arr: np.ndarray) -> np.ndarray:
            out = np.empty(arr.size + 1, dtype=float)
            out[0] = 0.0
            np.nancumsum(arr, out=out[1:])
            return out

        csum_t = _nancumsum_with_zero(t)
        csum_y = _nancumsum_with_zero(y_vals)
        csum_tt = _nancumsum_with_zero(t * t)
        csum_yy = _nancumsum_with_zero(y_vals * y_vals)
        csum_ty = _nancumsum_with_zero(t * y_vals)

        # Track which windows contain NaN values to invalidate them
        nan_mask = np.isnan(y_vals)
        csum_nan_count = _nancumsum_with_zero(nan_mask.astype(float))

        # Rolling window sums (length n - window + 1)
        window_slice = slice(window, None)
        sum_t = csum_t[window_slice] - csum_t[:-window]
        sum_y = csum_y[window_slice] - csum_y[:-window]
        sum_tt = csum_tt[window_slice] - csum_tt[:-window]
        sum_yy = csum_yy[window_slice] - csum_yy[:-window]
        sum_ty = csum_ty[window_slice] - csum_ty[:-window]

        # Count of NaN values in each rolling window
        nan_count_in_window = csum_nan_count[window_slice] - csum_nan_count[:-window]
        windows_with_nan = nan_count_in_window > 0

        window_float = float(window)
        cov_ty = window_float * sum_ty - sum_t * sum_y
        var_t = window_float * sum_tt - sum_t * sum_t
        var_y_raw = window_float * sum_yy - sum_y * sum_y

        # Guard against catastrophic cancellation in variance calculation
        # For large windows, the subtraction can produce small negative numbers
        # due to floating-point precision issues, even when variance should be positive
        var_y = np.maximum(var_y_raw, 0.0)

        # Detect cases where cancellation likely occurred (small negative values)
        # These should be treated as effectively zero variance
        cancellation_threshold = 1e-12 * window_float * np.maximum(np.abs(sum_yy), 1.0)
        likely_cancellation = (var_y_raw < 0) & (np.abs(var_y_raw) <= cancellation_threshold)

        # Valid slope calculation requires non-zero variance and no NaN in window
        valid_slope = (var_t > 0) & (~windows_with_nan)
        slope_vals = np.full_like(sum_ty, np.nan, dtype=float)
        slope_vals[valid_slope] = cov_ty[valid_slope] / var_t[valid_slope]

        # Valid R² calculation requires valid slope and non-zero y variance
        # For backward compatibility, when var_y is very small (effectively zero),
        # we set R² to 0 instead of NaN to match the naive implementation behavior
        var_y_threshold = 1e-9  # Threshold for numerical precision
        valid_r2 = valid_slope & (var_y > var_y_threshold) & (~likely_cancellation)
        near_zero_var_y = valid_slope & ((var_y <= var_y_threshold) | likely_cancellation)

        r2_vals = np.full_like(sum_ty, np.nan, dtype=float)
        r2_vals[valid_r2] = (cov_ty[valid_r2] ** 2) / (var_t[valid_r2] * var_y[valid_r2])
        r2_vals[valid_r2] = np.clip(r2_vals[valid_r2], 0.0, 1.0)

        # Set R² to 0 for cases where var_y is effectively zero (constant y values)
        r2_vals[near_zero_var_y] = 0.0

        # Validate array bounds before assignment to prevent index out of bounds errors
        start_idx = window - 1
        expected_length = n - start_idx  # Length of slopes[start_idx:]
        actual_length = len(slope_vals)

        if actual_length != expected_length:
            raise RuntimeError(
                f"Array length mismatch in OLS calculation: "
                f"expected {expected_length} values for slopes[{start_idx}:], "
                f"but computed {actual_length} slope values. "
                f"n={n}, window={window}. This indicates a bug in rolling window calculation."
            )

        # Safe assignment after validation
        slopes[start_idx:] = slope_vals
        r2s[start_idx:] = r2_vals

        return pd.Series(slopes, index=y.index), pd.Series(r2s, index=y.index)

    @staticmethod
    def _atr(df: pd.DataFrame, window: int) -> pd.Series:
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=window, min_periods=window).mean()

    def _label_trend(self, trend_score: float) -> TrendLabel:
        if pd.isna(trend_score):
            return TrendLabel.RANGE
        if trend_score > self.config.trend_threshold:
            return TrendLabel.TREND_UP
        if trend_score < -self.config.trend_threshold:
            return TrendLabel.TREND_DOWN
        return TrendLabel.RANGE

    @staticmethod
    def _percentile_rank(series: pd.Series, lookback: int) -> pd.Series:
        """Optimized percentile rank calculation"""

        def rank_last(window: pd.Series | np.ndarray) -> float:
            arr = np.asarray(window)
            if np.isnan(arr).any():
                return np.nan
            last = arr[-1]
            return np.mean(arr <= last)

        # Fast path: vectorised percentile rank using sliding window view
        values = series.to_numpy()
        n = len(values)
        if lookback <= 0 or n == 0 or n < lookback:
            return pd.Series(np.nan, index=series.index, dtype=float)

        try:
            windows = sliding_window_view(values, lookback)
            last_vals = windows[:, -1]
            nan_mask = np.isnan(windows).any(axis=1) | np.isnan(last_vals)
            ranks = np.where(
                nan_mask,
                np.nan,
                (windows <= last_vals[:, None]).mean(axis=1),
            )
            result = np.full(n, np.nan, dtype=float)
            result[lookback - 1 :] = ranks
            return pd.Series(result, index=series.index, dtype=float)
        except Exception:
            # Fall back to rolling apply paths when sliding window is unavailable (e.g., older numpy)
            pass

        # Use engine='numba' for better performance if available
        try:
            return series.rolling(window=lookback, min_periods=lookback).apply(
                rank_last, raw=True, engine="numba"
            )
        except Exception:
            # Fallback to default engine
            return series.rolling(window=lookback, min_periods=lookback).apply(rank_last, raw=False)

    def annotate(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        out = df.copy()
        # Trend
        slope, r2 = self._rolling_ols_slope_and_r2(out["close"], cfg.slope_window)
        trend_score = slope * r2
        trend_score[r2 < cfg.r2_min] = 0.0
        out["trend_score"] = trend_score
        trend_label = trend_score.apply(self._label_trend)
        # Volatility
        atr = self._atr(out, cfg.atr_window)
        out["atr"] = atr
        atr_pct = self._percentile_rank(atr, cfg.atr_percentile_lookback)
        out["atr_percentile"] = atr_pct
        vol_label = atr_pct.apply(
            lambda p: (
                VolLabel.HIGH if (not pd.isna(p) and p >= cfg.atr_high_percentile) else VolLabel.LOW
            )
        )
        # Hysteresis on combined trend label only (vol used as overlay)
        labels = []
        dwell = self._dwell
        cons = self._consecutive
        last = self._last_label
        for tl in trend_label:
            proposed = str(tl.value)
            # Initialize if unset
            if last is None:
                last = proposed
                cons = 1
                dwell = 1
                labels.append(last)
                continue
            # If same as current state, increase dwell/cons and continue
            if proposed == last:
                cons += 1
                dwell += 1
                labels.append(last)
                continue
            # If different, require both dwell and confirmations to switch
            cons += 1  # confirmation counter on proposed label
            if dwell >= cfg.min_dwell and cons >= cfg.hysteresis_k:
                last = proposed
                dwell = 1
                cons = 1
            else:
                # do not switch; keep current
                pass
            labels.append(last)
        self._last_label = last
        self._consecutive = cons
        self._dwell = dwell
        out["trend_label"] = labels
        out["vol_label"] = vol_label.astype(str)
        out["regime_label"] = out["trend_label"].astype(str) + ":" + out["vol_label"].astype(str)
        # Confidence from normalized |trend_score|
        ts = trend_score.copy()
        ts_mean = ts.rolling(252, min_periods=cfg.slope_window).mean()
        ts_std = ts.rolling(252, min_periods=cfg.slope_window).std(ddof=0)

        # Robust division: handle zero, near-zero, NaN, and inf in std
        # Use threshold to prevent division by very small numbers that might be numerical noise
        std_threshold = 1e-9
        ts_std_safe = ts_std.copy()
        # Replace zeros, near-zeros, NaN, and inf with NaN to prevent corrupt confidence scores
        invalid_std = (ts_std_safe.abs() < std_threshold) | ts_std_safe.isna() | np.isinf(ts_std_safe)
        ts_std_safe[invalid_std] = np.nan

        z = (ts - ts_mean) / ts_std_safe
        conf = z.abs().clip(0, 3) / 3.0
        out["regime_confidence"] = conf
        return out

    def current_labels(self, df: pd.DataFrame) -> tuple[str, str, float]:
        if df.empty or "regime_label" not in df.columns:
            return "unknown", "unknown", 0.0
        last = df.iloc[-1]
        return (
            str(last.get("trend_label", "unknown")),
            str(last.get("vol_label", "unknown")),
            float(last.get("regime_confidence", 0.0)),
        )

    def long_position_multiplier(
        self, trend_label: str, vol_label: str, confidence: float
    ) -> float:
        # Conservative defaults
        mult = 1.0
        if vol_label == VolLabel.HIGH.value:
            mult *= 0.8
        if trend_label == TrendLabel.RANGE.value:
            mult *= 0.9
        if trend_label == TrendLabel.TREND_DOWN.value:
            mult *= 0.7
        if confidence < 0.5:
            mult *= 0.8
        return float(max(0.2, min(1.0, mult)))
