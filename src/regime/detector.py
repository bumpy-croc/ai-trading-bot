from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd


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

    def __init__(self, config: Optional[RegimeConfig] = None):
        self.config = config or RegimeConfig()
        self._last_label: Optional[str] = None
        self._consecutive: int = 0
        self._dwell: int = 0

    @staticmethod
    def _rolling_ols_slope_and_r2(x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        """Vectorised rolling OLS slope and R^2 for log-price vs time index."""

        if window <= 0:
            raise ValueError("window must be positive")

        y = np.log(x.clip(lower=1e-8)).astype(float)
        n = len(y)
        index = x.index

        slopes = np.full(n, np.nan, dtype=float)
        r2s = np.full(n, np.nan, dtype=float)

        if n < window:
            return pd.Series(slopes, index=index), pd.Series(r2s, index=index)

        t = np.arange(n, dtype=float)

        y_valid = ~np.isnan(y)
        counts = np.cumsum(y_valid.astype(int))
        counts = counts[window - 1 :] - np.concatenate(([0], counts[:-window]))
        valid_mask = counts == window

        y_filled = np.where(y_valid, y, 0.0)

        def _rolling_sum(arr: np.ndarray) -> np.ndarray:
            csum = np.cumsum(arr, dtype=float)
            return csum[window - 1 :] - np.concatenate(([0.0], csum[:-window]))

        sum_y = _rolling_sum(y_filled)
        sum_t = _rolling_sum(t)
        sum_ty = _rolling_sum(t * y_filled)
        sum_t2 = _rolling_sum(t * t)
        sum_y2 = _rolling_sum(y_filled * y_filled)

        w = float(window)
        denom = w * sum_t2 - sum_t**2
        denom_zero = denom == 0

        slope = np.full_like(sum_y, np.nan)
        valid_denom = ~denom_zero
        slope[valid_denom] = (w * sum_ty - sum_t * sum_y)[valid_denom] / denom[valid_denom]

        mean_t = sum_t / w
        mean_y = sum_y / w
        intercept = mean_y - slope * mean_t

        ss_tot = sum_y2 - (sum_y**2) / w
        ss_res = (
            sum_y2
            - 2 * slope * sum_ty
            - 2 * intercept * sum_y
            + (slope**2) * sum_t2
            + 2 * slope * intercept * sum_t
            + w * (intercept**2)
        )
        ss_res = np.maximum(ss_res, 0.0)

        positive_ss_tot = ss_tot > 0
        r2 = np.full_like(sum_y, np.nan)
        r2[positive_ss_tot] = 1 - ss_res[positive_ss_tot] / ss_tot[positive_ss_tot]

        valid_slope = valid_mask & valid_denom
        valid_r2 = valid_slope & positive_ss_tot

        slopes_out = slopes.copy()
        r2_out = r2s.copy()
        slope_idx = np.nonzero(valid_slope)[0] + (window - 1)
        r2_idx = np.nonzero(valid_r2)[0] + (window - 1)
        slopes_out[slope_idx] = slope[valid_slope]
        r2_out[r2_idx] = r2[valid_r2]

        return pd.Series(slopes_out, index=index), pd.Series(r2_out, index=index)

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
        if lookback <= 0:
            raise ValueError("lookback must be positive")

        values = series.to_numpy(dtype=float, copy=False)
        n = len(values)
        result = np.full(n, np.nan, dtype=float)
        if n < lookback:
            return pd.Series(result, index=series.index)

        windows = np.lib.stride_tricks.sliding_window_view(values, lookback)
        invalid = np.isnan(windows).any(axis=1)
        last = windows[:, -1]
        pct = (windows <= last[:, None]).mean(axis=1)
        pct[invalid] = np.nan
        result[lookback - 1 :] = pct
        return pd.Series(result, index=series.index)

    def annotate(self, df: pd.DataFrame, *, inplace: bool = False) -> pd.DataFrame:
        cfg = self.config
        out = df if inplace else df.copy(deep=False)
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
        z = (ts - ts_mean) / ts_std.replace(0, np.nan)
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
