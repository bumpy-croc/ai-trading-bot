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
        # Cached rolling metrics for incremental updates
        self._cache_index: Optional[pd.Index] = None
        self._cache_len: int = 0
        self._cache_slope: Optional[pd.Series] = None
        self._cache_r2: Optional[pd.Series] = None
        self._cache_atr: Optional[pd.Series] = None
        self._cache_atr_pct: Optional[pd.Series] = None
        self._cache_trend_score: Optional[pd.Series] = None
        self._cache_regime_conf: Optional[pd.Series] = None
        self._cache_vol_label_values: Optional[pd.Series] = None
        self._cache_result: Optional[pd.DataFrame] = None
        self._last_processed_index: Optional[pd.Timestamp] = None

    def _reset_cache(self) -> None:
        self._cache_index = None
        self._cache_len = 0
        self._cache_slope = None
        self._cache_r2 = None
        self._cache_atr = None
        self._cache_atr_pct = None
        self._cache_trend_score = None
        self._cache_regime_conf = None
        self._cache_vol_label_values = None
        self._cache_result = None
        self._last_processed_index = None

    @staticmethod
    def _ols_slope_and_r2_block(window: pd.Series) -> tuple[float, float]:
        if len(window) == 0:
            return np.nan, np.nan
        y = np.log(window.astype(float).clip(lower=1e-8))
        t = np.arange(len(y), dtype=float)
        t_mean = t.mean()
        y_mean = y.mean()
        tt = t - t_mean
        yy = y - y_mean
        denom = (tt**2).sum()
        if denom == 0:
            return np.nan, np.nan
        slope = float((tt * yy).sum() / denom)
        y_hat = y_mean + slope * tt
        ss_tot = float((yy**2).sum())
        ss_res = float(((y - y_hat) ** 2).sum())
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        return slope, r2

    @staticmethod
    def _rolling_ols_slope_and_r2(x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
        # Compute rolling OLS slope and R^2 for y = log(price) vs time index
        y = np.log(x.clip(lower=1e-8))
        idx = np.arange(len(y))
        df = pd.DataFrame({"y": y.values, "t": idx}, index=y.index)

        # Rolling OLS helper
        def _ols(block: pd.DataFrame):
            t = block["t"].values.astype(float)
            yb = block["y"].values.astype(float)
            t_mean = t.mean()
            y_mean = yb.mean()
            tt = t - t_mean
            yy = yb - y_mean
            denom = (tt**2).sum()
            if denom == 0:
                return pd.Series([np.nan, np.nan])
            slope = (tt * yy).sum() / denom
            y_hat = y_mean + slope * tt
            ss_tot = (yy**2).sum()
            ss_res = ((yb - y_hat) ** 2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            return pd.Series([slope, r2])

        # Compute using a simple sliding window to avoid DataFrame.rolling apply quirks
        slopes = []
        r2s = []
        for i in range(len(df)):
            if i + 1 < window:
                slopes.append(np.nan)
                r2s.append(np.nan)
                continue
            block = df.iloc[i + 1 - window : i + 1]
            vals = _ols(block)
            slopes.append(vals.iloc[0])
            r2s.append(vals.iloc[1])
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
        def rank_last(window: pd.Series) -> float:
            if window.isna().any():
                return np.nan
            last = window.iloc[-1]
            return (window <= last).mean()

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
        vol_label_values = vol_label.apply(
            lambda v: v.value if isinstance(v, VolLabel) else str(v)
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
        out["regime_label"] = out["trend_label"].astype(str) + ":" + vol_label_values.astype(str)
        # Confidence from normalized |trend_score|
        ts = trend_score.copy()
        ts_mean = ts.rolling(252, min_periods=cfg.slope_window).mean()
        ts_std = ts.rolling(252, min_periods=cfg.slope_window).std(ddof=0)
        z = (ts - ts_mean) / ts_std.replace(0, np.nan)
        conf = z.abs().clip(0, 3) / 3.0
        out["regime_confidence"] = conf
        # Update cache for incremental path
        self._cache_index = out.index
        self._cache_len = len(out)
        self._cache_slope = slope.to_numpy(copy=True)
        self._cache_r2 = r2.to_numpy(copy=True)
        self._cache_atr = atr.to_numpy(copy=True)
        self._cache_atr_pct = atr_pct.to_numpy(copy=True)
        self._cache_trend_score = trend_score.to_numpy(copy=True)
        self._cache_regime_conf = conf.to_numpy(copy=True)
        self._cache_vol_label_values = vol_label_values.astype(object).to_numpy(copy=True)
        self._cache_result = out.copy()
        self._last_processed_index = out.index[-1] if len(out) else None
        return out

    def _can_use_incremental(self, df: pd.DataFrame) -> bool:
        if self._cache_result is None or self._cache_index is None:
            return False
        if df.empty or self._cache_len == 0:
            return False
        if len(df) != self._cache_len:
            return False
        prev_index = self._cache_index
        new_index = df.index
        if len(prev_index) < 2:
            return False
        if not prev_index[1:].equals(new_index[:-1]):
            return False
        return True

    def annotate_incremental(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._can_use_incremental(df):
            return self.annotate(df)

        cfg = self.config
        prev_out = self._cache_result
        assert prev_out is not None

        length = len(df)
        prev_slope = self._cache_slope
        prev_r2 = self._cache_r2
        prev_atr = self._cache_atr
        prev_atr_pct = self._cache_atr_pct
        prev_vol_values = self._cache_vol_label_values

        if (
            prev_slope is None
            or prev_r2 is None
            or prev_atr is None
            or prev_atr_pct is None
            or prev_vol_values is None
            or len(prev_slope) != length
        ):
            return self.annotate(df)

        out = df.copy()

        closes = df["close"].astype(float)
        if length >= cfg.slope_window:
            window = closes.iloc[-cfg.slope_window :]
            new_slope, new_r2 = self._ols_slope_and_r2_block(window)
        else:
            new_slope, new_r2 = np.nan, np.nan

        slope_vals = np.roll(prev_slope, -1)
        r2_vals = np.roll(prev_r2, -1)
        slope_vals[-1] = new_slope
        r2_vals[-1] = new_r2

        invalid_span = max(cfg.slope_window - 1, 0)
        if invalid_span > 0:
            slope_vals[:invalid_span] = np.nan
            r2_vals[:invalid_span] = np.nan

        trend_score_vals = slope_vals * r2_vals
        mask_low_r2 = r2_vals < cfg.r2_min
        trend_score_vals[mask_low_r2] = 0.0

        atr_vals = np.roll(prev_atr, -1)
        atr_window_data = df.iloc[-(cfg.atr_window + 1) :]
        atr_tail = self._atr(atr_window_data, cfg.atr_window)
        new_atr = float(atr_tail.iloc[-1]) if len(atr_tail) and not pd.isna(atr_tail.iloc[-1]) else np.nan
        atr_vals[-1] = new_atr
        atr_invalid = max(cfg.atr_window - 1, 0)
        if atr_invalid > 0:
            atr_vals[:atr_invalid] = np.nan

        atr_pct_vals = np.roll(prev_atr_pct, -1)
        window_vals = atr_vals[-cfg.atr_percentile_lookback :]
        if len(window_vals) == cfg.atr_percentile_lookback and not np.isnan(window_vals).any():
            last_val = window_vals[-1]
            new_pct = float(np.count_nonzero(window_vals <= last_val) / len(window_vals))
        else:
            new_pct = np.nan
        atr_pct_vals[-1] = new_pct
        pct_invalid_total = max(cfg.atr_percentile_lookback - 1, 0) + atr_invalid
        if pct_invalid_total > 0:
            atr_pct_vals[:pct_invalid_total] = np.nan

        prev_trend_labels = prev_out["trend_label"].to_numpy(copy=True)
        trend_labels = np.roll(prev_trend_labels, -1)
        proposed_label = str(self._label_trend(trend_score_vals[-1]).value)
        last = self._last_label
        cons = self._consecutive
        dwell = self._dwell
        if last is None:
            last = proposed_label
            cons = 1
            dwell = 1
        elif proposed_label == last:
            cons += 1
            dwell += 1
        else:
            cons += 1
            if dwell >= cfg.min_dwell and cons >= cfg.hysteresis_k:
                last = proposed_label
                dwell = 1
                cons = 1
        trend_labels[-1] = last
        if invalid_span > 0:
            trend_labels[:invalid_span] = TrendLabel.RANGE.value
        self._last_label = last
        self._consecutive = cons
        self._dwell = dwell

        prev_vol_labels = prev_out["vol_label"].to_numpy(copy=True)
        vol_labels = np.roll(prev_vol_labels, -1)
        vol_label_last = (
            VolLabel.HIGH if (not pd.isna(new_pct) and new_pct >= cfg.atr_high_percentile) else VolLabel.LOW
        )
        vol_labels[-1] = str(vol_label_last)
        if pct_invalid_total > 0:
            vol_labels[:pct_invalid_total] = str(VolLabel.LOW)

        vol_value_labels = np.roll(prev_vol_values, -1)
        vol_value_labels[-1] = vol_label_last.value
        if pct_invalid_total > 0:
            vol_value_labels[:pct_invalid_total] = VolLabel.LOW.value

        regime_labels = np.array([f"{t}:{v}" for t, v in zip(trend_labels, vol_value_labels)], dtype=object)

        trend_series = pd.Series(trend_score_vals, index=df.index)
        ts_mean = trend_series.rolling(252, min_periods=cfg.slope_window).mean()
        ts_std = trend_series.rolling(252, min_periods=cfg.slope_window).std(ddof=0)
        z = (trend_series - ts_mean) / ts_std.replace(0, np.nan)
        conf_series = z.abs().clip(0, 3) / 3.0
        regime_conf_vals = conf_series.to_numpy(copy=True)

        out["trend_score"] = trend_score_vals
        out["trend_label"] = trend_labels
        out["atr"] = atr_vals
        out["atr_percentile"] = atr_pct_vals
        out["vol_label"] = vol_labels
        out["regime_label"] = regime_labels
        out["regime_confidence"] = regime_conf_vals

        self._cache_index = out.index
        self._cache_len = len(out)
        self._cache_slope = slope_vals.copy()
        self._cache_r2 = r2_vals.copy()
        self._cache_atr = atr_vals.copy()
        self._cache_atr_pct = atr_pct_vals.copy()
        self._cache_trend_score = trend_score_vals.copy()
        self._cache_regime_conf = regime_conf_vals.copy()
        self._cache_vol_label_values = vol_value_labels.copy()
        self._cache_result = out.copy()
        self._last_processed_index = out.index[-1]

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
