import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


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
    def _rolling_ols_slope_and_r2(x: pd.Series, window: int) -> Tuple[pd.Series, pd.Series]:
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
            denom = (tt ** 2).sum()
            if denom == 0:
                return pd.Series([np.nan, np.nan])
            slope = (tt * yy).sum() / denom
            y_hat = y_mean + slope * tt
            ss_tot = (yy ** 2).sum()
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
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
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
<<<<<<< HEAD
        # Trend
=======
>>>>>>> feature/regime-detection-mvp
        slope, r2 = self._rolling_ols_slope_and_r2(out["close"], cfg.slope_window)
        trend_score = slope * r2
        trend_score[r2 < cfg.r2_min] = 0.0
        out["trend_score"] = trend_score
        trend_label = trend_score.apply(self._label_trend)
<<<<<<< HEAD
        # Volatility
=======
>>>>>>> feature/regime-detection-mvp
        atr = self._atr(out, cfg.atr_window)
        out["atr"] = atr
        atr_pct = self._percentile_rank(atr, cfg.atr_percentile_lookback)
        out["atr_percentile"] = atr_pct
        vol_label = atr_pct.apply(lambda p: VolLabel.HIGH if (not pd.isna(p) and p >= cfg.atr_high_percentile) else VolLabel.LOW)
<<<<<<< HEAD
        # Hysteresis on combined trend label only (vol used as overlay)
=======
>>>>>>> feature/regime-detection-mvp
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

    def current_labels(self, df: pd.DataFrame) -> Tuple[str, str, float]:
        if df.empty or "regime_label" not in df.columns:
            return "unknown", "unknown", 0.0
        last = df.iloc[-1]
        return str(last.get("trend_label", "unknown")), str(last.get("vol_label", "unknown")), float(last.get("regime_confidence", 0.0))

    def long_position_multiplier(self, trend_label: str, vol_label: str, confidence: float) -> float:
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