"""US spot ETF net-flow data provider (#803).

US spot BTC/ETH ETF net flows are the marginal buyer/seller this cycle — price
legs have tracked multi-day flow streaks — but the bot has had no flow
awareness. This provider ingests daily net flows, caches them to parquet (like
market data), degrades gracefully when the upstream source is unreachable, and
exposes derived features (5d/20d net-flow z-scores, consecutive-outflow days)
for both the flow *gate* (rule-based, works today) and the flow *feature
extractor* (model input, inert until a compatible model is retrained).

Data source: a pluggable ``fetch_fn`` (default targets Farside Investors' daily
CSV). When the fetch fails or is unavailable (offline/CI), the provider falls
back to the cache and then to a bundled seed dataset, so it never hard-fails a
trading loop — it degrades to neutral/last-known flows and logs.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config.constants import (
    DEFAULT_ETF_FLOW_CACHE_TTL_HOURS,
    DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
    DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
)
from src.config.paths import get_cache_dir, resolve_data_path

logger = logging.getLogger(__name__)

# Net-flow columns (USD). One per tracked asset.
BTC_FLOW_COL = "btc_etf_netflow_usd"
ETH_FLOW_COL = "eth_etf_netflow_usd"
FLOW_COLUMNS = [BTC_FLOW_COL, ETH_FLOW_COL]

SEED_FILENAME = "etf_flows_seed.csv"
CACHE_FILENAME = "etf_flows.parquet"

FetchFn = Callable[[datetime, datetime], "pd.DataFrame | None"]


class ETFFlowProvider:
    """Daily US spot ETF net flows with parquet cache + graceful degradation."""

    def __init__(
        self,
        *,
        cache_dir: Path | None = None,
        fetch_fn: FetchFn | None = None,
        cache_ttl_hours: int = DEFAULT_ETF_FLOW_CACHE_TTL_HOURS,
        seed_path: Path | None = None,
    ) -> None:
        self._cache_dir = (
            Path(cache_dir) if cache_dir is not None else get_cache_dir() / "etf_flows"
        )
        self._fetch_fn = fetch_fn if fetch_fn is not None else _default_fetch
        self._cache_ttl_hours = cache_ttl_hours
        self._seed_path = (
            Path(seed_path) if seed_path is not None else resolve_data_path(SEED_FILENAME)
        )

    @property
    def _cache_file(self) -> Path:
        return self._cache_dir / CACHE_FILENAME

    # --- public API --------------------------------------------------------

    def get_flows(self, start: datetime, end: datetime) -> pd.DataFrame:
        """Return a UTC-daily-indexed frame of net flows over ``[start, end]``.

        Resolution order: fresh cache → upstream fetch (merged into cache) →
        stale cache → bundled seed. Always returns a frame (possibly empty) and
        never raises on source failure.
        """
        start = _as_utc(start)
        end = _as_utc(end)

        cached = self._load_cache()
        if cached is not None and self._cache_is_fresh() and _covers(cached, start, end):
            return _slice(cached, start, end)

        fetched = self._try_fetch(start, end)
        if fetched is not None and not fetched.empty:
            merged = _merge(cached, fetched)
            self._save_cache(merged)
            return _slice(merged, start, end)

        if cached is not None and not cached.empty:
            logger.warning("ETF flows: upstream unavailable; serving cached data (may be stale)")
            return _slice(cached, start, end)

        seed = self._load_seed()
        if seed is not None and not seed.empty:
            logger.warning("ETF flows: no cache/upstream; serving bundled seed dataset")
            return _slice(seed, start, end)

        logger.warning("ETF flows: no data available from any source; returning empty frame")
        return _empty_frame()

    def flow_features(
        self,
        as_of: datetime,
        *,
        asset_col: str = BTC_FLOW_COL,
        short_window: int = DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
        long_window: int = DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
    ) -> dict[str, float | None]:
        """Return derived flow features as of ``as_of`` (inclusive).

        Keys: ``netflow_zscore_5d``, ``netflow_zscore_20d``,
        ``consecutive_outflow_days``. Values are ``None`` when there is
        insufficient history (the caller decides how to treat unknown flow).
        """
        as_of = _as_utc(as_of)
        # Pull enough history to compute the long-window z-score up to as_of.
        start = as_of - timedelta(days=long_window * 2 + 5)
        flows = self.get_flows(start, as_of)
        return compute_flow_features(
            flows, as_of, asset_col=asset_col, short_window=short_window, long_window=long_window
        )

    # --- internals ---------------------------------------------------------

    def _try_fetch(self, start: datetime, end: datetime) -> pd.DataFrame | None:
        try:
            raw = self._fetch_fn(start, end)
        except Exception as exc:  # noqa: BLE001 - network/parse errors must not break trading
            logger.warning("ETF flow fetch failed: %s", exc)
            return None
        if raw is None or raw.empty:
            return None
        return _normalize(raw)

    def _cache_is_fresh(self) -> bool:
        try:
            mtime = self._cache_file.stat().st_mtime
        except OSError:
            return False
        age_hours = (datetime.now(UTC).timestamp() - mtime) / 3600.0
        return age_hours < self._cache_ttl_hours

    def _load_cache(self) -> pd.DataFrame | None:
        if not self._cache_file.exists():
            return None
        try:
            return _normalize(pd.read_parquet(self._cache_file))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ETF flow cache unreadable (%s); ignoring", exc)
            return None

    def _save_cache(self, df: pd.DataFrame) -> None:
        """Atomic parquet write (temp + os.replace), mirroring CachedDataProvider."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._cache_file.with_suffix(".parquet.tmp")
            df.to_parquet(tmp)
            # Verify the temp file reads back before swapping it in.
            pd.read_parquet(tmp)
            os.replace(tmp, self._cache_file)
        except Exception as exc:  # noqa: BLE001 - caching is best-effort
            logger.warning("ETF flow cache write failed: %s", exc)

    def _load_seed(self) -> pd.DataFrame | None:
        if not self._seed_path.exists():
            return None
        try:
            return _normalize(pd.read_csv(self._seed_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("ETF flow seed unreadable (%s)", exc)
            return None


# --- module-level helpers (pure; unit-tested directly) ---------------------


def compute_flow_features(
    flows: pd.DataFrame,
    as_of: datetime,
    *,
    asset_col: str = BTC_FLOW_COL,
    short_window: int = DEFAULT_ETF_FLOW_ZSCORE_SHORT_WINDOW,
    long_window: int = DEFAULT_ETF_FLOW_ZSCORE_LONG_WINDOW,
) -> dict[str, float | None]:
    """Compute net-flow z-scores and the consecutive-outflow-day count.

    The z-score is the latest day's flow relative to the mean/std over the
    trailing window (so a large outflow prints a strongly negative z-score).
    """
    empty = {"netflow_zscore_5d": None, "netflow_zscore_20d": None, "consecutive_outflow_days": 0.0}
    if flows is None or flows.empty or asset_col not in flows.columns:
        return empty
    as_of = _as_utc(as_of)
    series = flows.loc[flows.index <= as_of, asset_col].dropna()
    if series.empty:
        return empty

    def _zscore(window: int, baseline: int = long_window) -> float | None:
        """Z-score of the latest ``window``-day rolling-mean flow, standardized
        against its own last ``baseline`` observations.

        This measures the flow *regime*, not a single-day anomaly: a sustained
        outflow streak pushes the rolling mean well below its recent norm and
        prints a strongly negative z-score (which the long gate keys on), while
        a recovering flow trend prints positive.
        """
        if window < 1 or baseline < 2:
            return None
        roll = series.rolling(window).mean().dropna()
        if len(roll) < baseline:
            return None
        recent = roll.iloc[-baseline:]
        std = float(recent.std(ddof=0))
        mean = float(recent.mean())
        if not math.isfinite(std) or std == 0.0:
            return 0.0
        z = (float(roll.iloc[-1]) - mean) / std
        return z if math.isfinite(z) else None

    # Consecutive trailing days of net outflow (flow < 0).
    consecutive = 0
    for value in reversed(series.tolist()):
        if value < 0:
            consecutive += 1
        else:
            break

    return {
        "netflow_zscore_5d": _zscore(short_window),
        "netflow_zscore_20d": _zscore(long_window),
        "consecutive_outflow_days": float(consecutive),
    }


def _default_fetch(start: datetime, end: datetime) -> pd.DataFrame | None:  # pragma: no cover
    """Best-effort upstream fetch (Farside Investors daily CSV).

    Kept intentionally simple and defensive: any failure returns ``None`` so the
    provider degrades to cache/seed. Network access to the source is not
    guaranteed in all environments; operators with access can override
    ``fetch_fn`` with a richer scraper/API client.
    """
    try:
        import requests  # type: ignore[import-untyped]  # requests ships no inline types

        url = "https://farside.co.uk/bitcoin-etf-flow-all-data/"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            return None
        return tables[0]
    except Exception as exc:  # noqa: BLE001
        logger.info("ETF flow default fetch unavailable: %s", exc)
        return None


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def _empty_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="date")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in FLOW_COLUMNS}, index=idx)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce arbitrary input into a UTC daily DatetimeIndex + flow columns."""
    out = df.copy()
    # Locate a date column if the index is not already datetime.
    if not isinstance(out.index, pd.DatetimeIndex):
        date_col = next(
            (c for c in out.columns if str(c).lower() in {"date", "day", "timestamp"}), None
        )
        if date_col is not None:
            out = out.set_index(date_col)
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()]
    out.index = out.index.normalize()
    out.index.name = "date"
    # Keep only known flow columns that are present; coerce to float.
    keep = [c for c in FLOW_COLUMNS if c in out.columns]
    out = out[keep] if keep else out
    for c in keep:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def _covers(df: pd.DataFrame, start: datetime, end: datetime) -> bool:
    if df.empty:
        return False
    return df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end).normalize()


def _slice(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (df.index >= pd.Timestamp(start).normalize()) & (df.index <= pd.Timestamp(end))
    return df.loc[mask]


def _merge(cached: pd.DataFrame | None, fresh: pd.DataFrame) -> pd.DataFrame:
    if cached is None or cached.empty:
        return fresh
    combined = pd.concat([cached, fresh])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined
