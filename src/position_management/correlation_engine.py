from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from src.config.constants import (
	DEFAULT_CORRELATION_SAMPLE_MIN_SIZE,
	DEFAULT_CORRELATION_THRESHOLD,
	DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS,
	DEFAULT_CORRELATION_WINDOW_DAYS,
	DEFAULT_MAX_CORRELATED_EXPOSURE,
)


@dataclass
class CorrelationConfig:
	correlation_window_days: int = DEFAULT_CORRELATION_WINDOW_DAYS
	correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD
	max_correlated_exposure: float = DEFAULT_MAX_CORRELATED_EXPOSURE
	correlation_update_frequency_hours: int = DEFAULT_CORRELATION_UPDATE_FREQUENCY_HOURS
	sample_min_size: int = DEFAULT_CORRELATION_SAMPLE_MIN_SIZE


class CorrelationEngine:
	"""
	Computes rolling correlations between symbols and tracks correlated exposure.
	
	- calculate_position_correlations: compute correlation matrix for provided price series
	- get_correlation_groups: group symbols by threshold
	- get_correlated_exposure: compute total exposure in each correlated group
	
	Note: Persistence is handled by the database layer; the engine exposes data for logging/caching.
	"""

	def __init__(self, config: CorrelationConfig | None = None):
		self.config = config or CorrelationConfig()
		self._last_update_at: Optional[datetime] = None
		self._last_matrix: Optional[pd.DataFrame] = None
		# Lightweight memoization to avoid expensive recomputation on each call
		self._last_signature: Optional[Tuple[Tuple[str, ...], Optional[pd.Timestamp], int]] = None

	def should_update(self, now: Optional[datetime] = None) -> bool:
		now = now or datetime.utcnow()
		if self._last_update_at is None:
			return True
		return now - self._last_update_at >= timedelta(hours=self.config.correlation_update_frequency_hours)

	def calculate_position_correlations(
		self,
		price_series_by_symbol: Dict[str, pd.Series],
		now: Optional[datetime] = None,
	) -> pd.DataFrame:
		"""Compute rolling Pearson correlation on aligned close-price series.
		
		Args:
			price_series_by_symbol: mapping symbol -> price series indexed by datetime
		Returns:
			Correlation matrix DataFrame (symbols x symbols)
		"""
		if not price_series_by_symbol:
			return pd.DataFrame()

		# Align and restrict to window
		end_time = None
		for s in price_series_by_symbol.values():
			if not s.empty:
				end_time = max(end_time or s.index.max(), s.index.max())
		if end_time is None:
			return pd.DataFrame()
		start_time = end_time - timedelta(days=self.config.correlation_window_days)

		aligned: List[pd.Series] = []
		cols: List[str] = []
		for symbol, series in price_series_by_symbol.items():
			if series is None or series.empty:
				continue
			windowed = series.loc[(series.index >= start_time) & (series.index <= end_time)]
			if len(windowed) < self.config.sample_min_size:
				continue
			aligned.append(windowed.rename(symbol))
			cols.append(symbol)

		if not aligned:
			return pd.DataFrame()

		prices = pd.concat(aligned, axis=1).dropna(how="any")
		if prices.shape[0] < self.config.sample_min_size or prices.shape[1] < 2:
			return pd.DataFrame()

		# Memoization guard: if we recently computed the same window/signature, return cached matrix
		signature: Tuple[Tuple[str, ...], Optional[pd.Timestamp], int] = (
			tuple(sorted(prices.columns)),
			prices.index[-1] if not prices.empty else None,
			prices.shape[0],
		)
		if (
			self._last_matrix is not None
			and self._last_signature == signature
			and not self.should_update(now)
		):
			return self._last_matrix

		# Use returns for correlation robustness
		returns = prices.pct_change().dropna(how="any")
		corr = returns.corr()
		self._last_matrix = corr
		self._last_update_at = now or datetime.utcnow()
		self._last_signature = signature
		return corr

	def get_correlation_groups(self, corr_matrix: pd.DataFrame | None = None) -> List[List[str]]:
		"""Group symbols where pairwise correlation >= threshold using union-find style clustering."""
		corr = corr_matrix if corr_matrix is not None else self._last_matrix
		if corr is None or corr.empty:
			return []

		symbols = list(corr.columns)
		parent = {s: s for s in symbols}

		def find(x: str) -> str:
			while parent[x] != x:
				parent[x] = parent[parent[x]]
				x = parent[x]
			return x

		def union(a: str, b: str) -> None:
			ra = find(a)
			rb = find(b)
			if ra != rb:
				parent[rb] = ra

		thr = float(self.config.correlation_threshold)
		for i, a in enumerate(symbols):
			for j in range(i + 1, len(symbols)):
				b = symbols[j]
				val = corr.at[a, b]
				if pd.notna(val) and val >= thr:
					union(a, b)

		groups: Dict[str, List[str]] = {}
		for s in symbols:
			root = find(s)
			groups.setdefault(root, []).append(s)
		# Only groups of size >= 2 are correlated groups
		return [g for g in groups.values() if len(g) >= 2]

	def get_correlated_exposure(
		self,
		positions: Dict[str, Dict[str, Any]],
		groups: List[List[str]],
	) -> Dict[Tuple[str, ...], float]:
		"""Aggregate exposure per correlation group.
		positions: mapping symbol -> { 'size': fraction }
		Returns mapping group(tuple(symbols)) -> total_exposure(float)
		"""
		exposures: Dict[Tuple[str, ...], float] = {}
		for group in groups:
			total = 0.0
			for sym in group:
				info = positions.get(sym)
				if info:
					total += float(info.get("size", 0.0))
			exposures[tuple(sorted(group))] = round(total, 8)
		return exposures

	def compute_size_reduction_factor(
		self,
		positions: Dict[str, Dict[str, Any]],
		corr_matrix: pd.DataFrame | None,
		candidate_symbol: str,
		candidate_fraction: float,
	) -> float:
		"""Return factor in [0,1] to reduce candidate_fraction if group exposure exceeds max.
		- Find candidate's correlated group
		- Compute projected exposure with candidate included
		- If projected > max_correlated_exposure, scale down proportionally
		"""
		if not candidate_symbol or candidate_fraction <= 0:
			return 1.0
		groups = self.get_correlation_groups(corr_matrix)
		if not groups:
			return 1.0
		# Identify groups containing candidate
		affected_groups = [g for g in groups if candidate_symbol in g]
		if not affected_groups:
			return 1.0
		max_allowed = float(self.config.max_correlated_exposure)
		factor = 1.0
		for g in affected_groups:
			current = 0.0
			for sym in g:
				current += float(positions.get(sym, {}).get("size", 0.0))
			projected = current + candidate_fraction
			if projected > max_allowed and projected > 0:
				factor = min(factor, max(0.0, max_allowed / projected))
		return factor


class CorrelationGroupManager:
	"""Manages named correlation groups and exposures for reporting/enforcement."""

	def __init__(self, group_definitions: Optional[Dict[str, List[str]]] = None):
		self.group_definitions = group_definitions or {}

	def set_groups(self, group_definitions: Dict[str, List[str]]) -> None:
		self.group_definitions = group_definitions or {}

	def calculate_group_exposures(self, positions: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
		"""Return mapping group_name -> { total_exposure, position_count, symbols }"""
		out: Dict[str, Dict[str, Any]] = {}
		for name, symbols in self.group_definitions.items():
			total = 0.0
			present: List[str] = []
			for s in symbols:
				if s in positions:
					present.append(s)
					total += float(positions[s].get("size", 0.0))
			out[name] = {
				"total_exposure": round(total, 8),
				"position_count": len(present),
				"symbols": present,
				"last_updated": datetime.utcnow().isoformat(),
			}
		return out