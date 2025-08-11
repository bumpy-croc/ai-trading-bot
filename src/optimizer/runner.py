from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pathlib import Path

from src.optimizer.schemas import ExperimentConfig, ExperimentResult
from src.backtesting.engine import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.mock_data_provider import MockDataProvider
from src.data_providers.data_provider import DataProvider
from src.strategies.ml_basic import MlBasic
from src.risk.risk_manager import RiskParameters
import pandas as pd


class _FixtureProvider(DataProvider):
    """Lightweight provider that serves data from tests/data feather file if available."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        import pandas as pd
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_feather(self.path)
        df.set_index("timestamp", inplace=True)
        return df

    def get_historical_data(self, symbol: str, timeframe: str, start: datetime, end: Optional[datetime] = None) -> pd.DataFrame:  # type: ignore[override]
        if self.df.empty:
            return self.df
        end = end or pd.Timestamp.now()
        df = self.df.loc[(self.df.index >= pd.Timestamp(start)) & (self.df.index <= pd.Timestamp(end))].copy()
        return df


class ExperimentRunner:
    """Runs backtests for given experiment configurations."""

    def _load_provider(self, name: str, use_cache: bool, cache_ttl_hours: int = 24):
        name = (name or "binance").lower()
        if name == "mock":
            # Use hourly candles over ~90 days for stability
            return MockDataProvider(interval_seconds=3600, num_candles=24 * 90)
        if name == "fixture":
            fixture_path = Path("tests/data/BTCUSDT_1h_2023-01-01_2024-12-31.feather")
            return _FixtureProvider(fixture_path)

        if name == "coinbase":
            provider = CoinbaseProvider()
        else:
            provider = BinanceProvider()
        if use_cache:
            return CachedDataProvider(provider, cache_ttl_hours=cache_ttl_hours)
        return provider

    def _load_strategy(self, strategy_name: str) -> MlBasic:
        if strategy_name == "ml_basic":
            return MlBasic()
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        strategy = self._load_strategy(config.strategy_name)
        provider = self._load_provider(config.provider, config.use_cache)

        risk_params = RiskParameters(**config.risk_parameters) if config.risk_parameters else RiskParameters()

        backtester = Backtester(
            strategy=strategy,
            data_provider=provider,
            sentiment_provider=None,
            risk_parameters=risk_params,
            initial_balance=config.initial_balance,
            log_to_database=False,
        )

        results = backtester.run(
            symbol=config.symbol,
            timeframe=config.timeframe,
            start=config.start,
            end=config.end,
        )

        return ExperimentResult(
            config=config,
            total_trades=int(results.get("total_trades", 0)),
            win_rate=float(results.get("win_rate", 0.0)),
            total_return=float(results.get("total_return", 0.0)),
            annualized_return=float(results.get("annualized_return", 0.0)),
            max_drawdown=float(results.get("max_drawdown", 0.0)),
            sharpe_ratio=float(results.get("sharpe_ratio", 0.0)),
            final_balance=float(results.get("final_balance", config.initial_balance)),
            session_id=results.get("session_id"),
        )

    def run_sweep(self, configs: List[ExperimentConfig]) -> List[ExperimentResult]:
        return [self.run(cfg) for cfg in configs]