from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtesting.engine import Backtester
from src.data_providers.binance_provider import BinanceProvider
from src.data_providers.cached_data_provider import CachedDataProvider
from src.data_providers.coinbase_provider import CoinbaseProvider
from src.data_providers.data_provider import DataProvider
from src.optimizer.schemas import ExperimentConfig, ExperimentResult
from src.risk.risk_manager import RiskParameters
from src.strategies.components import Strategy
from src.strategies.ml_basic import create_ml_basic_strategy


class _FixtureProvider(DataProvider):
    """Lightweight provider that serves data from tests/data feather file if available."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.df = self._load()

    def _load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_feather(self.path)
        df.set_index("timestamp", inplace=True)
        return df

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:  # type: ignore[override]
        if self.df.empty:
            return self.df
        end = end or pd.Timestamp.now()
        df = self.df.loc[
            (self.df.index >= pd.Timestamp(start)) & (self.df.index <= pd.Timestamp(end))
        ].copy()
        return df


class _RandomWalkProvider(DataProvider):
    """Generates synthetic OHLCV series using a random walk for offline experiments."""

    def __init__(
        self,
        start: datetime,
        end: datetime,
        timeframe: str = "1h",
        start_price: float = 30000.0,
        vol: float = 0.01,
        seed: int | None = None,
    ):
        super().__init__()
        self.timeframe = timeframe
        self.seed = seed
        self.df = self._generate(start, end, timeframe, start_price, vol)

    def _freq(self, timeframe: str) -> str:
        mapping = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "4h": "4h", "1d": "1d"}
        return mapping.get(timeframe, "1h")

    def _generate(
        self, start: datetime, end: datetime, timeframe: str, start_price: float, vol: float
    ) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)
        idx = pd.date_range(
            start=pd.Timestamp(start), end=pd.Timestamp(end), freq=self._freq(timeframe)
        )
        if len(idx) < 2:
            return pd.DataFrame(
                index=idx, columns=["open", "high", "low", "close", "volume"]
            ).fillna(0.0)
        prices = [start_price]
        for _ in range(1, len(idx)):
            shock = np.random.normal(0, vol)
            prices.append(max(1.0, prices[-1] * (1.0 + shock)))
        prices = np.array(prices)
        highs = prices * (1.0 + np.abs(np.random.normal(0, vol / 2, size=len(prices))))
        lows = prices * (1.0 - np.abs(np.random.normal(0, vol / 2, size=len(prices))))
        opens = np.r_[prices[0], prices[:-1]]
        volume = np.random.uniform(1000.0, 10000.0, size=len(prices))
        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volume,
            },
            index=idx,
        )
        return df

    def get_historical_data(
        self, symbol: str, timeframe: str, start: datetime, end: datetime | None = None
    ) -> pd.DataFrame:  # type: ignore[override]
        end = end or pd.Timestamp.now()
        return self.df.loc[
            (self.df.index >= pd.Timestamp(start)) & (self.df.index <= pd.Timestamp(end))
        ].copy()

    def get_live_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:  # type: ignore[override]
        tail = self.df.tail(limit).copy()
        return tail

    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:  # type: ignore[override]
        return self.get_live_data(symbol, timeframe, limit=1)

    def get_current_price(self, symbol: str) -> float:  # type: ignore[override]
        if self.df.empty:
            return 0.0
        return float(self.df["close"].iloc[-1])


class ExperimentRunner:
    """Runs backtests for given experiment configurations."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def _load_provider(
        self,
        name: str,
        use_cache: bool,
        cache_ttl_hours: int = 24,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        timeframe: str = "1h",
        seed: int | None = None,
    ):
        name = (name or "binance").lower()
        if name == "mock":
            # Internal random-walk provider
            return _RandomWalkProvider(
                start or (datetime.utcnow() - timedelta(days=30)),
                end or datetime.utcnow(),
                timeframe=timeframe,
                seed=seed,
            )
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

    def _load_strategy(self, strategy_name: str) -> Strategy:
        if strategy_name == "ml_basic":
            return create_ml_basic_strategy()
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def _apply_parameter_overrides(self, strategy: Strategy, config: ExperimentConfig) -> None:
        if config.parameters and config.parameters.values:
            for key, value in config.parameters.values.items():
                if key.startswith("MlBasic."):
                    attr = key.split(".", 1)[1]
                    if hasattr(strategy, attr):
                        try:
                            setattr(strategy, attr, value)
                        except Exception as e:
                            # Ignore invalid attribute assignments to keep runner robust
                            self.logger.debug(f"Failed to set attribute {attr}={value}: {e}")
                            pass

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        strategy = self._load_strategy(config.strategy_name)
        # Apply any parameter overrides for strategy-level tuning
        self._apply_parameter_overrides(strategy, config)
        provider = self._load_provider(
            config.provider,
            config.use_cache,
            start=config.start,
            end=config.end,
            timeframe=config.timeframe,
            seed=config.random_seed,
        )

        risk_params = (
            RiskParameters(**config.risk_parameters) if config.risk_parameters else RiskParameters()
        )

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

    def run_sweep(self, configs: list[ExperimentConfig]) -> list[ExperimentResult]:
        return [self.run(cfg) for cfg in configs]
