from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import cast

# Ensure project root and src are in sys.path for absolute imports
from src.infrastructure.runtime.paths import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(1, str(SRC_PATH))

from src.infrastructure.logging.config import configure_logging
from src.strategies import (
    call_strategy_factory,
    create_adaptive_trend_strategy,
    create_ensemble_weighted_strategy,
    create_hyper_growth_strategy,
    create_kelly_momentum_strategy,
    create_leveraged_regime_strategy,
    create_ml_adaptive_strategy,
    create_ml_basic_strategy,
    create_ml_sentiment_strategy,
    create_momentum_leverage_strategy,
)
from src.strategies.components import Strategy
from src.strategies.exam_target_redesign import (
    MODEL_VERSION_OVERRIDE_ENV_VAR,
    create_exam_binary_direction_strategy,
    create_exam_meta_label_strategy,
    create_exam_smoothed_return_strategy,
    create_exam_triple_barrier_strategy,
)
from src.trading.symbols.factory import SymbolFactory

logger = logging.getLogger("atb.backtest")

# Registry model type each pinnable strategy scores with — required to
# resolve --model-as-of (and to detect promotion boundaries) BEFORE the
# strategy is constructed. Mirrors the model_type each factory wires into
# its MLBasicSignalGenerator; exam strategies are absent deliberately (they
# consume MODEL_VERSION_OVERRIDE_ENV_VAR and span several registry
# namespaces, so --model-version is their only supported pin).
_PINNABLE_STRATEGY_MODEL_TYPES: dict[str, str] = {
    "ml_basic": "basic",
    "hyper_growth": "basic",
}

# Exam-only strategies read the pin from MODEL_VERSION_OVERRIDE_ENV_VAR at
# construction (PR #950's mechanism) instead of a model_version kwarg.
_EXAM_STRATEGY_NAMES = frozenset(
    {
        "exam_binary_direction",
        "exam_triple_barrier",
        "exam_smoothed_return",
        "exam_meta_label",
    }
)


def _load_strategy(strategy_name: str, symbol: str | None = None, model_version: str | None = None):
    """Load a strategy by name, threading the trading symbol when supported.

    Mirrors the live runner (backtest-live parity): the symbol must reach ML
    signal generators so model registry selection matches the traded pair.

    ``model_version`` pins ML predictions to that exact registry version
    (GH #988): mainline factories receive it as an explicit kwarg via
    ``call_strategy_factory`` (which refuses factories that cannot honor
    it); exam factories pick it up from ``MODEL_VERSION_OVERRIDE_ENV_VAR``,
    set here only for the duration of construction.
    """
    # Define available strategies with their import paths and classes
    available_strategies: dict[str, Callable[..., Strategy]] = {
        "ml_basic": create_ml_basic_strategy,
        "ml_sentiment": create_ml_sentiment_strategy,
        "ml_adaptive": create_ml_adaptive_strategy,
        "ensemble_weighted": create_ensemble_weighted_strategy,
        "momentum_leverage": create_momentum_leverage_strategy,
        "adaptive_trend": create_adaptive_trend_strategy,
        "kelly_momentum": create_kelly_momentum_strategy,
        "leveraged_regime": create_leveraged_regime_strategy,
        "hyper_growth": create_hyper_growth_strategy,
        # TARGET-REDESIGN tournament exam-only strategies (GH #933, Phase 2b
        # item 4). EXAM-ONLY BY DESIGN -- backtest CLI is the ONLY place
        # these are registered. Do NOT add these to src/strategies/__init__
        # .py's general exports or src/engines/live/runner.py's strategy
        # dict; the tournament's research harness (ConfidenceWeightedSizer
        # + ratified risk-limits defaults) is deliberately separate from
        # anything a live/paper trading session could select.
        "exam_binary_direction": create_exam_binary_direction_strategy,
        "exam_triple_barrier": create_exam_triple_barrier_strategy,
        "exam_smoothed_return": create_exam_smoothed_return_strategy,
        "exam_meta_label": create_exam_meta_label_strategy,
    }

    try:
        builder = available_strategies.get(strategy_name)
        if builder is not None:
            if model_version is not None and strategy_name in _EXAM_STRATEGY_NAMES:
                return _call_exam_factory_pinned(builder, symbol, model_version)
            return call_strategy_factory(builder, symbol=symbol, model_version=model_version)

        print(f"Unknown strategy: {strategy_name}")
        print(f"Available strategies: {', '.join(available_strategies.keys())}")
        raise SystemExit(1)
    except Exception as exc:
        logger.error(f"Error loading strategy: {exc}")
        raise


def _call_exam_factory_pinned(
    builder: Callable[..., Strategy], symbol: str | None, model_version: str
) -> Strategy:
    """Construct an exam strategy with the version pin set in its env var.

    Exam factories read ``MODEL_VERSION_OVERRIDE_ENV_VAR`` at construction
    time only, so the override is scoped to this call and restored after —
    it must never leak into the wider process (or a later unpinned
    construction in the same process would silently inherit the pin).
    """
    prior = os.environ.get(MODEL_VERSION_OVERRIDE_ENV_VAR)
    os.environ[MODEL_VERSION_OVERRIDE_ENV_VAR] = model_version
    try:
        return call_strategy_factory(builder, symbol=symbol)
    finally:
        if prior is None:
            os.environ.pop(MODEL_VERSION_OVERRIDE_ENV_VAR, None)
        else:
            os.environ[MODEL_VERSION_OVERRIDE_ENV_VAR] = prior


def _parse_as_of(value: str) -> datetime:
    """argparse type for --model-as-of: ISO date/datetime, naive means UTC."""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r} — use YYYY-MM-DD or an ISO 8601 datetime"
        ) from exc
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _resolve_model_pin(
    ns: argparse.Namespace, start_date: datetime, end_date: datetime
) -> str | None:
    """Resolve --model-version/--model-as-of into a registry version id.

    For --model-as-of, resolution asks "which version WAS latest at that
    date?" using bundle metadata timestamps (see
    ``src/prediction/models/version_resolver.py``). Either way, when the
    backtest window spans a model-promotion boundary a loud warning maps
    each window segment to the version that was live then — the run is
    still pinned to ONE version (never switched mid-backtest), so segments
    live-traded by another version are not comparable (GH #988).

    Returns None when no pin was requested (zero behavior change).
    """
    model_version: str | None = getattr(ns, "model_version", None)
    model_as_of: datetime | None = getattr(ns, "model_as_of", None)
    if model_version is None and model_as_of is None:
        return None

    from src.prediction.config import PredictionConfig
    from src.prediction.models.version_resolver import (
        list_version_records,
        promotion_segments,
        resolve_version_as_of,
    )

    # Same normalization the ML signal generator applies for registry
    # selection, so resolution looks at the directory the pin will load from.
    registry_symbol = SymbolFactory.to_exchange_symbol(ns.symbol, "binance")
    registry_path = PredictionConfig.from_config_manager().model_registry_path
    model_type = _PINNABLE_STRATEGY_MODEL_TYPES.get(ns.strategy)

    if model_as_of is not None:
        if model_type is None:
            print(
                f"--model-as-of is not supported for strategy '{ns.strategy}': its "
                f'registry model type is unknown, so "which version was latest at '
                f'that date" cannot be resolved. Pass --model-version with an '
                f"explicit version id instead."
            )
            raise SystemExit(1)
        record = resolve_version_as_of(registry_path, registry_symbol, model_type, model_as_of)
        model_version = record.version_id
        logger.info(
            "Resolved --model-as-of %s to %s/%s version %s (effective %s, from %s)",
            model_as_of.isoformat(),
            registry_symbol,
            model_type,
            model_version,
            record.effective_at.isoformat(),
            record.source,
        )

    if model_type is not None:
        records = list_version_records(registry_path, registry_symbol, model_type)
        segments = promotion_segments(records, start_date, end_date)
        if len(segments) > 1:
            segment_lines = "\n".join(
                f"  - {segment.version_id or 'NO VERSION YET (live ran a cross-symbol substitute or no model)'}: "
                f"{segment.start.isoformat()} -> {segment.end.isoformat()}"
                for segment in segments
            )
            logger.warning(
                "MODEL PROMOTION BOUNDARY INSIDE BACKTEST WINDOW: %d different "
                "%s/%s model registry states between %s and %s:\n%s\n"
                "The ENTIRE window will be scored with pinned version %s — the "
                "backtest never switches models mid-window, so segments that "
                "live-traded under a different version are NOT comparable to "
                "live results (GH #988; promotion record: "
                "docs/research/model-promotions.md).",
                len(segments),
                registry_symbol,
                model_type,
                start_date.isoformat(),
                end_date.isoformat(),
                segment_lines,
                model_version,
            )

    return model_version


def _get_date_range(args):
    if args.start and args.end:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
    elif args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end_date = datetime.now(UTC)
    elif args.days:
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=args.days)
    else:
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=30)
    return start_date, end_date


def _handle(ns: argparse.Namespace) -> int:
    try:
        from src.data_providers.feargreed_provider import FearGreedProvider
        from src.engines.backtest.engine import Backtester
        from src.engines.shared.risk_configuration import resolve_strategy_max_position_size
        from src.risk.risk_manager import RiskParameters

        configure_logging()

        start_date, end_date = _get_date_range(ns)

        model_version = _resolve_model_pin(ns, start_date, end_date)
        strategy = _load_strategy(ns.strategy, symbol=ns.symbol, model_version=model_version)
        logger.info(f"Loaded strategy: {strategy.name}")
        if model_version is not None:
            logger.info(f"Model version pinned for this backtest: {model_version}")

        if getattr(ns, "mock_data", False):
            # Test-only escape hatch (mirrors `atb live --mock-data`,
            # src/engines/live/runner.py): deterministic synthetic OHLCV,
            # zero network, zero disk cache -- never a real provider or
            # create_data_provider() dispatch. Never set for a real backtest.
            from src.data_providers.data_provider import DataProvider
            from src.data_providers.mock_data_provider import MockDataProvider

            data_provider: DataProvider = MockDataProvider(seed=42)
            logger.info("Using MockDataProvider (--mock-data): no network, no cache")
        else:
            # Provider - use factory for automatic failover support
            from src.data_providers.provider_factory import create_data_provider

            provider = create_data_provider(provider_type=ns.provider)
            if ns.no_cache:
                data_provider = provider
                logger.info("Data caching disabled")
            else:
                from src.data_providers.cached_data_provider import CachedDataProvider

                # Determine appropriate cache TTL based on provider state
                from src.infrastructure.runtime.cache import (
                    DataProviderProtocol,
                    get_cache_ttl_for_provider,
                )

                # cast: get_cache_ttl_for_provider only reads _client behind a hasattr guard,
                # so any provider instance is safe even if it lacks the attribute
                cache_ttl = get_cache_ttl_for_provider(
                    cast(DataProviderProtocol, provider), ns.cache_ttl
                )
                cached_provider = CachedDataProvider(provider, cache_ttl_hours=cache_ttl)
                data_provider = cached_provider
                logger.info(f"Using cached data provider (TTL: {cache_ttl} hours)")
                cache_info = cached_provider.get_cache_info()
                logger.info(
                    f"Cache info: {cache_info['total_files']} files, "
                    f"{cache_info['total_size_mb']} MB"
                )

        sentiment_provider = None
        if ns.use_sentiment:
            sentiment_provider = FearGreedProvider()
            logger.info("Using sentiment analysis in backtest")

        risk_params_kwargs = {
            "base_risk_per_trade": ns.risk_per_trade,
            "max_risk_per_trade": ns.max_risk_per_trade,
            "max_drawdown": ns.max_drawdown,
        }
        if ns.max_position_size is not None:
            if not 0 < ns.max_position_size <= 1:
                raise ValueError(
                    f"--max-position-size must be in (0, 1], got {ns.max_position_size}"
                )
            risk_params_kwargs["max_position_size"] = ns.max_position_size
        else:
            # Honor strategy-level max_fraction (e.g., trend-following uses 95%
            # allocation instead of the default 10% cap) via the seam shared
            # with ExperimentRunner, so CLI and harness sizing cannot drift.
            strategy_max_position = resolve_strategy_max_position_size(strategy)
            if strategy_max_position is not None:
                risk_params_kwargs["max_position_size"] = strategy_max_position
        risk_params = RiskParameters(**risk_params_kwargs)

        # Default to no database logging for performance, unless explicitly enabled
        enable_db_logging = ns.log_to_db

        # Determine engine risk exit behavior
        enable_engine_risk_exits = not ns.disable_engine_sl

        # Disable dynamic risk management for strategies that manage their own risk
        # via signal timing (e.g., trend-following). Dynamic risk reduces position
        # sizes after losses, which undermines fully-invested strategies.
        # Strategies opt out by setting enable_dynamic_risk=False in risk overrides.
        enable_dynamic_risk = True
        if hasattr(strategy, "get_risk_overrides"):
            overrides = strategy.get_risk_overrides()
            if isinstance(overrides, dict) and overrides.get("enable_dynamic_risk") is False:
                enable_dynamic_risk = False

        backtester = Backtester(
            strategy=strategy,
            data_provider=data_provider,
            sentiment_provider=sentiment_provider,
            risk_parameters=risk_params,
            initial_balance=ns.initial_balance,
            log_to_database=enable_db_logging,
            enable_engine_risk_exits=enable_engine_risk_exits,
            enable_dynamic_risk=enable_dynamic_risk,
        )

        # Map provider types to exchange names for symbol conversion
        # "auto" uses Binance first, "coingecko" handles conversion internally
        provider_for_symbol = {
            "auto": "binance",  # Auto tries Binance first
            "coingecko": "binance",  # Use Binance format, CoinGecko converts internally
        }.get(ns.provider, ns.provider)

        trading_symbol = SymbolFactory.to_exchange_symbol(ns.symbol, provider_for_symbol)

        results = backtester.run(
            symbol=trading_symbol, timeframe=ns.timeframe, start=start_date, end=end_date
        )

        print("\nBacktest Results:")
        print("=" * 50)
        print(f"Strategy: {strategy.name}")
        print(f"Symbol: {trading_symbol}")
        print(f"Period: {start_date.date()} to {end_date.date()}")
        print(f"Timeframe: {ns.timeframe}")
        print(f"Using Sentiment: {ns.use_sentiment}")
        print(f"Using Cache: {not ns.no_cache}")
        print(f"Database Logging: {enable_db_logging}")
        print("-" * 50)
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annualized_return']:.2f}%")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Final Balance: ${results['final_balance']:.2f}")
        print(f"Hold Return: {results['hold_return']:.2f}%")
        print(f"Trading vs Hold: {results['trading_vs_hold_difference']:+.2f}%")
        print("=" * 50)

        if enable_db_logging and results.get("session_id"):
            print(f"Database Session ID: {results['session_id']}")
            print("=" * 50)

        if results.get("yearly_returns"):
            print("Yearly Returns:")
            print(f"{'Year':<8} {'Return (%)':>12}")
            for year in sorted(results["yearly_returns"].keys()):
                print(f"{year:<8} {results['yearly_returns'][year]:>12.2f}")
            print("=" * 50)

        if not ns.no_cache and not getattr(ns, "mock_data", False):
            # cached_provider is always bound here: it is assigned on the same
            # `not ns.no_cache and not mock_data` branch above.
            final_cache_info = cached_provider.get_cache_info()
            logger.info(
                f"Final cache info: {final_cache_info['total_files']} files, {final_cache_info['total_size_mb']} MB"
            )

        try:
            import re

            duration_years = round((end_date - start_date).days / 365.25, 2)
            timestamp_for_file = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            sanitized_strategy_name = re.sub(r"[^a-zA-Z0-9_-]", "_", strategy.name)
            filename = f"{timestamp_for_file}_{sanitized_strategy_name}_{duration_years}yrs.json"
            logs_dir = PROJECT_ROOT / "logs" / "backtest"
            logs_dir.mkdir(parents=True, exist_ok=True)
            filepath = logs_dir / filename
            with open(filepath, "w") as _f:
                json.dump(
                    {
                        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
                        "strategy": strategy.name,
                        "symbol": trading_symbol,
                        "timeframe": ns.timeframe,
                        "start_date": start_date.isoformat(),
                        "end_date": end_date.isoformat(),
                        "duration_years": duration_years,
                        "initial_balance": ns.initial_balance,
                        "use_sentiment": ns.use_sentiment,
                        "use_cache": not ns.no_cache,
                        "database_logging": enable_db_logging,
                        "results": results,
                    },
                    _f,
                    indent=2,
                )
            logger.info(f"Backtest log saved to {filepath.relative_to(PROJECT_ROOT)}")
        except Exception as log_err:
            logger.warning(f"Failed to write backtest log: {log_err}")

        if sentiment_provider is not None:
            df = data_provider.get_historical_data(ns.symbol, ns.timeframe, start_date, end_date)
            sentiment_df = sentiment_provider.get_historical_sentiment(
                ns.symbol, start_date, end_date
            )
            if not sentiment_df.empty:
                sentiment_df = sentiment_provider.aggregate_sentiment(
                    sentiment_df, window=ns.timeframe
                )
                aligned_df = df.join(sentiment_df, how="left")
                print(f"Shape of aligned DataFrame: {aligned_df.shape}")
                if aligned_df.empty:
                    print("Warning: aligned DataFrame is empty. No file will be written.")
                output_path = PROJECT_ROOT / "data" / "sentiment_aligned_output.csv"
                try:
                    aligned_df.to_csv(output_path)
                    print(f"Aligned sentiment and price data saved to {output_path}")
                except Exception as file_err:
                    print(f"Error writing {output_path}: {file_err}")
                    logger.error(f"Error writing {output_path}: {file_err}")

        return 0
    except SystemExit:
        raise
    except Exception as exc:
        logger.error(f"Error running backtest: {exc}")
        return 1


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("backtest", help="Run strategy backtest")
    p.add_argument("strategy", help="Strategy name - e.g., ml_basic")
    p.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol")
    p.add_argument("--timeframe", default="1h", help="Candle timeframe")
    p.add_argument("--days", type=int, default=30, help="Number of days to backtest")
    p.add_argument("--start", help="Start date - YYYY-MM-DD")
    p.add_argument("--end", help="End date - YYYY-MM-DD")
    from src.config.constants import DEFAULT_INITIAL_BALANCE

    p.add_argument(
        "--initial-balance", type=float, default=DEFAULT_INITIAL_BALANCE, help="Initial balance"
    )
    p.add_argument(
        "--risk-per-trade", type=float, default=0.01, help="Risk per trade - 1 percent equals 0.01"
    )
    p.add_argument("--max-risk-per-trade", type=float, default=0.02, help="Maximum risk per trade")
    p.add_argument(
        "--use-sentiment", action="store_true", help="Use sentiment analysis in backtest"
    )
    p.add_argument("--no-cache", action="store_true", help="Disable data caching")
    p.add_argument("--cache-ttl", type=int, default=24, help="Cache TTL in hours - default: 24")
    p.add_argument(
        "--log-to-db",
        action="store_true",
        help="Enable database logging for this backtest - slower but provides detailed logs",
    )
    p.add_argument(
        "--provider",
        choices=["auto", "binance", "coinbase", "coingecko"],
        default="auto",
        help="Data provider: auto=Binance→CoinGecko failover (recommended), binance=Binance only, coinbase=Coinbase only, coingecko=CoinGecko only - default: auto",
    )
    p.add_argument(
        "--mock-data",
        action="store_true",
        help="Use deterministic synthetic OHLCV data instead of a real "
        "provider (mirrors `atb live --mock-data`). Test/CI use only -- "
        "never set for a real backtest.",
    )
    p.add_argument(
        "--max-drawdown",
        type=float,
        default=0.5,
        help="Maximum drawdown before stopping - default: 0.5 (50 percent)",
    )
    p.add_argument(
        "--max-position-size",
        type=float,
        default=None,
        help="Maximum position size as fraction of balance (0.1 = 10 percent) - default: 0.1",
    )
    p.add_argument(
        "--disable-engine-sl",
        action="store_true",
        help="Disable engine-level stop loss and take profit checks (strategy signals only)",
    )
    pin_group = p.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--model-version",
        default=None,
        help="Pin ML predictions to this exact registry version (e.g. "
        "2026-07-04_22h_v1) instead of resolving 'latest' at invocation time. "
        "Required for honest comparisons against live windows that predate "
        "the current 'latest' model.",
    )
    pin_group.add_argument(
        "--model-as-of",
        type=_parse_as_of,
        default=None,
        metavar="DATE",
        help="Pin ML predictions to whichever registry version was 'latest' "
        "at this UTC date/datetime (YYYY-MM-DD or ISO 8601), resolved from "
        "bundle metadata timestamps. Warns when the backtest window spans a "
        "model-promotion boundary.",
    )
    p.set_defaults(func=_handle)
