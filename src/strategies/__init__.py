import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .adaptive_trend import create_adaptive_trend_strategy
from .ensemble_weighted import create_ensemble_weighted_strategy
from .hyper_growth import create_hyper_growth_strategy
from .kelly_momentum import create_kelly_momentum_strategy
from .leveraged_regime import create_leveraged_regime_strategy
from .ml_adaptive import create_ml_adaptive_strategy
from .ml_basic import create_ml_basic_strategy
from .ml_sentiment import create_ml_sentiment_strategy
from .momentum_leverage import create_momentum_leverage_strategy

if TYPE_CHECKING:
    from .components import Strategy

__all__ = [
    "create_ml_basic_strategy",
    "create_ml_adaptive_strategy",
    "create_ml_sentiment_strategy",
    "create_ensemble_weighted_strategy",
    "create_momentum_leverage_strategy",
    "create_adaptive_trend_strategy",
    "create_kelly_momentum_strategy",
    "create_leveraged_regime_strategy",
    "create_hyper_growth_strategy",
    "call_strategy_factory",
    "factory_accepts_symbol",
]


def factory_accepts_symbol(factory: Callable[..., Any]) -> bool:
    """Return whether ``factory`` has a ``symbol`` parameter (or ``**kwargs``).

    Shared by every caller that must decide whether to thread a trading
    symbol into a strategy factory — passing ``symbol=`` to a factory that
    doesn't accept it raises ``TypeError``, so callers check this first.
    """
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return False
    return "symbol" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def call_strategy_factory(
    factory: Callable[..., Any],
    *,
    symbol: str | None = None,
    model_version: str | None = None,
) -> "Strategy":
    """Invoke a strategy factory, threading the trading symbol when supported.

    Runners (live and backtest) must construct strategies through this helper
    so the symbol they trade reaches ML signal generators for model registry
    selection — otherwise a strategy silently scores with the default
    (BTCUSDT) model. Factories without a ``symbol`` parameter (e.g.
    chaos_test) are called unchanged.

    ``model_version`` is the backtest harness's point-in-time model pin
    (GH #988). Unlike ``symbol``, it is threaded only to factories that
    declare an explicit ``model_version`` parameter — never through
    ``**kwargs`` (a factory forwarding unknown kwargs elsewhere would crash
    or, worse, silently drop the pin). A pin the factory cannot honor
    raises so a "pinned" run can never silently score with ``latest``.

    Raises:
        ValueError: ``model_version`` was requested but the factory has no
            explicit ``model_version`` parameter.
    """
    kwargs: dict[str, Any] = {}
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = None

    if model_version is not None:
        if parameters is None or "model_version" not in parameters:
            raise ValueError(
                f"Strategy factory {getattr(factory, '__name__', factory)!r} does not "
                f"support model_version pinning — refusing to run a 'pinned' backtest "
                f"that would silently resolve 'latest' instead."
            )
        kwargs["model_version"] = model_version

    if symbol is not None and factory_accepts_symbol(factory):
        kwargs["symbol"] = symbol

    return factory(**kwargs)
