"""Per-deployment strategy configuration — single source of truth (GH #1020).

Deployment-level decisions (a strategy locked to long-only on a specific
symbol) live here, in exactly one named place, instead of scattered factory
defaults or environment variables. Both engines consume these values through
the same path — ``call_strategy_factory`` -> strategy factory -> this module —
so a backtest of a deployment resolves the identical effective configuration
as live with zero extra flags (CODE.md Backtest-Live Parity; risk-review
condition C1 on proposal 2026-07-12-01).

Current entries:

- ``hyper_growth`` / ``ETHUSDT`` is long-only by design (board-approved
  proposal 2026-07-12-01, GH #1020): short trades' standalone P&L was
  negative in every counterfactual fold tested, so the accidental short
  suppression found in GH #990 is codified as an explicit configuration.
  Re-enable criteria are recorded in the risk review
  (docs/research/risk-snapshots/2026-07-12_2000_risk-review_1020-*.md).

A repo-wide risk-config consolidation (GH #986) is being designed in
parallel; this module is deliberately small so it can migrate wholesale.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from src.trading.symbols.factory import SymbolFactory

# Strategy factory key -> symbols whose deployments never ENTER shorts.
# Keys match the registry names used by the live runner and backtest CLI.
# Symbols are stored in Binance exchange format; lookups normalize
# provider-specific formats (e.g. Coinbase "ETH-USD") before matching.
LONG_ONLY_DEPLOYMENTS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "hyper_growth": frozenset({"ETHUSDT"}),
    }
)


def resolve_allow_shorts(strategy_key: str, symbol: str | None) -> bool:
    """Return whether the ``strategy_key``/``symbol`` deployment may enter shorts.

    ``True`` (shorts allowed) for every deployment not listed in
    :data:`LONG_ONLY_DEPLOYMENTS`, including unknown strategies, ``None``
    symbols, and symbols that fail normalization — an invalid symbol fails
    fast later at signal-generator construction, which owns that validation.

    Args:
        strategy_key: Factory registry name (e.g. ``"hyper_growth"``).
        symbol: Trading symbol in any supported format, or None.
    """
    if symbol is None:
        return True
    long_only_symbols = LONG_ONLY_DEPLOYMENTS.get(strategy_key)
    if not long_only_symbols:
        return True
    try:
        normalized = SymbolFactory.to_exchange_symbol(symbol, "binance")
    except ValueError:
        return True
    return normalized not in long_only_symbols


__all__ = ["LONG_ONLY_DEPLOYMENTS", "resolve_allow_shorts"]
