"""
SymbolFactory: Centralized utility for formatting trading symbols for different exchanges.

Usage:
    SymbolFactory.to_exchange_symbol('BTC-USD', 'binance')  # 'BTCUSDT'
    SymbolFactory.to_exchange_symbol('BTCUSDT', 'coinbase') # 'BTC-USD'
    SymbolFactory.to_exchange_symbol('ETH-USD', 'binance')  # 'ETHUSDT'
    SymbolFactory.to_exchange_symbol('ETHUSDT', 'coinbase') # 'ETH-USD'

Add more exchanges as needed.
"""

import re


def base_asset_from_symbol(symbol: str) -> str:
    """Best-effort base asset for ``symbol`` by stripping a known quote suffix.

    Fallback only — prefer the exchange-authoritative ``base_asset`` from
    ``get_symbol_info`` when available. Quotes are checked longest-first so e.g.
    ``ETHUSDT`` resolves to ``ETH``. Returns ``symbol`` unchanged when no known
    quote matches.
    """
    for quote in ("USDT", "BUSD", "USDC", "USD"):
        if symbol.endswith(quote) and len(symbol) > len(quote):
            return symbol[: -len(quote)]
    return symbol


class SymbolFactory:
    @staticmethod
    def to_exchange_symbol(symbol: str, exchange: str) -> str:
        """
        Convert a generic or other-exchange symbol to the format required by the target exchange.
        Args:
            symbol: Symbol in any supported format (e.g., 'BTC-USD', 'BTCUSDT')
            exchange: Target exchange ('binance', 'coinbase', ...)
        Returns:
            str: Symbol formatted for the target exchange
        """
        symbol = symbol.upper()
        if exchange == "binance":
            # Convert 'BTC-USD' or 'BTC/USD' to 'BTCUSDT'
            if "-" in symbol:
                parts = symbol.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Invalid symbol format '{symbol}': expected 'BASE-QUOTE'")
                base, quote = parts
                if quote == "USD":
                    quote = "USDT"  # Binance uses USDT
                return f"{base}{quote}"
            if "/" in symbol:
                parts = symbol.split("/")
                if len(parts) != 2:
                    raise ValueError(f"Invalid symbol format '{symbol}': expected 'BASE/QUOTE'")
                base, quote = parts
                if quote == "USD":
                    quote = "USDT"
                return f"{base}{quote}"
            # Already Binance style
            return symbol
        elif exchange == "coinbase":
            # Convert 'BTCUSDT' or 'BTC/USDT' to 'BTC-USD'
            if symbol.endswith("USDT"):
                base = symbol[:-4]
                return f"{base}-USD"
            if symbol.endswith("USD"):
                base = symbol[:-3]
                return f"{base}-USD"
            if "/" in symbol:
                parts = symbol.split("/")
                if len(parts) != 2:
                    raise ValueError(f"Invalid symbol format '{symbol}': expected 'BASE/QUOTE'")
                base, quote = parts
                return f"{base}-{quote}"
            if "-" in symbol:
                return symbol
            # Fallback: try regex for 3-4 letter base/quote
            m = re.match(r"([A-Z]{3,5})([A-Z]{3,5})", symbol)
            if m:
                return f"{m.group(1)}-{m.group(2)}"
            raise ValueError(f"Could not parse symbol '{symbol}' for Coinbase format.")
        else:
            raise ValueError(f"Exchange '{exchange}' not supported in SymbolFactory.")
