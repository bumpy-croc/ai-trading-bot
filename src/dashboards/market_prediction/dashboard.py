from __future__ import annotations

# --- Ensure greenlet/eventlet is configured BEFORE any other imports.
# This is critical because eventlet.monkey_patch() must be called before
# importing any network-related modules like Flask, requests, etc.
import os

_WEB_SERVER_USE_EVENTLET = os.environ.get("WEB_SERVER_USE_EVENTLET", "0") == "1"
if _WEB_SERVER_USE_EVENTLET:
    import eventlet
    # Full monkey patching for production WSGI server
    eventlet.monkey_patch()
    _ASYNC_MODE = "eventlet"
else:
    _ASYNC_MODE = "threading"

# --- ALL imports must happen AFTER monkey patching to avoid threading issues ---

# Standard library imports
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template

# Project imports - all must happen after monkey patching
try:
    from src.data_providers.binance_provider import BinanceProvider
except Exception:  # pragma: no cover – fallback when deps unavailable
    BinanceProvider = None  # type: ignore

from src.data_providers.feargreed_provider import FearGreedProvider

logger = logging.getLogger(__name__)


class MarketPredictionDashboard:
    """Dashboard that shows BTC price forecasts over multiple horizons."""

    # Forecast horizons we will compute (in days)
    _HORIZONS_DAYS = [7, 30, 90]

    def __init__(self, symbol: str = "BTCUSDT", lookback_days: int = 365):
        self.symbol = symbol
        self.lookback_days = lookback_days
        base_path = Path(__file__).parent
        self.app = Flask(
            __name__,
            template_folder=str(base_path / "templates"),
            static_folder=str(base_path / "static"),
        )
        self._setup_routes()

        # Providers – create lazily to avoid heavy imports during unit tests
        self._price_provider = None
        self._sentiment_provider = FearGreedProvider()

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------
    def _setup_routes(self) -> None:  # noqa: D401
        @self.app.route("/")
        def index():  # noqa: D401
            return render_template("market_prediction_dashboard.html")

        @self.app.route("/api/predictions")
        def api_predictions():  # noqa: D401
            from flask import request

            symbol = request.args.get("symbol", self.symbol).upper()
            # Basic validation – ensure ends with 'USDT'
            if not symbol.endswith("USDT"):
                return jsonify({"error": "Symbol must end with USDT"}), 400
            data = self._generate_prediction_payload(symbol)
            return jsonify(data)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------
    def _get_price_provider(self):
        if self._price_provider is not None:
            return self._price_provider
        if BinanceProvider is None:
            logger.warning("BinanceProvider unavailable – falling back to offline mode")
            self._price_provider = None
            return None
        try:
            self._price_provider = BinanceProvider()
        except Exception as exc:  # pragma: no cover
            logger.warning("BinanceProvider init failed (%s). Using offline mode", exc)
            self._price_provider = None
        return self._price_provider

    def _load_price_history(self, symbol: str) -> pd.DataFrame:
        """Load historical daily OHLCV for the symbol (lookback_days)."""
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=self.lookback_days)

        provider = self._get_price_provider()
        df: pd.DataFrame | None = None
        if provider is not None:
            try:
                df = provider.get_historical_data(symbol, "1d", start=start_dt, end=end_dt)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed fetching Binance data: %s", exc)
                df = None
        if df is None or df.empty:
            # Fallback to bundled CSV (offline mode)
            symbol_csv = "BTCUSDT_1d.csv" if symbol == "BTCUSDT" else "ETHUSDT_1d.csv"
            from src.utils.project_paths import get_project_root
            csv_path = get_project_root() / "src" / "data" / symbol_csv
            if not csv_path.exists():
                logger.error("Offline price CSV not found at %s", csv_path)
                return pd.DataFrame()
            df = pd.read_csv(csv_path, parse_dates=["timestamp"])
            df = df.set_index("timestamp").sort_index()
            df = df.loc[start_dt:]  # trim to lookback window
        # Ensure datetime index is timezone aware
        if df.index.tzinfo is None or df.index.tz is None:
            df.index = df.index.tz_localize(timezone.utc)
        return df

    def _linear_regression_forecast(self, close: pd.Series, horizon: int) -> dict[str, float]:
        """Simple linear regression forecast returning predicted price & r2 score."""
        # Use ordinal day count as x variable
        y = close.values.astype(float)
        x = np.arange(len(y), dtype=float)
        if len(x) < 2:
            return {"predicted_price": float(y[-1]) if len(y) else 0.0, "r2": 0.0}
        # Fit line: y = a*x + b
        a, b = np.polyfit(x, y, 1)
        # Prediction day index
        future_idx = len(y) + horizon
        pred_price = a * future_idx + b
        # R^2 computation
        y_pred = a * x + b
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 0.0 if ss_tot == 0 else (1 - ss_res / ss_tot)
        return {"predicted_price": float(pred_price), "r2": max(0.0, min(1.0, r2))}

    def _adjust_confidence_with_sentiment(self, conf: float, direction: int) -> float:
        """Boost or dampen confidence based on latest Fear & Greed sentiment."""
        try:
            latest_sentiment = self._sentiment_provider.data["sentiment_primary"].iloc[-1]
        except Exception:
            return conf
        # sentiment_primary is 0..1 where >0.5 is greed, <0.5 is fear
        if (direction > 0 and latest_sentiment > 0.6) or (direction < 0 and latest_sentiment < 0.4):
            conf = min(1.0, conf + 0.1)
        else:
            conf = max(0.0, conf - 0.05)
        return conf

    def _generate_prediction_payload(self, symbol: str) -> dict[str, Any]:
        df = self._load_price_history(symbol)
        if df.empty or "close" not in df.columns:
            return {"error": "No price data available"}
        close = df["close"].astype(float)
        current_price = float(close.iloc[-1])

        predictions: list[dict[str, Any]] = []
        for horizon in self._HORIZONS_DAYS:
            res = self._linear_regression_forecast(close, horizon)
            pred_price = res["predicted_price"]
            base_conf = res["r2"]
            direction = 1 if pred_price > current_price else -1 if pred_price < current_price else 0
            conf = self._adjust_confidence_with_sentiment(base_conf, direction)

            pct_change = (pred_price - current_price) / current_price if current_price else 0.0
            recommendation: str
            threshold = 0.05  # 5% move threshold
            if pct_change > threshold and conf >= 0.55:
                recommendation = "Long / Buy"
            elif pct_change < -threshold and conf >= 0.55:
                recommendation = "Short / Sell"
            else:
                recommendation = "Hold"

            predictions.append(
                {
                    "horizon_days": horizon,
                    "predicted_price": round(pred_price, 2),
                    "pct_change": round(pct_change * 100, 2),
                    "confidence": round(conf, 2),
                    "recommendation": recommendation,
                }
            )

        # Latest Fear & Greed info
        sentiment_info: dict[str, Any] = {}
        if not self._sentiment_provider.data.empty:
            latest_row = self._sentiment_provider.data.iloc[-1]
            sentiment_info = {
                "index_value": float(latest_row["value"]) if "value" in latest_row else None,
                "sentiment_primary": float(latest_row.get("sentiment_primary", 0.5)),
                "classification": latest_row.get("classification", ""),
                "timestamp": latest_row.name.isoformat(),
            }

        payload = {
            "symbol": symbol,
            "current_price": round(current_price, 2),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "predictions": predictions,
            "sentiment": sentiment_info,
            "sources": [
                "Historical price (Binance or local)",
                "Linear Regression trend",
                "Fear & Greed sentiment adjustment",
            ],
        }
        return payload

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run(self, host: str = "127.0.0.1", port: int = 8002, debug: bool = False):
        logger.info("MarketPredictionDashboard available at http://%s:%d", host, port)

        # Decide server kwargs based on whether eventlet is enabled.
        # With eventlet enabled, Flask runs a production-safe eventlet server.
        # Without eventlet, allow Werkzeug only for local development.
        server_kwargs = {
            "host": host,
            "port": port,
            "debug": debug,
        }
        if not _WEB_SERVER_USE_EVENTLET:
            server_kwargs["allow_unsafe_werkzeug"] = True

        self.app.run(**server_kwargs)


if __name__ == "__main__":
    MarketPredictionDashboard().run(debug=False)
