#!/usr/bin/env python3
"""
Live Trading Example

This example shows how to set up and run the live trading system safely.
It demonstrates both paper trading and live trading configuration.

IMPORTANT: This is for educational purposes. Always start with paper trading!
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.constants import DEFAULT_INITIAL_BALANCE, DEFAULT_PERFORMANCE_MONITOR_INTERVAL
from src.data_providers.binance_provider import BinanceProvider
from src.live.trading_engine import LiveTradingEngine
from src.risk.risk_manager import RiskParameters
from src.strategies.ml_basic import MlBasic


def setup_paper_trading():
    """Example: Safe paper trading setup"""
    print("🚀 Setting up PAPER TRADING (Safe Mode)")

    # Load strategy
    strategy = MlBasic()
    print(f"Strategy loaded: {strategy.name}")

    # Setup data provider
    data_provider = BinanceProvider()
    print("Data provider initialized")

    # Setup sentiment provider (optional)
    print("Sentiment analysis not available - sentiment providers have been removed")
    sentiment_provider = None

    # Risk parameters
    risk_params = RiskParameters(
        base_risk_per_trade=0.01,  # 1% risk per trade
        max_risk_per_trade=0.02,  # 2% maximum risk
        max_drawdown=0.2,  # 20% max drawdown
    )

    # Create trading engine (PAPER TRADING MODE)
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        sentiment_provider=sentiment_provider,
        risk_parameters=risk_params,
        check_interval=60,  # Check every minute
        initial_balance=DEFAULT_INITIAL_BALANCE,  # Virtual balance
        max_position_size=0.1,  # Max 10% per position
        enable_live_trading=False,  # PAPER TRADING ONLY
        log_trades=True,
    )

    print("\n" + "=" * 60)
    print("📄 PAPER TRADING MODE CONFIGURED")
    print("=" * 60)
    print("- Strategy: MlBasic")
    print("- Symbol: BTC-USD")
    print(f"- Balance: ${DEFAULT_INITIAL_BALANCE:,.0f} (virtual)")
    print("- Risk per trade: 1%")
    print("- Max position: 10%")
    print("- Check interval: 60 seconds")
    print("- Sentiment: Enabled" if sentiment_provider else "- Sentiment: Disabled")
    print("=" * 60)

    return engine


def setup_live_trading():
    """Example: Live trading setup (DANGEROUS - REAL MONEY)"""
    print("🚨 Setting up LIVE TRADING (REAL MONEY AT RISK)")

    # Check API credentials
    if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
        print("❌ ERROR: Binance API credentials not found!")
        print("Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return None

    # Confirmation check
    print("\n⚠️  WARNING: This will trade with REAL MONEY")
    print("⚠️  Only proceed if you understand the risks")
    confirmation = input("Type 'I UNDERSTAND THE RISKS' to continue: ")

    if confirmation != "I UNDERSTAND THE RISKS":
        print("Live trading setup cancelled")
        return None

    # Load strategy
    strategy = MlBasic()

    # Setup data provider
    data_provider = BinanceProvider()

    # Risk parameters (more conservative for live trading)
    risk_params = RiskParameters(
        base_risk_per_trade=0.005,  # 0.5% risk per trade
        max_risk_per_trade=0.01,  # 1% maximum risk
        max_drawdown=0.1,  # 10% max drawdown
    )

    # Create trading engine (LIVE TRADING MODE)
    engine = LiveTradingEngine(
        strategy=strategy,
        data_provider=data_provider,
        risk_parameters=risk_params,
        check_interval=300,  # Check every 5 minutes (less frequent)
        initial_balance=DEFAULT_INITIAL_BALANCE,  # Start with defined balance
        max_position_size=0.05,  # Max 5% per position (conservative)
        enable_live_trading=True,  # REAL TRADING ENABLED
        log_trades=True,
        alert_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),  # Optional alerts
    )

    print("\n" + "=" * 60)
    print("🔴 LIVE TRADING MODE CONFIGURED")
    print("=" * 60)
    print("- Strategy: MlBasic")
    print("- Symbol: BTC-USD")
    print(f"- Balance: ${DEFAULT_INITIAL_BALANCE:,.0f} (REAL MONEY)")
    print("- Risk per trade: 0.5%")
    print("- Max position: 5%")
    print("- Check interval: 300 seconds")
    print("- REAL ORDERS WILL BE EXECUTED")
    print("=" * 60)

    return engine


def monitor_performance(engine):
    """Monitor trading performance"""
    import time

    print("\n📊 Monitoring trading performance...")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while engine.is_running:
            # Get performance summary
            perf = engine.get_performance_summary()

            # Display status
            print(
                f"\r📊 Balance: ${perf['current_balance']:,.2f} | "
                f"Return: {perf['total_return_pct']:+.2f}% | "
                f"Trades: {perf['total_trades']} | "
                f"Win Rate: {perf['win_rate_pct']:.1f}% | "
                f"Positions: {perf['active_positions']}",
                end="",
                flush=True,
            )

            time.sleep(DEFAULT_PERFORMANCE_MONITOR_INTERVAL)  # Configurable monitoring interval

    except KeyboardInterrupt:
        print("\n\n🛑 Monitoring stopped")


def main():
    """Main example function"""
    print("🤖 Live Trading System Example")
    print("=" * 50)

    print("\nChoose trading mode:")
    print("1. Paper Trading (Safe - Recommended)")
    print("2. Live Trading (REAL MONEY - Advanced Users Only)")
    print("3. Exit")

    choice = input("\nEnter your choice (1-3): ").strip()

    if choice == "1":
        # Paper trading
        engine = setup_paper_trading()
        if engine:
            try:
                print("\n🚀 Starting paper trading...")
                print("Press Ctrl+C to stop")

                # Start trading in a separate thread for monitoring
                import threading

                trading_thread = threading.Thread(target=lambda: engine.start("BTC-USD", "1h"))
                trading_thread.daemon = True
                trading_thread.start()

                # Monitor performance
                monitor_performance(engine)

            except KeyboardInterrupt:
                print("\n🛑 Stopping paper trading...")
                engine.stop()

    elif choice == "2":
        # Live trading
        engine = setup_live_trading()
        if engine:
            try:
                print("\n🔴 Starting LIVE trading...")
                print("⚠️  REAL MONEY AT RISK")
                print("Press Ctrl+C to stop")

                # Final countdown
                import time

                for i in range(10, 0, -1):
                    print(f"Starting in {i}...", end="\r")
                    time.sleep(1)

                # Start trading
                engine.start("BTC-USD", "1h")

            except KeyboardInterrupt:
                print("\n🛑 Stopping live trading...")
                engine.stop()

    elif choice == "3":
        print("👋 Goodbye!")

    else:
        print("❌ Invalid choice")


if __name__ == "__main__":
    # Check prerequisites
    try:
        pass
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Run: pip install -r requirements.txt")
        sys.exit(1)

    # Run example
    main()
