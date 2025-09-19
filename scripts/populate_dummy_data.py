#!/usr/bin/env python3
"""
Script to populate the database with dummy data for all tables.
Creates approximately 100 trades with related data across all database tables.
"""

import logging
import random
import sys
from datetime import datetime, timedelta
from decimal import Decimal

# Add src to path for imports
sys.path.insert(0, "src")

from src.database.manager import DatabaseManager
from src.database.models import (
    EventType,
    PositionSide,
    PredictionPerformance,
    Trade,
    TradeSource,
)
from src.utils.logging_config import configure_logging

# Configure logging
configure_logging()
logger = logging.getLogger("atb.populate_dummy")


class DummyDataPopulator:
    """Populates database with realistic dummy data for testing and development."""

    def __init__(self, database_url: str = None):
        """Initialize the populator with database connection."""
        self.db_manager = DatabaseManager(database_url)
        self.symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"]
        self.strategies = ["MlBasic", "BullStrategy", "BearStrategy", "TestStrategy"]
        self.timeframes = ["1h", "4h", "1d"]

        # Price ranges for different symbols
        self.price_ranges = {
            "BTCUSDT": (25000, 65000),
            "ETHUSDT": (1500, 4000),
            "ADAUSDT": (0.3, 1.2),
            "DOTUSDT": (5, 25),
            "LINKUSDT": (5, 20),
        }

        # Base balance for simulations
        self.base_balance = 10000.0

    def generate_random_price(self, symbol: str, base_price: float = None) -> float:
        """Generate a realistic price for a symbol."""
        min_price, max_price = self.price_ranges[symbol]
        if base_price is None:
            base_price = random.uniform(min_price, max_price)

        # Add some volatility (±5%)
        volatility = random.uniform(-0.05, 0.05)
        new_price = base_price * (1 + volatility)

        # Ensure price stays within reasonable bounds
        return max(min_price * 0.8, min(max_price * 1.2, new_price))

    def generate_trade_data(self, trade_id: int) -> dict:
        """Generate realistic trade data."""
        symbol = random.choice(self.symbols)
        side = random.choice(["LONG", "SHORT"])
        strategy = random.choice(self.strategies)

        # Generate entry and exit times (trades last 1-48 hours)
        entry_time = datetime.utcnow() - timedelta(
            days=random.randint(1, 30), hours=random.randint(0, 23), minutes=random.randint(0, 59)
        )
        duration_hours = random.randint(1, 48)
        exit_time = entry_time + timedelta(hours=duration_hours)

        # Generate prices
        entry_price = self.generate_random_price(symbol)
        exit_price = self.generate_random_price(symbol, entry_price)

        # Ensure realistic P&L based on side
        if side == "LONG":
            # 60% chance of profit for long positions
            if random.random() < 0.6:
                exit_price = entry_price * random.uniform(1.01, 1.15)
            else:
                exit_price = entry_price * random.uniform(0.85, 0.99)
        else:
            # 60% chance of profit for short positions
            if random.random() < 0.6:
                exit_price = entry_price * random.uniform(0.85, 0.99)
            else:
                exit_price = entry_price * random.uniform(1.01, 1.15)

        # Calculate P&L
        size = random.uniform(0.01, 0.1)  # 1-10% of balance
        quantity = (self.base_balance * size) / entry_price

        if side == "LONG":
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        # Add some commission
        commission = abs(pnl) * random.uniform(0.0005, 0.001)  # 0.05-0.1%

        return {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "quantity": quantity,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl - commission,
            "commission": commission,
            "exit_reason": random.choice(["take_profit", "stop_loss", "manual_close", "time_exit"]),
            "strategy_name": strategy,
            "confidence_score": random.uniform(0.3, 0.95),
            "order_id": f"order_{trade_id}_{random.randint(1000, 9999)}",
            "stop_loss": entry_price * (0.95 if side == "LONG" else 1.05),
            "take_profit": entry_price * (1.05 if side == "LONG" else 0.95),
            "strategy_config": {
                "base_position_size": random.uniform(0.01, 0.05),
                "use_abs_confidence": random.choice([True, False]),
                "stop_loss_pct": random.uniform(0.02, 0.05),
                "take_profit_pct": random.uniform(0.03, 0.08),
            },
        }

    def create_trading_sessions(self) -> list:
        """Create multiple trading sessions."""
        session_ids = []

        for i in range(3):  # Create 3 sessions
            strategy = random.choice(self.strategies)
            symbol = random.choice(self.symbols)
            timeframe = random.choice(self.timeframes)
            mode = random.choice([TradeSource.LIVE, TradeSource.BACKTEST, TradeSource.PAPER])

            session_id = self.db_manager.create_trading_session(
                strategy_name=strategy,
                symbol=symbol,
                timeframe=timeframe,
                mode=mode,
                initial_balance=self.base_balance,
                strategy_config={
                    "base_position_size": random.uniform(0.01, 0.05),
                    "use_abs_confidence": random.choice([True, False]),
                },
                session_name=f"dummy_session_{i + 1}_{datetime.utcnow().strftime('%Y%m%d_%H%M')}",
            )
            session_ids.append(session_id)
            logger.info(f"Created trading session {session_id}")

        return session_ids

    def populate_trades(self, session_ids: list, num_trades: int = 100):
        """Populate trades table with dummy data."""
        logger.info(f"Creating {num_trades} trades...")

        for i in range(num_trades):
            session_id = random.choice(session_ids)
            trade_data = self.generate_trade_data(i + 1)

            self.db_manager.log_trade(session_id=session_id, **trade_data)

            if (i + 1) % 20 == 0:
                logger.info(f"Created {i + 1} trades")

        logger.info(f"Successfully created {num_trades} trades")

    def validate_trade_exports(self, session_ids: list[int]):
        """Ensure live/backtest trade exports report cash PnL."""

        logger.info("Validating trade export PnL units across live/backtest sessions...")
        tolerance = Decimal("0.01")

        with self.db_manager.get_session() as session:
            for session_id in session_ids:
                trades = (
                    session.query(Trade)
                    .filter(Trade.session_id == session_id)
                    .all()
                )

                for trade in trades:
                    if trade.source not in {TradeSource.LIVE, TradeSource.BACKTEST}:
                        continue

                    quantity = Decimal(trade.quantity or 0)
                    if quantity == 0:
                        continue

                    entry_price = Decimal(trade.entry_price)
                    exit_price = Decimal(trade.exit_price)
                    commission = Decimal(trade.commission or 0)
                    side = (
                        trade.side.value
                        if isinstance(trade.side, PositionSide)
                        else str(trade.side)
                    ).lower()

                    if side == "long":
                        expected_cash = (exit_price - entry_price) * quantity - commission
                    else:
                        expected_cash = (entry_price - exit_price) * quantity - commission

                    actual_cash = Decimal(trade.pnl)

                    if (expected_cash - actual_cash).copy_abs() > tolerance:
                        raise ValueError(
                            "PnL unit mismatch detected: expected cash value "
                            f"{expected_cash} but saw {actual_cash} for session {session_id}"
                        )

        logger.info("Trade export validation passed: cash PnL stored for live/backtest trades")

    def populate_positions(self, session_ids: list, num_positions: int = 15):
        """Populate positions table with dummy data."""
        logger.info(f"Creating {num_positions} positions...")

        for i in range(num_positions):
            session_id = random.choice(session_ids)
            symbol = random.choice(self.symbols)
            side = random.choice(["LONG", "SHORT"])
            strategy = random.choice(self.strategies)

            entry_price = self.generate_random_price(symbol)
            size = random.uniform(0.01, 0.1)
            quantity = (self.base_balance * size) / entry_price

            position_id = self.db_manager.log_position(
                session_id=session_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                size=size,
                quantity=quantity,
                strategy_name=strategy,
                order_id=f"pos_order_{i + 1}_{random.randint(1000, 9999)}",
                stop_loss=entry_price * (0.95 if side == "LONG" else 1.05),
                take_profit=entry_price * (1.05 if side == "LONG" else 0.95),
                confidence_score=random.uniform(0.3, 0.95),
            )

            # Update some positions with current market data
            if random.random() < 0.7:  # 70% of positions get updated
                current_price = self.generate_random_price(symbol, entry_price)
                # Calculate unrealized P&L
                if side == "LONG":
                    unrealized_pnl_percent = ((current_price - entry_price) / entry_price) * 100
                else:
                    unrealized_pnl_percent = ((entry_price - current_price) / entry_price) * 100

                self.db_manager.update_position(
                    position_id=position_id,
                    current_price=current_price,
                    unrealized_pnl_percent=unrealized_pnl_percent,
                )

        logger.info(f"Successfully created {num_positions} positions")

    def populate_account_history(self, session_ids: list, num_snapshots: int = 50):
        """Populate account history table with dummy data."""
        logger.info(f"Creating {num_snapshots} account history snapshots...")

        for session_id in session_ids:
            current_balance = self.base_balance

            for _i in range(num_snapshots // len(session_ids)):
                # Simulate balance changes over time
                _ = datetime.utcnow() - timedelta(
                    days=random.randint(0, 30), hours=random.randint(0, 23)
                )

                # Random balance fluctuation
                balance_change = random.uniform(-500, 800)
                current_balance += balance_change
                current_balance = max(1000, current_balance)  # Minimum balance

                equity = current_balance + random.uniform(-200, 300)
                total_pnl = current_balance - self.base_balance
                daily_pnl = random.uniform(-200, 300)
                drawdown = random.uniform(0, 0.15)  # 0-15% drawdown

                self.db_manager.log_account_snapshot(
                    session_id=session_id,
                    balance=current_balance,
                    equity=equity,
                    total_pnl=total_pnl,
                    daily_pnl=daily_pnl,
                    drawdown=drawdown,
                    open_positions=random.randint(0, 5),
                    total_exposure=random.uniform(0, current_balance * 0.3),
                    margin_used=random.uniform(0, current_balance * 0.1),
                )

        logger.info(f"Successfully created {num_snapshots} account history snapshots")

    def populate_system_events(self, session_ids: list, num_events: int = 30):
        """Populate system events table with dummy data."""
        logger.info(f"Creating {num_events} system events...")

        event_types = list(EventType)
        components = [
            "trading_engine",
            "strategy_manager",
            "risk_manager",
            "data_provider",
            "ml_model",
        ]
        severities = ["info", "warning", "error", "critical"]

        for i in range(num_events):
            session_id = random.choice(session_ids)
            event_type = random.choice(event_types)
            component = random.choice(components)
            severity = random.choice(severities)

            # Generate appropriate messages based on event type
            if event_type == EventType.ENGINE_START:
                message = f"Trading engine started for session {session_id}"
            elif event_type == EventType.ENGINE_STOP:
                message = f"Trading engine stopped for session {session_id}"
            elif event_type == EventType.STRATEGY_CHANGE:
                message = f"Strategy changed to {random.choice(self.strategies)}"
            elif event_type == EventType.MODEL_UPDATE:
                message = "ML model updated with new training data"
            elif event_type == EventType.ERROR:
                message = f"Error occurred in {component}: {random.choice(['Connection timeout', 'Invalid data', 'API rate limit'])}"
            elif event_type == EventType.WARNING:
                message = f"Warning in {component}: {random.choice(['High volatility detected', 'Low confidence signal', 'Unusual market conditions'])}"
            elif event_type == EventType.ALERT:
                message = f"Alert: {random.choice(['Large position opened', 'Drawdown threshold reached', 'Unusual trading volume'])}"
            else:
                message = f"System event: {event_type.value}"

            self.db_manager.log_event(
                session_id=session_id,
                event_type=event_type,
                message=message,
                severity=severity,
                component=component,
                details={
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                    "additional_info": f"Event {i + 1} details",
                },
            )

        logger.info(f"Successfully created {num_events} system events")

    def populate_strategy_executions(self, session_ids: list, num_executions: int = 80):
        """Populate strategy executions table with dummy data."""
        logger.info(f"Creating {num_executions} strategy executions...")

        signal_types = ["entry", "exit", "hold"]
        action_taken_options = ["opened_long", "opened_short", "closed_position", "no_action"]

        for _i in range(num_executions):
            session_id = random.choice(session_ids)
            symbol = random.choice(self.symbols)
            strategy = random.choice(self.strategies)
            signal_type = random.choice(signal_types)
            action_taken = random.choice(action_taken_options)

            # Generate realistic indicators
            indicators = {
                "rsi": random.uniform(20, 80),
                "macd": random.uniform(-2, 2),
                "bollinger_upper": self.generate_random_price(symbol) * 1.02,
                "bollinger_lower": self.generate_random_price(symbol) * 0.98,
                "volume_sma": random.uniform(1000, 10000),
                "price_sma_20": self.generate_random_price(symbol),
                "price_sma_50": self.generate_random_price(symbol),
            }

            # Generate sentiment data
            sentiment_data = {
                "fear_greed_index": random.uniform(0, 100),
                "social_sentiment": random.uniform(-1, 1),
                "news_sentiment": random.uniform(-1, 1),
                "overall_sentiment": random.uniform(-1, 1),
            }

            # Generate ML predictions
            ml_predictions = {
                "price_prediction": self.generate_random_price(symbol),
                "confidence": random.uniform(0.3, 0.95),
                "prediction_horizon": random.randint(1, 24),
                "model_version": f"v{random.randint(1, 3)}.{random.randint(0, 9)}",
            }

            self.db_manager.log_strategy_execution(
                session_id=session_id,
                strategy_name=strategy,
                symbol=symbol,
                signal_type=signal_type,
                action_taken=action_taken,
                price=self.generate_random_price(symbol),
                timeframe=random.choice(self.timeframes),
                signal_strength=random.uniform(0.1, 1.0),
                confidence_score=random.uniform(0.3, 0.95),
                indicators=indicators,
                sentiment_data=sentiment_data,
                ml_predictions=ml_predictions,
                position_size=random.uniform(0.01, 0.1),
                reasons=[
                    random.choice(
                        ["Strong signal", "Risk management", "Market conditions", "ML prediction"]
                    )
                ],
                volume=random.uniform(1000, 50000),
                volatility=random.uniform(0.01, 0.05),
            )

        logger.info(f"Successfully created {num_executions} strategy executions")

    def populate_account_balances(self, session_ids: list):
        """Populate account balances table with dummy data."""
        logger.info("Creating account balance records...")

        for session_id in session_ids:
            # Create initial balance
            self.db_manager.update_balance(
                new_balance=self.base_balance,
                update_reason="Initial balance setup",
                updated_by="system",
                session_id=session_id,
            )

            # Create some balance updates over time
            current_balance = self.base_balance
            for _i in range(5):
                balance_change = random.uniform(-1000, 1500)
                current_balance += balance_change
                current_balance = max(1000, current_balance)

                self.db_manager.update_balance(
                    new_balance=current_balance,
                    update_reason=random.choice(
                        [
                            "Trade P&L",
                            "Manual adjustment",
                            "Deposit",
                            "Withdrawal",
                            "Fee adjustment",
                        ]
                    ),
                    updated_by=random.choice(["system", "user", "admin"]),
                    session_id=session_id,
                )

        logger.info("Successfully created account balance records")

    def populate_optimization_cycles(self, session_ids: list, num_cycles: int = 10):
        """Populate optimization cycles table with dummy data."""
        logger.info(f"Creating {num_cycles} optimization cycles...")

        for _i in range(num_cycles):
            session_id = random.choice(session_ids)
            strategy = random.choice(self.strategies)
            symbol = random.choice(self.symbols)
            timeframe = random.choice(self.timeframes)

            # Generate baseline metrics
            baseline_metrics = {
                "total_return": random.uniform(-0.1, 0.3),
                "sharpe_ratio": random.uniform(0.5, 2.0),
                "max_drawdown": random.uniform(0.05, 0.25),
                "win_rate": random.uniform(0.4, 0.7),
                "profit_factor": random.uniform(0.8, 2.5),
            }

            # Generate candidate parameters
            candidate_params = {
                "base_position_size": random.uniform(0.01, 0.05),
                "stop_loss_pct": random.uniform(0.02, 0.05),
                "take_profit_pct": random.uniform(0.03, 0.08),
                "confidence_threshold": random.uniform(0.3, 0.8),
            }

            # Generate candidate metrics (slightly different from baseline)
            candidate_metrics = {
                "total_return": baseline_metrics["total_return"] + random.uniform(-0.05, 0.05),
                "sharpe_ratio": baseline_metrics["sharpe_ratio"] + random.uniform(-0.3, 0.3),
                "max_drawdown": baseline_metrics["max_drawdown"] + random.uniform(-0.05, 0.05),
                "win_rate": baseline_metrics["win_rate"] + random.uniform(-0.1, 0.1),
                "profit_factor": baseline_metrics["profit_factor"] + random.uniform(-0.3, 0.3),
            }

            # Generate validator report
            validator_report = {
                "p_value": random.uniform(0.01, 0.1),
                "effect_size": random.uniform(0.1, 0.5),
                "statistical_significance": random.choice([True, False]),
                "pass_threshold": random.choice([True, False]),
            }

            decision = random.choice(["propose", "reject", "apply"])

            self.db_manager.record_optimization_cycle(
                strategy_name=strategy,
                symbol=symbol,
                timeframe=timeframe,
                baseline_metrics=baseline_metrics,
                candidate_params=candidate_params,
                candidate_metrics=candidate_metrics,
                validator_report=validator_report,
                decision=decision,
                session_id=session_id,
            )

        logger.info(f"Successfully created {num_cycles} optimization cycles")

    def populate_prediction_performance(self, session_ids: list, num_records: int = 20):
        """Populate prediction performance table with dummy data."""
        logger.info(f"Creating {num_records} prediction performance records...")

        model_names = ["price_model_v1", "sentiment_model_v2", "ensemble_model_v1"]
        horizons = [1, 4, 24]  # 1h, 4h, 24h

        for _i in range(num_records):
            _ = random.choice(session_ids)
            strategy = random.choice(self.strategies)
            symbol = random.choice(self.symbols)
            timeframe = random.choice(self.timeframes)
            model_name = random.choice(model_names)
            horizon = random.choice(horizons)

            # Generate performance metrics
            mae = random.uniform(0.01, 0.05)
            rmse = random.uniform(0.02, 0.08)
            mape = random.uniform(0.5, 3.0)
            ic = random.uniform(0.1, 0.6)  # Information coefficient

            # Distribution metrics
            mean_pred = random.uniform(0.4, 0.6)
            std_pred = random.uniform(0.1, 0.3)
            mean_real = random.uniform(0.4, 0.6)
            std_real = random.uniform(0.1, 0.3)

            with self.db_manager.get_session() as session:
                performance = PredictionPerformance(
                    timestamp=datetime.utcnow() - timedelta(days=random.randint(0, 30)),
                    model_name=model_name,
                    horizon=horizon,
                    mae=mae,
                    rmse=rmse,
                    mape=mape,
                    ic=ic,
                    mean_pred=mean_pred,
                    std_pred=std_pred,
                    mean_real=mean_real,
                    std_real=std_real,
                    strategy_name=strategy,
                    symbol=symbol,
                    timeframe=timeframe,
                )
                session.add(performance)
                session.commit()

        logger.info(f"Successfully created {num_records} prediction performance records")

    def end_sessions(self, session_ids: list):
        """End all trading sessions."""
        logger.info("Ending trading sessions...")

        for session_id in session_ids:
            # Calculate final balance based on trades
            with self.db_manager.get_session() as session:
                trades = session.query(Trade).filter(Trade.session_id == session_id).all()
                total_pnl = sum(float(trade.pnl) for trade in trades)
                final_balance = self.base_balance + total_pnl

            self.db_manager.end_trading_session(session_id=session_id, final_balance=final_balance)

        logger.info("Successfully ended all trading sessions")

    def populate_all_data(self, num_trades: int = 100):
        """Populate all tables with dummy data."""
        logger.info("Starting database population with dummy data...")

        try:
            # Create trading sessions
            session_ids = self.create_trading_sessions()

            # Populate all tables
            self.populate_trades(session_ids, num_trades)
            self.validate_trade_exports(session_ids)
            self.populate_positions(session_ids, 15)
            self.populate_account_history(session_ids, 50)
            self.populate_system_events(session_ids, 30)
            self.populate_strategy_executions(session_ids, 80)
            self.populate_account_balances(session_ids)
            self.populate_optimization_cycles(session_ids, 10)
            self.populate_prediction_performance(session_ids, 20)

            # End sessions
            self.end_sessions(session_ids)

            logger.info("✅ Successfully populated database with dummy data!")
            logger.info(f"Created {num_trades} trades across {len(session_ids)} trading sessions")

        except Exception as e:
            logger.error(f"Error populating database: {e}")
            raise


def main():
    """Main function to run the dummy data population."""
    import argparse

    parser = argparse.ArgumentParser(description="Populate database with dummy data")
    parser.add_argument(
        "--trades", type=int, default=100, help="Number of trades to create (default: 100)"
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Database URL (optional, uses environment variable if not provided)",
    )
    parser.add_argument("--confirm", action="store_true", help="Skip confirmation prompt")

    args = parser.parse_args()

    if not args.confirm:
        response = input(
            f"This will populate the database with {args.trades} trades and related data. Continue? (y/N): "
        )
        if response.lower() != "y":
            print("Operation cancelled.")
            return

    try:
        populator = DummyDataPopulator(args.database_url)
        populator.populate_all_data(args.trades)
        print("✅ Database population completed successfully!")

    except Exception as e:
        logger.error(f"Failed to populate database: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
