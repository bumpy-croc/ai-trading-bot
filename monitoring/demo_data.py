#!/usr/bin/env python3
"""
Demo Data Generator for Monitoring Dashboard

Generates sample trading data for testing the dashboard when no real
trading data is available.
"""

import sqlite3
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path

class DemoDataGenerator:
    """
    Generates realistic demo data for testing the monitoring dashboard
    """
    
    def __init__(self, db_path: str = "demo_trading.db"):
        self.db_path = db_path
        self.connection = None
        self.setup_database()
    
    def setup_database(self):
        """Create database tables for demo data"""
        self.connection = sqlite3.connect(self.db_path)
        cursor = self.connection.cursor()
        
        # Create tables matching the trading bot schema
        cursor.executescript("""
        CREATE TABLE IF NOT EXISTS trading_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            mode TEXT NOT NULL,
            initial_balance REAL NOT NULL,
            final_balance REAL,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            strategy_config TEXT
        );
        
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            quantity REAL NOT NULL,
            entry_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exit_time TIMESTAMP,
            stop_loss REAL,
            take_profit REAL,
            order_id TEXT UNIQUE,
            session_id INTEGER,
            FOREIGN KEY (session_id) REFERENCES trading_sessions (id)
        );
        
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,
            quantity REAL NOT NULL,
            entry_time TIMESTAMP NOT NULL,
            exit_time TIMESTAMP NOT NULL,
            pnl REAL NOT NULL,
            exit_reason TEXT,
            strategy_name TEXT,
            session_id INTEGER,
            FOREIGN KEY (session_id) REFERENCES trading_sessions (id)
        );
        
        CREATE TABLE IF NOT EXISTS account_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            balance REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            session_id INTEGER,
            FOREIGN KEY (session_id) REFERENCES trading_sessions (id)
        );
        
        CREATE TABLE IF NOT EXISTS system_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            severity TEXT DEFAULT 'info',
            component TEXT,
            stack_trace TEXT,
            session_id INTEGER
        );
        """)
        
        self.connection.commit()
    
    def generate_demo_session(self, duration_hours: int = 24) -> int:
        """Generate a complete demo trading session"""
        cursor = self.connection.cursor()
        
        # Create trading session
        strategies = ['adaptive', 'ml_with_sentiment', 'enhanced', 'ml_basic']
        strategy = random.choice(strategies)
        
        cursor.execute("""
        INSERT INTO trading_sessions 
        (strategy_name, symbol, timeframe, mode, initial_balance, start_time)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (strategy, 'BTCUSDT', '1h', 'PAPER', 10000.0, 
              datetime.now() - timedelta(hours=duration_hours)))
        
        session_id = cursor.lastrowid
        
        # Generate trades and account snapshots
        current_balance = 10000.0
        current_time = datetime.now() - timedelta(hours=duration_hours)
        end_time = datetime.now()
        
        trade_count = 0
        winning_trades = 0
        
        # Generate hourly snapshots and random trades
        while current_time < end_time:
            # Maybe generate a trade (20% chance per hour)
            if random.random() < 0.2:
                trade_data = self.generate_trade(current_time, session_id)
                current_balance += trade_data['pnl']
                trade_count += 1
                if trade_data['pnl'] > 0:
                    winning_trades += 1
            
            # Generate account snapshot
            # Add some volatility to balance
            balance_change = random.uniform(-50, 50)
            snapshot_balance = max(current_balance + balance_change, 100)  # Don't go below $100
            
            cursor.execute("""
            INSERT INTO account_snapshots (balance, timestamp, session_id)
            VALUES (?, ?, ?)
            """, (snapshot_balance, current_time, session_id))
            
            current_time += timedelta(hours=1)
        
        # Generate some active positions
        self.generate_active_positions(session_id, 2)
        
        # Generate system events
        self.generate_system_events(session_id, duration_hours)
        
        # Update session with final balance
        cursor.execute("""
        UPDATE trading_sessions 
        SET final_balance = ?, end_time = ?
        WHERE id = ?
        """, (current_balance, datetime.now(), session_id))
        
        self.connection.commit()
        
        print(f"✅ Generated demo session {session_id}")
        print(f"   Strategy: {strategy}")
        print(f"   Duration: {duration_hours} hours")
        if trade_count == 0:
            win_rate = 0.0
        else:
            win_rate = (winning_trades / trade_count) * 100
        print(f"   Trades: {trade_count} (Win rate: {win_rate:.1f}%)")
        print(f"   Final balance: ${current_balance:.2f}")
        
        return session_id
    
    def generate_trade(self, entry_time: datetime, session_id: int) -> Dict:
        """Generate a single realistic trade"""
        cursor = self.connection.cursor()
        
        # Random trade parameters
        side = random.choice(['long', 'short'])
        entry_price = random.uniform(40000, 70000)  # BTC price range
        quantity = random.uniform(0.001, 0.01)  # Small BTC amounts
        
        # Exit after 1-12 hours
        exit_time = entry_time + timedelta(hours=random.uniform(1, 12))
        
        # Generate realistic price movement
        price_change_pct = random.uniform(-0.05, 0.05)  # ±5% price movement
        exit_price = entry_price * (1 + price_change_pct)
        
        # Calculate P&L
        if side == 'long':
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity
        
        # Add some trading fees
        pnl -= abs(pnl) * 0.001  # 0.1% fee
        
        exit_reasons = ['Strategy signal', 'Stop loss', 'Take profit', 'Time limit']
        exit_reason = random.choice(exit_reasons)
        
        # Insert position
        order_id = f"demo_{int(time.time())}_{random.randint(1000, 9999)}"
        cursor.execute("""
        INSERT INTO positions 
        (symbol, side, entry_price, exit_price, quantity, entry_time, exit_time, order_id, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('BTCUSDT', side, entry_price, exit_price, quantity, 
              entry_time, exit_time, order_id, session_id))
        
        # Insert trade
        cursor.execute("""
        INSERT INTO trades 
        (symbol, side, entry_price, exit_price, quantity, entry_time, exit_time, pnl, exit_reason, session_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, ('BTCUSDT', side, entry_price, exit_price, quantity,
              entry_time, exit_time, pnl, exit_reason, session_id))
        
        return {
            'pnl': pnl,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price
        }
    
    def generate_active_positions(self, session_id: int, count: int = 2):
        """Generate active positions (not yet closed)"""
        cursor = self.connection.cursor()
        
        for _ in range(count):
            side = random.choice(['long', 'short'])
            entry_price = random.uniform(40000, 70000)
            quantity = random.uniform(0.001, 0.01)
            entry_time = datetime.now() - timedelta(hours=random.uniform(1, 6))
            
            order_id = f"active_{int(time.time())}_{random.randint(1000, 9999)}"
            
            cursor.execute("""
            INSERT INTO positions 
            (symbol, side, entry_price, quantity, entry_time, order_id, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, ('BTCUSDT', side, entry_price, quantity, entry_time, order_id, session_id))
    
    def generate_system_events(self, session_id: int, duration_hours: int):
        """Generate system events and errors"""
        cursor = self.connection.cursor()
        
        event_types = ['INFO', 'WARNING', 'ERROR']
        components = ['TradingEngine', 'DataProvider', 'Strategy', 'RiskManager']
        
        # Generate some events
        for _ in range(random.randint(5, 15)):
            event_type = random.choice(event_types)
            component = random.choice(components)
            
            if event_type == 'ERROR':
                messages = [
                    'API rate limit exceeded',
                    'Connection timeout to exchange',
                    'Invalid order parameters',
                    'Insufficient balance for trade'
                ]
            elif event_type == 'WARNING':
                messages = [
                    'High volatility detected',
                    'Position approaching stop loss',
                    'API latency increased',
                    'Market sentiment changed'
                ]
            else:
                messages = [
                    'Strategy signal generated',
                    'Position opened successfully',
                    'Risk check passed',
                    'Data update completed'
                ]
            
            message = random.choice(messages)
            timestamp = datetime.now() - timedelta(hours=random.uniform(0, duration_hours))
            
            cursor.execute("""
            INSERT INTO system_events 
            (event_type, message, timestamp, component, session_id)
            VALUES (?, ?, ?, ?, ?)
            """, (event_type, message, timestamp, component, session_id))
        
        self.connection.commit()
    
    def update_realtime_data(self):
        """Update data in real-time for demo purposes"""
        cursor = self.connection.cursor()
        
        # Update account balance with small random change
        cursor.execute("SELECT id, balance FROM account_snapshots ORDER BY timestamp DESC LIMIT 1")
        result = cursor.fetchone()
        
        if result:
            latest_id, current_balance = result
            new_balance = current_balance + random.uniform(-10, 20)  # Small change
            
            cursor.execute("""
            INSERT INTO account_snapshots (balance, timestamp)
            VALUES (?, ?)
            """, (new_balance, datetime.now()))
        
        # Maybe generate a new trade (5% chance)
        if random.random() < 0.05:
            cursor.execute("SELECT id FROM trading_sessions ORDER BY start_time DESC LIMIT 1")
            session_result = cursor.fetchone()
            if session_result:
                self.generate_trade(datetime.now(), session_result[0])
        
        self.connection.commit()
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()

def main():
    """Generate demo data"""
    print("🎭 Generating demo trading data...")
    
    generator = DemoDataGenerator()
    
    try:
        # Generate a 24-hour trading session
        session_id = generator.generate_demo_session(duration_hours=24)
        
        print(f"\n✅ Demo data generated successfully!")
        print(f"📁 Database: {generator.db_path}")
        print(f"🔗 To use with dashboard:")
        print(f"   python monitoring/dashboard.py --db-url sqlite:///{generator.db_path}")
        
    finally:
        generator.close()

if __name__ == '__main__':
    main()