#!/usr/bin/env python3
"""
Script to inspect the trading database and show current data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from datetime import datetime

def inspect_database(db_path="data/trading_bot.db"):
    """Inspect the trading database and display summary information"""
    
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    print("🗄️  Trading Bot Database Inspection")
    print("=" * 60)
    
    # Get list of tables
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    print("\n📊 Database Tables:")
    for table in tables:
        table_name = table[0]
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  - {table_name}: {count} records")
    
    # Show recent trading sessions
    print("\n🎯 Recent Trading Sessions:")
    sessions_df = pd.read_sql_query("""
        SELECT id, session_name, mode, strategy_name, symbol, 
               start_time, end_time, initial_balance, final_balance,
               total_trades, win_rate, max_drawdown
        FROM trading_sessions
        ORDER BY start_time DESC
        LIMIT 5
    """, conn)
    
    if not sessions_df.empty:
        for _, session in sessions_df.iterrows():
            duration = "Active" if pd.isna(session['end_time']) else f"{(pd.to_datetime(session['end_time']) - pd.to_datetime(session['start_time'])).total_seconds() / 3600:.1f}h"
            print(f"\n  Session #{session['id']}: {session['session_name']}")
            print(f"    Strategy: {session['strategy_name']} | Symbol: {session['symbol']} | Mode: {session['mode']}")
            print(f"    Started: {session['start_time']} | Duration: {duration}")
            print(f"    Balance: ${session['initial_balance']:,.2f} → ${session['final_balance'] or session['initial_balance']:,.2f}")
            if session['total_trades']:
                print(f"    Trades: {session['total_trades']} | Win Rate: {session['win_rate']:.1f}% | Max DD: {session['max_drawdown']:.1f}%")
    else:
        print("  No sessions found")
    
    # Show recent trades
    print("\n💹 Recent Trades:")
    trades_df = pd.read_sql_query("""
        SELECT symbol, side, entry_price, exit_price, size, pnl, pnl_percent,
               exit_reason, strategy_name, exit_time
        FROM trades
        ORDER BY exit_time DESC
        LIMIT 10
    """, conn)
    
    if not trades_df.empty:
        for _, trade in trades_df.iterrows():
            pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴"
            print(f"  {pnl_emoji} {trade['symbol']} {trade['side']} | "
                  f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
                  f"P&L: ${trade['pnl']:.2f} ({trade['pnl_percent']:.2f}%) | "
                  f"{trade['exit_reason']}")
    else:
        print("  No trades found")
    
    # Show active positions
    print("\n📈 Active Positions:")
    positions_df = pd.read_sql_query("""
        SELECT symbol, side, entry_price, current_price, size, 
               unrealized_pnl, unrealized_pnl_percent, stop_loss, take_profit,
               entry_time, strategy_name
        FROM positions
        WHERE status != 'filled'
        ORDER BY entry_time DESC
    """, conn)
    
    if not positions_df.empty:
        for _, pos in positions_df.iterrows():
            pnl_emoji = "🟢" if (pos['unrealized_pnl'] or 0) > 0 else "🔴"
            print(f"  {pnl_emoji} {pos['symbol']} {pos['side']} @ ${pos['entry_price']:.2f}")
            print(f"    Current: ${pos['current_price'] or pos['entry_price']:.2f} | "
                  f"Size: {pos['size']*100:.1f}% | "
                  f"Unrealized: ${pos['unrealized_pnl'] or 0:.2f}")
            print(f"    SL: ${pos['stop_loss']:.2f} | TP: ${pos['take_profit'] or 0:.2f}")
    else:
        print("  No active positions")
    
    # Show recent system events
    print("\n🔔 Recent System Events:")
    events_df = pd.read_sql_query("""
        SELECT timestamp, event_type, severity, message
        FROM system_events
        ORDER BY timestamp DESC
        LIMIT 5
    """, conn)
    
    if not events_df.empty:
        for _, event in events_df.iterrows():
            severity_emoji = {"info": "ℹ️", "warning": "⚠️", "error": "❌", "critical": "🚨"}.get(event['severity'], "📝")
            print(f"  {severity_emoji} [{event['timestamp'][:19]}] {event['event_type']}: {event['message'][:60]}...")
    else:
        print("  No events found")
    
    # Show performance summary
    print("\n📊 Overall Performance:")
    perf_df = pd.read_sql_query("""
        SELECT 
            COUNT(*) as total_trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
            AVG(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100 as win_rate,
            SUM(pnl) as total_pnl,
            AVG(pnl) as avg_pnl,
            MAX(pnl) as best_trade,
            MIN(pnl) as worst_trade
        FROM trades
    """, conn)
    
    if not perf_df.empty and perf_df.iloc[0]['total_trades'] > 0:
        perf = perf_df.iloc[0]
        print(f"  Total Trades: {perf['total_trades']}")
        print(f"  Win Rate: {perf['win_rate']:.1f}%")
        print(f"  Total P&L: ${perf['total_pnl']:.2f}")
        print(f"  Average Trade: ${perf['avg_pnl']:.2f}")
        print(f"  Best Trade: ${perf['best_trade']:.2f}")
        print(f"  Worst Trade: ${perf['worst_trade']:.2f}")
    else:
        print("  No performance data available")
    
    print("\n" + "=" * 60)
    print(f"Database location: {os.path.abspath(db_path)}")
    print(f"Database size: {os.path.getsize(db_path) / 1024:.1f} KB")
    
    conn.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Inspect trading database")
    parser.add_argument("--db", default="data/trading_bot.db", help="Database path")
    args = parser.parse_args()
    
    inspect_database(args.db) 