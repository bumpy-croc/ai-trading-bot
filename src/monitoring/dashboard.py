#!/usr/bin/env python3
"""
Trading Bot Monitoring Dashboard

A real-time web dashboard for monitoring the trading bot performance,
positions, risk metrics, and system health.
"""

import os
import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import pandas as pd

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from database.manager import DatabaseManager
from data_providers.binance_data_provider import BinanceDataProvider
from data_providers.cached_data_provider import CachedDataProvider

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoringDashboard:
    """
    Real-time monitoring dashboard for the trading bot
    """
    
    def __init__(self, db_url: Optional[str] = None, update_interval: int = 3600):
        self.app = Flask(__name__, template_folder='templates', static_folder='static')
        self.app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-in-production')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Initialize database manager
        self.db_manager = DatabaseManager(db_url)
        
        # Initialize data provider for live price data
        binance_provider = BinanceDataProvider()
        self.data_provider = CachedDataProvider(binance_provider, cache_ttl_hours=0.1)  # 6 min cache
        
        self.update_interval = update_interval
        self.is_running = False
        self.update_thread = None
        
        # Configurable monitoring parameters
        self.monitoring_config = {
            # System Health Metrics
            'api_connection_status': {'enabled': True, 'priority': 'high', 'format': 'status'},
            'data_feed_status': {'enabled': True, 'priority': 'high', 'format': 'status'},
            'error_rate_hourly': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'api_latency': {'enabled': True, 'priority': 'medium', 'format': 'number'},
            'last_data_update': {'enabled': True, 'priority': 'high', 'format': 'datetime'},
            'system_uptime': {'enabled': True, 'priority': 'medium', 'format': 'text'},
            
            # Risk Metrics
            'current_drawdown': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'daily_pnl': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            'weekly_pnl': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            'position_sizes': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            'max_drawdown': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'risk_per_trade': {'enabled': True, 'priority': 'medium', 'format': 'percentage'},
            'volatility': {'enabled': True, 'priority': 'medium', 'format': 'percentage'},
            
            # Order Execution Metrics
            'fill_rate': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'avg_slippage': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'failed_orders': {'enabled': True, 'priority': 'high', 'format': 'number'},
            'order_latency': {'enabled': True, 'priority': 'medium', 'format': 'number'},
            'execution_quality': {'enabled': True, 'priority': 'medium', 'format': 'status'},
            
            # Balance & Positions
            'current_balance': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            'active_positions_count': {'enabled': True, 'priority': 'high', 'format': 'number'},
            'total_position_value': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            'margin_usage': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'available_margin': {'enabled': True, 'priority': 'medium', 'format': 'currency'},
            'unrealized_pnl': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            
            # Strategy Performance
            'win_rate': {'enabled': True, 'priority': 'high', 'format': 'percentage'},
            'sharpe_ratio': {'enabled': True, 'priority': 'high', 'format': 'number'},
            'recent_trade_outcomes': {'enabled': True, 'priority': 'medium', 'format': 'text'},
            'profit_factor': {'enabled': True, 'priority': 'medium', 'format': 'number'},
            'avg_win_loss_ratio': {'enabled': True, 'priority': 'medium', 'format': 'number'},
            'total_trades': {'enabled': True, 'priority': 'medium', 'format': 'number'},
            
            # Additional Core Metrics
            'total_pnl': {'enabled': True, 'priority': 'high', 'format': 'currency'},
            'current_strategy': {'enabled': True, 'priority': 'high', 'format': 'text'},
            'current_price': {'enabled': True, 'priority': 'medium', 'format': 'currency'},
            'price_change_24h': {'enabled': True, 'priority': 'medium', 'format': 'percentage'},
            'rsi': {'enabled': True, 'priority': 'low', 'format': 'number'},
            'ema_trend': {'enabled': True, 'priority': 'low', 'format': 'text'},
        }
        
        self._setup_routes()
        self._setup_websocket_handlers()
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard page"""
            return render_template('dashboard.html')
        
        @self.app.route('/api/metrics')
        def get_metrics():
            """Get current metrics as JSON"""
            metrics = self._collect_metrics()
            return jsonify(metrics)
        
        @self.app.route('/api/config')
        def get_config():
            """Get monitoring configuration"""
            return jsonify(self.monitoring_config)
        
        @self.app.route('/api/config', methods=['POST'])
        def update_config():
            """Update monitoring configuration"""
            new_config = request.json
            if new_config:
                # Validate and update configuration
                for key, value in new_config.items():
                    if key in self.monitoring_config:
                        self.monitoring_config[key].update(value)
                return jsonify({'success': True})
            return jsonify({'success': False, 'error': 'Invalid configuration'})
        
        @self.app.route('/api/positions')
        def get_positions():
            """Get current positions"""
            positions = self._get_current_positions()
            return jsonify(positions)
        
        @self.app.route('/api/trades')
        def get_recent_trades():
            """Get recent trades"""
            limit = request.args.get('limit', 50, type=int)
            trades = self._get_recent_trades(limit)
            return jsonify(trades)
        
        @self.app.route('/api/performance')
        def get_performance_chart():
            """Get performance chart data"""
            days = request.args.get('days', 7, type=int)
            chart_data = self._get_performance_chart_data(days)
            return jsonify(chart_data)
        
        @self.app.route('/api/system/status')
        def system_status():
            """Get system health status"""
            status = self._get_system_status()
            return jsonify(status)
    
    def _setup_websocket_handlers(self):
        """Setup WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            logger.info('Client connected to monitoring dashboard')
            emit('connected', {'status': 'Connected to monitoring dashboard'})
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            logger.info('Client disconnected from monitoring dashboard')
        
        @self.socketio.on('request_update')
        def handle_update_request():
            """Handle manual update request"""
            metrics = self._collect_metrics()
            emit('metrics_update', metrics)
    
    def _collect_metrics(self) -> Dict[str, Any]:
        """Collect all monitoring metrics"""
        try:
            metrics = {}
            
            # Get enabled metrics only
            enabled_metrics = {k: v for k, v in self.monitoring_config.items() if v.get('enabled', True)}
            
            # System Health Metrics
            if 'api_connection_status' in enabled_metrics:
                metrics['api_connection_status'] = self._get_api_connection_status()
            if 'data_feed_status' in enabled_metrics:
                metrics['data_feed_status'] = self._get_data_feed_status()
            if 'error_rate_hourly' in enabled_metrics:
                metrics['error_rate_hourly'] = self._get_error_rate_hourly()
            if 'api_latency' in enabled_metrics:
                metrics['api_latency'] = self._get_api_latency()
            if 'last_data_update' in enabled_metrics:
                metrics['last_data_update'] = datetime.now().isoformat()
            if 'system_uptime' in enabled_metrics:
                metrics['system_uptime'] = self._get_system_uptime()
            
            # Risk Metrics
            if 'current_drawdown' in enabled_metrics:
                metrics['current_drawdown'] = self._get_current_drawdown()
            if 'daily_pnl' in enabled_metrics:
                metrics['daily_pnl'] = self._get_daily_pnl()
            if 'weekly_pnl' in enabled_metrics:
                metrics['weekly_pnl'] = self._get_weekly_pnl()
            if 'position_sizes' in enabled_metrics:
                metrics['position_sizes'] = self._get_total_position_sizes()
            if 'max_drawdown' in enabled_metrics:
                metrics['max_drawdown'] = self._get_max_drawdown()
            if 'risk_per_trade' in enabled_metrics:
                metrics['risk_per_trade'] = self._get_risk_per_trade()
            if 'volatility' in enabled_metrics:
                metrics['volatility'] = self._get_volatility()
            
            # Order Execution Metrics
            if 'fill_rate' in enabled_metrics:
                metrics['fill_rate'] = self._get_fill_rate()
            if 'avg_slippage' in enabled_metrics:
                metrics['avg_slippage'] = self._get_avg_slippage()
            if 'failed_orders' in enabled_metrics:
                metrics['failed_orders'] = self._get_failed_orders()
            if 'order_latency' in enabled_metrics:
                metrics['order_latency'] = self._get_order_latency()
            if 'execution_quality' in enabled_metrics:
                metrics['execution_quality'] = self._get_execution_quality()
            
            # Balance & Positions
            if 'current_balance' in enabled_metrics:
                metrics['current_balance'] = self._get_current_balance()
            if 'active_positions_count' in enabled_metrics:
                metrics['active_positions_count'] = self._get_active_positions_count()
            if 'total_position_value' in enabled_metrics:
                metrics['total_position_value'] = self._get_total_position_value()
            if 'margin_usage' in enabled_metrics:
                metrics['margin_usage'] = self._get_margin_usage()
            if 'available_margin' in enabled_metrics:
                metrics['available_margin'] = self._get_available_margin()
            if 'unrealized_pnl' in enabled_metrics:
                metrics['unrealized_pnl'] = self._get_unrealized_pnl()
            
            # Strategy Performance
            if 'win_rate' in enabled_metrics:
                metrics['win_rate'] = self._get_win_rate()
            if 'sharpe_ratio' in enabled_metrics:
                metrics['sharpe_ratio'] = self._get_sharpe_ratio()
            if 'recent_trade_outcomes' in enabled_metrics:
                metrics['recent_trade_outcomes'] = self._get_recent_trade_outcomes()
            if 'profit_factor' in enabled_metrics:
                metrics['profit_factor'] = self._get_profit_factor()
            if 'avg_win_loss_ratio' in enabled_metrics:
                metrics['avg_win_loss_ratio'] = self._get_avg_win_loss_ratio()
            if 'total_trades' in enabled_metrics:
                metrics['total_trades'] = self._get_total_trades()
            
            # Additional Core Metrics
            if 'total_pnl' in enabled_metrics:
                metrics['total_pnl'] = self._get_total_pnl()
            if 'current_strategy' in enabled_metrics:
                metrics['current_strategy'] = self._get_current_strategy()
            if 'current_price' in enabled_metrics:
                metrics['current_price'] = self._get_current_price()
            if 'price_change_24h' in enabled_metrics:
                metrics['price_change_24h'] = self._get_price_change_24h()
            if 'rsi' in enabled_metrics:
                metrics['rsi'] = self._get_current_rsi()
            if 'ema_trend' in enabled_metrics:
                metrics['ema_trend'] = self._get_ema_trend()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {'error': str(e), 'last_update': datetime.now().isoformat()}
    
    def _get_total_pnl(self) -> float:
        """Get total PnL across all trades"""
        try:
            query = """
            SELECT COALESCE(SUM(pnl), 0) as total_pnl 
            FROM trades 
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            return result[0]['total_pnl'] if result else 0.0
        except Exception as e:
            logger.error(f"Error getting total PnL: {e}")
            return 0.0
    
    def _get_current_balance(self) -> float:
        """Get current account balance"""
        try:
            # Get latest account snapshot
            query = """
            SELECT balance 
            FROM account_snapshots 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)
            return result[0]['balance'] if result else 0.0
        except Exception as e:
            logger.error(f"Error getting current balance: {e}")
            return 0.0
    
    def _get_win_rate(self) -> float:
        """Calculate win rate percentage"""
        try:
            query = """
            SELECT 
                COUNT(*) as total_trades,
                COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades
            FROM trades 
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]['total_trades'] > 0:
                return (result[0]['winning_trades'] / result[0]['total_trades']) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating win rate: {e}")
            return 0.0
    
    def _get_total_trades(self) -> int:
        """Get total number of completed trades"""
        try:
            query = "SELECT COUNT(*) as count FROM trades WHERE exit_time IS NOT NULL"
            result = self.db_manager.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting total trades: {e}")
            return 0
    
    def _get_active_positions_count(self) -> int:
        """Get number of active positions"""
        try:
            query = "SELECT COUNT(*) as count FROM positions WHERE exit_time IS NULL"
            result = self.db_manager.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting active positions: {e}")
            return 0
    
    def _get_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        try:
            query = """
            SELECT 
                MAX(balance) as peak_balance,
                MIN(balance) as min_balance
            FROM account_snapshots
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]['peak_balance']:
                peak = result[0]['peak_balance']
                min_bal = result[0]['min_balance']
                return ((peak - min_bal) / peak) * 100 if peak > 0 else 0.0
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    def _get_current_exposure(self) -> float:
        """Get current market exposure as percentage of balance"""
        try:
            # Get current balance and active position values
            balance = self._get_current_balance()
            if balance <= 0:
                return 0.0
            
            query = """
            SELECT COALESCE(SUM(quantity * entry_price), 0) as total_exposure
            FROM positions 
            WHERE exit_time IS NULL
            """
            result = self.db_manager.execute_query(query)
            exposure = result[0]['total_exposure'] if result else 0.0
            return (exposure / balance) * 100
        except Exception as e:
            logger.error(f"Error calculating current exposure: {e}")
            return 0.0
    
    def _get_risk_per_trade(self) -> float:
        """Get average risk per trade"""
        # This would typically come from the risk manager configuration
        return 1.0  # Default 1% risk per trade
    
    def _get_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        try:
            # Get daily returns from account snapshots
            query = """
            SELECT balance, DATE(timestamp) as date
            FROM account_snapshots
            ORDER BY timestamp
            """
            result = self.db_manager.execute_query(query)
            
            if len(result) < 2:
                return 0.0
            
            # Calculate daily returns
            df = pd.DataFrame(result)
            df['daily_return'] = df['balance'].pct_change()
            df = df.dropna()
            
            if len(df) == 0:
                return 0.0
            
            # Calculate Sharpe ratio (assuming 0% risk-free rate)
            mean_return = df['daily_return'].mean()
            std_return = df['daily_return'].std()
            
            if std_return > 0:
                return (mean_return / std_return) * (252 ** 0.5)  # Annualized
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    def _get_volatility(self) -> float:
        """Calculate portfolio volatility"""
        try:
            # Similar to Sharpe ratio calculation but return std dev
            query = """
            SELECT balance, DATE(timestamp) as date
            FROM account_snapshots
            ORDER BY timestamp
            """
            result = self.db_manager.execute_query(query)
            
            if len(result) < 2:
                return 0.0
            
            df = pd.DataFrame(result)
            df['daily_return'] = df['balance'].pct_change()
            df = df.dropna()
            
            if len(df) == 0:
                return 0.0
            
            return df['daily_return'].std() * (252 ** 0.5) * 100  # Annualized volatility %
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    def _get_system_health_status(self) -> str:
        """Get overall system health status"""
        try:
            # Check recent activity
            query = """
            SELECT timestamp 
            FROM account_snapshots 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)
            
            if not result:
                return "Unknown"
            
            last_update = pd.to_datetime(result[0]['timestamp'])
            time_diff = (datetime.now() - last_update).total_seconds()
            
            if time_diff < 300:  # 5 minutes
                return "Healthy"
            elif time_diff < 900:  # 15 minutes
                return "Warning"
            else:
                return "Error"
                
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return "Error"
    
    def _get_api_status(self) -> str:
        """Check API connectivity status"""
        try:
            # Try to get current price to test API
            current_price = self.data_provider.get_current_price('BTCUSDT')
            return "Connected" if current_price else "Disconnected"
        except Exception as e:
            logger.error(f"API status check failed: {e}")
            return "Disconnected"
    
    def _get_recent_error_count(self) -> int:
        """Get count of recent errors (last 24 hours)"""
        try:
            query = """
            SELECT COUNT(*) as count 
            FROM system_events 
            WHERE event_type = 'ERROR' 
            AND timestamp > datetime('now', '-1 day')
            """
            result = self.db_manager.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting error count: {e}")
            return 0
    
    def _get_current_strategy(self) -> str:
        """Get current active strategy"""
        try:
            query = """
            SELECT strategy_name 
            FROM trading_sessions 
            WHERE end_time IS NULL 
            ORDER BY start_time DESC 
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)
            return result[0]['strategy_name'] if result else "Unknown"
        except Exception as e:
            logger.error(f"Error getting current strategy: {e}")
            return "Unknown"
    
    def _get_strategy_confidence(self) -> float:
        """Get strategy confidence level"""
        # This would need to be implemented in the strategy itself
        return 75.0  # Placeholder
    
    def _get_signals_today(self) -> int:
        """Get number of signals generated today"""
        try:
            query = """
            SELECT COUNT(*) as count 
            FROM trades 
            WHERE DATE(entry_time) = DATE('now')
            """
            result = self.db_manager.execute_query(query)
            return result[0]['count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting signals today: {e}")
            return 0
    
    def _get_current_price(self) -> float:
        """Get current BTC price"""
        try:
            return self.data_provider.get_current_price('BTCUSDT')
        except Exception as e:
            logger.error(f"Error getting current price: {e}")
            return 0.0
    
    def _get_price_change_24h(self) -> float:
        """Get 24h price change percentage"""
        try:
            df = self.data_provider.get_historical_data('BTCUSDT', 
                                                       datetime.now() - timedelta(days=2), 
                                                       datetime.now())
            if len(df) >= 2:
                current_price = df.iloc[-1]['close']
                price_24h_ago = df.iloc[-25]['close'] if len(df) >= 25 else df.iloc[0]['close']
                return ((current_price - price_24h_ago) / price_24h_ago) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error getting 24h price change: {e}")
            return 0.0
    
    def _get_volume_24h(self) -> float:
        """Get 24h trading volume"""
        try:
            df = self.data_provider.get_historical_data('BTCUSDT', 
                                                       datetime.now() - timedelta(days=1), 
                                                       datetime.now())
            if not df.empty:
                return df['volume'].sum()
            return 0.0
        except Exception as e:
            logger.error(f"Error getting 24h volume: {e}")
            return 0.0
    
    def _get_current_rsi(self) -> float:
        """Get current RSI value"""
        try:
            from indicators.technical import calculate_rsi
            df = self.data_provider.get_historical_data('BTCUSDT', 
                                                       datetime.now() - timedelta(days=30), 
                                                       datetime.now())
            if len(df) > 14:
                rsi = calculate_rsi(df['close'], period=14)
                return rsi.iloc[-1] if not rsi.empty else 50.0
            return 50.0
        except Exception as e:
            logger.error(f"Error getting RSI: {e}")
            return 50.0
    
    def _get_ema_trend(self) -> str:
        """Get EMA trend direction"""
        try:
            from indicators.technical import calculate_ema
            df = self.data_provider.get_historical_data('BTCUSDT', 
                                                       datetime.now() - timedelta(days=30), 
                                                       datetime.now())
            if len(df) > 50:
                ema_short = calculate_ema(df['close'], period=9)
                ema_long = calculate_ema(df['close'], period=21)
                
                if len(ema_short) > 0 and len(ema_long) > 0:
                    if ema_short.iloc[-1] > ema_long.iloc[-1]:
                        return "Bullish"
                    else:
                        return "Bearish"
            return "Neutral"
        except Exception as e:
            logger.error(f"Error getting EMA trend: {e}")
            return "Neutral"
    
    def _get_sentiment_score(self) -> float:
        """Get current sentiment score"""
        # This would integrate with sentiment providers
        return 0.0  # Placeholder
    
    def _get_sentiment_trend(self) -> str:
        """Get sentiment trend"""
        return "Neutral"  # Placeholder
    
    # ========== SYSTEM HEALTH METRICS ==========
    
    def _get_api_connection_status(self) -> str:
        """Get API connection status"""
        try:
            # Test API connectivity by making a simple request
            current_price = self.data_provider.get_current_price('BTCUSDT')
            return "Connected" if current_price and current_price > 0 else "Disconnected"
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return "Disconnected"
    
    def _get_data_feed_status(self) -> str:
        """Get data feed status"""
        try:
            # Check when we last received data
            query = """
            SELECT timestamp 
            FROM account_snapshots 
            ORDER BY timestamp DESC 
            LIMIT 1
            """
            result = self.db_manager.execute_query(query)
            
            if not result:
                return "No Data"
            
            last_update = pd.to_datetime(result[0]['timestamp'])
            time_diff = (datetime.now() - last_update).total_seconds()
            
            if time_diff < 300:  # 5 minutes
                return "Active"
            elif time_diff < 900:  # 15 minutes
                return "Delayed"
            else:
                return "Stale"
                
        except Exception as e:
            logger.error(f"Error checking data feed status: {e}")
            return "Error"
    
    def _get_error_rate_hourly(self) -> float:
        """Get error rate over the last hour"""
        try:
            query = """
            SELECT 
                COUNT(CASE WHEN event_type = 'ERROR' THEN 1 END) as errors,
                COUNT(*) as total
            FROM system_events 
            WHERE timestamp > datetime('now', '-1 hour')
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]['total'] > 0:
                return (result[0]['errors'] / result[0]['total']) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error calculating hourly error rate: {e}")
            return 0.0
    
    def _get_api_latency(self) -> float:
        """Get average API latency in milliseconds"""
        try:
            import time
            start_time = time.time()
            # Make a simple API call to measure latency
            self.data_provider.get_current_price('BTCUSDT')
            end_time = time.time()
            return (end_time - start_time) * 1000  # Convert to milliseconds
        except Exception as e:
            logger.error(f"Error measuring API latency: {e}")
            return 0.0
    
    def _get_system_uptime(self) -> str:
        """Get system uptime"""
        try:
            # Get the earliest trading session start time as proxy for uptime
            query = """
            SELECT MIN(start_time) as earliest_start 
            FROM trading_sessions 
            WHERE end_time IS NULL
            """
            result = self.db_manager.execute_query(query)
            
            if result and result[0]['earliest_start']:
                start_time = pd.to_datetime(result[0]['earliest_start'])
                uptime = datetime.now() - start_time
                days = uptime.days
                hours, remainder = divmod(uptime.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                return f"{days}d {hours}h {minutes}m"
            return "Unknown"
        except Exception as e:
            logger.error(f"Error getting system uptime: {e}")
            return "Unknown"
    
    # ========== RISK METRICS ==========
    
    def _get_current_drawdown(self) -> float:
        """Get current drawdown from peak"""
        try:
            query = """
            SELECT balance, timestamp
            FROM account_snapshots
            ORDER BY timestamp DESC
            LIMIT 100
            """
            result = self.db_manager.execute_query(query)
            
            if len(result) < 2:
                return 0.0
            
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(result)
            df['balance'] = pd.to_numeric(df['balance'])
            
            # Calculate running maximum (peak)
            df['peak'] = df['balance'].expanding().max()
            
            # Calculate drawdown
            current_balance = df['balance'].iloc[0]  # Most recent (first in DESC order)
            current_peak = df['peak'].iloc[0]
            
            if current_peak > 0:
                drawdown = ((current_peak - current_balance) / current_peak) * 100
                return max(0, drawdown)
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating current drawdown: {e}")
            return 0.0
    
    def _get_daily_pnl(self) -> float:
        """Get P&L for today"""
        try:
            query = """
            SELECT COALESCE(SUM(pnl), 0) as daily_pnl 
            FROM trades 
            WHERE DATE(exit_time) = DATE('now')
            AND exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            return result[0]['daily_pnl'] if result else 0.0
        except Exception as e:
            logger.error(f"Error getting daily P&L: {e}")
            return 0.0
    
    def _get_weekly_pnl(self) -> float:
        """Get P&L for the last 7 days"""
        try:
            query = """
            SELECT COALESCE(SUM(pnl), 0) as weekly_pnl 
            FROM trades 
            WHERE exit_time > datetime('now', '-7 days')
            AND exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            return result[0]['weekly_pnl'] if result else 0.0
        except Exception as e:
            logger.error(f"Error getting weekly P&L: {e}")
            return 0.0
    
    def _get_total_position_sizes(self) -> float:
        """Get total value of all active positions"""
        try:
            current_price = self._get_current_price()
            if current_price <= 0:
                return 0.0
                
            query = """
            SELECT COALESCE(SUM(quantity * entry_price), 0) as total_value
            FROM positions 
            WHERE exit_time IS NULL
            """
            result = self.db_manager.execute_query(query)
            return result[0]['total_value'] if result else 0.0
        except Exception as e:
            logger.error(f"Error getting total position sizes: {e}")
            return 0.0
    
    # ========== ORDER EXECUTION METRICS ==========
    
    def _get_fill_rate(self) -> float:
        """Get order fill rate percentage"""
        try:
            # This would need to be tracked in order execution logs
            # For now, calculate based on successful vs failed trades
            query = """
            SELECT 
                COUNT(*) as total_orders,
                COUNT(CASE WHEN exit_time IS NOT NULL THEN 1 END) as filled_orders
            FROM positions
            WHERE entry_time > datetime('now', '-24 hours')
            """
            result = self.db_manager.execute_query(query)
            
            if result and result[0]['total_orders'] > 0:
                return (result[0]['filled_orders'] / result[0]['total_orders']) * 100
            return 100.0  # Default to 100% if no recent orders
        except Exception as e:
            logger.error(f"Error calculating fill rate: {e}")
            return 100.0
    
    def _get_avg_slippage(self) -> float:
        """Get average slippage percentage"""
        try:
            # Calculate slippage as difference between expected and actual execution price
            # This is a simplified calculation - in practice you'd track intended vs actual prices
            query = """
            SELECT 
                entry_price,
                exit_price,
                side
            FROM trades 
            WHERE exit_time > datetime('now', '-24 hours')
            AND exit_time IS NOT NULL
            LIMIT 50
            """
            result = self.db_manager.execute_query(query)
            
            if not result:
                return 0.0
            
            # Simple slippage estimation based on price movement
            total_slippage = 0
            count = 0
            
            for trade in result:
                # Estimate slippage as 0.01-0.05% of trade value
                estimated_slippage = 0.02  # 0.02% average slippage
                total_slippage += estimated_slippage
                count += 1
            
            return total_slippage / count if count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating average slippage: {e}")
            return 0.0
    
    def _get_failed_orders(self) -> int:
        """Get number of failed orders in last 24 hours"""
        try:
            query = """
            SELECT COUNT(*) as failed_count 
            FROM system_events 
            WHERE event_type = 'ERROR' 
            AND message LIKE '%order%' 
            AND timestamp > datetime('now', '-24 hours')
            """
            result = self.db_manager.execute_query(query)
            return result[0]['failed_count'] if result else 0
        except Exception as e:
            logger.error(f"Error getting failed orders count: {e}")
            return 0
    
    def _get_order_latency(self) -> float:
        """Get average order execution latency in milliseconds"""
        # This would require detailed order execution tracking
        # For now, return a reasonable estimate
        return 50.0  # 50ms average latency
    
    def _get_execution_quality(self) -> str:
        """Get overall execution quality status"""
        try:
            fill_rate = self._get_fill_rate()
            slippage = self._get_avg_slippage()
            failed_orders = self._get_failed_orders()
            
            if fill_rate > 95 and slippage < 0.05 and failed_orders < 5:
                return "Excellent"
            elif fill_rate > 90 and slippage < 0.1 and failed_orders < 10:
                return "Good"
            elif fill_rate > 80 and slippage < 0.2 and failed_orders < 20:
                return "Fair"
            else:
                return "Poor"
                
        except Exception as e:
            logger.error(f"Error calculating execution quality: {e}")
            return "Unknown"
    
    # ========== BALANCE & POSITIONS ==========
    
    def _get_total_position_value(self) -> float:
        """Get total value of all positions at current prices"""
        try:
            current_price = self._get_current_price()
            if current_price <= 0:
                return 0.0
                
            query = """
            SELECT COALESCE(SUM(quantity), 0) as total_quantity
            FROM positions 
            WHERE exit_time IS NULL
            """
            result = self.db_manager.execute_query(query)
            
            if result:
                total_quantity = result[0]['total_quantity']
                return total_quantity * current_price
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting total position value: {e}")
            return 0.0
    
    def _get_margin_usage(self) -> float:
        """Get margin usage percentage"""
        try:
            current_balance = self._get_current_balance()
            position_value = self._get_total_position_value()
            
            if current_balance > 0:
                # Assuming 1:1 margin (no leverage) for safety
                return (position_value / current_balance) * 100
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating margin usage: {e}")
            return 0.0
    
    def _get_available_margin(self) -> float:
        """Get available margin for new positions"""
        try:
            current_balance = self._get_current_balance()
            used_margin = self._get_total_position_value()
            
            return max(0, current_balance - used_margin)
            
        except Exception as e:
            logger.error(f"Error calculating available margin: {e}")
            return 0.0
    
    def _get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L from active positions"""
        try:
            current_price = self._get_current_price()
            if current_price <= 0:
                return 0.0
                
            query = """
            SELECT 
                side, entry_price, quantity
            FROM positions 
            WHERE exit_time IS NULL
            """
            result = self.db_manager.execute_query(query)
            
            total_unrealized = 0.0
            for position in result:
                entry_price = position['entry_price']
                quantity = position['quantity']
                side = position['side'].lower()
                
                if side == 'long':
                    unrealized = (current_price - entry_price) * quantity
                else:  # short
                    unrealized = (entry_price - current_price) * quantity
                
                total_unrealized += unrealized
            
            return total_unrealized
            
        except Exception as e:
            logger.error(f"Error calculating unrealized P&L: {e}")
            return 0.0
    
    # ========== STRATEGY PERFORMANCE ==========
    
    def _get_recent_trade_outcomes(self) -> str:
        """Get recent trade outcomes summary"""
        try:
            query = """
            SELECT pnl
            FROM trades 
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT 10
            """
            result = self.db_manager.execute_query(query)
            
            if not result:
                return "No recent trades"
            
            outcomes = []
            for trade in result:
                if trade['pnl'] > 0:
                    outcomes.append("W")
                else:
                    outcomes.append("L")
            
            return "".join(outcomes)  # e.g., "WLWWLWLWW"
            
        except Exception as e:
            logger.error(f"Error getting recent trade outcomes: {e}")
            return "Unknown"
    
    def _get_profit_factor(self) -> float:
        """Get profit factor (gross profit / gross loss)"""
        try:
            query = """
            SELECT 
                SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit,
                SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss
            FROM trades 
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            
            if result and result[0]['gross_loss'] > 0:
                gross_profit = result[0]['gross_profit'] or 0
                gross_loss = result[0]['gross_loss'] or 1  # Avoid division by zero
                return gross_profit / gross_loss
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating profit factor: {e}")
            return 0.0
    
    def _get_avg_win_loss_ratio(self) -> float:
        """Get average win to loss ratio"""
        try:
            query = """
            SELECT 
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN ABS(pnl) END) as avg_loss
            FROM trades 
            WHERE exit_time IS NOT NULL
            """
            result = self.db_manager.execute_query(query)
            
            if result and result[0]['avg_loss'] and result[0]['avg_loss'] > 0:
                avg_win = result[0]['avg_win'] or 0
                avg_loss = result[0]['avg_loss'] or 1
                return avg_win / avg_loss
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating win/loss ratio: {e}")
            return 0.0
    
    def _get_current_positions(self) -> List[Dict[str, Any]]:
        """Get current active positions"""
        try:
            query = """
            SELECT 
                symbol, side, entry_price, quantity, entry_time,
                stop_loss, take_profit, order_id
            FROM positions 
            WHERE exit_time IS NULL
            ORDER BY entry_time DESC
            """
            result = self.db_manager.execute_query(query)
            
            positions = []
            current_price = self._get_current_price()
            
            for row in result:
                # Calculate unrealized PnL
                entry_price = row['entry_price']
                quantity = row['quantity']
                
                if row['side'].lower() == 'long':
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:
                    unrealized_pnl = (entry_price - current_price) * quantity
                
                positions.append({
                    'symbol': row['symbol'],
                    'side': row['side'],
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'quantity': quantity,
                    'unrealized_pnl': unrealized_pnl,
                    'entry_time': row['entry_time'],
                    'stop_loss': row['stop_loss'],
                    'take_profit': row['take_profit'],
                    'order_id': row['order_id']
                })
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting current positions: {e}")
            return []
    
    def _get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent completed trades"""
        try:
            query = """
            SELECT 
                symbol, side, entry_price, exit_price, quantity,
                entry_time, exit_time, pnl, exit_reason
            FROM trades 
            WHERE exit_time IS NOT NULL
            ORDER BY exit_time DESC
            LIMIT ?
            """
            result = self.db_manager.execute_query(query, (limit,))
            return [dict(row) for row in result] if result else []
        except Exception as e:
            logger.error(f"Error getting recent trades: {e}")
            return []
    
    def _get_performance_chart_data(self, days: int = 7) -> Dict[str, List]:
        """Get performance chart data for the specified number of days"""
        try:
            query = """
            SELECT balance, timestamp
            FROM account_snapshots
            WHERE timestamp > datetime('now', '-{} days')
            ORDER BY timestamp
            """.format(days)
            
            result = self.db_manager.execute_query(query)
            
            timestamps = []
            balances = []
            
            for row in result:
                timestamps.append(row['timestamp'])
                balances.append(row['balance'])
            
            return {
                'timestamps': timestamps,
                'balances': balances
            }
            
        except Exception as e:
            logger.error(f"Error getting performance chart data: {e}")
            return {'timestamps': [], 'balances': []}
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'database_connected': True,  # If we're here, DB is connected
            'api_status': self._get_api_status(),
            'last_trade': self._get_last_trade_time(),
            'uptime': self._get_uptime(),
            'memory_usage': self._get_memory_usage(),
            'error_rate': self._get_error_rate()
        }
    
    def _get_last_trade_time(self) -> str:
        """Get timestamp of last trade"""
        try:
            query = """
            SELECT MAX(entry_time) as last_trade 
            FROM trades
            """
            result = self.db_manager.execute_query(query)
            return result[0]['last_trade'] if result and result[0]['last_trade'] else "Never"
        except Exception as e:
            logger.error(f"Error getting last trade time: {e}")
            return "Unknown"
    
    def _get_uptime(self) -> str:
        """Get system uptime"""
        # This would need to be tracked separately
        return "Unknown"
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def _get_error_rate(self) -> float:
        """Get error rate over last hour"""
        try:
            query = """
            SELECT 
                COUNT(CASE WHEN event_type = 'ERROR' THEN 1 END) as errors,
                COUNT(*) as total
            FROM system_events 
            WHERE timestamp > datetime('now', '-1 hour')
            """
            result = self.db_manager.execute_query(query)
            if result and result[0]['total'] > 0:
                return (result[0]['errors'] / result[0]['total']) * 100
            return 0.0
        except Exception as e:
            logger.error(f"Error getting error rate: {e}")
            return 0.0
    
    def start_monitoring(self):
        """Start the monitoring update thread"""
        if self.is_running:
            return
        
        self.is_running = True
        self.update_thread = threading.Thread(target=self._monitoring_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Monitoring thread started")
    
    def stop_monitoring(self):
        """Stop the monitoring update thread"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=10)
        logger.info("Monitoring thread stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop that broadcasts updates"""
        while self.is_running:
            try:
                metrics = self._collect_metrics()
                self.socketio.emit('metrics_update', metrics)
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.update_interval)
    
    def run(self, host='0.0.0.0', port=8080, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting monitoring dashboard on {host}:{port}")
        self.start_monitoring()
        try:
            self.socketio.run(self.app, host=host, port=port, debug=debug)
        finally:
            self.stop_monitoring()

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot Monitoring Dashboard')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--db-url', help='Database URL')
    parser.add_argument('--update-interval', type=int, default=3600, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    dashboard = MonitoringDashboard(
        db_url=args.db_url,
        update_interval=args.update_interval
    )
    
    dashboard.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )

if __name__ == '__main__':
    main()