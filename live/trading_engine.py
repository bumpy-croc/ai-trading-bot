import logging
import time
from typing import Optional
from datetime import datetime
import pandas as pd

from core.data import DataProvider
from strategies import BaseStrategy
from core.risk import RiskManager, RiskParameters

logger = logging.getLogger(__name__)

class LiveTradingEngine:
    """Live trading engine that executes strategies in real-time"""
    
    def __init__(
        self,
        strategy: BaseStrategy,
        data_provider: DataProvider,
        risk_parameters: Optional[RiskParameters] = None,
        check_interval: int = 60  # seconds
    ):
        self.strategy = strategy
        self.data_provider = data_provider
        self.risk_manager = RiskManager(risk_parameters)
        self.check_interval = check_interval
        self.is_running = False
        self.current_position = None
        
    def start(self, symbol: str, timeframe: str):
        """Start the trading engine"""
        self.is_running = True
        logger.info(f"Starting live trading for {symbol} on {timeframe} timeframe")
        
        while self.is_running:
            try:
                # Fetch latest data
                df = self.data_provider.get_live_data(symbol, timeframe)
                if df.empty:
                    logger.warning("No data received")
                    time.sleep(self.check_interval)
                    continue
                    
                # Prepare data with indicators
                df = self.strategy.prepare_data(df)
                
                # Get current market state
                current_index = len(df) - 1
                
                # Check for exit if in position
                if self.current_position is not None:
                    if self.strategy.check_exit_conditions(df, current_index, self.current_position['entry_price']):
                        self._exit_position(df.iloc[-1])
                        
                # Check for entry if not in position
                elif self.strategy.check_entry_conditions(df, current_index):
                    self._enter_position(df, df.iloc[-1], symbol)
                    
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(self.check_interval)
                
    def stop(self):
        """Stop the trading engine"""
        self.is_running = False
        logger.info("Stopping trading engine")
        
        # Close any open position
        if self.current_position is not None:
            logger.info("Closing open position")
            self._exit_position(None)  # Force exit
            
    def _enter_position(self, data: pd.DataFrame, candle: pd.Series, symbol: str):
        """Enter a new position"""
        try:
            # Calculate position size
            size = self.strategy.calculate_position_size(
                data,
                len(data) - 1,
                self.risk_manager.get_total_exposure()
            )
            
            # Create new position
            self.current_position = {
                'symbol': symbol,
                'side': 'long',  # Currently only supporting long positions
                'entry_price': candle['close'],
                'entry_time': candle.name,
                'size': size,
                'stop_loss': self.strategy.calculate_stop_loss(data, len(data) - 1, candle['close'], 'long')
            }
            
            # Update risk tracking
            self.risk_manager.update_position(
                symbol,
                'long',
                size,
                candle['close']
            )
            
            logger.info(f"Entered long position at {candle['close']}")
            
        except Exception as e:
            logger.error(f"Error entering position: {e}")
            
    def _exit_position(self, candle: Optional[pd.Series]):
        """Exit current position"""
        if self.current_position is None:
            return
            
        try:
            exit_price = candle['close'] if candle is not None else None
            exit_time = candle.name if candle is not None else datetime.now()
            
            logger.info(f"Exited position at {exit_price}")
            
            # Clear position and risk tracking
            self.risk_manager.close_position(self.current_position['symbol'])
            self.current_position = None
            
        except Exception as e:
            logger.error(f"Error exiting position: {e}") 