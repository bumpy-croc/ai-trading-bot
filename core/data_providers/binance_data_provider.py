from datetime import datetime
import pandas as pd
from binance.client import Client
from typing import Optional
import logging
from .data_provider import DataProvider
from config.settings import API_KEY, API_SECRET

logger = logging.getLogger(__name__)

class BinanceDataProvider(DataProvider):
    """Implementation of DataProvider for Binance exchange"""
    
    TIMEFRAME_MAPPING = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
    }
    
    def __init__(self):
        super().__init__()
        self.client = Client(API_KEY, API_SECRET)
        
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert generic timeframe to Binance-specific interval"""
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        return self.TIMEFRAME_MAPPING[timeframe]
        
    def _process_klines(self, klines: list) -> pd.DataFrame:
        """Convert raw klines data to DataFrame"""
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignored'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.set_index('timestamp')
        
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Fetch historical klines data from Binance"""
        try:
            interval = self._convert_timeframe(timeframe)
            start_ts = int(start.timestamp() * 1000)
            end_ts = int(end.timestamp() * 1000) if end else None
            
            klines = self.client.get_historical_klines(
                symbol,
                interval,
                start_ts,
                end_ts
            )
            
            df = self._process_klines(klines)
            self.data = df
            
            logger.info(f"Fetched {len(df)} candles from {df.index.min()} to {df.index.max()}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            raise
            
    def get_live_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> pd.DataFrame:
        """Fetch current market data"""
        try:
            interval = self._convert_timeframe(timeframe)
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = self._process_klines(klines)
            self.data = df
            return df
            
        except Exception as e:
            logger.error(f"Error fetching live data: {str(e)}")
            raise
            
    def update_live_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Update the latest market data"""
        try:
            interval = self._convert_timeframe(timeframe)
            latest_kline = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1
            )
            
            if not latest_kline:
                return self.data
                
            latest_df = self._process_klines(latest_kline)
            
            if self.data is not None:
                # Update or append the latest candle
                self.data = pd.concat([
                    self.data[~self.data.index.isin(latest_df.index)],
                    latest_df
                ]).sort_index()
            else:
                self.data = latest_df
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error updating live data: {str(e)}")
            raise 