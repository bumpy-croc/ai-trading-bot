import ccxt
import pandas as pd
from datetime import datetime

# Initialize Binance exchange
exchange = ccxt.binance()

# Define the symbol and timeframe
symbol = 'BTC/USDT'
timeframe = '1d'

# Define the date range
start_date = '2017-01-01T00:00:00Z'
end_date = '2023-10-01T00:00:00Z'

# Convert dates to timestamps
start_timestamp = exchange.parse8601(start_date)
end_timestamp = exchange.parse8601(end_date)

# Fetch historical data
ohlcv = []
since = start_timestamp
while since < end_timestamp:
    data = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
    if not data:
        break
    ohlcv.extend(data)
    since = data[-1][0] + 86400000  # Move to the next day

# Convert to DataFrame
columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
df = pd.DataFrame(ohlcv, columns=columns)

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# Save to CSV
output_file = 'data/btcusdt.csv'
df.to_csv(output_file, index=False)

print(f"Data saved to {output_file}") 