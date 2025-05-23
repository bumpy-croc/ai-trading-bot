import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('data/btcusdt.csv')

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Set timestamp as index
df.set_index('timestamp', inplace=True)

# Plot closing price
df['close'].plot(title='BTC/USDT Closing Price', figsize=(14, 7))
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.grid(True)
plt.show()

# Calculate moving averages
df['50_MA'] = df['close'].rolling(window=50).mean()
df['200_MA'] = df['close'].rolling(window=200).mean()

# Plot moving averages
df[['close', '50_MA', '200_MA']].plot(title='BTC/USDT with Moving Averages', figsize=(14, 7))
plt.xlabel('Date')
plt.ylabel('Price (USDT)')
plt.grid(True)
plt.show()

# Calculate RSI
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Plot RSI
df['RSI'].plot(title='BTC/USDT RSI', figsize=(14, 7))
plt.axhline(70, color='r', linestyle='--')
plt.axhline(30, color='g', linestyle='--')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.grid(True)
plt.show()

# Print summary statistics
print(df.describe()) 