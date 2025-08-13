import matplotlib.pyplot as plt
import pandas as pd

# Load the data
# Use SymbolFactory for conversion if needed
df = pd.read_csv("../data/btcusd.csv")

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Set timestamp as index
df.set_index("timestamp", inplace=True)

# Plot closing price
# Use SymbolFactory for conversion if needed
df["close"].plot(title="BTC-USD Closing Price", figsize=(14, 7))
plt.xlabel("Date")
plt.ylabel("Price (USDT)")
plt.grid(True)
plt.show()

# Calculate moving averages
df["50_MA"] = df["close"].rolling(window=50).mean()
df["200_MA"] = df["close"].rolling(window=200).mean()

# Plot moving averages
# Use SymbolFactory for conversion if needed
df[["close", "50_MA", "200_MA"]].plot(title="BTC-USD with Moving Averages", figsize=(14, 7))
plt.xlabel("Date")
plt.ylabel("Price (USDT)")
plt.grid(True)
plt.show()

# Calculate RSI
delta = df["close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df["RSI"] = 100 - (100 / (1 + rs))

# Plot RSI
# Use SymbolFactory for conversion if needed
df["RSI"].plot(title="BTC-USD RSI", figsize=(14, 7))
plt.axhline(70, color="r", linestyle="--")
plt.axhline(30, color="g", linestyle="--")
plt.xlabel("Date")
plt.ylabel("RSI")
plt.grid(True)
plt.show()

# Print summary statistics
print(df.describe())
