#!/usr/bin/env python3
"""
Train a price-only model compatible with MlBasic (5 normalized OHLCV features, predict next normalized close).

Usage:
  python scripts/train_price_only_model.py --symbol ETHUSDT --timeframe 1h \
      --start 2018-01-01 --end 2025-01-01 --seq_len 120 --epochs 5 \
      --output src/ml/ethusdt_price.onnx

Notes:
- Uses Binance data provider (public if no keys set).
- Exports ONNX with input shape (1, seq_len, 5) and a single scalar output.
- Saves minimal metadata JSON alongside the ONNX.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Lazy import torch so the script can show a helpful error if missing
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for training. Please install with: pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from None

# Ensure project modules are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

from data_providers import BinanceDataProvider  # noqa: E402
from data_providers.coinbase_provider import CoinbaseProvider  # noqa: E402
from prediction.features.price_only import PriceOnlyFeatureExtractor  # noqa: E402


class PriceOnlySequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 120):
        self.seq_len = seq_len
        # Build normalized OHLCV features in fixed order to match MlBasic
        extractor = PriceOnlyFeatureExtractor(normalization_window=seq_len)
        df = extractor.extract(df)
        # Ensure no NaNs
        df = df.dropna()
        feature_names = extractor.get_feature_names()
        data = df[feature_names].values.astype(np.float32)
        target = df["close_normalized"].values.astype(np.float32)
        # Build sliding windows (X: seq_len x 5, y: scalar next normalized close)
        X, y = [], []
        # We predict value at index i using previous seq_len rows ending at (i-1)
        for i in range(seq_len, len(data)):
            X.append(data[i - seq_len : i, :])
            y.append(target[i])
        self.X = np.stack(X, axis=0) if X else np.zeros((0, seq_len, 5), dtype=np.float32)
        self.y = (
            np.array(y, dtype=np.float32).reshape(-1, 1)
            if y
            else np.zeros((0, 1), dtype=np.float32)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class GRUPriceOnly(nn.Module):
    def __init__(self, input_size: int = 5, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1])


def _to_ccxt_symbol(symbol: str) -> str:
    if "/" in symbol:
        return symbol
    # Convert e.g., BTCUSDT -> BTC/USDT
    if symbol.endswith("USDT"):
        return symbol[:-4] + "/USDT"
    if symbol.endswith("USD"):
        return symbol[:-3] + "/USD"
    # Fallback: insert slash before last 3 or 4 chars
    return symbol[:-4] + "/" + symbol[-4:]


def _fetch_with_ccxt(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    import ccxt  # lazy import

    ex = ccxt.binance({"enableRateLimit": True, "options": {"adjustForTimeDifference": True}})
    market = _to_ccxt_symbol(symbol)
    tf = timeframe
    ms_start = int(start.timestamp() * 1000)
    ms_end = int(end.timestamp() * 1000)
    limit = 1000
    all_rows = []
    since = ms_start
    # Loop in chunks to cover the full range
    while since < ms_end:
        batch = ex.fetch_ohlcv(market, timeframe=tf, since=since, limit=limit)
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + 1  # move forward
        # Safety: stop if no progress
        if len(batch) < limit:
            break
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    # First try Binance provider (python-binance). If unavailable or empty, use Coinbase, then ccxt.
    try:
        provider = BinanceDataProvider()
        df = provider.get_historical_data(symbol, timeframe, start, end)
        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
        ):
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            return df.dropna()
    except Exception:
        pass
    # Coinbase fallback
    try:
        cb = CoinbaseProvider()
        df = cb.get_historical_data(symbol, timeframe, start, end)
        if (
            isinstance(df, pd.DataFrame)
            and not df.empty
            and set(["open", "high", "low", "close", "volume"]).issubset(df.columns)
        ):
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            print("Falling back to Coinbase provider …")
            return df.dropna()
    except Exception:
        pass
    # CCXT fallback
    print("Falling back to ccxt for historical data …")
    df = _fetch_with_ccxt(symbol, timeframe, start, end)
    if df.empty:
        raise RuntimeError("Failed to fetch historical data from Binance, Coinbase, and ccxt.")
    return df


def train_model(df: pd.DataFrame, seq_len: int, epochs: int = 10, lr: float = 1e-3) -> nn.Module:
    ds = PriceOnlySequenceDataset(df, seq_len)
    if len(ds) == 0:
        raise ValueError(
            "Not enough data to build training windows. Try reducing seq_len or extending date range."
        )
    dl = DataLoader(ds, batch_size=512, shuffle=True, drop_last=True)
    model = GRUPriceOnly()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        losses = []
        for X, y in dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs} | Loss: {np.mean(losses):.6f} | Batches: {len(dl)}")
    return model.cpu()


def export_onnx(model: nn.Module, seq_len: int, output_path: Path):
    model.eval()
    dummy = torch.zeros((1, seq_len, 5), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input_sequence"],
        output_names=["next_normalized_close"],
        dynamic_axes={"input_sequence": {0: "batch"}},
        opset_version=17,
    )
    print(f"Saved ONNX model to {output_path}")
    # Write minimal metadata JSON for engine discovery
    metadata_path = output_path.with_suffix("").as_posix() + "_metadata.json"
    metadata = {
        "sequence_length": seq_len,
        "feature_count": 5,
        "normalization_params": {},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)
    print(f"Saved metadata to {metadata_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--timeframe", default="1h", choices=["1h", "4h", "1d"])
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--output", default="src/ml/ethusdt_price.onnx")
    return p.parse_args()


def main():
    args = parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    # Avoid future timestamps for ccxt
    if end_dt > datetime.utcnow():
        end_dt = datetime.utcnow() - timedelta(hours=1)
    print("Fetching data …")
    df = fetch_data(args.symbol, args.timeframe, start_dt, end_dt)
    print(f"Loaded {len(df)} candles from {df.index.min()} to {df.index.max()}")
    model = train_model(df, seq_len=args.seq_len, epochs=args.epochs)
    export_onnx(model, args.seq_len, Path(args.output))


if __name__ == "__main__":
    main()
