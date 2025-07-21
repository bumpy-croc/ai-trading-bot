#!/usr/bin/env python3
"""
Train an updated price-prediction model (v2) with log-return / EMA normalization
and optionally different candle timeframes.

Usage:
    python scripts/train_price_model.py --symbol BTCUSDT --timeframe 1h \
        --start 2018-01-01 --end 2024-12-31 --epochs 10 --output ml/btcusdt_price_v2.onnx

The script:
1. Pulls historical OHLCV data via the existing data-provider utilities
2. Resamples to desired timeframe (1h default, can be 4h)
3. Computes log-returns and volume z-scores for scale-invariant inputs
4. Trains a simple 2-layer GRU network in PyTorch
5. Exports the network to ONNX for inference in production strategies

Note: Designed for quick experimentation ‑ not heavy hyper-parameter tuning.
"""
import argparse
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Ensure project modules are importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.extend([str(ROOT), str(ROOT / "src")])

from data_providers import BinanceDataProvider  # noqa: E402

# ----------------------- Dataset ------------------------------------------- #

class PriceSequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int = 120):
        self.seq_len = seq_len
        # Use log-returns for price (close) & pct-change for high/low
        df = df.copy()
        df["log_return"] = np.log(df["close"]).diff()
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["volume_z"] = (df["volume"] - df["volume"].rolling(1000).mean()) / (
            df["volume"].rolling(1000).std() + 1e-9
        )
        df = df.dropna()
        feats = ["log_return", "hl_range", "volume_z"]
        self.data = df[feats].values.astype(np.float32)
        self.targets = df["log_return"].shift(-1).values.astype(np.float32)  # next bar return
        # Trim
        self.data = self.data[:-1]
        self.targets = self.targets[:-1]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.from_numpy(x), torch.from_numpy(np.array([y]))

# ----------------------- Model -------------------------------------------- #

class GRUModel(nn.Module):
    def __init__(self, input_size: int = 3, hidden_size: int = 32, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.gru(x)
        return self.fc(out[:, -1])  # prediction only from last output

# ----------------------- Training routine --------------------------------- #

def train_model(df: pd.DataFrame, seq_len: int, epochs: int = 10, lr: float = 1e-3):
    ds = PriceSequenceDataset(df, seq_len)
    dl = DataLoader(ds, batch_size=256, shuffle=True, drop_last=True)
    model = GRUModel()
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
        print(f"Epoch {epoch+1}/{epochs}  |  Loss: {np.mean(losses):.6f}")

    return model.cpu()

# ----------------------- Main ------------------------------------------------ #

def fetch_data(symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
    provider = BinanceDataProvider()
    df = provider.get_historical_data(symbol, timeframe, start, end)
    # Ensure numeric cols
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna()


def export_onnx(model: nn.Module, seq_len: int, output_path: Path):
    model.eval()
    dummy = torch.zeros((1, seq_len, 3), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input_sequence"],
        output_names=["next_log_return"],
        dynamic_axes={"input_sequence": {0: "batch"}},
        opset_version=17,
    )
    print(f"✅ Saved ONNX model to {output_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--timeframe", default="1h", choices=["1h", "4h"], help="Candle timeframe for training")
    p.add_argument("--seq_len", type=int, default=120)
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=datetime.utcnow().strftime("%Y-%m-%d"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--output", default="ml/btcusdt_price_v2.onnx")
    return p.parse_args()


def main():
    args = parse_args()
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    print("📥 Fetching data …")
    df = fetch_data(args.symbol, args.timeframe, start_dt, end_dt)
    print(f"Loaded {len(df)} candles")
    model = train_model(df, seq_len=args.seq_len, epochs=args.epochs)
    export_onnx(model, args.seq_len, Path(args.output))

if __name__ == "__main__":
    main()