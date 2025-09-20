# ML Models by Symbol

This directory contains organized ML models grouped by trading symbol.

## Structure

```
models/
├── BTCUSDT/           # Bitcoin models
│   ├── basic/         # Basic price prediction models
│   └── sentiment/     # Sentiment analysis models
└── ETHUSDT/           # Ethereum models
    ├── basic/         # Basic price prediction models  
    └── sentiment/     # Sentiment analysis models
```

## Usage

Models in this directory are organized for future expansion but are currently discovered by the prediction engine from the parent `ml/` directory. The symbol-specific organization helps maintain model versions and training artifacts by trading pair.

## Model Types

- **basic/**: Price prediction models using technical indicators
- **sentiment/**: Sentiment-based prediction models using external data

See the parent `ml/README.md` and `docs/MODEL_TRAINING_AND_INTEGRATION_GUIDE.md` for more details on model training and deployment.