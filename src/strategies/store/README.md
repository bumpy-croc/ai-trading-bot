# Strategy Version Store

This directory contains version history for all strategies as JSON files.

## Contents

- `ml_basic.json` - ML Basic strategy versions
- `ml_adaptive.json` - ML Adaptive strategy versions
- `ml_sentiment.json` - ML Sentiment strategy versions
- `ensemble_weighted.json` - Ensemble Weighted strategy versions
- `momentum_leverage.json` - Momentum Leverage strategy versions

## Management

These files are **automatically managed** by the pre-commit hook in `.git/hooks/pre-commit`.

When you modify a strategy file, the hook:
1. Detects configuration changes
2. Prompts for version update
3. Saves the new version to this directory
4. Auto-stages the file for commit

See [docs/development.md](../../../docs/development.md#strategy-versioning) for details.

## Loading Versions

Load strategy versions at application startup:

```python
from pathlib import Path
from src.strategies.components import StrategyRegistry
import json

def load_strategy_configs():
    registry = StrategyRegistry()
    
    store_dir = Path("src/strategies/store")
    for config_file in store_dir.glob("*.json"):
        data = json.loads(config_file.read_text())
        registry.deserialize_strategy(data)
    
    return registry
```
