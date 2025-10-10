# Strategy Versioning Guide

## Overview

Strategy versions are automatically tracked when you modify strategy files. The pre-commit hook detects changes and prompts you to version them before committing.

## How It Works

### 1. Automatic Detection

When you commit changes to strategy files (e.g., `src/strategies/ml_basic.py`), the pre-commit hook:
- Detects modified strategy files
- Calculates configuration hash
- Compares with previously saved version
- Prompts for version update if configuration changed

### 2. Versioning Workflow

```bash
# 1. Modify your strategy
vim src/strategies/ml_basic.py

# 2. Make your changes (update thresholds, logic, etc.)

# 3. Stage the changes
git add src/strategies/ml_basic.py

# 4. Commit (hook runs automatically)
git commit -m "feat: improve ML basic stop loss"

# The hook will:
# - Detect the change
# - Show old vs new configuration hash
# - Prompt you for change description
# - Ask if it's a major version
# - Save the new version to strategies/ml_basic.json
# - Auto-stage the JSON file for commit
```

### 3. Interactive Prompts

When the hook detects changes, you'll see:

```
============================================================
Processing: src/strategies/ml_basic.py
============================================================
⚠ Configuration changed detected!
  Old hash: a1b2c3d4e5f6g7h8...
  New hash: z9y8x7w6v5u4t3s2...

Describe the changes made (one per line, empty line to finish):
  - Reduced stop loss from 2% to 1.5%
  - Updated confidence threshold
  - 

Is this a major version change? (y/N): n

✓ Updated ml_basic to version 1.0.2
✓ Saved ml_basic version to src/strategies/store/ml_basic.json
============================================================
✓ Updated 1 strategy version(s)
  Config files staged for commit in src/strategies/store/
============================================================
```

## Version Storage

### File Structure

```
src/strategies/store/
├── ml_basic.json           # ML Basic strategy versions
├── ml_adaptive.json        # ML Adaptive strategy versions
├── ml_sentiment.json       # ML Sentiment strategy versions
└── ensemble_weighted.json  # Ensemble strategy versions
```

### Version File Format

```json
{
  "metadata": {
    "id": "ml_basic_a1b2c3d4",
    "name": "MlBasic",
    "version": "1.0.2",
    "created_at": "2025-10-10T12:34:56",
    "created_by": "developer",
    "description": "ML-based trading strategy",
    "tags": ["ml"],
    "status": "EXPERIMENTAL",
    "signal_generator_config": {...},
    "risk_manager_config": {...},
    "position_sizer_config": {...},
    "config_hash": "abc123...",
    "component_hash": "def456..."
  },
  "versions": [
    {
      "version": "1.0.0",
      "strategy_id": "ml_basic_a1b2c3d4",
      "created_at": "2025-10-08T10:00:00",
      "changes": ["Initial version"],
      "is_major": true,
      "signal_generator_config": {...},
      "config_hash": "..."
    },
    {
      "version": "1.0.1",
      "strategy_id": "ml_basic_a1b2c3d4",
      "created_at": "2025-10-09T14:30:00",
      "changes": ["Updated stop loss threshold"],
      "is_major": false,
      "signal_generator_config": {...},
      "config_hash": "..."
    },
    {
      "version": "1.0.2",
      "strategy_id": "ml_basic_a1b2c3d4",
      "created_at": "2025-10-10T12:34:56",
      "changes": [
        "Reduced stop loss from 2% to 1.5%",
        "Updated confidence threshold"
      ],
      "is_major": false,
      "signal_generator_config": {...},
      "config_hash": "..."
    }
  ],
  "lineage": {...}
}
```

## Semantic Versioning

The system uses semantic versioning (MAJOR.MINOR.PATCH):

- **Patch** (1.0.0 → 1.0.1): Minor changes, parameter tweaks
- **Major** (1.0.0 → 2.0.0): Breaking changes, complete rewrites

```bash
# When prompted:
Is this a major version change? (y/N): 

# Answer 'n' or just press Enter for patch version
# Answer 'y' for major version bump
```

## When Versions are Updated

✅ **Version IS updated when:**
- Strategy parameters change (thresholds, stop loss, etc.)
- Component configurations change
- Model paths or settings change
- Trading logic modifications

❌ **Version is NOT updated when:**
- Only code formatting changes
- Comments added/updated
- Refactoring with no behavior change
- Test file updates
- Documentation updates

The hook uses configuration hash comparison to detect meaningful changes.

## Git Integration

### Reviewing Version Changes

```bash
# See version history
git log src/strategies/store/ml_basic.json

# See what changed in a specific version
git show abc123:src/strategies/store/ml_basic.json

# Compare versions
git diff HEAD~1 HEAD -- src/strategies/store/ml_basic.json
```

### Rollback

```bash
# Rollback to previous version
git revert HEAD

# Rollback to specific version
git checkout abc123 -- src/strategies/store/ml_basic.json src/strategies/ml_basic.py
git commit -m "Revert to version 1.0.1"
```

### Pull Request Review

Reviewers can see both:
- **Code changes** in `src/strategies/ml_basic.py`
- **Version changes** in `src/strategies/store/ml_basic.json`

This ensures configuration changes are reviewed alongside code changes.

## Skipping the Hook

If you need to commit without running the hook (not recommended):

```bash
git commit --no-verify -m "message"
```

## Loading Strategy Versions

At application startup, load strategy configurations:

```python
from pathlib import Path
from src.strategies.components import StrategyRegistry
import json

def load_strategy_configs():
    """Load all strategy configs from disk"""
    registry = StrategyRegistry()
    
    configs_dir = Path("src/strategies/store")
    for config_file in configs_dir.glob("*.json"):
        data = json.loads(config_file.read_text())
        registry.deserialize_strategy(data)
        print(f"Loaded {config_file.stem} version {data['metadata']['version']}")
    
    return registry
```

## Manual Version Update

If you need to manually update a version (without the hook):

```bash
# Run the versioning script directly
python scripts/update_strategy_versions.py
```

## Troubleshooting

### Hook not running

```bash
# Check if hook is executable
ls -la .git/hooks/pre-commit

# Make it executable
chmod +x .git/hooks/pre-commit
```

### Import errors

If the hook fails due to import errors, it will skip versioning and allow the commit to proceed. Fix the import errors and run:

```bash
python scripts/update_strategy_versions.py
git add src/strategies/store/*.json
git commit --amend
```

### Configuration not detected as changed

The hook uses configuration hash (based on `get_parameters()` output). If you modified strategy behavior but the hash didn't change, manually update the version file.

## Best Practices

1. **Descriptive change messages**: Be specific about what changed
2. **Atomic commits**: One logical change per commit (including version)
3. **Review version files**: Check the JSON diffs in PRs
4. **Track breaking changes**: Use major versions for breaking changes
5. **Test before commit**: Ensure your changes work before committing

