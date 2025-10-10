#!/usr/bin/env python3
"""
Pre-commit hook to automatically version strategy changes.

This script:
1. Detects modified strategy files
2. Computes configuration hash
3. Compares with saved version
4. Prompts for version update if changed
5. Saves new version to JSON
"""

import hashlib
import json
import sys
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


STRATEGIES_DIR = Path("src/strategies/store")
STRATEGIES_DIR.mkdir(exist_ok=True, parents=True)


def lazy_import_strategies():
    """Lazy import strategies to avoid import-time issues."""
    from src.strategies import EnsembleWeighted, MlAdaptive, MlBasic, MlSentiment
    from src.strategies.components import StrategyRegistry
    
    return {
        "StrategyRegistry": StrategyRegistry,
        "MlBasic": MlBasic,
        "MlAdaptive": MlAdaptive,
        "MlSentiment": MlSentiment,
        "EnsembleWeighted": EnsembleWeighted,
    }


def get_modified_strategy_files() -> list[Path]:
    """Get list of modified strategy Python files from git."""
    import subprocess
    
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
            capture_output=True,
            text=True,
            check=True
        )
        files = result.stdout.strip().split('\n')
        
        # Filter for strategy files
        strategy_files = [
            Path(f) for f in files 
            if f.startswith('src/strategies/') 
            and f.endswith('.py')
            and not f.endswith('__init__.py')
            and not f.endswith('base.py')
            and 'components/' not in f
            and 'adapters/' not in f
            and 'migration/' not in f
        ]
        
        return strategy_files
    except subprocess.CalledProcessError:
        return []


def calculate_strategy_hash(strategy_instance) -> str:
    """Calculate hash of strategy configuration."""
    try:
        params = strategy_instance.get_parameters()
        params_str = json.dumps(params, sort_keys=True)
        return hashlib.sha256(params_str.encode()).hexdigest()
    except Exception as e:
        print(f"Warning: Could not calculate hash: {e}")
        return ""


def load_existing_version(strategy_name: str) -> Optional[dict]:
    """Load existing strategy version from JSON file."""
    config_file = STRATEGIES_DIR / f"{strategy_name}.json"
    if config_file.exists():
        return json.loads(config_file.read_text())
    return None


def prompt_for_changes() -> tuple[list[str], bool]:
    """Prompt user for change description and version type."""
    print("\nDescribe the changes made (one per line, empty line to finish):")
    changes = []
    while True:
        line = input("  - ").strip()
        if not line:
            break
        changes.append(line)
    
    if not changes:
        changes = ["Updated strategy implementation"]
    
    is_major = input("\nIs this a major version change? (y/N): ").lower() == 'y'
    
    return changes, is_major


def save_strategy_version(registry, strategy_id: str, strategy_name: str):
    """Save strategy version to JSON file."""
    data = registry.serialize_strategy(strategy_id)
    config_file = STRATEGIES_DIR / f"{strategy_name}.json"
    config_file.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"✓ Saved {strategy_name} version to {config_file}")


def process_strategy_file(strategy_file: Path) -> bool:
    """
    Process a modified strategy file and update version if needed.
    
    Returns:
        True if version was updated, False otherwise
    """
    # Lazy import to avoid import errors during hook installation
    try:
        imports = lazy_import_strategies()
        StrategyRegistry = imports["StrategyRegistry"]
        strategy_classes = {
            "MlBasic": imports["MlBasic"],
            "MlAdaptive": imports["MlAdaptive"],
            "MlSentiment": imports["MlSentiment"],
            "EnsembleWeighted": imports["EnsembleWeighted"],
        }
    except Exception as e:
        print(f"✗ Failed to import strategies: {e}")
        print("Skipping strategy versioning (imports failed)")
        return False
    
    # Extract strategy name from filename
    strategy_name = strategy_file.stem
    
    # Convert to class name (e.g., ml_basic -> MlBasic)
    class_name = ''.join(word.capitalize() for word in strategy_name.split('_'))
    
    if class_name not in strategy_classes:
        print(f"⊘ Skipping {strategy_file} (not in registry)")
        return False
    
    print(f"\n{'='*60}")
    print(f"Processing: {strategy_file}")
    print(f"{'='*60}")
    
    # Instantiate strategy
    try:
        strategy_class = strategy_classes[class_name]
        strategy_instance = strategy_class()
    except Exception as e:
        print(f"✗ Failed to instantiate {class_name}: {e}")
        return False
    
    # Calculate current configuration hash
    current_hash = calculate_strategy_hash(strategy_instance)
    
    # Load existing version
    existing_data = load_existing_version(strategy_name)
    
    # Check if configuration changed
    if existing_data:
        # Get the latest version's component hash
        latest_version = existing_data['versions'][-1] if existing_data.get('versions') else None
        existing_hash = latest_version.get('component_hash', '') if latest_version else ''
        
        if current_hash == existing_hash:
            print("✓ No configuration changes detected, skipping version update")
            return False
        
        print("⚠ Configuration changed detected!")
        print(f"  Old hash: {existing_hash[:16]}...")
        print(f"  New hash: {current_hash[:16]}...")
    else:
        print("⚠ No existing version found, will create initial version")
    
    # Create registry and load existing data if available
    registry = StrategyRegistry()
    
    if existing_data:
        # Deserialize existing strategy
        strategy_id = registry.deserialize_strategy(existing_data)
        
        # Prompt for changes
        changes, is_major = prompt_for_changes()
        
        # Update strategy
        new_version = registry.update_strategy(
            strategy_id=strategy_id,
            strategy=strategy_instance.component_strategy,  # Get the component strategy
            changes=changes,
            is_major=is_major
        )
        
        print(f"✓ Updated {strategy_name} to version {new_version}")
    else:
        # Register new strategy
        strategy_id = registry.register_strategy(
            strategy=strategy_instance.component_strategy,
            metadata={
                'created_by': 'developer',
                'description': f'{class_name} trading strategy',
                'tags': ['ml'] if 'Ml' in class_name else [],
                'status': 'EXPERIMENTAL'
            }
        )
        print(f"✓ Registered {strategy_name} as version 1.0.0")
    
    # Save to JSON
    save_strategy_version(registry, strategy_id, strategy_name)
    
    # Stage the JSON file for commit
    import subprocess
    subprocess.run(['git', 'add', str(STRATEGIES_DIR / f"{strategy_name}.json")])
    
    return True


def main():
    """Main pre-commit hook logic."""
    print("Checking for modified strategy files...")
    
    modified_files = get_modified_strategy_files()
    
    if not modified_files:
        print("✓ No strategy files modified")
        return 0
    
    print(f"Found {len(modified_files)} modified strategy file(s):")
    for f in modified_files:
        print(f"  - {f}")
    
    updated_count = 0
    for strategy_file in modified_files:
        if process_strategy_file(strategy_file):
            updated_count += 1
    
    if updated_count > 0:
        print(f"\n{'='*60}")
        print(f"✓ Updated {updated_count} strategy version(s)")
        print(f"  Config files staged for commit in {STRATEGIES_DIR}/")
        print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

