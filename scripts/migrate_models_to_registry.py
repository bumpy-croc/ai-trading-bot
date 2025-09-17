"""
Migrate legacy flat ONNX models under src/ml/ into the structured registry:

  src/ml/models/{SYMBOL}/{model_type}/{version_id}/

Usage (via CLI runner):
  atb scripts run migrate_models_to_registry -- --timeframe 1h

Notes:
- Detects `model_type` by filename: contains 'sentiment' -> sentiment, else basic.
- `version_id` defaults to {YYYY-MM-DD}_{timeframe}_v1 unless overridden.
- Creates metadata.json with symbol/model_type/timeframe/version.
- Creates/updates `latest` symlink for the symbol/model_type.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from src.prediction.config import PredictionConfig

logger = logging.getLogger("migrate_models_to_registry")


def infer_symbol_and_type(stem: str) -> tuple[str, str]:
    s = stem.lower()
    # Symbol inference: take leading word until underscore
    parts = s.split("_")
    symbol = parts[0].upper()
    model_type = "sentiment" if "sentiment" in s else "basic"
    return symbol, model_type


def migrate(timeframe: str, version_id: str | None = None) -> int:
    cfg = PredictionConfig.from_config_manager()
    legacy_base = Path("src/ml")
    target_base = Path(cfg.model_registry_path)
    target_base.mkdir(parents=True, exist_ok=True)

    moved = 0
    for onnx in legacy_base.glob("*.onnx"):
        stem = onnx.stem
        symbol, model_type = infer_symbol_and_type(stem)
        ver = version_id or f"{datetime.utcnow().date()}_{timeframe}_v1"
        dest_dir = target_base / symbol / model_type / ver
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Move/copy model file
        dest_model = dest_dir / "model.onnx"
        if not dest_model.exists():
            dest_model.write_bytes(onnx.read_bytes())

        # Write metadata if missing
        metadata_path = dest_dir / "metadata.json"
        if not metadata_path.exists():
            md = {
                "model_id": stem,
                "symbol": symbol,
                "timeframe": timeframe,
                "model_type": model_type,
                "version_id": ver,
                "framework": "onnx",
                "model_file": "model.onnx",
                "created_at": datetime.utcnow().isoformat(timespec="seconds"),
                "stage": "candidate",
            }
            metadata_path.write_text(json.dumps(md, indent=2))

        # Promote latest symlink
        latest = dest_dir.parent / "latest"
        if latest.exists() or latest.is_symlink():
            try:
                if latest.is_symlink():
                    latest.unlink()
                else:
                    # refuse to remove non-symlink directory named latest
                    logger.warning("Refusing to overwrite non-symlink 'latest' at %s", latest)
            except Exception:
                pass
        if not latest.exists():
            relative_target = dest_dir.relative_to(dest_dir.parent)
            latest.symlink_to(relative_target, target_is_directory=True)

        moved += 1

    logger.info("Migrated %d models", moved)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--version-id", default=None)
    ns = p.parse_args(argv)
    return migrate(timeframe=ns.timeframe, version_id=ns.version_id)


if __name__ == "__main__":
    raise SystemExit(main())


