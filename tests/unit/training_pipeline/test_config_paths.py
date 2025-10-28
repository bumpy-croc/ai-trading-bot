from pathlib import Path

import src.ml.training_pipeline.config as cfg


def test_training_paths_default_points_to_model_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "get_project_root", lambda: Path(tmp_path))
    paths = cfg.TrainingPaths.default()
    assert paths.models_dir == Path(tmp_path) / "src" / "ml" / "models"
    assert paths.data_dir == Path(tmp_path) / "data"
