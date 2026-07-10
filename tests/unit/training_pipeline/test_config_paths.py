from pathlib import Path

import src.ml.training_pipeline.config as cfg


def test_training_paths_default_points_to_model_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "get_project_root", lambda: Path(tmp_path))
    # models_dir is resolved by get_model_registry_root() (shared with
    # promote_model_version and PredictionConfig.model_registry_path, PR
    # #950 review item 5) -- also monkeypatch its own get_project_root()
    # call so the default (no MODEL_REGISTRY_PATH override) case matches.
    monkeypatch.setattr("src.infrastructure.runtime.paths.get_project_root", lambda: Path(tmp_path))
    monkeypatch.delenv("MODEL_REGISTRY_PATH", raising=False)

    paths = cfg.TrainingPaths.default()

    assert paths.models_dir == Path(tmp_path) / "src" / "ml" / "models"
    assert paths.data_dir == Path(tmp_path) / "data"


def test_training_paths_default_honors_model_registry_path_override(tmp_path, monkeypatch):
    """PR #950 review item 5: MODEL_REGISTRY_PATH (the same key
    PredictionConfig.model_registry_path reads) must redirect models_dir --
    the lever acceptance tests use to keep trained artifacts out of the
    real src/ml/models/ registry."""
    monkeypatch.setattr(cfg, "get_project_root", lambda: Path(tmp_path))
    override_dir = tmp_path / "isolated-registry"
    monkeypatch.setenv("MODEL_REGISTRY_PATH", str(override_dir))

    paths = cfg.TrainingPaths.default()

    assert paths.models_dir == override_dir
    assert override_dir.is_dir()
