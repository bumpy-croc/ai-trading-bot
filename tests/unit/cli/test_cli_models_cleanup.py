"""Tests for `atb models cleanup`."""

import argparse
from unittest.mock import patch

import pytest

from cli.commands.models import _handle_cleanup, register

pytestmark = pytest.mark.fast


@pytest.fixture
def registry(tmp_path):
    base = tmp_path / "models"
    type_dir = base / "BTCUSDT" / "basic"
    for version in ("2025-01-01_10h_v1", "2025-02-01_10h_v1", "2025-03-01_10h_v1"):
        (type_dir / version).mkdir(parents=True)
        (type_dir / version / "model.onnx").write_bytes(b"x" * 2048)
    (type_dir / "latest").symlink_to("2025-03-01_10h_v1")
    return base, type_dir


def make_args(base, **overrides):
    values = {
        "registry": str(base),
        "keep": 1,
        "reference_path": None,
        "verbose": False,
        "apply": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def run_cleanup(base, **overrides):
    # No default reference paths: the repo's own docs must not influence the temp registry.
    with patch("cli.commands.models.DEFAULT_REFERENCE_PATHS", ()):
        return _handle_cleanup(make_args(base, **overrides))


def test_dry_run_is_the_default_and_deletes_nothing(registry, capsys):
    base, type_dir = registry

    result = run_cleanup(base)

    out = capsys.readouterr().out
    assert result == 0
    assert "Would remove 2 version(s)" in out
    assert "BTCUSDT/basic/2025-01-01_10h_v1" in out
    assert "Space that would be freed: 4.0 KB" in out
    assert "Dry run" in out
    assert len(list(type_dir.iterdir())) == 4


def test_apply_deletes_candidates_and_keeps_latest(registry):
    base, type_dir = registry

    result = run_cleanup(base, apply=True)

    assert result == 0
    assert sorted(p.name for p in type_dir.iterdir()) == ["2025-03-01_10h_v1", "latest"]
    assert (type_dir / "latest").resolve().name == "2025-03-01_10h_v1"


def test_extra_reference_path_protects_a_version(registry, tmp_path):
    base, type_dir = registry
    notes = tmp_path / "notes.md"
    notes.write_text("Pinned 2025-01-01_10h_v1 for the exam.")

    result = run_cleanup(base, apply=True, reference_path=[str(notes)])

    assert result == 0
    assert sorted(p.name for p in type_dir.iterdir()) == [
        "2025-01-01_10h_v1",
        "2025-03-01_10h_v1",
        "latest",
    ]


def test_dangling_latest_aborts_without_deleting(registry, capsys):
    base, type_dir = registry
    (type_dir / "latest").unlink()
    (type_dir / "latest").symlink_to("missing")

    result = run_cleanup(base, apply=True)

    assert result == 1
    assert "Dangling" in capsys.readouterr().out
    assert len(list(type_dir.iterdir())) == 4


def test_negative_keep_is_reported_as_an_error(registry, capsys):
    base, _ = registry

    result = run_cleanup(base, keep=-1)

    assert result == 1
    assert "keep_newest" in capsys.readouterr().out


def test_apply_flag_is_opt_in():
    parser = argparse.ArgumentParser()
    register(parser.add_subparsers())

    assert parser.parse_args(["models", "cleanup"]).apply is False
    assert parser.parse_args(["models", "cleanup", "--apply"]).apply is True
