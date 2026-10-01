from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "release_gate.py"


def _load():
    spec = importlib.util.spec_from_file_location("release_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


gate = _load()


def test_artifact_mode_never_falls_back_to_source(tmp_path):
    args = gate.parse_args(["--mode", "artifact", "--artifact", str(tmp_path)])
    assert gate.package_candidates(args) == []


def test_artifact_mode_selects_only_designated_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    package = runtime / "movie_review_factory"
    package.mkdir(parents=True)
    (package / "pipeline.py").write_text("", encoding="utf-8")
    args = gate.parse_args(["--mode", "artifact", "--artifact", str(runtime)])
    assert gate.package_candidates(args) == [runtime.resolve()]


def test_source_mode_is_explicitly_compatible():
    args = gate.parse_args(["--mode", "source"])
    assert args.mode == "source"
    assert gate.package_candidates(args) == [gate.REPO_ROOT / "src"]
