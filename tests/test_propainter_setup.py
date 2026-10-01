from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from movie_review_factory import propainter_setup as setup


def test_download_rejects_checksum_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size: int = -1) -> bytes:
            if hasattr(self, "done"):
                return b""
            self.done = True
            return b"tampered"

    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        setup._download("https://example.invalid/file", tmp_path / "file", "0" * 64)


def test_extract_source_rejects_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(f"ProPainter-{setup.PROPAINTER_COMMIT}/../../escape.txt", "bad")
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(RuntimeError, match="Unsafe path"):
        setup._extract_source(archive, destination)


def test_is_ready_checks_marker_models_and_python(tmp_path: Path) -> None:
    (tmp_path / "inference_propainter.py").write_text("# stub", encoding="utf-8")
    python = setup.env_python(tmp_path)
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    weights = tmp_path / "weights"
    weights.mkdir()
    original = setup.MODEL_FILES
    files = {}
    for name in original:
        data = name.encode()
        (weights / name).write_bytes(data)
        files[name] = ("https://example.invalid", hashlib.sha256(data).hexdigest())
    (tmp_path / ".mrf-propainter").write_text(setup.PROPAINTER_COMMIT, encoding="utf-8")
    try:
        setup.MODEL_FILES = files
        assert setup.is_ready(tmp_path)
    finally:
        setup.MODEL_FILES = original
