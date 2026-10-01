"""Pinned, checksum-verified ProPainter installer for the local watermark stage."""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable

PROPAINTER_COMMIT = "9243b7abf14098675da057114c63d9e502d4741d"
PROPAINTER_SOURCE_URL = f"https://github.com/sczhou/ProPainter/archive/{PROPAINTER_COMMIT}.zip"
PROPAINTER_SOURCE_SHA256 = "1f695a8265d3d85c93885f4b876e353050c4eb4e3a436a9fea1f4592924cae96"
PROPAINTER_ENV = "MRF_PROPAINTER_DIR"
PROPAINTER_PYTHON_ENV = "MRF_PROPAINTER_PYTHON"

MODEL_FILES = {
    "ProPainter.pth": (
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth",
        "12c070c4b48f374c91d8a2a17851140b85c159621080989f9e191bbc18bd6591",
    ),
    "recurrent_flow_completion.pth": (
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth",
        "22939a1a7900da878dbe1ccd011d646b1bfb30b8290039d8ff0e0c2fefbfd283",
    ),
    "raft-things.pth": (
        "https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth",
        "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1",
    ),
}

_RUNTIME_PACKAGES = (
    "av==12.3.0",
    "addict==2.4.0",
    "einops==0.8.1",
    "future==1.0.0",
    "numpy==1.26.4",
    "scipy==1.11.4",
    "opencv-python==4.10.0.84",
    "matplotlib==3.9.4",
    "scikit-image==0.24.0",
    "imageio-ffmpeg==0.5.1",
    "pyyaml==6.0.2",
    "requests==2.32.5",
    "timm==1.0.20",
    "yapf==0.43.0",
)


def install_root() -> Path:
    configured = (os.environ.get(PROPAINTER_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    cache_root = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(cache_root).expanduser() if cache_root else Path.home() / ".cache"
    return base / "movie-review-factory" / "propainter" / PROPAINTER_COMMIT


def env_python(home: Path | None = None) -> Path:
    configured = (os.environ.get(PROPAINTER_PYTHON_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    root = install_root() if home is None else Path(home)
    return root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, target: Path, expected_sha256: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and _sha256(target) == expected_sha256:
        return
    partial = target.with_suffix(target.suffix + ".part")
    partial.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "movie-review-factory"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        actual = _sha256(partial)
        if actual != expected_sha256:
            raise RuntimeError(f"Checksum mismatch for {target.name}: {actual}")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def _extract_source(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        files = bundle.infolist()
        roots = {Path(item.filename).parts[0] for item in files if Path(item.filename).parts}
        if len(roots) != 1:
            raise RuntimeError("Unexpected ProPainter archive layout")
        root_name = roots.pop()
        for item in files:
            relative = Path(*Path(item.filename).parts[1:])
            if not relative.parts:
                continue
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents:
                raise RuntimeError("Unsafe path in ProPainter archive")
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if root_name != f"ProPainter-{PROPAINTER_COMMIT}":
            raise RuntimeError("ProPainter archive did not match the pinned commit")


def _run(args: list[str], *, timeout: int = 3600) -> None:
    try:
        completed = subprocess.run(args, capture_output=True, text=True, shell=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {args[0]}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "command failed").strip()[-2000:]
        raise RuntimeError(detail)


def _create_environment(home: Path) -> Path:
    python = env_python(home)
    if python.is_file():
        return python
    uv = shutil.which("uv")
    python.parent.parent.mkdir(parents=True, exist_ok=True)
    if uv:
        _run([uv, "venv", str(python.parent.parent), "--python", "3.11", "--no-project"])
    else:
        launcher = shutil.which("py")
        if not launcher:
            raise RuntimeError("Install uv or Python 3.11 to create the ProPainter environment")
        _run([launcher, "-3.11", "-m", "venv", str(python.parent.parent)])
    if not python.is_file():
        raise RuntimeError("Could not create the ProPainter Python 3.11 environment")
    return python


def _install_runtime(python: Path) -> None:
    uv = shutil.which("uv")
    prefix = [uv, "pip", "install", "--python", str(python)] if uv else [str(python), "-m", "pip", "install"]
    _run(prefix + ["torch==2.1.2", "torchvision==0.16.2", "--index-url", "https://download.pytorch.org/whl/cpu"], timeout=3600)
    _run(prefix + list(_RUNTIME_PACKAGES), timeout=3600)


def is_ready(home: Path | None = None) -> bool:
    root = install_root() if home is None else Path(home)
    python = env_python(root)
    if not (root / "inference_propainter.py").is_file() or not python.is_file():
        return False
    if any(_sha256(root / "weights" / name) != checksum for name, (_, checksum) in MODEL_FILES.items() if (root / "weights" / name).is_file()):
        return False
    if any(not (root / "weights" / name).is_file() for name in MODEL_FILES):
        return False
    marker = root / ".mrf-propainter"
    return marker.is_file() and marker.read_text(encoding="utf-8-sig").strip() == PROPAINTER_COMMIT


def install(*, downloader: Callable[[str, Path, str], None] = _download, install_runtime: bool = True) -> Path:
    target = install_root()
    if is_ready(target):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mrf-propainter-", dir=target.parent) as temp_dir:
        temp = Path(temp_dir)
        source_archive = temp / "source.zip"
        downloader(PROPAINTER_SOURCE_URL, source_archive, PROPAINTER_SOURCE_SHA256)
        source = temp / "source"
        source.mkdir()
        _extract_source(source_archive, source)
        weights = source / "weights"
        for name, (url, checksum) in MODEL_FILES.items():
            downloader(url, weights / name, checksum)
        python = _create_environment(source)
        if install_runtime:
            _install_runtime(python)
        _run([str(python), "-c", "import cv2, scipy, torch, torchvision; print(torch.__version__)"])
        (source / ".mrf-propainter").write_text(PROPAINTER_COMMIT, encoding="utf-8")
        if target.exists():
            shutil.rmtree(target)
        source.replace(target)
    os.environ[PROPAINTER_ENV] = str(target)
    os.environ[PROPAINTER_PYTHON_ENV] = str(env_python(target))
    return target
