from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

import movie_review_factory
from movie_review_factory import runtime_cache


DERIVED = {
    "XDG_CACHE_HOME": ("xdg",),
    "HF_HOME": ("huggingface",),
    "HF_HUB_CACHE": ("huggingface", "hub"),
    "TRANSFORMERS_CACHE": ("huggingface", "transformers"),
    "TORCH_HOME": ("torch",),
    "UV_CACHE_DIR": ("uv",),
    "PIP_CACHE_DIR": ("pip",),
    "PLAYWRIGHT_BROWSERS_PATH": ("playwright",),
    "MRF_WHISPER_CACHE": ("models", "whisper"),
    "MRF_EMBED_CACHE": ("models", "embeddings"),
}
CACHE_ENV_NAMES = (runtime_cache.CACHE_ROOT_ENV, *DERIVED)


@pytest.fixture(autouse=True)
def _restore_cache_environment():
    snapshot = {name: os.environ.get(name) for name in CACHE_ENV_NAMES}
    yield
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _clear(monkeypatch) -> None:
    for name in (runtime_cache.CACHE_ROOT_ENV, *DERIVED):
        monkeypatch.delenv(name, raising=False)


def test_cache_root_is_opt_in(tmp_path: Path, monkeypatch) -> None:
    _clear(monkeypatch)

    runtime_cache.configure_cache_environment()

    assert runtime_cache.CACHE_ROOT_ENV not in os.environ
    assert all(name not in os.environ for name in DERIVED)


def test_cache_root_populates_all_app_caches(tmp_path: Path, monkeypatch) -> None:
    _clear(monkeypatch)
    root = tmp_path / "cache"
    monkeypatch.setenv(runtime_cache.CACHE_ROOT_ENV, str(root))

    assert runtime_cache.configure_cache_environment() == root
    for name, parts in DERIVED.items():
        assert os.environ[name] == str(root.joinpath(*parts))


def test_specific_cache_override_wins(tmp_path: Path, monkeypatch) -> None:
    _clear(monkeypatch)
    root = tmp_path / "cache"
    explicit = tmp_path / "explicit-hf"
    monkeypatch.setenv(runtime_cache.CACHE_ROOT_ENV, str(root))
    monkeypatch.setenv("HF_HOME", str(explicit))

    runtime_cache.configure_cache_environment()

    assert os.environ["HF_HOME"] == str(explicit)
    assert os.environ["UV_CACHE_DIR"] == str(root / "uv")


def test_package_import_configures_cache_before_submodules(tmp_path: Path, monkeypatch) -> None:
    _clear(monkeypatch)
    root = tmp_path / "cache"
    monkeypatch.setenv(runtime_cache.CACHE_ROOT_ENV, str(root))

    importlib.reload(movie_review_factory)

    assert os.environ["HF_HOME"] == str(root / "huggingface")
    assert os.environ["MRF_WHISPER_CACHE"] == str(root / "models" / "whisper")
    assert os.environ["MRF_EMBED_CACHE"] == str(root / "models" / "embeddings")
