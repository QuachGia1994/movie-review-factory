from __future__ import annotations

import os
from pathlib import Path

CACHE_ROOT_ENV = "MRF_CACHE_ROOT"


def cache_root() -> Path:
    configured = (os.environ.get(CACHE_ROOT_ENV) or "").strip()
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt":
        local = (os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or "").strip()
        if local:
            return Path(local) / "MovieReviewFactory" / "cache"
    xdg = (os.environ.get("XDG_CACHE_HOME") or "").strip()
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "movie-review-factory"


def cache_environment(root: Path | None = None) -> dict[str, str]:
    base = cache_root() if root is None else Path(root)
    return {
        CACHE_ROOT_ENV: str(base),
        "XDG_CACHE_HOME": str(base / "xdg"),
        "HF_HOME": str(base / "huggingface"),
        "HF_HUB_CACHE": str(base / "huggingface" / "hub"),
        "TRANSFORMERS_CACHE": str(base / "huggingface" / "transformers"),
        "TORCH_HOME": str(base / "torch"),
        "UV_CACHE_DIR": str(base / "uv"),
        "PIP_CACHE_DIR": str(base / "pip"),
        "PLAYWRIGHT_BROWSERS_PATH": str(base / "playwright"),
        "MRF_WHISPER_CACHE": str(base / "models" / "whisper"),
        "MRF_EMBED_CACHE": str(base / "models" / "embeddings"),
    }


def configure_cache_environment(root: Path | None = None) -> Path:
    configured = (os.environ.get(CACHE_ROOT_ENV) or "").strip()
    target = Path(root) if root is not None else (Path(configured).expanduser() if configured else None)
    if target is None:
        return cache_root()
    values = cache_environment(target)
    for name, value in values.items():
        os.environ.setdefault(name, value)
    return target
