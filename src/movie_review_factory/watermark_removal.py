"""Optional full-frame watermark removal for source footage via ProPainter.

The base pipeline only *neutralises* corner/edge watermarks by masking two
bands (``JobConfig.brand_top_band`` / ``brand_bottom_band``). That cannot
recover a watermark that is alpha-blended across the whole frame ("đánh chìm"):

    observed = alpha * watermark + (1 - alpha) * original

For that case we reconstruct the frame instead of hiding it. This module wraps
the external ProPainter video-inpainting model (https://github.com/sczhou/
ProPainter): given a source video and a mask of the watermark region (either a
single image applied to every frame, or a folder of per-frame masks for a
watermark that moves over time), ProPainter propagates background pixels across
neighbouring frames to rebuild a clean video without flicker.

ProPainter is heavy (PyTorch + CUDA + downloaded weights) and is intentionally
*not* a Python dependency of this package. It is invoked as an external
subprocess and located at runtime through environment variables, so a machine
without it simply skips the optional ``watermark`` pipeline stage instead of
failing the whole job.

Environment
-----------
- ``MRF_PROPAINTER_DIR``    : clone dir containing ``inference_propainter.py`` (required).
- ``MRF_PROPAINTER_PYTHON`` : python interpreter for ProPainter's venv (default: current).
- ``MRF_PROPAINTER_DEVICE`` : ``cuda`` (default) or ``cpu`` (disables ``--fp16``).
- ``MRF_PROPAINTER_FP16``   : ``1``/``0`` toggle for half precision (default on).
- ``MRF_PROPAINTER_MASK_DILATION`` / ``_RESIZE_RATIO`` / ``_SUBVIDEO_LENGTH`` /
  ``_NEIGHBOR_LENGTH`` / ``_REF_STRIDE`` : numeric tuning knobs.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw

INFERENCE_SCRIPT = "inference_propainter.py"
_OUTPUT_VIDEO_NAME = "inpaint_out.mp4"
# ProPainter accepts a single mask image applied to every frame, OR a folder of
# per-frame masks (matched to frames in sorted filename order) for a watermark
# that moves or changes over time.
_MASK_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

_FALSE_TOKENS = {"", "0", "false", "no", "off"}


class ProPainterUnavailable(RuntimeError):
    """ProPainter is not installed/configured. The stage maps this to a skip."""


class ProPainterError(RuntimeError):
    """ProPainter launched but failed. The stage maps this to a hard failure."""


@dataclass(frozen=True)
class ProPainterConfig:
    """Resolved location and tuning knobs for a ProPainter invocation."""

    home: Path
    python_bin: str = sys.executable
    device: str = "cuda"
    fp16: bool = True
    mask_dilation: int = 8
    resize_ratio: float = 1.0
    subvideo_length: int = 80
    neighbor_length: int = 10
    ref_stride: int = 10
    extra_args: Sequence[str] = ()


def resolve_config(env: dict[str, str] | None = None) -> ProPainterConfig:
    """Build a :class:`ProPainterConfig` from the environment.

    Raises :class:`ProPainterUnavailable` (never a hard error) when ProPainter
    is not installed/configured, so the caller can turn it into a stage skip.
    """
    env = os.environ if env is None else env

    home_raw = (env.get("MRF_PROPAINTER_DIR") or "").strip()
    if not home_raw:
        raise ProPainterUnavailable(
            "MRF_PROPAINTER_DIR not set - install ProPainter and point it here "
            "to enable full-frame watermark removal"
        )
    home = Path(home_raw).expanduser()
    if not (home / INFERENCE_SCRIPT).is_file():
        raise ProPainterUnavailable(f"{INFERENCE_SCRIPT} not found under {home}")

    python_bin = (env.get("MRF_PROPAINTER_PYTHON") or sys.executable).strip()
    if shutil.which(python_bin) is None and not Path(python_bin).exists():
        raise ProPainterUnavailable(f"python interpreter not found: {python_bin}")

    def _num(key: str, default, cast):
        raw = (env.get(key) or "").strip()
        if not raw:
            return default
        try:
            return cast(raw)
        except ValueError:
            return default

    fp16 = (env.get("MRF_PROPAINTER_FP16", "1") or "").strip().lower() not in _FALSE_TOKENS
    return ProPainterConfig(
        home=home,
        python_bin=python_bin,
        device=(env.get("MRF_PROPAINTER_DEVICE") or "cuda").strip() or "cuda",
        fp16=fp16,
        mask_dilation=_num("MRF_PROPAINTER_MASK_DILATION", 8, int),
        resize_ratio=_num("MRF_PROPAINTER_RESIZE_RATIO", 1.0, float),
        subvideo_length=_num("MRF_PROPAINTER_SUBVIDEO_LENGTH", 80, int),
        neighbor_length=_num("MRF_PROPAINTER_NEIGHBOR_LENGTH", 10, int),
        ref_stride=_num("MRF_PROPAINTER_REF_STRIDE", 10, int),
    )


# --- mask helpers -----------------------------------------------------------
# ProPainter expects a single-channel mask where white (255) = "reconstruct
# this pixel" and black (0) = "keep original". A single PNG applies to every
# frame, which suits a static watermark.

def _to_px(value: float, extent: int) -> int:
    """A value <= 1 is treated as a fraction of ``extent``; otherwise pixels."""
    return int(round(value * extent)) if 0 <= value <= 1 else int(round(value))


def _save_mask(image: Image.Image, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(out_path)
    return out_path


def rect_mask_from_bands(
    width: int,
    height: int,
    out_path: Path,
    *,
    top_fraction: float = 0.0,
    bottom_fraction: float = 0.0,
) -> Path:
    """Write a mask covering the top/bottom bands (mirrors ``brand_*_band``)."""
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    top_px = _to_px(top_fraction, height)
    bottom_px = _to_px(bottom_fraction, height)
    if top_px > 0:
        draw.rectangle([0, 0, width, top_px], fill=255)
    if bottom_px > 0:
        draw.rectangle([0, height - bottom_px, width, height], fill=255)
    return _save_mask(image, out_path)


def rect_mask_from_boxes(
    width: int,
    height: int,
    boxes: Iterable[Sequence[float]],
    out_path: Path,
) -> Path:
    """Write a mask from ``(x, y, w, h)`` boxes given as fractions or pixels."""
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    marked = False
    for box in boxes:
        x, y, w, h = box
        x0, y0 = _to_px(x, width), _to_px(y, height)
        x1, y1 = x0 + _to_px(w, width), y0 + _to_px(h, height)
        if x1 > x0 and y1 > y0:
            draw.rectangle([x0, y0, min(x1, width), min(y1, height)], fill=255)
            marked = True
    if not marked:
        raise ValueError("no valid watermark boxes to build a mask from")
    return _save_mask(image, out_path)


def frame_mask_paths(mask_dir: Path) -> list[Path]:
    """Return the per-frame mask images in ``mask_dir``, sorted by filename.

    ProPainter matches masks to video frames in sorted filename order, so the
    files should be zero-padded (``00000.png``, ``00001.png`` ...). One mask per
    frame is ideal; ProPainter reuses the last mask when there are fewer.
    """
    mask_dir = Path(mask_dir)
    return sorted(
        path for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in _MASK_IMAGE_SUFFIXES
    )


def _resolve_mask(mask: Path) -> Path:
    """Validate the mask is a single image file or a non-empty per-frame folder."""
    mask = Path(mask)
    if mask.is_dir():
        if not frame_mask_paths(mask):
            raise ProPainterError(
                f"watermark mask folder has no {'/'.join(_MASK_IMAGE_SUFFIXES)} images: {mask}"
            )
        return mask
    if mask.is_file():
        return mask
    raise ProPainterError(f"watermark mask not found: {mask}")


# --- ProPainter invocation --------------------------------------------------

def build_command(
    cfg: ProPainterConfig,
    source: Path,
    mask: Path,
    work_dir: Path,
) -> list[str]:
    """Assemble the ProPainter CLI command (paths are passed as absolute)."""
    command = [
        cfg.python_bin,
        INFERENCE_SCRIPT,
        "--video", str(source),
        "--mask", str(mask),
        "--output", str(work_dir),
        "--mask_dilation", str(cfg.mask_dilation),
        "--ref_stride", str(cfg.ref_stride),
        "--neighbor_length", str(cfg.neighbor_length),
        "--subvideo_length", str(cfg.subvideo_length),
    ]
    if abs(cfg.resize_ratio - 1.0) > 1e-6:
        command += ["--resize_ratio", str(cfg.resize_ratio)]
    if cfg.fp16 and cfg.device != "cpu":
        command.append("--fp16")
    command += list(cfg.extra_args)
    return command


def _find_output(work_dir: Path) -> Path | None:
    """Locate ProPainter's rendered video under its output directory."""
    named = list(Path(work_dir).rglob(_OUTPUT_VIDEO_NAME))
    if named:
        return max(named, key=lambda p: p.stat().st_mtime)
    any_mp4 = list(Path(work_dir).rglob("*.mp4"))
    return max(any_mp4, key=lambda p: p.stat().st_mtime) if any_mp4 else None


def remove_watermark(
    source: Path,
    mask: Path,
    output: Path,
    cfg: ProPainterConfig | None = None,
    *,
    work_dir: Path | None = None,
    on_log: Callable[[str], None] | None = None,
    timeout: float | None = None,
) -> Path:
    """Run ProPainter to reconstruct ``source`` behind ``mask`` into ``output``.

    Returns the path to the cleaned video. Raises :class:`ProPainterError` on a
    runtime failure and :class:`ProPainterUnavailable` if the model cannot be
    launched at all.
    """
    cfg = cfg or resolve_config()
    source, mask, output = Path(source), Path(mask), Path(output)
    if not source.is_file():
        raise ProPainterError(f"source video not found: {source}")
    mask = _resolve_mask(mask)  # single image file, or a folder of per-frame masks

    work_dir = Path(work_dir) if work_dir else output.parent / f"{output.stem}.propainter"
    work_dir.mkdir(parents=True, exist_ok=True)

    command = build_command(cfg, source, mask, work_dir)
    try:
        proc = subprocess.run(
            command,
            cwd=str(cfg.home),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:  # interpreter/script vanished mid-run
        raise ProPainterUnavailable(f"cannot launch ProPainter: {exc}") from exc

    if on_log and proc.stdout:
        on_log(proc.stdout)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-500:]
        raise ProPainterError(f"ProPainter exited {proc.returncode}: {tail}")

    produced = _find_output(work_dir)
    if produced is None:
        raise ProPainterError(f"ProPainter produced no {_OUTPUT_VIDEO_NAME} under {work_dir}")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    shutil.move(str(produced), str(output))
    shutil.rmtree(work_dir, ignore_errors=True)
    return output
