"""Optional full-frame watermark removal for source footage.

Three methods share one mask format (see docs/watermark-removal.md):

- ``propainter``: AI video inpainting, cleanest but needs a GPU to be practical.
- ``delogo``: FFmpeg ``delogo`` over the mask bounding box, fast on any CPU.
- ``blur``: FFmpeg blur composited through the mask shape, fastest; hides
  rather than removes.

The FFmpeg methods need a static mask, so a per-frame mask folder is merged
into the pixels marked in enough frames first (:func:`static_mask`).

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

import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageChops, ImageDraw

from . import propainter_setup

INFERENCE_SCRIPT = "inference_propainter.py"
_OUTPUT_VIDEO_NAME = "inpaint_out.mp4"
# ProPainter accepts a single mask image applied to every frame, OR a folder of
# per-frame masks (matched to frames in sorted filename order) for a watermark
# that moves or changes over time.
_MASK_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp")

_FALSE_TOKENS = {"", "0", "false", "no", "off"}


class WatermarkToolUnavailable(RuntimeError):
    """The removal tool is not installed. The stage maps this to a skip."""


class WatermarkRemovalError(RuntimeError):
    """The removal tool ran but failed. The stage maps this to a hard failure."""


class ProPainterUnavailable(WatermarkToolUnavailable):
    """ProPainter is not installed/configured. The stage maps this to a skip."""


class ProPainterError(WatermarkRemovalError):
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
    chunk_seconds: float = 10.0
    chunk_max_height: int = 480
    extra_args: Sequence[str] = ()


def resolve_config(env: dict[str, str] | None = None) -> ProPainterConfig:
    """Build a :class:`ProPainterConfig` from the environment.

    Raises :class:`ProPainterUnavailable` (never a hard error) when ProPainter
    is not installed/configured, so the caller can turn it into a stage skip.
    """
    env = os.environ if env is None else env

    home_raw = (env.get("MRF_PROPAINTER_DIR") or "").strip()
    home = Path(home_raw).expanduser() if home_raw else propainter_setup.install_root()
    if not home_raw and not propainter_setup.is_ready(home):
        raise ProPainterUnavailable(
            "ProPainter is not installed - use the automatic installer to enable full-frame watermark removal"
        )
    if not (home / INFERENCE_SCRIPT).is_file():
        raise ProPainterUnavailable(f"{INFERENCE_SCRIPT} not found under {home}")

    default_python = str(propainter_setup.env_python(home)) if propainter_setup.is_ready(home) else sys.executable
    python_bin = (env.get("MRF_PROPAINTER_PYTHON") or default_python).strip()
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
        chunk_seconds=_num("MRF_PROPAINTER_CHUNK_SECONDS", 10.0, float),
        chunk_max_height=_num("MRF_PROPAINTER_CHUNK_MAX_HEIGHT", 480, int),
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


def _run_checked(command: list[str], *, cwd: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(command, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as exc:
        raise ProPainterUnavailable(f"cannot launch {command[0]}: {exc}") from exc
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        raise ProPainterError(f"command exited {proc.returncode}: {tail}")
    return proc


def _probe_duration(source: Path) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise ProPainterUnavailable("ffprobe not found - install FFmpeg to run chunked ProPainter")
    proc = _run_checked([ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(source)])
    try:
        duration = float(proc.stdout.strip())
    except ValueError as exc:
        raise ProPainterError(f"could not read video duration: {proc.stdout!r}") from exc
    if duration <= 0:
        raise ProPainterError("source video has no positive duration")
    return duration


def _extract_chunk(source: Path, target: Path, start: float, duration: float, max_height: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProPainterUnavailable("ffmpeg not found - install FFmpeg to run chunked ProPainter")
    scale = f"scale=-2:'min({max_height},ih)'" if max_height > 0 else "null"
    _run_checked([ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source), "-an", "-vf", scale, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", str(target)])


def _concat_chunks(chunks: list[Path], source: Path, output: Path, work_dir: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ProPainterUnavailable("ffmpeg not found - install FFmpeg to join ProPainter chunks")
    concat_file = work_dir / "concat.txt"
    concat_file.write_text("".join(f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in chunks), encoding="utf-8")
    video = work_dir / "video.mp4"
    _run_checked([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(video)])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    _run_checked([ffmpeg, "-y", "-i", str(video), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output)])


def _remove_watermark_chunked(source: Path, mask: Path, output: Path, cfg: ProPainterConfig, work_dir: Path, on_log: Callable[[str], None] | None, timeout: float | None) -> Path:
    duration = _probe_duration(source)
    chunk_seconds = max(1.0, cfg.chunk_seconds)
    chunks_dir = work_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    cleaned: list[Path] = []
    start = 0.0
    index = 0
    while start < duration - 0.01:
        length = min(chunk_seconds, duration - start)
        chunk = chunks_dir / f"chunk-{index:05d}.mp4"
        _extract_chunk(source, chunk, start, length, cfg.chunk_max_height)
        chunk_work = chunks_dir / f"work-{index:05d}"
        proc = _run_checked(build_command(cfg, chunk, mask, chunk_work), cwd=cfg.home, timeout=timeout)
        if on_log and proc.stdout:
            on_log(proc.stdout)
        produced = _find_output(chunk_work)
        if produced is None:
            raise ProPainterError(f"ProPainter produced no output for chunk {index}")
        cleaned_chunk = chunks_dir / f"clean-{index:05d}.mp4"
        shutil.move(str(produced), str(cleaned_chunk))
        cleaned.append(cleaned_chunk)
        shutil.rmtree(chunk_work, ignore_errors=True)
        chunk.unlink(missing_ok=True)
        start += length
        index += 1
    _concat_chunks(cleaned, source, output, work_dir)
    shutil.rmtree(work_dir, ignore_errors=True)
    return output


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
    source, mask, output = Path(source).resolve(), Path(mask).resolve(), Path(output).resolve()
    if not source.is_file():
        raise ProPainterError(f"source video not found: {source}")
    mask = _resolve_mask(mask)  # single image file, or a folder of per-frame masks

    work_dir = Path(work_dir) if work_dir else output.parent / f"{output.stem}.propainter"
    work_dir.mkdir(parents=True, exist_ok=True)
    if cfg.chunk_seconds > 0:
        return _remove_watermark_chunked(source, mask, output, cfg, work_dir, on_log, timeout)

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


# --- FFmpeg methods (delogo / blur) ----------------------------------------

FFMPEG_METHODS = ("delogo", "blur")
_BLUR_FILTER = "boxblur=20:2"
# Per-frame detectors flag stray pixels whose union can cover most of the frame; keep pixels marked in >= 5% of frames.
STATIC_MASK_MIN_FRACTION = 0.05
# Frame counters saturate at 255, so the needed count must stay below that.
_STATIC_MASK_MAX_FRAMES = 5000


def _binary(image: Image.Image) -> Image.Image:
    return image.convert("L").point(lambda value: 255 if value > 127 else 0)


def static_mask(mask: Path, out_path: Path, min_fraction: float = STATIC_MASK_MIN_FRACTION) -> Path:
    """Return a single mask image for the FFmpeg methods.

    A per-frame folder becomes the pixels marked in at least ``min_fraction``
    of the frames (the persistent watermark), falling back to the plain union
    when nothing is that persistent.
    """
    mask = _resolve_mask(mask)
    if mask.is_file():
        return mask
    paths = frame_mask_paths(mask)
    paths = paths[:: max(1, -(-len(paths) // _STATIC_MASK_MAX_FRAMES))]
    needed = max(1, math.ceil(min_fraction * len(paths)))
    union: Image.Image | None = None
    counts: Image.Image | None = None
    for path in paths:
        with Image.open(path) as frame:
            current = _binary(frame)
        if union is not None and current.size != union.size:
            current = current.resize(union.size, Image.NEAREST)
        ones = current.point(lambda value: 1 if value else 0)
        union = current if union is None else ImageChops.lighter(union, current)
        counts = ones if counts is None else ImageChops.add(counts, ones)
    persistent = counts.point(lambda value: 255 if value >= needed else 0)
    return _save_mask(persistent if persistent.getbbox() else union, out_path)


def mask_stats(mask: Path) -> tuple[float, tuple[int, int, int, int] | None, tuple[int, int]]:
    """Return ``(coverage 0..1, bbox (x0, y0, x1, y1) or None, mask size)``."""
    with Image.open(mask) as image:
        binary = _binary(image)
    width, height = binary.size
    marked = binary.histogram()[255]
    return marked / float(width * height), binary.getbbox(), (width, height)


def _delogo_rect(bbox: tuple[int, int, int, int], mask_size: tuple[int, int], frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Scale the mask bbox to the video and keep it strictly inside the frame."""
    mask_w, mask_h = mask_size
    frame_w, frame_h = frame_size
    sx, sy = frame_w / mask_w, frame_h / mask_h
    x0, y0 = max(1, int(bbox[0] * sx)), max(1, int(bbox[1] * sy))
    x1, y1 = min(frame_w - 2, int(round(bbox[2] * sx))), min(frame_h - 2, int(round(bbox[3] * sy)))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise WatermarkRemovalError("watermark mask is too small or touches only the frame border")
    return x0, y0, x1 - x0, y1 - y0


def build_ffmpeg_command(
    ffmpeg: str,
    source: Path,
    mask: Path,
    output: Path,
    method: str,
    frame_size: tuple[int, int],
) -> list[str]:
    """Assemble the FFmpeg command for ``delogo`` or ``blur``."""
    if method not in FFMPEG_METHODS:
        raise ValueError(f"unsupported FFmpeg watermark method: {method}")
    _, bbox, mask_size = mask_stats(mask)
    if bbox is None:
        raise WatermarkRemovalError(f"watermark mask is empty: {mask}")
    encode = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "copy"]
    if method == "delogo":
        x, y, w, h = _delogo_rect(bbox, mask_size, frame_size)
        return [ffmpeg, "-y", "-i", str(source), "-vf", f"delogo=x={x}:y={y}:w={w}:h={h}",
                "-map", "0:v:0", "-map", "0:a?", *encode, str(output)]
    width, height = frame_size
    graph = (
        f"[1:v]format=gray,scale={width}:{height}[m];"
        f"[0:v]split[base][soft];[soft]{_BLUR_FILTER}[blurred];"
        "[blurred][m]alphamerge[patch];[base][patch]overlay=shortest=1[v]"
    )
    return [ffmpeg, "-y", "-i", str(source), "-loop", "1", "-i", str(mask), "-filter_complex", graph,
            "-map", "[v]", "-map", "0:a?", *encode, str(output)]


def remove_watermark_ffmpeg(
    source: Path,
    mask: Path,
    output: Path,
    method: str,
    frame_size: tuple[int, int],
    *,
    timeout: float | None = None,
) -> Path:
    """Remove (delogo) or hide (blur) the watermark with FFmpeg, CPU only.

    ``mask`` must be a single image (use :func:`static_mask` for a folder).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise WatermarkToolUnavailable("ffmpeg not found - install FFmpeg to use the delogo/blur methods")
    source, mask, output = Path(source).resolve(), Path(mask).resolve(), Path(output).resolve()
    if not source.is_file():
        raise WatermarkRemovalError(f"source video not found: {source}")
    if not mask.is_file():
        raise WatermarkRemovalError(f"watermark mask image not found: {mask}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.tmp{output.suffix}")
    command = build_ffmpeg_command(ffmpeg, source, mask, temporary, method, frame_size)
    try:
        proc = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except FileNotFoundError as exc:
        raise WatermarkToolUnavailable(f"cannot launch ffmpeg: {exc}") from exc
    if proc.returncode != 0 or not temporary.is_file():
        temporary.unlink(missing_ok=True)
        tail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        raise WatermarkRemovalError(f"ffmpeg {method} exited {proc.returncode}: {tail}")
    temporary.replace(output)
    return output
