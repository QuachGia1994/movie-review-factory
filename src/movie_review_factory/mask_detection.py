"""Auto-generate a per-frame watermark mask folder from a video (no hand-drawing).

Feeds :mod:`watermark_removal`: instead of a single hand-drawn PNG, this builds
a folder of zero-padded masks (``00000.png``, ``00001.png`` ...) — one per video
frame — so a watermark that *moves or changes over time* is tracked. Masks are
single-channel PNGs where white (255) = "reconstruct this pixel".

Three strategies, matched to how much tooling you have:

- ``color``    : threshold pixels near a target colour (e.g. a white/grey
                 semi-transparent overlay), independently per frame. Pillow only.
- ``temporal`` : mark pixels whose luminance barely changes across frames (a
                 fixed overlay). One static mask, replicated per frame. Pillow only.
- ``external`` : run a user-supplied detector command (Florence-2 / SAM / …) that
                 writes the numbered masks itself, configured via
                 ``MRF_MASK_DETECTOR_CMD`` with ``{video}`` / ``{out}``
                 placeholders — so heavy ML dependencies are never bundled here.

``color`` / ``temporal`` extract frames with FFmpeg; a machine without FFmpeg (or
without the external detector) raises :class:`MaskDetectorUnavailable`, which the
``watermark`` stage turns into a clean skip.
"""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from PIL import Image, ImageChops, ImageFilter

FRAME_PATTERN = "%05d.png"  # FFmpeg image2 output pattern (zero-padded, from 0)


class MaskDetectorUnavailable(RuntimeError):
    """FFmpeg or an external detector is missing. The stage maps this to a skip."""


class MaskDetectorError(RuntimeError):
    """Detection ran but failed. The stage maps this to a hard failure."""


@dataclass(frozen=True)
class DetectSettings:
    """Tuning knobs for :func:`generate_frame_masks`."""

    method: str = "color"                              # "color" | "temporal" | "external"
    target_rgb: tuple[int, int, int] = (255, 255, 255)  # colour to match (color method)
    tolerance: int = 30                                # +/- per channel (color method)
    dilation: int = 4                                  # grow the mask by N px (0 = off)
    threshold: int = 12                                # max luminance range (temporal method)
    fps: float | None = None                           # None = every frame; else subsample
    external_cmd: str = ""                             # command template (external method)


# --- pure per-pixel detectors (Pillow only; unit-tested without FFmpeg) -----

def _dilate(mask: Image.Image, radius: int) -> Image.Image:
    if radius > 0:
        return mask.filter(ImageFilter.MaxFilter(2 * radius + 1))
    return mask


def color_mask(
    image: Image.Image,
    target_rgb: tuple[int, int, int] = (255, 255, 255),
    tolerance: int = 30,
    dilation: int = 0,
) -> Image.Image:
    """Return an 'L' mask: white where the pixel is within tolerance of target_rgb."""
    rgb = image.convert("RGB")
    combined: Image.Image | None = None
    for channel, target in zip(rgb.split(), target_rgb):
        near = channel.point(lambda v, t=target: 255 if abs(v - t) <= tolerance else 0)
        combined = near if combined is None else ImageChops.multiply(combined, near)
    mask = combined if combined is not None else Image.new("L", rgb.size, 0)
    return _dilate(mask, dilation)


def temporal_static_mask(
    frames: Iterable[Image.Image],
    threshold: int = 12,
    dilation: int = 0,
) -> Image.Image:
    """Mark pixels whose luminance range across frames is <= threshold (static overlay).

    Note: this also catches genuinely static *background* regions, so it works
    best when the watermark sits over moving content.
    """
    iterator = iter(frames)
    try:
        first = next(iterator).convert("L")
    except StopIteration as exc:
        raise MaskDetectorError("no frames to analyse") from exc
    darkest = first
    brightest = first.copy()
    for frame in iterator:
        luminance = frame.convert("L")
        darkest = ImageChops.darker(darkest, luminance)
        brightest = ImageChops.lighter(brightest, luminance)
    span = ImageChops.difference(brightest, darkest)
    mask = span.point(lambda v: 255 if v <= threshold else 0)
    return _dilate(mask, dilation)


# --- FFmpeg frame extraction + orchestration --------------------------------

def _mask_name(index: int) -> str:
    return f"{index:05d}.png"


def _write_mask(mask: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    mask.save(temporary, format="PNG")
    temporary.replace(path)
    return path


def _read_ppm_token(stream) -> bytes | None:
    token = bytearray()
    while True:
        byte = stream.read(1)
        if not byte:
            return bytes(token) if token else None
        if byte == b"#" and not token:
            while byte not in (b"", b"\n"):
                byte = stream.read(1)
            continue
        if byte.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(byte)


def _stream_video_frames(
    video: Path,
    *,
    fps: float | None = None,
    ffmpeg: str | None = None,
) -> Iterator[Image.Image]:
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise MaskDetectorUnavailable("ffmpeg not on PATH - install FFmpeg to detect watermark masks")
    command = [
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(video),
    ]
    if fps and fps > 0:
        command += ["-vf", f"fps={fps}"]
    command += ["-an", "-sn", "-f", "image2pipe", "-vcodec", "ppm", "-"]
    try:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise MaskDetectorUnavailable(f"cannot launch ffmpeg: {exc}") from exc
    assert proc.stdout is not None
    assert proc.stderr is not None
    emitted = 0
    try:
        while True:
            magic = _read_ppm_token(proc.stdout)
            if magic is None:
                break
            if magic != b"P6":
                raise MaskDetectorError(f"unexpected ffmpeg frame format: {magic!r}")
            width_raw = _read_ppm_token(proc.stdout)
            height_raw = _read_ppm_token(proc.stdout)
            max_value_raw = _read_ppm_token(proc.stdout)
            try:
                width = int(width_raw or b"")
                height = int(height_raw or b"")
                max_value = int(max_value_raw or b"")
            except ValueError as exc:
                raise MaskDetectorError("invalid PPM header from ffmpeg") from exc
            if width <= 0 or height <= 0 or max_value != 255:
                raise MaskDetectorError(
                    f"unsupported PPM frame from ffmpeg: {width}x{height}, max={max_value}"
                )
            expected = width * height * 3
            payload = proc.stdout.read(expected)
            if len(payload) != expected:
                raise MaskDetectorError(
                    f"truncated ffmpeg frame: expected {expected} bytes, got {len(payload)}"
                )
            emitted += 1
            yield Image.frombytes("RGB", (width, height), payload)
        returncode = proc.wait()
        if returncode != 0:
            tail = proc.stderr.read().decode("utf-8", errors="replace").strip()[-500:]
            raise MaskDetectorError(f"ffmpeg frame stream failed: {tail or f'exit {returncode}'}")
        if emitted == 0:
            raise MaskDetectorError("ffmpeg produced no frames")
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        try:
            proc.stderr.close()
        except Exception:
            pass


def extract_frames(
    video: Path,
    out_dir: Path,
    *,
    fps: float | None = None,
    ffmpeg: str | None = None,
) -> list[Path]:
    """Extract frames from ``video`` into ``out_dir`` as zero-padded PNGs.

    ``fps=None`` extracts every frame (needed to align per-frame masks with a
    moving watermark); a positive value subsamples.
    """
    ffmpeg = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg:
        raise MaskDetectorUnavailable("ffmpeg not on PATH - install FFmpeg to detect watermark masks")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(video)]
    if fps and fps > 0:
        command += ["-vf", f"fps={fps}"]
    command += ["-start_number", "0", str(out_dir / FRAME_PATTERN)]
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise MaskDetectorUnavailable(f"cannot launch ffmpeg: {exc}") from exc
    if proc.returncode != 0:
        for frame in out_dir.glob("*.png"):
            frame.unlink(missing_ok=True)
        raise MaskDetectorError(f"ffmpeg frame extraction failed: {(proc.stderr or '').strip()[-500:]}")
    frames = sorted(out_dir.glob("*.png"))
    if not frames:
        raise MaskDetectorError(f"ffmpeg produced no frames under {out_dir}")
    return frames


def resolve_external_command(env: dict[str, str] | None = None) -> str:
    env = os.environ if env is None else env
    template = (env.get("MRF_MASK_DETECTOR_CMD") or "").strip()
    if not template:
        raise MaskDetectorUnavailable(
            "MRF_MASK_DETECTOR_CMD not set - configure an external detector "
            "(e.g. a Florence-2/SAM script) to auto-detect watermark masks"
        )
    return template


def build_external_command(template: str, video: Path, out_dir: Path) -> list[str]:
    """Expand ``{video}`` / ``{out}`` placeholders and split into argv."""
    filled = template.replace("{video}", str(video)).replace("{out}", str(out_dir))
    return shlex.split(filled, posix=(os.name != "nt"))


def _run_external(video: Path, out_dir: Path, settings: DetectSettings) -> Path:
    template = settings.external_cmd or resolve_external_command()
    command = build_external_command(template, video, out_dir)
    try:
        proc = subprocess.run(command, capture_output=True, text=True)
    except OSError as exc:
        raise MaskDetectorUnavailable(f"cannot launch mask detector: {exc}") from exc
    if proc.returncode != 0:
        raise MaskDetectorError(f"mask detector exited {proc.returncode}: {(proc.stderr or '').strip()[-300:]}")
    if not sorted(out_dir.glob("*.png")):
        raise MaskDetectorError(f"mask detector wrote no PNG masks to {out_dir}")
    return out_dir


def generate_frame_masks(
    video: Path,
    out_dir: Path,
    settings: DetectSettings | None = None,
    *,
    work_dir: Path | None = None,
) -> Path:
    """Detect the watermark and write a per-frame mask folder; return ``out_dir``.

    ``MaskDetectorUnavailable`` signals a missing tool (skippable); other
    problems raise ``MaskDetectorError``.
    """
    settings = settings or DetectSettings()
    video, out_dir = Path(video), Path(out_dir)
    if not video.is_file():
        raise MaskDetectorError(f"source video not found: {video}")
    if settings.method not in ("color", "temporal", "external"):
        raise MaskDetectorError(f"unknown mask detection method: {settings.method!r}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        if settings.method == "external":
            return _run_external(video, out_dir, settings)
        if settings.method == "color":
            frames = _stream_video_frames(video, fps=settings.fps)
            try:
                for index, frame in enumerate(frames):
                    try:
                        mask = color_mask(frame, settings.target_rgb, settings.tolerance, settings.dilation)
                        _write_mask(mask, out_dir / _mask_name(index))
                    finally:
                        frame.close()
            finally:
                frames.close()
        else:
            frame_count = 0
            frames = _stream_video_frames(video, fps=settings.fps)

            def counted_frames() -> Iterator[Image.Image]:
                nonlocal frame_count
                for frame in frames:
                    frame_count += 1
                    try:
                        yield frame
                    finally:
                        frame.close()

            try:
                static = temporal_static_mask(counted_frames(), settings.threshold, settings.dilation)
            finally:
                frames.close()
            try:
                first_mask = _write_mask(static, out_dir / _mask_name(0))
                for index in range(1, frame_count):
                    shutil.copyfile(first_mask, out_dir / _mask_name(index))
            finally:
                static.close()
    except Exception:
        shutil.rmtree(out_dir, ignore_errors=True)
        raise

    if not sorted(out_dir.glob("*.png")):
        shutil.rmtree(out_dir, ignore_errors=True)
        raise MaskDetectorError("no masks were produced")
    return out_dir


# --- external detector self-test --------------------------------------------

@dataclass(frozen=True)
class DetectorProbe:
    """Result of running the external detector on a single frame."""

    ok: bool
    masks: int
    message: str
    command: list[str] = field(default_factory=list)


def probe_external_detector(
    video: Path,
    settings: DetectSettings | None = None,
    *,
    ffmpeg: str | None = None,
    timeout: float | None = 120,
    env: dict[str, str] | None = None,
) -> DetectorProbe:
    """Validate the external detector by running it on ONE extracted frame.

    Cheap pre-flight before a full run: it never raises for a misconfigured or
    failing detector — it returns a :class:`DetectorProbe` with ``ok=False`` and
    a human-readable message so a CLI/API can show it. Raises only for a missing
    source video (a usage error).
    """
    settings = settings or DetectSettings(method="external")
    env = os.environ if env is None else env
    template = (settings.external_cmd or "").strip() or (env.get("MRF_MASK_DETECTOR_CMD") or "").strip()
    if not template:
        return DetectorProbe(False, 0, "no detector command configured (set external_cmd or MRF_MASK_DETECTOR_CMD)")

    video = Path(video)
    if not video.is_file():
        raise MaskDetectorError(f"source video not found: {video}")

    ffmpeg_bin = ffmpeg or shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return DetectorProbe(False, 0, "ffmpeg not on PATH - install FFmpeg to probe the detector")

    with tempfile.TemporaryDirectory(prefix="mrf-probe-") as tmp:
        tmp_dir = Path(tmp)
        clip = tmp_dir / "probe.mp4"
        out = tmp_dir / "masks"
        out.mkdir(parents=True, exist_ok=True)

        try:
            extract = subprocess.run(
                [ffmpeg_bin, "-y", "-i", str(video), "-frames:v", "1", str(clip)],
                capture_output=True, text=True, timeout=timeout,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return DetectorProbe(False, 0, f"could not extract a probe frame: {exc}")
        if extract.returncode != 0 or not clip.is_file():
            return DetectorProbe(False, 0, f"could not extract a probe frame: {(extract.stderr or '').strip()[-200:]}")

        command = build_external_command(template, clip, out)
        try:
            run = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return DetectorProbe(False, 0, f"detector timed out after {timeout}s", command)
        except OSError as exc:
            return DetectorProbe(False, 0, f"cannot launch detector: {exc}", command)

        masks = len(sorted(out.glob("*.png")))
        if run.returncode != 0:
            tail = (run.stderr or run.stdout or "").strip()[-300:]
            return DetectorProbe(False, masks, f"detector exited {run.returncode}: {tail}", command)
        if masks == 0:
            return DetectorProbe(False, 0, "detector ran but wrote no PNG masks", command)
        return DetectorProbe(True, masks, f"ok - detector wrote {masks} mask(s) from 1 frame", command)
