"""Export portrait highlights of an approved, QA-passed finished review."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw

from . import branding
from .pipeline import _escape_ffmpeg_filter_path, _srt_timestamp, caption_ass, load_manifest

SHORT_SECONDS_MIN = 3
SHORT_SECONDS_MAX = 60
OUTRO_SECONDS = 2
WIDTH = 1080
HEIGHT = 1920
# Caption panel geometry. build_short draws a dark panel at y=1190..1440; with a
# frame-sized ASS PlayRes (1080x1920) Alignment=2 + this bottom margin place the
# block inside that panel (text bottom ~= HEIGHT - MARGIN_V = 1330). Font size is
# tuned for the narrow <=25-char portrait wrap done by _portrait_cues, not the
# landscape height fraction. Burning through caption_ass (not force_style) is
# what makes these pixels accurate -- libass ignores force_style/original_size
# PlayRes for SRT and would otherwise scale a 384x288 default and shove the
# caption off the panel.
SHORT_CAPTION_MARGIN_V = 590
SHORT_CAPTION_FONT_SIZE = 45


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _outro(jobs_root: Path, target: Path) -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), (15, 18, 27))
    logo = Image.open(branding.logo_path(jobs_root)).convert("RGBA")
    logo.thumbnail((280, 280), Image.Resampling.LANCZOS)
    image.paste(logo, ((WIDTH - logo.width) // 2, 580), logo)
    draw = ImageDraw.Draw(image)
    name = branding.load_settings(jobs_root)["name"]
    for line, y, size in (
        (name, 910, 78),
        ("Xem bản review đầy đủ", 1040, 50),
        ("Theo dõi để xem phần tiếp theo", 1120, 42),
    ):
        font = branding._fitted(draw, line, WIDTH - 170, size)
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(((WIDTH - (box[2] - box[0])) / 2, y), line,
                  font=font, fill=(248, 249, 251))
    image.save(target, format="PNG")


def _promote_export(files: tuple[tuple[Path, Path], ...]) -> None:
    """Replace a complete export while retaining the previous set for rollback."""
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for _, target in files:
            if target.exists():
                backup = target.with_name(f"{target.name}.previous-{uuid4().hex}")
                os.replace(target, backup)
                backups.append((target, backup))
        for staged, target in files:
            os.replace(staged, target)
            promoted.append(target)
    except Exception:
        for target in promoted:
            target.unlink(missing_ok=True)
        for target, backup in reversed(backups):
            os.replace(backup, target)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)


def _portrait_cues(start: float, end: float, text: str) -> list[tuple[float, float, str]]:
    words = text.split()
    lines: list[str] = []
    for word in words:
        if len(word) > 25:
            raise ValueError("short caption contains a word wider than portrait safe area")
        if lines and len(lines[-1]) + 1 + len(word) <= 25:
            lines[-1] += " " + word
        else:
            lines.append(word)
    groups = ["\n".join(lines[index:index + 2]) for index in range(0, len(lines), 2)]
    segment = (end - start) / len(groups)
    return [
        (start + segment * index, end if index == len(groups) - 1
         else start + segment * (index + 1), group)
        for index, group in enumerate(groups)
    ]


def build_short(
    root: Path,
    start_seconds: float,
    end_seconds: float,
    *,
    ffmpeg: str | None = None,
) -> Path:
    """Create a 9:16 excerpt strictly from final.mp4 with offset captions and branded outro.

    The creator selects an approved commentary span; output is local and never uploaded.
    A failed render preserves the previous short-review.mp4.
    """
    root = Path(root)
    manifest = load_manifest(root)
    required = ("script", "alignment", "render", "qa")
    pending = [stage for stage in required
               if not manifest.stage(stage) or manifest.stage(stage).status != "ready"]
    if pending:
        raise ValueError("short blocked: stages not ready: " + ", ".join(pending))
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    if not script.get("approved") or not qa.get("passed") or qa.get("output_file") != "final.mp4":
        raise ValueError("short blocked: approved script and passing final.mp4 QA required")
    final = root / "final.mp4"
    if not final.is_file() or not final.stat().st_size:
        raise ValueError("short blocked: final.mp4 missing")
    duration_check = next(
        (item for item in qa.get("checks", [])
         if item.get("check") == "positive_duration" and item.get("passed")), None)
    duration = float(duration_check["value"]) if duration_check else None
    start, end = float(start_seconds), float(end_seconds)
    if (not math.isfinite(start) or not math.isfinite(end) or start < 0
            or end <= start or end - start < SHORT_SECONDS_MIN
            or end - start > SHORT_SECONDS_MAX
            or duration is None or not math.isfinite(duration) or end > duration):
        raise ValueError("short range must be 3–60 seconds inside QA-verified review duration")
    aligned = json.loads((root / "alignment.json").read_text(encoding="utf-8"))
    cues = []
    for cue in aligned.get("cues", []):
        cue_start = max(start, float(cue["start_seconds"]))
        cue_end = min(end, float(cue["end_seconds"]))
        label = str(cue.get("text", "")).strip()
        if cue_end > cue_start and label:
            lines = label.splitlines()
            if len(lines) > 2 or any(len(line) > 42 for line in lines):
                raise ValueError("short captions exceed the two-line review subtitle limit")
            cues.extend(_portrait_cues(cue_start - start, cue_end - start, label))
    if not cues:
        raise ValueError("short requires a highlight containing finished review commentary")
    source_stat = final.stat()
    input_fingerprints = {
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "qa_sha256": _digest(root / "qa.json"),
        "approved_script_sha256": _digest(root / "script.json"),
        "alignment_sha256": _digest(root / "alignment.json"),
    }
    encoder = ffmpeg or shutil.which("ffmpeg")
    if not encoder:
        raise RuntimeError("ffmpeg not found")
    folder = root / "shorts"
    folder.mkdir(exist_ok=True)
    subtitle_path = folder / "short-review.srt"
    subtitle_tmp = folder / "short-review-render.srt"
    subtitle_text = "\n".join(
        f"{i}\n{_srt_timestamp(cue_start)} --> {_srt_timestamp(cue_end)}\n{label}\n"
        for i, (cue_start, cue_end, label) in enumerate(cues, 1)
    )
    outro_path = folder / "short-outro-render.png"
    _outro(root.parent, outro_path)
    subtitle_tmp.write_text(subtitle_text, encoding="utf-8")
    # Same caption generator as the landscape render/preview (single source of
    # truth), with the Shorts panel's bespoke pixel geometry.
    subtitle_ass_tmp = folder / "short-review-render.ass"
    subtitle_ass_tmp.write_text(
        caption_ass(
            subtitle_text, WIDTH, HEIGHT,
            margin_v=SHORT_CAPTION_MARGIN_V, font_size=SHORT_CAPTION_FONT_SIZE,
        ),
        encoding="utf-8",
    )
    target = folder / "short-review.mp4"
    temporary = folder / "short-review-render.tmp"
    receipt_path = folder / "short-review.json"
    receipt_tmp = folder / "short-review.json.tmp"
    filter_graph = (
        f"[0:v]split[background][foreground];"
        f"[background]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},boxblur=20:1[blurred];"
        f"[foreground]scale=960:960:force_original_aspect_ratio=decrease,"
        f"drawbox=x=0:y=ih*0.78:w=iw:h=ih*0.22:color=0x11151f@0.92:t=fill[review];"
        f"[blurred][review]overlay=(W-w)/2:(H-h)/2,drawbox=x=80:y=1190:w=920:h=250:"
        f"color=0x11151f@0.84:t=fill,"
        f"ass='{_escape_ffmpeg_filter_path(subtitle_ass_tmp)}',"
        f"setsar=1,fps=30,format=yuv420p[main];"
        f"[1:v]scale={WIDTH}:{HEIGHT},setsar=1,fps=30,format=yuv420p[outro];"
        f"[0:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[speech];"
        f"anullsrc=r=48000:cl=stereo,atrim=duration={OUTRO_SECONDS}[silence];"
        f"[main][speech][outro][silence]concat=n=2:v=1:a=1[v][a]"
    )
    command = [
        encoder, "-y", "-ss", str(start), "-t", str(end - start),
        "-i", str(final), "-loop", "1", "-t", str(OUTRO_SECONDS),
        "-i", str(outro_path), "-filter_complex", filter_graph,
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac",
        "-b:a", "160k", "-movflags", "+faststart",
        "-t", str(end - start + OUTRO_SECONDS), "-f", "mp4", str(temporary),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode or not temporary.is_file() or not temporary.stat().st_size:
            raise RuntimeError("short FFmpeg render failed: " + result.stderr[-4000:])
        current_stat = final.stat()
        if (current_stat.st_size != input_fingerprints["source_bytes"]
                or current_stat.st_mtime_ns != input_fingerprints["source_mtime_ns"]
                or _digest(root / "script.json") != input_fingerprints["approved_script_sha256"]
                or _digest(root / "qa.json") != input_fingerprints["qa_sha256"]
                or _digest(root / "alignment.json") != input_fingerprints["alignment_sha256"]):
            raise RuntimeError("review inputs changed while the short was rendering")
        receipt = {
            "source": "final.mp4", "source_sha256": _digest(final),
            **input_fingerprints,
            "qa_passed": True,
            "review_range_seconds": [start, end],
            "subtitle_file": subtitle_path.name,
            "subtitle_sha256": _digest(subtitle_tmp),
            "subtitle_bytes": subtitle_tmp.stat().st_size,
            "subtitle_mtime_ns": subtitle_tmp.stat().st_mtime_ns,
            "outro_seconds": OUTRO_SECONDS,
            "output_sha256": _digest(temporary),
            "output_bytes": temporary.stat().st_size,
            "output_mtime_ns": temporary.stat().st_mtime_ns,
        }
        receipt_tmp.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        _promote_export((
            (subtitle_tmp, subtitle_path),
            (temporary, target),
            (receipt_tmp, receipt_path),
        ))
    finally:
        temporary.unlink(missing_ok=True)
        subtitle_tmp.unlink(missing_ok=True)
        subtitle_ass_tmp.unlink(missing_ok=True)
        outro_path.unlink(missing_ok=True)
        receipt_tmp.unlink(missing_ok=True)
    return target


def short_artifact(root: Path, name: str, *, verify_output: bool = True) -> Path:
    """Resolve only current, approved portrait exports for a download route."""
    root = Path(root)
    if name not in ("short-review.mp4", "short-review.srt", "short-review.json"):
        raise ValueError("short artifact name is not allowed")
    manifest = load_manifest(root)
    if any(not manifest.stage(stage) or manifest.stage(stage).status != "ready"
           for stage in ("script", "alignment", "render", "qa")):
        raise ValueError("short stale: review stages must be ready")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    if not script.get("approved") or not qa.get("passed") or qa.get("output_file") != "final.mp4":
        raise ValueError("short download requires approved script and passed QA")
    folder = root / "shorts"
    receipt = json.loads((folder / "short-review.json").read_text(encoding="utf-8"))
    final = root / "final.mp4"
    fingerprints = {
        "source_bytes": final.stat().st_size,
        "source_mtime_ns": final.stat().st_mtime_ns,
        "qa_sha256": _digest(root / "qa.json"),
        "approved_script_sha256": _digest(root / "script.json"),
        "alignment_sha256": _digest(root / "alignment.json"),
    }
    if any(receipt.get(key) != current for key, current in fingerprints.items()):
        raise ValueError("short stale: review inputs changed; render the variant again")
    video = folder / "short-review.mp4"
    subtitle = folder / "short-review.srt"
    for item in (video, subtitle):
        if not item.is_file():
            raise FileNotFoundError(f"short artifact missing: {item.name}")
    if (video.stat().st_size != receipt.get("output_bytes")
            or video.stat().st_mtime_ns != receipt.get("output_mtime_ns")
            or subtitle.stat().st_size != receipt.get("subtitle_bytes")
            or subtitle.stat().st_mtime_ns != receipt.get("subtitle_mtime_ns")):
        raise ValueError("short stale: exported video or subtitle changed")
    if verify_output and (
            _digest(video) != receipt.get("output_sha256")
            or _digest(subtitle) != receipt.get("subtitle_sha256")):
        raise ValueError("short stale: exported video or subtitle checksum changed")
    path = folder / name
    if not path.is_file():
        raise FileNotFoundError(f"short artifact missing: {name}")
    return path
