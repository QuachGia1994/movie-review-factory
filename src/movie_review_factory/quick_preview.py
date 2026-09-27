"""Small, revision-bound section preview assembled from selected source shots.

The preview uses the approved narration and exact section timing. It never
reads or modifies final.mp4, and is not itself an export approval.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import uuid
from pathlib import Path

from .pipeline import (
    RENDER_FRAME_RATE,
    SUBTITLE_BOTTOM_FRACTION,
    _escape_ffmpeg_filter_path,
    _render_source_ranges,
    _srt_timestamp,
    caption_ass,
    load_manifest,
)


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprints(root: Path, source: Path) -> dict:
    source_stat = source.stat()
    voice_stat = (root / "narration.mp3").stat()
    return {
        "script_sha256": _digest(root / "script.json"),
        "scene_plan_sha256": _digest(root / "scene_plan.json"),
        "alignment_sha256": _digest(root / "alignment.json"),
        "source_bytes": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "narration_bytes": voice_stat.st_size,
        "narration_mtime_ns": voice_stat.st_mtime_ns,
    }


def _probe_duration_seconds(path: Path, ffprobe: str) -> float:
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if result.returncode:
        raise RuntimeError("preview probe failed: " + result.stderr[-1000:])
    return float(result.stdout.strip())


def _required_inputs(root: Path) -> tuple[dict, dict, dict, Path, Path]:
    manifest = load_manifest(root)
    for stage in ("script", "scene_plan", "tts", "alignment"):
        if not manifest.stage(stage) or manifest.stage(stage).status != "ready":
            raise ValueError(f"section preview requires ready {stage} stage")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    if not script.get("approved"):
        raise ValueError("section preview requires approved script")
    plan = json.loads((root / "scene_plan.json").read_text(encoding="utf-8"))
    alignment = json.loads((root / "alignment.json").read_text(encoding="utf-8"))
    source = Path(manifest.config.source_video or "")
    audio = root / "narration.mp3"
    if not source.is_file() or not audio.is_file():
        raise FileNotFoundError("section preview requires source footage and narration.mp3")
    return script, plan, alignment, source, audio


def build_section_preview(
    root: Path, section_index: int, *, ffmpeg: str | None = None, ffprobe: str | None = None,
) -> Path:
    """Render just one script section at 540p, with approved voice and timed SRT."""
    root = Path(root)
    script, plan, alignment, source, audio = _required_inputs(root)
    if isinstance(section_index, bool) or not isinstance(section_index, int) or not 1 <= section_index <= len(script.get("sections") or []):
        raise ValueError("section index must identify a script section")
    spans = [b for b in alignment.get("section_bounds") or []
             if b.get("section_index") == section_index]
    if len(spans) != 1:
        raise ValueError("precise section bounds from narration alignment are required")
    start = float(spans[0]["start_seconds"])
    end = float(spans[0]["end_seconds"])
    duration = end - start
    if not all(math.isfinite(v) for v in (start, end)) or start < 0 or duration <= 0:
        raise ValueError("invalid section bounds")
    encoder = ffmpeg or shutil.which("ffmpeg")
    prober = ffprobe or shutil.which("ffprobe")
    if not encoder or not prober:
        raise RuntimeError("FFmpeg and ffprobe are required for section preview")
    source_duration = _probe_duration_seconds(source, prober)
    narration_duration = _probe_duration_seconds(audio, prober)
    if end > narration_duration + 0.05:
        raise ValueError("section bounds extend beyond narration")
    clips = [c for c in plan.get("clips") or [] if c.get("section_index") == section_index]
    if not clips:
        raise ValueError("section preview requires selected source clips")
    ranges = _render_source_ranges(clips, source_duration)
    width, height = {"16:9": (960, 540), "9:16": (304, 540)}.get(
        plan.get("aspect_ratio") or manifest_ratio(root), (0, 0)
    )
    if not width:
        raise ValueError("unsupported preview aspect ratio")
    cues = []
    for cue in alignment.get("cues") or []:
        a, b = max(start, float(cue["start_seconds"])), min(end, float(cue["end_seconds"]))
        label = str(cue.get("text") or "").strip()
        if a >= b or not label:
            continue
        lines = label.splitlines()
        if len(lines) > 2 or any(len(line) > 42 for line in lines):
            raise ValueError("preview subtitles exceed two lines or 42 characters per line")
        cues.append((a - start, b - start, label))
    if not cues:
        raise ValueError("section preview needs aligned narration subtitles")
    fingerprints = _fingerprints(root, source)
    folder = root / "previews"
    folder.mkdir(exist_ok=True)
    stem = f"section-{section_index}"
    nonce = uuid.uuid4().hex
    tmp_srt = folder / f"{stem}.{nonce}.srt"
    tmp_ass = folder / f"{stem}.{nonce}.ass"
    tmp_video = folder / f"{stem}.{nonce}.rendering"
    tmp_receipt = folder / f"{stem}.{nonce}.json"
    target = folder / f"{stem}.mp4"
    srt = folder / f"{stem}.srt"
    receipt_path = folder / f"{stem}.json"
    tmp_srt.write_text("\n".join(
        f"{index}\n{_srt_timestamp(a)} --> {_srt_timestamp(b)}\n{label}\n"
        for index, (a, b, label) in enumerate(cues, 1)
    ), encoding="utf-8")
    # Same caption layout policy as the final render (single source of truth):
    # a frame-sized ASS so margins/font are pixel-accurate at preview resolution.
    tmp_ass.write_text(
        caption_ass(tmp_srt.read_text(encoding="utf-8"), width, height, SUBTITLE_BOTTOM_FRACTION),
        encoding="utf-8",
    )
    filters = []
    video_inputs = []
    for index, item in enumerate(ranges):
        source_seconds = item["source_seconds"]
        # Exact voice section span controls the preview, matching the main render.
        target_seconds = duration / len(ranges) if index < len(ranges) - 1 else duration - duration / len(ranges) * index
        piece = (f"[0:v]trim=start={item['start_seconds']:.6f}:end={item['end_seconds']:.6f},"
                 "setpts=PTS-STARTPTS")
        if target_seconds > source_seconds:
            loops = math.ceil(target_seconds / source_seconds) - 1
            size = max(1, round(source_seconds * RENDER_FRAME_RATE))
            piece += (f",fps={RENDER_FRAME_RATE},loop=loop={loops}:size={size}:start=0,"
                      f"setpts=N/{RENDER_FRAME_RATE}/TB,trim=end={target_seconds:.6f},setpts=PTS-STARTPTS")
        elif target_seconds < source_seconds:
            piece += f",trim=end={target_seconds:.6f},setpts=PTS-STARTPTS"
        filters.append(
            f"{piece},scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={RENDER_FRAME_RATE},format=yuv420p[v{index}]"
        )
        video_inputs.append(f"[v{index}]")
    filters.extend([
        f"{''.join(video_inputs)}concat=n={len(ranges)}:v=1:a=0[video]",
        f"[video]tpad=stop_mode=clone:stop_duration=1,"
        f"ass='{_escape_ffmpeg_filter_path(tmp_ass)}'[v]",
        f"[1:a]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS[a]",
    ])
    command = [
        encoder, "-y", "-i", str(source), "-i", str(audio),
        "-filter_complex", ";".join(filters), "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-r", str(RENDER_FRAME_RATE),
        "-c:a", "aac", "-b:a", "128k", "-t", f"{duration:.6f}",
        "-movflags", "+faststart", "-f", "mp4", str(tmp_video),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode or not tmp_video.is_file() or not tmp_video.stat().st_size:
            raise RuntimeError("section preview render failed: " + result.stderr[-3000:])
        if _fingerprints(root, source) != fingerprints:
            raise RuntimeError("preview inputs changed during render")
        receipt = {
            "section_index": section_index,
            "duration_seconds": duration,
            "source_ranges": [[item["start_seconds"], item["end_seconds"]] for item in ranges],
            "inputs": fingerprints,
            "output_sha256": _digest(tmp_video),
            "subtitle_sha256": _digest(tmp_srt),
            "resolution": [width, height],
        }
        tmp_receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_srt, srt)
        os.replace(tmp_video, target)
        os.replace(tmp_receipt, receipt_path)
        return target
    finally:
        for path in (tmp_srt, tmp_ass, tmp_video, tmp_receipt):
            path.unlink(missing_ok=True)


def manifest_ratio(root: Path) -> str:
    return load_manifest(root).config.aspect_ratio


def section_preview_artifact(root: Path, section_index: int, kind: str = "mp4") -> Path:
    """Resolve a current preview artifact; block stale or unapproved previews."""
    if isinstance(section_index, bool) or not isinstance(section_index, int) or section_index < 1:
        raise ValueError("invalid section index")
    if kind not in {"mp4", "srt", "json"}:
        raise ValueError("invalid preview artifact kind")
    root = Path(root)
    script, _plan, _alignment, source, _audio = _required_inputs(root)
    if section_index > len(script.get("sections") or []):
        raise ValueError("invalid section index")
    folder = root / "previews"
    receipt_path = folder / f"section-{section_index}.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("section_index") != section_index
            or receipt.get("inputs") != _fingerprints(root, source)):
        raise ValueError("section preview is stale; render again")
    path = folder / f"section-{section_index}.{kind}"
    if not path.is_file():
        raise FileNotFoundError(f"section preview artifact missing: {kind}")
    if kind == "mp4" and receipt.get("output_sha256") != _digest(path):
        raise ValueError("section preview is stale; video changed")
    if kind == "srt" and receipt.get("subtitle_sha256") != _digest(path):
        raise ValueError("section preview is stale; captions changed")
    return path
