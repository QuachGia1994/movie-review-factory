"""Hook Crafter - pick the most dramatic scene and cut a 3-5s opening teaser.

Short-form review retention lives or dies in the first ~5 seconds, so this
module finds the single most dramatic scene in the scene index and cuts a tight,
punchy teaser (a slow punch-in zoom for energy plus an optional whoosh/impact
SFX) that the creator can drop on the front of the review.

Selection is a deterministic, offline heuristic over ``scenes.json`` - action /
tension keywords (English + Vietnamese) in the dialogue and visual tags/actions,
weighted and normalised - so it needs no GPU or network and is fully
unit-testable. It never reads or modifies ``final.mp4`` and is not an export
approval; it only produces the standalone ``hook.mp4`` + ``hook.json`` preview.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .pipeline import (
    RENDER_CANVASES,
    RENDER_FRAME_RATE,
    ffmpeg_thread_cap,
    load_manifest,
)

HOOK_MIN_SECONDS = 3.0
HOOK_MAX_SECONDS = 5.0
HOOK_TARGET_SECONDS = 4.0

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# Action / tension cues that make an opening irresistible. Kept broad and
# bilingual (English + Vietnamese) because the source films and their transcripts
# are mixed. Vietnamese multi-syllable cues are listed per token because the
# tokenizer splits on whitespace (e.g. "tấn công" -> "tấn", "công").
_DRAMA_KEYWORDS = {
    # English
    "fight", "punch", "kick", "run", "chase", "scream", "shout", "explosion",
    "blast", "gun", "shoot", "kill", "killed", "die", "died", "death", "blood",
    "crash", "fire", "attack", "betray", "betrayal", "reveal", "twist", "cry",
    "kiss", "fear", "panic", "escape", "war", "battle", "knife", "fall", "danger",
    "save", "rescue", "chaos", "shock", "monster", "ghost", "kidnap", "scream",
    # Vietnamese
    "đánh", "đấm", "đá", "chạy", "rượt", "đuổi", "hét", "la", "nổ", "súng",
    "bắn", "giết", "chết", "máu", "đâm", "cháy", "tấn", "công", "phản", "bội",
    "tiết", "lộ", "khóc", "hôn", "sợ", "hoảng", "trốn", "thoát", "chiến", "đấu",
    "dao", "ngã", "nguy", "hiểm", "cứu", "bí", "mật", "kinh", "hoàng", "ám", "sát",
}


def _tokens(value: object) -> list[str]:
    return _TOKEN_RE.findall(str(value or "").casefold())


def _keyword_hits(*values: object) -> int:
    return sum(
        1
        for value in values
        for token in _tokens(value)
        if token in _DRAMA_KEYWORDS
    )


def dramatic_score(scene: dict) -> float:
    """Heuristic 0..1 score of how dramatic / high-tension a scene is.

    Visual actions weigh most (action *is* drama on screen), then dialogue and
    the visual tags/description. Normalised by a soft cap so a handful of strong
    cues already saturates the score.
    """
    actions = " ".join(str(v) for v in scene.get("visual_actions") or [])
    tags = " ".join(str(v) for v in scene.get("visual_tags") or [])
    weighted = (
        2.0 * _keyword_hits(actions)
        + 1.0 * _keyword_hits(scene.get("text"))
        + 1.0 * _keyword_hits(tags)
        + 1.0 * _keyword_hits(scene.get("visual_description"))
    )
    return max(0.0, min(1.0, weighted / 6.0))


def _duration(scene: dict) -> float:
    try:
        return max(0.0, float(scene.get("end_seconds", 0)) - float(scene.get("start_seconds", 0)))
    except (TypeError, ValueError):
        return 0.0


def select_dramatic_scene(scenes: list[dict]) -> dict | None:
    """Return the most dramatic scene, or None when there are no usable scenes.

    Ranks by dramatic score, then longer duration, then earlier index, so the
    choice is fully deterministic. When no scene shows any drama cue it falls back
    to the longest scene, so a teaser can always be produced.
    """
    usable = [s for s in scenes if isinstance(s, dict) and _duration(s) > 0]
    if not usable:
        return None

    def _key(scene: dict) -> tuple[float, float, int]:
        return (dramatic_score(scene), _duration(scene), -int(scene.get("index") or 0))

    return max(usable, key=_key)


def hook_teaser_range(
    scene: dict,
    *,
    target: float = HOOK_TARGET_SECONDS,
    minimum: float = HOOK_MIN_SECONDS,
    maximum: float = HOOK_MAX_SECONDS,
) -> dict:
    """Center a min..max-second teaser window on the scene, clamped to its bounds."""
    start = float(scene.get("start_seconds", 0.0))
    end = float(scene.get("end_seconds", 0.0))
    span = max(0.0, end - start)
    duration = min(maximum, max(minimum, target))
    if span <= duration:
        window_start, duration = start, span
    else:
        mid = (start + end) / 2.0
        window_start = max(start, mid - duration / 2.0)
        if window_start + duration > end:
            window_start = end - duration
    window_start = max(0.0, window_start)
    return {
        "start_seconds": round(window_start, 3),
        "end_seconds": round(window_start + duration, 3),
        "duration_seconds": round(duration, 3),
    }


def hook_advisory(advice: dict | None) -> list[str]:
    """Hook-relevant advisory lines from :func:`analytics.retention_advice` (read-only).

    Returns the Vietnamese suggestion strings whose target is the opening hook, or an
    empty list when advice is missing / disabled. Never changes the teaser window.
    """
    if not isinstance(advice, dict) or not advice.get("enabled"):
        return []
    return [
        item["message_vi"]
        for item in advice.get("suggestions") or []
        if isinstance(item, dict) and item.get("target") == "hook" and item.get("message_vi")
    ]


def plan_hook(scenes: list[dict], *, advice: dict | None = None, **kwargs: object) -> dict | None:
    """Full teaser plan for the most dramatic scene, or None when unavailable.

    ``advice`` (optional, from :func:`analytics.retention_advice`) only *annotates*
    the plan with read-only ``advisory`` notes about early drop-off; it never changes
    the selected scene or the teaser window.
    """
    scene = select_dramatic_scene(scenes)
    if scene is None:
        return None
    window = hook_teaser_range(scene, **kwargs)  # type: ignore[arg-type]
    score = dramatic_score(scene)
    plan = {
        "scene_index": int(scene.get("index") or 0),
        "dramatic_score": round(score, 6),
        "reason": "top dramatic scene" if score > 0 else "longest scene (no drama cue found)",
        **window,
    }
    notes = hook_advisory(advice)
    if notes:
        plan["advisory"] = notes
    return plan


def build_teaser_command(
    ffmpeg: str,
    source: Path,
    plan: dict,
    dest: Path,
    *,
    width: int,
    height: int,
    fps: int = RENDER_FRAME_RATE,
    sfx_path: Path | None = None,
    zoom: bool = True,
) -> list[str]:
    """ffmpeg command that cuts the teaser with a punch-in zoom (adds energy) and
    an optional whoosh/impact SFX mixed over the source audio.

    The clip is read with a seeked input (``-ss``/``-t``) so only the teaser range
    is decoded, then normalised to the target canvas.
    """
    start = float(plan["start_seconds"])
    duration = float(plan["duration_seconds"])
    frames = max(1, round(duration * fps))
    if zoom:
        geometry = (
            f"scale={width * 2}:{height * 2}:force_original_aspect_ratio=increase,"
            f"crop={width * 2}:{height * 2},"
            f"zoompan=z='min(zoom+0.0018,1.18)':d={frames}:s={width}x{height}:fps={fps},"
        )
    else:
        geometry = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
        )
    video_filter = f"[0:v]{geometry}setsar=1,fps={fps},format=yuv420p[v]"

    command = [
        ffmpeg, "-y", "-ss", f"{start:.6f}", "-t", f"{duration:.6f}", "-i", str(source),
    ]
    if sfx_path is not None:
        command += ["-i", str(sfx_path)]
        filter_complex = (
            f"{video_filter};"
            "[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a0];"
            "[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[a1];"
            "[a0][a1]amix=inputs=2:duration=first:dropout_transition=0[a]"
        )
        audio_map = "[a]"
    else:
        filter_complex = video_filter
        audio_map = "0:a?"

    command += [
        "-filter_complex", filter_complex,
        "-map", "[v]", "-map", audio_map,
        "-c:v", "libx264", "-threads", str(ffmpeg_thread_cap()),
        "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-r", str(fps),
        "-t", f"{duration:.6f}",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(dest),
    ]
    return command


def _load_scenes(root: Path) -> list[dict]:
    raw = json.loads((root / "scenes.json").read_text(encoding="utf-8"))
    scenes = raw.get("scenes") if isinstance(raw, dict) else raw
    return [scene for scene in (scenes or []) if isinstance(scene, dict)]


def build_hook_teaser(
    root: Path,
    *,
    ratio: str | None = None,
    ffmpeg: str | None = None,
) -> Path:
    """Select the most dramatic scene and render ``hook.mp4`` (+ ``hook.json``).

    Reads ``scenes.json`` in the job root; raises a clear error when a
    prerequisite is missing. Set ``MRF_HOOK_SFX`` to a whoosh/impact audio file to
    mix it under the teaser. Returns the ``hook.mp4`` path.
    """
    root = Path(root)
    if not (root / "scenes.json").is_file():
        raise FileNotFoundError("hook teaser requires scenes.json (run the scenes stage first)")
    manifest = load_manifest(root)
    source = Path(manifest.config.source_video or "")
    if not source.is_file():
        raise FileNotFoundError("hook teaser requires the configured source video")
    ratio = ratio or getattr(manifest.config, "aspect_ratio", None) or "16:9"
    width, height = RENDER_CANVASES.get(ratio, RENDER_CANVASES["16:9"])
    from .analytics import retention_advice  # lazy import avoids a module import cycle
    plan = plan_hook(_load_scenes(root), advice=retention_advice(root))
    if plan is None:
        raise ValueError("no usable scene found for a hook teaser")
    encoder = ffmpeg or shutil.which("ffmpeg")
    if not encoder:
        raise RuntimeError("FFmpeg is required to render the hook teaser")

    sfx_env = os.environ.get("MRF_HOOK_SFX", "").strip()
    sfx_path = Path(sfx_env) if sfx_env and Path(sfx_env).is_file() else None
    dest = root / "hook.mp4"
    command = build_teaser_command(
        encoder, source, plan, dest, width=width, height=height, sfx_path=sfx_path,
    )
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or not dest.exists():
        raise RuntimeError("hook teaser render failed: " + (result.stderr or "")[-1000:])
    (root / "hook.json").write_text(
        json.dumps(
            {
                "source_video": str(source),
                "ratio": ratio,
                "width": width,
                "height": height,
                "sfx": str(sfx_path) if sfx_path else None,
                **plan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return dest
