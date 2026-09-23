import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable

from .content_agent import run_claude_json
from .models import Artifact, JobConfig, JobManifest, StageResult

STAGES = ("ingest","research","transcript","scenes","outline","script","scene_plan","tts","alignment","render","qa","metadata","thumbnail","publish")

# Stages that still need source video / media or external models and have no
# real handler yet. These are honestly marked "skipped" (never faked).
MEDIA_STAGES = frozenset({"transcript", "scenes", "alignment"})

MANIFEST_NAME = "manifest.json"
MANIFEST_RETRY_ATTEMPTS = 10
MANIFEST_RETRY_DELAY_SECONDS = 0.01
_KNOWN_ARTIFACTS = (
    "ingest.json", "research.json", "transcript.json", "captions.srt",
    "scenes.json", "outline.json", "script.json", "script.md", "scene_plan.json",
    "voice.json", "narration.mp3", "alignment.json", "aligned.srt",
    "render.json", "final.mp4", "qa.json",
    "youtube_metadata.json", "thumbnails.json", "thumbnail.jpg",
    "thumbnail-1.jpg", "thumbnail-2.jpg", "thumbnail-3.jpg",
    "publish_record.json",
)

_STAGE_ARTIFACTS = {
    "ingest": ("ingest.json",),
    "research": ("research.json",),
    "transcript": ("transcript.json", "captions.srt"),
    "scenes": ("scenes.json",),
    "outline": ("outline.json",),
    "script": ("script.json", "script.md"),
    "scene_plan": ("scene_plan.json",),
    "tts": ("voice.json", "narration.mp3"),
    "alignment": ("alignment.json", "aligned.srt"),
    "render": ("render.json", "final.mp4"),
    "qa": ("qa.json",),
    "metadata": ("youtube_metadata.json",),
    "thumbnail": (
        "thumbnails.json", "thumbnail.jpg",
        "thumbnail-1.jpg", "thumbnail-2.jpg", "thumbnail-3.jpg",
    ),
    "publish": ("publish_record.json",),
}

RENDER_CANVASES = {"16:9": (1920, 1080), "9:16": (1080, 1920)}
RENDER_FRAME_RATE = 25
RENDER_DURATION_DRIFT_SECONDS = 0.25
THUMBNAIL_SIZE = (1280, 720)
THUMBNAIL_COUNT = 3
SCENE_PLAN_MAX_SHOTS_PER_SECTION = 6
SCENE_PLAN_TARGET_SHOT_SECONDS = 8.0


# --- manifest I/O -----------------------------------------------------------

def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def load_manifest(root: Path) -> JobManifest:
    path = manifest_path(root)
    for attempt in range(MANIFEST_RETRY_ATTEMPTS):
        try:
            manifest = JobManifest.model_validate_json(path.read_text(encoding="utf-8"))
            actual = [stage.stage for stage in manifest.stages]
            legacy_without_thumbnail = [name for name in STAGES if name != "thumbnail"]
            if actual == legacy_without_thumbnail:
                existing = {stage.stage: stage for stage in manifest.stages}
                manifest.stages = [
                    existing.get(name, StageResult(stage=name, status="pending"))
                    for name in STAGES
                ]
            return manifest
        except PermissionError:
            if attempt == MANIFEST_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(MANIFEST_RETRY_DELAY_SECONDS)


def save_manifest(root: Path, manifest: JobManifest) -> Path:
    """Atomically replace the manifest so concurrent dashboard reads are valid."""
    path = manifest_path(root)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    for attempt in range(3):
        try:
            temporary.replace(path)
            return path
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.01)


def create_job(root: Path, config: JobConfig) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in _KNOWN_ARTIFACTS:
        (root / name).unlink(missing_ok=True)
    manifest = JobManifest(config=config, stages=[StageResult(stage=s, status="pending") for s in STAGES])
    return save_manifest(root, manifest)


def validate_job(root: Path) -> list[str]:
    path = manifest_path(root)
    if not path.exists():
        return ["manifest.json is missing"]
    try:
        manifest = load_manifest(root)
    except Exception as exc:
        return [f"invalid manifest: {exc}"]
    actual = [s.stage for s in manifest.stages]
    if actual != list(STAGES):
        return [f"stage order mismatch: expected {list(STAGES)}, got {actual}"]
    return []


# --- stage handler registry -------------------------------------------------

# A handler does deterministic, local work and returns (artifacts, message).
# Raise SkipStage to mark the stage "skipped" (a controlled, non-error skip,
# e.g. a required input is missing). Any other exception marks it "failed" and
# stops the run so the job stays resumable.
StageHandler = Callable[[Path, JobManifest], tuple[list[Artifact], str]]

STAGE_HANDLERS: dict[str, StageHandler] = {}


class SkipStage(Exception):
    """Raised by a handler to mark its stage 'skipped' rather than 'failed'."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def register_stage(name: str) -> Callable[[StageHandler], StageHandler]:
    if name not in STAGES:
        raise ValueError(f"unknown stage: {name}")

    def decorator(fn: StageHandler) -> StageHandler:
        STAGE_HANDLERS[name] = fn
        return fn

    return decorator


def _write_json(root: Path, name: str, data: dict) -> Artifact:
    out = root / name
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(out)
    return Artifact(name=name, path=out, status="ready")


def _write_text(root: Path, name: str, text: str) -> Artifact:
    out = root / name
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(out)
    return Artifact(name=name, path=out, status="ready")


def _read_json(root: Path, name: str) -> dict:
    """Read a JSON artifact; return {} if it does not exist yet."""
    p = root / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def invalidate_downstream(root: Path, changed_stage: str) -> JobManifest:
    """Invalidate every stage derived from changed_stage and remove stale outputs."""
    if changed_stage not in STAGES:
        raise ValueError(f"unknown stage: {changed_stage!r}")

    manifest = load_manifest(root)
    changed_index = STAGES.index(changed_stage)
    for stage in manifest.stages:
        if STAGES.index(stage.stage) <= changed_index:
            continue
        for name in _STAGE_ARTIFACTS.get(stage.stage, ()):
            (root / name).unlink(missing_ok=True)
        stage.status = "pending"
        stage.artifacts = []
        stage.message = ""
        stage.updated_at = None
    save_manifest(root, manifest)
    return manifest


# --- ingest: probe source video with ffprobe --------------------------------

@register_stage("ingest")
def _ingest(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Probe the source video with ffprobe and record duration/codecs.

    Self-skips when there is no source video or ffprobe is not installed,
    so a job with no media still runs cleanly.
    """
    cfg = manifest.config
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to ingest")
    src = Path(cfg.source_video)
    if not src.exists():
        raise SkipStage(f"source_video not found: {src}")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to ingest")
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(src)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:200]}")
    probe = json.loads(proc.stdout)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})
    data = {
        "source_video": str(src),
        "duration_seconds": float(fmt.get("duration") or 0),
        "video_codec": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "audio_codec": audio.get("codec_name"),
        "has_audio": bool(audio),
    }
    return [_write_json(root, "ingest.json", data)], "probed source video with ffprobe"


# --- transcript: timed speech-to-text with faster-whisper -------------------

def _srt_timestamp(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    seconds, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"


def _whisper_model_options() -> dict[str, object]:
    """Return optional app-owned cache/offline settings for faster-whisper."""
    options: dict[str, object] = {}
    cache = os.environ.get("MRF_WHISPER_CACHE", "").strip()
    if cache:
        options["download_root"] = cache
    if os.environ.get("MRF_WHISPER_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        options["local_files_only"] = True
    return options


@register_stage("transcript")
def _transcript(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Transcribe a local source video into timed JSON and SRT artifacts."""
    cfg = manifest.config
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to transcribe")
    src = Path(cfg.source_video)
    if not src.exists():
        raise SkipStage(f"source_video not found: {src}")
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SkipStage("faster-whisper not installed - install the media extra") from exc

    model = WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
        **_whisper_model_options(),
    )
    raw_segments, _ = model.transcribe(str(src), language=cfg.language, vad_filter=True)
    segments = [
        {
            "start_seconds": float(segment.start),
            "end_seconds": float(segment.end),
            "text": segment.text.strip(),
        }
        for segment in raw_segments
    ]
    transcript = {
        "job_id": cfg.job_id,
        "source_video": str(src),
        "language": cfg.language,
        "segments": segments,
    }
    srt = "\n".join(
        f"{index}\n{_srt_timestamp(segment['start_seconds'])} --> "
        f"{_srt_timestamp(segment['end_seconds'])}\n{segment['text']}\n"
        for index, segment in enumerate(segments, start=1)
    )
    artifacts = [
        _write_json(root, "transcript.json", transcript),
        _write_text(root, "captions.srt", srt),
    ]
    return artifacts, f"transcribed {len(segments)} segments"


# --- scenes: deterministic scene index bounded to the source video ----------

# A new scene begins after a silent gap longer than SCENE_GAP_SECONDS or once a
# running scene would grow past SCENE_MAX_SECONDS. These keep the index
# deterministic and coarse enough to drive the downstream scene_plan.
SCENE_GAP_SECONDS = 1.5
SCENE_MAX_SECONDS = 30.0


def _probe_duration_seconds(src: Path) -> float | None:
    """Return the source video duration in seconds via ffprobe.

    Returns None when ffprobe is not installed so the scenes stage can honestly
    skip. Kept as a seam so scene indexing can be tested without a real binary.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    proc = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format", str(src)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {proc.stderr.strip()[:200]}")
    duration = json.loads(proc.stdout).get("format", {}).get("duration")
    return float(duration) if duration is not None else 0.0


def _clamp(value: float, upper: float) -> float:
    """Clamp value into the bounded range [0, upper]."""
    return max(0.0, min(float(value), upper))


def _build_scenes(segments: list[dict], duration: float) -> list[dict]:
    """Group timed transcript segments into deterministic, bounded scenes.

    All timestamps are clamped to [0, duration]; segments with no positive
    length after clamping are dropped. When there are no usable segments a
    single scene spanning the whole video is returned so the index is never
    empty for an owned video.
    """
    bounded: list[dict] = []
    for seg in segments:
        start = _clamp(seg.get("start_seconds", 0.0), duration)
        end = _clamp(seg.get("end_seconds", start), duration)
        if end <= start:
            continue
        bounded.append({"start_seconds": start, "end_seconds": end, "text": str(seg.get("text", "")).strip()})
    bounded.sort(key=lambda s: (s["start_seconds"], s["end_seconds"]))
    if not bounded:
        return [{"index": 1, "start_seconds": 0.0, "end_seconds": duration, "segment_count": 0, "text": ""}]

    scenes: list[dict] = []
    current: list[dict] = []
    for segment in bounded:
        if current and (
            segment["start_seconds"] - current[-1]["end_seconds"] > SCENE_GAP_SECONDS
            or segment["end_seconds"] - current[0]["start_seconds"] > SCENE_MAX_SECONDS
        ):
            scenes.append(_scene_from_segments(len(scenes) + 1, current))
            current = []
        current.append(segment)
    if current:
        scenes.append(_scene_from_segments(len(scenes) + 1, current))
    return scenes


def _scene_from_segments(index: int, segments: list[dict]) -> dict:
    return {
        "index": index,
        "start_seconds": segments[0]["start_seconds"],
        "end_seconds": segments[-1]["end_seconds"],
        "segment_count": len(segments),
        "text": " ".join(s["text"] for s in segments if s["text"]),
    }


@register_stage("scenes")
def _scenes(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Create a deterministic scene index from transcript timestamps."""
    cfg = manifest.config
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to index scenes")
    src = Path(cfg.source_video)
    if not src.exists():
        raise SkipStage(f"source_video not found: {src}")
    duration = _probe_duration_seconds(src)
    if duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to index scenes")
    transcript = _read_json(root, "transcript.json")
    scenes = _build_scenes(transcript.get("segments", []), duration)
    data = {
        "job_id": cfg.job_id,
        "source_video": str(src),
        "duration_seconds": duration,
        "scene_count": len(scenes),
        "scenes": scenes,
    }
    return [_write_json(root, "scenes.json", data)], f"indexed {len(scenes)} scenes from transcript"


# --- reasoning-agent contracts -----------------------------------------------

_RESEARCH_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "brief": {"type": "string"},
        "facts": {"type": "array", "items": {"type": "string"}},
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["title", "url", "note"],
                "additionalProperties": False,
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["brief", "facts", "sources", "uncertainties"],
    "additionalProperties": False,
}

_OUTLINE_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "budget_minutes": {"type": "number"},
                    "purpose": {"type": "string"},
                },
                "required": ["title", "budget_minutes", "purpose"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["sections", "notes"],
    "additionalProperties": False,
}

_SCENE_PLAN_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "shots": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": SCENE_PLAN_MAX_SHOTS_PER_SECTION,
                        "items": {
                            "type": "object",
                            "properties": {
                                "start_scene_index": {"type": "integer"},
                                "end_scene_index": {"type": "integer"},
                                "rationale": {"type": "string"},
                            },
                            "required": [
                                "start_scene_index", "end_scene_index", "rationale"
                            ],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["shots"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["assignments", "notes"],
    "additionalProperties": False,
}


_SCRIPT_AGENT_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "minItems": 3,
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "narration": {"type": "string"},
                },
                "required": ["title", "narration"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["sections", "notes"],
    "additionalProperties": False,
}


def _movie_title(config: JobConfig) -> str:
    if config.movie_title and config.movie_title.strip():
        return config.movie_title.strip()
    if config.source_video:
        return Path(config.source_video).stem
    return config.job_id


def _scene_context(root: Path) -> list[dict]:
    scenes = _read_json(root, "scenes.json").get("scenes") or []
    context: list[dict] = []
    for scene in scenes:
        if not isinstance(scene, dict):
            continue
        context.append({
            "index": scene.get("index"),
            "start_seconds": scene.get("start_seconds"),
            "end_seconds": scene.get("end_seconds"),
            "text": str(scene.get("text") or "")[:600],
        })
    return context


def _normalize_outline_sections(raw_sections: list[dict], target_minutes: float) -> list[dict]:
    cleaned: list[dict] = []
    for section in raw_sections:
        if not isinstance(section, dict):
            continue
        title = str(section.get("title") or "").strip()
        if not title:
            continue
        try:
            weight = float(section.get("budget_minutes") or 0)
        except (TypeError, ValueError):
            weight = 0.0
        cleaned.append({
            "title": title,
            "budget_minutes": max(weight, 0.0),
            "purpose": str(section.get("purpose") or "").strip(),
        })
    if not cleaned:
        raise ValueError("Claude outline contained no usable sections")

    total_weight = sum(section["budget_minutes"] for section in cleaned)
    weights = (
        [section["budget_minutes"] for section in cleaned]
        if total_weight > 0
        else [1.0] * len(cleaned)
    )
    weight_sum = sum(weights)
    assigned = 0.0
    for index, (section, weight) in enumerate(zip(cleaned, weights)):
        if index == len(cleaned) - 1:
            budget = round(float(target_minutes) - assigned, 1)
        else:
            budget = max(0.1, round(float(target_minutes) * weight / weight_sum, 1))
            assigned = round(assigned + budget, 1)
        section["budget_minutes"] = budget
    if cleaned[-1]["budget_minutes"] <= 0:
        raise ValueError("Claude outline could not be normalized to the target duration")
    return cleaned


def _run_reasoning_agent(
    *,
    root: Path,
    manifest: JobManifest,
    stage: str,
    instruction: str,
    context: dict,
    schema: dict,
    allowed_tools: list[str] | None = None,
) -> dict | None:
    if manifest.config.content_agent != "claude":
        return None

    prompt = (
        "You are the reasoning worker for Movie Review Factory. "
        "Produce original review/recap material, not copied dialogue. "
        "Do not invent facts or source URLs; put unresolved claims in uncertainties when the schema permits it. "
        "Return only the structured result requested by the JSON schema enforced by the caller.\n\n"
        f"STAGE: {stage}\n"
        f"INSTRUCTION: {instruction}\n"
        "JOB CONTEXT:\n"
        + json.dumps(context, ensure_ascii=False)
    )
    return run_claude_json(
        root=root,
        stage=stage,
        prompt=prompt,
        schema=schema,
        allowed_tools=allowed_tools,
    )


# --- research: deterministic scaffold or Claude research ---------------------

@register_stage("research")
def _research(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    cfg = manifest.config
    title = _movie_title(cfg)
    agent = _run_reasoning_agent(
        root=root,
        manifest=manifest,
        stage="research",
        instruction=(
            f"Research the film '{title}' for an original {cfg.language} review/recap. "
            "Use web research when available. Record only claims you can support; "
            "never fabricate URLs. Focus on premise, characters, themes, reception/context "
            "that helps a reviewer, and uncertainty that the story editor must not overstate."
        ),
        context={
            "movie_title": title,
            "language": cfg.language,
            "target_minutes": cfg.target_minutes,
            "source_video": str(cfg.source_video) if cfg.source_video else None,
            "ingest": _read_json(root, "ingest.json"),
        },
        schema=_RESEARCH_AGENT_SCHEMA,
        allowed_tools=["WebSearch", "WebFetch"],
    )
    if agent is not None:
        data = {
            "job_id": cfg.job_id,
            "movie_title": title,
            "language": cfg.language,
            "target_minutes": cfg.target_minutes,
            "brief": str(agent.get("brief") or "").strip(),
            "facts": list(agent.get("facts") or []),
            "sources": list(agent.get("sources") or []),
            "uncertainties": list(agent.get("uncertainties") or []),
            "status": "ready",
            "generator": "claude",
        }
        if not data["brief"]:
            raise ValueError("Claude research returned an empty brief")
        return [_write_json(root, "research.json", data)], "research generated by Claude"

    data = {
        "job_id": cfg.job_id,
        "movie_title": title,
        "language": cfg.language,
        "target_minutes": cfg.target_minutes,
        "brief": "Research brief scaffold - add verified sources and notes before final script.",
        "facts": [],
        "sources": [],
        "uncertainties": [],
        "status": "draft",
        "generator": "scaffold",
    }
    return [_write_json(root, "research.json", data)], "research brief scaffold written"


# --- outline: section/time-budget scaffold ----------------------------------

@register_stage("outline")
def _outline(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    cfg = manifest.config
    research = _read_json(root, "research.json")
    target_min = float(research.get("target_minutes") or cfg.target_minutes)
    agent = _run_reasoning_agent(
        root=root,
        manifest=manifest,
        stage="outline",
        instruction=(
            f"Design a coherent {target_min:g}-minute {cfg.language} review/recap outline. "
            "Use the research and source-scene chronology below. Balance recap with original "
            "analysis, keep the hook useful, and do not invent scenes that are absent from the "
            "provided scene context. Return 3-8 sections with relative time budgets."
        ),
        context={
            "movie_title": _movie_title(cfg),
            "language": cfg.language,
            "target_minutes": target_min,
            "research": research,
            "scenes": _scene_context(root),
        },
        schema=_OUTLINE_AGENT_SCHEMA,
    )
    if agent is not None:
        sections = _normalize_outline_sections(
            list(agent.get("sections") or []),
            target_min,
        )
        outline = {
            "job_id": cfg.job_id,
            "language": cfg.language,
            "target_minutes": target_min,
            "sections": sections,
            "notes": str(agent.get("notes") or "").strip(),
            "generator": "claude",
        }
        return [_write_json(root, "outline.json", outline)], "outline generated by Claude"

    hook_min, cta_min = 0.5, 0.5
    body_min = max(target_min - hook_min - cta_min, 1.0)
    body_titles = [
        "Bối cảnh & tiền đề",
        "Diễn biến chính (hạn chế spoiler)",
        "Điểm nhấn phân tích",
        "Đánh giá & kết luận",
    ]
    n = len(body_titles)
    per_body = round(body_min / n, 1)
    # Last section absorbs rounding slack so total never exceeds target_min.
    last_body = round(body_min - per_body * (n - 1), 1)
    sections = [
        {"title": "Mở đầu / hook", "budget_minutes": hook_min},
        *[{"title": t, "budget_minutes": per_body} for t in body_titles[:-1]],
        {"title": body_titles[-1], "budget_minutes": last_body},
        {"title": "Call to action", "budget_minutes": cta_min},
    ]
    outline = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "target_minutes": target_min,
        "sections": sections,
        "notes": "Outline scaffold - reorder or split sections as needed before script.",
        "generator": "scaffold",
    }
    return [_write_json(root, "outline.json", outline)], "outline scaffold written"


# --- script: narration scaffold from outline --------------------------------

def _script_markdown(script: dict) -> str:
    sections = script.get("sections") or []
    lines = [
        f"# Script: {script.get('job_id', '')}",
        "",
        f"Language: `{script.get('language', '')}`  |  "
        f"Target: `{script.get('target_minutes', 0)} min`  |  "
        f"**approved: {str(bool(script.get('approved'))).lower()}**",
        "",
        "> Review and edit each draft below, then approve the final script.",
        "",
    ]
    for sec in sections:
        budget = float(sec.get("budget_minutes") or 0)
        budget_note = f" _{budget} min_" if budget else ""
        lines += [f"## {sec.get('title', '')}{budget_note}", "", str(sec.get("narration", "")), ""]
    return "\n".join(lines)


@register_stage("script")
def _script(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Turn the outline into script.json + script.md behind a human approval gate."""
    cfg = manifest.config
    outline = _read_json(root, "outline.json")
    raw_sections = outline.get("sections") or [
        {"title": "Mở đầu / hook", "budget_minutes": 0.5},
        {"title": "Nội dung chính", "budget_minutes": cfg.target_minutes - 1.0},
        {"title": "Kết luận & CTA", "budget_minutes": 0.5},
    ]

    agent = _run_reasoning_agent(
        root=root,
        manifest=manifest,
        stage="script",
        instruction=(
            f"Write natural {cfg.language} narration for an original movie review/recap. "
            "Keep exactly the same section count and order as the outline. Ground plot claims "
            "in the supplied research and scene context, add analysis instead of merely retelling, "
            "avoid long verbatim dialogue, and write enough narration to fit each section budget."
        ),
        context={
            "movie_title": _movie_title(cfg),
            "language": cfg.language,
            "target_minutes": cfg.target_minutes,
            "research": _read_json(root, "research.json"),
            "outline": outline,
            "scenes": _scene_context(root),
        },
        schema=_SCRIPT_AGENT_SCHEMA,
    )

    sections = []
    if agent is not None:
        generated = list(agent.get("sections") or [])
        if len(generated) != len(raw_sections):
            raise ValueError(
                "Claude script section count does not match the approved outline shape"
            )
        for outline_section, generated_section in zip(raw_sections, generated):
            title = (
                outline_section
                if isinstance(outline_section, str)
                else str(outline_section.get("title") or "")
            )
            budget = (
                0.0
                if isinstance(outline_section, str)
                else float(outline_section.get("budget_minutes") or 0)
            )
            narration = (
                str(generated_section.get("narration") or "").strip()
                if isinstance(generated_section, dict)
                else ""
            )
            if not narration:
                raise ValueError(f"Claude script returned empty narration for {title!r}")
            sections.append({
                "title": title,
                "budget_minutes": budget,
                "narration": narration,
                "duration_seconds": round(budget * 60),
            })
        notes = str(agent.get("notes") or "").strip()
        generator = "claude"
        message = "script generated by Claude (approval required before TTS)"
    else:
        # Normalise: sections may arrive as plain strings (legacy) or dicts.
        for section in raw_sections:
            title = section if isinstance(section, str) else section.get("title", "")
            budget = (
                0.0
                if isinstance(section, str)
                else float(section.get("budget_minutes") or 0)
            )
            narration = (
                f"Bản thảo lời dẫn cho phần {title}. "
                "Hãy rà soát và chỉnh sửa nội dung này trước khi duyệt."
            )
            sections.append({
                "title": title,
                "budget_minutes": budget,
                "narration": narration,
                "duration_seconds": round(budget * 60),
            })
        notes = "Review and edit each draft narration, then set approved=true when ready for TTS."
        generator = "scaffold"
        message = "script scaffold written (set approved=true before TTS)"

    script_data = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "target_minutes": cfg.target_minutes,
        "approval_required": True,
        "approved": False,
        "sections": sections,
        "notes": notes,
        "generator": generator,
    }
    json_art = _write_json(root, "script.json", script_data)
    md_art = _write_text(root, "script.md", _script_markdown(script_data))
    return [json_art, md_art], message


# --- scene_plan: deterministic source-clip assignment + clip slots ----------


def _scene_range_from_indexes(
    scenes: list[dict],
    positions: dict[int, int],
    start_index: object,
    end_index: object,
    video_duration: float,
) -> dict:
    """Resolve one indexed scene range to validated source timestamps."""
    if not isinstance(start_index, int) or not isinstance(end_index, int):
        raise ValueError("Claude scene plan indexes must be integers")
    if start_index not in positions or end_index not in positions:
        raise ValueError("Claude scene plan referenced an unknown scene index")
    start_pos, end_pos = positions[start_index], positions[end_index]
    if end_pos < start_pos:
        raise ValueError("Claude scene plan end scene precedes start scene")

    selected = scenes[start_pos:end_pos + 1]
    start = float(selected[0].get("start_seconds") or 0.0)
    end = float(selected[-1].get("end_seconds") or 0.0)
    if (
        not math.isfinite(start) or not math.isfinite(end)
        or start < 0 or end <= start
        or (video_duration > 0 and end > video_duration)
    ):
        raise ValueError("Claude scene plan resolved to an invalid source range")
    return {"start_seconds": start, "end_seconds": end}


def _agent_scene_assignments(
    sections: list[dict],
    scenes_doc: dict,
    assignments: list[dict],
) -> list[list[tuple[dict, str]]]:
    """Validate Claude shot choices and resolve scene IDs to source timestamps."""
    scenes = [scene for scene in (scenes_doc.get("scenes") or []) if isinstance(scene, dict)]
    if len(assignments) != len(sections):
        raise ValueError("Claude scene plan assignment count does not match script sections")
    if not scenes:
        raise ValueError("Claude scene plan cannot run without indexed scenes")

    positions: dict[int, int] = {}
    for position, scene in enumerate(scenes):
        index = scene.get("index")
        if isinstance(index, int):
            positions[index] = position

    video_duration = float(scenes_doc.get("duration_seconds") or 0.0)
    result: list[list[tuple[dict, str]]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("Claude scene plan assignment must be an object")
        raw_shots = assignment.get("shots")
        if not isinstance(raw_shots, list) or not raw_shots:
            raise ValueError("Claude scene plan assignment must contain at least one shot")
        if len(raw_shots) > SCENE_PLAN_MAX_SHOTS_PER_SECTION:
            raise ValueError("Claude scene plan assignment contains too many shots")

        shots: list[tuple[dict, str]] = []
        seen_ranges: set[tuple[float, float]] = set()
        for shot in raw_shots:
            if not isinstance(shot, dict):
                raise ValueError("Claude scene plan shot must be an object")
            source_clip = _scene_range_from_indexes(
                scenes,
                positions,
                shot.get("start_scene_index"),
                shot.get("end_scene_index"),
                video_duration,
            )
            key = (source_clip["start_seconds"], source_clip["end_seconds"])
            if key in seen_ranges:
                raise ValueError("Claude scene plan repeated the same shot within one section")
            seen_ranges.add(key)
            shots.append((source_clip, str(shot.get("rationale") or "").strip()))
        result.append(shots)
    return result


def _assign_source_clips(sections: list[dict], scenes_doc: dict) -> "list[dict | None]":
    """Map each script section to a bounded source_clip range from scenes.json.

    Proportional assignment: each section's share of total narration time maps
    to the same share of the video timeline; scenes overlapping that video range
    are merged into a single source_clip dict.  When no scenes overlap the
    mapped range the nearest scene (by midpoint) is used so every section
    always gets a clip when scenes are available.

    Returns a list of source_clip dicts (or None) parallel to sections.
    Gracefully returns all-None when scenes is empty or video_duration <= 0.
    """
    scenes = scenes_doc.get("scenes") or []
    video_duration = float(scenes_doc.get("duration_seconds") or 0.0)

    if not scenes or video_duration <= 0:
        return [None] * len(sections)

    # Total narration budget; fall back to equal weights when all are zero.
    raw_durations = [float(s.get("duration_seconds") or 0) for s in sections]
    total_narration = sum(raw_durations)
    if total_narration <= 0:
        total_narration = float(len(sections)) or 1.0
        durations = [1.0] * len(sections)
    else:
        durations = raw_durations

    result: list[dict | None] = []
    elapsed = 0.0
    for dur in durations:
        frac_start = elapsed / total_narration
        elapsed += dur
        frac_end = elapsed / total_narration

        video_start = frac_start * video_duration
        video_end = frac_end * video_duration

        # Scenes whose interval overlaps (video_start, video_end).
        overlapping = [
            s for s in scenes
            if s["end_seconds"] > video_start and s["start_seconds"] < video_end
        ]
        if not overlapping:
            # Snap to nearest scene by midpoint.
            mid = (video_start + video_end) / 2.0
            overlapping = [min(scenes, key=lambda s: abs(
                (s["start_seconds"] + s["end_seconds"]) / 2.0 - mid
            ))]

        result.append({
            "start_seconds": overlapping[0]["start_seconds"],
            "end_seconds": overlapping[-1]["end_seconds"],
        })

    return result


def _assign_source_shots(
    sections: list[dict],
    scenes_doc: dict,
) -> list[list[tuple[dict, str]]]:
    """Split deterministic section ranges into several representative scene shots."""
    broad_ranges = _assign_source_clips(sections, scenes_doc)
    scenes = sorted(
        [scene for scene in (scenes_doc.get("scenes") or []) if isinstance(scene, dict)],
        key=lambda scene: (
            float(scene.get("start_seconds") or 0.0),
            float(scene.get("end_seconds") or 0.0),
        ),
    )
    result: list[list[tuple[dict, str]]] = []
    for section, broad in zip(sections, broad_ranges):
        if not isinstance(broad, dict):
            result.append([])
            continue
        overlapping = [
            scene for scene in scenes
            if float(scene.get("end_seconds") or 0.0) > float(broad["start_seconds"])
            and float(scene.get("start_seconds") or 0.0) < float(broad["end_seconds"])
        ]
        if not overlapping:
            result.append([(broad, "")])
            continue

        duration = max(float(section.get("duration_seconds") or 0.0), 0.0)
        desired = max(
            1,
            min(
                SCENE_PLAN_MAX_SHOTS_PER_SECTION,
                math.ceil(duration / SCENE_PLAN_TARGET_SHOT_SECONDS)
                if duration > 0 else 1,
            ),
        )
        count = min(desired, len(overlapping))
        if count == 1:
            chosen = [overlapping[len(overlapping) // 2]]
        elif count == len(overlapping):
            chosen = overlapping
        else:
            indexes = [
                round(i * (len(overlapping) - 1) / (count - 1))
                for i in range(count)
            ]
            chosen = [overlapping[index] for index in indexes]

        shots = [
            ({
                "start_seconds": float(scene["start_seconds"]),
                "end_seconds": float(scene["end_seconds"]),
            }, "")
            for scene in chosen
        ]
        result.append(shots)
    return result


def _expand_scene_plan_clips(
    sections: list[dict],
    assignments: list[list[tuple[dict, str]]],
) -> tuple[list[dict], float]:
    """Expand each narration section into ordered top-level visual shot clips."""
    if len(assignments) != len(sections):
        raise ValueError("scene plan assignment count does not match script sections")

    clips: list[dict] = []
    cursor = 0.0
    for section_index, (section, shots) in enumerate(zip(sections, assignments), start=1):
        duration = max(float(section.get("duration_seconds") or 0.0), 0.0)
        shot_entries = shots or [(None, "")]
        shot_count = len(shot_entries)
        allocated = 0.0
        for shot_index, (source_clip, rationale) in enumerate(shot_entries, start=1):
            if duration > 0:
                shot_duration = (
                    duration - allocated
                    if shot_index == shot_count
                    else duration / shot_count
                )
            else:
                shot_duration = 0.0
            allocated += shot_duration
            clips.append({
                "section": section.get("title", ""),
                "section_index": section_index,
                "shot_index": shot_index,
                "shot_count": shot_count,
                "type": "narration",
                "start_seconds": cursor,
                "duration_seconds": shot_duration,
                "source_clip": source_clip,
                "notes": rationale,
            })
            cursor += shot_duration
    return clips, cursor


@register_stage("scene_plan")
def _scene_plan(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Map script sections to time-stamped clip slots.

    When scenes.json is present, each clip's source_clip is automatically
    populated by proportional assignment (_assign_source_clips), eliminating
    manual edits.  When scenes.json is absent the field stays None so the
    stage still completes and the render stage will skip with a clear message.
    """
    cfg = manifest.config
    script = _read_json(root, "script.json")
    sections = script.get("sections") or []
    scenes_doc = _read_json(root, "scenes.json")
    scene_assignments: list[list[tuple[dict, str]]]
    agent = None
    if scenes_doc.get("scenes"):
        agent = _run_reasoning_agent(
            root=root,
            manifest=manifest,
            stage="scene_plan",
            instruction=(
                f"Choose 1-{SCENE_PLAN_MAX_SHOTS_PER_SECTION} ordered visual shots for each "
                "script section. Keep exactly the same section count/order as the script. "
                "Each shot may reference one scene or a short contiguous scene range. "
                "Select only supplied scene indexes; never invent timestamps or indexes. "
                "Prefer varied, representative shots that support the narration and avoid "
                "repeating the same source range inside a section."
            ),
            context={
                "movie_title": _movie_title(cfg),
                "language": cfg.language,
                "script_sections": sections,
                "scenes": _scene_context(root),
            },
            schema=_SCENE_PLAN_AGENT_SCHEMA,
        )

    if agent is not None:
        scene_assignments = _agent_scene_assignments(
            sections,
            scenes_doc,
            list(agent.get("assignments") or []),
        )
        generator = "claude"
        plan_notes = str(agent.get("notes") or "").strip()
    else:
        scene_assignments = _assign_source_shots(sections, scenes_doc)
        generator = "deterministic"
        plan_notes = (
            "multiple representative source shots auto-assigned from scenes.json."
            if any(scene_assignments)
            else "Scene plan scaffold. Populate source shots from scenes.json after transcript/scenes stages."
        )

    clips, cursor = _expand_scene_plan_clips(sections, scene_assignments)
    resolved = sum(1 for c in clips if c["source_clip"] is not None)
    data = {
        "job_id": cfg.job_id,
        "aspect_ratio": cfg.aspect_ratio,
        "total_seconds": cursor,
        "clips": clips,
        "notes": plan_notes,
        "generator": generator,
    }
    if generator == "claude":
        msg = (
            f"scene_plan generated by Claude ({len(clips)} clips, "
            f"{round(cursor / 60, 1)} min total, {resolved} source_clips resolved)"
        )
    else:
        msg = (
            f"scene_plan written ({len(clips)} clips, {round(cursor / 60, 1)} min total, "
            f"{resolved} source_clips resolved)"
        )
    return [_write_json(root, "scene_plan.json", data)], msg



# --- tts: approved script narration with edge-tts ---------------------------

VOICE_BY_LANGUAGE = {
    "vi": "vi-VN-HoaiMyNeural",
    "en": "en-US-AriaNeural",
}
DEFAULT_TTS_VOICE = "en-US-AriaNeural"


@register_stage("tts")
def _tts(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Synthesize approved, non-empty narration into a single MP3 artifact."""
    script_path = root / "script.json"
    if not script_path.exists():
        raise SkipStage("script.json missing - generate and approve a script before TTS")
    script = _read_json(root, "script.json")
    if not script.get("approved"):
        raise SkipStage("script.json not approved (set approved=true before TTS)")

    sections = [
        {
            "title": str(section.get("title", "")),
            "narration": str(section.get("narration", "")).strip(),
        }
        for section in script.get("sections", [])
        if isinstance(section, dict) and str(section.get("narration", "")).strip()
    ]
    if not sections:
        raise SkipStage("script has no non-empty narration to synthesize")

    try:
        import edge_tts
    except ImportError as exc:
        raise SkipStage("edge-tts not installed - install the tts extra") from exc

    cfg = manifest.config
    voice = VOICE_BY_LANGUAGE.get(cfg.language, DEFAULT_TTS_VOICE)
    narration = "\n\n".join(section["narration"] for section in sections)
    audio_path = root / "narration.mp3"
    temporary_audio = root / "narration.synthesizing.mp3"
    temporary_audio.unlink(missing_ok=True)
    try:
        edge_tts.Communicate(narration, voice).save_sync(str(temporary_audio))
        if not temporary_audio.exists() or temporary_audio.stat().st_size <= 0:
            raise RuntimeError("edge-tts completed without producing narration audio")
        temporary_audio.replace(audio_path)
    finally:
        temporary_audio.unlink(missing_ok=True)

    metadata = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "engine": "edge-tts",
        "voice": voice,
        "audio_file": audio_path.name,
        "section_count": len(sections),
        "sections": sections,
    }
    return [
        _write_json(root, "voice.json", metadata),
        Artifact(name=audio_path.name, path=audio_path, status="ready"),
    ], f"synthesized narration.mp3 from {len(sections)} sections"


# --- alignment: fit captions to the synthesized narration ------------------

# Matches an SRT timestamp like 00:01:02,500 (comma or dot as the ms separator).
_SRT_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})[,.](\d{1,3})")


def _parse_srt_timestamp(value: str) -> float | None:
    """Parse a single SRT timestamp into seconds; return None if unparseable."""
    match = _SRT_TIME_RE.search(value)
    if not match:
        return None
    hours, minutes, seconds, millis = match.groups()
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000
    )


def _parse_srt(text: str) -> list[dict]:
    """Parse SRT text into cues. Blocks without a valid timing line are skipped."""
    cues: list[dict] = []
    for block in re.split(r"\r?\n\r?\n", text.strip()):
        lines = block.splitlines()
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        start_str, _, end_str = lines[timing_index].partition("-->")
        start = _parse_srt_timestamp(start_str)
        end = _parse_srt_timestamp(end_str)
        if start is None or end is None:
            continue
        cue_text = "\n".join(lines[timing_index + 1:]).strip()
        cues.append({"start_seconds": start, "end_seconds": end, "text": cue_text})
    return cues


def _align_cues(cues: list[dict], duration: float) -> list[dict]:
    """Clamp cues to [0, duration] and drop any with no positive length.

    Preserves cue order and re-indexes sequentially so every output cue stays
    strictly inside the narration bounds.
    """
    aligned: list[dict] = []
    for cue in cues:
        start = _clamp(cue.get("start_seconds", 0.0), duration)
        end = _clamp(cue.get("end_seconds", start), duration)
        if end <= start:
            continue
        aligned.append({
            "index": len(aligned) + 1,
            "start_seconds": start,
            "end_seconds": end,
            "text": str(cue.get("text", "")),
        })
    return aligned


def _narration_section_cues(sections: list[object], duration: float) -> list[dict]:
    """Build deterministic, contiguous cues from narration section text."""
    narration = [
        str(section.get("narration", "")).strip()
        for section in sections
        if isinstance(section, dict) and str(section.get("narration", "")).strip()
    ]
    if not narration:
        return []
    word_counts = [len(text.split()) for text in narration]
    total_words = sum(word_counts)
    if total_words <= 0:
        return []
    elapsed_words = 0
    cues: list[dict] = []
    for index, (text, word_count) in enumerate(zip(narration, word_counts), start=1):
        start = duration * elapsed_words / total_words
        elapsed_words += word_count
        end = duration if index == len(narration) else duration * elapsed_words / total_words
        cues.append({
            "index": index,
            "start_seconds": start,
            "end_seconds": end,
            "text": text,
        })
    return cues


def _script_fallback_cues(script: dict, duration: float) -> list[dict]:
    """Build deterministic cues from approved script narration for legacy jobs."""
    if not script.get("approved"):
        return []
    return _narration_section_cues(list(script.get("sections") or []), duration)


@register_stage("alignment")
def _alignment(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Build narration subtitles aligned to the synthesized audio, locally.

    Consumes voice.json + narration.mp3 and measures narration duration with
    ffprobe. For current jobs, subtitle text comes from voice.json.sections —
    the exact narration text sent to TTS. Approved script.json is the legacy
    fallback; source captions.srt is used only when neither narration source is
    available. Writes alignment.json + aligned.srt and never publishes.
    """
    cfg = manifest.config
    voice = _read_json(root, "voice.json")
    if not voice:
        raise SkipStage("voice.json missing - run tts before alignment")
    audio_name = str(voice.get("audio_file") or "narration.mp3")
    audio_path = root / audio_name
    if not audio_path.exists():
        raise SkipStage(f"{audio_name} missing - run tts before alignment")
    captions_path = root / "captions.srt"

    duration = _probe_duration_seconds(audio_path)
    if duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to align narration")
    if duration <= 0:
        raise SkipStage("narration has no positive duration - cannot align")

    aligned = _narration_section_cues(list(voice.get("sections") or []), duration)
    dropped = 0
    cue_source = "voice.json"

    if not aligned:
        aligned = _script_fallback_cues(_read_json(root, "script.json"), duration)
        if aligned:
            cue_source = "script.json"

    if not aligned:
        if not captions_path.exists():
            raise SkipStage(
                "no narration sections available and captions.srt missing - "
                "run TTS/script or transcript before alignment"
            )
        cues = _parse_srt(captions_path.read_text(encoding="utf-8"))
        aligned = _align_cues(cues, duration)
        dropped = len(cues) - len(aligned)
        cue_source = "captions.srt"

    data = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "audio_file": audio_name,
        "narration_seconds": duration,
        "source_captions": cue_source,
        "cue_source": cue_source,
        "cue_count": len(aligned),
        "dropped_cues": dropped,
        "cues": aligned,
    }
    srt = "\n".join(
        f"{cue['index']}\n{_srt_timestamp(cue['start_seconds'])} --> "
        f"{_srt_timestamp(cue['end_seconds'])}\n{cue['text']}\n"
        for cue in aligned
    )
    artifacts = [
        _write_json(root, "alignment.json", data),
        _write_text(root, "aligned.srt", srt),
    ]
    source_note = (
        " from voice" if cue_source == "voice.json"
        else " from script" if cue_source == "script.json"
        else ""
    )
    msg = f"aligned {len(aligned)} cues within narration bounds (dropped {dropped}){source_note}"
    return artifacts, msg


# --- render: deterministic local FFmpeg assembly ----------------------------


def _render_source_ranges(clips: list[object], source_duration: float) -> list[dict]:
    """Validate ordered, explicit ranges from a scene plan for one owned video."""
    ranges: list[dict] = []
    for index, clip in enumerate(clips):
        if not isinstance(clip, dict):
            raise SkipStage(f"scene_plan clip {index} must be an object")
        source_clip = clip.get("source_clip")
        if not isinstance(source_clip, dict):
            raise SkipStage(
                f"scene_plan clip {index} has no source_clip range - populate it before render"
            )
        start = source_clip.get("start_seconds")
        end = source_clip.get("end_seconds")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise SkipStage(f"scene_plan clip {index} has non-numeric source_clip timestamps")
        start, end = float(start), float(end)
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise SkipStage(f"scene_plan clip {index} has an invalid source_clip range")
        if end > source_duration:
            raise SkipStage(
                f"scene_plan clip {index} ends at {end:.3f}s beyond source duration "
                f"{source_duration:.3f}s"
            )
        source_seconds = end - start
        target = clip.get("duration_seconds")
        if (
            isinstance(target, (int, float))
            and math.isfinite(float(target))
            and float(target) > 0
        ):
            target_seconds = float(target)
        else:
            target_seconds = source_seconds
        item = {
            "index": index,
            "section": str(clip.get("section", "")),
            "type": str(clip.get("type", "")),
            "start_seconds": start,
            "end_seconds": end,
            "source_seconds": source_seconds,
            "duration_seconds": target_seconds,
        }
        for key in ("section_index", "shot_index", "shot_count"):
            if key in clip:
                item[key] = clip[key]
        ranges.append(item)
    if not ranges:
        raise SkipStage("scene_plan.json has no clips to render")
    return ranges


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """Return an absolute, FFmpeg-filter-safe path for the subtitles filter."""
    return str(path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", r"\'")


@register_stage("render")
def _render(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Render selected ranges of the configured source video with narration and SRT."""
    cfg = manifest.config
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to render")
    source_path = Path(cfg.source_video)
    if not source_path.exists():
        raise SkipStage(f"source_video not found: {source_path}")

    narration_path = root / "narration.mp3"
    if not narration_path.exists():
        raise SkipStage("narration.mp3 missing - run tts before render")
    subtitles_path = root / "aligned.srt"
    if not subtitles_path.exists():
        raise SkipStage("aligned.srt missing - run alignment before render")
    scene_plan_path = root / "scene_plan.json"
    if not scene_plan_path.exists():
        raise SkipStage("scene_plan.json missing - run scene_plan before render")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SkipStage("ffmpeg not on PATH - install FFmpeg to render")
    source_duration = _probe_duration_seconds(source_path)
    if source_duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to render")
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise SkipStage("source_video has no positive duration - cannot render")

    plan = _read_json(root, "scene_plan.json")
    ratio = plan.get("aspect_ratio") or cfg.aspect_ratio
    if ratio not in RENDER_CANVASES:
        raise SkipStage(f"unsupported aspect_ratio {ratio!r}")
    ranges = _render_source_ranges(plan.get("clips") or [], source_duration)
    width, height = RENDER_CANVASES[ratio]

    filter_parts: list[str] = []
    concat_inputs: list[str] = []
    for item in ranges:
        index = item["index"]
        start = item["start_seconds"]
        end = item["end_seconds"]
        source_seconds = item["source_seconds"]
        target_seconds = item["duration_seconds"]
        segment = f"[0:v]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS"
        # When a clip must play longer than its trimmed source range, normalise
        # the frame rate then loop the trimmed frames until they cover the target
        # duration and trim to the exact length. This keeps the composed video at
        # least as long as the narration so -shortest clamps the output to the
        # narration track instead of truncating the video (audio/video drift).
        if target_seconds > source_seconds:
            size = max(1, round(source_seconds * RENDER_FRAME_RATE))
            loops = math.ceil(target_seconds / source_seconds) - 1
            segment += (
                f",fps={RENDER_FRAME_RATE},"
                f"loop=loop={loops}:size={size}:start=0,"
                f"setpts=N/{RENDER_FRAME_RATE}/TB,"
                f"trim=end={target_seconds:.6f},setpts=PTS-STARTPTS"
            )
        elif target_seconds < source_seconds:
            segment += f",trim=end={target_seconds:.6f},setpts=PTS-STARTPTS"
        filter_parts.append(
            f"{segment},"
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            f"fps={RENDER_FRAME_RATE},format=yuv420p[v{index}]"
        )
        concat_inputs.append(f"[v{index}]")
    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(ranges)}:v=1:a=0[video]")
    if subtitles_path.stat().st_size:
        filter_parts.append(
            f"[video]subtitles='{_escape_ffmpeg_filter_path(subtitles_path)}'[rendered]"
        )
    else:
        filter_parts.append("[video]null[rendered]")
    filter_complex = ";".join(filter_parts)

    temporary_path = root / "final.rendering.mp4"
    final_path = root / "final.mp4"
    temporary_path.unlink(missing_ok=True)

    # Narration is the master track: probe its duration up front and hard-cap the
    # muxed output to it with -t. -shortest alone does not cut the filtered video
    # exactly at the audio EOF - with the concat/loop filtergraph the video
    # overruns the narration by the filter's buffered tail (~0.7s on the smoke
    # job), which exceeds RENDER_DURATION_DRIFT_SECONDS. -t pins both streams to
    # the narration length; -shortest stays as a secondary guard.
    narration_duration = _probe_duration_seconds(narration_path)
    if narration_duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to render")
    if not math.isfinite(narration_duration) or narration_duration <= 0:
        raise SkipStage("narration.mp3 has no positive duration - cannot render")

    command = [
        ffmpeg, "-y", "-i", str(source_path), "-i", str(narration_path),
        "-filter_complex", filter_complex,
        "-map", "[rendered]", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-r", str(RENDER_FRAME_RATE),
        "-t", f"{narration_duration:.6f}",
        "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
        if not temporary_path.exists():
            raise RuntimeError("ffmpeg completed without producing final output")
        output_duration = _probe_duration_seconds(temporary_path)
        if output_duration is None:
            raise RuntimeError("ffprobe unavailable while verifying rendered output")
        if output_duration <= 0:
            raise RuntimeError("rendered output has no positive duration")
        if abs(output_duration - narration_duration) > RENDER_DURATION_DRIFT_SECONDS:
            raise RuntimeError(
                f"rendered audio/video duration drift exceeds {RENDER_DURATION_DRIFT_SECONDS}s"
            )
        temporary_path.replace(final_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    render_data = {
        "job_id": cfg.job_id,
        "source_video": str(source_path),
        "narration_file": narration_path.name,
        "subtitle_file": subtitles_path.name,
        "output_file": final_path.name,
        "aspect_ratio": ratio,
        "width": width,
        "height": height,
        "frame_rate": RENDER_FRAME_RATE,
        "video_codec": "libx264",
        "audio_codec": "aac",
        "framing_policy": "scale_pad",
        "source_duration_seconds": source_duration,
        "narration_duration_seconds": narration_duration,
        "output_duration_seconds": output_duration,
        "duration_drift_seconds": abs(output_duration - narration_duration),
        "clips": ranges,
    }
    artifacts = [
        _write_json(root, "render.json", render_data),
        Artifact(name=final_path.name, path=final_path, status="ready"),
    ]
    return artifacts, f"rendered {len(ranges)} clips to final.mp4"


# --- qa: validate rendered final.mp4 against configured targets -------------


@register_stage("qa")
def _qa(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Validate the rendered final.mp4 against configured targets.

    Self-skips when final.mp4 is absent or ffprobe is unavailable.
    Writes qa.json recording every check, its measured value, and any
    failure message. Raises RuntimeError when any check fails (after
    writing the report) so the job is resumable from this stage.
    """
    final_path = root / "final.mp4"
    if not final_path.exists():
        raise SkipStage("final.mp4 missing - run render before qa")

    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to qa")

    render = _read_json(root, "render.json")
    expected_width: int | None = render.get("width")
    expected_height: int | None = render.get("height")
    expected_frame_rate: int = int(render.get("frame_rate", RENDER_FRAME_RATE))
    narration_duration: float | None = render.get("narration_duration_seconds")

    proc = subprocess.run(
        [ffprobe_bin, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(final_path)],
        capture_output=True, text=True,
    )

    checks: list[dict] = []
    failures: list[str] = []

    def _record(name: str, value: object, passed: bool, message: str = "") -> None:
        checks.append({"check": name, "value": value, "passed": passed, "message": message})
        if not passed:
            failures.append(message or name)

    if proc.returncode != 0:
        err = proc.stderr.strip()[:200]
        _record("ffprobe_decode", None, False, f"ffprobe failed: {err}")
        _write_json(root, "qa.json", {
            "job_id": manifest.config.job_id, "output_file": "final.mp4",
            "passed": False, "checks": checks,
        })
        raise RuntimeError(f"ffprobe failed: {err}")

    probe = json.loads(proc.stdout)
    fmt = probe.get("format", {})
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = next((s for s in streams if s.get("codec_type") == "audio"), {})

    _record("ffprobe_decode", True, True)

    video_codec = video.get("codec_name")
    ok = video_codec == "h264"
    _record("video_codec", video_codec, ok, "" if ok else f"expected h264, got {video_codec!r}")

    audio_codec = audio.get("codec_name")
    ok = audio_codec == "aac"
    _record("audio_codec", audio_codec, ok, "" if ok else f"expected aac, got {audio_codec!r}")

    actual_width = video.get("width")
    actual_height = video.get("height")
    if expected_width is not None and expected_height is not None:
        ok = actual_width == expected_width and actual_height == expected_height
        _record(
            "canvas_dimensions",
            {"width": actual_width, "height": actual_height},
            ok,
            "" if ok else f"expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}",
        )
    else:
        _record("canvas_dimensions", {"width": actual_width, "height": actual_height}, True)

    r_frame_rate = video.get("r_frame_rate", "")
    try:
        num_s, den_s = r_frame_rate.split("/")
        den = int(den_s)
        actual_frame_rate = int(num_s) // den if den else None
    except (ValueError, AttributeError):
        actual_frame_rate = None
    ok = actual_frame_rate == expected_frame_rate
    _record(
        "frame_rate",
        actual_frame_rate,
        ok,
        "" if ok else f"expected {expected_frame_rate} fps, got {actual_frame_rate!r}",
    )

    raw_duration = fmt.get("duration")
    try:
        output_duration = float(raw_duration) if raw_duration is not None else None
    except (TypeError, ValueError):
        output_duration = None
    ok = output_duration is not None and output_duration > 0
    _record(
        "positive_duration",
        output_duration,
        ok,
        "" if ok else f"expected positive duration, got {output_duration!r}",
    )

    if narration_duration is not None and output_duration is not None:
        drift = abs(output_duration - narration_duration)
        ok = drift <= RENDER_DURATION_DRIFT_SECONDS
        _record(
            "duration_drift",
            drift,
            ok,
            "" if ok else f"output/narration drift {drift:.3f}s exceeds limit {RENDER_DURATION_DRIFT_SECONDS}s",
        )

    alignment = _read_json(root, "alignment.json")
    if alignment:
        nar_dur = narration_duration if narration_duration is not None else (output_duration or 0.0)
        cues = alignment.get("cues") or []
        bad: list[object] = [
            cue.get("index", "?")
            for cue in cues
            if isinstance(cue, dict) and float(cue.get("end_seconds", 0.0)) > nar_dur
        ]
        ok = not bad
        _record(
            "alignment_cue_bounds",
            {"cue_count": len(cues), "out_of_bounds": bad},
            ok,
            "" if ok else f"cues exceed narration duration: indices {bad}",
        )

    artifact = _write_json(root, "qa.json", {
        "job_id": manifest.config.job_id,
        "output_file": "final.mp4",
        "passed": not failures,
        "checks": checks,
    })
    if failures:
        raise RuntimeError("; ".join(failures))
    return [artifact], f"qa passed — {len(checks)} checks OK"


# --- metadata: YouTube/platform draft from script + outline -----------------

@register_stage("metadata")
def _metadata(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Draft platform metadata (title, description, tags) from script + outline."""
    cfg = manifest.config
    script = _read_json(root, "script.json")
    outline = _read_json(root, "outline.json")

    # Prefer script sections (they inherit outline titles); fall back to outline.
    raw_sections = script.get("sections") or outline.get("sections") or []
    section_titles = [
        s["title"] if isinstance(s, dict) else s for s in raw_sections
    ]
    description_lines = [
        f"Review / Recap phim – {cfg.job_id}",
        "",
        "Nội dung video:",
        *[f"• {t}" for t in section_titles],
        "",
        "---",
        "⚠️ Bản nháp – chỉnh sửa trước khi publish.",
    ]
    tags = (
        ["review", "recap", "phim", cfg.language]
        + [t.lower().replace(" ", "-") for t in section_titles[:5]]
    )
    meta = {
        "job_id": cfg.job_id,
        "title": f"[{cfg.language.upper()}] Review/Recap – {cfg.job_id}",
        "description": "\n".join(description_lines),
        "tags": tags,
        "language": cfg.language,
        "aspect_ratio": cfg.aspect_ratio,
        "approved": False,
        "publish": False,
        "notes": "Edit title/description/tags, set approved=true before publish stage.",
    }
    return [_write_json(root, "youtube_metadata.json", meta)], "draft metadata written"


# --- thumbnail: deterministic clean-frame candidates ------------------------

def _thumbnail_timestamps(root: Path, source_duration: float) -> list[float]:
    """Pick representative source-frame timestamps from the scene plan."""
    plan = _read_json(root, "scene_plan.json")
    midpoints: list[float] = []
    for clip in plan.get("clips") or []:
        source_clip = clip.get("source_clip") if isinstance(clip, dict) else None
        if not isinstance(source_clip, dict):
            continue
        try:
            start = float(source_clip.get("start_seconds"))
            end = float(source_clip.get("end_seconds"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(start) and math.isfinite(end) and end > start:
            midpoints.append((start + end) / 2.0)

    fractions = [(index + 1) / (THUMBNAIL_COUNT + 1) for index in range(THUMBNAIL_COUNT)]
    chosen: list[float] = []
    if midpoints:
        for fraction in fractions:
            index = min(int(fraction * len(midpoints)), len(midpoints) - 1)
            chosen.append(midpoints[index])

    chosen.extend(source_duration * fraction for fraction in fractions)
    upper = max(source_duration - 0.05, 0.0)
    unique: list[float] = []
    for value in chosen:
        bounded = round(max(0.0, min(float(value), upper)), 3)
        if bounded not in unique:
            unique.append(bounded)
        if len(unique) == THUMBNAIL_COUNT:
            break
    return unique or [0.0]


@register_stage("thumbnail")
def _thumbnail(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Extract three clean 1280x720 source frames and select a default primary."""
    cfg = manifest.config
    source_path = Path(cfg.source_video) if cfg.source_video else root / "final.mp4"
    if not source_path.exists():
        fallback = root / "final.mp4"
        if not fallback.exists():
            raise SkipStage("source video/final.mp4 missing - run render before thumbnail")
        source_path = fallback

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SkipStage("ffmpeg not on PATH - install FFmpeg to generate thumbnails")
    duration = _probe_duration_seconds(source_path)
    if duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to generate thumbnails")
    if not math.isfinite(duration) or duration <= 0:
        raise SkipStage("thumbnail source has no positive duration")

    width, height = THUMBNAIL_SIZE
    timestamps = _thumbnail_timestamps(root, duration)
    candidates: list[dict] = []
    artifacts: list[Artifact] = []
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )

    for index, timestamp in enumerate(timestamps, start=1):
        output = root / f"thumbnail-{index}.jpg"
        subprocess.run(
            [
                ffmpeg, "-y", "-ss", f"{timestamp:.3f}", "-i", str(source_path),
                "-frames:v", "1", "-vf", video_filter, "-q:v", "2", str(output),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        if not output.exists() or output.stat().st_size <= 0:
            raise RuntimeError(f"ffmpeg did not produce {output.name}")
        artifacts.append(Artifact(name=output.name, path=output, status="ready"))
        candidates.append({
            "index": index,
            "file": output.name,
            "source_seconds": timestamp,
            "width": width,
            "height": height,
        })

    primary_candidate = candidates[len(candidates) // 2]["file"]
    primary_path = root / "thumbnail.jpg"
    shutil.copyfile(root / primary_candidate, primary_path)
    artifacts.append(Artifact(name=primary_path.name, path=primary_path, status="ready"))

    metadata = _read_json(root, "youtube_metadata.json")
    data = {
        "job_id": cfg.job_id,
        "source_file": source_path.name,
        "source_duration_seconds": duration,
        "title_hint": str(metadata.get("title") or ""),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "primary_thumbnail": primary_path.name,
        "primary_candidate": primary_candidate,
        "width": width,
        "height": height,
    }
    artifacts.insert(0, _write_json(root, "thumbnails.json", data))
    return artifacts, f"generated {len(candidates)} thumbnail candidates"


def select_thumbnail(root: Path, candidate_name: str) -> dict:
    """Select one generated candidate as thumbnail.jpg and invalidate publish."""
    thumbnails = _read_json(root, "thumbnails.json")
    if not thumbnails:
        raise FileNotFoundError("thumbnails.json missing - run thumbnail stage first")

    candidates = {
        str(candidate.get("file"))
        for candidate in thumbnails.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("file")
    }
    if candidate_name not in candidates:
        raise ValueError("thumbnail candidate is not part of this job")

    source = root / candidate_name
    if source.resolve().parent != root.resolve():
        raise ValueError("thumbnail candidate path escapes the job directory")
    if not source.is_file():
        raise FileNotFoundError(f"thumbnail candidate missing: {candidate_name}")

    target = root / "thumbnail.jpg"
    temporary = root / "thumbnail.jpg.tmp"
    shutil.copyfile(source, temporary)
    temporary.replace(target)

    thumbnails["primary_candidate"] = candidate_name
    thumbnails["primary_thumbnail"] = target.name
    _write_json(root, "thumbnails.json", thumbnails)

    # Selection changes the publish handoff, but thumbnail generation remains ready.
    invalidate_downstream(root, "thumbnail")
    return thumbnails


# --- publish: gate + publish record -----------------------------------------

@register_stage("publish")
def _publish(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Gate stage: verify approvals, then write publish_record.json.

    Never pushes to any external service. The publish_record.json is the
    handoff artefact for a separate upload script / operator action.
    Raises SkipStage when required approvals are missing.
    """
    script = _read_json(root, "script.json")
    meta = _read_json(root, "youtube_metadata.json")
    qa = _read_json(root, "qa.json")
    final_path = root / "final.mp4"

    blockers: list[str] = []
    if not script:
        blockers.append("script.json missing")
    elif not script.get("approved"):
        blockers.append("script.json not approved (set approved=true)")
    if not meta:
        blockers.append("youtube_metadata.json missing")
    elif not meta.get("approved"):
        blockers.append("youtube_metadata.json not approved (set approved=true)")
    if not final_path.is_file() or final_path.stat().st_size <= 0:
        blockers.append("final.mp4 missing or empty")
    if not qa:
        blockers.append("qa.json missing")
    elif not qa.get("passed"):
        blockers.append("qa.json not passed")

    if blockers:
        raise SkipStage("publish blocked: " + "; ".join(blockers))

    cfg = manifest.config
    thumbnails = _read_json(root, "thumbnails.json")
    record = {
        "job_id": cfg.job_id,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "tags": meta.get("tags", []),
        "language": cfg.language,
        "aspect_ratio": cfg.aspect_ratio,
        "video_file": final_path.name,
        "qa_file": "qa.json",
        "qa_passed": True,
        "thumbnail_file": thumbnails.get("primary_thumbnail", ""),
        "thumbnail_candidates": [
            candidate.get("file")
            for candidate in thumbnails.get("candidates", [])
            if isinstance(candidate, dict) and candidate.get("file")
        ],
        "publish_ready": True,
        "notes": "All approvals complete. Hand this file to your upload script.",
    }
    return [_write_json(root, "publish_record.json", record)], "publish record written — ready for upload"


# --- runner -----------------------------------------------------------------

def _skip_reason(stage_name: str) -> str:
    if stage_name in MEDIA_STAGES:
        return "needs source video/media - skipped (not implemented)"
    return "no handler - skipped"


def run_job(root: Path, *, force: bool = False, until: str | None = None) -> JobManifest:
    """Walk stages in order, run any registered handler, and persist the manifest
    after every stage so the job is resumable. Stages without a handler are
    honestly marked "skipped". A handler may raise SkipStage for a controlled
    skip; any other exception marks it "failed" and stops the run unless
    force=True (which continues past failures without re-running ready stages)."""
    if until and until not in STAGES:
        raise ValueError(f"unknown stage: {until!r}")
    manifest = load_manifest(root)
    for stage in manifest.stages:
        # Only ready stages are final; skipped stages are recoverable and retryable.
        permanently_done = stage.status == "ready"
        if permanently_done:
            if until and stage.stage == until:
                break
            continue
        handler = STAGE_HANDLERS.get(stage.stage)
        if handler is None:
            stage.mark("skipped", _skip_reason(stage.stage))
            save_manifest(root, manifest)
        else:
            stage.mark("running")
            save_manifest(root, manifest)
            try:
                artifacts, message = handler(root, manifest)
            except SkipStage as exc:
                stage.mark("skipped", exc.reason)
            except Exception as exc:
                stage.mark("failed", str(exc))
                save_manifest(root, manifest)
                if not force:
                    return manifest
            else:
                stage.mark("ready", message, artifacts)
            save_manifest(root, manifest)
        if until and stage.stage == until:
            break
    return manifest


def job_status(root: Path) -> dict:
    """Return a serialisable summary for CLI/API callers."""
    manifest = load_manifest(root)
    stages = [stage.model_dump(mode="json") for stage in manifest.stages]
    # Tally stages by status. Zero-fill every known status so the shape is
    # stable for callers (the CLI `status` command prints this verbatim) and
    # sum(counts.values()) always equals the stage count.
    counts = {status: 0 for status in ("pending", "running", "ready", "failed", "skipped")}
    for stage in stages:
        counts[stage["status"]] = counts.get(stage["status"], 0) + 1
    return {
        "job_id": manifest.config.job_id,
        "complete": manifest.is_complete,
        "counts": counts,
        "stages": stages,
    }


# --- job discovery / approval helpers (used by the CLI and the web UI) -------

METADATA_NAME = "youtube_metadata.json"


def list_jobs(jobs_root: Path) -> list[dict]:
    """List every job (a directory holding a manifest.json) under jobs_root.

    Returns a serialisable summary per job for CLI/API callers. Directories
    without a readable manifest are skipped rather than raising, so a stray
    folder never breaks the listing.
    """
    jobs_root = Path(jobs_root)
    jobs: list[dict] = []
    if not jobs_root.exists():
        return jobs
    for entry in sorted(jobs_root.iterdir()):
        if not entry.is_dir() or not manifest_path(entry).exists():
            continue
        try:
            manifest = load_manifest(entry)
        except Exception:
            continue
        ready = sum(1 for stage in manifest.stages if stage.status == "ready")
        jobs.append({
            "job_id": entry.name,
            "language": manifest.config.language,
            "aspect_ratio": manifest.config.aspect_ratio,
            "target_minutes": manifest.config.target_minutes,
            "complete": manifest.is_complete,
            "stage_count": len(manifest.stages),
            "ready_count": ready,
        })
    return jobs


def approve_metadata(root: Path) -> dict:
    """Explicitly approve a job's YouTube metadata (set approved=true).

    This is a deliberate, human-triggered lift of the publish gate: it reads
    youtube_metadata.json, sets approved=true, writes it back, and returns the
    updated document. It never runs the publish stage and never uploads
    anything. Raises FileNotFoundError when metadata has not been drafted yet
    (the metadata stage must have run first).
    """
    path = root / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"{METADATA_NAME} missing - run the metadata stage first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["approved"] = True
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def approve_script(root: Path) -> dict:
    """Explicitly approve the current script and keep script.md synchronized.

    Approval only lifts the TTS gate; it does not run any downstream stage.
    Script edits already invalidate every derived artifact before this point.
    """
    path = root / "script.json"
    if not path.exists():
        raise FileNotFoundError("script.json missing - run the script stage first")
    script = json.loads(path.read_text(encoding="utf-8"))
    script["approved"] = True
    _write_json(root, "script.json", script)
    _write_text(root, "script.md", _script_markdown(script))
    return script


def update_script(root: Path, fields: dict) -> dict:
    """Update the script, clear approval, and invalidate every derived stage."""
    path = root / "script.json"
    if not path.exists():
        raise FileNotFoundError("script.json missing - run the script stage first")
    script = json.loads(path.read_text(encoding="utf-8"))
    if "sections" in fields and fields["sections"] is not None:
        if not isinstance(fields["sections"], list):
            raise ValueError("sections must be a JSON array")
        script["sections"] = fields["sections"]
    if "notes" in fields and fields["notes"] is not None:
        script["notes"] = str(fields["notes"])
    script["approved"] = False
    path.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_text(root, "script.md", _script_markdown(script))
    invalidate_downstream(root, "script")
    return script


def update_metadata(root: Path, fields: dict) -> dict:
    """Update editable metadata fields, clear approval, and invalidate publish.

    Editing clears approval (approved=false) so a human must re-approve the
    edited draft - approval always follows the final edit, it can never be
    carried over from a previous version. Unknown fields are ignored. Raises
    FileNotFoundError when metadata has not been drafted yet.
    """
    path = root / METADATA_NAME
    if not path.exists():
        raise FileNotFoundError(f"{METADATA_NAME} missing - run the metadata stage first")
    meta = json.loads(path.read_text(encoding="utf-8"))
    for key in ("title", "description", "tags"):
        if key in fields and fields[key] is not None:
            meta[key] = fields[key]
    meta["approved"] = False
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    invalidate_downstream(root, "metadata")
    return meta
