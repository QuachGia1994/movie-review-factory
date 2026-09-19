import json
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from .models import Artifact, JobConfig, JobManifest, StageResult

STAGES = ("ingest","research","transcript","scenes","outline","script","scene_plan","tts","alignment","render","qa","metadata","publish")

# Stages that still need source video / media or external models and have no
# real handler yet. These are honestly marked "skipped" (never faked).
MEDIA_STAGES = frozenset({"transcript", "scenes", "tts", "alignment", "render", "qa"})

MANIFEST_NAME = "manifest.json"


# --- manifest I/O -----------------------------------------------------------

def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def load_manifest(root: Path) -> JobManifest:
    return JobManifest.model_validate_json(manifest_path(root).read_text(encoding="utf-8"))


def save_manifest(root: Path, manifest: JobManifest) -> Path:
    path = manifest_path(root)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def create_job(root: Path, config: JobConfig) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    manifest = JobManifest(config=config, stages=[StageResult(stage=s, status="pending") for s in STAGES])
    return save_manifest(root, manifest)


def validate_job(root: Path) -> list[str]:
    path = manifest_path(root)
    if not path.exists():
        return ["manifest.json is missing"]
    try:
        manifest = JobManifest.model_validate_json(path.read_text(encoding="utf-8"))
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
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return Artifact(name=name, path=out, status="ready")


def _write_text(root: Path, name: str, text: str) -> Artifact:
    out = root / name
    out.write_text(text, encoding="utf-8")
    return Artifact(name=name, path=out, status="ready")


def _read_json(root: Path, name: str) -> dict:
    """Read a JSON artifact; return {} if it does not exist yet."""
    p = root / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


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


# --- research: scaffold enriched with ingest data ---------------------------

@register_stage("research")
def _research(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Build a research scaffold, enriched with ingest data when available."""
    cfg = manifest.config
    ingest = _read_json(root, "ingest.json")
    video_info = {k: ingest[k] for k in ("video_codec", "width", "height", "audio_codec", "has_audio") if k in ingest}
    data = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "target_minutes": cfg.target_minutes,
        "aspect_ratio": cfg.aspect_ratio,
        "source_video": str(cfg.source_video) if cfg.source_video else None,
        "duration_seconds": ingest.get("duration_seconds"),
        "video_info": video_info,
        "sources": [],
        "open_questions": [
            "Xác nhận nguồn phim và quyền sử dụng.",
            "Chốt góc nhìn / định hướng bài review.",
        ],
        "notes": "Research scaffold - fill sources and open_questions before outline.",
    }
    return [_write_json(root, "research.json", data)], "research scaffold written"


# --- outline: time-budgeted sections from research --------------------------

@register_stage("outline")
def _outline(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Generate a default review outline with per-section time budgets.

    Hook and CTA are fixed short bookends; the four body sections share the
    remaining time proportionally.
    """
    cfg = manifest.config
    research = _read_json(root, "research.json")
    target_min = float(research.get("target_minutes") or cfg.target_minutes)

    hook_min, cta_min = 0.5, 0.5
    body_min = max(target_min - hook_min - cta_min, 1.0)
    body_titles = [
        "Bối cảnh & tiền đề",
        "Diễn biến chính (hạn chế spoiler)",
        "Điểm nhấn phân tích",
        "Đánh giá & kết luận",
    ]
    per_body = round(body_min / len(body_titles), 1)
    sections = [
        {"title": "Mở đầu / hook", "budget_minutes": hook_min},
        *[{"title": t, "budget_minutes": per_body} for t in body_titles],
        {"title": "Call to action", "budget_minutes": cta_min},
    ]
    outline = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "target_minutes": target_min,
        "sections": sections,
        "notes": "Outline scaffold - reorder or split sections as needed before script.",
    }
    return [_write_json(root, "outline.json", outline)], "outline scaffold written"


# --- script: narration scaffold from outline --------------------------------

@register_stage("script")
def _script(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Turn the outline into a script scaffold (script.json + script.md).

    Reads outline.json for section titles and time budgets; falls back to
    sensible defaults when outline is missing. Marks approved=false — the
    script must be approved before TTS/render stages consume it.
    """
    cfg = manifest.config
    outline = _read_json(root, "outline.json")
    raw_sections = outline.get("sections") or [
        {"title": "Mở đầu / hook", "budget_minutes": 0.5},
        {"title": "Nội dung chính", "budget_minutes": cfg.target_minutes - 1.0},
        {"title": "Kết luận & CTA", "budget_minutes": 0.5},
    ]
    # Normalise: sections may arrive as plain strings (legacy) or dicts.
    sections = []
    for s in raw_sections:
        if isinstance(s, str):
            sections.append({"title": s, "budget_minutes": 0, "narration": "", "duration_seconds": 0})
        else:
            budget = float(s.get("budget_minutes") or 0)
            sections.append({
                "title": s.get("title", ""),
                "budget_minutes": budget,
                "narration": "",
                "duration_seconds": round(budget * 60),
            })

    script_data = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "target_minutes": cfg.target_minutes,
        "approval_required": True,
        "approved": False,
        "sections": sections,
        "notes": "Fill narration for each section, set approved=true when ready for TTS.",
    }
    json_art = _write_json(root, "script.json", script_data)

    # Human-readable markdown mirror for easy editing.
    md_lines = [
        f"# Script: {cfg.job_id}",
        "",
        f"Language: `{cfg.language}`  |  Target: `{cfg.target_minutes} min`  |  **approved: false**",
        "",
        "> Fill each section below, then set `approved: true` in script.json.",
        "",
    ]
    for sec in sections:
        budget_note = f" _{sec['budget_minutes']} min_" if sec["budget_minutes"] else ""
        md_lines += [f"## {sec['title']}{budget_note}", "", "_(narration here)_", ""]
    md_path = root / "script.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    md_art = Artifact(name="script.md", path=md_path, status="ready")

    return [json_art, md_art], "script scaffold written (set approved=true before TTS)"


# --- scene_plan: clip slots from script sections ----------------------------

@register_stage("scene_plan")
def _scene_plan(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Map script sections to time-stamped clip slots.

    Uses script section budgets to pre-populate start/duration for each
    narration block. Real source_clip timestamps come from the scenes stage.
    """
    cfg = manifest.config
    script = _read_json(root, "script.json")
    sections = script.get("sections") or []
    clips = []
    cursor = 0.0
    for sec in sections:
        dur = float(sec.get("duration_seconds") or 0)
        clips.append({
            "section": sec.get("title", ""),
            "type": "narration",        # narration | broll | overlay
            "start_seconds": cursor,
            "duration_seconds": dur,
            "source_clip": None,        # fill from scenes.json when available
            "notes": "",
        })
        cursor += dur
    data = {
        "job_id": cfg.job_id,
        "aspect_ratio": cfg.aspect_ratio,
        "total_seconds": cursor,
        "clips": clips,
        "notes": "Scene plan scaffold. Populate source_clip from scenes.json after transcript/scenes stages.",
    }
    msg = f"scene_plan written ({len(clips)} clips, {round(cursor / 60, 1)} min total)"
    return [_write_json(root, "scene_plan.json", data)], msg


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

    blockers: list[str] = []
    if not script:
        blockers.append("script.json missing")
    elif not script.get("approved"):
        blockers.append("script.json not approved (set approved=true)")
    if not meta:
        blockers.append("youtube_metadata.json missing")
    elif not meta.get("approved"):
        blockers.append("youtube_metadata.json not approved (set approved=true)")

    if blockers:
        raise SkipStage("publish blocked: " + "; ".join(blockers))

    cfg = manifest.config
    record = {
        "job_id": cfg.job_id,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "tags": meta.get("tags", []),
        "language": cfg.language,
        "aspect_ratio": cfg.aspect_ratio,
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
    skip; any other exception marks the stage "failed" and stops the run."""
    manifest = load_manifest(root)
    for stage in manifest.stages:
        if not force and stage.status in ("ready", "skipped"):
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
            except SkipStage as skip:
                stage.mark("skipped", skip.reason)
                save_manifest(root, manifest)
            except Exception as exc:
                stage.mark("failed", f"{type(exc).__name__}: {exc}")
                save_manifest(root, manifest)
                break
            else:
                stage.mark("ready", message, artifacts)
                save_manifest(root, manifest)
        if until and stage.stage == until:
            break
    return manifest


def job_status(root: Path) -> dict:
    manifest = load_manifest(root)
    counts: dict[str, int] = {}
    for s in manifest.stages:
        counts[s.status] = counts.get(s.status, 0) + 1
    return {
        "job_id": manifest.config.job_id,
        "complete": manifest.is_complete,
        "counts": counts,
        "stages": [
            {"stage": s.stage, "status": s.status, "message": s.message}
            for s in manifest.stages
        ],
    }
