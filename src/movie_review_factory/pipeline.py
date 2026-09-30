import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterator

from . import agy_vision, branding, cancellation, copyright_bypass, mask_detection, scene_scoring, semantic_search, visual_rhythm, watermark_removal
from .agy_agent import run_agy_json
from .content_agent import run_claude_json
from .creative_brief import prompt_creative_brief, script_evidence_issues, stale_script_tags
from .media_store import MediaStore
from .models import Artifact, JobConfig, JobManifest, MediaAsset, Shot, StageResult, TranscriptSegment, VisualObservation

STAGES = ("ingest","watermark","research","transcript","scenes","outline","script","scene_plan","tts","alignment","render","qa","metadata","thumbnail","publish")

MANIFEST_NAME = "manifest.json"
MANIFEST_RETRY_ATTEMPTS = 10
MANIFEST_RETRY_DELAY_SECONDS = 0.01
_KNOWN_ARTIFACTS = (
    "ingest.json", "research.json", "transcript.json", "captions.srt",
    "scenes.json", "outline.json", "script.json", "script.md", "scene_plan.json",
    "voice.json", "narration.mp3", "alignment.json", "aligned.srt",
    "render.json", "final.mp4", "aligned.ass", "intro-card.png", "outro-card.png", "qa.json",
    "youtube_metadata.json", "thumbnails.json", "thumbnail.jpg",
    "thumbnail-1.jpg", "thumbnail-2.jpg", "thumbnail-3.jpg",
    "media_index.sqlite3", "publish_record.json",
    "source_clean.mp4", "watermark.json",
)

_STAGE_ARTIFACTS = {
    "ingest": ("ingest.json",),
    "watermark": ("source_clean.mp4", "watermark.json"),
    "research": ("research.json",),
    "transcript": ("transcript.json", "captions.srt"),
    "scenes": ("scenes.json", "media_index.sqlite3"),
    "outline": ("outline.json",),
    "script": ("script.json", "script.md"),
    "scene_plan": ("scene_plan.json",),
    "tts": ("voice.json", "narration.mp3"),
    "alignment": ("alignment.json", "aligned.srt"),
    "render": ("render.json", "final.mp4", "aligned.ass"),
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
THUMBNAIL_SIZES = {"16:9": (1280, 720), "9:16": (720, 1280)}
THUMBNAIL_COUNT = 3
SCENE_PLAN_MAX_SHOTS_PER_SECTION = 6
SCENE_PLAN_TARGET_SHOT_SECONDS = 8.0
# Burned-in caption geometry, expressed as fractions of the real frame so the
# same rule holds for 16:9 and 9:16 and for both the web preview and the export
# (the preview plays the rendered MP4, so there is a single source of truth).
SUBTITLE_BOTTOM_FRACTION = 0.10   # caption block sits ~10% above the frame bottom (lower-third safe area)
SUBTITLE_SIDE_FRACTION = 0.075    # left/right margin -> caption width capped at ~85%
SUBTITLE_FONT_FRACTION = 0.042    # caption font size as a fraction of frame height


# --- manifest I/O -----------------------------------------------------------

def manifest_path(root: Path) -> Path:
    return root / MANIFEST_NAME


def load_manifest(root: Path) -> JobManifest:
    path = manifest_path(root)
    for attempt in range(MANIFEST_RETRY_ATTEMPTS):
        try:
            manifest = JobManifest.model_validate_json(path.read_text(encoding="utf-8"))
            actual = [stage.stage for stage in manifest.stages]
            # Reconcile older manifests when stages are added/reordered (e.g. the
            # optional "watermark" stage): keep every known stage's status and
            # insert any missing stage as pending, in canonical STAGES order.
            # Only reconcile when all present stages are known, so a genuinely
            # corrupt manifest is still surfaced by validate_job().
            if actual != list(STAGES) and all(name in STAGES for name in actual):
                existing = {stage.stage: stage for stage in manifest.stages}
                manifest.stages = [
                    existing.get(name, StageResult(stage=name, status="pending"))
                    for name in STAGES
                ]
                save_manifest(root, manifest)
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


def _clear_media_cache(root: Path) -> None:
    for pattern in ("shot-*.jpg", "highlight-h-*.mp4"):
        for path in root.glob(pattern):
            path.unlink(missing_ok=True)
    (root / "transcript.vtt").unlink(missing_ok=True)


def create_job(root: Path, config: JobConfig) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in _KNOWN_ARTIFACTS:
        (root / name).unlink(missing_ok=True)
    _clear_media_cache(root)
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
    if changed_index < STAGES.index("scenes"):
        _clear_media_cache(root)
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


def _source_video(root: Path, cfg: JobConfig) -> Path:
    """Resolve the source video, preferring a watermark-cleaned copy if present.

    The optional ``watermark`` stage writes ``source_clean.mp4``; when it exists
    every downstream stage (scenes, render) reads it instead of the original so
    the full-frame watermark never reaches the finished review.
    """
    clean = root / "source_clean.mp4"
    if clean.exists():
        return clean
    return Path(cfg.source_video)


def _band_or_box_mask(root: Path, cfg: JobConfig) -> Path | None:
    """Generate a rectangular mask from boxes or top/bottom bands.

    Uses the frame size recorded by the ingest stage; returns None when neither
    a box nor a band is configured (or the frame size is unknown).
    """
    wm = cfg.watermark_removal
    ingest = _read_json(root, "ingest.json")
    width, height = ingest.get("width"), ingest.get("height")
    if not width or not height:
        return None
    out = root / "watermark_mask.png"
    if wm.boxes:
        return watermark_removal.rect_mask_from_boxes(int(width), int(height), wm.boxes, out)
    if wm.top_band or wm.bottom_band:
        return watermark_removal.rect_mask_from_bands(
            int(width), int(height), out,
            top_fraction=wm.top_band, bottom_fraction=wm.bottom_band,
        )
    return None


def _resolve_watermark_mask(root: Path, cfg: JobConfig, src: Path) -> Path | None:
    """Resolve a mask for the watermark region.

    Order: an explicit ``mask`` (single image or per-frame folder), then an
    auto-detected per-frame mask folder (``detect``), then a generated
    rectangular mask from boxes/bands. Raises SkipStage only when a detector is
    unavailable (so the job still runs); other detector failures propagate.
    """
    wm = cfg.watermark_removal
    if wm.mask:
        explicit = Path(wm.mask)
        return explicit if explicit.exists() else None
    if wm.detect:
        color = tuple(wm.detect.color[:3]) if len(wm.detect.color) >= 3 else (255, 255, 255)
        settings = mask_detection.DetectSettings(
            method=wm.detect.method,
            target_rgb=color,
            tolerance=wm.detect.tolerance,
            dilation=wm.detect.dilation,
            threshold=wm.detect.threshold,
            fps=wm.detect.fps or None,
            external_cmd=wm.detect.external_cmd,
        )
        try:
            return mask_detection.generate_frame_masks(src, root / "watermark_masks", settings)
        except mask_detection.MaskDetectorUnavailable as exc:
            raise SkipStage(str(exc))
    return _band_or_box_mask(root, cfg)


# --- watermark: optional full-frame watermark removal via ProPainter --------

@register_stage("watermark")
def _watermark(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Reconstruct a watermark-free source video with ProPainter (optional).

    Self-skips unless ``watermark_removal.enabled`` is set, a mask/band/box is
    configured, and ProPainter is installed (``MRF_PROPAINTER_DIR``), so a job
    without the model still runs cleanly.
    """
    cfg = manifest.config
    wm = cfg.watermark_removal
    if not wm.enabled:
        raise SkipStage("watermark removal disabled - set watermark_removal.enabled to clean full-frame watermarks")
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to clean")
    src = Path(cfg.source_video)
    if not src.exists():
        raise SkipStage(f"source_video not found: {src}")
    mask = _resolve_watermark_mask(root, cfg, src)
    if mask is None:
        raise SkipStage("no watermark mask configured - set watermark_removal.mask, detect, boxes, or a band")
    try:
        pp_config = watermark_removal.resolve_config()
    except watermark_removal.ProPainterUnavailable as exc:
        raise SkipStage(str(exc))
    output = root / "source_clean.mp4"
    watermark_removal.remove_watermark(src, mask, output, pp_config)
    per_frame = watermark_removal.frame_mask_paths(mask) if mask.is_dir() else []
    data = {
        "job_id": cfg.job_id,
        "source_video": str(src),
        "clean_video": output.name,
        "mask": str(mask),
        "mask_kind": "per-frame" if mask.is_dir() else "static",
        "mask_frames": len(per_frame),
        "propainter_home": str(pp_config.home),
    }
    artifacts = [
        Artifact(name="source_clean.mp4", path=output, status="ready"),
        _write_json(root, "watermark.json", data),
    ]
    return artifacts, "removed full-frame watermark with ProPainter"


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


# When faster-whisper returns per-word timings, a coarse VAD segment is cut into
# sentence-sized transcript rows so captions and the explorer list stay anchored to
# real speech instead of being interpolated by character count. A row ends at a
# sentence mark once it is at least MIN long, at a silence GAP between two spoken
# words, or once it reaches MAX seconds.
_TRANSCRIPT_SPLIT_GAP_SECONDS = 0.6
_TRANSCRIPT_SPLIT_MIN_SECONDS = 2.5
_TRANSCRIPT_SPLIT_MAX_SECONDS = 8.0


def _cuda_available() -> bool:
    """Best-effort probe for a usable CUDA device via ctranslate2 (whisper's backend)."""
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count()) > 0
    except Exception:
        return False


def _whisper_runtime() -> tuple[str, str, str]:
    """Resolve (model, device, compute_type) for faster-whisper.

    Honours MRF_WHISPER_MODEL / MRF_WHISPER_DEVICE / MRF_WHISPER_COMPUTE_TYPE and
    otherwise auto-detects: a CUDA GPU -> float16 (8-15x faster than CPU int8),
    else CPU int8. Point MRF_WHISPER_MODEL at a distilled checkpoint such as
    "distil-large-v3" for a further 4-5x speed-up at near-identical accuracy.
    """
    model_name = os.environ.get("MRF_WHISPER_MODEL", "").strip() or "small"
    device = os.environ.get("MRF_WHISPER_DEVICE", "").strip().lower()
    compute_type = os.environ.get("MRF_WHISPER_COMPUTE_TYPE", "").strip().lower()
    if device not in {"cpu", "cuda", "auto", ""}:
        device = ""
    if not device or device == "auto":
        device = "cuda" if _cuda_available() else "cpu"
    if not compute_type:
        compute_type = "float16" if device == "cuda" else "int8"
    return model_name, device, compute_type


def _extract_wav_16k_mono(src: Path, dest: Path) -> bool:
    """Demux the audio to 16kHz mono PCM so whisper reads a tiny WAV instead of
    demuxing the multi-GB video container on every run. Returns True on success."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    dest.unlink(missing_ok=True)
    command = [
        ffmpeg, "-y", "-i", str(src), "-vn",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(dest),
    ]
    try:
        subprocess.run(command, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, OSError):
        dest.unlink(missing_ok=True)
        return False
    return dest.exists() and dest.stat().st_size > 0


def _prepare_whisper_audio(root: Path, src: Path) -> Path:
    """Return a fast 16kHz mono WAV for whisper, or the original source when
    extraction is unavailable (kept as a seam so tests can stub it)."""
    wav_path = root / ".mrf_whisper_audio.wav"
    if _extract_wav_16k_mono(src, wav_path):
        return wav_path
    return src


def _srt_time_to_seconds(value: str) -> float:
    value = value.strip().replace(",", ".")
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _parse_srt_segments(text: str) -> list[dict]:
    """Parse SRT / converted-subtitle text into timed transcript rows."""
    segments: list[dict] = []
    for block in re.split(r"\r?\n[ \t]*\r?\n", text.strip()):
        lines = [line for line in block.splitlines() if line.strip()]
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        match = re.search(
            r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})",
            lines[timing],
        )
        if not match:
            continue
        body = " ".join(lines[timing + 1:]).strip()
        body = re.sub(r"<[^>]+>", "", body)
        body = re.sub(r"\{[^}]*\}", "", body).strip()
        if not body:
            continue
        start = _srt_time_to_seconds(match.group(1))
        end = _srt_time_to_seconds(match.group(2))
        segments.append({
            "start_seconds": start,
            "end_seconds": end if end > start else start,
            "text": body,
        })
    return segments


def _embedded_subtitle_segments(
    src: Path, want_langs: list[str], root: Path
) -> tuple[list[dict], str] | None:
    """Reuse a subtitle track shipped inside the container instead of running
    Whisper. Returns (segments, language) or None when nothing usable is found."""
    ffprobe = shutil.which("ffprobe")
    ffmpeg = shutil.which("ffmpeg")
    if not ffprobe or not ffmpeg:
        return None
    try:
        probe = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "s",
             "-show_entries", "stream=index:stream_tags=language",
             "-of", "json", str(src)],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(probe.stdout or "{}").get("streams") or []
    except (subprocess.CalledProcessError, OSError, json.JSONDecodeError):
        return None
    wanted = [lang.strip().lower() for lang in want_langs if lang.strip()]
    chosen_rel: int | None = None
    chosen_lang = "unknown"
    for rel_index, stream in enumerate(streams):
        lang = str((stream.get("tags") or {}).get("language", "")).lower()
        if not wanted or lang in wanted:
            chosen_rel, chosen_lang = rel_index, (lang or "unknown")
            break
    if chosen_rel is None:
        return None
    dest = root / ".mrf_embedded_sub.srt"
    dest.unlink(missing_ok=True)
    raw = ""
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src), "-map", f"0:s:{chosen_rel}", str(dest)],
            capture_output=True, text=True, check=True,
        )
        if dest.exists():
            raw = dest.read_text(encoding="utf-8", errors="replace")
    except (subprocess.CalledProcessError, OSError):
        raw = ""
    finally:
        dest.unlink(missing_ok=True)
    segments = _parse_srt_segments(raw)
    if not segments:
        return None
    return segments, chosen_lang


@register_stage("transcript")
def _transcript(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Transcribe a local source video into timed JSON and SRT artifacts."""
    cfg = manifest.config
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to transcribe")
    src = Path(cfg.source_video)
    if not src.exists():
        raise SkipStage(f"source_video not found: {src}")
    # Ingest already recorded the fact: a container with zero audio streams makes faster-whisper/PyAV die with a bare "tuple index out of range".
    if _read_json(root, "ingest.json").get("has_audio") is False:
        raise SkipStage("source video has no audio track - nothing to transcribe")
    # Optional fast path: if the container already ships a usable subtitle track,
    # reuse it verbatim and skip Whisper entirely (set MRF_TRANSCRIPT_EMBEDDED_SUBS=1).
    if os.environ.get("MRF_TRANSCRIPT_EMBEDDED_SUBS", "").strip().lower() in {"1", "true", "yes", "on"}:
        requested = os.environ.get("MRF_TRANSCRIPT_SUB_LANGS", "").strip()
        want_langs = [part for part in re.split(r"[,\s]+", requested) if part]
        if not want_langs and cfg.language:
            want_langs = [cfg.language]
        embedded = _embedded_subtitle_segments(src, want_langs, root)
        if embedded is not None:
            segments, sub_language = embedded
            transcript = {
                "job_id": cfg.job_id,
                "source_video": str(src),
                "language": sub_language,
                "output_language": cfg.language,
                "segments": segments,
                "source": "embedded-subtitles",
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
            return artifacts, f"reused {len(segments)} embedded subtitle segments (skipped Whisper)"
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SkipStage("faster-whisper not installed - install the media extra") from exc

    logical_cores = os.cpu_count() or 4
    cpu_threads = min(8, max(1, logical_cores // 2))
    configured_threads = os.environ.get("MRF_WHISPER_CPU_THREADS", "").strip()
    if configured_threads:
        cpu_threads = max(1, min(32, int(configured_threads)))
    model_name, device, compute_type = _whisper_runtime()
    try:
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            **_whisper_model_options(),
        )
    except (RuntimeError, ValueError, OSError):
        # A GPU runtime was requested/auto-selected but is not usable on this box
        # (missing CUDA libraries, insufficient VRAM, ...). Fall back to CPU int8 so
        # the job still completes rather than aborting the whole pipeline.
        if device == "cpu":
            raise
        device, compute_type = "cpu", "int8"
        model = WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            **_whisper_model_options(),
        )
    # Feed whisper a small pre-extracted 16kHz mono WAV instead of demuxing the
    # full video container on every decode pass.
    audio_input = _prepare_whisper_audio(root, src)
    batch_size = max(1, min(16, int(os.environ.get("MRF_WHISPER_BATCH_SIZE", "4"))))
    # word_timestamps=True makes faster-whisper emit per-word times derived from
    # the model's cross-attention. Whisper's coarse segment.start is
    # systematically early (Silero VAD's speech_pad_ms padding plus the model's
    # ~400ms lead-in), which surfaces as captions that sit a constant beat ahead
    # of the audio; anchoring each segment to its own first/last word removes
    # that fixed offset without a hand-tuned magic constant.
    if batch_size > 1:
        from faster_whisper import BatchedInferencePipeline

        transcriber = BatchedInferencePipeline(model=model)
        raw_segments, detected = transcriber.transcribe(
            str(audio_input), language=None, vad_filter=True, batch_size=batch_size,
            word_timestamps=True,
        )
    else:
        raw_segments, detected = model.transcribe(
            str(audio_input), language=None, vad_filter=True, word_timestamps=True,
        )

    def _split_segment(segment) -> list[dict]:
        """Cut one coarse VAD segment into sentence-sized, word-anchored rows.

        faster-whisper groups speech into segments that can span many seconds and
        several sentences. Keeping such a block whole forces the UI to interpolate
        per-sentence timing by character count, which assumes a constant speaking
        pace and drifts ahead of the audio across any pause. With per-word timings we
        break at sentence-ending punctuation, at a silent gap between two spoken
        words, or once a row grows too long — so every row starts and ends on real
        speech. Falls back to the whole segment when word timings are unavailable.
        """
        def _whole() -> list[dict]:
            return [{
                "start_seconds": float(segment.start),
                "end_seconds": float(segment.end),
                "text": segment.text.strip(),
                "words": [],
            }]
        words = [
            word for word in (getattr(segment, "words", None) or [])
            if getattr(word, "start", None) is not None and getattr(word, "end", None) is not None
        ]
        if not words:
            return _whole()
        rows: list[dict] = []
        buffer: list = []

        def _flush() -> None:
            if not buffer:
                return
            text = "".join(str(getattr(word, "word", "")) for word in buffer).strip()
            start, end = float(buffer[0].start), float(buffer[-1].end)
            if text and end > start:
                word_times = [
                    {"word": str(getattr(word, "word", "")).strip(),
                     "start": float(word.start), "end": float(word.end)}
                    for word in buffer
                ]
                rows.append({"start_seconds": start, "end_seconds": end, "text": text, "words": word_times})

        for index, word in enumerate(words):
            buffer.append(word)
            token = str(getattr(word, "word", "")).strip()
            row_seconds = float(word.end) - float(buffer[0].start)
            ends_sentence = token.endswith((".", "?", "!", "…")) and row_seconds >= _TRANSCRIPT_SPLIT_MIN_SECONDS
            gap_ahead = (
                index + 1 < len(words)
                and float(words[index + 1].start) - float(word.end) >= _TRANSCRIPT_SPLIT_GAP_SECONDS
            )
            if ends_sentence or gap_ahead or row_seconds >= _TRANSCRIPT_SPLIT_MAX_SECONDS:
                _flush()
                buffer = []
        _flush()
        return rows or _whole()

    segments = []
    for segment in raw_segments:
        segments.extend(_split_segment(segment))
    if audio_input != src:
        Path(audio_input).unlink(missing_ok=True)
    transcript = {
        "job_id": cfg.job_id,
        "source_video": str(src),
        "language": getattr(detected, "language", "unknown"),
        "output_language": cfg.language,
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
    length after clamping are dropped. Silent spans are kept as bounded
    selectable ranges, including when the source has no speech.
    """
    bounded: list[dict] = []
    for seg in segments:
        start = _clamp(seg.get("start_seconds", 0.0), duration)
        end = _clamp(seg.get("end_seconds", start), duration)
        if end <= start:
            continue
        bounded.append({"start_seconds": start, "end_seconds": end, "text": str(seg.get("text", "")).strip()})
    bounded.sort(key=lambda s: (s["start_seconds"], s["end_seconds"]))
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

    timeline: list[dict] = []
    cursor = 0.0

    def add_silent_span(start: float, end: float) -> None:
        while start < end:
            stop = min(start + SCENE_MAX_SECONDS, end)
            timeline.append({
                "index": len(timeline) + 1,
                "start_seconds": start,
                "end_seconds": stop,
                "segment_count": 0,
                "text": "",
            })
            start = stop

    for scene in scenes:
        if scene["start_seconds"] - cursor > SCENE_GAP_SECONDS:
            add_silent_span(cursor, scene["start_seconds"])
        scene["index"] = len(timeline) + 1
        timeline.append(scene)
        cursor = max(cursor, scene["end_seconds"])
    if duration - cursor > SCENE_GAP_SECONDS or not timeline:
        add_silent_span(cursor, duration)
    return timeline


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
    src = _source_video(root, cfg)
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
    database_path = root / "media_index.sqlite3"
    _clear_media_cache(root)
    database_path.unlink(missing_ok=True)
    semantic_mode = "unavailable"
    semantic_error = ""
    with MediaStore(database_path) as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=src, duration_seconds=duration),
            [
                Shot(
                    media_asset_id=1,
                    start_seconds=scene["start_seconds"],
                    end_seconds=scene["end_seconds"],
                    label=scene["text"] or f"Scene {scene['index']}",
                )
                for scene in scenes
            ],
            [
                TranscriptSegment(
                    media_asset_id=1,
                    start_seconds=float(segment["start_seconds"]),
                    end_seconds=float(segment["end_seconds"]),
                    text=str(segment.get("text", "")).strip(),
                    words=segment.get("words") or [],
                )
                for segment in transcript.get("segments", [])
                if str(segment.get("text", "")).strip()
                and float(segment["end_seconds"]) > float(segment["start_seconds"])
            ],
        )
        try:
            semantic_search.refresh_store_embeddings(store)
            semantic_mode = "fastembed"
        except semantic_search.EmbeddingUnavailable as exc:
            semantic_error = str(exc)
    data["semantic_mode"] = semantic_mode
    data["semantic_model"] = semantic_search.model_name()
    data["semantic_error"] = semantic_error
    artifacts = [
        _write_json(root, "scenes.json", data),
        Artifact(name=database_path.name, path=database_path, status="ready"),
    ]
    return artifacts, f"indexed {len(scenes)} scenes from transcript"


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


SCENE_CANDIDATE_LIMIT = 12


def _retrieve_scene_candidates(sections: list[dict], scenes_doc: dict) -> list[list[dict]]:
    scenes = [item for item in scenes_doc.get("scenes", []) if isinstance(item, dict)]
    if not scenes:
        return [[] for _ in sections]
    result: list[list[dict]] = []
    previous_top: set[int] = set()
    for section_index, section in enumerate(sections):
        anchor = round(section_index * (len(scenes) - 1) / max(len(sections) - 1, 1))
        ranked = scene_scoring.rank_scenes(
            section,
            scenes,
            anchor=anchor,
            previous_scene_indexes=previous_top,
            limit=SCENE_CANDIDATE_LIMIT,
        )
        result.append(ranked)
        if ranked:
            previous_top.add(int(ranked[0]["index"]))
    return result


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
        raise ValueError("content agent outline contained no usable sections")

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
        raise ValueError("content agent outline could not be normalized to the target duration")
    return cleaned


def _agent_display(mode: str) -> str:
    return "AGY" if mode == "agy" else "Claude"


# --- AGY prompt budget fitting ------------------------------------------------

AGY_PROMPT_MAX_DEFAULT = 26000
AGY_PROMPT_MAX_MINIMUM = 2000

# tier -> (floor, floor after the single relaxation pass)
_AGY_TRIM_FLOORS = {"A": (120, 60), "B": (200, 150), "C": (400, 300)}
# Shortest string each tier may cut while its floors are still the first-pass ones.
_AGY_TRIM_MINIMUM = {"A": 200, "B": 200, "C": 400}
_AGY_TRIM_TIERS = ("A", "B", "C")


def _agy_prompt_max() -> int:
    """Characters allowed for one AGY prompt (``MRF_AGY_PROMPT_MAX``)."""
    raw = os.environ.get("MRF_AGY_PROMPT_MAX", str(AGY_PROMPT_MAX_DEFAULT))
    try:
        value = int(raw)
    except ValueError:
        return AGY_PROMPT_MAX_DEFAULT
    if value < AGY_PROMPT_MAX_MINIMUM:
        return AGY_PROMPT_MAX_DEFAULT
    return value


def _agy_string_tier(path: str) -> str:
    if "scenes" in path or "scene_candidates" in path:
        return "A"
    if "script_sections" in path:
        return "C"
    return "B"


def _agy_string_leaves(node: object, path: str = "") -> Iterator[tuple[dict | list, object, str, str]]:
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(value, str):
                yield node, key, value, child
            elif isinstance(value, (dict, list)):
                yield from _agy_string_leaves(value, child)
    elif isinstance(node, list):
        for index, value in enumerate(node):
            child = f"{path}[{index}]"
            if isinstance(value, str):
                yield node, index, value, child
            elif isinstance(value, (dict, list)):
                yield from _agy_string_leaves(value, child)


def _fit_agy_context(context: dict, budget: int) -> tuple[dict, int]:
    """Shorten string values until ``context`` serialises within ``budget``.

    ``budget`` is the share of the final AGY prompt left for the serialised
    context; ``_run_reasoning_agent`` subtracts the fixed prompt head first, so
    the whole prompt stays under ``MRF_AGY_PROMPT_MAX``. Keys, structure, list
    indexes, numbers and booleans are never touched - only string values get
    cut, always to ``max(floor, len // 2)``.

    Tiers are drained in order: A (scenes / scene_candidates), then B (research
    and every other string), then C (script_sections), so the cheapest context
    is given up first. Candidates inside a tier rank by (longest first, then
    json-path ascending) for a deterministic order. Once every string sits at
    its floor and the budget is still missed, all floors relax once
    (60 / 150 / 300) for a final pass; a second stall ends the loop even if the
    prompt remains over budget.
    """
    original = json.dumps(context, ensure_ascii=False)
    if len(original) <= budget:
        return context, 0

    fitted = copy.deepcopy(context)
    floors = {tier: bounds[0] for tier, bounds in _AGY_TRIM_FLOORS.items()}
    trimmed: set[str] = set()
    size = len(original)
    relaxed = False
    while size > budget:
        progress = False
        for tier in _AGY_TRIM_TIERS:
            while size > budget:
                floor = floors[tier]
                minimum = floor if relaxed else _AGY_TRIM_MINIMUM[tier]
                best: tuple[str, dict | list, object, str] | None = None
                best_rank: tuple[int, str] | None = None
                for holder, key, value, path in _agy_string_leaves(fitted):
                    length = len(value)
                    if length <= floor or length < minimum:
                        continue
                    if _agy_string_tier(path) != tier:
                        continue
                    rank = (-length, path)
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best = (path, holder, key, value)
                if best is None:
                    break
                path, holder, key, value = best
                holder[key] = value[: max(floor, len(value) // 2)]
                trimmed.add(path)
                size = len(json.dumps(fitted, ensure_ascii=False))
                progress = True
            if size <= budget:
                break
        if progress:
            continue
        if not relaxed:
            floors = {tier: bounds[1] for tier, bounds in _AGY_TRIM_FLOORS.items()}
            relaxed = True
            continue
        break
    if trimmed:
        print(
            f"agy prompt trimmed: {len(original)} -> {size} chars ({len(trimmed)} strings)",
            flush=True,
        )
    return fitted, len(trimmed)


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
    if manifest.config.content_agent not in ("claude", "agy"):
        return None

    prompt_head = (
        "You are the reasoning worker for Movie Review Factory. "
        "Produce original review/recap material, not copied dialogue. "
        "Do not invent facts or source URLs; put unresolved claims in uncertainties when the schema permits it. "
        "Return only the structured result requested by the JSON schema enforced by the caller.\n\n"
        f"STAGE: {stage}\n"
        f"INSTRUCTION: {instruction}\n"
        "JOB CONTEXT:\n"
    )
    if manifest.config.content_agent == "agy":
        context, _ = _fit_agy_context(context, _agy_prompt_max() - len(prompt_head))
        return run_agy_json(
            stage=stage,
            prompt=prompt_head + json.dumps(context, ensure_ascii=False),
            schema=schema,
        )
    prompt = prompt_head + json.dumps(context, ensure_ascii=False)
    return run_claude_json(
        root=root,
        stage=stage,
        prompt=prompt,
        schema=schema,
        allowed_tools=allowed_tools,
    )


# --- research: deterministic scaffold or content-agent research ---------------------

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
            "generator": cfg.content_agent,
        }
        if not data["brief"]:
            raise ValueError(f"{_agent_display(cfg.content_agent)} research returned an empty brief")
        return [_write_json(root, "research.json", data)], f"research generated by {_agent_display(cfg.content_agent)}"

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


def _retention_advice_for(root: Path) -> dict | None:
    """Read-only retention advice for the content-agent context (never mutates output).

    Lazy import avoids an analytics<->pipeline cycle; failure-safe so a malformed
    analytics folder can never break the critical generation path. Returns ``None``
    when unavailable, which keeps :func:`prompt_creative_brief` at its legacy shape.
    """
    try:
        from . import analytics

        return analytics.retention_advice(root)
    except Exception:
        return None


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
            "provided scene context. Treat creative brief as editorial preferences, not film facts. "
            "Return 3-8 sections with relative time budgets."
        ),
        context={
            "creative_brief": prompt_creative_brief(cfg, retention_advice=_retention_advice_for(root)),
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
            "generator": cfg.content_agent,
        }
        return [_write_json(root, "outline.json", outline)], f"outline generated by {_agent_display(cfg.content_agent)}"

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

def _script_tag_issues(root: Path, script: dict) -> list[str]:
    stale = stale_script_tags(script)
    issues = [f"stale evidence tags at {stale}"] if stale else []
    issues.extend(script_evidence_issues(
        script, _read_json(root, "scenes.json"), _read_json(root, "transcript.json")
    ))
    return issues


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
            "avoid long verbatim dialogue, and write enough narration to fit each section budget. "
            "Treat creative brief as editorial preferences, not film facts."
        ),
        context={
            "creative_brief": __import__("movie_review_factory.creative_brief", fromlist=["prompt_creative_brief"]).prompt_creative_brief(cfg, retention_advice=_retention_advice_for(root)),
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
                f"{_agent_display(cfg.content_agent)} script section count does not match the approved outline shape"
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
                raise ValueError(f"{_agent_display(cfg.content_agent)} script returned empty narration for {title!r}")
            sections.append({
                "title": title,
                "budget_minutes": budget,
                "narration": narration,
                "duration_seconds": round(budget * 60),
            })
        notes = str(agent.get("notes") or "").strip()
        generator = cfg.content_agent
        message = f"script generated by {_agent_display(cfg.content_agent)} (approval required before TTS)"
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
        raise ValueError("content agent scene plan indexes must be integers")
    if start_index not in positions or end_index not in positions:
        raise ValueError("content agent scene plan referenced an unknown scene index")
    start_pos, end_pos = positions[start_index], positions[end_index]
    if end_pos < start_pos:
        raise ValueError("content agent scene plan end scene precedes start scene")

    selected = scenes[start_pos:end_pos + 1]
    start = float(selected[0].get("start_seconds") or 0.0)
    end = float(selected[-1].get("end_seconds") or 0.0)
    if (
        not math.isfinite(start) or not math.isfinite(end)
        or start < 0 or end <= start
        or (video_duration > 0 and end > video_duration)
    ):
        raise ValueError("content agent scene plan resolved to an invalid source range")
    return {"start_seconds": start, "end_seconds": end}


def _agent_scene_assignments(
    sections: list[dict],
    scenes_doc: dict,
    assignments: list[dict],
    candidates: list[list[dict]] | None = None,
) -> list[list[tuple[dict, str]]]:
    """Validate content agent shot choices and resolve scene IDs to source timestamps."""
    scenes = [scene for scene in (scenes_doc.get("scenes") or []) if isinstance(scene, dict)]
    if len(assignments) != len(sections):
        raise ValueError("content agent scene plan assignment count does not match script sections")
    if not scenes:
        raise ValueError("content agent scene plan cannot run without indexed scenes")

    positions: dict[int, int] = {}
    for position, scene in enumerate(scenes):
        index = scene.get("index")
        if isinstance(index, int):
            positions[index] = position

    video_duration = float(scenes_doc.get("duration_seconds") or 0.0)
    result: list[list[tuple[dict, str]]] = []
    for section_index, assignment in enumerate(assignments):
        allowed = (
            {scene["index"] for scene in candidates[section_index]}
            if candidates is not None else None
        )
        if not isinstance(assignment, dict):
            raise ValueError("content agent scene plan assignment must be an object")
        raw_shots = assignment.get("shots")
        if not isinstance(raw_shots, list) or not raw_shots:
            raise ValueError("content agent scene plan assignment must contain at least one shot")
        if len(raw_shots) > SCENE_PLAN_MAX_SHOTS_PER_SECTION:
            raise ValueError("content agent scene plan assignment contains too many shots")

        shots: list[tuple[dict, str]] = []
        seen_ranges: set[tuple[float, float]] = set()
        for shot in raw_shots:
            if not isinstance(shot, dict):
                raise ValueError("content agent scene plan shot must be an object")
            source_clip = _scene_range_from_indexes(
                scenes,
                positions,
                shot.get("start_scene_index"),
                shot.get("end_scene_index"),
                video_duration,
            )
            if allowed is not None:
                # The agent only references the range endpoints, and the per-section
                # candidate set is a sparse ranked subset, so intermediate scenes in
                # a contiguous range are frequently not candidates through no fault of
                # the agent. Validate only the chosen endpoints against candidates.
                if (shot["start_scene_index"] not in allowed
                        or shot["end_scene_index"] not in allowed):
                    raise ValueError("content agent scene plan selected a scene outside section candidates")
            key = (source_clip["start_seconds"], source_clip["end_seconds"])
            if key in seen_ranges:
                raise ValueError("content agent scene plan repeated the same shot within one section")
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


def _scene_range_key(scene: dict) -> tuple[float, float]:
    return (
        round(float(scene.get("start_seconds") or 0.0), 6),
        round(float(scene.get("end_seconds") or 0.0), 6),
    )


def _load_scene_visual_observations(database: Path, scenes: list[dict]) -> dict[int, dict]:
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
        rows = store.list_visual_observations()
    shot_by_range = {_scene_range_key(shot.model_dump()): shot.id for shot in shots if shot.id}
    row_by_shot = {int(row["shot_id"]): row for row in rows}
    result: dict[int, dict] = {}
    for scene in scenes:
        scene_index = scene.get("index")
        shot_id = shot_by_range.get(_scene_range_key(scene))
        if not isinstance(scene_index, int) or not shot_id or shot_id not in row_by_shot:
            continue
        row = row_by_shot[shot_id]
        result[scene_index] = {
            "description": row["description"],
            "tags": list(row.get("tags") or []),
            "people": list(row.get("people") or []),
            "actions": list(row.get("actions") or []),
        }
    return result


def _save_scene_visual_observations(
    database: Path,
    scenes: list[dict],
    observations: dict[int, dict],
) -> None:
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
        shot_by_range = {_scene_range_key(shot.model_dump()): shot.id for shot in shots if shot.id}
        records: list[VisualObservation] = []
        for scene in scenes:
            scene_index = scene.get("index")
            if not isinstance(scene_index, int) or scene_index not in observations:
                continue
            shot_id = shot_by_range.get(_scene_range_key(scene))
            if not shot_id:
                continue
            observation = observations[scene_index]
            records.append(VisualObservation(
                shot_id=shot_id,
                description=str(observation.get("description") or "").strip(),
                tags=list(observation.get("tags") or []),
                people=list(observation.get("people") or []),
                actions=list(observation.get("actions") or []),
                source="agy",
            ))
        store.replace_visual_observations(records)


def _apply_scene_visual_observations(
    scenes: list[dict], observations: dict[int, dict]
) -> None:
    for scene in scenes:
        scene_index = scene.get("index")
        observation = observations.get(scene_index) if isinstance(scene_index, int) else None
        if not observation:
            continue
        scene["visual_description"] = observation["description"]
        scene["visual_tags"] = list(observation.get("tags") or [])
        scene["visual_people"] = list(observation.get("people") or [])
        scene["visual_actions"] = list(observation.get("actions") or [])


def _save_person_identity(database: Path, scenes: list[dict], identity: dict) -> None:
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
        shot_by_range = {_scene_range_key(shot.model_dump()): shot.id for shot in shots if shot.id}
        scene_to_shot = {
            int(scene["index"]): shot_by_range.get(_scene_range_key(scene))
            for scene in scenes if isinstance(scene.get("index"), int)
        }
        appearances = []
        for item in identity.get("appearances", []):
            shot_id = scene_to_shot.get(int(item.get("scene_index") or 0))
            if not shot_id:
                continue
            appearances.append({
                "label": item["label"],
                "shot_id": shot_id,
                "description": item.get("description", ""),
                "clothing": item.get("clothing", ""),
                "ambiguous": bool(item.get("ambiguous", False)),
                "confidence": item.get("confidence", 0.0),
                "evidence": item.get("evidence", ""),
            })
        store.replace_person_tracks(list(identity.get("tracks") or []), appearances)


def _apply_person_identity(database: Path, scenes: list[dict]) -> list[dict]:
    with MediaStore(database) as store:
        store.migrate()
        tracks = store.list_person_tracks()
        shots = store.list_shots(1)
        labels_by_shot = {int(shot.id): store.person_labels_for_shot(int(shot.id))
                          for shot in shots if shot.id}
    shot_by_range = {_scene_range_key(shot.model_dump()): int(shot.id)
                     for shot in shots if shot.id}
    for scene in scenes:
        shot_id = shot_by_range.get(_scene_range_key(scene))
        scene["person_tracks"] = labels_by_shot.get(shot_id, [])
    return tracks


def _save_story_graph(database: Path, scenes: list[dict], graph: dict) -> None:
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
        shot_by_range = {
            _scene_range_key(shot.model_dump()): int(shot.id)
            for shot in shots if shot.id
        }
        scene_to_shot = {
            int(scene["index"]): shot_by_range.get(_scene_range_key(scene))
            for scene in scenes if isinstance(scene.get("index"), int)
        }
        scene_entities = []
        for item in graph.get("scene_entities", []):
            shot_id = scene_to_shot.get(int(item.get("scene_index") or 0))
            if shot_id:
                scene_entities.append({**item, "shot_id": shot_id})
        relations = []
        for item in graph.get("relations", []):
            shot_id = scene_to_shot.get(int(item.get("scene_index") or 0))
            if shot_id:
                relations.append({**item, "shot_id": shot_id})
        store.replace_story_graph(
            list(graph.get("entities") or []),
            scene_entities,
            relations,
        )


def _apply_story_graph(database: Path, scenes: list[dict]) -> dict:
    with MediaStore(database) as store:
        store.migrate()
        graph = store.list_story_graph()
        shots = store.list_shots(1)
    shot_by_range = {
        _scene_range_key(shot.model_dump()): int(shot.id)
        for shot in shots if shot.id
    }
    entities_by_shot: dict[int, list[dict]] = {}
    for item in graph["scene_entities"]:
        entities_by_shot.setdefault(int(item["shot_id"]), []).append({
            "type": item["type"],
            "label": item["label"],
            "confidence": item["confidence"],
        })
    relations_by_shot: dict[int, list[dict]] = {}
    for item in graph["relations"]:
        relations_by_shot.setdefault(int(item["shot_id"]), []).append({
            "subject_type": item["subject_type"],
            "subject_label": item["subject_label"],
            "predicate": item["predicate"],
            "object_type": item["object_type"],
            "object_label": item["object_label"],
            "confidence": item["confidence"],
        })
    for scene in scenes:
        shot_id = shot_by_range.get(_scene_range_key(scene))
        scene["story_entities"] = entities_by_shot.get(shot_id, [])
        scene["story_relations"] = relations_by_shot.get(shot_id, [])
    return graph


def _merge_semantic_scene_candidates(
    sections: list[dict],
    scenes: list[dict],
    database: Path,
    candidates: list[list[dict]],
) -> list[list[dict]]:
    del candidates  # scorer v2 reranks the full bounded scene set.
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
        shot_to_scene: dict[int, int] = {}
        range_to_scene = {
            _scene_range_key(scene): int(scene["index"])
            for scene in scenes if isinstance(scene.get("index"), int)
        }
        for shot in shots:
            if shot.id:
                scene_index = range_to_scene.get(_scene_range_key(shot.model_dump()))
                if scene_index is not None:
                    shot_to_scene[int(shot.id)] = scene_index

        merged: list[list[dict]] = []
        previous_top: set[int] = set()
        for section_index, section in enumerate(sections):
            query = " ".join([
                str(section.get("title") or ""),
                str(section.get("narration") or ""),
            ]).strip()
            semantic_scores: dict[int, float] = {}
            if query:
                for item in semantic_search.search_store(
                    store, query, limit=max(SCENE_CANDIDATE_LIMIT * 3, len(scenes))
                ):
                    if item["kind"] != "visual":
                        continue
                    scene_index = shot_to_scene.get(int(item["shot_id"]))
                    if scene_index is not None:
                        semantic_scores[scene_index] = float(item["score"])
            anchor = round(
                section_index * (len(scenes) - 1) / max(len(sections) - 1, 1)
            )
            ranked = scene_scoring.rank_scenes(
                section,
                scenes,
                anchor=anchor,
                semantic_scores=semantic_scores,
                previous_scene_indexes=previous_top,
                limit=SCENE_CANDIDATE_LIMIT,
            )
            merged.append(ranked)
            if ranked:
                previous_top.add(int(ranked[0]["index"]))
    return merged


def index_scene_memory(
    root: Path,
    cfg: JobConfig,
    scenes_doc: dict,
    scenes: list[dict],
) -> dict:
    """Populate per-job scene memory into media_index.sqlite3 and return a summary.

    Builds visual observations, anonymous person tracks, the story graph, and
    semantic embeddings for the indexed scenes. Extracted verbatim from the head
    of ``_scene_plan`` so the same work can also run standalone in the background
    indexing queue (roadmap #14) right after import, decoupled from content
    generation. Mutates ``scenes`` in place with the cached visual/identity/story
    fields and returns the per-facet mode/error summary. Only meaningful for
    ``content_agent`` in ("claude", "agy") with a source video; callers guard that.
    """
    visual_mode = "not_requested"
    visual_error = ""
    visual_observations: dict[int, dict] = {}
    identity_mode = "not_requested"
    identity_error = ""
    person_tracks: list[dict] = []
    semantic_mode = "not_requested"
    semantic_error = ""
    story_mode = "not_requested"
    story_error = ""
    story_graph: dict = {"entities": [], "scene_entities": [], "relations": []}
    database = root / "media_index.sqlite3"

    indexed_source = scenes_doc.get("source_video")
    if indexed_source and Path(cfg.source_video).resolve() != Path(indexed_source).resolve():
        raise ValueError("scene index belongs to a different source video")
    if database.is_file():
        visual_observations = _load_scene_visual_observations(database, scenes)
        if visual_observations:
            visual_mode = "agy_cached"
        try:
            fresh = agy_vision.describe_candidate_observations(
                Path(cfg.source_video), [scenes]
            )
            if fresh:
                visual_observations = fresh
                _save_scene_visual_observations(database, scenes, fresh)
                visual_mode = "agy"
        except agy_vision.VisionUnavailable as exc:
            visual_error = str(exc)
            if not visual_observations:
                visual_mode = "unavailable"
        _apply_scene_visual_observations(scenes, visual_observations)

        try:
            identity = agy_vision.track_anonymous_people(
                Path(cfg.source_video), scenes
            )
            _save_person_identity(database, scenes, identity)
            identity_mode = "agy"
        except agy_vision.VisionUnavailable as exc:
            identity_error = str(exc)
            with MediaStore(database) as store:
                store.migrate()
                identity_mode = "agy_cached" if store.list_person_tracks() else "unavailable"
        person_tracks = _apply_person_identity(database, scenes)

        try:
            fresh_story = agy_vision.extract_story_graph(scenes)
            _save_story_graph(database, scenes, fresh_story)
            story_mode = "agy"
        except agy_vision.VisionUnavailable as exc:
            story_error = str(exc)
            with MediaStore(database) as store:
                store.migrate()
                cached_story = store.list_story_graph()
            story_mode = (
                "agy_cached"
                if cached_story["entities"] or cached_story["relations"]
                else "unavailable"
            )
        story_graph = _apply_story_graph(database, scenes)

        try:
            with MediaStore(database) as store:
                store.migrate()
                semantic_search.refresh_store_embeddings(store)
            semantic_mode = "fastembed"
        except semantic_search.EmbeddingUnavailable as exc:
            semantic_mode = "unavailable"
            semantic_error = str(exc)

    return {
        "visual_observations": visual_observations,
        "visual_mode": visual_mode,
        "visual_error": visual_error,
        "identity_mode": identity_mode,
        "identity_error": identity_error,
        "person_tracks": person_tracks,
        "semantic_mode": semantic_mode,
        "semantic_error": semantic_error,
        "story_mode": story_mode,
        "story_error": story_error,
        "story_graph": story_graph,
    }


@register_stage("scene_plan")
def _scene_plan(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Map script sections to time-stamped visual clips."""
    cfg = manifest.config
    script = _read_json(root, "script.json")
    sections = script.get("sections") or []
    scenes_doc = _read_json(root, "scenes.json")
    scenes = [scene for scene in scenes_doc.get("scenes", []) if isinstance(scene, dict)]
    scene_assignments: list[list[tuple[dict, str]]]
    agent = None
    candidates: list[list[dict]] = []
    visual_mode = "not_requested"
    visual_error = ""
    visual_observations: dict[int, dict] = {}
    identity_mode = "not_requested"
    identity_error = ""
    person_tracks: list[dict] = []
    semantic_mode = "not_requested"
    semantic_error = ""
    story_mode = "not_requested"
    story_error = ""
    story_graph: dict = {"entities": [], "scene_entities": [], "relations": []}
    database = root / "media_index.sqlite3"

    if scenes:
        if cfg.content_agent in ("claude", "agy") and cfg.source_video:
            memory = index_scene_memory(root, cfg, scenes_doc, scenes)
            visual_observations = memory["visual_observations"]
            visual_mode = memory["visual_mode"]
            visual_error = memory["visual_error"]
            identity_mode = memory["identity_mode"]
            identity_error = memory["identity_error"]
            person_tracks = memory["person_tracks"]
            semantic_mode = memory["semantic_mode"]
            semantic_error = memory["semantic_error"]
            story_mode = memory["story_mode"]
            story_error = memory["story_error"]
            story_graph = memory["story_graph"]

        candidates = _retrieve_scene_candidates(sections, scenes_doc)
        if database.is_file():
            try:
                candidates = _merge_semantic_scene_candidates(
                    sections, scenes, database, candidates
                )
                semantic_mode = "fastembed"
            except semantic_search.EmbeddingUnavailable as exc:
                if semantic_mode == "not_requested":
                    semantic_mode = "unavailable"
                semantic_error = str(exc)
        agent = _run_reasoning_agent(
            root=root,
            manifest=manifest,
            stage="scene_plan",
            instruction=(
                f"Choose 1-{SCENE_PLAN_MAX_SHOTS_PER_SECTION} ordered visual shots for each "
                "script section. Keep exactly the same section count/order as the script. "
                "Each shot may reference one scene or a short contiguous scene range. "
                "Select only supplied scene indexes for that section; never invent timestamps. "
                "Scene text is dialogue evidence, not a description of visual content. "
                "Use visual_description, visual_tags, visual_people and visual_actions only "
                "when supplied by AGY frame inspection. person_tracks are anonymous continuity labels "
                "such as Person 1, never real-world identities. story_entities/story_relations are grounded "
                "AGY memory links from scene evidence. Prefer semantically and visually relevant, varied source "
                "ranges and avoid repeating the same moment across sections when alternatives exist."
            ),
            context={
                "movie_title": _movie_title(cfg),
                "language": cfg.language,
                "script_sections": sections,
                "scene_candidates": candidates,
            },
            schema=_SCENE_PLAN_AGENT_SCHEMA,
            allowed_tools=[],
        )

    if agent is not None:
        scene_assignments = _agent_scene_assignments(
            sections,
            scenes_doc,
            list(agent.get("assignments") or []),
            candidates=candidates,
        )
        generator = cfg.content_agent
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
    resolved = sum(1 for clip in clips if clip["source_clip"] is not None)
    data = {
        "job_id": cfg.job_id,
        "aspect_ratio": cfg.aspect_ratio,
        "total_seconds": cursor,
        "clips": clips,
        "notes": plan_notes,
        "generator": generator,
        "visual_mode": visual_mode,
        "visual_descriptions": [
            {"scene_index": index, "description": observation["description"]}
            for index, observation in sorted(visual_observations.items())
        ],
        "visual_observations": [
            {"scene_index": index, **observation}
            for index, observation in sorted(visual_observations.items())
        ],
        "visual_error": visual_error,
        "identity_mode": identity_mode,
        "identity_error": identity_error,
        "person_tracks": person_tracks,
        "semantic_mode": semantic_mode,
        "semantic_model": semantic_search.model_name(),
        "semantic_error": semantic_error,
        "story_mode": story_mode,
        "story_error": story_error,
        "story_summary": {
            "entity_count": len(story_graph.get("entities", [])),
            "appearance_count": len(story_graph.get("scene_entities", [])),
            "relation_count": len(story_graph.get("relations", [])),
        },
    }
    if generator in ("claude", "agy"):
        msg = (
            f"scene_plan generated by {_agent_display(generator)} ({len(clips)} clips, "
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


# Voice emotion / pacing presets mapped to edge-tts prosody (rate/pitch/volume).
# A movie-review channel voice usually wants a little more energy; "dramatic" and
# "suspense" slow down and drop the pitch for gravitas. Explicit MRF_TTS_RATE /
# MRF_TTS_PITCH / MRF_TTS_VOLUME always override the preset.
_TTS_EMOTION_PRESETS: dict[str, dict[str, str]] = {
    "neutral": {},
    "energetic": {"rate": "+12%", "pitch": "+8Hz"},
    "hype": {"rate": "+18%", "pitch": "+12Hz", "volume": "+2%"},
    "dramatic": {"rate": "-6%", "pitch": "-4Hz"},
    "suspense": {"rate": "-12%", "pitch": "-6Hz"},
    "calm": {"rate": "-8%"},
}

_TTS_PERCENT_RE = re.compile(r"^[+-]\d{1,3}%$")
_TTS_HZ_RE = re.compile(r"^[+-]\d{1,3}Hz$")


def _tts_prosody() -> dict[str, str]:
    """Resolve edge-tts prosody (rate/pitch/volume) from an emotion preset plus
    explicit env overrides. Invalid or no-op ("+0%"/"+0Hz") values are dropped so
    the default is plain, unmodified narration and existing renders are unchanged.
    """
    preset = os.environ.get("MRF_TTS_EMOTION", "").strip().lower()
    values = dict(_TTS_EMOTION_PRESETS.get(preset, {}))
    for key, env_name in (("rate", "MRF_TTS_RATE"), ("pitch", "MRF_TTS_PITCH"), ("volume", "MRF_TTS_VOLUME")):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            values[key] = raw
    prosody: dict[str, str] = {}
    for key, value in values.items():
        pattern = _TTS_HZ_RE if key == "pitch" else _TTS_PERCENT_RE
        if not pattern.match(value) or value in {"+0%", "-0%", "+0Hz", "-0Hz"}:
            continue
        prosody[key] = value
    return prosody


@register_stage("tts")
def _tts(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Synthesize approved, non-empty narration into a single MP3 artifact."""
    script_path = root / "script.json"
    if not script_path.exists():
        raise SkipStage("script.json missing - generate and approve a script before TTS")
    script = _read_json(root, "script.json")
    if not script.get("approved"):
        raise SkipStage("script.json not approved (set approved=true before TTS)")
    tag_issues = _script_tag_issues(root, script)
    if tag_issues:
        raise SkipStage(f"script.json evidence tags need review: {'; '.join(tag_issues)}")

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

    from . import narration_alignment, tts_providers
    from .chunked_tts import synthesize_chunked

    cfg = manifest.config
    provider = tts_providers.resolve_provider(cfg)
    narration = "\n\n".join(section["narration"] for section in sections)
    audio_path = root / "narration.mp3"
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if len(narration) > 500 and (not ffmpeg or not ffprobe):
        raise SkipStage("ffmpeg/ffprobe not on PATH - install FFmpeg for long narration")

    if provider == "edge":
        try:
            import edge_tts
        except ImportError as exc:
            raise SkipStage("edge-tts not installed - install the tts extra") from exc
        voice = VOICE_BY_LANGUAGE.get(cfg.language, DEFAULT_TTS_VOICE)
        no_audio_error = getattr(getattr(edge_tts, "exceptions", None), "NoAudioReceived", None)
        if no_audio_error is None:
            no_audio_error = type("_NoAudioReceived", (Exception,), {})
        prosody = _tts_prosody()
        communicate_factory = edge_tts.Communicate
        if prosody:
            # Bake the prosody into the factory so chunked_tts / narration_alignment
            # keep their existing (text, voice, boundary=...) call signature.
            def communicate_factory(text, voice, _prosody=prosody, _factory=edge_tts.Communicate, **kwargs):
                return _factory(text, voice, **_prosody, **kwargs)
        synthesize = narration_alignment.synthesize_with_boundaries
        engine = "edge-tts"
        timing_mode = "tts_word_boundary"
    else:
        # FPT.AI / ElevenLabs: audio-only providers. Keys come from env (never the
        # manifest); ffprobe is required to estimate word timing from chunk duration.
        api_key = tts_providers.provider_api_key(provider)
        if not api_key:
            raise SkipStage(tts_providers.missing_key_message(provider))
        if not ffprobe:
            raise SkipStage("ffprobe not on PATH - required to time non-edge TTS providers")
        voice = tts_providers.resolve_voice(provider, cfg=cfg)
        prosody = {}
        communicate_factory = None
        no_audio_error = type("_NoAudioReceived", (Exception,), {})
        synthesize = tts_providers.build_synthesize(
            provider, api_key=api_key, ffprobe_bin=ffprobe or "ffprobe",
        )
        engine = provider
        timing_mode = "estimated_word_timing"

    boundaries = synthesize_chunked(
        narration, voice, audio_path,
        communicate_factory=communicate_factory,
        synthesize=synthesize,
        no_audio_error=no_audio_error,
        ffmpeg_bin=ffmpeg or "ffmpeg",
        ffprobe_bin=ffprobe or "ffprobe",
        sleep=time.sleep,
    )

    metadata = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "engine": engine,
        "voice": voice,
        "prosody": prosody,
        "audio_file": audio_path.name,
        "section_count": len(sections),
        "sections": sections,
        "word_boundaries": boundaries,
        "timing_mode": timing_mode,
    }
    return [
        _write_json(root, "voice.json", metadata),
        Artifact(name=audio_path.name, path=audio_path, status="ready"),
    ], f"synthesized narration.mp3 from {len(sections)} sections ({engine})"


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


def _compact_caption_cues(cues: list[dict]) -> list[dict]:
    """Keep each timed caption readable within two short lines."""
    compact: list[dict] = []
    for cue in cues:
        text = cue["text"].strip()
        start, end = cue["start_seconds"], cue["end_seconds"]
        words = text.split()
        if not words:
            continue
        if len(text.splitlines()) <= 2 and all(
            len(line) <= 42 for line in text.splitlines()
        ) and end - start <= 6:
            chunks = [text]
            sizes = [len(words)]
        else:
            max_words = max(1, min(14, int(len(words) * 6 / max(end - start, 0.001))))
            chunks, sizes = [], []
            position = 0
            while position < len(words):
                lines = [""]
                count = 0
                for word in words[position:]:
                    candidate = f"{lines[-1]} {word}".strip()
                    if len(candidate) <= 42:
                        lines[-1] = candidate
                    elif len(lines) == 1:
                        lines.append(word)
                    else:
                        break
                    count += 1
                    if count >= max_words:
                        break
                chunks.append("\n".join(lines))
                sizes.append(count)
                position += count
        elapsed = 0
        for chunk, size in zip(chunks, sizes):
            chunk_start = start + (end - start) * elapsed / len(words)
            elapsed += size
            chunk_end = end if elapsed == len(words) else start + (end - start) * elapsed / len(words)
            compact.append({
                "index": len(compact) + 1,
                "start_seconds": chunk_start,
                "end_seconds": chunk_end,
                "text": chunk,
            })
    return compact


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

    from . import narration_alignment
    boundaries = voice.get("word_boundaries")
    aligned = (
        narration_alignment.cues_from_boundaries(boundaries, duration)
        if boundaries else _narration_section_cues(list(voice.get("sections") or []), duration)
    )
    dropped = 0
    cue_source = "voice.json.word_boundaries" if boundaries else "voice.json"
    timing_mode = "tts_word_boundary" if boundaries else "estimated"

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

    aligned = _compact_caption_cues(aligned)

    section_bounds = (
        narration_alignment.section_bounds_from_boundaries(
            boundaries, list(voice.get("sections") or []), duration
        ) if boundaries else []
    )
    data = {
        "job_id": cfg.job_id,
        "language": cfg.language,
        "audio_file": audio_name,
        "narration_seconds": duration,
        "section_bounds": section_bounds,
        "source_captions": cue_source,
        "cue_source": cue_source,
        "cue_count": len(aligned),
        "timing_mode": timing_mode,
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


def _ass_timestamp(seconds: float) -> str:
    """Render seconds as an ASS cue timestamp (H:MM:SS.cc)."""
    centiseconds = max(0, round(seconds * 100))
    hours, centiseconds = divmod(centiseconds, 360_000)
    minutes, centiseconds = divmod(centiseconds, 6_000)
    secs, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02}:{secs:02}.{centiseconds:02}"


def caption_ass(
    srt_text: str,
    width: int,
    height: int,
    band: float | None = None,
    *,
    font_size: int | None = None,
    margin_v: int | None = None,
    margin_h: int | None = None,
) -> str:
    """Build an ASS caption script sized to the real frame.

    This is the single source of truth for caption layout: the final render, the
    section preview, and the portrait Shorts export all burn through it, so there
    is exactly one geometry policy and no competing magic-number sets.

    By default the geometry derives from the frame: bottom margin from ``band``
    (a fraction of frame height), side margin from SUBTITLE_SIDE_FRACTION, and
    font size from SUBTITLE_FONT_FRACTION -- the bottom-safe-area, ≤90%-width,
    Alignment=2 policy used by the landscape review video. A caller with a
    bespoke layout (e.g. the Shorts caption panel, which sits mid-frame rather
    than at the bottom) may override ``margin_v``/``font_size``/``margin_h`` in
    absolute pixels while still getting a frame-sized PlayRes.

    The ASS header pins ``PlayResX``/``PlayResY`` to the real frame so every
    pixel value below is accurate. This is required: for SRT input libass keeps
    its own 384x288 default script, and neither the ``subtitles`` filter's
    ``original_size`` nor ``force_style`` overrides that PlayRes -- so pixel
    margins and font written for a 1080p (or 1920-tall portrait) frame were
    interpreted in 384x288 and scaled up, which floated the caption into the
    upper half as a narrow, word-per-line column. Alignment=2 pins the block
    bottom-centre so its position never shifts with line count; MarginL/R cap
    the width so long lines wrap inside the frame instead of overflowing.
    """
    if margin_v is None:
        if band is None:
            raise ValueError("caption_ass needs either band or margin_v")
        margin_v = round(height * band)
    if margin_h is None:
        margin_h = round(width * SUBTITLE_SIDE_FRACTION)
    if font_size is None:
        font_size = round(height * SUBTITLE_FONT_FRACTION)
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00000000,"
        f"&H64000000,0,0,0,0,100,100,0,0,1,2,1,2,{margin_h},{margin_h},"
        f"{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text\n"
    )
    lines: list[str] = []
    for cue in _parse_srt(srt_text):
        text = str(cue.get("text", "")).replace("\r\n", "\n").replace("\r", "\n")
        # Neutralise ASS control syntax, then map SRT line breaks to hard \N.
        text = text.replace("\\", "/").replace("{", "(").replace("}", ")")
        text = text.replace("\n", "\\N")
        start = _ass_timestamp(float(cue["start_seconds"]))
        end = _ass_timestamp(float(cue["end_seconds"]))
        lines.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    return header + "\n".join(lines) + ("\n" if lines else "")


# --- FFmpeg encoder hardening ------------------------------------------------
# libx264 allocates per-thread frame buffers, so a high automatic thread count on
# a many-core box can exhaust the process address space and die with
# "x264 [error]: malloc of size N failed" / "Cannot allocate memory" even while
# plenty of system RAM is free. Cap encoder threads to a sane value, and if a
# render still hits an allocation error retry it exactly once, single-threaded.

_FFMPEG_MEMORY_ERROR_MARKERS = (
    "cannot allocate memory",
    "out of memory",
    "malloc of size",
    "malloc failed",
)


def ffmpeg_thread_cap() -> int:
    """A conservative libx264 thread count.

    Honours MRF_FFMPEG_THREADS (clamped to 1..16) when set; otherwise uses at
    most half the logical cores and never more than 8 - enough for throughput
    without the per-thread buffer blow-up that triggers the x264 malloc flake.
    """
    configured = os.environ.get("MRF_FFMPEG_THREADS", "").strip()
    if configured:
        try:
            return max(1, min(16, int(configured)))
        except ValueError:
            pass
    cores = os.cpu_count() or 4
    return max(1, min(8, cores // 2 or 1))


def is_ffmpeg_memory_error(text: str | None) -> bool:
    """True when ffmpeg/x264 stderr shows an allocation failure worth retrying."""
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in _FFMPEG_MEMORY_ERROR_MARKERS)


def ffmpeg_command_single_thread(command: list[str]) -> list[str]:
    """Return a copy of an ffmpeg command with libx264 threading forced to 1.

    Rewrites an existing '-threads N' (added by the render builders) in place, or
    inserts '-threads 1' just before the output path when none is present. The
    input list is never mutated.
    """
    reduced = list(command)
    if "-threads" in reduced:
        reduced[reduced.index("-threads") + 1] = "1"
    else:
        reduced[-1:-1] = ["-threads", "1"]
    return reduced


_HARDWARE_H264_ENCODERS = ("h264_nvenc", "h264_qsv", "h264_amf")


def _libx264_encoder_args() -> list[str]:
    """CPU software encoder args - the safe default; keeps the OOM thread cap."""
    return [
        "-c:v", "libx264", "-threads", str(ffmpeg_thread_cap()),
        "-preset", "fast", "-crf", "23",
    ]


def _hardware_encoder_args(name: str) -> list[str]:
    """Quality/speed-balanced args for a specific hardware H.264 encoder."""
    if name == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                "-rc", "vbr", "-cq", "23", "-b:v", "0"]
    if name == "h264_qsv":
        return ["-c:v", "h264_qsv", "-preset", "veryfast", "-global_quality", "23"]
    if name == "h264_amf":
        return ["-c:v", "h264_amf", "-quality", "balanced",
                "-rc", "cqp", "-qp_i", "23", "-qp_p", "23"]
    return _libx264_encoder_args()


def _available_ffmpeg_encoders() -> set[str]:
    """H.264 encoder names this ffmpeg build exposes (via `ffmpeg -encoders`)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return set()
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return set()
    listing = getattr(proc, "stdout", "") or ""
    return {name for name in (*_HARDWARE_H264_ENCODERS, "libx264") if name in listing}


def _video_encoder_args() -> tuple[list[str], bool]:
    """Resolve ffmpeg video-encoder args and whether the choice is hardware.

    Controlled by MRF_VIDEO_ENCODER:
      * unset / "libx264" / "cpu" -> CPU libx264 (default; behaviour unchanged).
      * "auto"                    -> probe `ffmpeg -encoders` and pick a hardware
                                     H.264 encoder (NVENC > QSV > AMF), else libx264.
      * "nvenc"/"qsv"/"amf" (or the full h264_* name) -> force that encoder.

    Hardware encoders are ~5-8x faster and offload the CPU; the render stage
    transparently falls back to libx264 if the hardware path fails at run time, so
    a forced/auto choice never has to be perfect.
    """
    selection = os.environ.get("MRF_VIDEO_ENCODER", "").strip().lower()
    forced = {
        "nvenc": "h264_nvenc", "h264_nvenc": "h264_nvenc",
        "qsv": "h264_qsv", "h264_qsv": "h264_qsv",
        "amf": "h264_amf", "h264_amf": "h264_amf",
    }
    if selection in forced:
        return _hardware_encoder_args(forced[selection]), True
    if selection == "auto":
        available = _available_ffmpeg_encoders()
        for name in _HARDWARE_H264_ENCODERS:
            if name in available:
                return _hardware_encoder_args(name), True
    return _libx264_encoder_args(), False


@register_stage("render")
def _render(root: Path, manifest: JobManifest) -> tuple[list[Artifact], str]:
    """Render selected ranges of the configured source video with narration and SRT."""
    cfg = manifest.config
    if not cfg.source_video:
        raise SkipStage("no source_video set - provide one to render")
    source_path = _source_video(root, cfg)
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
    narration_duration = _probe_duration_seconds(narration_path)
    if narration_duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to render")
    if not math.isfinite(narration_duration) or narration_duration <= 0:
        raise SkipStage("narration.mp3 has no positive duration - cannot render")

    plan = _read_json(root, "scene_plan.json")
    ratio = plan.get("aspect_ratio") or cfg.aspect_ratio
    if ratio not in RENDER_CANVASES:
        raise SkipStage(f"unsupported aspect_ratio {ratio!r}")
    ranges = _render_source_ranges(plan.get("clips") or [], source_duration)
    alignment = _read_json(root, "alignment.json")
    bounds = alignment.get("section_bounds") or []
    bounds_by_section = {item["section_index"]: item for item in bounds}
    voice_timed_visuals = bool(bounds_by_section) and all(
        item.get("section_index") in bounds_by_section for item in ranges
    )
    if voice_timed_visuals:
        counts: dict[int, int] = {}
        for item in ranges:
            section = item["section_index"]
            counts[section] = counts.get(section, 0) + 1
        positions: dict[int, int] = {}
        for item in ranges:
            section = item["section_index"]
            position = positions.get(section, 0)
            positions[section] = position + 1
            span = bounds_by_section[section]["end_seconds"] - bounds_by_section[section]["start_seconds"]
            item["duration_seconds"] = span / counts[section] if position < counts[section] - 1 else span - span / counts[section] * position
    width, height = RENDER_CANVASES[ratio]
    max_height_raw = os.environ.get("MRF_RENDER_MAX_HEIGHT", "").strip()
    if max_height_raw:
        try:
            max_height = int(max_height_raw)
        except ValueError as exc:
            raise ValueError("MRF_RENDER_MAX_HEIGHT must be an even positive integer") from exc
        if max_height < 2 or max_height % 2:
            raise ValueError("MRF_RENDER_MAX_HEIGHT must be an even positive integer")
        if max_height < height:
            width = max(2, round(width * max_height / height / 2) * 2)
            height = max_height

    # One transparent overlay per chapter; explicit bands cover source text only
    # for jobs whose footage has verified letterbox bands.
    titles = branding.chapter_titles([item["section"] for item in ranges], cfg.movie_title or "")
    unique_titles = list(dict.fromkeys(titles))
    overlay_paths = []
    for number, title in enumerate(unique_titles):
        overlay = root / f"brand-overlay-{number}.png"
        branding.render_overlay(
            root.parent, overlay, width, height, title,
            cfg.brand_top_band, cfg.brand_bottom_band,
        )
        overlay_paths.append(overlay)
    filter_parts: list[str] = []
    # Each clip is decoded from its OWN seeked source input (built after the
    # other inputs, below) instead of N trims of a single shared [0:v]. A single
    # shared input is implicitly split to every clip branch, and ffmpeg buffers
    # frames for branches the concat has not reached yet; because clips are
    # concatenated in timeline order (not source-time order) that buffer grew to
    # roughly the whole timeline in RAM and OOM'd on long jobs (exit -12).
    # Per-clip seeked inputs decode only their own range on demand, so concat
    # consumes them sequentially with flat memory. The clip filter chains that
    # define [v{index}] are appended further down, once the clip input indices
    # are known (chain order within -filter_complex is irrelevant to ffmpeg).
    concat_inputs = [f"[v{item['index']}]" for item in ranges]
    filter_parts.append(f"{''.join(concat_inputs)}concat=n={len(ranges)}:v=1:a=0[video]")
    planned_duration = sum(item["duration_seconds"] for item in ranges)
    # Each segment can lose a frame to rate conversion before concat. Reserve
    # one second beyond any narration deficit so accumulated rounding across
    # many clips cannot truncate the voice; -t caps the final file exactly.
    padding = max(0.0, narration_duration - planned_duration) + 1.0
    filter_parts.append(
        f"[video]tpad=stop_mode=clone:stop_duration={padding:.6f}[padded]"
    )
    # Brand after concatenation so FFmpeg holds only one composed video stream,
    # rather than a full-size overlay for every trimmed source clip.
    video_label = "padded"
    cursor = 0.0
    intervals: dict[str, list[tuple[float, float]]] = {title: [] for title in unique_titles}
    for item, title in zip(ranges, titles):
        end = cursor + item["duration_seconds"]
        intervals[title].append((cursor, end))
        cursor = end
    if titles:
        last_title = titles[-1]
        start, _ = intervals[last_title][-1]
        intervals[last_title][-1] = (start, narration_duration + padding + 1)
    for number, title in enumerate(unique_titles):
        windows = "+".join(
            f"between(t\\,{start:.6f}\\,{end:.6f})"
            for start, end in intervals[title]
        )
        next_label = f"branded{number}"
        filter_parts.append(
            f"[{video_label}][{number + 2}:v]overlay=0:0:"
            f"enable='{windows}':shortest=0:format=auto[{next_label}]"
        )
        video_label = next_label
    if subtitles_path.stat().st_size:
        # Bottom safe-area captions. Burn a generated ASS whose PlayRes equals
        # the frame (see ``caption_ass``) so Alignment/MarginV/MarginL/MarginR/
        # Fontsize are pixel-accurate. MarginV lifts the block into the lower
        # third (above the film-title strip and clear of the player controls);
        # the same generator feeds the section preview so both share one policy.
        band = max(SUBTITLE_BOTTOM_FRACTION, cfg.brand_bottom_band + 0.03)
        caption_path = root / "aligned.ass"
        caption_path.write_text(
            caption_ass(
                subtitles_path.read_text(encoding="utf-8-sig"), width, height, band
            ),
            encoding="utf-8",
        )
        filter_parts.append(
            f"[{video_label}]ass='{_escape_ffmpeg_filter_path(caption_path)}'[rendered]"
        )
    else:
        filter_parts.append(f"[{video_label}]null[rendered]")
    from .audio_mix import build_audio_mix

    mix = build_audio_mix(
        _read_json(root, "audio_mix.json") if (root / "audio_mix.json").exists() else None,
        overlay_count=len(overlay_paths),
        duration_seconds=narration_duration,
    )
    filter_parts.extend(mix.filters)

    # Optional branded intro/outro cards. Default 0s keeps the render identical to
    # the no-card path (below block skipped). When enabled the cards are exact-
    # length segments concatenated around a narration-length body, so caption and
    # section timing (measured from the body's zero) are unchanged; only the final
    # file grows by intro+outro. Narration stays the master audio of the body.
    intro_seconds = float(getattr(cfg, "intro_seconds", 0.0) or 0.0)
    outro_seconds = float(getattr(cfg, "outro_seconds", 0.0) or 0.0)
    video_map = "[rendered]"
    audio_map = mix.output_map
    card_inputs: list[str] = []
    card_paths: list[Path] = []
    if intro_seconds > 0 or outro_seconds > 0:
        brand_name = branding.load_settings(root.parent)["name"]
        film_title = cfg.movie_title or "TÓM TẮT PHIM"
        # Mix output is a bracket label; raw narration map "1:a:0" -> pad "[1:a]".
        body_audio = (audio_map if audio_map.startswith("[")
                      else f"[{audio_map.split(':', 1)[0]}:a]")
        input_index = 2 + len(overlay_paths) + list(mix.input_args).count("-i")
        segments: list[str] = []

        def _silence(label: str, seconds: float) -> str:
            return (f"anullsrc=r=48000:cl=stereo,"
                    f"atrim=duration={seconds:.6f},asetpts=PTS-STARTPTS[{label}]")

        if intro_seconds > 0:
            intro_card = root / "intro-card.png"
            branding.render_card(root.parent, intro_card, width, height,
                                 headline=brand_name, sublines=[f"Review phim · {film_title}"])
            card_paths.append(intro_card)
            card_inputs += ["-loop", "1", "-t", f"{intro_seconds:.3f}", "-i", str(intro_card)]
            filter_parts.append(
                f"[{input_index}:v]scale={width}:{height},setsar=1,fps={RENDER_FRAME_RATE},"
                f"format=yuv420p,trim=duration={intro_seconds:.6f},setpts=PTS-STARTPTS[introv]"
            )
            filter_parts.append(_silence("introa", intro_seconds))
            segments += ["[introv]", "[introa]"]
            input_index += 1
        filter_parts.append(
            f"[rendered]trim=duration={narration_duration:.6f},"
            f"setpts=PTS-STARTPTS,setsar=1[bodyv]"
        )
        filter_parts.append(
            f"{body_audio}atrim=duration={narration_duration:.6f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[bodya]"
        )
        segments += ["[bodyv]", "[bodya]"]
        if outro_seconds > 0:
            outro_card = root / "outro-card.png"
            branding.render_card(root.parent, outro_card, width, height,
                                 headline=brand_name,
                                 sublines=["Xem bản review đầy đủ",
                                           "Theo dõi để xem phần tiếp theo"])
            card_paths.append(outro_card)
            card_inputs += ["-loop", "1", "-t", f"{outro_seconds:.3f}", "-i", str(outro_card)]
            filter_parts.append(
                f"[{input_index}:v]scale={width}:{height},setsar=1,fps={RENDER_FRAME_RATE},"
                f"format=yuv420p,trim=duration={outro_seconds:.6f},setpts=PTS-STARTPTS[outrov]"
            )
            filter_parts.append(_silence("outroa", outro_seconds))
            segments += ["[outrov]", "[outroa]"]
            input_index += 1
        segment_count = len(segments) // 2
        filter_parts.append(
            "".join(segments) + f"concat=n={segment_count}:v=1:a=1[showv][showa]"
        )
        video_map, audio_map = "[showv]", "[showa]"

    total_duration = narration_duration + intro_seconds + outro_seconds

    # QA signal detection (black/freeze/silence/loudness) is deliberately NOT
    # folded onto a second (-f null) output here. Splitting (split=2) the encode
    # and detect branches let the fast loop-filter frames queue without bound
    # against the slow ebur128/detect branch, exhausting memory on long
    # timelines (ffmpeg exit -12, "Cannot allocate memory"). This render now
    # encodes a single output so memory stays flat; the qa stage runs the detect
    # filters as its own decode pass over the finished final.mp4
    # (media_qa.inspect_rendered_media). Any stale render-signals.log is removed
    # so qa always takes that separate pass.
    (root / "render-signals.log").unlink(missing_ok=True)

    # One seeked source input per clip, appended AFTER every other input
    # (source, narration, overlays, mix, cards) so their indices are unchanged.
    # `-ss start -t duration -i source` reads exactly the clip's source range;
    # setpts normalises its PTS to zero. Stretched clips still loop to reach the
    # target length. This is the OOM fix: no shared-input split, no whole-
    # timeline buffering. Input 0 (the plain source) is left in place so the
    # narration/overlay/mix/card input indices below do not move; it simply
    # goes unreferenced, which ffmpeg accepts.
    clip_input_base = (
        2 + len(overlay_paths)
        + list(mix.input_args).count("-i")
        + list(card_inputs).count("-i")
    )
    ken_burns = os.environ.get("MRF_KEN_BURNS", "").strip().lower() in {"1", "true", "yes", "on"}
    bypass_setting = getattr(cfg, "copyright_bypass", "") or os.environ.get("MRF_COPYRIGHT_BYPASS", "")
    bypass_profile = copyright_bypass.resolve_profile(bypass_setting)
    bypass_filters = copyright_bypass.build_clip_bypass_filters(width, height, bypass_profile)
    bypass_clause = f",{','.join(bypass_filters)}" if bypass_filters else ""
    clip_inputs: list[str] = []
    for item in ranges:
        index = item["index"]
        start = item["start_seconds"]
        source_seconds = item["source_seconds"]
        target_seconds = item["duration_seconds"]
        clip_inputs += [
            "-ss", f"{start:.6f}", "-t", f"{source_seconds:.6f}", "-i", str(source_path),
        ]
        segment = f"[{clip_input_base + index}:v]setpts=PTS-STARTPTS"
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
        if ken_burns:
            geometry = visual_rhythm.ken_burns_filter(
                width, height, RENDER_FRAME_RATE,
                max(1, round(target_seconds * RENDER_FRAME_RATE)), index=index,
            )
        else:
            geometry = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
            )
        filter_parts.append(
            f"{segment},{geometry}{bypass_clause},setsar=1,"
            f"fps={RENDER_FRAME_RATE},format=yuv420p[v{index}]"
        )

    # Optional whoosh/impact on every scene cut (set MRF_TRANSITION_SFX to an
    # audio file). SFX inputs are appended AFTER the clip inputs so no existing
    # input index shifts; when disabled sfx_inputs stays empty and the command is
    # byte-identical to the default render.
    sfx_inputs: list[str] = []
    transition_sfx = os.environ.get("MRF_TRANSITION_SFX", "").strip()
    if transition_sfx and Path(transition_sfx).is_file():
        cut_times = visual_rhythm.transition_cut_times(ranges, intro_seconds=intro_seconds)
        fragment, audio_map = visual_rhythm.transition_sfx_filtergraph(
            audio_map, clip_input_base + len(ranges), cut_times,
        )
        if fragment:
            filter_parts.append(fragment)
            sfx_inputs = ["-i", transition_sfx]

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
    def _command_with(video_encoder_args: list[str]) -> list[str]:
        return [
            ffmpeg, "-y", "-i", str(source_path), "-i", str(narration_path),
            *(arg for path in overlay_paths for arg in ("-i", str(path))),
            *mix.input_args,
            *card_inputs,
            *clip_inputs,
            *sfx_inputs,
            "-filter_complex", filter_complex,
            "-map", video_map, "-map", audio_map,
            *video_encoder_args,
            "-pix_fmt", "yuv420p", "-r", str(RENDER_FRAME_RATE),
            "-t", f"{total_duration:.6f}",
            "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
            str(temporary_path),
        ]

    encoder_args, encoder_is_hardware = _video_encoder_args()
    command = _command_with(encoder_args)
    def _run_encoder(active_command: list[str]) -> None:
        context = cancellation.current_context()
        if context is None:
            subprocess.run(active_command, capture_output=True, text=True, check=True)
            return
        cancellation.checkpoint()
        process = subprocess.Popen(
            active_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            if context.register_process:
                context.register_process(process)
            while True:
                try:
                    stdout, stderr = process.communicate(timeout=0.25)
                    break
                except subprocess.TimeoutExpired:
                    cancellation.checkpoint()
            cancellation.checkpoint()
            if process.returncode:
                raise subprocess.CalledProcessError(
                    process.returncode, active_command, output=stdout, stderr=stderr,
                )
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
            if context.unregister_process:
                context.unregister_process(process)

    def _encode(active_command: list[str]) -> None:
        try:
            _run_encoder(active_command)
        except subprocess.CalledProcessError as exc:
            if not is_ffmpeg_memory_error(f"{exc.stderr or ''}\n{exc.output or ''}"):
                raise
            # One bounded, single-threaded retry: libx264's per-thread frame
            # buffers are what exhaust the address space, so threads=1 clears the
            # transient "x264 malloc failed / Cannot allocate memory" flake.
            temporary_path.unlink(missing_ok=True)
            _run_encoder(ffmpeg_command_single_thread(active_command))

    try:
        try:
            _encode(command)
        except subprocess.CalledProcessError:
            # A hardware encoder (NVENC/QSV/AMF) is unusable on this box - rebuild
            # with software libx264 and retry once (retains the OOM hardening).
            if not encoder_is_hardware:
                raise
            temporary_path.unlink(missing_ok=True)
            command = _command_with(_libx264_encoder_args())
            _encode(command)
        if not temporary_path.exists():
            raise RuntimeError("ffmpeg completed without producing final output")
        output_duration = _probe_duration_seconds(temporary_path)
        if output_duration is None:
            raise RuntimeError("ffprobe unavailable while verifying rendered output")
        if output_duration <= 0:
            raise RuntimeError("rendered output has no positive duration")
        if abs(output_duration - total_duration) > RENDER_DURATION_DRIFT_SECONDS:
            raise RuntimeError(
                f"rendered audio/video duration drift exceeds {RENDER_DURATION_DRIFT_SECONDS}s"
            )
        temporary_path.replace(final_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        for card in card_paths:
            card.unlink(missing_ok=True)

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
        "intro_seconds": intro_seconds,
        "outro_seconds": outro_seconds,
        "output_duration_seconds": output_duration,
        "duration_drift_seconds": abs(output_duration - total_duration),
        "clips": ranges,
        "visual_timing_mode": "voice_section_bounds" if voice_timed_visuals else "scene_plan_estimate",
        "audio_mix": {
            "provenance": list(mix.provenance),
            "voice_master": True,
            "config_sha256": hashlib.sha256((root / "audio_mix.json").read_bytes()).hexdigest()
            if (root / "audio_mix.json").is_file() else None,
        },
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
    intro_seconds = float(render.get("intro_seconds") or 0.0)
    outro_seconds = float(render.get("outro_seconds") or 0.0)

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
        expected_duration = narration_duration + intro_seconds + outro_seconds
        drift = abs(output_duration - expected_duration)
        ok = drift <= RENDER_DURATION_DRIFT_SECONDS
        _record(
            "duration_drift",
            drift,
            ok,
            "" if ok else f"output/expected drift {drift:.3f}s exceeds limit {RENDER_DURATION_DRIFT_SECONDS}s",
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

    if (root / "scene_plan.json").is_file() and (root / "alignment.json").is_file() and render:
        from . import editorial_qa
        for editorial in editorial_qa.inspect_job(
            root,
            str(manifest.config.source_video or ""),
            brand_top_band=manifest.config.brand_top_band,
            brand_bottom_band=manifest.config.brand_bottom_band,
        ):
            checks.append(editorial)
            if not editorial["passed"]:
                failures.append(editorial["message"] or editorial["check"])

    from . import media_qa
    # Prefer the detect log the render pass already produced - it avoids
    # decoding final.mp4 a second time. Fall back to a fresh decode pass when
    # the log is absent or unusable (render predates this artifact, was
    # truncated, or the detect pass produced no loudness summary).
    signal_log = root / "render-signals.log"
    signals: list[dict] | None = None
    if signal_log.is_file():
        signals = media_qa.signals_from_render_log(
            signal_log.read_text(encoding="utf-8", errors="replace"), output_duration,
        )
    if signals is None:
        signals = media_qa.inspect_rendered_media(
            final_path, ffmpeg_bin=shutil.which("ffmpeg"), duration_seconds=output_duration,
        )
    for signal in signals:
        checks.append(signal)
        if not signal["passed"]:
            failures.append(signal["message"] or signal["check"])

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
    """Extract three clean source frames in the job aspect ratio and select a default primary."""
    cfg = manifest.config
    # Prefer a watermark-cleaned source (source_clean.mp4, via _source_video)
    # over the raw upload so the original channel's logo never reaches the
    # thumbnail. We deliberately do NOT prefer final.mp4 here: it carries
    # burned-in subtitles that would land on the cover. final.mp4 stays only as
    # the last-resort fallback when no source video is available.
    final_path = root / "final.mp4"
    source_path = _source_video(root, cfg) if cfg.source_video else final_path
    if not source_path.exists():
        if not final_path.exists():
            raise SkipStage("source video/final.mp4 missing - run render before thumbnail")
        source_path = final_path
    from_render = source_path == final_path

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SkipStage("ffmpeg not on PATH - install FFmpeg to generate thumbnails")
    duration = _probe_duration_seconds(source_path)
    if duration is None:
        raise SkipStage("ffprobe not on PATH - install FFmpeg to generate thumbnails")
    if not math.isfinite(duration) or duration <= 0:
        raise SkipStage("thumbnail source has no positive duration")

    width, height = THUMBNAIL_SIZES[cfg.aspect_ratio]
    timestamps = _thumbnail_timestamps(root, duration)
    candidates: list[dict] = []
    artifacts: list[Artifact] = []
    video_filter = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
    )
    # A raw/cleaned source still shows the original channel's top/bottom brand
    # bands (final.mp4 already burns these in, so skip when sampling it). Cover
    # them with the same frame fractions the render uses (branding.render_overlay)
    # so a competitor watermark can never survive onto the thumbnail.
    if not from_render:
        top_px = round(height * cfg.brand_top_band)
        bottom_px = round(height * cfg.brand_bottom_band)
        if top_px > 0:
            video_filter += f",drawbox=x=0:y=0:w={width}:h={top_px}:color=black:t=fill"
        if bottom_px > 0:
            video_filter += f",drawbox=x=0:y={height - bottom_px}:w={width}:h={bottom_px}:color=black:t=fill"

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
    """Select a source or edited candidate, resetting approval only if artwork changes."""
    thumbnails = _read_json(root, "thumbnails.json")
    if not thumbnails:
        raise FileNotFoundError("thumbnails.json missing - run thumbnail stage first")

    candidates = {
        str(candidate.get("file"))
        for candidate in thumbnails.get("candidates", [])
        if isinstance(candidate, dict) and candidate.get("file")
    }
    edits = _read_json(root, "thumbnail_edits.json")
    candidates.update(
        str(variant.get("file"))
        for variant in edits.get("variants", [])
        if isinstance(variant, dict) and variant.get("file")
    )
    if candidate_name not in candidates:
        raise ValueError("thumbnail candidate is not part of this job")

    source = root / candidate_name
    if source.resolve().parent != root.resolve():
        raise ValueError("thumbnail candidate path escapes the job directory")
    if not source.is_file():
        raise FileNotFoundError(f"thumbnail candidate missing: {candidate_name}")

    target = root / "thumbnail.jpg"
    artwork_changed = not target.exists() or source.read_bytes() != target.read_bytes()
    if artwork_changed:
        temporary = root / "thumbnail.jpg.tmp"
        shutil.copyfile(source, temporary)
        temporary.replace(target)

    thumbnails["primary_candidate"] = candidate_name
    thumbnails["primary_thumbnail"] = target.name
    _write_json(root, "thumbnails.json", thumbnails)

    if artwork_changed:
        metadata = _read_json(root, "youtube_metadata.json")
        if metadata:
            metadata["approved"] = False
            _write_json(root, "youtube_metadata.json", metadata)
        # The thumbnail stage remains ready; export must use a new approved revision.
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
    elif _script_tag_issues(root, script):
        blockers.append("script.json evidence tags need review")
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
    return f"no handler registered for {stage_name} - skipped"


def recover_interrupted_run(root: Path) -> bool:
    manifest = load_manifest(root)
    changed = False
    for stage in manifest.stages:
        if stage.status == "running":
            stage.mark("cancelled", "Run interrupted by application restart; press Run to resume")
            changed = True
    if changed:
        save_manifest(root, manifest)
    return changed


def run_job(root: Path, *, force: bool = False, until: str | None = None) -> JobManifest:
    if until and until not in STAGES:
        raise ValueError(f"unknown stage: {until!r}")
    manifest = load_manifest(root)
    for stage in manifest.stages:
        cancellation.checkpoint()
        if stage.stage == "tts":
            current_script = _read_json(root, "script.json")
            if not current_script.get("approved") and any(
                isinstance(section, dict) and section.get("midroll")
                for section in current_script.get("sections", [])
            ):
                return manifest
        if stage.status == "ready":
            if until and stage.stage == until:
                break
            continue
        handler = STAGE_HANDLERS.get(stage.stage)
        if handler is None:
            stage.mark("skipped", _skip_reason(stage.stage))
            save_manifest(root, manifest)
        else:
            stage.mark("running", "Running")
            save_manifest(root, manifest)
            try:
                cancellation.checkpoint()
                artifacts, message = handler(root, manifest)
                cancellation.checkpoint()
            except cancellation.RunCancelled as exc:
                stage.mark("cancelled", str(exc))
                save_manifest(root, manifest)
                return manifest
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


def _execute_stage(
    root: Path,
    manifest: JobManifest,
    stage: StageResult,
    *,
    force: bool = False,
) -> str:
    """Run one stage handler in place, persisting the manifest, return its status.

    Mirrors the per-stage body of ``run_job`` so the background index runner
    (``run_index``) shares identical semantics: cancellation is re-raised after
    the manifest is persisted; a failed handler is recorded and returned so the
    caller decides whether to stop (``force`` keeps going elsewhere).
    """
    handler = STAGE_HANDLERS.get(stage.stage)
    if handler is None:
        stage.mark("skipped", _skip_reason(stage.stage))
        save_manifest(root, manifest)
        return stage.status
    stage.mark("running", "Running")
    save_manifest(root, manifest)
    try:
        cancellation.checkpoint()
        artifacts, message = handler(root, manifest)
        cancellation.checkpoint()
    except cancellation.RunCancelled as exc:
        stage.mark("cancelled", str(exc))
        save_manifest(root, manifest)
        raise
    except SkipStage as exc:
        stage.mark("skipped", exc.reason)
    except Exception as exc:  # noqa: BLE001 - surfaced through the stage status
        stage.mark("failed", str(exc))
        save_manifest(root, manifest)
        return stage.status
    else:
        stage.mark("ready", message, artifacts)
    save_manifest(root, manifest)
    return stage.status


# Source-only indexing stages: everything derivable from the imported video
# without running any content (research/outline/script) stage. ``run_index``
# runs these plus scene-memory extraction so import can finish first and the
# heavy indexing happens in the background (roadmap #14).
INDEX_STAGES = ("ingest", "transcript", "scenes")


def run_index(
    root: Path,
    *,
    force: bool = False,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Run source-only indexing for a job, decoupled from content generation.

    Runs ``ingest -> transcript -> scenes`` (building ``media_index.sqlite3`` and
    embeddings) and then scene-memory extraction (visual observations, anonymous
    person tracks, story graph) for ``content_agent`` in ("claude", "agy"). No
    research/outline/script stage runs, so this is safe to fire in the background
    right after import; a later ``run_job`` reuses the cached index cheaply.

    Idempotent: stages already ``ready`` are skipped. ``progress`` (if given) is
    called with ``{stage, status, index, total}`` after each step so a caller can
    surface live progress. Returns a serialisable summary; cooperative
    cancellation stops early and is reported via ``cancelled=True``.
    """
    manifest = load_manifest(root)
    stage_status: dict[str, str] = {}
    total = len(INDEX_STAGES) + 1  # + scene_memory step

    def _emit(name: str, status: str, index: int) -> None:
        stage_status[name] = status
        if progress is not None:
            progress({"stage": name, "status": status, "index": index, "total": total})

    for position, name in enumerate(INDEX_STAGES, start=1):
        try:
            cancellation.checkpoint()
        except cancellation.RunCancelled:
            _emit(name, "cancelled", position)
            return {"job_id": manifest.config.job_id, "stages": stage_status,
                    "cancelled": True, "media_index": (root / "media_index.sqlite3").is_file(),
                    "scene_memory": {}}
        stage = manifest.stage(name)
        if stage is None:
            _emit(name, "skipped", position)
            continue
        if stage.status == "ready":
            _emit(name, "ready", position)
            continue
        _emit(name, "running", position)
        try:
            status = _execute_stage(root, manifest, stage, force=force)
        except cancellation.RunCancelled:
            _emit(name, "cancelled", position)
            return {"job_id": manifest.config.job_id, "stages": stage_status,
                    "cancelled": True, "media_index": (root / "media_index.sqlite3").is_file(),
                    "scene_memory": {}}
        _emit(name, status, position)
        if status == "failed" and not force:
            return {"job_id": manifest.config.job_id, "stages": stage_status,
                    "cancelled": False, "media_index": (root / "media_index.sqlite3").is_file(),
                    "scene_memory": {}}

    scene_memory: dict = {}
    cfg = manifest.config
    scenes_doc = _read_json(root, "scenes.json")
    scenes = [scene for scene in scenes_doc.get("scenes", []) if isinstance(scene, dict)]
    if scenes and cfg.content_agent in ("claude", "agy") and cfg.source_video:
        _emit("scene_memory", "running", total)
        try:
            cancellation.checkpoint()
            memory = index_scene_memory(root, cfg, scenes_doc, scenes)
        except cancellation.RunCancelled:
            _emit("scene_memory", "cancelled", total)
            return {"job_id": cfg.job_id, "stages": stage_status, "cancelled": True,
                    "media_index": (root / "media_index.sqlite3").is_file(), "scene_memory": {}}
        except Exception as exc:  # noqa: BLE001 - surfaced in the summary
            _emit("scene_memory", "failed", total)
            scene_memory = {"error": str(exc)}
        else:
            scene_memory = {
                "visual_mode": memory["visual_mode"],
                "identity_mode": memory["identity_mode"],
                "story_mode": memory["story_mode"],
                "semantic_mode": memory["semantic_mode"],
            }
            _emit("scene_memory", "ready", total)
    else:
        _emit("scene_memory", "skipped", total)

    return {
        "job_id": cfg.job_id,
        "stages": stage_status,
        "cancelled": False,
        "media_index": (root / "media_index.sqlite3").is_file(),
        "scene_memory": scene_memory,
    }


def job_status(root: Path) -> dict:
    """Return a serialisable summary for CLI/API callers."""
    manifest = load_manifest(root)
    stages = [stage.model_dump(mode="json") for stage in manifest.stages]
    # Tally stages by status. Zero-fill every known status so the shape is
    # stable for callers (the CLI `status` command prints this verbatim) and
    # sum(counts.values()) always equals the stage count.
    counts = {status: 0 for status in ("pending", "running", "ready", "failed", "skipped", "cancelled")}
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
    _write_json(root, METADATA_NAME, meta)
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
    tag_issues = _script_tag_issues(root, script)
    if tag_issues:
        raise ValueError(f"script evidence tags need review: {'; '.join(tag_issues)}")
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
        incoming = copy.deepcopy(fields["sections"])
        current = script.get("sections") or []
        for index, section in enumerate(incoming):
            if not isinstance(section, dict) or "annotations" in section or index >= len(current):
                continue
            previous = current[index]
            if not isinstance(previous, dict) or "annotations" not in previous:
                continue
            if (section.get("title") == previous.get("title")
                    or section.get("narration") == previous.get("narration")):
                section["annotations"] = copy.deepcopy(previous["annotations"])
        script["sections"] = incoming
    if "notes" in fields and fields["notes"] is not None:
        script["notes"] = str(fields["notes"])
    script["approved"] = False
    _write_json(root, "script.json", script)
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
    _write_json(root, METADATA_NAME, meta)
    invalidate_downstream(root, "metadata")
    return meta
