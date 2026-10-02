"""Local web UI + JSON API for the movie-review-factory pipeline.

Python-native and dependency-free: it uses only the standard library
(``http.server`` + ``threading``) so it adds nothing to pyproject and needs no
JavaScript build step. The page is a single inline HTML document that talks to
a small JSON API. Bind to localhost - this is a single-operator tool, not a
public service.

Publishing safety is built into the shape of the API, not just the UI:

* Creating a project and importing its source never start the pipeline;
  every run is triggered explicitly by the operator.
* The web "run" action stops at the ``script`` stage for review and only
  advances toward the ``thumbnail`` stage once the script is approved - it
  never reaches ``publish`` on its own.
* Metadata approval (``approved=true``) is a separate, explicit endpoint.
* Publishing requires an explicit ``confirm`` and, even then, only runs the
  ``publish`` stage - which merely writes the handoff record
  ``publish_record.json`` when approvals are complete. Nothing is ever uploaded.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import ipaddress
import queue
import re
import shutil
import socket
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import analytics, audio_mix, branding, cancellation, content_scout, creative_brief, editor_ops, hook_crafter, licensing, link_download, localization, mask_detection, midroll, narration_style, pipeline, propainter_setup, semantic_search, tts_providers, versions, visual_variety
from .content_agent import _terminate_process_tree
from .media_store import MediaStore
from .creator_library import CreatorLibrary
from . import media_intelligence
from .models import CONTENT_AGENT_MODES, WATERMARK_METHODS, JobConfig

# A job id / artifact name must be a single safe path segment. This is the only thing standing between a URL and the filesystem, so it is deliberately strict.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")

# The web "run" button intentionally stops here; publish is a separate action.
RUN_UNTIL_STAGE = "thumbnail"
SCRIPT_REVIEW_STAGE = "script"

# Optional bearer-token auth.  Set DASHBOARD_TOKEN in the environment to require
# a token on every request.  When the variable is absent or empty every request
# is allowed — explicit local-dev mode, no silent open access on a deployed host.
_DASHBOARD_TOKEN: str = os.environ.get("DASHBOARD_TOKEN", "")

# Maximum JSON request-body size accepted before returning 413.
_MAX_BODY_BYTES = 1_048_576  # 1 MiB
_REQUEST_TIMEOUT_SECONDS = 15.0
_MAX_CONCURRENT_HANDLERS = 32


def _is_loopback_host(host: str) -> bool:
    """Return whether a bind target is unambiguously local-only."""
    candidate = host.strip().lower().rstrip(".")
    if candidate == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def _require_safe_bind(host: str) -> None:
    if not _is_loopback_host(host) and not _DASHBOARD_TOKEN.strip():
        raise ValueError("DASHBOARD_TOKEN is required when binding to a non-loopback host")


class RequestTimeoutError(TimeoutError):
    pass


# Conservative security headers added to every response.
_SECURITY_HEADERS: dict[str, str] = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    # The page uses only inline scripts/styles and self-hosted media.
    "Content-Security-Policy": (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "media-src 'self' blob:; "
        "img-src 'self' data: blob:; "
        "connect-src 'self'"
    ),
}

_CONTENT_TYPES = {
    ".mp4": "video/mp4",
    ".mp3": "audio/mpeg",
    ".json": "application/json; charset=utf-8",
    ".srt": "text/plain; charset=utf-8",
    ".vtt": "text/vtt; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
    ".txt": "text/plain; charset=utf-8",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".zip": "application/zip",
    ".woff2": "font/woff2",
}

_UI_FONTS = branding.ASSETS / "fonts"

_ARTIFACT_KINDS = {
    ".mp4": "video",
    ".mp3": "audio",
    ".json": "json",
    ".srt": "subtitle",
    ".vtt": "subtitle",
    ".md": "markdown",
    ".txt": "text",
    ".jpg": "image",
    ".jpeg": "image",
    ".zip": "archive",
}


def _is_safe_segment(name: str) -> bool:
    return bool(_SAFE_SEGMENT.match(name)) and name not in (".", "..")


def _content_type(name: str) -> str:
    return _CONTENT_TYPES.get(Path(name).suffix.lower(), "application/octet-stream")


def _artifact_kind(name: str) -> str:
    return _ARTIFACT_KINDS.get(Path(name).suffix.lower(), "other")


def _format_vtt_timestamp(seconds: float) -> str:
    """Render seconds as a WebVTT cue timestamp (HH:MM:SS.mmm)."""
    milliseconds = int(round(max(0.0, seconds) * 1000))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


def _render_webvtt(segments) -> str:
    """Serialize indexed transcript segments into a valid WebVTT document.

    Speaker names, when present, are emitted as WebVTT voice tags so players
    can attribute dialogue - the web-native complement to the SRT captions the
    transcript stage already writes.
    """
    lines = ["WEBVTT", ""]
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{_format_vtt_timestamp(segment.start_seconds)} --> "
            f"{_format_vtt_timestamp(segment.end_seconds)}"
        )
        text = segment.text
        if segment.speaker:
            text = f"<v {segment.speaker}>{text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


# Whitelist (pip name -> import module) for POST /api/system/install-package so arbitrary input never reaches pip; token-gated on a localhost bind.
_INSTALLABLE_PACKAGES = {
    "vieneu": "vieneu",
    "edge-tts": "edge_tts",
}


class JobsService:
    """Filesystem-backed operations behind the API, decoupled from HTTP.

    Wraps a ``jobs_root`` directory and tracks in-memory run state per job so
    the UI can poll progress. All heavy pipeline work is delegated to
    ``pipeline``; this class only adds job discovery, localization, background
    running, and the explicit approval/publish gates.
    """

    def __init__(self, jobs_root: Path):
        self.jobs_root = Path(jobs_root)
        self.creator_library = CreatorLibrary(self.jobs_root)
        self._runs: dict[str, dict] = {}
        self._short_exports: dict[str, dict] = {}
        self._section_previews: dict[str, dict] = {}
        # Tracks the "re-index dialogue" action (reset + re-run transcript+scenes)
        # per job so the dashboard can disable the button and poll progress.
        self._reindex: dict[str, dict] = {}
        self._uploads: set[str] = set()
        self._deleting: set[str] = set()
        # Overnight batch queue (roadmap P2.1): one sequential worker runs many
        # pasted links to the script-review gate; state is polled by the dashboard.
        self._batch: dict = {"running": False, "stop": False, "items": [], "started_at": None}
        self._lock = threading.Lock()
        self._version_lock = threading.Lock()
        # 1-click TTS library installer (POST /api/system/install-package): per-package
        # {status: running|done|error} tracked here and polled by the settings UI.
        self._installs: dict[str, dict] = {}
        # Background indexing queue (roadmap #14): a single FIFO worker builds
        # media_index/embeddings/scene-memory for imported jobs one at a time so
        # import returns immediately and other projects stay usable meanwhile.
        self._indexing: dict[str, dict] = {}
        self._index_queue: "queue.Queue[str | None]" = queue.Queue()
        self._index_cv = threading.Condition(self._lock)
        self._closed = threading.Event()
        self._index_auto = os.environ.get("MRF_AUTO_INDEX", "0").strip().lower() not in (
            "0", "false", "no", "off",
        )
        self._index_worker_thread = threading.Thread(
            target=self._index_worker, name="mrf-index-queue", daemon=True
        )
        self._index_worker_thread.start()
        for job in pipeline.list_jobs(self.jobs_root):
            pipeline.recover_interrupted_run(self._job_root(job["job_id"]))

    # -- helpers -------------------------------------------------------------

    def close(self, timeout: float = 5.0) -> None:
        """Stop accepting background work and boundedly join owned workers."""
        if self._closed.is_set():
            return
        self._closed.set()
        with self._lock:
            self._batch["stop"] = True
            for state in self._indexing.values():
                state["cancel"].set()
            for state in self._runs.values():
                event = state.get("event")
                if event is not None:
                    event.set()
        self._index_queue.put(None)
        if self._index_worker_thread is not threading.current_thread():
            self._index_worker_thread.join(max(0.0, timeout))

    shutdown = close

    def _ensure_open(self) -> None:
        if self._closed.is_set():
            raise RuntimeError("service is shutting down")

    def _job_root(self, job_id: str) -> Path:
        if not _is_safe_segment(job_id):
            raise ValueError(f"job_id không hợp lệ: {job_id!r}")
        return self.jobs_root / job_id

    def _require_job(self, job_id: str) -> Path:
        root = self._job_root(job_id)
        if not pipeline.manifest_path(root).exists():
            raise FileNotFoundError(f"không tìm thấy job: {job_id}")
        return root

    @staticmethod
    def _read_json(root: Path, name: str) -> dict:
        path = root / name
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {}

    def _is_running(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._runs.get(job_id, {}).get("running"))

    def _assert_no_preview(self, job_id: str) -> None:
        with self._lock:
            if self._section_previews.get(job_id, {}).get("running"):
                raise RuntimeError("Chờ dựng preview phần hiện tại xong trước khi sửa project.")

    # -- read ----------------------------------------------------------------

    def list_creator_series(self) -> dict:
        return {"series": self.creator_library.list_series()}

    def save_creator_series(self, payload: dict) -> dict:
        return self.creator_library.save_series(
            payload.get("series_id"), payload.get("title"), payload.get("entries")
        )

    def list_creator_briefs(self) -> dict:
        return {"briefs": self.creator_library.list_briefs()}

    def save_creator_brief(self, payload: dict) -> dict:
        return self.creator_library.save_brief(payload.get("name"), payload.get("brief"))

    def search_creator_projects(self, query: str) -> dict:
        return {"projects": self.creator_library.search_projects(query)}

    def list_channels(self) -> dict:
        return self.creator_library.list_channels()

    def save_channel(self, payload: dict) -> dict:
        return self.creator_library.save_channel(str(payload.get("id") or ""), payload)

    def activate_channel(self, profile_id: str) -> dict:
        return self.creator_library.activate_channel(profile_id)

    def delete_channel(self, profile_id: str) -> dict:
        self.creator_library.delete_channel(profile_id)
        return {"deleted": profile_id, **self.creator_library.list_channels()}

    def save_channel_logo(self, profile_id: str, data: bytes) -> dict:
        self.creator_library.set_channel_logo(profile_id, data)
        return {"logo_url": f"/api/channels/{profile_id}/logo"}

    def channel_logo_file(self, profile_id: str) -> Path:
        return branding.channel_logo_path(self.jobs_root, profile_id) or branding.logo_path(self.jobs_root)

    def save_channel_sfx(self, profile_id: str, payload: dict) -> dict:
        return self.creator_library.save_channel_sfx(profile_id, payload)

    def save_channel_sfx_file(self, profile_id: str, slug: str, content_type: str, data: bytes) -> dict:
        return self.creator_library.set_channel_sfx_file(profile_id, slug, content_type, data)

    def delete_channel_sfx(self, profile_id: str, slug: str) -> dict:
        return self.creator_library.delete_channel_sfx(profile_id, slug)

    def channel_sfx_file(self, profile_id: str, slug: str) -> Path:
        path = branding.channel_sfx_path(self.jobs_root, profile_id, slug)
        if path is None:
            raise FileNotFoundError("SFX chưa có tệp")
        return path

    def active_channel_sfx(self) -> dict:
        return self.creator_library.active_channel_sfx()

    def add_audio_mix_sfx(self, job_id: str, payload: dict) -> dict:
        """Append an active-channel SFX to a job's audio mix at the given timestamp."""
        root = self._require_job(job_id)
        path, meta = self.creator_library.resolve_active_sfx(str(payload.get("slug") or ""))
        config = self._read_json(root, "audio_mix.json") or {"voice_gain_db": 0, "music": None, "effects": []}
        effects = list(config.get("effects") or [])
        if len(effects) >= 16:
            raise ValueError("Đã đạt tối đa 16 hiệu ứng âm thanh; xoá bớt trước khi thêm.")
        try:
            at = float(payload.get("at_seconds", 0))
        except (TypeError, ValueError):
            at = 0.0
        effects.append({
            "path": str(path),
            "rights_note": meta.get("rights_note", ""),
            "gain_db": float(meta.get("gain_db", -8)),
            "at_seconds": max(0.0, at),
        })
        return self.update_audio_mix(job_id, {
            "voice_gain_db": config.get("voice_gain_db", 0),
            "music": config.get("music"),
            "effects": effects,
        })

    def place_transition_sfx(self, job_id: str) -> dict:
        """Auto-place channel SFX at each scene cut using render.json timings (mode B).

        Cut offsets are body-relative cumulative clip durations from render.json;
        auto effects are tagged so a re-run replaces the previous set rather than
        stacking. Requires a prior render so the cut times are exact.
        """
        root = self._require_job(job_id)
        render = self._read_json(root, "render.json")
        if not render or not render.get("clips"):
            raise ValueError("Cần dựng video một lần trước để biết mốc cắt cảnh, rồi tự rải SFX và dựng lại.")
        durations = [float(clip.get("duration_seconds") or 0) for clip in render["clips"]]
        cuts = pipeline.transition_cut_offsets(durations)
        narration = float(render.get("narration_duration_seconds") or 0)
        if narration > 0:
            cuts = [cut for cut in cuts if cut < narration - 0.05]
        if not cuts:
            raise ValueError("Không tìm thấy điểm chuyển cảnh phù hợp để chèn SFX.")
        palette = self.creator_library.active_channel_sfx()["sfx"]
        if not palette:
            raise ValueError("Kênh đang dùng chưa có SFX (đã tải tệp). Thêm SFX trong mục Kênh trước.")
        config = self._read_json(root, "audio_mix.json") or {"voice_gain_db": 0, "music": None, "effects": []}
        manual = [e for e in (config.get("effects") or []) if e.get("source") != "channel-transition"]
        budget = 16 - len(manual)
        if budget <= 0:
            raise ValueError("Đã đủ 16 hiệu ứng âm thanh; xoá bớt trước khi tự rải.")
        chosen = pipeline.select_evenly(cuts, budget)
        resolved = {item["slug"]: self.creator_library.resolve_active_sfx(item["slug"]) for item in palette}
        auto = []
        for position, at in enumerate(chosen):
            path, meta = resolved[palette[position % len(palette)]["slug"]]
            auto.append({
                "path": str(path),
                "rights_note": meta.get("rights_note", ""),
                "gain_db": float(meta.get("gain_db", -8)),
                "at_seconds": float(at),
                "source": "channel-transition",
            })
        result = self.update_audio_mix(job_id, {
            "voice_gain_db": config.get("voice_gain_db", 0),
            "music": config.get("music"),
            "effects": manual + auto,
        })
        result["placed"] = len(auto)
        return result

    def list_creator_rights(self, job_id: str) -> dict:
        self._require_job(job_id)
        return {"rights": self.creator_library.list_asset_rights(job_id)}

    def save_creator_rights(self, job_id: str, payload: dict) -> dict:
        self._require_job(job_id)
        return self.creator_library.record_asset_rights(
            job_id, payload.get("path"), source=payload.get("source"),
            usage=payload.get("usage"),
            permission_status=payload.get("permission_status", "unreviewed"),
            evidence_note=str(payload.get("evidence_note") or ""),
        )

    def list_jobs(self) -> list[dict]:
        jobs = pipeline.list_jobs(self.jobs_root)
        for job in jobs:
            job["running"] = self._is_running(job["job_id"])
            index_state = self.index_state(job["job_id"])
            job["indexing"] = bool(index_state and (index_state["queued"] or index_state["running"]))
            with self._lock:
                job["uploading"] = job["job_id"] in self._uploads
        return jobs

    def status(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        info = pipeline.job_status(root)
        for stage in info["stages"]:
            stage["stage_label"] = localization.stage_label(stage["stage"])
            stage["status_label"] = localization.status_label(stage["status"])
            stage["status_hint"] = localization.status_hint(stage["status"])
            stage["message_vi"] = localization.localize_message(stage.get("message", ""))

        with self._lock:
            run_state = dict(self._runs.get(job_id, {}))
        run_error = run_state.get("error")
        info["running"] = bool(run_state.get("running"))
        with self._lock:
            short_state = dict(self._short_exports.get(
                job_id, {"running": False, "error": None, "href": None, "srt_href": None}
            ))
        if not short_state["running"] and (root / "shorts" / "short-review.mp4").is_file():
            from .short_variants import short_artifact

            try:
                short_artifact(root, "short-review.mp4", verify_output=False)
            except (ValueError, FileNotFoundError):
                short_state.update(href=None, srt_href=None, error="Video ngắn đã cũ; xuất lại.")
            else:
                short_state.update(
                    href=f"/api/jobs/{job_id}/shorts/short-review.mp4",
                    srt_href=f"/api/jobs/{job_id}/shorts/short-review.srt",
                )
        info["short_export"] = short_state
        with self._lock:
            info["section_preview"] = dict(self._section_previews.get(job_id, {"running": False}))
            info["reindex"] = dict(self._reindex.get(job_id, {"running": False}))
        info["stopping"] = bool(run_state.get("stopping"))
        info["cancelled"] = any(stage["status"] == "cancelled" for stage in info["stages"])
        info["can_stop"] = info["running"] and not info["stopping"]
        with self._lock:
            info["uploading"] = job_id in self._uploads
        cfg = pipeline.load_manifest(root).config
        info["has_source_video"] = bool(cfg.source_video)
        info["brand_top_band"] = cfg.brand_top_band
        info["brand_bottom_band"] = cfg.brand_bottom_band
        info["config"] = cfg.model_dump(mode="json")
        info["run_error"] = run_error
        info["run_error_vi"] = localization.localize_message(run_error) if run_error else None

        script = self._read_json(root, "script.json")
        meta = self._read_json(root, pipeline.METADATA_NAME)
        info["approvals"] = {
            "script_present": bool(script),
            "script_approved": bool(script.get("approved")),
            "metadata_present": bool(meta),
            "metadata_approved": bool(meta.get("approved")),
        }
        qa = self._read_json(root, "qa.json")
        info["qa_findings"] = [
            {"check": check.get("check"), "message": check.get("message"),
             "passed": check.get("passed"), "review_required": bool(check.get("review_required")),
             "value": check.get("value")}
            for check in qa.get("checks", []) if isinstance(check, dict)
            and (check.get("passed") is False or check.get("review_required"))
        ]
        voice_meta = self._read_json(root, "voice.json")
        info["voice"] = {
            "engine": voice_meta.get("engine"),
            "voice": voice_meta.get("voice"),
            "timing_mode": voice_meta.get("timing_mode"),
        } if voice_meta else None
        info["artifacts"] = self.list_artifacts(job_id)
        info["has_final_video"] = (root / "final.mp4").exists()
        info["has_thumbnail"] = (root / "thumbnail.jpg").exists()
        info["has_media_index"] = (root / "media_index.sqlite3").exists()
        index_state = self.index_state(job_id)
        info["indexing"] = index_state
        info["is_indexing"] = bool(index_state and (index_state["queued"] or index_state["running"]))
        return info

    def list_artifacts(self, job_id: str) -> list[dict]:
        root = self._job_root(job_id)
        if not root.exists():
            return []
        items: list[dict] = []
        for path in sorted(root.iterdir()):
            if not path.is_file() or path.name == pipeline.MANIFEST_NAME or path.suffix == ".tmp":
                continue
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                continue
            items.append({
                "name": path.name,
                "size": size,
                "kind": _artifact_kind(path.name),
                "href": f"/api/jobs/{job_id}/artifacts/{path.name}",
            })
        return items

    def get_metadata(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        meta = self._read_json(root, pipeline.METADATA_NAME)
        return {"present": bool(meta), "metadata": meta}

    def get_thumbnails(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        thumbnails = self._read_json(root, "thumbnails.json")
        return {
            "present": bool(thumbnails), "thumbnails": thumbnails,
            "edits": self._read_json(root, "thumbnail_edits.json"),
            "channel_name": branding.load_settings(self.jobs_root)["name"],
        }

    def person_tracks(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        database = root / "media_index.sqlite3"
        if not database.is_file():
            return {"tracks": []}
        with MediaStore(database) as store:
            store.migrate()
            return {"tracks": store.list_person_tracks()}

    def set_person_alias(self, job_id: str, label: str, alias: str) -> dict:
        root = self._require_job(job_id)
        database = root / "media_index.sqlite3"
        if not database.is_file():
            raise FileNotFoundError("media index is not available")
        with MediaStore(database) as store:
            store.migrate()
            try:
                track = store.set_person_alias(label, alias)
            except KeyError as exc:
                raise FileNotFoundError(f"person track not found: {label}") from exc
            try:
                semantic_search.refresh_store_embeddings(store)
            except semantic_search.EmbeddingUnavailable:
                pass
        return {"track": track}

    def timeline(self, job_id: str) -> dict:
        return editor_ops.timeline(self._require_job(job_id))

    def edit_timeline(self, job_id: str, operation: dict) -> dict:
        if self._is_running(job_id):
            raise RuntimeError("job is running")
        self._assert_no_preview(job_id)
        with self._version_lock:
            root = self._require_job(job_id)
            versions.create(root, "Trước khi sửa timeline")
            return editor_ops.edit_timeline(root, operation)

    def similar_scenes(self, job_id: str, shot_id: int) -> dict:
        return {
            "results": [
                {
                    **item,
                    "id": item["shot_id"],
                    "thumbnail_href": f"/api/jobs/{job_id}/shots/{item['shot_id']}/thumbnail",
                }
                for item in editor_ops.similar_scenes(self._require_job(job_id), shot_id)
            ]
        }

    def broll(self, job_id: str, clip_index: int, *, apply: bool = False) -> dict:
        root = self._require_job(job_id)
        if apply:
            if self._is_running(job_id):
                raise RuntimeError("job is running")
            self._assert_no_preview(job_id)
            with self._version_lock:
                versions.create(root, "Trước khi thay cảnh tự động")
                return editor_ops.auto_replace_broll(root, apply=True)
        return {
            "clip_index": clip_index,
            "suggestions": editor_ops.broll_suggestions(root, clip_index),
        }

    def regenerate_section(
        self, job_id: str, section_index: int, instruction: str = ""
    ) -> dict:
        if self._is_running(job_id):
            raise RuntimeError("job is running")
        self._assert_no_preview(job_id)
        with self._version_lock:
            root = self._require_job(job_id)
            versions.create(root, "Trước khi tạo lại cảnh")
            return editor_ops.regenerate_section(
                root,
                section_index,
                instruction=instruction,
            )

    def _search_history_path(self) -> Path:
        return self.jobs_root / ".library-searches.json"

    def list_saved_searches(self) -> dict:
        path = self._search_history_path()
        if not path.is_file():
            return {"searches": []}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        return {"searches": data if isinstance(data, list) else []}

    def save_search(self, body: dict) -> dict:
        query = str(body.get("query") or "").strip()
        if not query or len(query) > 200:
            raise ValueError("search query must contain 1 to 200 characters")
        item = {
            "name": str(body.get("name") or query)[:120],
            "query": query,
            "kind": str(body.get("kind") or ""),
            "person": str(body.get("person") or ""),
            "action": str(body.get("action") or ""),
            "location": str(body.get("location") or ""),
            "object": str(body.get("object") or ""),
            "project": str(body.get("project") or ""),
            "scene_type": str(body.get("scene_type") or ""),
            "source": str(body.get("source") or ""),
            "date_from": str(body.get("date_from") or ""),
            "date_to": str(body.get("date_to") or ""),
            "min_duration": body.get("min_duration"),
            "max_duration": body.get("max_duration"),
            "min_confidence": body.get("min_confidence"),
        }
        path = self._search_history_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.list_saved_searches()["searches"]
        existing = [
            saved for saved in existing
            if not (
                isinstance(saved, dict)
                and saved.get("name") == item["name"]
                and saved.get("query") == item["query"]
            )
        ]
        existing.insert(0, item)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(existing[:30], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)
        return {"search": item, "searches": existing[:30]}

    def media_explorer(self, job_id: str, query: str = "") -> dict:
        root = self._require_job(job_id)
        database_path = root / "media_index.sqlite3"
        if not database_path.exists():
            return {
                "present": False,
                "query": query,
                "media_href": None,
                "source_language": None,
                "transcript_vtt_href": None,
                "shots": [],
                "transcript": [],
            }
        with MediaStore(database_path) as store:
            store.migrate()
            asset = store.get_media_asset(1)
            full_transcript = store.list_transcript(1)
            transcript = store.search_transcript(query, 1) if query else full_transcript
            visual_rows = (
                store.search_visual_observations(query)
                if query else store.list_visual_observations()
            )
            visual_by_shot = {int(item["shot_id"]): item for item in visual_rows}
            shots = store.search_shots(query, 1) if query else store.list_shots(1)
            semantic_mode = "not_requested"
            semantic_scores: dict[tuple[str, int], float] = {}
            if query:
                try:
                    has_embeddings = bool(
                        store.list_shot_embeddings(semantic_search.model_name())
                        or store.list_transcript_embeddings(semantic_search.model_name())
                    )
                    semantic_hits = (
                        semantic_search.search_store(store, query, limit=30)
                        if has_embeddings else []
                    )
                    semantic_mode = "fastembed" if has_embeddings else "unavailable"
                    for hit in semantic_hits:
                        entity_id = int(hit.get("shot_id") or hit.get("segment_id") or 0)
                        semantic_scores[(hit["kind"], entity_id)] = float(hit["score"])
                        if hit["kind"] == "visual":
                            visual_by_shot.setdefault(entity_id, {})
                            if all(int(shot.id or 0) != entity_id for shot in shots):
                                shot = store.get_shot(entity_id)
                                if shot is not None:
                                    shots.append(shot)
                        elif hit["kind"] == "transcript":
                            segment = store.get_transcript_segment(entity_id)
                            if segment is not None and all(item.id != segment.id for item in transcript):
                                transcript.append(segment)
                except semantic_search.EmbeddingUnavailable:
                    semantic_mode = "unavailable"
                found = {int(shot.id or 0) for shot in shots}
                shots.extend(
                    shot for shot_id in visual_by_shot
                    if shot_id not in found
                    for shot in [store.get_shot(shot_id)]
                    if shot is not None
                )
                shots.sort(key=lambda shot: (
                    -semantic_scores.get(("visual", int(shot.id or 0)), -2.0),
                    shot.start_seconds,
                ))
                transcript.sort(key=lambda segment: (
                    -semantic_scores.get(("transcript", int(segment.id or 0)), -2.0),
                    segment.start_seconds,
                ))
            all_visual = store.list_visual_observations()
            all_visual_by_shot = {int(item["shot_id"]): item for item in all_visual}
            person_labels = {
                int(shot.id): store.person_labels_for_shot(int(shot.id))
                for shot in store.list_shots(1) if shot.id
            }
        has_transcript = bool(full_transcript)
        media_href = None
        if asset:
            source_path = Path(asset.path)
            if source_path.is_file():
                media_href = f"/api/jobs/{job_id}/media/source"
            elif (root / "final.mp4").is_file():
                media_href = f"/api/jobs/{job_id}/artifacts/final.mp4"
        return {
            "present": asset is not None,
            "query": query,
            "media_href": media_href,
            "source_language": self._read_json(root, "transcript.json").get("language"),
            "transcript_vtt_href": (
                f"/api/jobs/{job_id}/transcript.vtt" if has_transcript else None
            ),
            "asset": asset.model_dump(mode="json") if asset else None,
            "shots": [
                {
                    **shot.model_dump(mode="json"),
                    "thumbnail_href": f"/api/jobs/{job_id}/shots/{shot.id}/thumbnail",
                    "visual_description": (
                        all_visual_by_shot.get(int(shot.id or 0), {}).get("description")
                    ),
                    "visual_tags": all_visual_by_shot.get(int(shot.id or 0), {}).get("tags", []),
                    "visual_people": all_visual_by_shot.get(int(shot.id or 0), {}).get("people", []),
                    "visual_actions": all_visual_by_shot.get(int(shot.id or 0), {}).get("actions", []),
                    "person_tracks": person_labels.get(int(shot.id or 0), []),
                    "semantic_score": semantic_scores.get(("visual", int(shot.id or 0))),
                }
                for shot in shots
            ],
            "transcript": [
                {
                    **segment.model_dump(mode="json"),
                    "semantic_score": semantic_scores.get(("transcript", int(segment.id or 0))),
                }
                for segment in transcript
            ],
            "visual_count": len(all_visual),
            "semantic_mode": semantic_mode,
        }

    def library_search(self, query: str, filters: dict | None = None) -> dict:
        query = query.strip()
        if len(query) > 200:
            raise ValueError("search query must contain at most 200 characters")
        parsed = semantic_search.parse_advanced_query(query)
        constraints: dict[str, list[str]] = {
            key: list(parsed.get(key) or [])
            for key in (
                "person", "action", "location", "object", "kind", "project",
                "scene_type", "source",
            )
        }
        numeric: dict[str, float | None] = {
            "min_duration": None,
            "max_duration": None,
            "min_confidence": None,
        }
        dates: dict[str, str] = {"date_from": "", "date_to": ""}
        for key, value in (filters or {}).items():
            if key in constraints and value:
                constraints[key].append(str(value))
            elif key in numeric and value not in (None, ""):
                numeric[key] = float(value)
            elif key in dates and value:
                candidate = str(value).strip()
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", candidate):
                    raise ValueError(f"{key} must use YYYY-MM-DD")
                try:
                    datetime.strptime(candidate, "%Y-%m-%d")
                except ValueError as exc:
                    raise ValueError(f"{key} must use a valid YYYY-MM-DD date") from exc
                dates[key] = candidate
        semantic_text = str(parsed.get("text") or "").strip()
        has_numeric = any(value is not None for value in numeric.values())
        has_dates = any(dates.values())
        if not semantic_text and not any(constraints.values()) and not has_numeric and not has_dates:
            raise ValueError("search query or filter is required")

        def wanted(values: list[str], candidates: list[str]) -> bool:
            lowered = [candidate.casefold() for candidate in candidates]
            return all(
                any(value.casefold() in candidate for candidate in lowered)
                for value in values
            )

        results: list[dict] = []
        semantic_available = False
        for job in self.list_jobs():
            job_id = job["job_id"]
            if constraints["project"] and not wanted(constraints["project"], [job_id]):
                continue
            root = self._job_root(job_id)
            manifest_file = pipeline.manifest_path(root)
            job_date = (
                datetime.fromtimestamp(
                    manifest_file.stat().st_mtime,
                    tz=timezone.utc,
                ).date().isoformat()
                if manifest_file.is_file() else ""
            )
            if dates["date_from"] and job_date < dates["date_from"]:
                continue
            if dates["date_to"] and job_date > dates["date_to"]:
                continue
            database = root / "media_index.sqlite3"
            if not database.is_file():
                continue
            with MediaStore(database) as store:
                store.migrate()
                visual_rows = {int(row["shot_id"]): row for row in store.list_visual_observations()}
                graph = store.list_story_graph()
                story_by_shot: dict[int, dict[str, list[str]]] = {}
                story_confidence: dict[int, float] = {}
                for item in graph["scene_entities"]:
                    shot_id = int(item["shot_id"])
                    story_by_shot.setdefault(shot_id, {}).setdefault(str(item["type"]), []).append(str(item["label"]))
                    story_confidence[shot_id] = max(
                        story_confidence.get(shot_id, 0.0),
                        float(item.get("confidence") or 0.0),
                    )
                track_labels: dict[int, list[str]] = {}
                track_confidence: dict[int, float] = {}
                track_sources: dict[int, list[str]] = {}
                for track in store.list_person_tracks():
                    display = str(track.get("alias") or track.get("label") or "")
                    source = str(track.get("source") or "").strip()
                    for appearance in track.get("appearances") or []:
                        shot_id = int(appearance["shot_id"])
                        track_labels.setdefault(shot_id, []).append(display)
                        if source:
                            track_sources.setdefault(shot_id, []).append(source)
                        track_confidence[shot_id] = max(
                            track_confidence.get(shot_id, 0.0),
                            float(appearance.get("confidence") or 0.0),
                        )
                try:
                    has_embeddings = bool(
                        store.list_shot_embeddings(semantic_search.model_name())
                        or store.list_transcript_embeddings(semantic_search.model_name())
                    )
                    semantic = (
                        semantic_search.search_store(store, semantic_text, limit=60)
                        if has_embeddings and semantic_text else []
                    )
                    semantic_available = semantic_available or has_embeddings
                except semantic_search.EmbeddingUnavailable:
                    semantic = []

                job_results: dict[tuple[str, int], dict] = {}
                for item in semantic:
                    entity_id = int(item.get("shot_id") or item.get("segment_id") or 0)
                    key = (item["kind"], entity_id)
                    payload = {"job_id": job_id, **item, "semantic_score": item["score"]}
                    if item["kind"] == "visual":
                        visual = visual_rows.get(entity_id, {})
                        facets = story_by_shot.get(entity_id, {})
                        payload.update({
                            "tags": visual.get("tags", []),
                            "people": visual.get("people", []),
                            "actions": visual.get("actions", []),
                            "person_tracks": track_labels.get(entity_id, []),
                            "locations": facets.get("location", []),
                            "objects": facets.get("object", []),
                            "events": facets.get("event", []),
                            "scene_type": f"Scene {entity_id}",
                            "source": (
                                track_sources.get(entity_id, [""])[0]
                                if track_sources.get(entity_id)
                                else str(visual.get("source") or "visual")
                            ),
                            "project_date": job_date,
                            "confidence": max(
                                track_confidence.get(entity_id, 0.0),
                                story_confidence.get(entity_id, 0.0),
                            ),
                            "thumbnail_href": f"/api/jobs/{job_id}/shots/{entity_id}/thumbnail",
                        })
                    else:
                        payload.update({
                            "scene_type": "transcript",
                            "source": "transcript",
                            "project_date": job_date,
                        })
                    job_results[key] = payload

                if semantic_text:
                    lexical_visual = store.search_visual_observations(semantic_text)
                    transcript = store.search_transcript(semantic_text, 1)
                else:
                    lexical_visual = store.list_visual_observations()
                    transcript = store.list_transcript(1)
                for item in lexical_visual:
                    shot_id = int(item["shot_id"])
                    facets = story_by_shot.get(shot_id, {})
                    key = ("visual", shot_id)
                    job_results.setdefault(key, {
                        "job_id": job_id,
                        "kind": "visual",
                        "shot_id": shot_id,
                        "start_seconds": item["start_seconds"],
                        "end_seconds": item["end_seconds"],
                        "text": item["description"],
                        "tags": item.get("tags", []),
                        "people": item.get("people", []),
                        "actions": item.get("actions", []),
                        "person_tracks": track_labels.get(shot_id, []),
                        "locations": facets.get("location", []),
                        "objects": facets.get("object", []),
                        "events": facets.get("event", []),
                        "scene_type": f"Scene {shot_id}",
                        "source": (
                            track_sources.get(shot_id, [""])[0]
                            if track_sources.get(shot_id)
                            else str(item.get("source") or "visual")
                        ),
                        "project_date": job_date,
                        "confidence": max(
                            track_confidence.get(shot_id, 0.0),
                            story_confidence.get(shot_id, 0.0),
                        ),
                        "thumbnail_href": f"/api/jobs/{job_id}/shots/{shot_id}/thumbnail",
                        "semantic_score": None,
                    })
                for segment in transcript:
                    key = ("transcript", int(segment.id or 0))
                    job_results.setdefault(key, {
                        "job_id": job_id,
                        "kind": "transcript",
                        "segment_id": segment.id,
                        "start_seconds": segment.start_seconds,
                        "end_seconds": segment.end_seconds,
                        "text": segment.text,
                        "speaker": segment.speaker,
                        "scene_type": "transcript",
                        "source": "transcript",
                        "project_date": job_date,
                        "semantic_score": None,
                    })
                for item in job_results.values():
                    if constraints["kind"] and not wanted(constraints["kind"], [str(item["kind"])]):
                        continue
                    if constraints["scene_type"] and not wanted(
                        constraints["scene_type"], [str(item.get("scene_type") or "")]
                    ):
                        continue
                    if constraints["source"] and not wanted(
                        constraints["source"], [str(item.get("source") or "")]
                    ):
                        continue
                    duration = float(item["end_seconds"]) - float(item["start_seconds"])
                    if numeric["min_duration"] is not None and duration < float(numeric["min_duration"]):
                        continue
                    if numeric["max_duration"] is not None and duration > float(numeric["max_duration"]):
                        continue
                    if item["kind"] == "visual":
                        if constraints["person"] and not wanted(constraints["person"], list(item.get("person_tracks") or [])):
                            continue
                        if constraints["action"] and not wanted(constraints["action"], list(item.get("actions") or [])):
                            continue
                        if constraints["location"] and not wanted(constraints["location"], list(item.get("locations") or [])):
                            continue
                        if constraints["object"] and not wanted(constraints["object"], list(item.get("objects") or [])):
                            continue
                        if numeric["min_confidence"] is not None and float(item.get("confidence") or 0.0) < float(numeric["min_confidence"]):
                            continue
                    elif any(constraints[key] for key in ("person", "action", "location", "object")):
                        continue
                    results.append(item)

        results.sort(key=lambda item: (
            -(float(item["semantic_score"]) if item.get("semantic_score") is not None else -2.0),
            item["job_id"],
            float(item["start_seconds"]),
        ))
        return {
            "query": query,
            "semantic_query": semantic_text,
            "filters": {**constraints, **numeric, **dates},
            "semantic_mode": "fastembed" if semantic_available else "unavailable",
            "semantic_model": semantic_search.model_name(),
            "results": results[:100],
        }

    def source_media_path(self, job_id: str) -> Path:
        root = self._require_job(job_id)
        manifest = pipeline.load_manifest(root)
        if not manifest.config.source_video:
            raise FileNotFoundError("job không có video nguồn")
        path = Path(manifest.config.source_video)
        if not path.is_file():
            raise FileNotFoundError("không tìm thấy video nguồn")
        return path

    def shot_thumbnail_path(self, job_id: str, shot_id: int) -> Path:
        root = self._require_job(job_id)
        if shot_id < 1:
            raise ValueError("shot_id không hợp lệ")
        output = root / f"shot-{shot_id}.jpg"
        if output.is_file():
            return output
        database_path = root / "media_index.sqlite3"
        if not database_path.exists():
            raise FileNotFoundError("chưa có chỉ mục media")
        with MediaStore(database_path) as store:
            shot = store.get_shot(shot_id)
        if not shot:
            raise FileNotFoundError(f"không tìm thấy shot: {shot_id}")
        source = self.source_media_path(job_id)
        ffmpeg = pipeline.shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not on PATH - install FFmpeg to generate shot thumbnails")
        temporary = root / f"shot-{shot_id}.extracting.jpg"
        temporary.unlink(missing_ok=True)
        try:
            pipeline.subprocess.run(
                [ffmpeg, "-y", "-ss", f"{shot.start_seconds:.6f}", "-i", str(source), "-frames:v", "1", "-vf", "scale=320:-2", str(temporary)],
                capture_output=True,
                text=True,
                check=True,
            )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError("ffmpeg completed without producing shot thumbnail")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def transcript_export_path(self, job_id: str, fmt: str = "vtt") -> Path:
        """Materialize the indexed transcript as a downloadable subtitle file."""
        root = self._require_job(job_id)
        if fmt != "vtt":
            raise ValueError("định dạng xuất bản ghi không được hỗ trợ")
        database_path = root / "media_index.sqlite3"
        if not database_path.exists():
            raise FileNotFoundError("chưa có chỉ mục media")
        with MediaStore(database_path) as store:
            store.migrate()
            segments = store.list_transcript(1)
        if not segments:
            raise FileNotFoundError("chưa có lời thoại trong chỉ mục media")
        output = root / "transcript.vtt"
        output.write_text(_render_webvtt(segments), encoding="utf-8")
        return output

    def highlights(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        database = root / "media_index.sqlite3"
        if not database.is_file():
            raise FileNotFoundError("media index is not available")
        items = media_intelligence.detect_highlights(database)
        for item in items:
            item["export_href"] = f"/api/jobs/{job_id}/highlights/{item['id']}/clip.mp4"
        return {"highlights": items}

    def highlight_clip_path(self, job_id: str, candidate_id: str) -> Path:
        if not re.fullmatch(r"h-[1-9][0-9]{0,8}", candidate_id):
            raise ValueError("invalid highlight candidate id")
        root = self._require_job(job_id)
        candidates = {x["id"]: x for x in media_intelligence.detect_highlights(root / "media_index.sqlite3", 20)}
        candidate = candidates.get(candidate_id)
        if not candidate:
            raise FileNotFoundError("highlight candidate was not found")
        output = root / f"highlight-{candidate_id}.mp4"
        if output.is_file() and output.stat().st_size > 0:
            return output
        ffmpeg = pipeline.shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not on PATH")
        start = float(candidate["start"]); duration = float(candidate["end"]) - start
        if start < 0 or not 0 < duration <= media_intelligence.MAX_CLIP_SECONDS:
            raise ValueError("invalid highlight time bounds")
        temporary = output.with_suffix(".exporting.mp4"); temporary.unlink(missing_ok=True)
        command = [ffmpeg, "-y", "-ss", f"{start:.6f}", "-i", str(self.source_media_path(job_id)), "-t", f"{duration:.6f}", "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2", "-c:v", "libx264", "-threads", str(pipeline.ffmpeg_thread_cap()), "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart", str(temporary)]
        try:
            try:
                pipeline.subprocess.run(command, capture_output=True, text=True, check=True, shell=False)
            except pipeline.subprocess.CalledProcessError as exc:
                if not pipeline.is_ffmpeg_memory_error(f"{exc.stderr or ''}\n{exc.output or ''}"):
                    raise
                # One bounded, single-threaded retry clears the transient x264
                # "malloc failed / Cannot allocate memory" allocation flake.
                temporary.unlink(missing_ok=True)
                pipeline.subprocess.run(
                    pipeline.ffmpeg_command_single_thread(command),
                    capture_output=True, text=True, check=True, shell=False,
                )
            if not temporary.is_file() or temporary.stat().st_size <= 0:
                raise RuntimeError("ffmpeg did not produce a highlight clip")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return output

    def chat(self, job_id: str, question: str) -> dict:
        return media_intelligence.answer_chat(self._require_job(job_id), question)

    def build_handoff(self, job_id: str) -> dict:
        if self._is_running(job_id):
            raise RuntimeError("Chờ pipeline hoàn tất trước khi đóng gói.")
        from .handoff import build_handoff

        root = self._require_job(job_id)
        path = build_handoff(root)
        return {"name": path.name, "href": f"/api/jobs/{job_id}/artifacts/{path.name}"}

    def list_analytics(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        return {"imports": analytics.list_imports(root), "advice": analytics.retention_advice(root)}

    def import_analytics(self, job_id: str, format_name: str, data: bytes,
                         cta_seconds: str | None, notes: str) -> dict:
        root = self._require_job(job_id)
        if format_name not in ("csv", "json"):
            raise ValueError("Chỉ nhận file Studio CSV hoặc JSON.")
        if len(data) > analytics.MAX_EXPORT_BYTES:
            raise OverflowError("Studio export exceeds 2 MB")
        if not data:
            raise ValueError("File Studio rỗng.")
        if self._is_running(job_id):
            raise RuntimeError("Chờ pipeline hoàn tất trước khi nhập thống kê.")
        with tempfile.NamedTemporaryFile(prefix=".studio-", suffix="." + format_name,
                                         dir=root, delete=False) as stream:
            path = Path(stream.name)
            stream.write(data)
        try:
            return analytics.import_studio_export(root, path,
                                                  cta_seconds=cta_seconds or None, notes=notes)
        finally:
            path.unlink(missing_ok=True)

    def start_short(self, job_id: str, payload: dict) -> dict:
        root = self._require_job(job_id)
        try:
            start = float(payload["start_seconds"])
            end = float(payload["end_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Nhập mốc đầu và cuối bằng số giây.") from exc
        state = {"running": True, "error": None, "href": None, "srt_href": None}
        with self._lock:
            if (self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting):
                raise RuntimeError("Project đang chạy hoặc đang xuất short.")
            self._short_exports[job_id] = state

        def worker() -> None:
            error = None
            try:
                from .short_variants import build_short

                build_short(root, start, end)
            except Exception as exc:
                error = str(exc)
            finally:
                with self._lock:
                    if self._short_exports.get(job_id) is state:
                        state.update(
                            running=False, error=error,
                            href=None if error else f"/api/jobs/{job_id}/shorts/short-review.mp4",
                            srt_href=None if error else f"/api/jobs/{job_id}/shorts/short-review.srt",
                        )

        threading.Thread(target=worker, name=f"mrf-short-{job_id}", daemon=True).start()
        return {"started": True}

    def start_section_preview(self, job_id: str, section_index: int) -> dict:
        from . import quick_preview

        root = self._require_job(job_id)
        if isinstance(section_index, bool) or not isinstance(section_index, int) or section_index < 1:
            raise ValueError("Mục preview phải là số thứ tự phần hợp lệ.")
        script, _plan, _alignment, _source, _audio = quick_preview._required_inputs(root)
        if section_index > len(script.get("sections") or []):
            raise ValueError("Mục preview vượt quá số phần kịch bản.")
        state = {"running": True, "section_index": section_index, "error": None,
                 "href": None, "srt_href": None}
        with self._lock:
            if (self._runs.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or self._short_exports.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing):
                raise RuntimeError("Project đang xử lý; chờ xong trước khi dựng preview.")
            self._section_previews[job_id] = state

        def worker() -> None:
            error = None
            try:
                quick_preview.build_section_preview(root, section_index)
            except Exception as exc:
                error = str(exc)
            finally:
                with self._lock:
                    if self._section_previews.get(job_id) is state:
                        state.update(running=False, error=error)

        threading.Thread(target=worker, name=f"mrf-preview-{job_id}-{section_index}", daemon=True).start()
        return {"started": True, "section_index": section_index}

    def section_preview_state(self, job_id: str, section_index: int) -> dict:
        from .quick_preview import section_preview_artifact

        root = self._require_job(job_id)
        if isinstance(section_index, bool) or not isinstance(section_index, int) or section_index < 1:
            raise ValueError("Mục preview không hợp lệ.")
        with self._lock:
            state = dict(self._section_previews.get(job_id, {}))
        if state.get("running") and state.get("section_index") == section_index:
            return state
        result = {"running": False, "section_index": section_index, "error": None,
                  "href": None, "srt_href": None}
        try:
            section_preview_artifact(root, section_index, "mp4")
            section_preview_artifact(root, section_index, "srt")
        except (ValueError, FileNotFoundError, OSError):
            if state.get("section_index") == section_index:
                result["error"] = state.get("error") or "Preview đã cũ hoặc chưa có; tạo lại để xem."
        else:
            prefix = f"/api/jobs/{job_id}/previews/{section_index}/"
            stamp = (root / "previews" / f"section-{section_index}.mp4").stat().st_mtime_ns
            result.update(href=prefix + f"mp4?v={stamp}", srt_href=prefix + f"srt?v={stamp}")
        return result

    def section_preview_path(self, job_id: str, section_index: int, kind: str) -> Path:
        from .quick_preview import section_preview_artifact

        with self._lock:
            if self._runs.get(job_id, {}).get("running") or self._section_previews.get(job_id, {}).get("running"):
                raise RuntimeError("Chờ xử lý xong trước khi tải preview.")
        return section_preview_artifact(self._require_job(job_id), section_index, kind)

    def short_path(self, job_id: str, name: str) -> Path:
        from .short_variants import short_artifact

        with self._lock:
            if self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running"):
                raise RuntimeError("Chờ xử lý xong trước khi tải short.")
        return short_artifact(self._require_job(job_id), name)

    def artifact_path(self, job_id: str, name: str) -> Path:
        root = self._require_job(job_id)
        if not _is_safe_segment(name):
            raise ValueError(f"tên artifact không hợp lệ: {name!r}")
        path = root / name
        # Defence in depth: the resolved file must stay inside the job dir.
        if path.resolve().parent != root.resolve():
            raise ValueError("đường dẫn artifact không hợp lệ")
        if not path.is_file():
            raise FileNotFoundError(f"không tìm thấy artifact: {name}")
        if name == "review-handoff.zip":
            from .handoff import ASSETS

            manifest = pipeline.load_manifest(root)
            if self._is_running(job_id) or any(
                not manifest.stage(stage) or manifest.stage(stage).status != "ready"
                for stage in ("script", "scene_plan", "alignment", "render", "qa", "metadata", "thumbnail")
            ):
                raise ValueError("Gói xuất đã hết hiệu lực; kiểm tra QA rồi đóng gói lại.")
            script = self._read_json(root, "script.json")
            metadata = self._read_json(root, pipeline.METADATA_NAME)
            qa = self._read_json(root, "qa.json")
            if not (script.get("approved") and metadata.get("approved") and qa.get("passed")):
                raise ValueError("Gói xuất yêu cầu kịch bản, metadata và QA đã duyệt.")
            inputs = [root / source for source, _ in ASSETS]
            inputs += [root / name for name in ("script.json", "scene_plan.json", "qa.json")]
            if any(not item.is_file() or item.stat().st_mtime_ns > path.stat().st_mtime_ns for item in inputs):
                raise ValueError("Gói xuất đã cũ; vui lòng đóng gói lại.")
            from .analytics import _current_receipt

            try:
                _current_receipt(root)
            except ValueError as exc:
                raise ValueError("Gói xuất đã hết hiệu lực; vui lòng đóng gói lại.") from exc
        return path

    # -- write ---------------------------------------------------------------

    def create_job(self, payload: dict) -> dict:
        job_id = str(payload.get("job_id", "")).strip()
        if not _is_safe_segment(job_id):
            raise ValueError("job_id chỉ gồm chữ, số, dấu chấm, gạch ngang, gạch dưới")
        root = self.jobs_root / job_id
        if root.exists():
            raise FileExistsError(f"job đã tồn tại: {job_id}")
        source_video = payload.get("source_video") or None
        content_agent = str(payload.get("content_agent") or "scaffold")
        if content_agent not in CONTENT_AGENT_MODES:
            raise ValueError(f"content_agent phải là {' hoặc '.join(CONTENT_AGENT_MODES)}")
        watermark_detect = str(payload.get("watermark_detect") or "").strip().lower()
        watermark_method = str(payload.get("watermark_method") or "propainter").strip().lower()
        if watermark_method not in WATERMARK_METHODS:
            raise ValueError(f"watermark_method phải là {', '.join(WATERMARK_METHODS)}")
        watermark_removal: dict = {}
        if watermark_detect:
            if watermark_detect not in ("color", "temporal", "external"):
                raise ValueError("watermark_detect phải là color, temporal, hoặc external")
            detect: dict = {"method": watermark_detect}
            detector_cmd = str(payload.get("detector_cmd") or "").strip()
            if detector_cmd:
                detect["external_cmd"] = detector_cmd
            watermark_removal = {"enabled": True, "method": watermark_method, "detect": detect}
        # Prefill unset fields from the active channel profile; existing jobs are never mutated by a channel switch.
        defaults = self.creator_library.active_channel_defaults() or {}

        def _with_default(key: str, fallback):
            value = payload.get(key)
            if value in (None, ""):
                value = defaults.get(key, fallback)
            return fallback if value in (None, "") else value

        config = JobConfig(
            job_id=job_id,
            language=str(_with_default("language", "vi")),
            target_minutes=float(payload.get("target_minutes") or 10),
            aspect_ratio=str(_with_default("aspect_ratio", "16:9")),
            source_video=Path(source_video) if source_video else None,
            movie_title=str(payload.get("movie_title") or "").strip() or None,
            creative_brief=payload.get("creative_brief") or {},
            content_agent=content_agent,
            intro_seconds=float(_with_default("intro_seconds", 0)),
            outro_seconds=float(_with_default("outro_seconds", 0)),
            brand_top_band=float(_with_default("brand_top_band", 0)),
            brand_bottom_band=float(_with_default("brand_bottom_band", 0)),
            watermark_removal=watermark_removal,
            visual_variety=str(_with_default("visual_variety", "off")),
            tts_provider=str(_with_default("tts_provider", "edge")),
            tts_voice=str(_with_default("tts_voice", "")).strip(),
        )
        pipeline.create_job(root, config)
        return self.status(job_id)

    def update_job_config(self, job_id: str, payload: dict) -> dict:
        """Validate and persist the operator-editable pre-production settings."""
        root = self._require_job(job_id)
        if self._is_running(job_id):
            raise RuntimeError("Chờ pipeline hoàn tất trước khi sửa cấu hình dự án.")
        if not isinstance(payload, dict):
            raise ValueError("cấu hình dự án phải là một object JSON")
        allowed = {
            "movie_title", "content_agent", "visual_variety", "watermark_removal",
            "tts_provider", "tts_voice", "target_minutes", "aspect_ratio",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError("trường cấu hình không được hỗ trợ: " + ", ".join(sorted(unknown)))

        manifest = pipeline.load_manifest(root)
        current = manifest.config.model_dump(mode="python")
        updates = dict(payload)
        if "movie_title" in updates:
            title = str(updates["movie_title"] or "").strip()
            if len(title) > 300:
                raise ValueError("movie_title tối đa 300 ký tự")
            updates["movie_title"] = title or None
        if "content_agent" in updates and updates["content_agent"] not in CONTENT_AGENT_MODES:
            raise ValueError(f"content_agent phải là {' hoặc '.join(CONTENT_AGENT_MODES)}")
        if "visual_variety" in updates and updates["visual_variety"] not in visual_variety.PROFILES:
            raise ValueError("visual_variety phải là off, light, balanced, hoặc aggressive")
        if "tts_provider" in updates and updates["tts_provider"] not in tts_providers.SUPPORTED_PROVIDERS:
            raise ValueError("tts_provider không được hỗ trợ")
        if "tts_voice" in updates:
            updates["tts_voice"] = str(updates["tts_voice"] or "").strip()
        if "watermark_removal" in updates:
            watermark = updates["watermark_removal"]
            if not isinstance(watermark, dict):
                raise ValueError("watermark_removal phải là một object JSON")
            detect = watermark.get("detect")
            if isinstance(detect, dict) and "detector_cmd" in detect:
                detect = dict(detect)
                detect["external_cmd"] = detect.pop("detector_cmd")
                watermark = {**watermark, "detect": detect}
            # Partial update: keep mask/boxes/bands/detector knobs the dialog does not edit.
            existing = current.get("watermark_removal") or {}
            merged = {**existing, **watermark}
            if isinstance(existing.get("detect"), dict) and isinstance(watermark.get("detect"), dict):
                merged["detect"] = {**existing["detect"], **watermark["detect"]}
            updates["watermark_removal"] = merged

        current.update(updates)
        manifest.config = JobConfig.model_validate(current)
        pipeline.save_manifest(root, manifest)
        result = self.status(job_id)
        completed = {stage["stage"] for stage in result["stages"] if stage["status"] == "ready"}
        warnings = []
        if "content_agent" in updates and completed.intersection({"research", "outline", "script"}):
            warnings.append("Bộ tạo nội dung mới chỉ áp dụng khi chạy lại các bước nghiên cứu/kịch bản.")
        if {"tts_provider", "tts_voice"}.intersection(updates) and "tts" in completed:
            warnings.append("Giọng đọc mới chỉ áp dụng khi chạy lại bước tạo giọng.")
        if "watermark_removal" in updates and "watermark" in completed:
            warnings.append("Thiết lập xoá watermark mới chỉ áp dụng khi chạy lại bước xoá watermark.")
        result["config_warnings"] = warnings
        return result

    def probe_detector(self, job_id: str) -> dict:
        """Run the external watermark detector on one frame to validate config."""
        root = self._require_job(job_id)
        cfg = pipeline.load_manifest(root).config
        if not cfg.source_video:
            raise ValueError("job chưa có video nguồn để thử detector")
        detect = cfg.watermark_removal.detect
        external_cmd = (detect.external_cmd if detect else "") or ""
        settings = mask_detection.DetectSettings(method="external", external_cmd=external_cmd)
        probe = mask_detection.probe_external_detector(Path(cfg.source_video), settings)
        return {"ok": probe.ok, "masks": probe.masks, "message": probe.message}

    def probe_link(self, url: str, runner=None) -> dict:
        """Fetch metadata for a video URL via yt-dlp."""
        url = str(url or "").strip()
        if not url:
            raise ValueError("nhập liên kết video hợp lệ")
        return link_download.fetch_metadata(url, runner=runner)

    def download_link_video(
        self,
        job_id: str,
        url: str,
        confirm_rights: bool = False,
        sub_langs: str = "vi,en",
        runner=None,
    ) -> dict:
        root = self._require_job(job_id)
        url = str(url or "").strip()
        if not url:
            raise ValueError("nhập liên kết video hợp lệ")
        if not confirm_rights and not link_download.rights_confirmed():
            raise link_download.RightsConfirmationRequired(
                "Xác nhận bạn có quyền sử dụng video này để tải về."
            )
        with self._lock:
            if (self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing):
                raise RuntimeError("job đang chạy hoặc đang tải video hay đang xuất short")
            if pipeline.load_manifest(root).config.source_video or (root / "source.mp4").exists():
                raise FileExistsError("job đã có video nguồn")
            self._uploads.add(job_id)
        try:
            result = link_download.download_video(
                url,
                root,
                confirm_rights=confirm_rights,
                sub_langs=sub_langs,
                runner=runner,
            )
            source_path = Path(result["source_video"])
            manifest = pipeline.load_manifest(root)
            manifest.config.source_video = source_path.resolve()
            pipeline.save_manifest(root, manifest)
        finally:
            with self._lock:
                self._uploads.discard(job_id)
        if self._index_auto:
            self._enqueue_index_quietly(job_id)
        return self.status(job_id)

    def hook_path(self, job_id: str) -> Path:
        root = self._require_job(job_id)
        path = root / "hook.mp4"
        if not path.is_file():
            raise FileNotFoundError("chưa có hook teaser cho project này")
        return path

    def get_hook_info(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        hook_file = root / "hook.mp4"
        meta_file = root / "hook.json"
        if not hook_file.is_file() or not meta_file.is_file():
            return {"present": False, "href": None, "meta": None}
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            meta = None
        return {"present": True, "href": f"/api/jobs/{job_id}/hook.mp4", "meta": meta}

    def generate_hook(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        if not (root / "scenes.json").is_file():
            raise ValueError("Cần lập chỉ mục cảnh (scenes.json) trước khi tạo hook teaser.")
        manifest = pipeline.load_manifest(root)
        if not manifest.config.source_video or not Path(manifest.config.source_video).is_file():
            raise ValueError("Project chưa có video nguồn để tạo hook teaser.")
        hook_crafter.build_hook_teaser(root)
        return self.get_hook_info(job_id)

    def import_video(self, job_id: str, filename: str, size: int, reader) -> dict:
        root = self._require_job(job_id)
        if Path(filename).suffix.lower() != ".mp4" or len(filename) > 255:
            raise ValueError("chọn video MP4 để import")
        if size <= 0:
            raise ValueError("video rỗng hoặc thiếu Content-Length")
        with self._lock:
            if (self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing):
                raise RuntimeError("job đang chạy hoặc đang tải video hay đang xuất short")
            if pipeline.load_manifest(root).config.source_video or (root / "source.mp4").exists():
                raise FileExistsError("job đã có video nguồn")
            self._uploads.add(job_id)
        temporary = root / "source.mp4.tmp"
        source = root / "source.mp4"
        moved = False
        try:
            with temporary.open("wb") as output:
                remaining = size
                while remaining:
                    chunk = reader.read(min(1_048_576, remaining))
                    if not chunk:
                        raise ValueError("video tải lên chưa đủ dữ liệu")
                    output.write(chunk)
                    remaining -= len(chunk)
            temporary.replace(source)
            moved = True
            manifest = pipeline.load_manifest(root)
            manifest.config.source_video = source.resolve()
            pipeline.save_manifest(root, manifest)
        except Exception:
            if moved:
                source.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)
            with self._lock:
                self._uploads.discard(job_id)
        # Importing the source never advances the pipeline: the operator starts
        # every run explicitly. Pre-warming the heavy transcript/scene/visual/
        # embedding work in the background is opt-in via MRF_AUTO_INDEX=1
        # (roadmap #14) for operators who want it ready before the run.
        if self._index_auto:
            self._enqueue_index_quietly(job_id)
        return self.status(job_id)

    def delete_job(self, job_id: str, confirmation: str) -> dict:
        if confirmation != job_id:
            raise ValueError("nhập đúng mã project để xác nhận xóa")
        result = self.delete_jobs([job_id], "XOA 1")
        if result["failed"]:
            raise RuntimeError("không xóa được project: " + job_id)
        return {"deleted": True, "job_id": job_id}

    def delete_jobs(self, job_ids: list[str], confirmation: str) -> dict:
        if not isinstance(job_ids, list) or not job_ids or any(not isinstance(job_id, str) for job_id in job_ids):
            raise ValueError("chọn ít nhất một project hợp lệ")
        if len(set(job_ids)) != len(job_ids):
            raise ValueError("danh sách project trùng mã")
        if confirmation != f"XOA {len(job_ids)}":
            raise ValueError("nhập đúng số project để xác nhận xóa")
        roots = [self._job_root(job_id) for job_id in job_ids]
        if any(root.is_symlink() for root in roots):
            raise ValueError("không thể xóa project liên kết thư mục")
        # Free any queued/active background index first (roadmap #14): a queued
        # index is dropped, an active one must be stopped before delete proceeds.
        for job_id in job_ids:
            self._preempt_index(job_id)
        with self._lock:
            for job_id in job_ids:
                if (self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running")
                        or self._section_previews.get(job_id, {}).get("running")
                        or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing):
                    raise RuntimeError("project đang chạy hoặc đang tải video: " + job_id)
                self._require_job(job_id)
            self._deleting.update(job_ids)
            deleted: list[str] = []
            failed: str | None = None
            try:
                for job_id, root in zip(job_ids, roots):
                    try:
                        shutil.rmtree(root)
                    except OSError:
                        failed = job_id
                        break
                    self._runs.pop(job_id, None)
                    self._section_previews.pop(job_id, None)
                    deleted.append(job_id)
            finally:
                self._deleting.difference_update(job_ids)
        return {"deleted": deleted, "failed": failed}

    def batch_status(self) -> dict:
        """Current overnight-queue state (safe copy) plus status counts."""
        with self._lock:
            items = [dict(item) for item in self._batch["items"]]
            running = bool(self._batch["running"])
            started_at = self._batch["started_at"]
        counts: dict[str, int] = {}
        for item in items:
            counts[item["status"]] = counts.get(item["status"], 0) + 1
        return {"running": running, "started_at": started_at, "items": items, "counts": counts}

    def stop_batch(self) -> dict:
        """Ask the queue to stop after the current job; remaining items are cancelled."""
        with self._lock:
            if self._batch["running"]:
                self._batch["stop"] = True
        return self.batch_status()

    def start_batch(self, payload: dict, *, process=None) -> dict:
        """Queue many links and run each to the script-review gate, one at a time.

        Respects the approval gate: each job stops at ``script`` (never auto-approved),
        so the operator reviews drafts afterwards. Fault-tolerant per item - a bad
        link is logged on its row and the queue continues. ``process`` is injectable
        for tests; production builds the real create -> download -> run worker.
        """
        from . import batch_queue

        self._ensure_open()
        urls = batch_queue.parse_batch_input(str(payload.get("links") or ""))
        confirm_rights = bool(payload.get("confirm_rights"))
        if process is None and not confirm_rights:
            raise link_download.RightsConfirmationRequired(
                "Xác nhận bạn có quyền dùng các video này trước khi chạy hàng đợi."
            )
        language = str(payload.get("language") or "vi")
        sub_langs = str(payload.get("sub_langs") or "vi,en")
        with self._lock:
            if self._batch["running"]:
                raise RuntimeError("hàng đợi đang chạy; chờ xong hoặc bấm dừng")
            self._batch = {"running": True, "stop": False,
                           "items": batch_queue.initial_items(urls),
                           "started_at": datetime.now(timezone.utc).isoformat()}

        worker_process = process or self._build_batch_process(
            confirm_rights=confirm_rights, language=language, sub_langs=sub_langs)

        def on_event(item: dict) -> None:
            with self._lock:
                for row in self._batch["items"]:
                    if row["index"] == item["index"]:
                        row.update(status=item["status"], job_id=item["job_id"], error=item["error"])
                        break

        def should_stop() -> bool:
            with self._lock:
                return bool(self._batch["stop"])

        def worker() -> None:
            try:
                results = batch_queue.run_batch(urls, process=worker_process,
                                                on_event=on_event, should_stop=should_stop)
                try:
                    batch_queue.notify_batch_complete(results)
                except Exception:  # a webhook problem must never fail the queue
                    pass
            finally:
                with self._lock:
                    self._batch["running"] = False
                    self._batch["stop"] = False

        threading.Thread(target=worker, name="mrf-batch-queue", daemon=True).start()
        return self.batch_status()

    def _build_batch_process(self, *, confirm_rights: bool, language: str, sub_langs: str):
        """Real per-item worker: create job -> download link -> run to script gate."""
        from . import batch_queue

        def process(url: str, index: int) -> str:
            job_id = batch_queue.slug_for(url, index)
            root = self.jobs_root / job_id
            suffix = 1
            while root.exists():
                suffix += 1
                job_id = f"{batch_queue.slug_for(url, index)}-{suffix}"
                root = self.jobs_root / job_id
            pipeline.create_job(root, JobConfig(job_id=job_id, language=language))
            result = link_download.download_video(
                url, root, confirm_rights=confirm_rights, sub_langs=sub_langs)
            manifest = pipeline.load_manifest(root)
            manifest.config.source_video = Path(result["source_video"]).resolve()
            pipeline.save_manifest(root, manifest)
            # Approval gate stays intact: stop at the script draft, never auto-approve.
            pipeline.run_job(root, until=SCRIPT_REVIEW_STAGE)
            return job_id

        return process

    def start_run(self, job_id: str, *, until: str | None = None) -> dict:
        self._ensure_open()
        root = self._require_job(job_id)
        if until is None:
            # The first run stops at the script for review; a run only advances
            # toward the finished video after the operator approves the script.
            approved = bool(self._read_json(root, "script.json").get("approved"))
            until = RUN_UNTIL_STAGE if approved else SCRIPT_REVIEW_STAGE
        # A background index for this job must yield before a full run starts so
        # the two never write the same media_index concurrently (roadmap #14).
        self._preempt_index(job_id)
        event = threading.Event()
        state = {"running": True, "stopping": False, "error": None, "until": until, "event": event, "process": None}
        with self._lock:
            if (self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing):
                raise RuntimeError("job đang chạy hoặc đang xuất short hoặc đang tải video")
            self._runs[job_id] = state

        def register(process) -> None:
            with self._lock:
                if self._runs.get(job_id) is state:
                    state["process"] = process

        def unregister(process) -> None:
            with self._lock:
                if self._runs.get(job_id) is state and state.get("process") is process:
                    state["process"] = None

        def worker() -> None:
            error = None
            try:
                context = cancellation.CancellationContext(event, register, unregister)
                with cancellation.cancellation_scope(context):
                    pipeline.run_job(root, until=until)
            except Exception as exc:
                error = str(exc)
            finally:
                with self._lock:
                    if self._runs.get(job_id) is state:
                        state.update(running=False, stopping=False, error=error, process=None)

        threading.Thread(target=worker, name=f"mrf-run-{job_id}", daemon=True).start()
        return {"started": True, "until": until}

    def reindex_transcript(self, job_id: str) -> dict:
        """Reset and re-run only the transcript + scenes stages for a job.

        Regenerates ``transcript.json``/``captions.srt`` (so faster-whisper runs
        again with word-level timestamps) and rebuilds ``media_index.sqlite3``
        from that fresh transcript, then stops before ``outline`` so the already
        produced script/render are left untouched. Runs in a background thread;
        progress is exposed via ``status.reindex``.
        """
        root = self._require_job(job_id)
        cfg = pipeline.load_manifest(root).config
        if not cfg.source_video:
            raise ValueError("Project chưa có video nguồn để lập chỉ mục lại lời thoại.")
        # A queued/running background index must yield first so the two never
        # write media_index.sqlite3 at the same time.
        self._preempt_index(job_id)
        state = {"running": True, "error": None,
                 "message": "Đang bóc lại lời thoại và dựng chỉ mục…"}
        with self._lock:
            if (self._runs.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or self._short_exports.get(job_id, {}).get("running")
                    or self._reindex.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing):
                raise RuntimeError("Project đang xử lý; chờ xong trước khi lập chỉ mục lại lời thoại.")
            # Flip only these two stages back to pending; the pipeline skips any
            # stage still marked ready, so re-running is otherwise a no-op.
            manifest = pipeline.load_manifest(root)
            for stage in manifest.stages:
                if stage.stage in ("transcript", "scenes"):
                    stage.status = "pending"
                    stage.message = ""
            pipeline.save_manifest(root, manifest)
            self._reindex[job_id] = state

        def worker() -> None:
            error = None
            try:
                pipeline.run_job(root, until="scenes")
                # run_job turns a missing dependency into a "skipped" stage
                # rather than raising, so a silent no-op would otherwise look
                # like success. If the transcript did not finish ready, the
                # captions were NOT recomputed (e.g. faster-whisper is not
                # installed) - surface that so the operator can act on it.
                transcript_stage = pipeline.load_manifest(root).stage("transcript")
                if transcript_stage is not None and transcript_stage.status in ("skipped", "failed"):
                    error = localization.localize_message(transcript_stage.message) or (
                        "Không bóc lại được lời thoại (kiểm tra phụ thuộc như faster-whisper)."
                    )
            except Exception as exc:
                error = str(exc)
            finally:
                with self._lock:
                    if self._reindex.get(job_id) is state:
                        state.update(
                            running=False,
                            error=error,
                            message=None if error else "Đã cập nhật lời thoại và mốc thời gian.",
                        )

        threading.Thread(target=worker, name=f"mrf-reindex-{job_id}", daemon=True).start()
        return {"started": True}

    def rerender_brand(self, job_id: str, payload: dict) -> dict:
        root = self._require_job(job_id)
        with self._lock:
            if (self._runs.get(job_id, {}).get("running") or self._short_exports.get(job_id, {}).get("running")
                    or self._section_previews.get(job_id, {}).get("running")
                    or job_id in self._uploads or job_id in self._deleting):
                raise RuntimeError("Project đang chạy; thử lại khi hoàn thành.")
            manifest = pipeline.load_manifest(root)
            if not self._read_json(root, "script.json").get("approved"):
                raise ValueError("Duyệt kịch bản mới trước khi dựng lại video.")
            if not (root / "scene_plan.json").exists() or not (root / "narration.mp3").exists():
                raise ValueError("Cần có scene plan và giọng đọc trước khi dựng lại.")
            versions.create(root, "Trước khi dựng lại nhận diện kênh")
            top = float(payload.get("top_band", 0))
            bottom = float(payload.get("bottom_band", 0))
            manifest.config.brand_top_band = top
            manifest.config.brand_bottom_band = bottom
            for stage in manifest.stages:
                if stage.stage in ("render", "qa"):
                    stage.status = "pending"
                    stage.message = ""
            pipeline.save_manifest(root, manifest)
        return self.start_run(job_id)

    def stop_run(self, job_id: str) -> dict:
        self._require_job(job_id)
        with self._lock:
            state = self._runs.get(job_id)
            if not state or not state.get("running"):
                index_state = self._indexing.get(job_id)
                if index_state is not None:
                    index_state["cancel"].set()
                    if not index_state.get("running"):
                        self._release_index(job_id)
                    return {"stopping": True, "already_stopped": False, "indexing": True}
                if state and state.get("event") is not None and state["event"].is_set():
                    return {"stopping": True, "already_stopped": True}
                return {"stopping": False, "already_stopped": True}
            state["stopping"] = True
            state["event"].set()
            process = state.get("process")
        if process is not None:
            _terminate_process_tree(process)
        return {"stopping": True, "already_stopped": False}

    # -- background indexing queue (roadmap #14) -----------------------------

    def enqueue_index(self, job_id: str) -> dict:
        """Queue a job for background source-only indexing (FIFO, one at a time).

        Idempotent: a job already queued or actively indexing is left in place.
        Refuses when the job has no source video or is being deleted.
        """
        self._ensure_open()
        root = self._require_job(job_id)
        if not pipeline.load_manifest(root).config.source_video:
            raise RuntimeError("job chưa có video nguồn để lập chỉ mục")
        with self._lock:
            if job_id in self._deleting:
                raise RuntimeError("job đang bị xóa")
            existing = self._indexing.get(job_id)
            if existing and (existing.get("queued") or existing.get("running")):
                return {"queued": True, "already": True, "job_id": job_id}
            self._indexing[job_id] = {
                "queued": True,
                "running": False,
                "stage": None,
                "done": 0,
                "total": len(pipeline.INDEX_STAGES) + 1,
                "error": None,
                "cancel": threading.Event(),
            }
        self._index_queue.put(job_id)
        return {"queued": True, "already": False, "job_id": job_id}

    def _enqueue_index_quietly(self, job_id: str) -> None:
        """Best-effort auto-enqueue used after import; never raises to the caller."""
        try:
            self.enqueue_index(job_id)
        except Exception:
            pass

    def index_state(self, job_id: str) -> dict | None:
        """Serialisable snapshot of a job's indexing progress, or None if idle."""
        with self._lock:
            state = self._indexing.get(job_id)
            if state is None:
                return None
            return {
                "queued": bool(state.get("queued")),
                "running": bool(state.get("running")),
                "stage": state.get("stage"),
                "done": int(state.get("done", 0)),
                "total": int(state.get("total", 0)),
                "error": state.get("error"),
            }

    def _index_progress(self, job_id: str, event: dict) -> None:
        with self._lock:
            state = self._indexing.get(job_id)
            if state is None:
                return
            state["stage"] = event.get("stage")
            state["done"] = int(event.get("index", state.get("done", 0)))
            state["total"] = int(event.get("total", state.get("total", 0)))

    def _release_index(self, job_id: str) -> None:
        # Caller must hold self._lock. Drop the job and wake run/delete waiters.
        self._indexing.pop(job_id, None)
        self._index_cv.notify_all()

    def _preempt_index(self, job_id: str) -> None:
        """Free a job from the index queue so a run/delete can take over.

        Cancels a queued index immediately; for an actively-indexing job it
        signals cancellation and waits briefly for a cooperative stop, raising
        if the worker has not released it (e.g. mid-transcription).
        """
        with self._index_cv:
            state = self._indexing.get(job_id)
            if state is None:
                return
            state["cancel"].set()
            if not state.get("running"):
                self._indexing.pop(job_id, None)
                self._index_cv.notify_all()
                return
            self._index_cv.wait_for(lambda: job_id not in self._indexing, timeout=8.0)
            if job_id in self._indexing:
                raise RuntimeError("job đang được lập chỉ mục nền, dừng chỉ mục rồi thử lại")

    def _index_worker(self) -> None:
        while True:
            job_id = self._index_queue.get()
            try:
                if job_id is None:
                    return
                self._run_index_job(job_id)
            except Exception:
                # The worker thread must survive any single job's failure.
                with self._lock:
                    self._release_index(job_id)
            finally:
                self._index_queue.task_done()

    def _run_index_job(self, job_id: str) -> None:
        with self._lock:
            state = self._indexing.get(job_id)
            if state is None or state["cancel"].is_set():
                self._release_index(job_id)
                return
            if (
                self._runs.get(job_id, {}).get("running")
                or job_id in self._uploads
                or job_id in self._deleting
            ):
                self._release_index(job_id)
                return
            try:
                root = self._require_job(job_id)
            except (FileNotFoundError, ValueError):
                self._release_index(job_id)
                return
            state["queued"] = False
            state["running"] = True
            cancel_event = state["cancel"]

        try:
            context = cancellation.CancellationContext(cancel_event)
            with cancellation.cancellation_scope(context):
                pipeline.run_index(
                    root, progress=lambda event: self._index_progress(job_id, event)
                )
        except cancellation.RunCancelled:
            pass
        except Exception:
            # Stage-level failures are already persisted to the manifest and are
            # surfaced through status(); nothing else to record here.
            pass
        finally:
            with self._lock:
                self._release_index(job_id)

    def approve_metadata(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        meta = pipeline.approve_metadata(root)
        return {"approved": True, "metadata": meta}

    def update_metadata(self, job_id: str, fields: dict) -> dict:
        with self._version_lock:
            root = self._require_job(job_id)
            if self._is_running(job_id):
                raise RuntimeError("job is running")
            versions.create(root, "Trước khi sửa metadata")
            meta = pipeline.update_metadata(root, fields)
            return {"approved": bool(meta.get("approved")), "metadata": meta}

    def get_audio_mix(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        config = self._read_json(root, "audio_mix.json")
        return {"audio_mix": config or {"voice_gain_db": 0, "music": None, "effects": []}}

    def update_audio_mix(self, job_id: str, config: dict) -> dict:
        self._assert_no_preview(job_id)
        with self._version_lock:
            root = self._require_job(job_id)
            if self._is_running(job_id):
                raise RuntimeError("job is running")
            if set(config) - {"voice_gain_db", "music", "effects"}:
                raise ValueError("Cấu hình âm thanh có trường không hợp lệ.")
            normalized = {
                "voice_gain_db": config.get("voice_gain_db", 0),
                "music": config.get("music"),
                "effects": config.get("effects", []),
            }
            audio_mix.build_audio_mix(normalized, overlay_count=0, duration_seconds=None)
            if self.get_audio_mix(job_id)["audio_mix"] == normalized:
                return {"changed": False, "audio_mix": normalized}
            versions.create(root, "Trước khi sửa âm thanh")
            pipeline._write_json(root, "audio_mix.json", normalized)
            pipeline.invalidate_downstream(root, "alignment")
            return {"changed": True, "audio_mix": normalized}

    def get_script(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        script = self._read_json(root, "script.json")
        return {"present": bool(script), "script": script}

    def update_script(self, job_id: str, fields: dict) -> dict:
        with self._version_lock:
            root = self._require_job(job_id)
            if self._is_running(job_id):
                raise RuntimeError("job is running")
            self._assert_no_preview(job_id)
            versions.create(root, "Trước khi sửa kịch bản")
            script = pipeline.update_script(root, fields)
            return {"approved": False, "script": script}

    def tag_script(self, job_id: str, fields: dict) -> dict:
        self._assert_no_preview(job_id)
        with self._version_lock:
            root = self._require_job(job_id)
            if self._is_running(job_id):
                raise RuntimeError("job is running")
            script = self._read_json(root, "script.json")
            if not script:
                raise FileNotFoundError("script.json missing")
            if not all(key in fields for key in ("section_index", "start", "end", "kind")):
                raise ValueError("Thiếu thông tin đoạn lời dẫn hoặc loại nhãn")
            refs = fields.get("evidence_refs") or []
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise ValueError("evidence_refs phải là danh sách mốc nguồn")
            updated = creative_brief.tag_script_span(
                script,
                int(fields["section_index"]),
                int(fields["start"]),
                int(fields["end"]),
                str(fields["kind"]),
                refs,
            )
            if updated != script:
                versions.create(root, "Trước khi gắn nhãn kịch bản")
                updated = pipeline.update_script(root, {"sections": updated["sections"]})
            return {"approved": bool(updated.get("approved")), "script": updated}

    def list_versions(self, job_id: str) -> dict:
        return {"versions": versions.list_versions(self._require_job(job_id))}

    def create_version(self, job_id: str, name: str) -> dict:
        if self._is_running(job_id):
            raise RuntimeError("job is running")
        with self._version_lock:
            return versions.create(self._require_job(job_id), name)

    def restore_version(self, job_id: str, version_id: str, kind: str) -> dict:
        if self._is_running(job_id):
            raise RuntimeError("job is running")
        self._assert_no_preview(job_id)
        with self._version_lock:
            return versions.restore(self._require_job(job_id), version_id, kind)

    def version_artifact(self, job_id: str, version_id: str, name: str) -> Path:
        return versions.artifact_path(self._require_job(job_id), version_id, name)

    def edit_thumbnail(self, job_id: str, headline: str, channel_name: str) -> dict:
        from .thumbnail_editor import render_thumbnail_variants

        root = self._require_job(job_id)
        if self._is_running(job_id):
            raise RuntimeError("Chờ pipeline hoàn tất trước khi sửa ảnh bìa.")
        return render_thumbnail_variants(root, headline=headline, channel_name=channel_name)

    def auto_generate_thumbnails(self, job_id: str) -> dict:
        """Zero-typing cover art: AGY writes three punchy headlines from the
        script (rule-based fallback when the AGY pool is offline), the channel
        name comes from branding, and the source channel's residual watermark
        bands are always covered (force_cover) before the new branding is drawn.

        Like :meth:`edit_thumbnail` this only stages variants; the user still
        picks one via :meth:`select_thumbnail`, which resets export approval.
        """
        from .thumbnail_editor import render_thumbnail_variants

        root = self._require_job(job_id)
        if self._is_running(job_id):
            raise RuntimeError("Chờ pipeline hoàn tất trước khi tạo ảnh bìa.")
        manifest = pipeline.load_manifest(root)
        channel_name = branding.load_settings(self.jobs_root)["name"]
        movie_title = (manifest.config.movie_title or job_id or "").strip()
        headlines = self._auto_thumbnail_headlines(root, movie_title, language=manifest.config.language)
        return render_thumbnail_variants(
            root,
            headline=headlines[0],
            channel_name=channel_name,
            headlines=headlines,
            force_cover=True,
        )

    def _auto_thumbnail_headlines(self, root, movie_title: str, language: str = "vi") -> list[str]:
        """Three legible headlines for the auto thumbnails.

        Tries the AGY pool first; on any AGY failure (offline pool, quota, bad
        output) it falls back to rule-based templates so the feature always
        produces three cards. Every headline is trimmed to a guaranteed-fit
        string, so rendering never raises on an over-long AGY line.
        """
        from .agy_agent import run_agy_json
        from .content_agent import ContentAgentError

        fallback = self._fallback_thumbnail_headlines(movie_title, language)
        script = self._read_json(root, "script.json")
        thesis = ""
        if isinstance(script, dict):
            sections = script.get("sections")
            if isinstance(sections, list) and sections and isinstance(sections[0], dict):
                thesis = str(sections[0].get("title") or "")[:160]
        schema = {
            "type": "object",
            "properties": {"headlines": {"type": "array", "items": {"type": "string"}}},
            "required": ["headlines"],
            "additionalProperties": False,
        }
        if narration_style.is_vietnamese(language):
            prompt = (
                "Bạn viết tiêu đề ảnh bìa YouTube tiếng Việt cho video review phim "
                f"'{movie_title or 'phim này'}'. Trả về đúng 3 tiêu đề giật gân, IN HOA, "
                "mỗi tiêu đề tối đa 38 ký tự để không tràn khung: câu 1 gợi tò mò, "
                "câu 2 tiết lộ cú twist sốc, câu 3 nhấn kịch tính sinh tử. "
                f"Bối cảnh mở đầu: '{thesis}'. Không bịa tình tiết, không dùng dấu ngoặc kép. "
                "Trả JSON đúng schema."
            )
        else:
            prompt = (
                f"Write YouTube thumbnail text in {narration_style.language_name(language)} for a movie "
                f"recap of '{movie_title or 'this movie'}'. Return exactly 3 ALL-CAPS headlines, "
                "each at most 4 words and 28 characters so it reads on a phone: 1 sparks curiosity, "
                "2 teases the twist without spoiling it, 3 raises the stakes. "
                f"Opening context: '{thesis}'. Do not invent plot points, no quotation marks, "
                "no clickbait the video cannot pay off. Return JSON matching the schema."
            )
        try:
            result = run_agy_json(stage="thumbnail", prompt=prompt, schema=schema)
        except ContentAgentError:
            return fallback
        raw = result.get("headlines") if isinstance(result, dict) else None
        headlines: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                text = self._fit_thumbnail_headline(str(item or ""))
                if text and text not in headlines:
                    headlines.append(text)
        while len(headlines) < 3:
            headlines.append(fallback[len(headlines)])
        return headlines[:3]

    @staticmethod
    def _fit_thumbnail_headline(text: str) -> str:
        """Trim ``text`` (dropping trailing words) to a headline that renders
        within the safe area; returns "" when nothing legible remains."""
        from .thumbnail_editor import headline_fits

        words = " ".join(str(text or "").split()).split(" ") if text else []
        while words:
            candidate = " ".join(words)
            if headline_fits(candidate):
                return candidate
            words.pop()
        return ""

    @staticmethod
    def _fallback_thumbnail_headlines(movie_title: str, language: str = "vi") -> list[str]:
        """Rule-based headlines used when the AGY pool is unavailable."""
        if narration_style.is_vietnamese(language):
            default = "PHIM NÀY"
            title = (movie_title or "").strip().upper() or default
            templates = [
                f"{title}: SỰ THẬT KINH HOÀNG",
                f"BÍ MẬT ĐẰNG SAU {title}",
                f"CÁI KẾT BẤT NGỜ CỦA {title}",
            ]
        else:
            default = "THIS MOVIE"
            title = (movie_title or "").strip().upper() or default
            templates = [
                f"{title}: THE DARK TRUTH",
                f"THE SECRET BEHIND {title}",
                f"{title} ENDING EXPLAINED",
            ]
        headlines: list[str] = []
        for template in templates:
            text = JobsService._fit_thumbnail_headline(template)
            if not text:
                text = JobsService._fit_thumbnail_headline(title) or default
            headlines.append(text)
        return headlines

    def select_thumbnail(self, job_id: str, candidate: str) -> dict:
        root = self._require_job(job_id)
        if self._is_running(job_id):
            raise RuntimeError("Chờ pipeline hoàn tất trước khi chọn ảnh bìa.")
        thumbnails = pipeline.select_thumbnail(root, candidate)
        return {
            "selected": thumbnails.get("primary_candidate", ""),
            "thumbnails": thumbnails,
        }

    def get_midroll(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        draft = self._read_json(root, "midroll-draft.json")
        script = self._read_json(root, "script.json")
        staged = next((s for s in script.get("sections", []) if isinstance(s, dict) and s.get("midroll")), None)
        start_seconds = None
        if isinstance(staged, dict):
            start_seconds = staged.get("start_seconds")
        if start_seconds is None and isinstance(draft, dict):
            start_seconds = draft.get("start_seconds")
        return {
            "draft": draft,
            "staged": staged,
            "approved": bool(script.get("approved")),
            "insertion": {
                "status": "inserted" if staged else "draft" if draft else "missing",
                "start_seconds": start_seconds,
                "location_label": (
                    f"Khoảng {round(float(start_seconds))} giây · gần giữa video"
                    if isinstance(start_seconds, (int, float)) else "Chưa xác định vị trí"
                ),
            },
        }

    def generate_midroll(self, job_id: str) -> dict:
        from .agy_agent import run_agy_json
        root = self._require_job(job_id)
        manifest = pipeline.load_manifest(root)
        script = self._read_json(root, "script.json")
        plan = self._read_json(root, "scene_plan.json")
        # The mid-roll CTA is always AGY-written through its own pool call, so it
        # runs regardless of the project's content_agent (scaffold/claude included).
        if not script.get("approved") or any(s.get("midroll") for s in script.get("sections", []) if isinstance(s, dict)):
            raise ValueError("Kịch bản cần được duyệt và chưa có CTA.")
        schema = {"type": "object", "properties": {"line": {"type": "string"}},
                  "required": ["line"], "additionalProperties": False}
        midpoint = len(script["sections"]) // 2
        brand_name = branding.load_settings(self.jobs_root)["name"]
        previous = script["sections"][midpoint - 1]["title"]
        following = script["sections"][midpoint]["title"]
        if narration_style.is_vietnamese(manifest.config.language):
            prompt = (
                "Viết một lời thoại CTA bằng tiếng Việt, 25–40 từ, một hoặc hai câu, "
                "hài hước tự nhiên theo chi tiết của phim. Nhắc bấm thích và đăng ký "
                f"kênh {brand_name} để không bỏ lỡ phần tiếp theo. Chèn ở 50% video giữa "
                f"'{previous}' và '{following}' của '{manifest.config.movie_title}'. "
                "Không bịa sự kiện, không lặp nguyên văn thoại phim. Trả JSON đúng schema."
            )
        else:
            prompt = (
                f"Write one spoken call-to-action line in "
                f"{narration_style.language_name(manifest.config.language)}, 25-40 words, "
                "one or two sentences, with light humor tied to a detail of the movie. Ask viewers "
                f"to like and subscribe to {brand_name} so they don't miss what comes next. It plays "
                f"at the 50% mark between '{previous}' and '{following}' of "
                f"'{manifest.config.movie_title}'. Do not invent events or quote dialogue verbatim; "
                "never say 'smash that like button'. Return JSON matching the schema."
            )
        result = run_agy_json(stage="midroll", prompt=prompt, schema=schema)
        line = str(result.get("line") or "").strip()
        render = self._read_json(root, "render.json")
        _, _, at = midroll.prepare(script, plan, line, 10, render.get("narration_duration_seconds"))
        draft = {"line": line, "start_seconds": at, "duration_seconds": 10, "generator": "agy"}
        pipeline._write_json(root, "midroll-draft.json", draft)
        return draft

    def license_status(self) -> dict:
        """Current machine's license state (ok / reason / machine fingerprint)."""
        return licensing.current_status().as_dict()

    def activate_license(self, license_text) -> dict:
        """Verify a pasted license for this machine and persist it only if valid."""
        return licensing.activate(str(license_text or "")).as_dict()

    def check_tts_connection(self, provider: str, *, http_request=None) -> dict:
        """Check a TTS provider's key/connection using its env key (no manifest)."""
        provider = (provider or "").strip().lower() or "edge"
        return tts_providers.check_provider(
            provider,
            api_key=tts_providers.provider_api_key(provider),
            http_request=http_request,
        )

    def install_package(self, package: str, *, runner=None) -> dict:
        """Start a background ``pip install`` for a whitelisted package (1-click TTS setup).

        Security: only keys of ``_INSTALLABLE_PACKAGES`` are accepted (arbitrary input
        can never reach pip), the command runs with ``shell=False`` via
        ``sys.executable -m pip``, and the route stays behind the dashboard bearer token
        on a localhost bind. ``runner`` is injectable so tests never touch real pip.
        """
        package = (package or "").strip().lower()
        allowed_packages = (*_INSTALLABLE_PACKAGES, "propainter")
        if package not in allowed_packages:
            allowed = ", ".join(sorted(allowed_packages))
            raise ValueError(f"Gói không được phép cài (chỉ {allowed}): {package!r}")
        with self._lock:
            current = self._installs.get(package)
            if current and current.get("status") == "running":
                return dict(current)
            self._installs[package] = {"package": package, "status": "running",
                                       "detail": f"Đang cài {package}…"}
        run = runner or (lambda selected: propainter_setup.install() if selected == "propainter" else self._pip_install(selected))
        threading.Thread(target=self._run_install, args=(package, run),
                         name=f"mrf-install-{package}", daemon=True).start()
        with self._lock:
            return dict(self._installs[package])

    def _run_install(self, package: str, run) -> None:
        try:
            run(package)
            state = {"package": package, "status": "done", "detail": f"Đã cài {package} thành công."}
        except Exception as exc:  # surface the failure detail to the polling UI
            state = {"package": package, "status": "error",
                     "detail": (str(exc) or "cài đặt thất bại")[:500]}
        with self._lock:
            self._installs[package] = state

    @staticmethod
    def _run_installer_command(args: list[str], *, timeout: int = 900) -> None:
        """Run one installer command without a shell and return a bounded error."""
        try:
            proc = pipeline.subprocess.run(
                args, capture_output=True, text=True, shell=False, timeout=timeout,
            )
        except pipeline.subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Cài đặt quá thời gian ({timeout} giây): {args[0]}") from exc
        except OSError as exc:
            raise RuntimeError(f"Không thể chạy {args[0]}: {exc}") from exc
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "installer failed").strip()[-1000:]
            raise RuntimeError(detail)

    @staticmethod
    def _pip_install(package: str) -> None:
        """Install a whitelisted package, isolating VieNeu from Windows 3.14+.

        VieNeu's compiled dependency may not publish a matching wheel for a newly
        released CPython. Prefer a Python 3.12 venv there rather than attempting a
        fragile local C++ build. Paths are discovered/configurable, never machine-
        specific; other packages and supported Python versions retain legacy pip.
        """
        package = (package or "").strip().lower()
        if package not in _INSTALLABLE_PACKAGES:
            raise ValueError(f"Gói không được phép: {package!r}")
        isolate = package == "vieneu" and os.name == "nt" and sys.version_info >= (3, 14)
        if not isolate:
            JobsService._run_installer_command([
                sys.executable, "-m", "pip", "install", "--disable-pip-version-check", package,
            ])
            return

        env_dir = tts_providers.vieneu_env_dir()
        env_python = tts_providers.vieneu_env_python(env_dir)
        env_dir.parent.mkdir(parents=True, exist_ok=True)
        uv = shutil.which("uv")
        if not env_python.is_file():
            if uv:
                JobsService._run_installer_command([
                    uv, "venv", str(env_dir), "--python", "3.12", "--no-project",
                ])
            else:
                py = shutil.which("py")
                if not py:
                    raise RuntimeError(
                        "Python 3.14 chưa có wheel VieNeu phù hợp. Hãy cài uv hoặc Python 3.12, "
                        f"rồi thử lại (có thể đặt {tts_providers.VIENEU_ENV_ENV})."
                    )
                JobsService._run_installer_command([
                    py, "-3.12", "-m", "venv", str(env_dir),
                ])
        if not env_python.is_file():
            raise RuntimeError("Không tạo được môi trường Python 3.12 riêng cho VieNeu-TTS.")
        if uv:
            command = [uv, "pip", "install", "--python", str(env_python), package]
        else:
            command = [str(env_python), "-m", "pip", "install", "--disable-pip-version-check", package]
        JobsService._run_installer_command(command, timeout=1800)
        JobsService._run_installer_command([
            str(env_python), "-c", "import vieneu",
        ], timeout=60)

    def install_status(self, package: str) -> dict:
        """Report install progress, including VieNeu's isolated environment."""
        package = (package or "").strip().lower()
        if package not in (*_INSTALLABLE_PACKAGES, "propainter"):
            raise ValueError(f"Gói không được phép: {package!r}")
        with self._lock:
            state = self._installs.get(package)
        if state:
            return dict(state)
        if package == "propainter":
            status = "installed" if propainter_setup.is_ready() else "absent"
            detail = "Đã sẵn sàng." if status == "installed" else "Chưa cài."
            return {"package": package, "status": status, "detail": detail}
        if package == "vieneu" and tts_providers.vieneu_available():
            return {"package": package, "status": "installed", "detail": "Đã sẵn sàng."}
        import importlib
        try:
            importlib.import_module(_INSTALLABLE_PACKAGES[package])
            return {"package": package, "status": "installed", "detail": "Đã sẵn sàng."}
        except Exception:
            return {"package": package, "status": "absent", "detail": "Chưa cài."}

    def agy_pool_status(self) -> dict:
        from .agy_vision import pool_status
        return pool_status()

    def probe_agy(self) -> dict:
        from .agy_agent import probe
        return probe()

    def stage_midroll(self, job_id: str, payload: dict) -> dict:
        root = self._require_job(job_id)
        with self._version_lock, self._lock:
            if self._runs.get(job_id, {}).get("running") or job_id in self._uploads or job_id in self._deleting:
                raise RuntimeError("Project đang chạy; thử lại sau.")
            draft = self._read_json(root, "midroll-draft.json")
            if not draft:
                raise ValueError("Hãy tạo câu CTA bằng AGY trước.")
            line = str(payload.get("line") or draft.get("line") or "").strip()
            versions.create(root, "Trước khi chèn CTA")
            result = midroll.stage(root, line, float(draft.get("duration_seconds") or 10))
        return result

    def approve_script(self, job_id: str) -> dict:
        self._assert_no_preview(job_id)
        root = self._require_job(job_id)
        script = pipeline.approve_script(root)
        return {"approved": True, "script": script}

    def prepare_publish(self, job_id: str, *, confirm: bool) -> dict:
        """The publishing gate. Without ``confirm`` it does nothing and reports
        the current gate state. With ``confirm`` it runs only the ``publish``
        stage, which writes ``publish_record.json`` when approvals are complete
        and otherwise stays ``skipped``. It never uploads anything."""
        root = self._require_job(job_id)
        meta = self._read_json(root, pipeline.METADATA_NAME)
        if not confirm:
            return {
                "confirmed": False,
                "published": False,
                "metadata_approved": bool(meta.get("approved")),
                "message_vi": "Cần xác nhận rõ ràng để tạo bản ghi xuất bản (không tải lên).",
            }
        manifest = pipeline.run_job(root, until="publish")
        stage = manifest.stage("publish")
        status = stage.status if stage else "pending"
        message = stage.message if stage else ""
        return {
            "confirmed": True,
            "published": status == "ready",
            "status": status,
            "status_label": localization.status_label(status),
            "message": message,
            "message_vi": localization.localize_message(message),
        }

    def scout_discover(self, topic: str = "all", source: str = "all", refresh: bool = False) -> dict:
        candidates = content_scout.discover_hidden_gems(topic=topic, source=source, refresh=refresh)
        return {"candidates": candidates, "total": len(candidates)}

    def scout_enqueue(self, payload: dict) -> dict:
        candidate_id = str(payload.get("candidate_id") or "")
        data = content_scout.enqueue_gem_for_review(candidate_id)
        if payload.get("auto_create"):
            candidate = data["candidate"]
            slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", candidate["title"]).strip("-").lower()[:32]
            job_id = f"scout-{slug}"
            count = 1
            unique_job_id = job_id
            while (self.jobs_root / unique_job_id).exists():
                count += 1
                unique_job_id = f"{job_id}-{count}"
            defaults = self.creator_library.active_channel_defaults() or {}
            title = str(payload.get("movie_title") or candidate.get("vietnamese_title") or candidate.get("title") or "").strip()
            create_payload = {
                **defaults,
                "job_id": unique_job_id,
                "movie_title": title,
                "content_agent": payload.get("content_agent") or "agy",
                "visual_variety": payload.get("visual_variety") or defaults.get("visual_variety") or "balanced",
                "tts_provider": payload.get("tts_provider") or defaults.get("tts_provider") or "edge",
                "tts_voice": payload.get("tts_voice") if payload.get("tts_voice") is not None else defaults.get("tts_voice", ""),
            }
            if payload.get("watermark_enabled", True):
                create_payload["watermark_detect"] = payload.get("watermark_detect") or "color"
                create_payload["watermark_method"] = payload.get("watermark_method") or "propainter"
            created = self.create_job(create_payload)
            data["created_job"] = created
        return data


# --- HTTP layer --------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: JobsService, *, require_license: bool = False):
        self.service = service
        self.require_license = require_license
        self._handler_slots = threading.BoundedSemaphore(_MAX_CONCURRENT_HANDLERS)
        super().__init__(address, MRFRequestHandler)

    def process_request(self, request, client_address) -> None:
        if not self._handler_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Connection: close\r\nContent-Length: 0\r\n\r\n"
                )
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._handler_slots.release()
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_slots.release()

    def server_close(self) -> None:
        try:
            super().server_close()
        finally:
            self.service.close()


class MRFRequestHandler(BaseHTTPRequestHandler):
    server_version = "MovieReviewFactory/0.2"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(_REQUEST_TIMEOUT_SECONDS)

    @property
    def service(self) -> JobsService:
        return self.server.service  # type: ignore[attr-defined]

    def _license_blocked(self) -> bool:
        """True only when this server enforces licensing and no valid license exists."""
        if not getattr(self.server, "require_license", False):
            return False
        return not self.service.license_status()["ok"]

    # -- low-level responders ------------------------------------------------

    def _send_json(self, status: int, obj: object) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if status == 200 and _DASHBOARD_TOKEN and self._has_bearer_token():
            self.send_header(
                "Set-Cookie",
                f"mrf_media={self._media_cookie()}; HttpOnly; SameSite=Strict; Path=/api/jobs/",
            )
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        if status in (401, 403):
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > _MAX_BODY_BYTES:
            raise OverflowError(f"request body too large ({length} > {_MAX_BODY_BYTES})")
        try:
            raw = self.rfile.read(length)
        except (socket.timeout, TimeoutError) as exc:
            raise RequestTimeoutError("request body read timed out") from exc
        if len(raw) != length:
            raise ValueError("request body was incomplete")
        try:
            data = json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise ValueError(f"JSON không hợp lệ: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("body phải là một đối tượng JSON")
        return data

    @staticmethod
    def _error_payload(exc: Exception) -> dict:
        message = str(exc)
        return {"error": message, "error_vi": localization.localize_message(message)}

    def _dispatch(self, handler) -> None:
        """Run a route handler, mapping domain exceptions to HTTP status codes."""
        try:
            handler()
        except RequestTimeoutError as exc:
            self._send_json(408, self._error_payload(exc))
        except ValueError as exc:
            self._send_json(400, self._error_payload(exc))
        except FileNotFoundError as exc:
            self._send_json(404, self._error_payload(exc))
        except KeyError as exc:
            self._send_json(404, self._error_payload(exc))
        except FileExistsError as exc:
            self._send_json(409, self._error_payload(exc))
        except RuntimeError as exc:
            self._send_json(409, self._error_payload(exc))
        except OverflowError:
            self._send_json(413, {"error": "request body too large", "error_vi": "body yêu cầu quá lớn"})
        except Exception:  # last resort — never leak a stack trace or exception text
            self._send_json(500, {"error": "internal server error", "error_vi": "lỗi máy chủ nội bộ"})

    # -- file streaming with HTTP range support (for video/audio preview) ----

    def _serve_file(self, path: Path) -> None:
        file_size = path.stat().st_size
        start, end, status = 0, file_size - 1, 200
        range_header = self.headers.get("Range")
        if range_header and range_header.startswith("bytes="):
            raw = range_header.split("=", 1)[1].split(",")[0].strip()
            bounds = re.fullmatch(r"(\d*)-(\d*)", raw)
            if bounds and (bounds[1] or bounds[2]):
                begin, finish = bounds.groups()
                if begin:
                    start = int(begin)
                    end = int(finish) if finish else file_size - 1
                else:
                    suffix = int(finish)
                    start = max(0, file_size - suffix)
                    end = file_size - 1
            valid_bounds = bool(bounds and (bounds[1] or bounds[2]))
            valid_suffix = bool(bounds and (bounds[1] or (bounds[2] and int(bounds[2]) > 0)))
            if not valid_bounds or not valid_suffix or start > end or start >= file_size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{file_size}")
                for hk, hv in _SECURITY_HEADERS.items():
                    self.send_header(hk, hv)
                self.end_headers()
                return
            end = min(end, file_size - 1)
            status = 206

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", _content_type(path.name))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        self.end_headers()
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (ConnectionError, TimeoutError):
                    # Browser stopped reading (seek, navigation, paused buffer); the socket is unusable.
                    self.close_connection = True
                    return
                remaining -= len(chunk)

    # -- auth ----------------------------------------------------------------

    @staticmethod
    def _media_cookie() -> str:
        return hmac.new(_DASHBOARD_TOKEN.encode("utf-8"), b"media-stream-v1", hashlib.sha256).hexdigest()

    def _has_bearer_token(self) -> bool:
        auth = self.headers.get("Authorization", "")
        return auth.startswith("Bearer ") and hmac.compare_digest(
            auth[len("Bearer "):].encode("utf-8"), _DASHBOARD_TOKEN.encode("utf-8")
        )

    def _check_auth(self) -> bool:
        if not _DASHBOARD_TOKEN or self._has_bearer_token():
            return True
        route = urlparse(self.path).path
        media_route = re.fullmatch(
            r"/api/jobs/[A-Za-z0-9._-]+/(?:media/source|artifacts/(?:final\.mp4|review-handoff\.zip))", route
        )
        if self.command != "GET" or not media_route:
            return False
        cookies = dict(
            part.strip().split("=", 1) for part in self.headers.get("Cookie", "").split(";")
            if "=" in part
        )
        return hmac.compare_digest(cookies.get("mrf_media", ""), self._media_cookie())

    def _send_401(self) -> None:
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Bearer realm="dashboard"')
        self.send_header("Content-Length", "0")
        for hk, hv in _SECURITY_HEADERS.items():
            self.send_header(hk, hv)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    # -- routing -------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 (http.server API)
        parsed = urlparse(self.path)
        route = parsed.path
        if route in ("/", "/index.html"):
            self._send_html(ACTIVATION_HTML if self._license_blocked() else INDEX_HTML)
            return
        # UI fonts are public static files: CSS @font-face requests carry no bearer token.
        font_match = re.fullmatch(r"/assets/fonts/([a-z-]+\.woff2)", route)
        if font_match and (_UI_FONTS / font_match[1]).is_file():
            self._serve_file(_UI_FONTS / font_match[1])
            return
        if not self._check_auth():
            self._send_401()
            return
        if route == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        parts = [unquote(p) for p in route.split("/") if p]
        if not parts or parts[0] != "api":
            self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})
            return
        self._dispatch(lambda: self._route_get(parts[1:], parse_qs(parsed.query)))

    def do_DELETE(self) -> None:  # noqa: N802 (http.server API)
        if not self._check_auth():
            self._send_401()
            return
        parts = [unquote(p) for p in urlparse(self.path).path.split("/") if p]
        if (
            parts == ["api", "jobs"]
            or (len(parts) == 3 and parts[:2] == ["api", "jobs"])
            or (len(parts) == 3 and parts[:2] == ["api", "channels"])
            or (len(parts) == 5 and parts[:2] == ["api", "channels"] and parts[3] == "sfx")
        ):
            self._dispatch(lambda: self._route_delete(parts))
            return
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def _route_delete(self, parts: list[str]) -> None:
        if parts[:2] == ["api", "channels"] and len(parts) == 5 and parts[3] == "sfx":
            self._send_json(200, self.service.delete_channel_sfx(parts[2], parts[4]))
            return
        if parts[:2] == ["api", "channels"]:
            self._send_json(200, self.service.delete_channel(parts[2]))
            return
        body = self._read_body()
        if len(parts) == 2:
            self._send_json(200, self.service.delete_jobs(body.get("job_ids"), str(body.get("confirm") or "")))
        else:
            self._send_json(200, self.service.delete_job(parts[2], str(body.get("confirm") or "")))

    def do_POST(self) -> None:  # noqa: N802 (http.server API)
        if not self._check_auth():
            self._send_401()
            return
        parsed = urlparse(self.path)
        parts = [unquote(p) for p in parsed.path.split("/") if p]
        if not parts or parts[0] != "api":
            self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})
            return
        self._dispatch(lambda: self._route_post(parts[1:]))

    def _route_get(self, parts: list[str], query: dict[str, list[str]]) -> None:
        # parts: [] | ["jobs"] | ["jobs", id] | ["jobs", id, "artifacts"] |
        #        ["jobs", id, "artifacts", name] | ["jobs", id, "metadata"]
        if parts == ["license"]:
            self._send_json(200, self.service.license_status())
            return
        if self._license_blocked():
            self._send_json(402, {"error": "license required", "error_vi": "cần kích hoạt license"})
            return
        if parts == ["brand"]:
            self._send_json(200, {**branding.load_settings(self.service.jobs_root), "logo_url": "/api/brand/logo"})
            return
        if parts == ["brand", "logo"]:
            self._serve_file(branding.logo_path(self.service.jobs_root))
            return
        if parts == ["brand", "logo.svg"]:
            self._serve_file(branding.ASSETS / "man-ke.svg")
            return
        if parts == ["channels"]:
            self._send_json(200, self.service.list_channels())
            return
        if len(parts) == 3 and parts[0] == "channels" and parts[2] == "logo":
            self._serve_file(self.service.channel_logo_file(parts[1]))
            return
        if parts == ["channel-sfx"]:
            self._send_json(200, self.service.active_channel_sfx())
            return
        if len(parts) == 5 and parts[0] == "channels" and parts[2] == "sfx" and parts[4] == "file":
            self._serve_file(self.service.channel_sfx_file(parts[1], parts[3]))
            return
        if parts == ["agy-pool"]:
            self._send_json(200, self.service.agy_pool_status())
            return
        if parts == ["tts", "test"]:
            self._send_json(200, self.service.check_tts_connection((query.get("provider") or [""])[0]))
            return
        if parts == ["system", "install-status"]:
            self._send_json(200, self.service.install_status((query.get("package") or [""])[0]))
            return
        if parts == ["jobs"]:
            self._send_json(200, {"jobs": self.service.list_jobs()})
            return
        if parts == ["scout", "discover"]:
            topic = (query.get("topic") or ["all"])[0]
            source = (query.get("source") or ["all"])[0]
            raw_refresh = (query.get("refresh") or ["0"])[0].lower()
            refresh = raw_refresh in ("1", "true", "yes")
            self._send_json(200, self.service.scout_discover(topic=topic, source=source, refresh=refresh))
            return
        if parts == ["batch"]:
            self._send_json(200, self.service.batch_status())
            return
        if parts == ["creator-library", "search"]:
            self._send_json(200, self.service.search_creator_projects((query.get("q") or [""])[0]))
            return
        if parts == ["creator-library", "series"]:
            self._send_json(200, self.service.list_creator_series())
            return
        if parts == ["creator-library", "briefs"]:
            self._send_json(200, self.service.list_creator_briefs())
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "rights":
            self._send_json(200, self.service.list_creator_rights(parts[1]))
            return
        if parts == ["library-search"]:
            search = (query.get("q") or [""])[0].strip()
            filters = {
                key: (query.get(key) or [""])[0].strip()
                for key in (
                    "person", "action", "location", "object", "kind", "project",
                    "scene_type", "source", "date_from", "date_to",
                    "min_duration", "max_duration", "min_confidence",
                )
            }
            self._send_json(200, self.service.library_search(search, filters))
            return
        if parts == ["library-searches"]:
            self._send_json(200, self.service.list_saved_searches())
            return
        if len(parts) == 2 and parts[0] == "jobs":
            self._send_json(200, self.service.status(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "analytics":
            self._send_json(200, self.service.list_analytics(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "versions":
            self._send_json(200, self.service.list_versions(parts[1]))
            return
        if len(parts) == 6 and parts[0] == "jobs" and parts[2] == "versions" and parts[4] == "artifacts":
            self._serve_file(self.service.version_artifact(parts[1], parts[3], parts[5]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "artifacts":
            self._send_json(200, {"artifacts": self.service.list_artifacts(parts[1])})
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "artifacts":
            self._serve_file(self.service.artifact_path(parts[1], parts[3]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "shorts":
            self._serve_file(self.service.short_path(parts[1], parts[3]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "previews":
            self._send_json(200, self.service.section_preview_state(parts[1], int(parts[3])))
            return
        if len(parts) == 5 and parts[0] == "jobs" and parts[2] == "previews":
            self._serve_file(self.service.section_preview_path(parts[1], int(parts[3]), parts[4]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "audio-mix":
            self._send_json(200, self.service.get_audio_mix(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "metadata":
            self._send_json(200, self.service.get_metadata(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "midroll":
            self._send_json(200, self.service.get_midroll(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "script":
            self._send_json(200, self.service.get_script(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "thumbnails":
            self._send_json(200, self.service.get_thumbnails(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "media-explorer":
            search = (query.get("q") or [""])[0].strip()
            self._send_json(200, self.service.media_explorer(parts[1], search))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "person-tracks":
            self._send_json(200, self.service.person_tracks(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "timeline":
            self._send_json(200, self.service.timeline(parts[1]))
            return
        if (
            len(parts) == 5
            and parts[0] == "jobs"
            and parts[2] == "shots"
            and parts[4] == "similar"
        ):
            self._send_json(200, self.service.similar_scenes(parts[1], int(parts[3])))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "highlights":
            self._send_json(200, self.service.highlights(parts[1]))
            return
        if len(parts) == 5 and parts[0] == "jobs" and parts[2] == "highlights" and parts[4] == "clip.mp4":
            self._serve_file(self.service.highlight_clip_path(parts[1], parts[3]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "transcript.vtt":
            self._serve_file(self.service.transcript_export_path(parts[1], "vtt"))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "media" and parts[3] == "source":
            self._serve_file(self.service.source_media_path(parts[1]))
            return
        if len(parts) == 5 and parts[0] == "jobs" and parts[2] == "shots" and parts[4] == "thumbnail":
            self._serve_file(self.service.shot_thumbnail_path(parts[1], int(parts[3])))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "hook.mp4":
            self._serve_file(self.service.hook_path(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "hook":
            self._send_json(200, self.service.get_hook_info(parts[1]))
            return
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def _route_post(self, parts: list[str]) -> None:
        if parts == ["license", "activate"]:
            self._send_json(200, self.service.activate_license(self._read_body().get("license")))
            return
        if self._license_blocked():
            self._send_json(402, {"error": "license required", "error_vi": "cần kích hoạt license"})
            return
        if parts == ["system", "install-package"]:
            self._send_json(200, self.service.install_package(str(self._read_body().get("package") or "")))
            return
        if parts == ["brand"]:
            name = self._read_body().get("name")
            self._send_json(200, branding.save_name(self.service.jobs_root, name))
            return
        if parts == ["brand", "logo"]:
            try:
                size = int(self.headers.get("Content-Length") or "0")
            except ValueError as exc:
                raise ValueError("Content-Length không hợp lệ") from exc
            if size <= 0 or size > 2_000_000:
                raise ValueError("Logo PNG cần dung lượng 1 byte–2 MB.")
            branding.save_logo(self.service.jobs_root, self.rfile.read(size))
            self._send_json(200, {"logo_url": "/api/brand/logo"})
            return
        if parts == ["channels"]:
            self._send_json(200, self.service.save_channel(self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "channels" and parts[2] == "activate":
            self._send_json(200, self.service.activate_channel(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "channels" and parts[2] == "logo":
            try:
                size = int(self.headers.get("Content-Length") or "0")
            except ValueError as exc:
                raise ValueError("Content-Length không hợp lệ") from exc
            if size <= 0 or size > 2_000_000:
                raise ValueError("Logo PNG cần dung lượng 1 byte–2 MB.")
            self._send_json(200, self.service.save_channel_logo(parts[1], self.rfile.read(size)))
            return
        if len(parts) == 3 and parts[0] == "channels" and parts[2] == "sfx":
            self._send_json(200, self.service.save_channel_sfx(parts[1], self._read_body()))
            return
        if len(parts) == 5 and parts[0] == "channels" and parts[2] == "sfx" and parts[4] == "file":
            try:
                size = int(self.headers.get("Content-Length") or "0")
            except ValueError as exc:
                raise ValueError("Content-Length không hợp lệ") from exc
            if size <= 0 or size > 3_000_000:
                raise ValueError("Tệp SFX cần 1 byte–3 MB.")
            ctype = self.headers.get("Content-Type") or ""
            self._send_json(200, self.service.save_channel_sfx_file(parts[1], parts[3], ctype, self.rfile.read(size)))
            return
        if parts == ["agy-pool", "probe"]:
            self._send_json(200, self.service.probe_agy())
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "probe-detector":
            self._send_json(200, self.service.probe_detector(parts[1]))
            return
        if parts == ["jobs"]:
            self._send_json(201, self.service.create_job(self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "config":
            self._send_json(200, self.service.update_job_config(parts[1], self._read_body()))
            return
        if parts == ["batch"]:
            self._send_json(202, self.service.start_batch(self._read_body()))
            return
        if parts == ["scout", "enqueue"]:
            self._send_json(200, self.service.scout_enqueue(self._read_body()))
            return
        if parts == ["batch", "stop"]:
            self._send_json(200, self.service.stop_batch())
            return
        if parts == ["link-probe"]:
            body = self._read_body()
            self._send_json(200, self.service.probe_link(body.get("url", "")))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "download-link":
            body = self._read_body()
            self._send_json(200, self.service.download_link_video(
                parts[1],
                body.get("url", ""),
                confirm_rights=bool(body.get("confirm_rights")),
                sub_langs=str(body.get("sub_langs") or "vi,en"),
            ))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "hook":
            self._send_json(202, self.service.generate_hook(parts[1]))
            return
        if parts == ["creator-library", "series"]:
            self._send_json(201, self.service.save_creator_series(self._read_body()))
            return
        if parts == ["creator-library", "briefs"]:
            self._send_json(201, self.service.save_creator_brief(self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "rights":
            self._send_json(201, self.service.save_creator_rights(parts[1], self._read_body()))
            return
        if parts == ["library-searches"]:
            self._send_json(201, self.service.save_search(self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "source":
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError as exc:
                raise ValueError("Content-Length không hợp lệ") from exc
            filename = self.headers.get("X-Source-Name") or ""
            self._send_json(200, self.service.import_video(parts[1], filename, length, self.rfile))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "versions":
            self._send_json(201, self.service.create_version(parts[1], str(self._read_body().get("name") or "")))
            return
        if len(parts) == 5 and parts[0] == "jobs" and parts[2] == "versions" and parts[4] == "restore":
            self._send_json(200, self.service.restore_version(parts[1], parts[3], str(self._read_body().get("kind") or "")))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "handoff":
            self._send_json(200, self.service.build_handoff(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "analytics":
            try:
                length = int(self.headers.get("Content-Length") or "0")
            except ValueError as exc:
                raise ValueError("Content-Length không hợp lệ") from exc
            if length > analytics.MAX_EXPORT_BYTES:
                raise OverflowError("Studio export exceeds 2 MB")
            if length <= 0:
                raise ValueError("File Studio rỗng.")
            content = self.rfile.read(length)
            if len(content) != length:
                raise ValueError("File Studio tải lên không đầy đủ.")
            self._send_json(201, self.service.import_analytics(
                parts[1], self.headers.get("X-Studio-Format") or "", content,
                self.headers.get("X-Studio-CTA-Seconds"),
                unquote(self.headers.get("X-Studio-Notes") or "")))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "shorts":
            self._send_json(202, self.service.start_short(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "previews":
            self._send_json(202, self.service.start_section_preview(parts[1], int(parts[3])))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "brand-render":
            self._send_json(202, self.service.rerender_brand(parts[1], self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "run":
            self._send_json(202, self.service.start_run(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "index":
            self._send_json(202, self.service.enqueue_index(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "reindex-transcript":
            self._send_json(202, self.service.reindex_transcript(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "stop":
            self._send_json(202, self.service.stop_run(parts[1]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "audio-mix" and parts[3] == "sfx":
            self._send_json(200, self.service.add_audio_mix_sfx(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "audio-mix" and parts[3] == "transitions":
            self._send_json(200, self.service.place_transition_sfx(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "audio-mix":
            self._send_json(200, self.service.update_audio_mix(parts[1], self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "metadata":
            self._send_json(200, self.service.update_metadata(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "metadata" and parts[3] == "approve":
            self._send_json(200, self.service.approve_metadata(parts[1]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "midroll" and parts[3] == "draft":
            self._send_json(200, self.service.generate_midroll(parts[1]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "midroll" and parts[3] == "stage":
            self._send_json(200, self.service.stage_midroll(parts[1], self._read_body()))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "script":
            self._send_json(200, self.service.update_script(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "script" and parts[3] == "tag":
            self._send_json(200, self.service.tag_script(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "script" and parts[3] == "approve":
            self._send_json(200, self.service.approve_script(parts[1]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "thumbnails" and parts[3] == "auto":
            self._send_json(200, self.service.auto_generate_thumbnails(parts[1]))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "thumbnails" and parts[3] == "edit":
            body = self._read_body()
            self._send_json(200, self.service.edit_thumbnail(
                parts[1], str(body.get("headline") or ""), str(body.get("channel_name") or "")))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "thumbnails" and parts[3] == "select":
            body = self._read_body()
            candidate = str(body.get("candidate") or "").strip()
            if not candidate:
                raise ValueError("thiếu candidate ảnh bìa")
            self._send_json(200, self.service.select_thumbnail(parts[1], candidate))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "chat":
            body = self._read_body()
            self._send_json(200, self.service.chat(parts[1], str(body.get("question") or "")))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "timeline":
            self._send_json(200, self.service.edit_timeline(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "broll":
            body = self._read_body()
            self._send_json(
                200,
                self.service.broll(
                    parts[1],
                    int(parts[3]),
                    apply=bool(body.get("apply", False)),
                ),
            )
            return
        if (
            len(parts) == 5
            and parts[0] == "jobs"
            and parts[2] == "person-tracks"
            and parts[4] == "alias"
        ):
            body = self._read_body()
            self._send_json(
                200,
                self.service.set_person_alias(
                    parts[1], parts[3], str(body.get("alias") or "")
                ),
            )
            return
        if (
            len(parts) == 5
            and parts[0] == "jobs"
            and parts[2] == "sections"
            and parts[4] == "regenerate"
        ):
            body = self._read_body()
            self._send_json(
                200,
                self.service.regenerate_section(
                    parts[1],
                    int(parts[3]),
                    str(body.get("instruction") or ""),
                ),
            )
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "publish":
            body = self._read_body()
            result = self.service.prepare_publish(parts[1], confirm=bool(body.get("confirm")))
            self._send_json(200, result)
            return
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        # Keep the console (and test output) quiet; override if debugging.
        return


def create_server(host: str = "127.0.0.1", port: int = 8765,
                  jobs_root: Path | str = "jobs", *, require_license: bool = False) -> _Server:
    """Build (but do not start) the local web server."""
    _require_safe_bind(host)
    service = JobsService(Path(jobs_root))
    try:
        return _Server((host, port), service, require_license=require_license)
    except Exception:
        service.close()
        raise


def run_server(host: str = "127.0.0.1", port: int = 8765,
               jobs_root: Path | str = "jobs", *, require_license: bool = False) -> None:
    """Start the local web server and serve until interrupted."""
    # The dashboard's operator-facing strings are Vietnamese; force UTF-8 on the
    # console streams so a legacy Windows code page (e.g. cp1258) cannot raise
    # UnicodeEncodeError on a startup line or an unhandled-exception traceback.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    server = create_server(host, port, jobs_root, require_license=require_license)
    bound_host, bound_port = server.server_address[:2]
    print(f"Movie Review Factory UI: http://{bound_host}:{bound_port}  (jobs: {Path(jobs_root)})")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        server.service.close()


# --- single-page UI ----------------------------------------------------------
# Vanilla HTML + JS (no framework, no build step). Kept as one inline document so the whole UI ships with the package and needs no static-file plumbing.

_UI_FONT_FACES = """  @font-face{ font-family:'Montserrat'; font-style:normal; font-weight:400 700; font-display:swap; src:url(/assets/fonts/montserrat-vietnamese.woff2) format('woff2');
    unicode-range:U+0102-0103, U+0110-0111, U+0128-0129, U+0168-0169, U+01A0-01A1, U+01AF-01B0, U+0300-0301, U+0303-0304, U+0308-0309, U+0323, U+0329, U+1EA0-1EF9, U+20AB; }
  @font-face{ font-family:'Montserrat'; font-style:normal; font-weight:400 700; font-display:swap; src:url(/assets/fonts/montserrat-latin.woff2) format('woff2');
    unicode-range:U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }
  @font-face{ font-family:'Cormorant'; font-style:normal; font-weight:500 700; font-display:swap; src:url(/assets/fonts/cormorant-vietnamese.woff2) format('woff2');
    unicode-range:U+0102-0103, U+0110-0111, U+0128-0129, U+0168-0169, U+01A0-01A1, U+01AF-01B0, U+0300-0301, U+0303-0304, U+0308-0309, U+0323, U+0329, U+1EA0-1EF9, U+20AB; }
  @font-face{ font-family:'Cormorant'; font-style:normal; font-weight:500 700; font-display:swap; src:url(/assets/fonts/cormorant-latin.woff2) format('woff2');
    unicode-range:U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD; }"""


ACTIVATION_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kích hoạt · Xưởng Review Phim</title>
<style>
/*@ui-fonts*/
  :root{ color-scheme:dark; --gold-grad:linear-gradient(135deg, #8c6a2f 0%, #c9a961 34%, #f3dfa8 52%, #c9a961 70%, #9a7735 100%); }
  body { margin:0; font-family:'Montserrat', system-ui, "Segoe UI", sans-serif; color:#f2ede4;
         background:radial-gradient(1100px 560px at 50% -12%, #221d15 0%, #0b0a08 62%) #0b0a08;
         display:flex; min-height:100vh; align-items:center; justify-content:center; }
  .box { width:min(560px, 92vw); background:#15130f; border:1px solid #2a251d; border-radius:18px; padding:30px;
         box-shadow:0 28px 70px rgba(0,0,0,.6), inset 0 1px 0 rgba(243,223,168,.07); }
  .brand { display:flex; align-items:center; gap:10px; margin-bottom:18px; color:#c9a961; font-size:11.5px; font-weight:600; letter-spacing:.16em; text-transform:uppercase; }
  .brand-mark { display:grid; place-items:center; width:34px; height:34px; border-radius:10px; background:var(--gold-grad); color:#1a1407;
                box-shadow:0 6px 20px rgba(201,169,97,.22), inset 0 1px 0 rgba(255,255,255,.35); }
  h1 { font-family:'Cormorant', Georgia, serif; font-size:32px; font-weight:600; line-height:1.12; margin:0 0 10px;
       background:var(--gold-grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
  p { color:#b8ae9c; font-size:13.5px; line-height:1.6; margin:0 0 8px; }
  code { display:inline-block; background:#100e0b; border:1px solid #3a3328; border-radius:8px; padding:4px 10px; word-break:break-all;
         color:#f3dfa8; font:12.5px/1.5 Consolas, "Cascadia Mono", monospace; }
  label { display:block; color:#948873; font-size:12px; font-weight:600; letter-spacing:.04em; margin-top:18px; }
  textarea { box-sizing:border-box; width:100%; min-height:96px; margin-top:8px; background:#100e0b; color:#f2ede4; resize:vertical;
             border:1px solid #3a3328; border-radius:12px; padding:10px 12px; font:12.5px/1.5 Consolas, "Cascadia Mono", monospace;
             transition:border-color .2s, box-shadow .2s; }
  textarea::placeholder { color:#6f6556; }
  textarea:focus, button:focus-visible { outline:none; border-color:#c9a961; box-shadow:0 0 0 3px rgba(201,169,97,.42); }
  button { margin-top:16px; padding:11px 22px; border:0; border-radius:999px; background:var(--gold-grad); color:#1a1407;
           font:600 14px 'Montserrat', system-ui, sans-serif; cursor:pointer; box-shadow:0 8px 22px rgba(201,169,97,.2);
           transition:filter .2s, transform .2s; }
  button:hover { filter:brightness(1.08); }
  button:active { transform:translateY(1px); }
  button[disabled] { opacity:.65; cursor:progress; }
  .msg { margin-top:12px; font-size:13px; min-height:18px; color:#b8ae9c; }
  .msg[data-kind="ok"] { color:#a9d69a; }
  .msg[data-kind="err"] { color:#f0a58e; }
  .msg[data-kind="busy"] { background-image:linear-gradient(100deg, #b8ae9c 40%, #f3dfa8 50%, #b8ae9c 60%); background-size:250% 100%;
                           -webkit-background-clip:text; background-clip:text; color:transparent; animation:mrf-sheen 1.8s linear infinite; }
  @keyframes mrf-sheen{ from{ background-position:100% 0; } to{ background-position:-150% 0; } }
  @media (prefers-reduced-motion:reduce){ .msg[data-kind="busy"]{ animation:none; } button{ transition:none; } }
</style>
</head>
<body>
  <div class="box">
    <div class="brand"><span class="brand-mark" aria-hidden="true"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="9" width="18" height="11" rx="2"/><path d="M3 9l2.5-5 15 2.5L18 9"/><path d="M8.5 4.6L7 9M13.5 5.4L12 9"/><path d="M10.5 12.5v4.5l3.8-2.25z" fill="currentColor" stroke="none"/></svg></span>Xưởng Review Phim</div>
    <h1>Cần kích hoạt bản quyền</h1>
    <p id="reason">Phần mềm chưa được kích hoạt trên máy này.</p>
    <p style="margin-top:14px">Mã máy (gửi cho nhà cung cấp để lấy license key):</p>
    <p><code id="machine">…</code></p>
    <label for="key">Dán license key:</label>
    <textarea id="key" placeholder="dán chuỗi license tại đây"></textarea>
    <button id="activate" type="button">Kích hoạt</button>
    <div class="msg" id="msg" role="status"></div>
  </div>
<script>
  const $ = (id) => document.getElementById(id);
  const say = (text, kind) => { $('msg').textContent = text; $('msg').dataset.kind = kind || ''; };
  async function loadStatus() {
    try {
      const res = await fetch('/api/license');
      const data = await res.json();
      $('machine').textContent = data.machine || '(không đọc được)';
      if (data.reason) $('reason').textContent = data.reason;
    } catch (error) { say(error.message, 'err'); }
  }
  $('activate').onclick = async () => {
    say('Đang kiểm tra…', 'busy');
    $('activate').disabled = true;
    try {
      const res = await fetch('/api/license/activate', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({license: $('key').value.trim()}),
      });
      const data = await res.json();
      if (data.ok) { say('Đã kích hoạt. Đang mở…', 'ok'); setTimeout(() => location.reload(), 700); return; }
      say('✗ ' + (data.reason || data.error_vi || 'License không hợp lệ'), 'err');
    } catch (error) { say(error.message, 'err'); }
    $('activate').disabled = false;
  };
  loadStatus();
</script>
</body>
</html>
""".replace("/*@ui-fonts*/", _UI_FONT_FACES)


INDEX_HTML = """<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Xưởng Review Phim — Bảng điều khiển</title>
<style>
  :root { color-scheme: light dark; --gap: 16px; --accent: #338ef7; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, "Segoe UI", Roboto, sans-serif;
         line-height: 1.5; background: #0f1115; color: #e6e8ee; }
  header { padding: 14px 20px; background: #171a21; border-bottom: 1px solid #262b36; }
  header h1 { margin: 0; font-size: 18px; }
  header .sub { color: #9aa3b2; font-size: 13px; }
  .card { background: #171a21; border: 1px solid #262b36; border-radius: 10px; padding: 14px; margin-bottom: var(--gap); }
  .card h2 { margin: 0 0 10px; font-size: 15px; }
  label { display: block; font-size: 12px; color: #9aa3b2; margin: 8px 0 2px; }
  input, select, textarea, button { font: inherit; }
  input, select, textarea { width: 100%; padding: 7px 9px; background: #0f1115; color: #e6e8ee;
      border: 1px solid #333a48; border-radius: 6px; }
  input[type="checkbox"], input[type="radio"] { width: auto !important; margin: 0; cursor: pointer; flex-shrink: 0; }
  .checkbox-row { display: flex; align-items: flex-start; gap: 8px; margin: 8px 0; font-size: 12px; color: var(--text, #e6e8ee); cursor: pointer; }
  .checkbox-row input { margin-top: 2px; }
  textarea { resize: vertical; }
  button { cursor: pointer; padding: 8px 12px; border-radius: 6px; border: 1px solid #333a48;
      background: #232838; color: #e6e8ee; }
  button.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  /* Modern SaaS glass layer: additive overrides preserve existing behavior. */
  html { min-height: 100%; background: #080b12; }
  body { min-height: 100vh; color: #f4f7ff; background: radial-gradient(circle at 12% -8%, rgba(109,140,255,.22), transparent 34rem), radial-gradient(circle at 92% 8%, rgba(139,92,246,.16), transparent 30rem), #080b12; }
  header { position: sticky; top: 0; z-index: 20; padding: 18px clamp(18px,3vw,36px); background: rgba(8,11,18,.72); border-color: rgba(148,163,184,.18); backdrop-filter: blur(18px) saturate(140%); }
  header h1 { font-size: clamp(18px,2vw,23px); letter-spacing: -.025em; }
  .card { background: rgba(20,25,38,.72); border-color: rgba(148,163,184,.18); border-radius: 16px; padding: clamp(14px,2vw,20px); box-shadow: 0 18px 50px rgba(0,0,0,.28); backdrop-filter: blur(18px) saturate(125%); }
  input, select, textarea { padding: 9px 11px; background: rgba(7,10,17,.72); border-color: rgba(148,163,184,.18); border-radius: 9px; }
  input:focus, select:focus, textarea:focus { outline: none; border-color: rgba(109,140,255,.8); box-shadow: 0 0 0 3px rgba(109,140,255,.16); }
  button { border-radius: 9px; border-color: rgba(148,163,184,.18); background: rgba(38,47,69,.82); transition: transform .16s, border-color .16s, background .16s; }
  button:hover:not(:disabled) { transform: translateY(-1px); border-color: rgba(148,163,184,.38); background: rgba(49,60,87,.92); }
  button:focus-visible, a:focus-visible { outline: 3px solid rgba(109,140,255,.55); outline-offset: 2px; }
  button.primary { background: linear-gradient(135deg,#6d8cff,#8b5cf6); border-color: rgba(255,255,255,.16); box-shadow: 0 8px 24px rgba(109,140,255,.24); }
  @media (max-width: 820px) { header { position: relative; } }
  .project-list, .artifact-grid { display: grid; gap: 8px; }
  .project-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
  .project-toolbar label { display: flex; align-items: center; gap: 6px; margin: 0; }
  .project-toolbar input, .project-check input { width: auto; }
  .project-check { display: inline-flex; align-items: center; gap: 6px; margin: 0; }
  .delete-targets { max-height: 140px; overflow: auto; overflow-wrap: anywhere; color: var(--text-dim); }
  .delete-actions { margin-top: 12px; }
  .project-card, .artifact-card { border: 1px solid var(--line); border-radius: 10px; background: var(--panel-raised); padding: 10px; }
  .project-card.active { border-color: var(--accent); background: #1c2438; }
  .project-card .open-project { display: block; width: 100%; padding: 2px; text-align: left; border: 0; background: transparent; }
  .project-card .open-project:hover { color: #bdd0ff; }
  .project-card .card-actions { display: flex; justify-content: space-between; align-items: center; gap: 8px; margin-top: 6px; }
  .project-card progress, .upload-progress { width: 100%; accent-color: var(--accent); }
  .danger { color: #ff9da5; border-color: #75444a; }
  .button-link { display: inline-block; padding: 8px 12px; border-radius: 6px; background: var(--accent); color: #fff; text-decoration: none; }
  .artifact-card a { color: #bdd0ff; overflow-wrap: anywhere; }
  .artifact-card .muted { display: block; }
  .action-note { margin-top: 10px; padding: 10px; border: 1px solid var(--line); border-radius: 8px; color: var(--text-dim); }
  dialog { width: min(440px, calc(100% - 32px)); border: 1px solid var(--line); border-radius: 12px; background: var(--panel); color: #e6e8ee; }
  dialog::backdrop { background: rgba(0,0,0,.72); }
  .muted { color: #9aa3b2; font-size: 12px; }
  .stage { display: flex; align-items: center; gap: 10px; padding: 6px 0; border-bottom: 1px solid #20242e; }
  .stage:last-child { border-bottom: 0; }
  .stage .name { width: 150px; font-weight: 600; }
  .badge { font-size: 11px; padding: 2px 8px; border-radius: 999px; white-space: nowrap; }
  .badge.pending { background: #2b303c; color: #b9c0cc; }
  .badge.running { background: #3a2f13; color: #f5c451; }
  .badge.ready   { background: #133a22; color: #57d98a; }
  .badge.failed  { background: #43181c; color: #ff7a86; }
  .badge.skipped { background: #2b303c; color: #9aa3b2; }
  .progress { height: 8px; background: #0f1115; border: 1px solid #333a48; border-radius: 999px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--accent); width: 0; transition: width .3s; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  .arts a { color: #9db4ff; text-decoration: none; }
  .arts li { margin: 3px 0; }
  video { width: 100%; border-radius: 8px; background: #000; margin-top: 8px; }
  .thumb-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 10px; }
  .thumb-item { border: 1px solid #333a48; border-radius: 8px; padding: 8px; }
  .thumb-item.selected { border-color: var(--accent); background: #1c2438; }
  .thumb-item img { width: 100%; aspect-ratio: 16/9; object-fit: cover; border-radius: 6px; background: #000; }
  .thumb-item button { width: 100%; margin-top: 6px; }
  .media-results { display: grid; gap: 8px; margin-top: 10px; max-height: 420px; overflow-y: auto; }
  .media-result { display: grid; grid-template-columns: 128px 1fr; gap: 10px; align-items: start; border: 1px solid #333a48; border-radius: 8px; padding: 8px; }
  .media-result img { width: 128px; aspect-ratio: 16/9; object-fit: cover; border-radius: 6px; background: #000; }
  .media-result button { margin-top: 6px; }
  .filter-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 6px; margin-top: 8px; }
  .timeline-list { display: grid; gap: 8px; margin-top: 10px; }
  .timeline-card, .track-card { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: var(--panel-raised); }
  .timeline-card.locked { border-color: #6b5324; }
  .timeline-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 8px; }
  .timeline-controls > button { flex: 1 1 0; min-width: 0; }
  .clip-more { position: relative; flex: 0 0 auto; }
  .clip-more > summary { list-style: none; cursor: pointer; padding: 6px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-raised); user-select: none; }
  .clip-more > summary::-webkit-details-marker { display: none; }
  .clip-more[open] > summary { border-color: var(--accent, #4f7cff); }
  .clip-more-menu { position: absolute; right: 0; top: calc(100% + 4px); z-index: 20; display: grid; gap: 4px; min-width: 190px; padding: 6px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-raised); box-shadow: 0 6px 18px rgba(0,0,0,0.35); }
  .clip-more-menu button { width: 100%; text-align: left; }
  .clip-more { position: relative; }
  .clip-more > summary { list-style: none; cursor: pointer; text-align: center; border: 1px solid var(--line); border-radius: 6px; padding: 5px 8px; background: var(--panel); color: var(--text-dim); }
  .clip-more > summary::-webkit-details-marker { display: none; }
  .clip-more[open] > summary { background: #283755; color: #fff; }
  .clip-more-menu { position: absolute; right: 0; z-index: 30; margin-top: 4px; display: grid; gap: 4px; padding: 6px; min-width: 168px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel-raised); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
  .clip-more-menu button { width: 100%; }
  .step-focus { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 8px; }
  .continuity-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 8px; margin-top: 10px; }
  .gate { border: 1px dashed #6b5324; background: #1c1706; border-radius: 8px; padding: 10px; }
  .ok { color: #57d98a; } .warn { color: #f5c451; } .err { color: #ff7a86; }
  .notice { font-size: 12px; color: #9aa3b2; margin-top: 6px; }
  code { background: #0f1115; padding: 1px 5px; border-radius: 4px; }
  .sr-only { position: absolute !important; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }

  /* Media Explorer: focused, clip-first editing workspace. */
  :root { --panel: #12161f; --panel-raised: #1b202c; --line: #303746; --text-dim: #aab3c3; }
  :focus-visible { outline: 3px solid #8eabff; outline-offset: 2px; }
  .media-shell { padding: 0; overflow: clip; }
  .media-topbar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; border-bottom: 1px solid var(--line); }
  .media-title h2 { margin: 0; font-size: 17px; }
  .media-title p { margin: 2px 0 0; color: var(--text-dim); font-size: 12px; }
  .export-tools { display: flex; gap: 6px; align-items: center; }
  .export-tools button { padding: 6px 10px; font-size: 12px; }
  .media-workspace { display: grid; grid-template-columns: minmax(320px, 1.12fr) minmax(340px, .88fr); min-height: 620px; }
  .player-pane { min-width: 0; padding: 16px; background: #0b0d12; border-right: 1px solid var(--line); }
  .sticky-player { position: sticky; top: 12px; }
  .source-frame { overflow: hidden; border: 1px solid #272d39; border-radius: 12px; background: #000; box-shadow: 0 16px 44px rgba(0,0,0,.28); }
  .source-frame video { display: block; margin: 0; border-radius: 0; aspect-ratio: 16/9; }
  .player-hint { display: flex; justify-content: space-between; gap: 10px; margin-top: 8px; color: var(--text-dim); font-size: 11px; }
  .highlight-section { margin-top: 18px; }
  .section-heading { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
  .section-heading h3 { margin: 0; font-size: 13px; }
  .highlight-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .highlight-card { display: grid; gap: 8px; padding: 11px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel-raised); }
  .highlight-card strong { font-size: 13px; }
  .highlight-card p { margin: 0; color: var(--text-dim); font-size: 11px; }
  .highlight-card .row button { flex: 1; padding: 6px; font-size: 11px; }
  .browser-pane { display: flex; min-width: 0; flex-direction: column; background: var(--panel); }
  .media-searchbar { padding: 14px; border-bottom: 1px solid var(--line); }
  .search-field { display: flex; gap: 7px; }
  .search-field input { min-width: 0; }
  .media-tabs { display: flex; gap: 4px; margin-top: 10px; padding: 3px; border-radius: 9px; background: #0d1017; }
  .media-tab { flex: 1; padding: 6px 8px; border: 0; background: transparent; color: var(--text-dim); font-size: 12px; }
  .media-tab[aria-selected="true"] { color: #fff; background: #2a3242; box-shadow: 0 1px 3px rgba(0,0,0,.3); }
  .media-list { display: grid; align-content: start; gap: 6px; padding: 10px; flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: contain; scroll-behavior: smooth; }
  .media-result { position: relative; display: grid; grid-template-columns: 116px minmax(0, 1fr); gap: 10px; align-items: start; border: 1px solid transparent; border-radius: 10px; padding: 9px; background: transparent; transition: border-color .15s, background .15s, transform .15s; }
  .media-result:hover { border-color: var(--line); background: #191e29; }
  .media-result.transcript-result { grid-template-columns: 1fr; cursor: pointer; }
  .media-result.active { border-color: var(--accent); background: #1c2740; }
  .media-result.active::before { content: ""; position: absolute; inset: 10px auto 10px 0; width: 3px; border-radius: 3px; background: var(--accent); }
  .media-result img { width: 116px; aspect-ratio: 16/9; object-fit: cover; border-radius: 7px; background: #08090c; }
  .result-meta { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px; }
  .time-chip, .speaker-chip, .kind-chip { display: inline-flex; align-items: center; min-height: 22px; padding: 2px 7px; border-radius: 999px; font-size: 10px; font-weight: 650; letter-spacing: .02em; }
  .time-chip { color: #bdd0ff; background: #213153; }
  .speaker-chip { color: #d6c9ff; background: #352958; }
  .kind-chip { color: #b9c2d0; background: #2b303a; }
  .result-copy { color: #e9ecf2; font-size: 13px; line-height: 1.45; }
  .seek-button { margin-top: 7px; padding: 5px 8px; color: #c9d7ff; border-color: #3b4f7c; background: #1c2945; font-size: 11px; }
  .state-panel { margin: 12px; padding: 26px 18px; text-align: center; border: 1px dashed var(--line); border-radius: 10px; color: var(--text-dim); }
  .state-panel[hidden] { display: none; }
  .chat-box { margin-top: 18px; padding-top: 14px; border-top: 1px solid var(--line); }
  .chat-answer { min-height: 22px; margin-top: 7px; padding: 8px 10px; border-radius: 8px; background: #121722; }
  @media (max-width: 1100px) { .media-workspace { grid-template-columns: 1fr; } .player-pane { border-right: 0; border-bottom: 1px solid var(--line); } .sticky-player { position: static; } .media-list { max-height: 560px; } }
  @media (max-width: 620px) { .media-topbar { align-items: flex-start; flex-direction: column; } .media-workspace { min-height: 0; } .player-pane { padding: 10px; } .highlight-grid { grid-template-columns: 1fr; } .media-result { grid-template-columns: 96px minmax(0, 1fr); } .media-result img { width: 96px; } .search-field { flex-wrap: wrap; } .search-field input { flex-basis: 100%; } }
  /* Compact project workspace: one task visible at a time. */
  body { background: #10141c; font-size: 14px; }
  header { padding: 11px 20px; }
  aside, main, #detail { min-width: 0; }
  aside .card { margin-bottom: 10px; }
  summary { cursor: pointer; font-weight: 650; }
  .create-panel > summary, .library-panel > summary { list-style-position: inside; }
  .create-panel form, .library-panel > .search-field { margin-top: 10px; }
  .advanced-fields { margin-top: 10px; padding: 9px; border: 1px solid var(--line); border-radius: 8px; }
  .project-list { max-height: min(48dvh, 450px); overflow: auto; overscroll-behavior: contain; }
  .project-card { padding: 8px; }
  .project-card .card-actions { margin-top: 3px; }
  .project-card progress { height: 6px; }
  .workspace-tabs { display: flex; gap: 5px; overflow-x: auto; margin-bottom: 10px; padding: 4px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel); }
  .workspace-tabs button { flex: 1; white-space: nowrap; min-width: max-content; background: transparent; border: 0; color: var(--text-dim); }
  .workspace-tabs button[aria-selected="true"] { background: #283755; color: #fff; }
  .workspace-view[hidden], [hidden] { display: none !important; }
  .next-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
  .next-row .action-note { flex: 1; min-width: 180px; margin-top: 8px; }
  .stages-panel { margin-top: 10px; color: var(--text-dim); }
  .stages-panel #stages { margin-top: 8px; }
  .extras-panel { margin-top: 12px; }
  .extras-panel .highlight-section { margin-top: 10px; }
  .media-workspace { min-height: 0; }
  .media-topbar { padding: 9px 12px; }
  .player-pane { padding: 10px; }
  .source-frame video { max-height: 52dvh; }
  @media (max-width: 1100px) { .media-list { max-height: min(62dvh, 650px); } }
  #view-review-form { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 320px), 1fr)); gap: 10px; align-items: start; }
  #view-review-form .card { min-width: 0; }
  @media (max-width: 820px) {
    .project-list { max-height: 210px; }
    .media-list { max-height: 380px; }
    .source-frame video { max-height: 32dvh; }
    .workspace-tabs button { padding: 8px; }
  }
  @media (max-width: 430px) {
    .card { padding: 10px; margin-bottom: 8px; }
    .project-list { max-height: 180px; }
    .stage { flex-wrap: wrap; }
    .stage .name { width: auto; }
    .next-row > button, .next-row > a { flex: 1; text-align: center; }
  }
  @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; } }
</style>
<style>
  /* ============================================================
     Visual upgrade — modern token-based design system.
     Layered as an override after the base sheet, so every existing
     class/id keeps working; only the visual treatment changes. Adds a
     real light theme selectable via [data-theme] (see header toggle).
     ============================================================ */
  :root{
    color-scheme: dark;
    --space-1:4px; --space-2:8px; --space-3:12px; --space-4:16px; --space-6:24px; --space-8:32px;
    --r-sm:8px; --r-md:12px; --r-lg:16px; --r-pill:999px;
    --bg-base:#0a0c11; --panel:#14171f; --panel-raised:#1b1f29; --field-bg:#0e1016;
    --line:#242a36; --line-strong:#333a48;
    --text:#e7e9f0; --text-dim:#9aa3b4; --text-muted:#6c7484;
    --accent:#338ef7; --accent-hover:#57a3f9; --accent-contrast:#ffffff;
    --accent-soft:rgba(51,142,247,.16); --ring:rgba(51,142,247,.5);
    --ok:#34d399; --ok-soft:rgba(16,185,129,.16); --ok-line:rgba(16,185,129,.34);
    --warn:#fbbf24; --warn-soft:rgba(245,158,11,.15); --warn-line:rgba(245,158,11,.32);
    --err:#fb7185; --err-soft:rgba(244,63,94,.15); --err-line:rgba(244,63,94,.34);
    --shadow-1:0 1px 2px rgba(0,0,0,.4); --shadow-2:0 8px 24px rgba(0,0,0,.36); --shadow-3:0 20px 50px rgba(0,0,0,.5);
    --z-toolbar:20;
    --gap:16px;
  }
  :root[data-theme="light"]{
    color-scheme: light;
    --bg-base:#f5f7fb; --panel:#ffffff; --panel-raised:#ffffff; --field-bg:#f1f4f9;
    --line:#e4e8f0; --line-strong:#cdd5e1;
    --text:#0f172a; --text-dim:#526078; --text-muted:#8a94a6;
    --accent:#1f7ae0; --accent-hover:#166bce; --accent-contrast:#ffffff;
    --accent-soft:rgba(51,142,247,.12); --ring:rgba(51,142,247,.35);
    --ok:#059669; --ok-soft:rgba(5,150,105,.12); --ok-line:rgba(5,150,105,.28);
    --warn:#b45309; --warn-soft:rgba(180,83,9,.12); --warn-line:rgba(180,83,9,.28);
    --err:#e11d48; --err-soft:rgba(225,29,72,.10); --err-line:rgba(225,29,72,.26);
    --shadow-1:0 1px 2px rgba(16,24,40,.06); --shadow-2:0 8px 24px rgba(16,24,40,.10); --shadow-3:0 24px 48px rgba(16,24,40,.16);
  }
  body{ background:var(--bg-base); color:var(--text); font-size:14px; -webkit-font-smoothing:antialiased;
        background-image:radial-gradient(1100px 520px at 100% -8%, var(--accent-soft), transparent 62%); background-attachment:fixed; }
  header{ position:sticky; top:0; z-index:30; display:flex; align-items:center; justify-content:space-between; gap:var(--space-4);
          padding:12px 22px; border-bottom:1px solid var(--line);
          background:var(--panel); background:color-mix(in srgb, var(--panel) 85%, transparent);
          -webkit-backdrop-filter:saturate(1.2) blur(10px); backdrop-filter:saturate(1.2) blur(10px); }
  header h1{ font-size:16px; letter-spacing:.01em; }
  header .sub{ color:var(--text-dim); font-size:12px; }
  .brand{ display:flex; align-items:center; gap:12px; min-width:0; }
  .brand-mark{ display:grid; place-items:center; width:38px; height:38px; border-radius:11px; font-size:19px; flex:none;
               background:linear-gradient(135deg, var(--accent), var(--accent-hover)); box-shadow:0 6px 18px var(--accent-soft); }
  .brand-text{ min-width:0; }
  .theme-toggle{ display:inline-flex; align-items:center; gap:8px; flex:none; padding:7px 13px; border-radius:var(--r-pill);
                 background:var(--panel-raised); border:1px solid var(--line-strong); color:var(--text); font-size:12.5px; font-weight:650; }
  .theme-toggle:hover{ border-color:var(--accent); color:var(--accent); }
  .theme-toggle-icon{ font-size:14px; line-height:1; }
  .card{ background:var(--panel); border:1px solid var(--line); border-radius:var(--r-md); padding:var(--space-4); box-shadow:var(--shadow-1); }
  aside .card{ background:var(--panel); }
  .card h2, .media-title h2{ letter-spacing:.01em; }
  .muted, .notice{ color:var(--text-dim); }
  .advanced-fields{ background:var(--field-bg); border-color:var(--line); border-radius:var(--r-sm); }
  input, select, textarea{ background:var(--field-bg); color:var(--text); border:1px solid var(--line-strong); border-radius:var(--r-sm);
                           padding:8px 10px; transition:border-color .15s, box-shadow .15s; }
  input::placeholder, textarea::placeholder{ color:var(--text-muted); }
  input:focus, select:focus, textarea:focus{ outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--ring); }
  /* Clipto-style: pill-shaped buttons, one solid accent (no gradient). */
  button{ background:var(--panel-raised); color:var(--text); border:1px solid var(--line-strong); border-radius:var(--r-pill);
          padding:8px 15px; font-weight:650; letter-spacing:.01em;
          transition:background .15s, border-color .15s, transform .06s, box-shadow .15s, filter .15s; }
  button:hover{ border-color:var(--accent); color:var(--accent); }
  button:active{ transform:translateY(1px); }
  button.primary, button.primary:hover{ background:var(--accent); border-color:transparent;
                                        color:var(--accent-contrast); box-shadow:0 6px 18px var(--accent-soft); }
  button.primary:hover{ filter:brightness(1.06); }
  .button-link{ background:var(--accent); color:var(--accent-contrast); border-radius:var(--r-pill); box-shadow:0 6px 18px var(--accent-soft); }
  button.danger, .danger{ color:var(--err); border-color:var(--err-line); background:var(--err-soft); }
  button.danger:hover{ border-color:var(--err); }
  button:disabled{ opacity:.45; }
  :focus-visible{ outline:2px solid var(--accent); outline-offset:2px; }
  .badge{ display:inline-flex; align-items:center; gap:6px; font-weight:650; border:1px solid transparent; font-variant-numeric:tabular-nums; }
  .badge::before{ content:""; width:6px; height:6px; border-radius:50%; background:currentColor; flex:none; }
  .badge.pending{ background:var(--panel-raised); color:var(--text-dim); border-color:var(--line); }
  .badge.running{ background:var(--warn-soft); color:var(--warn); border-color:var(--warn-line); }
  .badge.ready{ background:var(--ok-soft); color:var(--ok); border-color:var(--ok-line); }
  .badge.failed{ background:var(--err-soft); color:var(--err); border-color:var(--err-line); }
  .badge.skipped{ background:var(--panel-raised); color:var(--text-muted); border-color:var(--line); }
  .badge.running::before{ animation:mrfpulse 1.2s ease-in-out infinite; }
  @keyframes mrfpulse{ 0%,100%{opacity:1} 50%{opacity:.3} }
  .progress{ background:var(--field-bg); border-color:var(--line); }
  .progress > div{ background:linear-gradient(90deg, var(--accent), #7c83f6); }
  progress.upload-progress, .project-card progress{ accent-color:var(--accent); }
  .project-card{ background:var(--panel); border:1px solid var(--line); border-radius:var(--r-sm);
                 transition:border-color .15s, background .15s, transform .12s; }
  .project-card:hover{ border-color:var(--line-strong); }
  .project-card.active{ border-color:var(--accent); background:var(--accent-soft); }
  .project-card .open-project{ color:var(--text); }
  .project-card .open-project:hover{ color:var(--accent); }
  .stage{ border-bottom-color:var(--line); }
  .stage .name{ font-variant-numeric:tabular-nums; }
  .workspace-tabs, .media-tabs{ background:var(--field-bg); border:1px solid var(--line); border-radius:var(--r-pill); }
  .workspace-tabs button, .media-tab{ border-radius:var(--r-pill); font-weight:650; transition:background .15s, color .15s; }
  .workspace-tabs button:hover, .media-tab:hover{ color:var(--accent); }
  .workspace-tabs button[aria-selected="true"], .media-tab[aria-selected="true"]{ background:var(--accent); color:var(--accent-contrast); box-shadow:var(--shadow-1); }
  .media-shell, .browser-pane{ background:var(--panel); }
  .player-pane{ background:var(--bg-base); border-right-color:var(--line); }
  .media-topbar, .media-searchbar, .chat-box{ border-color:var(--line); }
  .media-title p, .player-hint, .highlight-card p{ color:var(--text-dim); }
  /* Clipto-style Media Explorer layout: softer/rounder shell, centered titles
     with more breathing room. CSS-only; dark theme and markup unchanged. */
  #mediaExplorerCard.media-shell{ border-radius:var(--r-lg); }
  #mediaExplorerCard .media-topbar{ flex-direction:column; align-items:center; text-align:center; gap:10px; padding:22px 20px; }
  #mediaExplorerCard .media-title{ max-width:640px; }
  #mediaExplorerCard .media-title h2{ font-size:20px; }
  #mediaExplorerCard .media-title p{ margin-top:6px; }
  #mediaExplorerCard .export-tools{ justify-content:center; }
  /* Centered, stacked section headings (title over subtitle) instead of the
     left title / right subtitle split. */
  #mediaExplorerCard .section-heading{ flex-direction:column; align-items:center; justify-content:center; text-align:center; gap:2px; margin-bottom:12px; }
  /* Extend the Clipto treatment to the other workspace areas (Biên tập,
     Duyệt & xuất) and the create-project panel: larger card radius + centered
     section titles. CSS-only, dark theme unchanged. */
  #view-explore .card, #view-edit .card, #view-review .card,
  #view-review-content .card, #view-review-form .card{ border-radius:var(--r-lg); }
  /* Center only direct card headings; headings that sit in a flex .row with
     action buttons (e.g. #editorCard, #detail job title) are .card > .row > h2
     and are intentionally left untouched. */
  #view-edit .card > h2, #view-review .card > h2,
  #view-review-content .card > h2, #view-review-form .card > h2{ text-align:center; }
  /* Give the create-project dropdown a softer, rounder frame and a centered
     panel title to match. */
  .create-panel > form{ border-radius:var(--r-lg); }
  .create-panel > summary{ justify-content:center; text-align:center; }
  .result-copy{ color:var(--text); }
  .media-result:hover{ border-color:var(--line-strong); background:var(--panel-raised); }
  .media-result.active{ border-color:var(--accent); background:var(--accent-soft); }
  .media-result.active::before{ background:var(--accent); }
  .highlight-card, .timeline-card, .track-card{ background:var(--panel); border-color:var(--line); border-radius:var(--r-sm); }
  .timeline-card.locked{ border-color:var(--warn-line); background:var(--warn-soft); }
  .time-chip{ color:var(--accent); background:var(--accent-soft); }
  .speaker-chip{ color:#c4b5fd; background:rgba(139,92,246,.16); }
  .kind-chip{ color:var(--text-dim); background:var(--panel-raised); }
  .seek-button{ color:var(--accent); background:var(--accent-soft); border-color:var(--line-strong); border-radius:var(--r-pill); }
  /* Overflow ("⋯ Thêm") trigger matches the pill button family; its menu items
     stay left-aligned but rounded to fit inside the popover. */
  .clip-more > summary{ border-radius:var(--r-pill); background:var(--panel-raised); color:var(--text); border-color:var(--line-strong); }
  .clip-more[open] > summary{ background:var(--accent); color:var(--accent-contrast); border-color:transparent; }
  .clip-more-menu button{ border-radius:var(--r-sm); text-align:left; }
  .button-link:hover{ filter:brightness(1.06); }
  .source-frame{ border-color:var(--line); box-shadow:var(--shadow-2); }
  .state-panel{ background:var(--panel); border-color:var(--line); color:var(--text-dim); }
  .chat-answer{ background:var(--field-bg); }
  .thumb-item{ background:var(--panel); border-color:var(--line); border-radius:var(--r-sm); }
  .thumb-item.selected{ border-color:var(--accent); background:var(--accent-soft); }
  .artifact-card{ background:var(--panel); }
  .artifact-card a, .arts a{ color:var(--accent); }
  .action-note{ background:var(--field-bg); border-color:var(--line); border-radius:var(--r-sm); color:var(--text-dim); }
  .gate{ background:var(--warn-soft); border-color:var(--warn-line); }
  .ok{ color:var(--ok); } .warn{ color:var(--warn); } .err{ color:var(--err); }
  code{ background:var(--field-bg); border:1px solid var(--line); }
  dialog{ background:var(--panel); color:var(--text); border:1px solid var(--line); border-radius:var(--r-lg); box-shadow:var(--shadow-3); }
  dialog::backdrop{ background:rgba(6,8,12,.6); -webkit-backdrop-filter:blur(4px); backdrop-filter:blur(4px); }
  *{ scrollbar-width:thin; scrollbar-color:var(--line-strong) transparent; }
  ::-webkit-scrollbar{ width:10px; height:10px; }
  ::-webkit-scrollbar-thumb{ background:var(--line-strong); border-radius:999px; border:2px solid transparent; background-clip:padding-box; }
  ::-webkit-scrollbar-thumb:hover{ background:var(--text-muted); }
  @media (max-width:560px){ header{ flex-wrap:wrap; padding:10px 14px; } .theme-toggle-label{ display:none; } }
</style>
<style>
  /* Workspace chrome: .dashboard-shell owns top-level layout; .layout only disables the legacy grid. */
  .layout{ display:block; }
  .dashboard-shell{ width:min(1800px,100%); margin:0 auto; padding:12px 18px 18px; }
  .top-toolbar{ position:relative; z-index:var(--z-toolbar); display:flex; align-items:center; gap:8px; min-width:0; margin:0 0 12px; }
  .toolbar-action{ min-height:42px; flex:0 0 auto; }
  .top-toolbar .project-panel{ position:relative; flex:0 1 auto; min-width:0; margin:0; padding:0; border-radius:var(--r-pill); }
  .project-panel > summary{ list-style:none; display:flex; align-items:center; gap:8px; min-height:42px; padding:8px 14px; white-space:nowrap; }
  .project-panel > summary::-webkit-details-marker{ display:none; }
  .project-panel > summary::before{ content:"▾"; color:var(--text-muted); font-size:11px; transition:transform .15s; }
  .project-panel[open] > summary::before{ transform:rotate(180deg); }
  .project-summary-prefix{ color:var(--text-dim); font-size:12px; font-weight:650; }
  .project-label{ max-width:360px; overflow:hidden; text-overflow:ellipsis; color:var(--text); }
  .project-popover{ position:absolute; top:calc(100% + 8px); left:0; z-index:80; width:min(560px,calc(100vw - 36px)); max-height:min(68dvh,560px); overflow:hidden; padding:12px; background:var(--panel); border:1px solid var(--line-strong); border-radius:var(--r-md); box-shadow:var(--shadow-3); }
  .project-popover .project-list{ display:grid; grid-template-columns:1fr; max-height:min(52dvh,440px); overflow-y:auto; overflow-x:hidden; padding:0 4px 4px 0; scroll-snap-type:none; }
  .project-popover .project-card{ width:100%; min-width:0; }

  .tool-dialog{ width:min(760px,calc(100vw - 32px)); max-width:none; max-height:min(88dvh,860px); padding:0; overflow:hidden; }
  .tool-dialog-wide{ width:min(1040px,calc(100vw - 32px)); }
  .tool-dialog::backdrop{ background:rgba(6,8,12,.68); -webkit-backdrop-filter:blur(4px); backdrop-filter:blur(4px); }
  .dialog-shell{ display:grid; grid-template-rows:auto minmax(0,1fr); max-height:min(88dvh,860px); }
  .dialog-header{ display:flex; align-items:center; justify-content:space-between; gap:16px; padding:14px 18px; border-bottom:1px solid var(--line); background:var(--panel); }
  .dialog-header h2{ margin:0; font-size:16px; }
  .dialog-header p{ margin:2px 0 0; color:var(--text-dim); font-size:12px; }
  .dialog-close{ display:grid; place-items:center; width:36px; height:36px; padding:0; flex:none; font-size:20px; line-height:1; }
  .dialog-body{ min-height:0; overflow:auto; padding:18px; }
  .create-form{ max-width:680px; margin:0 auto; }
  .create-form > label:first-of-type{ margin-top:0; }
  .brand-fields{ display:grid; gap:16px; }
  .brand-preview{ display:flex; align-items:center; gap:12px; margin:0; padding:12px; border:1px solid var(--line); border-radius:var(--r-md); background:var(--field-bg); }
  .brand-preview img{ width:56px; height:56px; object-fit:contain; background:var(--bg-base); border-radius:var(--r-md); }
  .settings-section{ display:grid; gap:10px; padding:14px; border:1px solid var(--line); border-radius:var(--r-md); background:var(--panel); }
  .settings-section h3, .library-section h3{ margin:0; font-size:14px; }
  .settings-section > p, .library-section > p{ margin:0; color:var(--text-dim); font-size:12px; }
  .settings-row{ display:grid; grid-template-columns:minmax(0,1fr) auto; gap:8px; align-items:end; }
  .settings-row label{ margin:0 0 2px; }
  .settings-row button{ white-space:nowrap; }
  /* A row of two equal inputs (e.g. the two mask-band fields) needs a balanced
     two-column grid; the default input+button template squeezes the second
     field and misaligns the pair. */
  .settings-row.band-row{ grid-template-columns:1fr 1fr; align-items:start; }
  .settings-row.band-row input{ width:100%; }
  .library-hub{ display:grid; grid-template-columns:minmax(0,1.15fr) minmax(300px,.85fr); gap:16px; align-items:start; }
  .library-section{ min-width:0; padding:14px; border:1px solid var(--line); border-radius:var(--r-md); background:var(--panel); }
  .library-section > .search-field{ margin-top:12px; }
  .library-section .project-list{ max-height:340px; overflow:auto; }
  .library-section details{ margin-top:10px; }

  .empty-state{ min-height:min(420px,calc(100dvh - 220px)); margin:0; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; padding:32px; text-align:center; border-style:dashed; }
  .empty-state-mark{ display:grid; place-items:center; width:52px; height:52px; border-radius:var(--r-lg); background:var(--accent-soft); color:var(--accent); font-size:28px; font-weight:750; }
  .empty-state h2{ margin:0; color:var(--text); font-size:20px; }
  .empty-state p{ max-width:620px; margin:0; color:var(--text-dim); }
  .empty-actions{ display:flex; flex-wrap:wrap; justify-content:center; gap:8px; margin-top:4px; }
  .empty-help{ font-size:12px; color:var(--text-muted); }
  .cta-status{ margin:8px 0; padding:9px 11px; border:1px solid var(--line); border-radius:9px; background:var(--field-bg); font-weight:650; }

  /* Review/export player stays constrained to the viewport. */
  #video{ display:block; width:100%; max-width:1040px; height:auto; aspect-ratio:16/9; max-height:calc(100dvh - 260px); object-fit:contain; margin:10px auto 0; background:#000; border-radius:10px; }

  @media (max-width:860px){
    .dashboard-shell{ padding:8px; }
    .top-toolbar{ flex-wrap:wrap; overflow:visible; padding-bottom:0; }
    .project-label{ max-width:220px; }
    .project-popover{ position:absolute; top:calc(100% + 8px); left:0; right:auto; bottom:auto; width:min(560px,calc(100vw - 16px)); max-height:min(60dvh,480px); }
    .tool-dialog, .tool-dialog-wide{ width:calc(100vw - 16px); max-height:92dvh; }
    .dialog-shell{ max-height:92dvh; }
    .dialog-header{ padding:12px 14px; }
    .dialog-body{ padding:14px; }
    .library-hub{ grid-template-columns:1fr; }
    .settings-row{ grid-template-columns:1fr; }
    .empty-state{ min-height:320px; padding:22px 16px; }
    #video{ max-height:56dvh; }
  }
</style>
<style>
  /* Noir & champagne skin: token overrides plus bounded loading shimmer; markup and ids unchanged. */
/*@ui-fonts*/
  :root{
    --bg-base:#0b0a08; --panel:#15130f; --panel-raised:#1d1a15; --field-bg:#100e0b;
    --line:#2a251d; --line-strong:#3a3328;
    --text:#f2ede4; --text-dim:#b8ae9c; --text-muted:#948873;
    --accent:#c9a961; --accent-hover:#e6c687; --accent-contrast:#1a1407;
    --accent-soft:rgba(201,169,97,.13); --ring:rgba(201,169,97,.42);
    --gold-deep:#8c6a2f; --gold-light:#f3dfa8;
    --gold-grad:linear-gradient(135deg, #8c6a2f 0%, #c9a961 34%, #f3dfa8 52%, #c9a961 70%, #9a7735 100%);
    --sheen:rgba(243,223,168,.07);
    --shadow-1:0 1px 2px rgba(0,0,0,.5), inset 0 1px 0 var(--sheen);
    --shadow-2:0 10px 30px rgba(0,0,0,.45);
    --shadow-3:0 28px 70px rgba(0,0,0,.6);
    --font-body:'Montserrat', system-ui, "Segoe UI", Roboto, sans-serif;
    --font-display:'Cormorant', Georgia, "Times New Roman", serif;
    --ease-out:cubic-bezier(.22,1,.36,1);
  }
  :root[data-theme="light"]{
    --bg-base:#faf8f3; --panel:#ffffff; --panel-raised:#ffffff; --field-bg:#f5f1e8;
    --line:#e8e1d3; --line-strong:#d6ccb8;
    --text:#1c1917; --text-dim:#57534e; --text-muted:#78716c;
    --accent:#8a6516; --accent-hover:#74540f; --accent-contrast:#ffffff;
    --accent-soft:rgba(138,101,22,.10); --ring:rgba(138,101,22,.32);
    --gold-grad:linear-gradient(135deg, #74540f 0%, #a07a28 40%, #c9a961 55%, #8a6516 100%);
    --sheen:rgba(255,255,255,.7);
    --shadow-1:0 1px 2px rgba(60,45,20,.08);
    --shadow-2:0 10px 28px rgba(60,45,20,.12);
    --shadow-3:0 28px 60px rgba(60,45,20,.18);
  }
  body{ font-family:var(--font-body); letter-spacing:.005em;
        background-image:radial-gradient(900px 420px at 50% -12%, rgba(201,169,97,.10), transparent 70%),
                         radial-gradient(700px 500px at 100% 0%, rgba(140,106,47,.08), transparent 65%); }
  h1, h2, .empty-state h2, .media-title h2, dialog h2{ font-family:var(--font-display); font-weight:650; letter-spacing:.015em; font-size-adjust:ex-height .5; }
  header{ box-shadow:0 1px 0 rgba(201,169,97,.16); }
  header h1{ font-size:21px; background:var(--gold-grad); -webkit-background-clip:text; background-clip:text; color:transparent; }
  header .sub{ letter-spacing:.06em; text-transform:uppercase; font-size:10.5px; }
  .brand-mark{ background:var(--gold-grad); color:var(--accent-contrast); box-shadow:0 6px 20px rgba(201,169,97,.22), inset 0 1px 0 rgba(255,255,255,.35); }
  :root[data-theme="light"] .brand-mark{ color:#ffffff; }
  .theme-toggle-icon{ display:inline-grid; }
  .theme-toggle-icon .icon-sun, :root[data-theme="light"] .theme-toggle-icon .icon-moon{ display:none; }
  :root[data-theme="light"] .theme-toggle-icon .icon-sun{ display:block; }
  .card, .library-section, .project-card, .highlight-card, .timeline-card, .track-card, .thumb-item{
    background-image:linear-gradient(180deg, rgba(243,223,168,.025), transparent 40%); }
  .card{ box-shadow:var(--shadow-1); transition:border-color .2s var(--ease-out), box-shadow .2s var(--ease-out); }
  .project-card:hover, .media-result:hover{ border-color:rgba(201,169,97,.35); }
  button.primary, button.primary:hover, .button-link, .workspace-tabs button[aria-selected="true"], .media-tab[aria-selected="true"], .clip-more[open] > summary{
    background:var(--gold-grad); color:var(--accent-contrast); border-color:transparent; }
  button.primary, .button-link{ position:relative; overflow:hidden; box-shadow:0 8px 22px rgba(201,169,97,.20), inset 0 1px 0 rgba(255,255,255,.3); }
  :root[data-theme="light"] button.primary, :root[data-theme="light"] .button-link,
  :root[data-theme="light"] .workspace-tabs button[aria-selected="true"], :root[data-theme="light"] .media-tab[aria-selected="true"]{ color:#ffffff; }
  /* One-shot hover glint, never looping. */
  button.primary::after, .button-link::after{ content:""; position:absolute; inset:0; pointer-events:none;
    background:linear-gradient(105deg, transparent 35%, rgba(255,255,255,.45) 50%, transparent 65%);
    transform:translateX(-120%); transition:transform .7s var(--ease-out); }
  button.primary:hover::after, .button-link:hover::after{ transform:translateX(120%); }
  button:hover{ border-color:rgba(201,169,97,.55); }
  .progress > div{ background:linear-gradient(90deg, var(--gold-deep), var(--accent), var(--gold-light)); }
  .badge.running{ background:var(--accent-soft); color:var(--accent); border-color:rgba(201,169,97,.35); }
  .speaker-chip{ color:var(--gold-light); background:rgba(201,169,97,.12); }
  :root[data-theme="light"] .speaker-chip{ color:var(--accent); }
  dialog{ background-image:linear-gradient(180deg, rgba(243,223,168,.03), transparent 30%); }
  dialog::backdrop{ background:rgba(8,6,3,.66); }
  ::selection{ background:rgba(201,169,97,.32); }
  /* Loading shimmer: animates only while a request is slow or a panel is in its loading state. */
  #mrfBusyBar{ position:fixed; inset:0 0 auto 0; height:2px; z-index:1000; pointer-events:none; opacity:0;
    background:linear-gradient(90deg, transparent 0%, var(--gold-deep) 30%, var(--gold-light) 50%, var(--accent) 70%, transparent 100%) 0 0 / 40% 100% no-repeat;
    transition:opacity .25s var(--ease-out); }
  :root[data-busy="1"] #mrfBusyBar{ opacity:1; animation:mrf-busy 1.25s cubic-bezier(.45,0,.55,1) infinite; box-shadow:0 0 10px rgba(243,223,168,.45); }
  @keyframes mrf-busy{ from{ background-position:-60% 0; } to{ background-position:160% 0; } }
  .mrf-loading{ display:inline-block; }
  #scoutLoading:not([hidden]), #mediaState[data-kind="loading"], .mrf-loading{
    background-image:linear-gradient(100deg, var(--text-dim) 40%, var(--gold-light) 50%, var(--text-dim) 60%); background-size:250% 100%;
    -webkit-background-clip:text; background-clip:text; color:transparent; animation:mrf-sheen 1.8s linear infinite; }
  :root[data-theme="light"] #scoutLoading:not([hidden]), :root[data-theme="light"] #mediaState[data-kind="loading"], :root[data-theme="light"] .mrf-loading{
    background-image:linear-gradient(100deg, var(--text-dim) 40%, var(--accent) 50%, var(--text-dim) 60%); }
  @keyframes mrf-sheen{ from{ background-position:100% 0; } to{ background-position:-150% 0; } }
  #mediaResults[aria-busy="true"]:empty{ min-height:180px; border-radius:var(--r-md);
    background:linear-gradient(100deg, transparent 30%, rgba(243,223,168,.09) 50%, transparent 70%) 0 0 / 250% 100%,
               repeating-linear-gradient(180deg, var(--panel-raised) 0 44px, transparent 44px 56px);
    animation:mrf-sheen 1.6s linear infinite; }
  @media (prefers-reduced-motion:reduce){
    :root[data-busy="1"] #mrfBusyBar, #scoutLoading:not([hidden]), #mediaState[data-kind="loading"], .mrf-loading, #mediaResults[aria-busy="true"]:empty{ animation:none; }
    :root[data-busy="1"] #mrfBusyBar{ background-size:100% 100%; }
    button.primary::after, .button-link::after{ display:none; }
  }
</style>
<script>
  (function(){
    try{
      var t = localStorage.getItem('mrf-theme');
      if(!t){ t = (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) ? 'light' : 'dark'; }
      document.documentElement.setAttribute('data-theme', t);
    }catch(e){ document.documentElement.setAttribute('data-theme','dark'); }
  })();
</script>
</head>
<body>
<div id="mrfBusyBar" aria-hidden="true"></div>
<header>
  <div class="brand">
    <span class="brand-mark" aria-hidden="true"><svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="9" width="18" height="11" rx="2"/><path d="M3 9l2.5-5 15 2.5L18 9"/><path d="M8.5 4.6L7 9M13.5 5.4L12 9"/><path d="M10.5 12.5v4.5l3.8-2.25z" fill="currentColor" stroke="none"/></svg></span>
    <div class="brand-text">
      <h1>Xưởng Review Phim</h1>
      <div class="sub">Import → khám phá → biên tập → duyệt và xuất</div>
    </div>
  </div>
  <button id="themeToggle" class="theme-toggle" type="button" title="Đổi giao diện sáng/tối" aria-label="Đổi giao diện sáng/tối" aria-pressed="false">
    <span class="theme-toggle-icon" aria-hidden="true"><svg class="icon-moon" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/></svg><svg class="icon-sun" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></span>
    <span class="theme-toggle-label">Tối</span>
  </button>
</header>
<script>
  (function(){
    var btn = document.getElementById('themeToggle');
    if(!btn){ return; }
    function sync(){
      var cur = document.documentElement.getAttribute('data-theme') || 'dark';
      var label = btn.querySelector('.theme-toggle-label');
      if(label){ label.textContent = cur === 'light' ? 'Sáng' : 'Tối'; }
      btn.setAttribute('aria-pressed', cur === 'light' ? 'true' : 'false');
    }
    sync();
    btn.addEventListener('click', function(){
      var next = (document.documentElement.getAttribute('data-theme') === 'light') ? 'dark' : 'light';
      document.documentElement.setAttribute('data-theme', next);
      try{ localStorage.setItem('mrf-theme', next); }catch(e){}
      sync();
    });
  })();
</script>
<div class="layout dashboard-shell">
  <aside class="top-toolbar" aria-label="Công cụ project">
    <details class="card project-panel" id="projectPanel" hidden>
      <summary>
        <span class="project-summary-prefix">Project</span>
        <strong id="projectSummaryLabel" class="project-label">Chưa chọn</strong>
      </summary>
      <div class="project-popover">
        <div class="project-toolbar">
          <label><input id="selectAllProjects" type="checkbox"> Chọn tất cả</label>
          <button id="deleteSelectedBtn" class="danger" type="button" disabled>Xóa đã chọn (0)</button>
        </div>
        <div id="bulkMsg" class="notice" role="status"></div>
        <div id="jobList" class="project-list" aria-live="polite"><span class="mrf-loading">Đang tải…</span></div>
      </div>
    </details>
    <button id="openCreateProject" class="primary toolbar-action" type="button" hidden>＋ Tạo project</button>
    <button id="openLibraryHub" class="toolbar-action" type="button">Thư viện</button>
    <button id="openChannelSwitcher" class="toolbar-action" type="button">Kênh</button>
    <button id="openBrandSettings" class="toolbar-action" type="button">Thương hiệu</button>
    <button id="openScoutHub" class="toolbar-action" type="button" hidden>Săn phim</button>
  </aside>

  <dialog id="createPanel" class="tool-dialog" aria-labelledby="createDialogTitle">
    <div class="dialog-shell">
      <div class="dialog-header">
        <div>
          <h2 id="createDialogTitle">Tạo project mới</h2>
          <p>Chọn MP4 trước. Các thiết lập nâng cao có thể để mặc định.</p>
        </div>
        <button id="closeCreateProject" class="dialog-close" type="button" aria-label="Đóng">×</button>
      </div>
      <div class="dialog-body">
        <form id="createForm" class="create-form">
          <label for="sourceFile">Video MP4</label>
          <input id="sourceFile" type="file" accept=".mp4,video/mp4">
          <details style="margin:8px 0 12px 0">
            <summary style="cursor:pointer;font-weight:500;color:var(--accent,#4f8ff7)">🔗 Hoặc dán link video (YouTube / Web / Trực tiếp)</summary>
            <div style="margin-top:6px;display:flex;gap:6px">
              <input id="linkUrl" type="url" placeholder="https://www.youtube.com/watch?v=..." style="flex:1">
              <button id="probeLinkBtn" type="button">Kiểm tra link</button>
            </div>
            <div id="linkPreview" class="notice" style="display:none;margin-top:6px"></div>
            <label class="checkbox-row" style="margin:10px 0 6px 0">
              <input id="confirmDownloadRights" type="checkbox">
              <span>Tôi xác nhận có quyền tải và sử dụng video này để review/bình luận</span>
            </label>
          </details>
          <input id="newJobId" name="job_id" type="hidden">
          <label>Tên phim / truy vấn nghiên cứu</label>
          <input name="movie_title" placeholder="vd: The Matrix (1999)">
          <details class="advanced-fields"><summary>Định hướng review</summary>
            <label for="briefTemplateSelect">Mẫu brief dùng lại</label>
            <select id="briefTemplateSelect"><option value="">Chọn mẫu để điền form…</option></select>
            <div class="row" style="margin:8px 0 12px 0">
              <button id="applyBriefTemplate" type="button">Áp dụng mẫu</button>
            </div>
            <label for="briefTemplateName">Lưu các trường bên dưới thành mẫu mới</label>
            <input id="briefTemplateName" maxlength="200" placeholder="Tên mẫu brief">
            <div class="row" style="margin:8px 0 12px 0;align-items:center;gap:8px">
              <button id="saveBriefTemplate" type="button">Lưu mẫu brief</button>
              <span id="briefTemplateMsg" class="notice" role="status"></span>
            </div>
            <label>Luận điểm chính</label><textarea name="review_thesis" rows="2" maxlength="500" placeholder="Điều bạn muốn người xem nhớ sau video"></textarea>
            <label>Giọng kể</label><input name="tone" maxlength="120" placeholder="Hài hước, phân tích, giàu cảm xúc...">
            <label>Khán giả</label><input name="target_audience" maxlength="200" placeholder="Người mới xem hay fan lâu năm">
            <label>Mức tiết lộ nội dung</label><select name="spoiler_policy"><option value="unspecified">Chưa chọn</option><option value="none">Không spoiler</option><option value="limited">Spoiler hạn chế</option><option value="full">Review toàn bộ</option></select>
            <label>Điều không được khẳng định (mỗi dòng một ý)</label><textarea name="forbidden_claims" rows="2" placeholder="Không đoán danh tính nhân vật..."></textarea>
          </details>
          <details class="advanced-fields"><summary>Tùy chọn dựng video</summary>
            <label>Ngôn ngữ</label>
            <input name="language" value="vi">
            <label>Bộ tạo nội dung</label>
            <select name="content_agent">
              <option value="scaffold">Scaffold (offline)</option>
              <option value="claude">Claude Code (nghiên cứu → dàn ý → kịch bản)</option>
              <option value="agy">AGY pool (nghiên cứu → dàn ý → kịch bản)</option>
            </select>
            <div class="row" id="agyPoolRow" style="margin-top:6px;align-items:center;gap:8px">
              <span id="agyPoolBadge" class="badge pending">AGY: chưa kiểm tra</span>
              <button type="button" id="agyProbeBtn">Kiểm tra AGY</button>
            </div>
            <label>Thời lượng mục tiêu (phút)</label>
            <input name="target_minutes" type="number" value="10" min="1" max="60" step="0.5">
            <label>Tỷ lệ khung hình</label>
            <select name="aspect_ratio"><option>16:9</option><option>9:16</option></select>
            <label>Biến đổi hình ảnh clip</label>
            <select name="visual_variety">
              <option value="off">Tắt</option>
              <option value="light">Nhẹ (Zoom 3%, chỉnh màu)</option>
              <option value="balanced" selected>Cân bằng (Lật ngang, Zoom 5%, chỉnh màu)</option>
              <option value="aggressive">Mạnh (Lật ngang, Zoom 8%, chỉnh màu, 1.06x)</option>
            </select>
            <label>Giọng đọc (TTS)</label>
            <select name="tts_provider">
              <option value="edge" selected>Edge-TTS (miễn phí, mặc định)</option>
              <option value="fptai">FPT.AI Voice (cần MRF_FPTAI_API_KEY)</option>
              <option value="elevenlabs">ElevenLabs (cần MRF_ELEVENLABS_API_KEY)</option>
              <option value="vieneu">VieNeu-TTS (miễn phí, offline 24kHz · cần: pip install vieneu)</option>
            </select>
            <label>Voice ID / tên giọng (FPT.AI / ElevenLabs / VieNeu, tùy chọn)</label>
            <input name="tts_voice" list="ttsVoiceHints" placeholder="vd: banmai (FPT.AI) hoặc 21m00Tcm4TlvDq8ikWAM (ElevenLabs)">
            <datalist id="ttsVoiceHints">
              <option value="banmai">FPT.AI · Ban Mai (nữ, miền Bắc)</option>
              <option value="lannhi">FPT.AI · Lan Nhi (nữ, miền Nam)</option>
              <option value="leminh">FPT.AI · Lê Minh (nam, miền Bắc)</option>
              <option value="minhquang">FPT.AI · Minh Quang (nam, miền Nam)</option>
              <option value="thuminh">FPT.AI · Thu Minh (nữ)</option>
              <option value="21m00Tcm4TlvDq8ikWAM">ElevenLabs · Rachel</option>
              <option value="AZnzlk1XvdvUeBnXmlld">ElevenLabs · Domi</option>
              <option value="EXAVITQu4vr4xnSDxMaL">ElevenLabs · Bella</option>
            </datalist>
            <p class="muted">Bỏ trống để dùng giọng mặc định. API key đặt qua biến môi trường, không lưu vào project.</p>
            <div class="tts-check" style="margin-top:6px; display:flex; gap:8px; align-items:center; flex-wrap:wrap">
              <button id="ttsTestBtn" type="button">Kiểm tra kết nối</button>
              <button id="ttsInstallBtn" type="button" class="accent-btn" hidden>⚡ Cài đặt tự động VieNeu-TTS</button>
              <span id="ttsTestMsg" class="muted" role="status"></span>
              <span id="ttsInstallMsg" class="muted" role="status"></span>
            </div>
            <label>Che dải watermark phía trên (0–20% chiều cao)</label>
            <input name="brand_top_band" type="number" value="0" min="0" max="0.2" step="0.01">
            <label>Che dải tiêu đề cũ phía dưới (0–20% chiều cao)</label>
            <input name="brand_bottom_band" type="number" value="0" min="0" max="0.2" step="0.01">
            <label>Tự động phát hiện &amp; xoá watermark chìm</label>
            <select name="watermark_detect">
              <option value="">Tắt</option>
              <option value="color">Ngưỡng màu (color)</option>
              <option value="temporal">Theo thời gian (temporal)</option>
              <option value="external">Detector ngoài (external)</option>
            </select>
            <label>Cách xoá watermark</label>
            <select name="watermark_method" class="wm-method"><option value="propainter">ProPainter</option></select>
            <p class="muted wm-method-help"></p>
            <div class="tts-check" style="margin-top:6px; display:flex; gap:8px; align-items:center; flex-wrap:wrap">
              <button id="propainterInstallBtn" type="button" class="accent-btn">⚡ Tải ProPainter an toàn</button>
              <span id="propainterInstallMsg" class="muted" role="status"></span>
            </div>
            <p class="muted">Nguồn và model được ghim checksum; ProPainter chỉ cấp phép phi thương mại, hãy bảo đảm quyền sử dụng.</p>
            <label>Lệnh detector ngoài (external) — để trống sẽ dùng MRF_MASK_DETECTOR_CMD</label>
            <input name="detector_cmd" placeholder="python detect.py --in VIDEO --out OUT">
          </details>
          <div class="row" style="margin-top:10px">
            <button class="primary" type="submit">Tạo project</button>
          </div>
          <progress id="uploadProgress" class="upload-progress" max="100" value="0" hidden></progress>
          <div id="createMsg" class="notice" role="status"></div>
        </form>
      </div>
    </div>
  </dialog>

  <dialog id="brandPanel" class="tool-dialog" aria-labelledby="brandDialogTitle">
    <div class="dialog-shell">
      <div class="dialog-header">
        <div>
          <h2 id="brandDialogTitle">Thương hiệu</h2>
          <p>Tên và logo dùng chung. Dải che chỉ áp dụng cho project đang mở.</p>
        </div>
        <button id="closeBrandSettings" class="dialog-close" type="button" aria-label="Đóng">×</button>
      </div>
      <div class="dialog-body">
        <div class="brand-fields">
          <div class="brand-preview"><img id="brandPreview" alt="Logo kênh"><strong id="brandPreviewName">Màn Kể</strong></div>
          <section class="settings-section">
            <h3>Nhận diện kênh</h3>
            <div class="settings-row">
              <div><label for="brandName">Tên kênh</label><input id="brandName" maxlength="40" value="Màn Kể" autocomplete="off"></div>
              <button id="saveBrandName" type="button">Lưu tên</button>
            </div>
            <div class="settings-row">
              <div><label for="brandLogo">Logo PNG nền trong suốt, tối đa 2 MB</label><input id="brandLogo" type="file" accept="image/png,.png"></div>
              <button id="saveBrandLogo" type="button">Lưu logo</button>
            </div>
            <div id="brandMsg" class="notice" role="status"></div>
            <small class="muted"><a href="/api/brand/logo.svg" download="man-ke.svg">Tải SVG gốc</a></small>
          </section>
          <section class="settings-section">
            <h3>Che chữ nguồn của project hiện tại</h3>
            <p>Chỉ dùng khi bạn có quyền xử lý nguồn. Kiểm tra lại bản dựng sau khi áp dụng.</p>
            <div class="settings-row band-row">
              <div><label for="brandTopBand">Dải phía trên (0–0,2)</label><input id="brandTopBand" type="number" min="0" max="0.2" step="0.01" value="0"></div>
              <div><label for="brandBottomBand">Dải phía dưới (0–0,2)</label><input id="brandBottomBand" type="number" min="0" max="0.2" step="0.01" value="0"></div>
            </div>
            <button id="brandRenderBtn" type="button" disabled>Dựng lại video đang chọn</button>
          </section>
        </div>
      </div>
    </div>
  </dialog>

  <dialog id="channelPanel" class="tool-dialog" aria-labelledby="channelDialogTitle">
    <div class="dialog-shell">
      <div class="dialog-header">
        <div>
          <h2 id="channelDialogTitle">Kênh (hồ sơ đa kênh)</h2>
          <p>Lưu sẵn logo, giọng, intro/outro và mức biến đổi hình ảnh cho từng kênh. Kích hoạt 1 chạm để đổi nhận diện cho các video dựng tiếp theo.</p>
        </div>
        <button id="closeChannelSwitcher" class="dialog-close" type="button" aria-label="Đóng">×</button>
      </div>
      <div class="dialog-body">
        <div class="settings-row">
          <div style="flex:1">
            <label for="channelSelect">Kênh đã lưu</label>
            <select id="channelSelect"></select>
          </div>
          <button id="activateChannelBtn" type="button">Kích hoạt</button>
          <button id="newChannelBtn" type="button">＋ Kênh mới</button>
        </div>
        <div id="channelActiveNote" class="muted" role="status" style="margin:4px 0 10px"></div>
        <div id="channelForm" class="create-form">
          <input id="chId" type="hidden">
          <label for="chName">Tên kênh</label>
          <input id="chName" maxlength="40" placeholder="vd: Màn Kể" autocomplete="off">
          <label for="chLogo">Logo PNG nền trong suốt (tối đa 2 MB) — tùy chọn</label>
          <div class="settings-row">
            <img id="chLogoPreview" alt="Logo kênh" style="width:40px;height:40px;object-fit:contain;border-radius:6px;background:#0d1017">
            <input id="chLogo" type="file" accept="image/png,.png" style="flex:1">
          </div>
          <label for="chTtsProvider">Giọng đọc (TTS)</label>
          <select id="chTtsProvider">
            <option value="edge">Edge-TTS (miễn phí, mặc định)</option>
            <option value="fptai">FPT.AI Voice</option>
            <option value="elevenlabs">ElevenLabs</option>
            <option value="vieneu">VieNeu-TTS (miễn phí, offline)</option>
          </select>
          <label for="chTtsVoice">Voice ID / tên giọng (tùy chọn)</label>
          <input id="chTtsVoice" maxlength="120" placeholder="vd: banmai hoặc 21m00Tcm4TlvDq8ikWAM">
          <label for="chAspect">Tỷ lệ khung hình</label>
          <select id="chAspect"><option value="16:9">16:9</option><option value="9:16">9:16</option></select>
          <label for="chLanguage">Ngôn ngữ</label>
          <input id="chLanguage" maxlength="20" value="vi">
          <div class="settings-row band-row">
            <div><label for="chIntro">Intro (giây, 0–15)</label><input id="chIntro" type="number" min="0" max="15" step="0.5" value="0"></div>
            <div><label for="chOutro">Outro (giây, 0–15)</label><input id="chOutro" type="number" min="0" max="15" step="0.5" value="0"></div>
          </div>
          <label for="chCopyright">Biến đổi hình ảnh clip</label>
          <select id="chCopyright">
            <option value="off">Tắt</option>
            <option value="light">Nhẹ</option>
            <option value="balanced">Cân bằng</option>
            <option value="aggressive">Mạnh</option>
          </select>
          <div class="settings-row band-row">
            <div><label for="chTopBand">Che dải trên (0–0,2)</label><input id="chTopBand" type="number" min="0" max="0.2" step="0.01" value="0"></div>
            <div><label for="chBottomBand">Che dải dưới (0–0,2)</label><input id="chBottomBand" type="number" min="0" max="0.2" step="0.01" value="0"></div>
          </div>
          <label style="margin-top:8px">SFX chuyển cảnh (whoosh/boom…) — tối đa 8</label>
          <div id="channelSfxList" class="muted" style="font-size:12px; margin:4px 0"></div>
          <div class="settings-row" style="flex-wrap:wrap; gap:6px">
            <input id="sfxLabel" maxlength="40" placeholder="Nhãn (vd: Whoosh)" style="flex:1; min-width:120px">
            <input id="sfxRights" maxlength="300" placeholder="Ghi chú quyền (bắt buộc)" style="flex:1; min-width:140px">
            <input id="sfxGain" type="number" min="-36" max="0" step="0.5" value="-8" title="dB" style="width:76px">
            <input id="sfxFile" type="file" accept="audio/*,.mp3,.wav,.m4a,.ogg" style="flex:1; min-width:160px">
            <button id="addSfxBtn" type="button">Thêm SFX</button>
          </div>
          <div class="row" style="margin-top:10px; gap:8px">
            <button id="saveChannelBtn" class="primary" type="button">Lưu kênh</button>
            <button id="deleteChannelBtn" type="button">Xoá kênh</button>
          </div>
          <div id="channelMsg" class="notice" role="status"></div>
        </div>
      </div>
    </div>
  </dialog>

  <dialog id="libraryPanel" class="tool-dialog tool-dialog-wide" aria-labelledby="libraryDialogTitle">
    <div class="dialog-shell">
      <div class="dialog-header">
        <div>
          <h2 id="libraryDialogTitle">Thư viện</h2>
          <p>Tìm cảnh, lời thoại, project và quản lý series ở một nơi.</p>
        </div>
        <button id="closeLibraryHub" class="dialog-close" type="button" aria-label="Đóng">×</button>
      </div>
      <div class="dialog-body library-hub">
        <section class="library-section">
          <h3>Tìm cảnh & lời thoại</h3>
          <p>Tìm xuyên tất cả project theo nội dung hoặc bộ lọc.</p>
          <div class="search-field" role="search">
            <label class="sr-only" for="librarySearch">Tìm lời thoại hoặc nội dung hình ảnh trong mọi project</label>
            <input id="librarySearch" type="search" maxlength="200" placeholder='vd: xe đỏ person:"Person 1" location:hospital'>
            <button id="librarySearchBtn" type="button">Tìm</button>
          </div>
          <details>
            <summary class="muted">Bộ lọc nâng cao</summary>
            <div class="filter-grid">
              <select id="libraryKind"><option value="">Mọi loại</option><option value="visual">Hình ảnh</option><option value="transcript">Lời thoại</option></select>
              <input id="libraryProject" placeholder="Dự án / Mã project">
              <input id="libraryPerson" placeholder="Nhân vật / tên gọi">
              <input id="libraryAction" placeholder="Hành động">
              <input id="libraryLocation" placeholder="Địa điểm">
              <input id="libraryObject" placeholder="Đồ vật">
              <input id="librarySceneType" placeholder="Loại cảnh / nhãn cảnh">
              <input id="librarySource" placeholder="Nguồn (agy/transcript/...)">
              <input id="libraryDateFrom" type="date" aria-label="Ngày dự án từ">
              <input id="libraryDateTo" type="date" aria-label="Ngày dự án đến">
              <input id="libraryMinDuration" type="number" min="0" step="0.1" placeholder="Số giây tối thiểu">
              <input id="libraryMaxDuration" type="number" min="0" step="0.1" placeholder="Số giây tối đa">
              <input id="libraryMinConfidence" type="number" min="0" max="1" step="0.05" placeholder="Độ tin cậy tối thiểu">
            </div>
          </details>
          <div class="row" style="margin-top:8px">
            <button id="saveLibrarySearchBtn" type="button">Lưu tìm kiếm</button>
            <select id="savedLibrarySearches" aria-label="Tìm kiếm thư viện đã lưu"><option value="">Tìm kiếm đã lưu…</option></select>
          </div>
          <div id="libraryResults" class="project-list muted" aria-live="polite">Nhập từ khóa hoặc bộ lọc để tìm xuyên mọi project.</div>
        </section>
        <section class="library-section" id="creatorLibraryPanel">
          <h3>Project & series</h3>
          <p>Tìm project theo nội dung và lưu thứ tự các phần trong series.</p>
          <label for="creatorProjectSearch">Tìm tên phim, brief, tiêu đề, mô tả, tag</label>
          <div class="row"><input id="creatorProjectSearch" type="search" maxlength="200" placeholder="Tìm project"><button id="creatorProjectSearchBtn" type="button">Tìm</button></div>
          <div id="creatorProjectResults" class="project-list notice" role="status"></div>
          <label for="seriesId">Mã series</label><input id="seriesId" maxlength="100" placeholder="vd: review-ben-10">
          <label for="seriesTitle">Tên series</label><input id="seriesTitle" maxlength="200" placeholder="Review thế giới Ben 10">
          <label for="seriesEntries">Phim theo thứ tự (mỗi dòng: tên phim | mã project, mã tùy chọn)</label>
          <textarea id="seriesEntries" rows="3" placeholder="Ben 10 Alien Swarm | ben-review&#10;Phần tiếp theo"></textarea>
          <button id="saveSeriesBtn" type="button">Lưu kế hoạch series</button>
          <div id="seriesMsg" class="notice" role="status"></div>
          <div id="seriesList" class="project-list notice"></div>
        </section>
      </div>
    </div>
  </dialog>

  <dialog id="scoutPanel" class="tool-dialog tool-dialog-wide" aria-labelledby="scoutDialogTitle">
    <div class="dialog-shell">
      <div class="dialog-header">
        <div>
          <h2 id="scoutDialogTitle">Tự động săn phim độc lạ</h2>
          <p>Tự động thu thập và xếp hạng phim xưa độc lạ (1985–2008), đoản kịch thịnh hành Trung Quốc và phim ít người biết theo công thức đánh giá tiềm năng lan truyền.</p>
        </div>
        <button id="closeScoutHub" class="dialog-close" type="button" aria-label="Đóng">×</button>
      </div>
      <div class="dialog-body">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; margin-bottom:14px">
          <div style="display:flex; gap:12px; flex-wrap:wrap; align-items:flex-end">
            <label style="min-width:180px">Chủ đề:
              <select id="scoutTopicSelect">
                <option value="all">Tất cả chủ đề</option>
                <option value="horror">Kinh dị / Quái vật / Rùng rợn</option>
                <option value="fantasy_mystery">Tiên hiệp / Huyền ảo / Bí ẩn</option>
                <option value="ceo_romance">Tổng tài / Nghịch tập / Đoản kịch</option>
                <option value="isekai_rebirth">Xuyên không / Trùng sinh</option>
                <option value="cult_classic">Phim xưa lạ / B-Movie độc lạ</option>
              </select>
            </label>
            <label style="min-width:180px">Nguồn dữ liệu:
              <select id="scoutSourceSelect">
                <option value="all">Tất cả 3 nguồn</option>
                <option value="tmdb_douban">1. TMDb / Douban (Phim xưa được đánh giá cao)</option>
                <option value="douyin_bilibili">2. Douyin / Bilibili (Đoản kịch thịnh hành)</option>
                <option value="youtube_obscure">3. YouTube (Phim đầy đủ, ít lượt xem)</option>
              </select>
            </label>
            <button id="scoutRefreshBtn" type="button" class="primary">Quét phim tiềm năng</button>
          </div>
          <div style="font-size:0.82rem; color:var(--text); background:var(--field-bg); padding:6px 12px; border-radius:6px; border:1px solid var(--line)">
            Công thức: <code style="color:var(--accent); font-weight:700">(Độ kịch tính × Điểm đánh giá) / (Độ phủ Việt Nam × Rủi ro bản quyền)</code>
          </div>
        </div>
        <div id="scoutLoading" class="muted" style="padding:16px; text-align:center" hidden>Đang quét nguồn dữ liệu và tính điểm tiềm năng lan truyền...</div>
        <div id="scoutGrid" style="display:grid; grid-template-columns:repeat(auto-fill, minmax(320px, 1fr)); gap:16px; max-height:65vh; overflow-y:auto; padding-right:4px"></div>
        <div id="scoutEmpty" class="muted card" style="text-align:center; padding:24px" hidden>Không tìm thấy phim phù hợp với bộ lọc hiện tại.</div>
      </div>
    </div>
  </dialog>
  <dialog id="scoutConfigDialog" class="tool-dialog" aria-labelledby="scoutConfigTitle">
    <div class="dialog-shell"><div class="dialog-header"><div><h2 id="scoutConfigTitle">Cấu hình nhanh dự án</h2><p>Kiểm tra thiết lập trước khi khởi tạo.</p></div><button id="closeScoutConfig" class="dialog-close" type="button" aria-label="Đóng">×</button></div>
      <div class="dialog-body"><form id="scoutConfigForm" class="create-form">
        <label for="scoutMovieTitle">Tên phim</label><input id="scoutMovieTitle" maxlength="300" required>
        <label class="checkbox-row"><input id="scoutWatermarkEnabled" type="checkbox" checked><span>Bật xoá watermark chìm</span></label>
        <label for="scoutWatermarkDetect">Phương pháp nhận diện</label><select id="scoutWatermarkDetect"><option value="color">Theo màu sắc</option><option value="temporal">Theo thời gian</option></select>
        <label for="scoutWatermarkMethod">Cách xoá watermark</label><select id="scoutWatermarkMethod" class="wm-method"><option value="propainter">ProPainter</option></select><p class="muted wm-method-help"></p>
        <label for="scoutContentAgent">Bộ tạo nội dung</label><select id="scoutContentAgent"><option value="agy">Nhóm AGY</option><option value="claude">Claude Code</option><option value="scaffold">Mẫu thử</option></select>
        <label for="scoutCopyright">Biến đổi hình ảnh clip</label><select id="scoutCopyright"><option value="balanced">Cân bằng (khuyên dùng)</option><option value="aggressive">Mạnh</option><option value="light">Nhẹ</option><option value="off">Tắt</option></select>
        <label for="scoutTtsProvider">Giọng đọc</label><select id="scoutTtsProvider"><option value="edge">Edge</option><option value="vieneu">VieNeu</option><option value="fptai">FPT.AI</option><option value="elevenlabs">ElevenLabs</option></select>
        <label for="scoutTtsVoice">Voice</label><input id="scoutTtsVoice" list="ttsVoiceSuggestions" maxlength="120" placeholder="Mặc định của provider"><datalist id="ttsVoiceSuggestions"><option value="banmai"><option value="vi-VN-HoaiMyNeural"><option value="vi-VN-NamMinhNeural"></datalist>
        <div class="row" style="margin-top:14px"><button type="button" id="cancelScoutConfig">Huỷ</button><button class="primary" type="submit">🚀 Khởi tạo dự án</button></div><div id="scoutConfigMsg" class="notice" role="status"></div>
      </form></div></div>
  </dialog>
  <dialog id="projectConfigDialog" class="tool-dialog" aria-labelledby="projectConfigTitle">
    <div class="dialog-shell"><div class="dialog-header"><div><h2 id="projectConfigTitle">Thiết lập dự án trước khi chạy</h2></div><button id="closeProjectConfig" class="dialog-close" type="button" aria-label="Đóng">×</button></div>
      <div class="dialog-body"><form id="projectConfigForm" class="create-form">
        <label for="projectMovieTitle">Tên phim</label><input id="projectMovieTitle" maxlength="300">
        <label class="checkbox-row"><input id="projectWatermarkEnabled" type="checkbox"><span>Bật xoá watermark chìm</span></label>
        <label for="projectWatermarkDetect">Phương pháp</label><select id="projectWatermarkDetect"><option value="color">Color</option><option value="temporal">Temporal</option><option value="external">External</option></select>
        <label for="projectWatermarkMethod">Cách xoá watermark</label><select id="projectWatermarkMethod" class="wm-method"><option value="propainter">ProPainter</option></select><p class="muted wm-method-help"></p>
        <label for="projectContentAgent">Bộ tạo nội dung</label><select id="projectContentAgent"><option value="agy">AGY Pool</option><option value="claude">Claude Code</option><option value="scaffold">Mẫu thử</option></select>
        <label for="projectCopyright">Biến đổi hình ảnh clip</label><select id="projectCopyright"><option value="balanced">Cân bằng</option><option value="aggressive">Mạnh</option><option value="light">Nhẹ</option><option value="off">Tắt</option></select>
        <label for="projectTtsProvider">TTS provider</label><select id="projectTtsProvider"><option value="edge">Edge</option><option value="vieneu">VieNeu</option><option value="fptai">FPT.AI</option><option value="elevenlabs">ElevenLabs</option></select>
        <label for="projectTtsVoice">Voice</label><input id="projectTtsVoice" maxlength="120">
        <div class="row" style="margin-top:14px"><button type="button" id="cancelProjectConfig">Huỷ</button><button id="saveProjectConfig" class="primary" type="submit">Lưu thiết lập</button></div><div id="projectConfigMsg" class="notice" role="status"></div>
      </form></div></div>
  </dialog>
  <main>
    <div id="empty" class="card empty-state" role="status">
      <div class="empty-state-mark" aria-hidden="true">＋</div>
      <h2>Bắt đầu một video review</h2>
      <p>Chọn MP4 để tạo project mới. Mặc định đã đủ để bắt đầu; brief và tùy chọn dựng chỉ cần mở khi bạn muốn chỉnh sâu.</p>
      <div class="empty-actions">
        <button id="emptyCreateBtn" class="primary" type="button">Tạo project từ MP4</button>
        <button id="emptyProjectsBtn" type="button">Mở project có sẵn</button>
        <button id="emptyScoutBtn" type="button" class="accent-btn">Săn phim tự động</button>
      </div>
      <div class="empty-help">Sau khi tạo, project mở thẳng vào workspace và giữ toàn bộ tiến trình ở một nơi.</div>
      <div id="batchPanel" class="card" style="margin-top:14px; text-align:left; width:100%; max-width:640px">
        <h3 style="margin:0 0 6px 0; font-size:1.05rem">Hàng đợi hàng loạt (chạy qua đêm)</h3>
        <p class="muted">Dán nhiều link, mỗi dòng một link. Hệ thống tải và dựng tuần tự tới bước duyệt kịch bản cho từng video; lỗi một video sẽ bỏ qua và chạy tiếp. Xong vào từng project bấm duyệt.</p>
        <textarea id="batchLinks" rows="5" style="width:100%" placeholder="https://..."></textarea>
        <label class="row" style="gap:6px; align-items:center; margin-top:6px">
          <input id="batchRights" type="checkbox" style="width:auto"> Tôi có quyền sử dụng các video này
        </label>
        <div class="row" style="margin-top:8px; gap:8px">
          <button id="batchStartBtn" type="button" class="primary">Chạy hàng đợi</button>
          <button id="batchStopBtn" type="button" class="danger" hidden>Dừng hàng đợi</button>
        </div>
        <div id="batchMsg" class="muted" role="status" style="margin-top:6px"></div>
        <div id="batchList" style="margin-top:6px"></div>
      </div>
    </div>
    <div id="detail" style="display:none">
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <h2 id="jobTitle" style="margin:0"></h2>
          <div class="row">
            <button id="runBtn" class="primary">Chạy pipeline</button>
            <button id="stopBtn" class="danger" hidden>Dừng project</button>
            <button id="refreshBtn">Làm mới</button>
            <button id="deleteBtn" class="danger" type="button">Xóa project</button>
          </div>
        </div>
        <div class="next-row"><div id="nextAction" class="action-note" role="status"></div><button id="reviewAction" type="button" hidden>Mở phần duyệt</button><a id="quickDownload" class="button-link" download="final.mp4" hidden>Tải MP4</a></div>
        <div class="notice">Chạy đến bước ảnh bìa. Duyệt kịch bản trước khi tạo giọng đọc; xuất bản là bước riêng.</div>
        <div id="projectConfigCard" class="card" style="margin-top:10px">
          <div class="row" style="justify-content:space-between;align-items:center"><div><h3 style="margin:0 0 6px">Cấu hình dự án tiền kỳ</h3><div id="projectConfigTags" class="row" style="gap:6px;flex-wrap:wrap"></div></div><button id="editProjectConfig" type="button">⚙️ Chỉnh sửa thiết lập</button></div>
          <div id="projectConfigWarning" class="notice" role="status"></div>
        </div>
        <div id="sourceRetryCard" hidden class="card" style="margin-top:10px; border:1px dashed var(--accent)">
          <h3 style="margin:0 0 6px 0; font-size:1.05rem">Project chưa có video nguồn</h3>
          <p class="muted" style="margin-bottom:10px">Tải video tự động từ link YouTube/Web qua yt-dlp hoặc chọn file MP4 có sẵn trên máy tính.</p>
          <div style="display:flex; flex-direction:column; gap:10px">
            <div style="padding:10px; border:1px solid var(--line); border-radius:8px; background:var(--field-bg)">
              <label style="font-weight:600; display:block; margin-bottom:4px">Cách 1: Tải video từ liên kết (yt-dlp)</label>
              <div class="row" style="gap:8px">
                <input id="sourceRetryUrl" type="url" placeholder="https://www.youtube.com/watch?v=..." style="flex:1">
                <button id="sourceRetryUrlBtn" type="button" class="primary">Tải video ngay</button>
              </div>
              <label class="row" style="gap:6px; align-items:center; margin-top:6px; font-size:0.85rem">
                <input id="sourceRetryRights" type="checkbox" checked style="width:auto"> Tôi có quyền sử dụng video này
              </label>
            </div>
            <div style="padding:10px; border:1px solid var(--line); border-radius:8px; background:var(--field-bg)">
              <label style="font-weight:600; display:block; margin-bottom:4px">Cách 2: Chọn file MP4 từ máy tính</label>
              <div class="row" style="gap:8px; align-items:center">
                <input id="sourceRetry" type="file" accept=".mp4,video/mp4" style="flex:1">
                <button id="sourceRetryBtn" type="button">Import MP4</button>
              </div>
            </div>
          </div>
          <div id="sourceRetryFallbackSection" style="margin-top:10px; padding:10px; border:1px solid var(--line); border-radius:8px; background:var(--field-bg)" hidden>
            <div style="font-weight:600; margin-bottom:4px; font-size:0.9rem; color:var(--text)">Không tìm thấy hoặc video bị lỗi?</div>
            <div class="muted" style="font-size:0.83rem; margin-bottom:8px">Tìm kiếm video thay thế cho bộ phim này trên YouTube bằng 1 click:</div>
            <a id="sourceRetrySearchFallbackBtn" class="button-link" target="_blank" rel="noopener noreferrer" style="display:inline-flex; align-items:center; gap:6px; font-weight:600; text-decoration:none">🔍 Tìm video thay thế trên YouTube</a>
          </div>
          <div id="sourceRetryMsg" class="notice" role="status" style="margin-top:8px"></div>
        </div>
        <div style="margin-top:10px" class="progress"><div id="progBar"></div></div>
        <div id="progText" class="muted" style="margin-top:6px"></div>
        <div id="runErr" class="err" style="margin-top:6px"></div>
        <div id="voiceInfo" class="muted" style="margin-top:6px"></div>
        <details class="stages-panel"><summary>Tiến trình các bước</summary><div id="stages"></div></details>
      </div>

      <nav class="workspace-tabs" role="tablist" aria-label="Khu vực làm việc">
        <button type="button" role="tab" id="tab-explore" aria-controls="view-explore" aria-selected="true" data-view="explore">Khám phá</button>
        <button type="button" role="tab" id="tab-edit" aria-controls="view-edit" aria-selected="false" data-view="edit" tabindex="-1">Biên tập</button>
        <button type="button" role="tab" id="tab-review" aria-controls="view-review" aria-selected="false" data-view="review" tabindex="-1">Duyệt &amp; xuất</button>
        <button type="button" role="tab" id="tab-files" aria-controls="view-files" aria-selected="false" data-view="files" tabindex="-1">Tệp</button>
      </nav>
      <section id="view-review" class="workspace-view" role="tabpanel" aria-labelledby="tab-review" hidden>
      <div class="card" id="videoCard" style="display:none">
        <h2>Xem trước bản dựng cuối</h2>
        <video id="video" controls preload="metadata"></video>
      </div>
      <div class="card" id="exportCard" hidden>
        <h2>Xuất bản &amp; bàn giao</h2>
        <div class="muted">Bản MP4 đã vượt qua bước kiểm tra chất lượng.</div>
        <a id="downloadFinal" class="button-link" download="final.mp4">Tải final.mp4</a>
        <button id="handoffBtn" type="button">Đóng gói bàn giao</button>
        <span id="handoffMsg" class="muted" role="status"></span>
        <details>
          <summary>Video ngắn 9:16 từ bản review</summary>
          <div class="muted">Chọn 3–60 giây lời bình trong video cuối đã duyệt.</div>
          <div class="filter-grid">
            <label for="shortStart">Mốc đầu (giây)<input id="shortStart" type="number" min="0" step="0.1"></label>
            <label for="shortEnd">Mốc cuối (giây)<input id="shortEnd" type="number" min="0" step="0.1"></label>
            <button id="shortMarkStart" type="button">Lấy mốc đầu từ video</button>
            <button id="shortMarkEnd" type="button">Lấy mốc cuối từ video</button>
          </div>
          <div class="row" style="margin-top:12px; gap:8px">
            <button id="shortExportBtn" type="button">Xuất video ngắn</button>
            <a id="shortDownload" hidden download="short-review.mp4">Tải MP4 9:16</a>
            <a id="shortSrtDownload" hidden download="short-review.srt">Tải phụ đề SRT</a>
          </div>
          <span id="shortExportMsg" class="muted" role="status"></span>
        </details>
        <details id="analyticsPanel">
          <summary>Học từ số liệu YouTube Studio</summary>
          <p class="muted">Sau khi tải gói bàn giao và đăng video thủ công, nhập CSV/JSON retention đã đo. Số liệu gắn với đúng phiên bản xuất; không dự đoán lượt xem.</p>
          <button id="studioGuideBtn" type="button">Cách lấy file CSV từ Studio (30 giây)</button>
          <label for="studioExport">File Studio (CSV/JSON, tối đa 2 MB)</label>
          <input id="studioExport" type="file" accept=".csv,.json,text/csv,application/json">
          <div class="filter-grid">
            <label for="analyticsCTA">Mốc bắt đầu CTA trong video cuối (giây, nếu có)<input id="analyticsCTA" type="number" min="0" step="0.1"></label>
            <label for="analyticsNotes">Ghi chú rút kinh nghiệm<textarea id="analyticsNotes" maxlength="2000" rows="2"></textarea></label>
          </div>
          <button id="analyticsUploadBtn" type="button">Nhập số liệu đã đo</button>
          <span id="analyticsMsg" role="status" class="muted"></span>
          <div id="analyticsResults" role="status" class="muted"></div>
          <div id="analyticsAdvice" role="status" class="muted"></div>
          <dialog id="studioGuide" style="max-width:540px; border:1px solid var(--border, rgba(128,128,128,.35)); border-radius:10px; padding:18px">
            <h3 style="margin:0 0 8px">Lấy file CSV giữ chân người xem từ YouTube Studio</h3>
            <ol style="padding-left:18px; line-height:1.7; margin:0">
              <li>Sau khi đăng video 24–48h, mở <a href="https://studio.youtube.com" target="_blank" rel="noopener noreferrer">YouTube Studio</a>.</li>
              <li>Mở video → tab <strong>Số liệu phân tích</strong> (Analytics) → <strong>Mức độ tương tác</strong> (Engagement).</li>
              <li>Tại biểu đồ <strong>Mức giữ chân người xem</strong> (Audience retention), bấm <strong>Chế độ nâng cao</strong> (Advanced mode) ở góc trên bên phải.</li>
              <li>Bấm <strong>Xuất dữ liệu</strong> (Export) → chọn <strong>Giá trị phân tách bằng dấu phẩy (.csv)</strong>.</li>
              <li>File cần có cột <code>time_seconds</code> và <code>retention_percent</code> (0–100); có thể thêm <code>impressions</code>, <code>ctr_percent</code>. Kéo vào ô “File Studio” phía trên rồi bấm “Nhập số liệu đã đo”.</li>
            </ol>
            <div class="row" style="justify-content:flex-end; margin-top:12px">
              <button id="studioGuideClose" type="button">Đã hiểu</button>
            </div>
          </dialog>
        </details>
        <h3 style="margin-top:16px; border-top:1px solid var(--border, rgba(128,128,128,.25)); padding-top:12px">Cổng xuất bản</h3>
        <div class="gate">
          <div>Thao tác này <strong>không tải lên</strong> bất kỳ đâu. Nó chỉ ghi tệp bàn giao
            <code>publish_record.json</code> khi kịch bản và siêu dữ liệu đã được duyệt.</div>
          <label class="row" style="margin-top:8px; gap:6px; align-items:center">
            <input id="confirmPub" type="checkbox" style="width:auto"> Tôi xác nhận tạo bản ghi xuất bản
          </label>
          <div class="row" style="margin-top:8px">
            <button id="publishBtn" disabled>Tạo bản ghi xuất bản</button>
          </div>
          <div id="pubMsg" class="notice"></div>
        </div>
      </div>
      <details id="rightsPanel" class="card">
        <summary>Nguồn và quyền sử dụng tài sản</summary>
        <p class="muted">Ghi chú do creator xác nhận; chưa kiểm chứng tự động quyền sử dụng.</p>
        <label for="rightsAssetPath">Đường dẫn tài sản</label><input id="rightsAssetPath" maxlength="200" placeholder="Video nguồn, nhạc hoặc logo">
        <label for="rightsSource">Nguồn tài sản</label><input id="rightsSource" maxlength="200" placeholder="Người cung cấp hoặc kho tài sản">
        <label for="rightsUsage">Cách dùng</label><input id="rightsUsage" maxlength="200" placeholder="Trích cảnh review">
        <label for="rightsStatus">Tình trạng quyền</label>
        <select id="rightsStatus"><option value="unreviewed">Chưa duyệt</option><option value="permitted">Có quyền theo ghi chú</option><option value="restricted">Hạn chế sử dụng</option></select>
        <label for="rightsEvidence">Ghi chú căn cứ / giấy phép</label><textarea id="rightsEvidence" rows="2" maxlength="1000"></textarea>
        <div class="row" style="margin:8px 0 10px 0">
          <button id="saveRightsBtn" type="button">Lưu ghi chú quyền</button>
        </div>
        <div id="rightsMsg" class="notice" role="status"></div>
        <div id="rightsList" class="project-list notice"></div>
      </details>
      <details id="versionsPanel" class="card">
        <summary>Phiên bản và khôi phục</summary>
        <div class="muted">Lưu mốc trước khi sửa. Bản MP4 đạt QA trước đó vẫn có thể tải về.</div>
        <label for="versionName">Tên phiên bản</label>
        <input id="versionName" type="text" maxlength="100" placeholder="Ví dụ: Trước khi sửa đoạn kết">
        <div class="row" style="margin:8px 0 12px 0">
          <button id="saveVersionBtn" type="button">Lưu phiên bản</button>
        </div>
        <label for="versionSelect">Phiên bản đã lưu</label>
        <select id="versionSelect" aria-label="Phiên bản đã lưu"></select>
        <label for="versionKind">Khôi phục phần</label>
        <select id="versionKind"><option value="script">Kịch bản</option><option value="scene_plan">Cảnh</option><option value="captions">Phụ đề</option><option value="metadata">Thông tin đăng</option><option value="final">Video đã QA</option></select>
        <div class="row" style="margin:8px 0 12px 0;align-items:center;gap:8px">
          <button id="restoreVersionBtn" type="button">Khôi phục</button>
          <a id="versionDownload" hidden download="final.mp4">Tải bản MP4 cũ</a>
          <span id="versionMsg" class="muted" role="status"></span>
        </div>
      </details>
      <div id="qaFindings" class="card muted" role="status" hidden></div>
      </section>
      <section id="view-explore" class="workspace-view" role="tabpanel" aria-labelledby="tab-explore">
      <section class="card media-shell" id="mediaExplorerCard" style="display:none" aria-labelledby="mediaExplorerTitle">
        <div class="media-topbar">
          <div class="media-title">
            <h2 id="mediaExplorerTitle">Trình khám phá tư liệu</h2>
            <p>Xem lại cảnh quay, theo dõi lời thoại và trích đoạn trong cùng một không gian làm việc.</p>
          </div>
          <div class="export-tools" aria-label="Công cụ xuất">
            <button id="reindexTranscriptBtn" type="button" title="Bóc lại lời thoại rồi dựng lại mốc thời gian phụ đề cho video nguồn (không đụng kịch bản/bản dựng)">↻ Index lại lời thoại</button>
            <span id="reindexMsg" class="muted" role="status" aria-live="polite"></span>
            <button id="mediaExportBtn" type="button" style="display:none" aria-label="Tải bản ghi lời thoại dạng VTT">↓ VTT</button>
          </div>
        </div>
        <div class="media-workspace">
          <div class="player-pane">
            <div class="sticky-player">
              <div class="source-frame">
                <video id="sourceVideo" controls preload="metadata" aria-label="Trình phát video nguồn"><track id="sourceCaptions" kind="subtitles" srclang="und" label="Lời thoại nguồn" default></video>
              </div>
              <div class="player-hint"><span>Bấm vào dòng lời thoại hoặc cảnh để tua tới</span><span id="playerTime" aria-live="off">00:00</span></div>
              <details class="extras-panel"><summary>Khoảnh khắc nổi bật và hỏi đáp video</summary>
              <section class="highlight-section" aria-labelledby="highlightsHeading">
                <div class="section-heading"><h3 id="highlightsHeading">Khoảnh khắc nổi bật thông minh</h3><span class="muted">Đoạn gợi ý cho video dọc</span></div>
                <div id="highlightResults" class="highlight-grid"><div class="state-panel"><span class="mrf-loading">Đang tải khoảnh khắc nổi bật…</span></div></div>
              </section>
              <section class="chat-box" aria-labelledby="askHeading">
                <div class="section-heading"><h3 id="askHeading">Hỏi đáp về video này</h3><span class="muted">Dựa trên lời thoại đã lập chỉ mục</span></div>
                <div class="search-field"><label class="sr-only" for="chatQuestion">Câu hỏi về video này</label><input id="chatQuestion" maxlength="500" placeholder="Chuyện gì xảy ra sau cảnh ngọn hải đăng?"><button id="chatAskBtn" type="button">Hỏi</button></div>
                <div id="chatAnswer" class="chat-answer notice" aria-live="polite">Đặt câu hỏi để tìm các đoạn liên quan trong video.</div>
              </section>
              </details>
            </div>
          </div>
          <div class="browser-pane">
            <div class="media-searchbar">
              <div class="search-field" role="search"><label class="sr-only" for="mediaSearch">Tìm trong lời thoại và cảnh</label><input id="mediaSearch" type="search" placeholder="Tìm từ khóa, người nói hoặc cảnh…" autocomplete="off"><button id="mediaSearchBtn" type="button">Tìm kiếm</button></div>
              <div class="media-tabs" role="tablist" aria-label="Bộ lọc kết quả tư liệu">
                <button class="media-tab" type="button" role="tab" aria-selected="true" data-media-filter="transcript">Lời thoại <span id="transcriptCount"></span></button>
                <button class="media-tab" type="button" role="tab" aria-selected="false" data-media-filter="scenes">Cảnh <span id="sceneCount"></span></button>
                <button class="media-tab" type="button" role="tab" aria-selected="false" data-media-filter="highlights">Nổi bật <span id="highlightCount"></span></button>
              </div>
            </div>
            <div id="mediaResults" class="media-list" aria-live="polite" aria-busy="false"></div>
            <div id="mediaState" class="state-panel" role="status" hidden></div>
            <div id="mediaMsg" class="sr-only" aria-live="polite"></div>
          </div>
        </div>
      </section>
      </section>
      <section id="view-edit" class="workspace-view" role="tabpanel" aria-labelledby="tab-edit" hidden>
      <details class="card" id="audioPanel">
        <summary>Âm thanh</summary>
        <p class="muted">Mặc định chỉ dùng lời đọc. Nhạc và hiệu ứng phải có ghi chú quyền sử dụng; nhạc tự hạ khi có lời đọc. Không lấy tiếng phim nguồn.</p>
        <div id="channelSfxPalette" hidden style="margin:6px 0 10px; padding:8px; border:1px solid var(--line, #333a48); border-radius:8px">
          <label>SFX chuyển cảnh của kênh — chèn tại giây đang xem</label>
          <div id="channelSfxButtons" class="row" style="flex-wrap:wrap; gap:6px; margin-top:6px"></div>
          <div class="row" style="gap:6px; margin-top:6px">
            <button id="autoTransitionSfxBtn" type="button">Tự rải theo cắt cảnh</button>
          </div>
          <div class="muted" style="font-size:12px; margin-top:4px">Bấm nút palette để chèn tại giây đang xem, hoặc “Tự rải theo cắt cảnh” để tự đặt tại các điểm chuyển cảnh (cần đã dựng video một lần; tối đa 16).</div>
        </div>
        <label for="audioVoiceGain">Lời đọc (dB)</label>
        <input id="audioVoiceGain" type="number" min="-12" max="12" step="0.5" value="0">
        <label for="audioMusicPath">Tệp nhạc nền trên máy (để trống nếu không dùng)</label>
        <input id="audioMusicPath" type="text" placeholder="Đường dẫn tệp âm thanh">
        <label for="audioMusicRights">Quyền sử dụng nhạc nền</label>
        <input id="audioMusicRights" type="text" placeholder="Ví dụ: tự sáng tác hoặc giấy phép sử dụng">
        <label for="audioMusicGain">Nhạc nền (dB)</label>
        <input id="audioMusicGain" type="number" min="-36" max="0" step="0.5" value="-18">
        <label for="audioEffectPath">Tệp hiệu ứng (để trống nếu không dùng)</label>
        <input id="audioEffectPath" type="text" placeholder="Đường dẫn tệp âm thanh">
        <label for="audioEffectRights">Quyền sử dụng hiệu ứng</label>
        <input id="audioEffectRights" type="text" placeholder="Ví dụ: tự thu âm">
        <label for="audioEffectTime">Vị trí hiệu ứng (giây)</label>
        <input id="audioEffectTime" type="number" min="0" step="0.1" value="0">
        <label for="audioEffectGain">Hiệu ứng (dB)</label>
        <input id="audioEffectGain" type="number" min="-36" max="0" step="0.5" value="-12">
        <button id="audioSaveBtn" type="button">Lưu âm thanh</button>
        <div id="audioMsg" class="notice" role="status"></div>
      </details>
      <div class="card" id="editorCard">
        <div class="row" style="justify-content:space-between">
          <h2 style="margin:0">Dòng thời gian &amp; Liền mạch</h2>
          <button id="autoBrollBtn" type="button">Tự chèn B-roll lặp cảnh</button>
        </div>
        <div class="muted">Khóa clip để giữ nguyên; cắt/thay chỉ làm mất hiệu lực các bước sau kế hoạch cảnh.</div>
        <div id="editorState" class="action-note" role="status" hidden></div>
        <details id="sectionPreviewPanel">
          <summary>Xem nhanh một phần · 540p</summary>
          <label for="sectionPreviewSelect" style="margin:8px 0 4px">Phần cần xem</label>
          <div class="row" style="align-items:center;gap:8px">
            <select id="sectionPreviewSelect" style="flex:1;min-width:0;max-width:520px"></select>
            <button id="sectionPreviewBtn" type="button" style="white-space:nowrap">Dựng nhanh phần này</button>
            <a id="sectionPreviewDownload" hidden download="section-preview.mp4">Tải MP4</a>
            <a id="sectionPreviewSrt" hidden download="section-preview.srt">Tải SRT</a>
          </div>
          <video id="sectionPreviewVideo" controls preload="metadata" style="width:100%;max-height:440px" hidden aria-label="Xem trước phần đã chọn"></video>
          <div id="sectionPreviewMsg" class="notice" role="status"></div>
        </details>
        <details id="hookTeaserPanel" style="margin-top:10px">
          <summary>Hook Teaser (3–5s) · Giữ chân người xem</summary>
          <p class="muted">Tự động chọn cảnh kịch tính nhất từ chỉ mục cảnh (scenes.json), cắt teaser ngắn 3–5 giây kèm hiệu ứng punch-in zoom để làm hook mở đầu video review.</p>
          <div class="row">
            <button id="buildHookBtn" type="button">Tạo Hook Teaser</button>
            <a id="hookDownload" hidden download="hook.mp4">Tải hook.mp4</a>
          </div>
          <video id="hookVideo" controls preload="metadata" style="width:100%;max-height:360px;margin-top:8px" hidden aria-label="Xem trước Hook Teaser"></video>
          <div id="hookMsg" class="notice" role="status"></div>
        </details>
        <div id="continuityTracks" class="continuity-grid"></div>
        <div id="timelineList" class="timeline-list"></div>
        <div id="editorMsg" class="notice" role="status"></div>
      </div>

      </section>
      <section id="view-review-content" class="workspace-view" hidden>
      <div class="card" id="thumbnailCard" style="display:none">
        <h2>Chọn ảnh bìa</h2>
        <div class="muted">3 phương án được AGY tự thiết kế từ kịch bản, đã che watermark của kênh nguồn — không cần gõ chữ. Ảnh được chọn sẽ trở thành <code>thumbnail.jpg</code> và cần duyệt lại metadata &amp; gói xuất.</div>
        <button id="thumbnailAutoBtn" type="button" style="margin-top:10px">✨ Tự động tạo 3 ảnh bìa với AGY</button>
        <div id="thumbnailVariantGrid" class="thumb-grid" style="margin-top:10px"></div>
        <details id="thumbnailEditor" style="margin-top:12px">
          <summary>Chọn khung hình gốc hoặc tự nhập chữ (nâng cao)</summary>
          <p class="muted">Chọn một khung hình gốc, hoặc tự nhập tiêu đề và tên kênh. Chữ nằm trong vùng an toàn; xem bản thu nhỏ trước khi chọn. Thay ảnh sẽ cần duyệt lại metadata và gói xuất.</p>
          <div id="thumbnailGrid" class="thumb-grid" style="margin-top:10px"></div>
          <div class="filter-grid">
            <label for="thumbnailHeadline">Dòng chính (tối đa 64 ký tự)<input id="thumbnailHeadline" maxlength="64" placeholder="BEN 10: AI ĐANG ĐIỀU KHIỂN THỜI GIAN?"></label>
            <label for="thumbnailChannel">Tên kênh trên ảnh<input id="thumbnailChannel" maxlength="40"></label>
          </div>
          <button id="thumbnailEditBtn" type="button">Tạo 3 phương án (thủ công)</button>
        </details>
        <div id="thumbnailMsg" class="notice" role="status"></div>
      </div>

      </section>
      <section id="view-files" class="workspace-view" role="tabpanel" aria-labelledby="tab-files" hidden>
        <div class="card"><h2>Tệp project</h2><div id="artifacts" class="artifact-grid muted"></div></div>
      </section>
      <section id="view-review-form" class="workspace-view" hidden>
      <div class="card" id="scriptReviewCard">
        <h2>Kịch bản &amp; Duyệt</h2>
        <div id="midrollStatus" class="cta-status" role="status">CTA: đang kiểm tra trạng thái…</div>
        <details id="midrollPanel">
          <summary>CTA giữa video · AGY</summary>
          <p class="muted">AGY viết câu hài ngắn tại ranh giới gần 50% video. Bạn duyệt câu trong kịch bản trước khi dựng lại.</p>
          <button id="midrollDraftBtn" type="button">AGY viết câu CTA</button>
          <label for="midrollLine">Lời thoại (có thể chỉnh sửa)</label>
          <textarea id="midrollLine" rows="3" maxlength="250" placeholder="Câu CTA sẽ hiện ở đây"></textarea>
          <div class="row"><button id="midrollStageBtn" type="button" disabled>Chèn vào kịch bản</button></div>
          <div id="midrollMsg" class="notice" role="status"></div>
        </details>
        <div id="scriptState" class="muted"></div>
        <label>Nội dung các phần (JSON sections)</label>
        <textarea id="scriptSections" rows="8"></textarea>
        <details><summary>Gắn nhãn lời dẫn và nguồn chứng cứ</summary>
          <label>Phần lời dẫn đã lưu</label><select id="tagSection"></select>
          <label>Bôi đen đoạn cần gắn nhãn</label><textarea id="tagNarration" rows="4" readonly></textarea>
          <div class="row"><select id="tagKind"><option value="plot_recap">Kể lại tình tiết</option><option value="opinion">Nhận xét riêng</option></select>
          <input id="tagEvidence" placeholder="scene:1 hoặc transcript:2"></div>
          <button id="tagScriptBtn" type="button" disabled style="margin-top:10px">Gắn nhãn đoạn đã chọn</button>
          <div class="muted">Mốc chỉ xác nhận nguồn có tồn tại; người làm review cần đối chiếu nội dung.</div>
          <div id="tagScriptMsg" class="notice" role="status"></div>
        </details>
        <div class="row" style="margin-top:10px">
          <button id="saveScriptBtn" disabled>Lưu kịch bản</button>
          <button id="approveScriptBtn" class="primary" disabled>Duyệt kịch bản</button>
        </div>
        <div class="notice">Lưu thay đổi sẽ đặt lại trạng thái duyệt (approved=false) — phải duyệt lại sau khi sửa.</div>
        <div id="scriptMsg" class="notice"></div>
      </div>

      <div class="card" id="metadataReviewCard">
        <h2>Siêu dữ liệu &amp; Duyệt</h2>
        <div id="metaState" class="muted"></div>
        <label>Tiêu đề</label>
        <input id="metaTitle">
        <label>Mô tả</label>
        <textarea id="metaDesc" rows="5"></textarea>
        <label>Thẻ (phân tách bằng dấu phẩy)</label>
        <input id="metaTags">
        <div class="row" style="margin-top:10px">
          <button id="saveMetaBtn">Lưu thay đổi</button>
          <button id="approveBtn" class="primary">Duyệt siêu dữ liệu</button>
        </div>
        <div class="notice">Lưu thay đổi sẽ đặt lại trạng thái duyệt (approved=false) — phải duyệt lại sau khi sửa.</div>
        <div id="metaMsg" class="notice"></div>
      </div>

      </section>
    </div>
  </main>
</div>
<dialog id="deleteDialog" aria-labelledby="deleteTitle">
  <form method="dialog" id="deleteForm">
    <h2 id="deleteTitle">Xóa project</h2>
    <p>Toàn bộ video nguồn và tệp xuất trong project sẽ bị xóa. Video gốc ở đường dẫn ngoài project không bị xóa.</p>
    <div id="deleteTargets" class="delete-targets"></div>
    <label id="deleteConfirmLabel" for="deleteConfirm">Nhập đúng mã project để xác nhận</label>
    <input id="deleteConfirm" autocomplete="off" required>
    <div class="row delete-actions"><button type="button" id="cancelDelete">Hủy</button><button id="confirmDelete" class="danger" type="submit" disabled>Xóa project</button></div>
    <div id="deleteMsg" class="err" role="alert"></div>
  </form>
</dialog>
<script>
const $ = (id) => document.getElementById(id);
let current = null;
let currentStatus = null;
let pendingScoutGem = null;
let poller = null;
let editorLoaded = null;
// Which review block matches the current next action: 'script' | 'metadata'
// | 'thumbnail' | 'export'. loadStatus() keeps this in sync with nextAction so
// opening "Duyệt & xuất" jumps straight to the block that needs attention.
let reviewFocus = 'script';
// The card that matches each focus. Opening the review view scrolls to this
// block; the FORM cards (script/metadata) collapse so only the
// one called out by the next action is expanded. videoCard / exportCard /
// thumbnailCard keep whatever visibility their own state gave them (a ready
// preview or export must never disappear just because it is not the focus).
const REVIEW_FOCUS_TARGET = {
  script: 'scriptReviewCard',
  metadata: 'metadataReviewCard',
  thumbnail: 'thumbnailCard',
  export: 'exportCard',
};
// Only these form cards collapse by focus; the rest are governed by loadStatus.
const REVIEW_FORM_CARDS = ['scriptReviewCard', 'metadataReviewCard'];
function applyReviewFocus(scroll = false) {
  const focus = REVIEW_FOCUS_TARGET[reviewFocus] ? reviewFocus : 'script';
  // Both review sub-sections stay visible so every state-driven card (final
  // preview, export, thumbnail) can still show; we only collapse the sibling
  // form cards that are not the current focus.
  $('view-review-content').hidden = false;
  $('view-review-form').hidden = false;
  const focusFormCard = REVIEW_FORM_CARDS.includes(REVIEW_FOCUS_TARGET[focus]) ? REVIEW_FOCUS_TARGET[focus] : null;
  for (const id of REVIEW_FORM_CARDS) {
    const el = $(id);
    if (el) el.hidden = focusFormCard ? id !== focusFormCard : false;
  }
  // Only scroll on an explicit user action (opening the review view); never on
  // background loadStatus polls, which would jerk the page during a run.
  const target = $(REVIEW_FOCUS_TARGET[focus]);
  if (scroll && target && target.offsetParent !== null) target.scrollIntoView({behavior: 'smooth', block: 'start'});
}
function setWorkspaceView(view) {
  for (const tab of document.querySelectorAll('.workspace-tabs [role="tab"]')) {
    const active = tab.dataset.view === view;
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  }
  for (const name of ['explore', 'edit', 'review', 'files']) {
    const el = $('view-' + name);
    if (el) el.hidden = name !== view;
  }
  // The section preview <video> lives inside a collapsible panel on the "Biên
  // tập" tab. Leaving that tab hides the panel (and its native controls), so a
  // still-playing preview would keep emitting audio with no reachable pause
  // button. Stop playback whenever we navigate away from the edit view.
  if (view !== 'edit') { const preview = $('sectionPreviewVideo'); if (preview && !preview.paused) preview.pause(); }
  if (view === 'review') {
    applyReviewFocus(true);
  } else {
    $('view-review-content').hidden = true;
    $('view-review-form').hidden = true;
  }
}
document.querySelectorAll('.workspace-tabs [role="tab"]').forEach(tab => {
  tab.addEventListener('click', () => setWorkspaceView(tab.dataset.view));
  tab.addEventListener('keydown', event => {
    const tabs = [...document.querySelectorAll('.workspace-tabs [role="tab"]')];
    const i = tabs.indexOf(tab);
    const target = event.key === 'ArrowRight' ? tabs[(i + 1) % tabs.length]
      : event.key === 'ArrowLeft' ? tabs[(i + tabs.length - 1) % tabs.length]
      : event.key === 'Home' ? tabs[0] : event.key === 'End' ? tabs[tabs.length - 1] : null;
    if (target) { event.preventDefault(); target.focus(); setWorkspaceView(target.dataset.view); }
  });
});
$('reviewAction').onclick = () => { reviewFocus = 'script'; setWorkspaceView('review'); $('scriptSections').focus(); };
// Fold Vietnamese diacritics and đ, keep only [a-z0-9_-], so the value always
// matches the server's _is_safe_segment instead of being rejected with no card.
function slugifyJobId(raw) {
  return (raw || '').normalize('NFKD').replace(/[\\u0300-\\u036f]/g, '')
    .replace(/[đĐ]/g, 'd').toLowerCase().replace(/[^a-z0-9_-]+/g, '-')
    .replace(/^-+|-+$/g, '').slice(0, 64).replace(/-+$/g, '');
}
// The project code is generated automatically and stays hidden from the form;
// the created code is shown in the success message after the project is made.
function autoJobId(title, file) {
  const fromTitle = slugifyJobId(title || '');
  const fromFile = file ? slugifyJobId(file.name.replace(/[.]mp4$/i, '')) : '';
  const base = fromTitle || fromFile || 'review';
  return (base + '-' + Date.now().toString(36).slice(-4)).slice(0, 64).replace(/-+$/g, '');
}

const selectedProjects = new Set();
let projectJobs = [];
let thumbnailObjectUrls = [];
let mediaObjectUrls = [];
let mediaLoaded = null;
let mediaExplorerData = { transcript: [], shots: [], highlights: [] };
let mediaFilter = 'transcript';
let manualTranscriptScrollUntil = 0;
let activeTranscriptId = null;
let pendingLibrarySeek = null;
let activeTimelinePreviewIndex = -1;
let currentTimelineData = null;

// Read bearer token from URL fragment (#token=...) — fragment is never sent
// to the server so the token never appears in access logs.  Not persisted.
let _tok = '';
(function () {
  const m = location.hash.replace(/^#/, '').match(/(?:^|&)token=([^&]*)/);
  if (m) _tok = decodeURIComponent(m[1]);
})();

function showError(error) {
  // Global rejection handler used by many actions (.catch(showError)). It was
  // referenced but never defined, so any failing action threw
  // "showError is not defined" (e.g. the mid-roll "Chèn vào kịch bản" flow).
  const message = (error && error.message) ? error.message : String(error || 'Đã xảy ra lỗi.');
  try { console.error('showError:', error); } catch (ignored) {}
  let toast = document.getElementById('globalErrorToast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'globalErrorToast';
    toast.setAttribute('role', 'alert');
    toast.style.cssText = 'position:fixed;right:24px;bottom:24px;max-width:520px;z-index:2147483647;'
      + 'background:#b3261e;color:#fff;padding:12px 16px;border-radius:8px;'
      + 'box-shadow:0 4px 16px rgba(0,0,0,.35);font:14px/1.4 system-ui,sans-serif;'
      + 'cursor:pointer;white-space:pre-wrap;word-break:break-word;';
    toast.addEventListener('click', () => toast.remove());
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  clearTimeout(showError._timer);
  showError._timer = setTimeout(() => { if (toast && toast.parentNode) toast.remove(); }, 8000);
}

// The gold busy bar appears only for requests slower than 300ms, so fast status polls never flash it.
let _busyCount = 0, _busyTimer = 0;
function setBusy(delta) {
  _busyCount = Math.max(0, _busyCount + delta);
  const root = document.documentElement;
  if (!_busyCount) { clearTimeout(_busyTimer); _busyTimer = 0; delete root.dataset.busy; return; }
  if (!_busyTimer && root.dataset.busy !== '1') { _busyTimer = setTimeout(() => { _busyTimer = 0; if (_busyCount) root.dataset.busy = '1'; }, 300); }
}

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  if (_tok) opts.headers['Authorization'] = 'Bearer ' + _tok;
  setBusy(1);
  try {
    const res = await fetch(path, opts);
    const text = await res.text();
    const data = text ? JSON.parse(text) : {};
    if (!res.ok) { throw new Error(data.error_vi || data.error || ('HTTP ' + res.status)); }
    return data;
  } finally {
    setBusy(-1);
  }
}

async function loadBrand() {
  const brand = await api('GET', '/api/brand');
  $('brandName').value = brand.name;
  $('brandPreviewName').textContent = brand.name;
  $('brandPreview').src = brand.logo_url + '?v=' + Date.now();
}
$('ttsTestBtn').onclick = async () => {
  const select = document.querySelector('#createForm select[name="tts_provider"]');
  const provider = select ? select.value : 'edge';
  const msg = $('ttsTestMsg');
  msg.textContent = 'Đang kiểm tra…';
  try {
    const data = await api('GET', '/api/tts/test?provider=' + encodeURIComponent(provider));
    msg.textContent = (data.ok ? '✓ ' : '✗ ') + (data.detail || (data.ok ? 'OK' : 'Không kết nối được'));
  } catch (error) {
    msg.textContent = '✗ ' + error.message;
  }
};
// 1-click auto-install for the local VieNeu-TTS package (shown only when selected).
(function setupTtsInstall() {
  const select = document.querySelector('#createForm select[name="tts_provider"]');
  const btn = $('ttsInstallBtn');
  if (!select || !btn) return;
  const msg = $('ttsInstallMsg');
  const sync = () => { btn.hidden = select.value !== 'vieneu'; if (btn.hidden && msg) msg.textContent = ''; };
  select.addEventListener('change', sync);
  sync();
  btn.onclick = async () => {
    btn.disabled = true;
    if (msg) msg.textContent = 'Đang cài VieNeu-TTS… Trên Windows/Python 3.14+, ứng dụng sẽ tự tạo môi trường Python 3.12 riêng để dùng wheel nhị phân (không biên dịch C++).';
    try {
      await api('POST', '/api/system/install-package', { package: 'vieneu' });
      let done = false;
      for (let i = 0; i < 80 && !done; i++) {
        await new Promise(r => setTimeout(r, 3000));
        const st = await api('GET', '/api/system/install-status?package=vieneu');
        if (st.status === 'done' || st.status === 'installed') {
          done = true;
          if (msg) msg.textContent = '✓ Cài đặt thành công! VieNeu-TTS đã sẵn sàng.';
          btn.hidden = true;
        } else if (st.status === 'error') {
          done = true;
          if (msg) msg.textContent = '✗ ' + (st.detail || 'Cài đặt thất bại');
        }
      }
      if (!done && msg) msg.textContent = 'Vẫn đang cài… bấm "Kiểm tra kết nối" sau ít phút.';
    } catch (error) {
      if (msg) msg.textContent = '✗ ' + error.message;
    } finally {
      btn.disabled = false;
    }
  };
})();
// Single source for the watermark method choices shown in every form.
const WATERMARK_METHOD_INFO = [
  {value: 'propainter', label: 'ProPainter (AI, cần GPU)', help: 'AI vẽ lại vùng watermark từ các khung hình lân cận. Sạch nhất nhưng rất nặng: cần GPU NVIDIA, máy chỉ có CPU có thể mất hàng chục giờ cho video 10 phút. Cần cài ProPainter.'},
  {value: 'delogo', label: 'Delogo (FFmpeg, nhanh)', help: 'FFmpeg lấp vùng watermark bằng màu nội suy từ viền xung quanh. Vài phút trên máy yếu, không cần GPU. Hợp với logo nhỏ đứng yên; vùng lớn sẽ thành mảng nhoè. Mask theo từng khung được gộp thành một vùng cố định.'},
  {value: 'blur', label: 'Làm mờ (FFmpeg, nhanh nhất)', help: 'FFmpeg phủ lớp làm mờ đúng theo hình mask. Nhanh nhất, không cần GPU. Không xoá hẳn mà che cho khó đọc; hợp với watermark chữ hoặc hình dạng phức tạp. Mask theo từng khung được gộp thành một vùng cố định.'},
];
function watermarkMethodInfo(value) { return WATERMARK_METHOD_INFO.find(item => item.value === value) || WATERMARK_METHOD_INFO[0]; }
function syncWatermarkMethodHelp(select) {
  const help = select.nextElementSibling;
  if (help && help.classList.contains('wm-method-help')) help.textContent = watermarkMethodInfo(select.value).help;
}
function setWatermarkMethod(id, value) { const select = $(id); select.value = watermarkMethodInfo(value).value; syncWatermarkMethodHelp(select); }
document.querySelectorAll('select.wm-method').forEach(select => {
  select.replaceChildren(...WATERMARK_METHOD_INFO.map(info => new Option(info.label, info.value)));
  select.addEventListener('change', () => syncWatermarkMethodHelp(select));
  syncWatermarkMethodHelp(select);
});
(function setupProPainterInstall() {
  const btn = $('propainterInstallBtn');
  const msg = $('propainterInstallMsg');
  if (!btn) return;
  const refresh = async () => {
    const st = await api('GET', '/api/system/install-status?package=propainter');
    if (st.status === 'installed' || st.status === 'done') {
      btn.hidden = true;
      if (msg) msg.textContent = '✓ ProPainter đã sẵn sàng.';
      return true;
    }
    return false;
  };
  refresh().catch(() => {});
  btn.onclick = async () => {
    btn.disabled = true;
    if (msg) msg.textContent = 'Đang tải bản ProPainter đã ghim và kiểm tra checksum model…';
    try {
      await api('POST', '/api/system/install-package', { package: 'propainter' });
      let done = false;
      for (let i = 0; i < 240 && !done; i++) {
        await new Promise(r => setTimeout(r, 3000));
        const st = await api('GET', '/api/system/install-status?package=propainter');
        if (st.status === 'done' || st.status === 'installed') {
          done = true;
          btn.hidden = true;
          if (msg) msg.textContent = '✓ ProPainter đã cài và sẵn sàng xoá watermark.';
        } else if (st.status === 'error') {
          done = true;
          if (msg) msg.textContent = '✗ ' + (st.detail || 'Cài đặt thất bại');
        }
      }
      if (!done && msg) msg.textContent = 'Vẫn đang cài ProPainter; bạn có thể tiếp tục chườ trong màn hình này.';
    } catch (error) {
      if (msg) msg.textContent = '✗ ' + error.message;
    } finally {
      btn.disabled = false;
    }
  };
})();
$('saveBrandName').onclick = async () => {
  try {
    const brand = await api('POST', '/api/brand', {name: $('brandName').value});
    $('brandPreviewName').textContent = brand.name;
    $('brandMsg').textContent = 'Đã lưu tên kênh cho các lần dựng tiếp theo.';
  } catch (error) { $('brandMsg').textContent = error.message; }
};
$('saveBrandLogo').onclick = async () => {
  const file = $('brandLogo').files[0];
  if (!file || file.type !== 'image/png' || file.size > 2000000) {
    $('brandMsg').textContent = 'Chọn PNG nền trong suốt, tối đa 2 MB.'; return;
  }
  try {
    const headers = {'Content-Type':'image/png'};
    if (_tok) headers.Authorization = 'Bearer ' + _tok;
    const response = await fetch('/api/brand/logo', {method:'POST', headers, body:file});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error_vi || result.error);
    $('brandPreview').src = result.logo_url + '?v=' + Date.now();
    $('brandMsg').textContent = 'Đã lưu logo cho các lần dựng tiếp theo.';
  } catch (error) { $('brandMsg').textContent = error.message; }
};
$('brandRenderBtn').onclick = async () => {
  if (!current) return;
  $('brandRenderBtn').disabled = true;
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/brand-render',
      {top_band:Number($('brandTopBand').value), bottom_band:Number($('brandBottomBand').value)});
    $('brandMsg').textContent = 'Đang dựng lại video với tên, logo và dải che đã chọn.';
    loadStatus();
  } catch (error) {
    $('brandMsg').textContent = error.message;
    $('brandRenderBtn').disabled = false;
  }
};
loadBrand().catch(error => { $('brandMsg').textContent = error.message; });

function uploadVideo(id, file, retry = false) {
  if (!file || !/[.]mp4$/i.test(file.name) || file.size <= 0) return Promise.reject(new Error('Chọn video MP4 không rỗng.'));
  const progress = $('uploadProgress');
  if (!retry) { progress.hidden = false; progress.value = 0; }
  $(retry ? 'sourceRetryMsg' : 'createMsg').textContent = 'Đang import video ' + file.name + '…';
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs/' + encodeURIComponent(id) + '/source');
    if (_tok) xhr.setRequestHeader('Authorization', 'Bearer ' + _tok);
    xhr.setRequestHeader('X-Source-Name', encodeURIComponent(file.name));
    xhr.setRequestHeader('Content-Type', 'video/mp4');
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) {
        const percent = Math.round(event.loaded * 100 / event.total);
        if (retry) $('sourceRetryMsg').textContent = 'Đang import video ' + percent + '%…';
        else progress.value = percent;
      }
    };
    xhr.onload = () => {
      progress.hidden = true;
      let result = {};
      try { result = JSON.parse(xhr.responseText || '{}'); } catch (_) { /* malformed response */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(result);
      else reject(new Error(result.error_vi || result.error || 'Import thất bại (HTTP ' + xhr.status + ')'));
    };
    xhr.onerror = () => { progress.hidden = true; reject(new Error('Không kết nối được khi import video.')); };
    xhr.send(file);
  });
}

async function authFetch(url) {
  if (!_tok) return url;
  try {
    const resp = await fetch(url, { headers: { Authorization: 'Bearer ' + _tok } });
    if (!resp.ok) return url;
    return URL.createObjectURL(await resp.blob());
  } catch (e) { return url; }
}

function clearThumbnailObjectUrls() {
  for (const url of thumbnailObjectUrls) URL.revokeObjectURL(url);
  thumbnailObjectUrls = [];
}

function clearMediaObjectUrls() {
  for (const url of mediaObjectUrls) URL.revokeObjectURL(url);
  mediaObjectUrls = [];
}

async function downloadHighlight(href, candidateId) {
  const resolved = await authFetch(href);
  const link = document.createElement('a');
  link.href = resolved;
  link.download = 'highlight-' + candidateId + '.mp4';
  document.body.appendChild(link);
  link.click();
  link.remove();
  if (resolved.startsWith('blob:')) setTimeout(() => URL.revokeObjectURL(resolved), 1000);
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value) || 0);
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.floor(seconds % 60);
  return String(minutes).padStart(2, '0') + ':' + String(remainder).padStart(2, '0');
}

function setMediaState(message, kind = 'empty') {
  const state = $('mediaState');
  state.hidden = !message;
  state.textContent = message || '';
  state.dataset.kind = kind;
}

function activateMediaFilter(filter) {
  mediaFilter = filter;
  document.querySelectorAll('[data-media-filter]').forEach(tab => {
    tab.setAttribute('aria-selected', String(tab.dataset.mediaFilter === filter));
  });
  renderMediaRows();
}

function makeChip(text, className) {
  const chip = document.createElement('span');
  chip.className = className;
  chip.textContent = text;
  return chip;
}

function makeSeekHandler(seconds) {
  return () => seekSource(seconds);
}

function renderMediaRows() {
  const results = $('mediaResults');
  results.replaceChildren();
  let rows = mediaFilter === 'transcript' ? mediaExplorerData.transcript : mediaExplorerData.shots;
  if (mediaFilter === 'highlights') rows = mediaExplorerData.highlights;
  if (!rows.length) {
    const searching = $('mediaSearch').value.trim();
    setMediaState(searching ? 'Không tìm thấy kết quả phù hợp. Hãy thử tìm kiếm rộng hơn.' : 'Chưa có dữ liệu.', 'empty');
    return;
  }
  setMediaState('');
  for (const row of rows) {
    if (mediaFilter === 'highlights') {
      const card = document.createElement('article');
      card.className = 'highlight-card';
      const heading = document.createElement('strong');
      heading.textContent = row.title || 'Khoảnh khắc nổi bật';
      const meta = document.createElement('div'); meta.className = 'result-meta';
      meta.append(makeChip(formatTime(row.start_seconds ?? row.start) + '–' + formatTime(row.end_seconds ?? row.end), 'time-chip'), makeChip('Nổi bật', 'kind-chip'));
      const reason = document.createElement('p'); reason.textContent = row.reason || 'Đoạn gợi ý';
      const actions = document.createElement('div'); actions.className = 'row';
      const preview = document.createElement('button'); preview.type = 'button'; preview.textContent = '▶ Xem trước'; preview.onclick = makeSeekHandler(row.start_seconds ?? row.start);
      const download = document.createElement('button'); download.type = 'button'; download.textContent = '↓ 9:16 MP4'; download.setAttribute('aria-label', 'Xuất ' + heading.textContent + ' dạng MP4 dọc'); download.onclick = () => downloadHighlight(row.export_href, row.id).catch(showError);
      actions.append(preview, download); card.append(heading, meta, reason, actions); results.append(card); continue;
    }
    const item = document.createElement('article');
    const isTranscript = mediaFilter === 'transcript';
    item.className = 'media-result' + (isTranscript ? ' transcript-result' : ' scene-result');
    item.dataset.start = String(Number(row.start_seconds) || 0);
    item.dataset.end = String(Number(row.end_seconds) || Number(row.start_seconds) || 0);
    if (isTranscript) item.dataset.transcriptId = String(row.id ?? row.start_seconds);
    if (!isTranscript) {
      const img = document.createElement('img');
      img.alt = 'Ảnh thu nhỏ cảnh tại ' + formatTime(row.start_seconds);
      img.loading = 'lazy';
      authFetch(row.thumbnail_href).then(resolved => { if (resolved.startsWith('blob:')) mediaObjectUrls.push(resolved); img.src = resolved; }).catch(error => { img.alt = 'Ảnh thu nhỏ không khả dụng: ' + error.message; });
      item.append(img);
    }
    const detail = document.createElement('div');
    const meta = document.createElement('div'); meta.className = 'result-meta';
    meta.append(makeChip(formatTime(row.start_seconds) + '–' + formatTime(row.end_seconds), 'time-chip'));
    if (isTranscript && row.speaker) meta.append(makeChip(row.speaker, 'speaker-chip'));
    meta.append(makeChip(isTranscript ? 'Lời thoại' : 'Cảnh', 'kind-chip'));
    if (row.semantic_score !== null && row.semantic_score !== undefined) {
      meta.append(makeChip('Ngữ nghĩa ' + Number(row.semantic_score).toFixed(2), 'kind-chip'));
    }
    const copy = document.createElement('div'); copy.className = 'result-copy';
    copy.textContent = isTranscript ? row.text : (row.visual_description || row.label || 'Cảnh đã lập chỉ mục');
    if (!isTranscript) {
      for (const value of [
        ...(row.visual_tags || []).slice(0, 3),
        ...(row.visual_people || []).slice(0, 2),
        ...(row.visual_actions || []).slice(0, 2),
        ...(row.person_tracks || []).slice(0, 3),
      ]) meta.append(makeChip(value, 'kind-chip'));
    }
    const seek = document.createElement('button'); seek.type = 'button'; seek.className = 'seek-button'; seek.textContent = '▶ Nhảy tới ' + formatTime(row.start_seconds); seek.setAttribute('aria-label', 'Tua video nguồn tới ' + formatTime(row.start_seconds)); seek.onclick = event => { event.stopPropagation(); seekSource(row.start_seconds); };
    detail.append(meta, copy, seek);
    if (!isTranscript && row.id) {
      const similar = document.createElement('button');
      similar.type = 'button';
      similar.textContent = 'Tìm cảnh tương tự';
      similar.onclick = event => { event.stopPropagation(); findSimilarScenes(row.id).catch(showError); };
      detail.append(similar);
    }
    item.append(detail);
    if (isTranscript) {
      item.tabIndex = 0; item.setAttribute('role', 'button'); item.setAttribute('aria-label', 'Phát lời thoại từ ' + formatTime(row.start_seconds) + ': ' + row.text);
      item.onclick = makeSeekHandler(row.start_seconds);
      item.onkeydown = event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); seekSource(row.start_seconds); } };
    }
    results.append(item);
  }
  syncActiveTranscript($('sourceVideo').currentTime, false);
}

function syncActiveTranscript(currentTime, allowScroll = true) {
  if (mediaFilter !== 'transcript') return;
  const rows = Array.from($('mediaResults').querySelectorAll('[data-transcript-id]'));
  const active = rows.find(row => currentTime >= Number(row.dataset.start) && currentTime < Number(row.dataset.end)) || null;
  const nextId = active ? active.dataset.transcriptId : null;
  if (nextId === activeTranscriptId) return;
  activeTranscriptId = nextId;
  rows.forEach(row => { const selected = row === active; row.classList.toggle('active', selected); row.setAttribute('aria-current', selected ? 'true' : 'false'); });
  if (active && allowScroll && Date.now() > manualTranscriptScrollUntil) active.scrollIntoView({block: 'nearest', behavior: 'smooth'});
}

async function loadHighlights() {
  const box = $('highlightResults');
  box.innerHTML = '<div class="state-panel"><span class="mrf-loading">Đang tải khoảnh khắc nổi bật…</span></div>';
  try {
    const data = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/highlights');
    mediaExplorerData.highlights = Array.isArray(data.highlights) ? data.highlights : [];
    $('highlightCount').textContent = mediaExplorerData.highlights.length ? '(' + mediaExplorerData.highlights.length + ')' : '';
    box.replaceChildren();
    if (!mediaExplorerData.highlights.length) { box.innerHTML = '<div class="state-panel">Chưa phát hiện khoảnh khắc nổi bật nào.</div>'; }
    for (const item of mediaExplorerData.highlights.slice(0, 4)) {
      const card = document.createElement('article'); card.className = 'highlight-card';
      const title = document.createElement('strong'); title.textContent = item.title || 'Khoảnh khắc nổi bật';
      const meta = document.createElement('div'); meta.className = 'result-meta'; meta.append(makeChip(formatTime(item.start_seconds ?? item.start) + '–' + formatTime(item.end_seconds ?? item.end), 'time-chip'));
      const reason = document.createElement('p'); reason.textContent = item.reason || 'Đoạn gợi ý';
      const actions = document.createElement('div'); actions.className = 'row';
      const play = document.createElement('button'); play.type = 'button'; play.textContent = '▶ Xem trước'; play.onclick = makeSeekHandler(item.start_seconds ?? item.start);
      const download = document.createElement('button'); download.type = 'button'; download.textContent = '↓ Xuất'; download.onclick = () => downloadHighlight(item.export_href, item.id).catch(showError);
      actions.append(play, download); card.append(title, meta, reason, actions); box.append(card);
    }
    if (mediaFilter === 'highlights') renderMediaRows();
  } catch (error) {
    mediaExplorerData.highlights = [];
    box.innerHTML = '<div class="state-panel">Khoảnh khắc nổi bật không khả dụng: ' + error.message + '</div>';
  }
}

function seekSource(seconds) {
  const video = $('sourceVideo');
  video.currentTime = Math.max(0, Number(seconds) || 0);
  syncActiveTranscript(video.currentTime);
  video.play().catch(() => {});
}

function seekSourceWhenReady(seconds) {
  const video = $('sourceVideo');
  if (video.readyState >= 1) { seekSource(seconds); return; }
  video.addEventListener('loadedmetadata', () => seekSource(seconds), {once:true});
}

async function downloadTranscript(href) {
  const resolved = await authFetch(href);
  const link = document.createElement('a'); link.href = resolved; link.download = 'transcript.vtt'; document.body.appendChild(link); link.click(); link.remove();
  if (resolved.startsWith('blob:')) setTimeout(() => URL.revokeObjectURL(resolved), 1000);
}

async function askVideo() {
  const question = $('chatQuestion').value.trim();
  if (!question) { $('chatAnswer').textContent = 'Hãy nhập câu hỏi trước.'; $('chatQuestion').focus(); return; }
  $('chatAskBtn').disabled = true; $('chatAnswer').textContent = 'Đang tìm trong lời thoại đã lập chỉ mục…';
  try {
    const data = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/chat', {question});
    const citations = Array.isArray(data.citations) ? data.citations.length : 0;
    $('chatAnswer').textContent = (data.answer || 'Không tìm thấy câu trả lời.') + (citations ? ' · ' + citations + ' đoạn trích dẫn' : '');
  } finally { $('chatAskBtn').disabled = false; }
}

// --- Client-side caption chunking -----------------------------------------
// Whisper-style transcript rows can span ~30s and hundreds of characters. Shown
// as one cue they cover 30-40% of the frame. We wrap each row to <=2 lines of
// ~42 characters and split long rows into time-sliced sub-cues so only a short,
// readable caption is on screen at any moment.
const CAPTION_MAX_CPL = 42;   // characters per line (broadcast/web standard)
const CAPTION_MAX_LINES = 2;  // lines visible at once
const CAPTION_MIN_CUE_SECONDS = 1.2;
// A sentence shorter than this is merged with the next one (when they still fit
// in CAPTION_MAX_LINES) so a caption never flashes a tiny fragment on its own.
const CAPTION_MERGE_MIN_CHARS = 25;

// Greedy word-wrap into lines of at most maxCpl characters. Words longer than
// maxCpl are hard-split so a single token can never overflow the frame.
function wrapCaptionLines(text, maxCpl) {
  const words = String(text || '').replace(/\\s+/g, ' ').trim().split(' ').filter(Boolean);
  const lines = [];
  let line = '';
  for (let word of words) {
    while (word.length > maxCpl) {
      if (line) { lines.push(line); line = ''; }
      lines.push(word.slice(0, maxCpl));
      word = word.slice(maxCpl);
    }
    if (!line) line = word;
    else if ((line + ' ' + word).length <= maxCpl) line += ' ' + word;
    else { lines.push(line); line = word; }
  }
  if (line) lines.push(line);
  return lines;
}

// Split a block of text into sentences at ., ?, !, … (and their repeats),
// keeping the terminal punctuation attached. This lets each caption cue align
// with a complete sentence instead of an arbitrary character window. Text with
// no sentence-ending punctuation comes back as a single sentence.
function splitIntoSentences(text) {
  const normalized = String(text || '').replace(/\\s+/g, ' ').trim();
  if (!normalized) return [];
  const matches = normalized.match(/[^.!?…]+(?:[.!?…]+["'”’)\\]]*|$)/g);
  const sentences = (matches || [normalized]).map(s => s.trim()).filter(Boolean);
  return sentences.length ? sentences : [normalized];
}

// Greedily merge a short sentence (< CAPTION_MERGE_MIN_CHARS) into the next one
// as long as the combined text still wraps within CAPTION_MAX_LINES, so tiny
// fragments ("Vâng." / "Được.") ride along with the following sentence instead
// of flashing as their own cue.
function mergeShortSentences(sentences) {
  const merged = [];
  let buffer = '';
  const fitsTwoLines = (text) => wrapCaptionLines(text, CAPTION_MAX_CPL).length <= CAPTION_MAX_LINES;
  for (const sentence of sentences) {
    if (!buffer) { buffer = sentence; continue; }
    const combined = buffer + ' ' + sentence;
    // Merge only while the running buffer is still short and the result fits.
    if (buffer.length < CAPTION_MERGE_MIN_CHARS && fitsTwoLines(combined)) {
      buffer = combined;
    } else {
      merged.push(buffer);
      buffer = sentence;
    }
  }
  if (buffer) merged.push(buffer);
  return merged;
}

// Turn one transcript row into 1..N sub-cues of <=CAPTION_MAX_LINES lines. We
// first split the row into whole sentences, then wrap each sentence to
// CAPTION_MAX_CPL and break it into <=2-line blocks. Every block's [start,end]
// is interpolated within the row proportionally to its character length, so a
// cue both aligns with a sentence and tracks the spoken pace.
function chunkTranscriptRow(row) {
  const start = Number(row.start_seconds);
  let end = Number(row.end_seconds);
  const text = String(row.text || '').trim();
  if (!text || !Number.isFinite(start)) return [];
  if (!Number.isFinite(end) || end <= start) end = start + CAPTION_MIN_CUE_SECONDS;
  // Sentence first (merging tiny fragments), then wrap + split into <=2 lines.
  const blocks = [];
  for (const sentence of mergeShortSentences(splitIntoSentences(text))) {
    const lines = wrapCaptionLines(sentence, CAPTION_MAX_CPL);
    for (let i = 0; i < lines.length; i += CAPTION_MAX_LINES) {
      blocks.push(lines.slice(i, i + CAPTION_MAX_LINES).join('\\n'));
    }
  }
  if (!blocks.length) return [];
  // Prefer real per-word timings when the row carries them: give each block the
  // start/end of the actual spoken words it holds, so a cue never drifts across a
  // pause. Fall back to interpolating by character length for legacy rows.
  const wordTimes = Array.isArray(row.words) ? row.words.filter(w =>
    w && Number.isFinite(Number(w.start)) && Number.isFinite(Number(w.end))) : [];
  if (wordTimes.length) {
    const NL = String.fromCharCode(10);
    const wordCues = [];
    let wi = 0;
    for (let i = 0; i < blocks.length; i++) {
      const tokens = blocks[i].split(NL).join(' ').split(' ').filter(Boolean).length || 1;
      const firstWord = wordTimes[Math.min(wi, wordTimes.length - 1)];
      const lastWord = wordTimes[Math.min(wi + tokens - 1, wordTimes.length - 1)];
      let cueStart = i === 0 ? start : Number(firstWord.start);
      let cueEnd = i === blocks.length - 1 ? end : Number(lastWord.end);
      if (!(cueEnd > cueStart)) cueEnd = Math.min(end, cueStart + CAPTION_MIN_CUE_SECONDS);
      wordCues.push({ start: cueStart, end: cueEnd, text: blocks[i] });
      wi += tokens;
    }
    return wordCues;
  }
  const totalChars = blocks.reduce((sum, block) => sum + block.replace(/\\n/g, '').length, 0) || 1;
  const span = end - start;
  const cues = [];
  let cursor = start;
  for (let i = 0; i < blocks.length; i++) {
    const block = blocks[i];
    const chars = block.replace(/\\n/g, '').length;
    let cueEnd = i === blocks.length - 1 ? end : cursor + span * (chars / totalChars);
    if (cueEnd - cursor < 0.2) cueEnd = Math.min(end, cursor + 0.2);
    cues.push({ start: cursor, end: cueEnd, text: block });
    cursor = cueEnd;
  }
  return cues;
}

// Rebuild the source video's caption track from transcript rows. Adds VTTCues
// directly (no raw .vtt blob) so we control wrapping and per-cue timing.
function renderSourceCaptions(trackEl, transcript) {
  const video = $('sourceVideo');
  const CueCtor = window.VTTCue || window.TextTrackCue;
  // Remove any previously built cues so re-opening a project starts clean.
  for (const existing of Array.from(video.textTracks || [])) {
    if (existing.cues) for (const cue of Array.from(existing.cues)) { try { existing.removeCue(cue); } catch (e) {} }
  }
  const tt = trackEl.track;
  if (!tt || !CueCtor) return;
  tt.mode = 'hidden';
  if (!Array.isArray(transcript) || !transcript.length) { tt.mode = 'disabled'; return; }
  for (const row of transcript) {
    for (const cue of chunkTranscriptRow(row)) {
      if (!(cue.end > cue.start)) continue;
      try { tt.addCue(new CueCtor(cue.start, cue.end, cue.text)); } catch (e) {}
    }
  }
  tt.mode = 'showing';
}

async function renderMediaExplorer(query = '') {
  const card = $('mediaExplorerCard'); const results = $('mediaResults'); const message = $('mediaMsg');
  clearMediaObjectUrls(); card.style.display = ''; results.setAttribute('aria-busy', 'true'); setMediaState('Đang tải chỉ mục tư liệu…', 'loading');
  let data;
  try {
    const suffix = query ? ('?q=' + encodeURIComponent(query)) : '';
    data = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/media-explorer' + suffix);
  } catch (error) {
    results.replaceChildren(); results.setAttribute('aria-busy', 'false'); setMediaState('Không tải được Trình khám phá tư liệu: ' + error.message, 'error'); message.textContent = 'Lỗi tải Trình khám phá tư liệu'; return;
  }
  if (!data.present || !data.media_href) {
    results.replaceChildren();
    results.setAttribute('aria-busy', 'false');
    const video = $('sourceVideo');
    if (video.dataset.src) { video.pause(); video.removeAttribute('src'); video.dataset.src = ''; video.load(); }
    setMediaState('Chưa có dữ liệu tư liệu cho project này. Hãy import video MP4 rồi chạy pipeline để lập chỉ mục lời thoại và cảnh.', 'empty');
    const jump = document.createElement('button'); jump.type = 'button'; jump.className = 'seek-button'; jump.style.marginTop = '10px'; jump.textContent = 'Tới bước import video';
    jump.onclick = () => {
      const retryCard = $('sourceRetryCard');
      if (retryCard && !retryCard.hidden) { retryCard.scrollIntoView({behavior:'smooth', block:'center'}); const retry = $('sourceRetry'); if (retry) retry.focus(); }
      else { openToolDialog('createPanel'); }
    };
    $('mediaState').appendChild(jump);
    return;
  }
  const exportBtn = $('mediaExportBtn');
  exportBtn.style.display = data.transcript_vtt_href ? '' : 'none'; exportBtn.onclick = data.transcript_vtt_href ? (() => downloadTranscript(data.transcript_vtt_href).catch(showError)) : null;
  const video = $('sourceVideo');
  if (video.dataset.src !== data.media_href) { video.dataset.src = data.media_href; video.src = data.media_href; }
  const track = $('sourceCaptions'); track.removeAttribute('src'); track.default = true;
  track.srclang = data.source_language || 'und';
  track.label = 'Lời thoại nguồn' + (data.source_language ? ' (' + data.source_language + ')' : '');
  mediaExplorerData.transcript = Array.isArray(data.transcript) ? data.transcript : [];
  mediaExplorerData.shots = Array.isArray(data.shots) ? data.shots : [];
  // Render captions from the transcript rows with client-side chunking so a
  // long (~30s) transcript block never dumps 6-7 lines over the frame. We build
  // the cues ourselves instead of loading the raw .vtt as a single track cue.
  renderSourceCaptions(track, mediaExplorerData.transcript);
  $('transcriptCount').textContent = '(' + mediaExplorerData.transcript.length + ')'; $('sceneCount').textContent = '(' + mediaExplorerData.shots.length + ')';
  results.setAttribute('aria-busy', 'false'); renderMediaRows(); loadHighlights();
  if (pendingLibrarySeek && pendingLibrarySeek.job_id === current) {
    const target = pendingLibrarySeek;
    pendingLibrarySeek = null;
    seekSourceWhenReady(target.start_seconds);
  }
}

async function findSimilarScenes(shotId) {
  const data = await api(
    'GET',
    '/api/jobs/' + encodeURIComponent(current) + '/shots/' + encodeURIComponent(shotId) + '/similar'
  );
  mediaFilter = 'scenes';
  mediaExplorerData.shots = Array.isArray(data.results) ? data.results : [];
  document.querySelectorAll('[data-media-filter]').forEach(tab => {
    tab.setAttribute('aria-selected', String(tab.dataset.mediaFilter === 'scenes'));
  });
  $('sceneCount').textContent = '(' + mediaExplorerData.shots.length + ')';
  renderMediaRows();
}

async function timelineAction(body) {
  await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/timeline', body);
  $('editorMsg').textContent = 'Đã cập nhật timeline; render/QA cũ đã được đánh dấu cần tạo lại.';
  await renderEditor();
  await loadStatus();
}

async function refreshSectionPreview() {
  if (!current || !$('sectionPreviewSelect').value) return;
  const index = $('sectionPreviewSelect').value;
  const state = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/previews/' + index);
  $('sectionPreviewBtn').disabled = !!state.running;
  $('sectionPreviewMsg').textContent = state.running ? 'Đang dựng phần ' + index + '…'
    : state.error ? state.error
    : state.href ? 'Bản xem nhanh đã sẵn sàng.' : 'Chưa có bản xem nhanh cho phần này.';
  $('sectionPreviewDownload').hidden = !state.href;
  $('sectionPreviewSrt').hidden = !state.srt_href;
  const video = $('sectionPreviewVideo');
  video.hidden = !state.href;
  if (state.href) {
    $('sectionPreviewDownload').href = state.href;
    $('sectionPreviewSrt').href = state.srt_href;
    if (video.dataset.src !== state.href) { video.dataset.src = state.href; video.src = state.href; }
  } else if (video.dataset.src) { video.pause(); video.removeAttribute('src'); video.load(); video.dataset.src = ''; }
}
$('sectionPreviewSelect').onchange = () => refreshSectionPreview().catch(showError);
// Collapsing the preview panel hides the <video> (and its controls) but does
// not stop playback, leaving audio running with no reachable pause button.
// Pause the preview whenever the panel is folded shut.
$('sectionPreviewPanel').addEventListener('toggle', () => {
  if (!$('sectionPreviewPanel').open) { const preview = $('sectionPreviewVideo'); if (preview && !preview.paused) preview.pause(); }
});
$('sectionPreviewBtn').onclick = async () => {
  if (!current) return;
  const index = $('sectionPreviewSelect').value;
  $('sectionPreviewBtn').disabled = true;
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/previews/' + index, {});
    $('sectionPreviewVideo').pause();
    $('sectionPreviewVideo').removeAttribute('src');
    $('sectionPreviewVideo').dataset.src = '';
    $('sectionPreviewVideo').hidden = true;
    $('sectionPreviewDownload').hidden = true;
    $('sectionPreviewSrt').hidden = true;
    $('sectionPreviewMsg').textContent = 'Đang dựng phần ' + index + '…';
    await loadStatus();
  } catch (error) { $('sectionPreviewMsg').textContent = error.message; $('sectionPreviewBtn').disabled = false; }
};

async function refreshHookTeaser() {
  if (!current) return;
  try {
    const data = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/hook');
    const download = $('hookDownload');
    const video = $('hookVideo');
    const msg = $('hookMsg');
    if (data.present && data.href) {
      if (download) { download.hidden = false; download.href = data.href; }
      if (video) {
        video.hidden = false;
        if (video.dataset.src !== data.href) { video.dataset.src = data.href; video.src = data.href; }
      }
      if (msg) {
        msg.textContent = data.meta
          ? ('Cảnh ' + data.meta.scene_index + ' (' + data.meta.duration_seconds + 's) · Điểm kịch tính: ' + Math.round((data.meta.dramatic_score || 0) * 100) + '%')
          : 'Đã có hook teaser.';
      }
    } else {
      if (download) download.hidden = true;
      if (video) {
        video.hidden = true;
        if (video.dataset.src) { video.pause(); video.removeAttribute('src'); video.dataset.src = ''; }
      }
      if (msg) msg.textContent = 'Chưa tạo hook teaser cho project này.';
    }
  } catch (e) {
    if ($('hookMsg')) $('hookMsg').textContent = e.message;
  }
}
if ($('hookTeaserPanel')) {
  $('hookTeaserPanel').addEventListener('toggle', () => {
    if ($('hookTeaserPanel').open) refreshHookTeaser().catch(showError);
    else { const v = $('hookVideo'); if (v && !v.paused) v.pause(); }
  });
}
if ($('buildHookBtn')) {
  $('buildHookBtn').onclick = async () => {
    if (!current) return;
    const btn = $('buildHookBtn');
    btn.disabled = true;
    $('hookMsg').textContent = 'Đang tìm cảnh kịch tính nhất và dựng hook teaser…';
    try {
      await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/hook', {});
      await refreshHookTeaser();
      $('hookMsg').textContent = 'Đã tạo hook teaser thành công!';
    } catch (e) {
      $('hookMsg').textContent = 'Lỗi tạo hook teaser: ' + e.message;
    } finally {
      btn.disabled = false;
    }
  };
}

// Keep the Timeline/Continuity card on screen even when there is nothing to
// edit yet; show a status line and a next action instead of hiding the card.
function setEditorState(message) {
  const note = $('editorState');
  if (!note) return;
  if (message) { note.textContent = message; note.hidden = false; }
  else { note.textContent = ''; note.hidden = true; }
}
async function renderEditor() {
  const card = $('editorCard');
  card.style.display = '';
  let timelineData, tracksData;
  try {
    [timelineData, tracksData] = await Promise.all([
      api('GET', '/api/jobs/' + encodeURIComponent(current) + '/timeline'),
      api('GET', '/api/jobs/' + encodeURIComponent(current) + '/person-tracks'),
    ]);
    currentTimelineData = timelineData;
  } catch (error) {
    $('timelineList').replaceChildren();
    $('continuityTracks').replaceChildren();
    setEditorState('Chưa có dòng thời gian để biên tập. Chạy pipeline tới bước kế hoạch cảnh để tạo các clip có thể chỉnh sửa.');
    return;
  }
  setEditorState('');
  const selector = $('sectionPreviewSelect');
  const previousSection = selector.value;
  const sectionRows = [...new Map((timelineData.clips || [])
    .filter(clip => Number.isInteger(clip.section_index) && clip.section_index > 0)
    .map(clip => [clip.section_index, clip.section || 'Phần ' + clip.section_index])).entries()];
  selector.replaceChildren();
  for (const [index, title] of sectionRows) {
    const option = document.createElement('option'); option.value = String(index);
    option.textContent = index + ' · ' + title; selector.appendChild(option);
  }
  if (sectionRows.some(([index]) => String(index) === previousSection)) selector.value = previousSection;
  $('sectionPreviewPanel').hidden = !sectionRows.length;
  if (sectionRows.length) refreshSectionPreview().catch(showError);
  const trackBox = $('continuityTracks');
  trackBox.replaceChildren();
  for (const track of tracksData.tracks || []) {
    const item = document.createElement('article');
    item.className = 'track-card';
    const title = document.createElement('strong');
    title.textContent = (track.alias || track.label) + (track.alias ? ' · ' + track.label : '');
    const meta = document.createElement('div');
    meta.className = 'muted';
    meta.textContent = 'độ tin cậy ' + Number(track.mean_confidence || 0).toFixed(2)
      + (track.ambiguous ? ' · không rõ ràng' : '')
      + ' · ' + ((track.appearances || []).length) + ' cảnh';
    const summary = document.createElement('div');
    summary.textContent = track.appearance_summary || track.description || '';
    const clothing = (track.traits || []).filter(x => x.trait_type === 'clothing').map(x => x.value);
    if (clothing.length) {
      const details = document.createElement('div');
      details.className = 'muted';
      details.textContent = 'Trang phục: ' + clothing.join(' → ');
      item.append(title, meta, summary, details);
    } else item.append(title, meta, summary);
    const aliasRow = document.createElement('div');
    aliasRow.className = 'row';
    const input = document.createElement('input');
    input.value = track.alias || '';
    input.placeholder = 'Tên thủ công cho ' + track.label;
    const save = document.createElement('button');
    save.type = 'button'; save.textContent = 'Lưu alias';
    save.onclick = async () => {
      await api(
        'POST',
        '/api/jobs/' + encodeURIComponent(current) + '/person-tracks/'
          + encodeURIComponent(track.label) + '/alias',
        {alias: input.value}
      );
      await renderEditor();
      await renderMediaExplorer($('mediaSearch').value.trim());
    };
    aliasRow.append(input, save); item.append(aliasRow); trackBox.append(item);
  }

  const list = $('timelineList');
  list.replaceChildren();
  for (const clip of timelineData.clips || []) {
    const item = document.createElement('article');
    item.className = 'timeline-card' + (clip.locked ? ' locked' : '');
    const source = clip.source_clip || {};
    const heading = document.createElement('strong');
    heading.textContent = '#' + (clip.timeline_index + 1) + ' · ' + (clip.section || 'Phần')
      + ' · ' + formatTime(source.start_seconds) + '–' + formatTime(source.end_seconds);
    const meta = document.createElement('div'); meta.className = 'muted';
    meta.textContent = 'Phần ' + clip.section_index + ' · cảnh ' + clip.shot_index + '/' + clip.shot_count
      + ' · đầu ra ' + formatTime(clip.start_seconds) + ' +' + Number(clip.duration_seconds || 0).toFixed(1) + 's';
    const controls = document.createElement('div'); controls.className = 'timeline-controls';

    // Three most-used clip actions stay inline; the rest move to an overflow menu.
    const preview = document.createElement('button');
    preview.type = 'button';
    preview.className = 'timeline-preview-btn';
    preview.dataset.clipIndex = String(clip.timeline_index);
    const isThisPlaying = activeTimelinePreviewIndex === clip.timeline_index && !$('sourceVideo').paused;
    preview.textContent = isThisPlaying ? '⏸ Tạm dừng' : '▶ Xem trước';
    preview.onclick = () => {
      const video = $('sourceVideo');
      if (activeTimelinePreviewIndex === clip.timeline_index && !video.paused) {
        video.pause();
        preview.textContent = '▶ Xem trước';
        activeTimelinePreviewIndex = -1;
      } else {
        activeTimelinePreviewIndex = clip.timeline_index;
        seekSource(source.start_seconds);
        document.querySelectorAll('.timeline-preview-btn').forEach(btn => {
          btn.textContent = (Number(btn.dataset.clipIndex) === clip.timeline_index) ? '⏸ Tạm dừng' : '▶ Xem trước';
        });
      }
    };

    const trim = document.createElement('button'); trim.type = 'button'; trim.textContent = 'Cắt';
    trim.onclick = () => {
      const start = prompt('Mốc đầu nguồn (giây)', String(source.start_seconds ?? 0));
      if (start === null) return;
      const end = prompt('Mốc cuối nguồn (giây)', String(source.end_seconds ?? 0));
      if (end === null) return;
      timelineAction({action:'trim', clip_index:clip.timeline_index, start_seconds:Number(start), end_seconds:Number(end)}).catch(showError);
    };

    const lock = document.createElement('button'); lock.type = 'button';
    lock.textContent = clip.locked ? 'Mở khóa' : 'Khóa';
    lock.onclick = () => timelineAction({action:'lock', clip_index:clip.timeline_index, locked:!clip.locked}).catch(showError);

    // Secondary actions live behind a "⋯ Thêm" overflow menu (native <details>).
    const replace = document.createElement('button'); replace.type = 'button'; replace.textContent = 'Thay cảnh';
    replace.onclick = () => {
      const shot = prompt('Shot ID thay thế');
      if (!shot) return;
      timelineAction({action:'replace', clip_index:clip.timeline_index, shot_id:Number(shot)}).catch(showError);
    };

    const broll = document.createElement('button'); broll.type = 'button'; broll.textContent = 'B-roll';
    broll.onclick = async () => {
      const data = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/broll/' + clip.timeline_index, {apply:false});
      const best = (data.suggestions || [])[0];
      if (!best) { $('editorMsg').textContent = 'Không có B-roll thay thế phù hợp.'; return; }
      if (confirm('Thay bằng shot ' + best.shot_id + ' · semantic ' + Number(best.semantic_score).toFixed(2) + '?')) {
        await timelineAction({action:'replace', clip_index:clip.timeline_index, shot_id:best.shot_id, notes:'B-roll suggestion'});
      }
    };

    const up = document.createElement('button'); up.type = 'button'; up.textContent = 'Chuyển lên ↑';
    up.disabled = clip.timeline_index <= 0;
    up.onclick = () => timelineAction({action:'reorder', from_index:clip.timeline_index, to_index:clip.timeline_index - 1}).catch(showError);

    const down = document.createElement('button'); down.type = 'button'; down.textContent = 'Chuyển xuống ↓';
    down.disabled = clip.timeline_index >= (timelineData.clips || []).length - 1;
    down.onclick = () => timelineAction({action:'reorder', from_index:clip.timeline_index, to_index:clip.timeline_index + 1}).catch(showError);

    const regen = document.createElement('button'); regen.type = 'button'; regen.textContent = 'Dựng lại phần';
    regen.onclick = async () => {
      const instruction = prompt('Yêu cầu cho phần hình ảnh này', 'more relevant visuals');
      if (instruction === null) return;
      await api(
        'POST',
        '/api/jobs/' + encodeURIComponent(current) + '/sections/' + clip.section_index + '/regenerate',
        {instruction}
      );
      $('editorMsg').textContent = 'Đã dựng lại hình ảnh cho phần ' + clip.section_index + '.';
      await renderEditor(); await loadStatus();
    };

    const more = document.createElement('details'); more.className = 'clip-more';
    const moreSummary = document.createElement('summary'); moreSummary.textContent = '⋯ Thêm';
    moreSummary.setAttribute('aria-label', 'Thêm thao tác cho clip');
    const moreMenu = document.createElement('div'); moreMenu.className = 'clip-more-menu';
    moreMenu.append(replace, broll, up, down, regen);
    // Close the menu after choosing an action or clicking outside.
    moreMenu.addEventListener('click', () => { more.open = false; });
    more.append(moreSummary, moreMenu);

    controls.append(preview, trim, lock, more);
    item.append(heading, meta, controls); list.append(item);
  }
}

$('autoBrollBtn').onclick = async () => {
  const data = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/broll/0', {apply:true});
  $('editorMsg').textContent = data.replacements?.length
    ? 'Đã thay ' + data.replacements.length + ' clip B-roll lặp/yếu.'
    : 'Không có clip lặp/yếu cần thay.';
  await renderEditor(); await loadStatus();
};

async function renderThumbnails() {
  const card = $('thumbnailCard');
  const grid = $('thumbnailGrid');
  const message = $('thumbnailMsg');
  const variantGrid = $('thumbnailVariantGrid');
  clearThumbnailObjectUrls();
  variantGrid.replaceChildren();

  let result;
  try {
    result = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/thumbnails');
  } catch (e) {
    card.style.display = '';
    grid.replaceChildren();
    message.textContent = 'Lỗi tải ảnh bìa: ' + e.message;
    return;
  }

  const doc = result.thumbnails || {};
  const candidates = Array.isArray(doc.candidates) ? doc.candidates : [];
  if (!result.present || !candidates.length) {
    card.style.display = 'none';
    grid.replaceChildren();
    message.textContent = '';
    return;
  }

  card.style.display = '';
  grid.replaceChildren();
  message.textContent = '';
  const edits = result.edits || {};
  const headlineField = $('thumbnailHeadline');
  if (headlineField.dataset.job !== current) {
    headlineField.value = edits.headline || '';
    $('thumbnailChannel').value = edits.channel_name || result.channel_name || '';
    headlineField.dataset.job = current;
  }

  for (const candidate of candidates) {
    if (!candidate || typeof candidate.file !== 'string') continue;

    const item = document.createElement('div');
    item.className = 'thumb-item' + (candidate.file === doc.primary_candidate ? ' selected' : '');

    const img = document.createElement('img');
    img.alt = 'Ảnh bìa ứng viên ' + (candidate.index || '');
    const artifactUrl = '/api/jobs/' + encodeURIComponent(current)
      + '/artifacts/' + encodeURIComponent(candidate.file);
    const resolved = await authFetch(artifactUrl);
    if (resolved.startsWith('blob:')) thumbnailObjectUrls.push(resolved);
    img.src = resolved;

    const meta = document.createElement('div');
    meta.className = 'muted';
    const seconds = Number(candidate.source_seconds || 0).toFixed(1);
    meta.textContent = candidate.file + ' · ' + seconds + 's';

    const button = document.createElement('button');
    const selected = candidate.file === doc.primary_candidate;
    button.textContent = selected ? 'Đang dùng' : 'Chọn ảnh này';
    button.disabled = selected;
    button.onclick = async () => {
      try {
        await api(
          'POST',
          '/api/jobs/' + encodeURIComponent(current) + '/thumbnails/select',
          { candidate: candidate.file }
        );
        message.innerHTML = '<span class="ok">Đã chọn ' + candidate.file + ' làm ảnh bìa.</span>';
        await loadStatus();
      } catch (e) {
        message.innerHTML = '<span class="err">' + e.message + '</span>';
      }
    };

    item.append(img, meta, button);
    grid.appendChild(item);
  }

  const edited = Array.isArray(edits.variants) ? edits.variants.slice() : [];
  if (doc.primary_candidate && doc.primary_candidate.startsWith('thumbnail-edit-')
      && !edited.some(item => item.file === doc.primary_candidate)) {
    edited.unshift({file: doc.primary_candidate, preview_file: doc.primary_candidate,
      layout: 'Bản đã chọn trước đây'});
  }
  for (const variant of edited) {
    if (!variant || typeof variant.file !== 'string' || typeof variant.preview_file !== 'string') continue;
    const item = document.createElement('div');
    item.className = 'thumb-item' + (variant.file === doc.primary_candidate ? ' selected' : '');
    const img = document.createElement('img');
    img.alt = 'Ảnh bìa thu nhỏ · ' + (variant.layout || '');
    const href = '/api/jobs/' + encodeURIComponent(current) + '/artifacts/' + encodeURIComponent(variant.preview_file);
    const resolved = await authFetch(href);
    if (resolved.startsWith('blob:')) thumbnailObjectUrls.push(resolved);
    img.src = resolved;
    const label = document.createElement('div');
    label.className = 'muted';
    label.textContent = (variant.layout || '') + ' · 320×180';
    const button = document.createElement('button');
    const selected = variant.file === doc.primary_candidate;
    button.textContent = selected ? 'Đang dùng' : 'Chọn ảnh này';
    button.disabled = selected;
    button.onclick = async () => {
      try {
        await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/thumbnails/select',
          {candidate: variant.file});
        await loadStatus();
        $('thumbnailMsg').textContent = 'Đã thay ảnh bìa; hãy xem lại và duyệt metadata trước khi xuất.';
      } catch (error) { $('thumbnailMsg').textContent = error.message; }
    };
    item.append(img, label, button);
    variantGrid.appendChild(item);
  }
}
$('thumbnailAutoBtn').onclick = async () => {
  if (!current) return;
  const button = $('thumbnailAutoBtn');
  button.disabled = true;
  $('thumbnailMsg').textContent = 'AGY đang tạo 3 ảnh bìa (tự đặt tiêu đề và che watermark kênh nguồn)…';
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/thumbnails/auto', {});
    await renderThumbnails();
    $('thumbnailMsg').textContent = 'Đã tạo 3 ảnh bìa tự động. Chọn 1 ảnh để lưu (sẽ cần duyệt lại metadata & gói xuất).';
  } catch (error) { $('thumbnailMsg').textContent = error.message; }
  finally { button.disabled = false; }
};
$('thumbnailEditBtn').onclick = async () => {
  if (!current) return;
  const button = $('thumbnailEditBtn');
  button.disabled = true;
  $('thumbnailMsg').textContent = 'Đang tạo ba phương án ảnh bìa…';
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/thumbnails/edit', {
      headline: $('thumbnailHeadline').value,
      channel_name: $('thumbnailChannel').value,
    });
    await renderThumbnails();
    $('thumbnailMsg').textContent = 'Đã tạo ba phương án. Xem bản thu nhỏ rồi chọn một ảnh.';
  } catch (error) { $('thumbnailMsg').textContent = error.message; }
  finally { button.disabled = false; }
};

function librarySearchState() {
  return {
    query: $('librarySearch').value.trim(),
    kind: $('libraryKind').value,
    project: $('libraryProject').value.trim(),
    person: $('libraryPerson').value.trim(),
    action: $('libraryAction').value.trim(),
    location: $('libraryLocation').value.trim(),
    object: $('libraryObject').value.trim(),
    scene_type: $('librarySceneType').value.trim(),
    source: $('librarySource').value.trim(),
    date_from: $('libraryDateFrom').value,
    date_to: $('libraryDateTo').value,
    min_duration: $('libraryMinDuration').value,
    max_duration: $('libraryMaxDuration').value,
    min_confidence: $('libraryMinConfidence').value,
  };
}

function applySavedSearch(item) {
  for (const [key, id] of Object.entries({
    query:'librarySearch', kind:'libraryKind', project:'libraryProject', person:'libraryPerson',
    action:'libraryAction', location:'libraryLocation', object:'libraryObject',
    scene_type:'librarySceneType', source:'librarySource', date_from:'libraryDateFrom', date_to:'libraryDateTo',
    min_duration:'libraryMinDuration', max_duration:'libraryMaxDuration', min_confidence:'libraryMinConfidence'
  })) $(id).value = item[key] ?? '';
}

async function loadSavedSearches() {
  const data = await api('GET', '/api/library-searches');
  const select = $('savedLibrarySearches');
  select.replaceChildren(new Option('Tìm kiếm đã lưu…', ''));
  (data.searches || []).forEach((item, index) => {
    const option = new Option(item.name || item.query, String(index));
    option.dataset.search = JSON.stringify(item);
    select.append(option);
  });
}

async function saveLibrarySearch() {
  const state = librarySearchState();
  if (!state.query) { $('libraryResults').textContent = 'Nhập từ khóa tìm kiếm trước khi lưu.'; return; }
  await api('POST', '/api/library-searches', state);
  await loadSavedSearches();
}

async function searchLibrary() {
  const state = librarySearchState();
  const box = $('libraryResults');
  if (!state.query && !Object.entries(state).some(([key,value]) => key !== 'query' && value)) {
    box.textContent = 'Nhập từ khóa hoặc bộ lọc để tìm xuyên mọi project.'; return;
  }
  box.textContent = 'Đang tìm…';
  try {
    const params = new URLSearchParams();
    for (const [key, value] of Object.entries(state)) if (value) params.set(key === 'query' ? 'q' : key, value);
    const data = await api('GET', '/api/library-search?' + params.toString());
    const rows = Array.isArray(data.results) ? data.results : [];
    box.replaceChildren();
    if (!rows.length) { box.textContent = 'Không tìm thấy lời thoại hoặc hình ảnh phù hợp.'; return; }
    for (const item of rows.slice(0, 30)) {
      const card = document.createElement('article');
      card.className = 'project-card';
      const open = document.createElement('button');
      open.type = 'button';
      open.className = 'open-project';
      open.textContent = item.job_id + ' · ' + formatTime(item.start_seconds);
      open.onclick = () => {
        pendingLibrarySeek = {job_id: item.job_id, start_seconds: item.start_seconds};
        selectJob(item.job_id);
      };
      const meta = document.createElement('div');
      meta.className = 'result-meta';
      meta.append(makeChip(item.kind === 'visual' ? 'Hình ảnh' : 'Lời thoại', 'kind-chip'));
      if (item.semantic_score !== null && item.semantic_score !== undefined) {
        meta.append(makeChip('Ngữ nghĩa ' + Number(item.semantic_score).toFixed(2), 'kind-chip'));
      }
      for (const value of [
        ...(item.tags || []).slice(0, 2),
        ...(item.people || []).slice(0, 1),
        ...(item.actions || []).slice(0, 1),
        ...(item.person_tracks || []).slice(0, 2),
      ]) meta.append(makeChip(value, 'kind-chip'));
      const copy = document.createElement('div');
      copy.className = 'result-copy';
      copy.textContent = item.text || '';
      card.append(open, meta, copy);
      box.append(card);
    }
  } catch (error) {
    box.textContent = 'Lỗi tìm thư viện: ' + error.message;
  }
}

$('librarySearchBtn').onclick = () => searchLibrary();
$('saveLibrarySearchBtn').onclick = () => saveLibrarySearch().catch(showError);
$('savedLibrarySearches').onchange = event => {
  const option = event.target.selectedOptions[0];
  if (!option?.dataset.search) return;
  applySavedSearch(JSON.parse(option.dataset.search));
  searchLibrary();
};
$('librarySearch').onkeydown = event => {
  if (event.key === 'Enter') { event.preventDefault(); searchLibrary(); }
};

function syncProjectSelection() {
  const available = projectJobs.filter(job => !job.running && !job.uploading);
  const all = $('selectAllProjects');
  all.disabled = !available.length;
  all.checked = !!available.length && available.every(job => selectedProjects.has(job.job_id));
  all.indeterminate = !all.checked && selectedProjects.size > 0;
  $('deleteSelectedBtn').disabled = !selectedProjects.size;
  $('deleteSelectedBtn').textContent = 'Xóa đã chọn (' + selectedProjects.size + ')';
}

async function loadJobs() {
  try {
    const { jobs } = await api('GET', '/api/jobs');
    projectJobs = jobs;
    const available = new Set(jobs.filter(job => !job.running && !job.uploading).map(job => job.job_id));
    for (const id of selectedProjects) if (!available.has(id)) selectedProjects.delete(id);
    const el = $('jobList');
    if ($('emptyProjectsBtn')) {
      $('emptyProjectsBtn').hidden = false;
      $('emptyProjectsBtn').disabled = !jobs.length;
      $('emptyProjectsBtn').title = jobs.length ? '' : 'Chưa có project nào';
    }
    $('projectSummaryLabel').textContent = current || (jobs.length ? 'Chọn project' : 'Chưa có project');
    if (!jobs.length) {
      el.textContent = 'Chưa có project.';
      $('projectPanel').open = false;
      syncProjectSelection();
      return;
    }
    el.replaceChildren();
    for (const j of jobs) {
      const card = document.createElement('article');
      card.className = 'project-card' + (j.job_id === current ? ' active' : '');
      const checkLabel = document.createElement('label');
      checkLabel.className = 'project-check';
      const check = document.createElement('input');
      check.type = 'checkbox';
      check.value = j.job_id;
      check.setAttribute('aria-label', 'Chọn project ' + j.job_id);
      check.disabled = j.running || j.uploading;
      check.checked = selectedProjects.has(j.job_id);
      check.onchange = () => {
        if (check.checked) selectedProjects.add(j.job_id);
        else selectedProjects.delete(j.job_id);
        syncProjectSelection();
      };
      checkLabel.append(check, document.createTextNode('Chọn'));
      const open = document.createElement('button');
      open.type = 'button'; open.className = 'open-project';
      open.textContent = j.job_id;
      open.onclick = () => selectJob(j.job_id);
      const progress = document.createElement('progress');
      progress.max = j.stage_count; progress.value = j.ready_count;
      progress.setAttribute('aria-label', 'Tiến độ ' + j.job_id);
      const actions = document.createElement('div');
      actions.className = 'card-actions';
      const info = document.createElement('span');
      info.className = 'muted';
      info.textContent = `${j.ready_count}/${j.stage_count} bước · ${j.aspect_ratio} · ${j.uploading ? 'đang import' : j.running ? 'đang chạy' : j.complete ? 'hoàn tất' : 'chờ tiếp tục'}`;
      const remove = document.createElement('button');
      remove.type = 'button'; remove.className = 'danger'; remove.textContent = 'Xóa';
      remove.disabled = j.running || j.uploading;
      remove.onclick = () => confirmDelete(j.job_id);
      actions.append(info, remove);
      card.append(checkLabel, open, progress, actions);
      el.appendChild(card);
    }
    syncProjectSelection();
  } catch (e) { $('jobList').textContent = 'Lỗi tải danh sách: ' + e.message; }
}
$('selectAllProjects').onchange = event => {
  for (const job of projectJobs) {
    if (job.running || job.uploading) continue;
    if (event.target.checked) selectedProjects.add(job.job_id);
    else selectedProjects.delete(job.job_id);
  }
  $('jobList').querySelectorAll('.project-check input').forEach(input => {
    input.checked = selectedProjects.has(input.value);
  });
  syncProjectSelection();
};
$('deleteSelectedBtn').onclick = () => {
  const ids = projectJobs.filter(job => selectedProjects.has(job.job_id)).map(job => job.job_id);
  if (ids.length) openDeleteDialog(ids, false);
};

function selectJob(id) {
  clearThumbnailObjectUrls();
  clearMediaObjectUrls();
  mediaLoaded = null;
  editorLoaded = null;
  setWorkspaceView('explore');
  closeToolDialog('createPanel');
  closeToolDialog('libraryPanel');
  closeToolDialog('scoutPanel');
  $('projectPanel').open = false;
  $('projectSummaryLabel').textContent = id;
  current = id;
  $('empty').style.display = 'none';
  $('detail').style.display = '';
  $('projectPanel').hidden = false;
  $('openCreateProject').hidden = false;
  if ($('openScoutHub')) $('openScoutHub').hidden = false;
  $('video').removeAttribute('src');
  delete $('video').dataset.src;
  $('video').load();
  scriptLoaded = null; metaLoaded = null;
  $('versionsPanel').open = false;
  $('analyticsPanel').open = false;
  $('analyticsResults').replaceChildren();
  $('analyticsAdvice').replaceChildren();
  $('analyticsMsg').textContent = '';
  $('versionSelect').replaceChildren();
  $('versionMsg').textContent = '';
  loadStatus();
  loadMidroll();
  loadAudioMix();
  loadRights();
  loadJobs();
}

let otherAudioEffects = [];
async function loadAudioMix() {
  const jobId = current;
  if (!jobId) return;
  try {
    const data = await api('GET', '/api/jobs/' + encodeURIComponent(jobId) + '/audio-mix');
    if (current !== jobId) return;
    const mix = data.audio_mix || {};
    $('audioVoiceGain').value = mix.voice_gain_db ?? 0;
    $('audioMusicPath').value = mix.music?.path || '';
    $('audioMusicRights').value = mix.music?.rights_note || '';
    $('audioMusicGain').value = mix.music?.gain_db ?? -18;
    otherAudioEffects = (mix.effects || []).slice(1);
    const effect = mix.effects?.[0] || {};
    $('audioEffectPath').value = effect.path || '';
    $('audioEffectRights').value = effect.rights_note || '';
    $('audioEffectTime').value = effect.at_seconds ?? 0;
    $('audioEffectGain').value = effect.gain_db ?? -12;
    $('audioMsg').textContent = otherAudioEffects.length
      ? 'Các hiệu ứng còn lại được giữ nguyên khi lưu.' : '';
    loadChannelSfxPalette();
  } catch (error) { $('audioMsg').textContent = error.message; }
}
$('audioSaveBtn').onclick = async () => {
  if (!current) return;
  const musicPath = $('audioMusicPath').value.trim();
  const effectPath = $('audioEffectPath').value.trim();
  const data = {
    voice_gain_db: Number($('audioVoiceGain').value),
    music: musicPath ? {path: musicPath, rights_note: $('audioMusicRights').value.trim(),
      gain_db: Number($('audioMusicGain').value)} : null,
    effects: effectPath ? [{path: effectPath, rights_note: $('audioEffectRights').value.trim(),
      at_seconds: Number($('audioEffectTime').value), gain_db: Number($('audioEffectGain').value)}, ...otherAudioEffects] : otherAudioEffects,
  };
  $('audioSaveBtn').disabled = true;
  try {
    const result = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/audio-mix', data);
    $('audioMsg').textContent = result.changed
      ? 'Đã lưu. Chạy tiếp để dựng lại video và kiểm tra chất lượng.' : 'Âm thanh không thay đổi.';
    await loadStatus();
  } catch (error) { $('audioMsg').textContent = error.message; }
  finally { $('audioSaveBtn').disabled = false; }
};

async function loadMidroll() {
  if (!current) return;
  try {
    const state = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/midroll');
    $('midrollLine').value = state.staged?.narration || state.draft?.line || '';
    $('midrollDraftBtn').disabled = !!state.staged || !state.approved;
    $('midrollStageBtn').disabled = !!state.staged || !state.draft;
    const insertion = state.insertion || {};
    const location = insertion.location_label || 'Chưa xác định vị trí';
    $('midrollStatus').textContent = insertion.status === 'inserted'
      ? ('CTA đã chèn · ' + location + (state.approved ? ' · đã duyệt' : ' · cần duyệt lại kịch bản'))
      : insertion.status === 'draft'
        ? ('CTA đã soạn · ' + location + ' · chưa chèn')
        : 'CTA chưa được tạo hoặc chèn vào video.';
    $('midrollStatus').className = 'cta-status ' + (insertion.status === 'inserted' ? 'ok' : insertion.status === 'draft' ? 'warn' : 'muted');
    $('midrollPanel').open = insertion.status === 'draft';
    $('midrollMsg').textContent = state.staged
      ? (state.approved ? 'CTA đã duyệt; chạy pipeline để tạo bản video mới.' : 'CTA đã chèn. Kiểm tra kịch bản và bấm Duyệt kịch bản.')
      : state.draft ? ('AGY đã soạn câu cho mốc ' + Math.round(state.draft.start_seconds) + ' giây.') : '';
  } catch (error) { $('midrollMsg').textContent = error.message; }
}
$('midrollDraftBtn').onclick = async () => {
  $('midrollDraftBtn').disabled = true;
  $('midrollMsg').textContent = 'AGY đang viết câu CTA…';
  try {
    const draft = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/midroll/draft', {});
    $('midrollLine').value = draft.line;
    $('midrollStageBtn').disabled = false;
    $('midrollMsg').textContent = 'Đã soạn câu cho mốc ' + Math.round(draft.start_seconds) + ' giây. Đọc lại trước khi chèn.';
  } catch (error) {
    $('midrollMsg').textContent = error.message;
  } finally { $('midrollDraftBtn').disabled = false; }
};
$('midrollStageBtn').onclick = async () => {
  $('midrollStageBtn').disabled = true;
  try {
    const staged = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/midroll/stage',
      {line:$('midrollLine').value});
    $('midrollMsg').textContent = 'Đã chèn tại ' + Math.round(staged.start_seconds)
      + ' giây. Hãy duyệt kịch bản rồi chạy pipeline. Bản MP4 cũ đã được lưu riêng.';
    scriptLoaded = null;
    await loadStatus();
    await loadMidroll();
    setWorkspaceView('review');
  } catch (error) {
    $('midrollMsg').textContent = error.message;
    $('midrollStageBtn').disabled = false;
  }
};

function openToolDialog(id) {
  const dialog = $(id);
  if (!dialog || dialog.open) return;
  $('projectPanel').open = false;
  document.querySelectorAll('dialog.tool-dialog[open]').forEach((other) => {
    if (other !== dialog) other.close();
  });
  dialog.showModal();
  if (id === 'createPanel') prefillCreateFromChannel();
  if (id === 'channelPanel') loadChannels();
  if (id === 'scoutPanel') { if (!_scoutLoaded) loadScoutGems(); }
}

function closeToolDialog(id) {
  const dialog = $(id);
  if (dialog && dialog.open) dialog.close();
}

$('openCreateProject').onclick = () => openToolDialog('createPanel');
$('openBrandSettings').onclick = () => openToolDialog('brandPanel');
$('openLibraryHub').onclick = () => openToolDialog('libraryPanel');
if ($('openScoutHub')) $('openScoutHub').onclick = () => openToolDialog('scoutPanel');
$('closeCreateProject').onclick = () => closeToolDialog('createPanel');
$('closeBrandSettings').onclick = () => closeToolDialog('brandPanel');
$('closeLibraryHub').onclick = () => closeToolDialog('libraryPanel');
if ($('closeScoutHub')) $('closeScoutHub').onclick = () => closeToolDialog('scoutPanel');
{ const b = $('emptyCreateBtn'); if (b) b.onclick = () => openToolDialog('createPanel'); }
{ const b = $('emptyProjectsBtn'); if (b) b.onclick = () => { $('projectPanel').hidden = false; $('projectPanel').open = true; }; }
if ($('emptyScoutBtn')) $('emptyScoutBtn').onclick = () => openToolDialog('scoutPanel');

// -- multi-channel profile switcher -----------------------------------------
{ const b = $('openChannelSwitcher'); if (b) b.onclick = () => openToolDialog('channelPanel'); }
{ const b = $('closeChannelSwitcher'); if (b) b.onclick = () => closeToolDialog('channelPanel'); }
let _channels = [];
function _channelSlug(name) {
  const base = (name || '').normalize('NFKD').replace(/[\\u0300-\\u036f]/g, '')
    .replace(/[đĐ]/g, 'd').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
  return (base || 'kenh') + '-' + Date.now().toString(36);
}
function _chSet(id, val) { const el = $(id); if (el != null) el.value = (val == null ? '' : val); }
function fillChannelForm(ch) {
  $('chId').value = (ch && ch.id) || '';
  _chSet('chName', ch && ch.name);
  _chSet('chTtsProvider', (ch && ch.tts_provider) || 'edge');
  _chSet('chTtsVoice', ch && ch.tts_voice);
  _chSet('chAspect', (ch && ch.aspect_ratio) || '16:9');
  _chSet('chLanguage', (ch && ch.language) || 'vi');
  _chSet('chIntro', ch ? ch.intro_seconds : 0);
  _chSet('chOutro', ch ? ch.outro_seconds : 0);
  _chSet('chCopyright', (ch && ch.visual_variety) || 'off');
  _chSet('chTopBand', ch ? ch.brand_top_band : 0);
  _chSet('chBottomBand', ch ? ch.brand_bottom_band : 0);
  $('chLogoPreview').src = (ch && ch.id && ch.has_logo)
    ? ('/api/channels/' + encodeURIComponent(ch.id) + '/logo?v=' + Date.now())
    : '/api/brand/logo.svg';
  $('chLogo').value = '';
  renderChannelSfx(ch);
}
async function loadChannels(selectId) {
  try {
    const data = await api('GET', '/api/channels');
    _channels = data.channels || [];
    const sel = $('channelSelect');
    sel.innerHTML = '';
    _channels.forEach((ch) => {
      const opt = document.createElement('option');
      opt.value = ch.id;
      opt.textContent = ch.name + (ch.id === data.active ? ' • đang dùng' : '');
      sel.appendChild(opt);
    });
    const pick = selectId || data.active || (_channels[0] && _channels[0].id) || '';
    if (pick) sel.value = pick;
    const chosen = _channels.find((c) => c.id === sel.value);
    fillChannelForm(chosen || null);
    const activeCh = _channels.find((c) => c.id === data.active);
    $('channelActiveNote').textContent = activeCh
      ? ('Kênh đang dùng: ' + activeCh.name)
      : 'Chưa có kênh nào được kích hoạt.';
    $('channelMsg').textContent = '';
  } catch (error) { $('channelMsg').textContent = error.message; }
}
$('channelSelect').onchange = () => {
  const chosen = _channels.find((c) => c.id === $('channelSelect').value);
  if (chosen) fillChannelForm(chosen);
};
$('newChannelBtn').onclick = () => {
  fillChannelForm(null);
  $('chName').focus();
  $('channelMsg').textContent = 'Điền thông tin rồi bấm “Lưu kênh”.';
};
function _channelPayload() {
  const id = $('chId').value || _channelSlug($('chName').value);
  return {
    id: id,
    name: $('chName').value,
    tts_provider: $('chTtsProvider').value,
    tts_voice: $('chTtsVoice').value.trim(),
    aspect_ratio: $('chAspect').value,
    language: $('chLanguage').value.trim() || 'vi',
    intro_seconds: Number($('chIntro').value) || 0,
    outro_seconds: Number($('chOutro').value) || 0,
    visual_variety: $('chCopyright').value,
    brand_top_band: Number($('chTopBand').value) || 0,
    brand_bottom_band: Number($('chBottomBand').value) || 0,
  };
}
$('saveChannelBtn').onclick = async () => {
  $('channelMsg').textContent = 'Đang lưu…';
  try {
    const saved = await api('POST', '/api/channels', _channelPayload());
    const file = $('chLogo').files[0];
    if (file) {
      if (file.type !== 'image/png' || file.size > 2000000) throw new Error('Logo phải là PNG ≤ 2 MB.');
      const headers = { 'Content-Type': 'image/png' };
      if (_tok) headers.Authorization = 'Bearer ' + _tok;
      const res = await fetch('/api/channels/' + encodeURIComponent(saved.id) + '/logo', { method: 'POST', headers: headers, body: file });
      const out = await res.json();
      if (!res.ok) throw new Error(out.error_vi || out.error);
    }
    $('channelMsg').textContent = 'Đã lưu kênh.';
    await loadChannels(saved.id);
    loadBrand().catch(() => {});
  } catch (error) { $('channelMsg').textContent = error.message; }
};
$('activateChannelBtn').onclick = async () => {
  const id = $('channelSelect').value;
  if (!id) return;
  $('channelMsg').textContent = 'Đang kích hoạt…';
  try {
    await api('POST', '/api/channels/' + encodeURIComponent(id) + '/activate');
    $('channelMsg').textContent = 'Đã kích hoạt kênh cho các video dựng tiếp theo.';
    await loadChannels(id);
    loadBrand().catch(() => {});
  } catch (error) { $('channelMsg').textContent = error.message; }
};
$('deleteChannelBtn').onclick = async () => {
  const id = $('channelSelect').value;
  if (!id) return;
  if (!confirm('Xoá kênh này?')) return;
  try {
    await api('DELETE', '/api/channels/' + encodeURIComponent(id));
    $('channelMsg').textContent = 'Đã xoá kênh.';
    await loadChannels();
  } catch (error) { $('channelMsg').textContent = error.message; }
};
async function prefillCreateFromChannel() {
  try {
    const data = await api('GET', '/api/channels');
    const active = (data.channels || []).find((c) => c.id === data.active);
    if (!active) return;
    const form = $('createForm');
    const put = (name, val) => {
      const el = form.querySelector('[name="' + name + '"]');
      if (el != null && val != null && val !== '') el.value = val;
    };
    put('language', active.language);
    put('aspect_ratio', active.aspect_ratio);
    put('tts_provider', active.tts_provider);
    put('tts_voice', active.tts_voice);
    put('visual_variety', active.visual_variety);
    put('brand_top_band', active.brand_top_band);
    put('brand_bottom_band', active.brand_bottom_band);
  } catch (error) { /* best-effort prefill */ }
}
function renderChannelSfx(ch) {
  const holder = $('channelSfxList');
  if (!holder) return;
  if (!ch || !ch.id) { holder.textContent = 'Lưu kênh trước để thêm SFX.'; return; }
  const items = ch.sfx || [];
  if (!items.length) { holder.textContent = 'Chưa có SFX.'; return; }
  holder.innerHTML = '';
  items.forEach((s) => {
    const row = document.createElement('div');
    row.className = 'row';
    row.style.cssText = 'gap:6px; align-items:center; margin:2px 0';
    const name = document.createElement('span');
    name.style.flex = '1';
    name.textContent = s.label + (s.has_file ? '' : ' (chưa có tệp)') + ' · ' + (s.gain_db ?? -8) + 'dB';
    const del = document.createElement('button');
    del.type = 'button';
    del.textContent = 'Xoá';
    del.onclick = () => deleteChannelSfx(ch.id, s.slug);
    row.appendChild(name);
    row.appendChild(del);
    holder.appendChild(row);
  });
}
$('addSfxBtn').onclick = async () => {
  const id = $('chId').value;
  if (!id) { $('channelMsg').textContent = 'Lưu kênh trước khi thêm SFX.'; return; }
  const label = $('sfxLabel').value.trim();
  const rights = $('sfxRights').value.trim();
  const file = $('sfxFile').files[0];
  if (!label || !rights) { $('channelMsg').textContent = 'Nhập nhãn và ghi chú quyền cho SFX.'; return; }
  if (!file) { $('channelMsg').textContent = 'Chọn tệp âm thanh SFX.'; return; }
  if (file.size > 3000000) { $('channelMsg').textContent = 'Tệp SFX ≤ 3 MB.'; return; }
  const slug = _channelSlug(label);
  $('channelMsg').textContent = 'Đang thêm SFX…';
  try {
    await api('POST', '/api/channels/' + encodeURIComponent(id) + '/sfx',
      { slug: slug, label: label, rights_note: rights, gain_db: Number($('sfxGain').value) || -8 });
    const headers = { 'Content-Type': file.type || 'audio/mpeg' };
    if (_tok) headers.Authorization = 'Bearer ' + _tok;
    const res = await fetch('/api/channels/' + encodeURIComponent(id) + '/sfx/' + encodeURIComponent(slug) + '/file',
      { method: 'POST', headers: headers, body: file });
    const out = await res.json();
    if (!res.ok) throw new Error(out.error_vi || out.error);
    $('sfxLabel').value = ''; $('sfxRights').value = ''; $('sfxFile').value = '';
    $('channelMsg').textContent = 'Đã thêm SFX “' + label + '”.';
    await loadChannels(id);
  } catch (error) { $('channelMsg').textContent = error.message; }
};
async function deleteChannelSfx(id, slug) {
  try {
    await api('DELETE', '/api/channels/' + encodeURIComponent(id) + '/sfx/' + encodeURIComponent(slug));
    await loadChannels(id);
  } catch (error) { $('channelMsg').textContent = error.message; }
}
async function loadChannelSfxPalette() {
  const wrap = $('channelSfxPalette');
  const holder = $('channelSfxButtons');
  if (!wrap || !holder) return;
  try {
    const data = await api('GET', '/api/channel-sfx');
    const items = data.sfx || [];
    holder.innerHTML = '';
    if (!items.length) { wrap.hidden = true; return; }
    items.forEach((sfx) => {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.textContent = '+ ' + sfx.label;
      btn.onclick = () => insertChannelSfx(sfx.slug, sfx.label);
      holder.appendChild(btn);
    });
    wrap.hidden = false;
  } catch (error) { wrap.hidden = true; }
}
async function insertChannelSfx(slug, label) {
  if (!current) return;
  const video = $('video');
  const at = (video && isFinite(video.currentTime)) ? Math.max(0, video.currentTime) : 0;
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/audio-mix/sfx',
      { slug: slug, at_seconds: at });
    $('audioMsg').textContent = 'Đã chèn “' + label + '” tại ' + at.toFixed(1) + 's. Chạy tiếp để dựng lại.';
    await loadAudioMix();
  } catch (error) { $('audioMsg').textContent = error.message; }
}
async function autoPlaceTransitionSfx() {
  if (!current) return;
  $('audioMsg').textContent = 'Đang tự rải SFX theo cắt cảnh…';
  try {
    const result = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/audio-mix/transitions');
    $('audioMsg').textContent = 'Đã tự rải ' + (result.placed || 0) + ' SFX tại các điểm chuyển cảnh. Chạy tiếp để dựng lại.';
    await loadAudioMix();
  } catch (error) { $('audioMsg').textContent = error.message; }
}
if ($('autoTransitionSfxBtn')) $('autoTransitionSfxBtn').onclick = autoPlaceTransitionSfx;

const projectPanel = $('projectPanel');
document.addEventListener('pointerdown', (event) => {
  if (projectPanel.open && !projectPanel.contains(event.target)) projectPanel.open = false;
});
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape') projectPanel.open = false;
});

function badge(stage) {
  return `<span class="badge ${stage.status}" title="${stage.status_hint||''}">${stage.status_label}</span>`;
}

let reindexWasRunning = false;
$('reindexTranscriptBtn').onclick = async () => {
  if (!current) return;
  const btn = $('reindexTranscriptBtn');
  btn.disabled = true;
  $('reindexMsg').textContent = 'Đang bắt đầu lập chỉ mục lại lời thoại…';
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/reindex-transcript', {});
    await loadStatus();
  } catch (error) {
    btn.disabled = false;
    $('reindexMsg').textContent = error.message;
  }
};
async function loadStatus() {
  if (!current) return;
  let s;
  try { s = await api('GET', '/api/jobs/' + encodeURIComponent(current)); }
  catch (e) { $('progText').textContent = 'Lỗi: ' + e.message; return; }

  currentStatus = s;
  $('jobTitle').textContent = 'Job: ' + s.job_id;
  renderProjectConfig(s);
  const total = s.stages.length;
  const ready = s.counts.ready || 0;
  $('progBar').style.width = Math.round(ready * 100 / total) + '%';
  $('progText').textContent = `${ready}/${total} bước hoàn tất · ${s.complete ? 'đã xong' : 'đang tiến hành'}`
    + (s.running ? ' · đang chạy…' : '')
    + (s.is_indexing ? ` · đang lập chỉ mục nền${s.indexing && s.indexing.stage ? ` (${s.indexing.stage} ${s.indexing.done}/${s.indexing.total})` : ''}…` : '');
  $('runErr').textContent = s.run_error_vi ? ('Lỗi chạy: ' + s.run_error_vi) : '';
  if ($('voiceInfo')) {
    const vm = s.voice;
    $('voiceInfo').textContent = vm && vm.engine
      ? ('Giọng đọc: ' + vm.engine + (vm.voice ? ' · ' + vm.voice : '')
         + (vm.timing_mode === 'estimated_word_timing' ? ' · nhịp từ ước lượng' : ''))
      : '';
  }
  $('runBtn').disabled = !!s.running || !!s.uploading || !s.has_source_video || !!(s.reindex && s.reindex.running);
  $('stopBtn').hidden = !s.running;
  $('stopBtn').disabled = !s.can_stop;
  $('deleteBtn').disabled = !!s.running || !!s.uploading;
  $('brandRenderBtn').disabled = !!s.running || !s.has_source_video;
  $('audioSaveBtn').disabled = !!s.running;
  const reindexState = s.reindex || { running: false };
  if ($('reindexTranscriptBtn')) {
    $('reindexTranscriptBtn').disabled = !!reindexState.running || !!s.running || !!s.uploading || !!s.is_indexing || !s.has_source_video;
    $('reindexMsg').textContent = reindexState.running
      ? (reindexState.message || 'Đang lập chỉ mục lại lời thoại…')
      : reindexState.error ? ('Lỗi: ' + reindexState.error) : (reindexState.message || '');
    // When a re-index finishes cleanly, refresh the explorer so the recomputed
    // caption timing is shown without a manual reload.
    if (reindexWasRunning && !reindexState.running && !reindexState.error
        && current && $('mediaExplorerCard').style.display !== 'none') {
      renderMediaExplorer($('mediaSearch').value.trim()).catch(() => {});
    }
    reindexWasRunning = !!reindexState.running;
  }
  if (document.activeElement !== $('brandTopBand')) $('brandTopBand').value = s.brand_top_band ?? 0;
  if (document.activeElement !== $('brandBottomBand')) $('brandBottomBand').value = s.brand_bottom_band ?? 0;
  $('sourceRetryCard').hidden = !!s.has_source_video;
  if (!s.has_source_video) {
    updateSourceRetryFallback();
  }
  const scriptStage = s.stages.find(stage => stage.stage === 'script');
  const renderStage = s.stages.find(stage => stage.stage === 'render');
  const qaStage = s.stages.find(stage => stage.stage === 'qa');
  $('runBtn').textContent = s.approvals.script_approved ? 'Chạy tiếp đến video' : 'Chạy đến kịch bản';
  $('nextAction').textContent = s.uploading ? 'Đang import video…' : !s.has_source_video ? 'Import video MP4 để bắt đầu.'
    : s.is_indexing ? 'Đang lập chỉ mục nền (transcript/cảnh/hình ảnh). Có thể mở project khác trong lúc chờ.'
    : s.running ? 'Pipeline đang xử lý. Tiến độ cập nhật tự động.'
    : s.has_final_video && qaStage?.status === 'ready' ? 'Video đã sẵn sàng: xem trước và tải MP4 bên dưới.'
    : scriptStage?.status === 'ready' && !s.approvals.script_approved ? 'Rà soát và duyệt kịch bản, rồi chạy tiếp để tạo video.'
    : renderStage?.status === 'failed' ? 'Bước dựng video lỗi: xem thông báo ở tiến trình rồi chạy lại.'
    : 'Chạy pipeline để tạo kịch bản và các tệp cần thiết.';
  const readyForDownload = !!(s.has_final_video && qaStage?.status === 'ready');
  // Keep reviewFocus aligned with nextAction so "Duyệt & xuất" opens the one
  // block that needs attention (same priority order as nextAction above).
  reviewFocus = readyForDownload ? 'export'
    : (scriptStage?.status === 'ready' && !s.approvals.script_approved) ? 'script'
    : s.has_thumbnail ? 'thumbnail'
    : s.approvals.script_approved && !s.approvals.metadata_approved ? 'metadata'
    : 'script';
  $('exportCard').hidden = !readyForDownload;
  $('reviewAction').hidden = !(scriptStage?.status === 'ready' && !s.approvals.script_approved);
  $('quickDownload').hidden = !readyForDownload;
  $('downloadFinal').href = '/api/jobs/' + encodeURIComponent(current) + '/artifacts/final.mp4';
  $('quickDownload').href = $('downloadFinal').href;
  $('handoffBtn').disabled = !readyForDownload || !s.approvals.script_approved || !s.approvals.metadata_approved || !!s.running;
  const shortState = s.short_export || {};
  $('shortExportBtn').disabled = !readyForDownload || !s.approvals.script_approved || !!s.running || !!shortState.running;
  $('shortMarkStart').disabled = !readyForDownload;
  $('shortMarkEnd').disabled = !readyForDownload;
  $('shortExportMsg').textContent = shortState.running ? 'Đang xuất video ngắn…'
    : shortState.error ? 'Không xuất được: ' + shortState.error
    : shortState.href ? 'Video ngắn đã sẵn sàng.' : '';
  $('shortDownload').hidden = !shortState.href;
  $('shortSrtDownload').hidden = !shortState.srt_href;
  if (shortState.href) $('shortDownload').href = shortState.href;
  if (shortState.srt_href) $('shortSrtDownload').href = shortState.srt_href;
  const qaFindings = $('qaFindings');
  qaFindings.replaceChildren();
  qaFindings.hidden = !(s.qa_findings || []).length;
  for (const finding of s.qa_findings || []) {
    const row = document.createElement('p');
    row.textContent = (finding.review_required ? 'Cần xem lại: ' : 'Lỗi QA: ')
      + finding.check + ' — ' + (finding.message || JSON.stringify(finding.value));
    qaFindings.appendChild(row);
  }

  $('stages').innerHTML = s.stages.map(st =>
    `<div class="stage"><div class="name">${st.stage_label}</div>${badge(st)}`
    + `<div class="muted">${st.message_vi || ''}</div></div>`).join('');

  const vc = $('videoCard');
  if (s.has_final_video) {
    vc.style.display = '';
    const src = '/api/jobs/' + encodeURIComponent(current) + '/artifacts/final.mp4';
    const v = $('video');
    if (v.dataset.src !== src) {
      v.dataset.src = src;
      v.src = src;
    }
  } else { vc.style.display = 'none'; }

  // Refresh after background indexing or pipeline scene memory completes.
  // Both can add AGY observations after the media index file first appears.
  const scenePlanStage = s.stages.find(stage => stage.stage === 'scene_plan');
  const mediaState = current + ':' + Boolean(s.has_media_index) + ':'
    + Boolean(s.is_indexing) + ':' + (scenePlanStage?.updated_at || '');
  if (mediaLoaded !== mediaState) {
    mediaLoaded = mediaState;
    await renderMediaExplorer();
  }
  const planStage = s.stages.find(stage => stage.stage === 'scene_plan');
  const editorState = current + ':' + (planStage?.status || '') + ':' + (planStage?.updated_at || '');
  if (s.has_media_index && editorLoaded !== editorState) {
    await renderEditor();
    editorLoaded = editorState;
  } else if (!s.has_media_index) {
    // Keep the editor card visible with guidance instead of hiding it.
    editorLoaded = null;
    $('editorCard').style.display = '';
    $('timelineList').replaceChildren();
    $('continuityTracks').replaceChildren();
    setEditorState(s.is_indexing
      ? 'Đang lập chỉ mục tư liệu. Dòng thời gian sẽ sẵn sàng để biên tập ngay khi lập chỉ mục xong.'
      : !s.has_source_video ? 'Import video MP4, sau đó chạy pipeline để tạo dòng thời gian có thể chỉnh sửa.'
      : 'Chạy pipeline tới bước kế hoạch cảnh để tạo dòng thời gian có thể chỉnh sửa.');
  }

  if (s.has_thumbnail) {
    await renderThumbnails();
  } else {
    clearThumbnailObjectUrls();
    $('thumbnailCard').style.display = 'none';
    $('thumbnailGrid').replaceChildren();
    $('thumbnailMsg').textContent = '';
  }

  const artifacts = $('artifacts');
  artifacts.replaceChildren();
  if (!s.artifacts.length) artifacts.textContent = 'Chưa có tệp project.';
  for (const item of s.artifacts) {
    const card = document.createElement('div'); card.className = 'artifact-card';
    const link = document.createElement('a'); link.href = item.href;
    link.textContent = item.name; link.download = item.name;
    if (_tok && item.kind !== 'video' && item.kind !== 'archive') link.onclick = async event => {
      event.preventDefault();
      const resolved = await authFetch(item.href);
      const tmp = document.createElement('a');
      tmp.href = resolved; tmp.download = item.name; tmp.click();
    };
    const details = document.createElement('small'); details.className = 'muted';
    details.textContent = `${item.kind} · ${(item.size / 1048576).toFixed(1)} MB`;
    card.append(link, details); artifacts.appendChild(card);
  }

  renderMeta(s.approvals);
  // If the review view is open, keep it focused on the block that matches the
  // freshly computed next action (reviewFocus) after state-driven visibility.
  if (!$('view-review').hidden) applyReviewFocus();
  if (poller) { clearInterval(poller); poller = null; }
  if (s.running || s.is_indexing || s.section_preview?.running || s.reindex?.running) { poller = setInterval(loadStatus, 1500); }
  if ($('sectionPreviewPanel').open && $('sectionPreviewSelect').value) {
    refreshSectionPreview().catch(showError);
  }
}

let metaLoaded = null;
let scriptLoaded = null;
let tagSections = [];
function showTagSection() {
  const section = tagSections[Number($('tagSection').value)];
  $('tagNarration').value = section ? section.narration || '' : '';
}
$('tagSection').onchange = showTagSection;
async function renderMeta(approvals) {
  $('scriptState').innerHTML = approvals.script_present
    ? (approvals.script_approved ? '<span class="ok">Kịch bản đã được duyệt.</span>'
        : '<span class="warn">Kịch bản chưa được duyệt.</span>')
    : '<span class="muted">Chưa có kịch bản (chạy pipeline tới bước Kịch bản).</span>';
  if (!approvals.script_present) {
    scriptLoaded = null; tagSections = []; $('scriptSections').value = '';
    $('tagSection').replaceChildren(); $('tagNarration').value = '';
  } else if (scriptLoaded !== current) {
    try {
      const { present, script } = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/script');
      if (present && script.sections !== undefined) {
        $('scriptSections').value = JSON.stringify(script.sections, null, 2);
        tagSections = script.sections;
        $('tagSection').replaceChildren(...tagSections.map((section, index) => {
          const option = document.createElement('option');
          option.value = String(index); option.textContent = section.title || 'Phần ' + (index + 1);
          return option;
        }));
        showTagSection();
        scriptLoaded = current;
      }
    } catch (e) { $('scriptMsg').textContent = 'Không tải được kịch bản: ' + e.message; }
  }
  $('saveScriptBtn').disabled = !approvals.script_present;
  $('approveScriptBtn').disabled = !approvals.script_present;
  $('tagScriptBtn').disabled = !approvals.script_present;

  $('metaState').innerHTML = approvals.metadata_present
    ? (approvals.metadata_approved ? '<span class="ok">Siêu dữ liệu đã được duyệt.</span>'
        : '<span class="warn">Siêu dữ liệu chưa được duyệt.</span>')
    : '<span class="muted">Chưa có siêu dữ liệu (chạy pipeline tới bước Siêu dữ liệu).</span>';
  if (!approvals.metadata_present) { metaLoaded = null; $('metaTitle').value = ''; $('metaDesc').value = ''; $('metaTags').value = ''; }
  else if (metaLoaded !== current) {
    try {
      const { present, metadata } = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/metadata');
      if (present) {
        $('metaTitle').value = metadata.title || '';
        $('metaDesc').value = metadata.description || '';
        $('metaTags').value = (metadata.tags || []).join(', ');
        metaLoaded = current;
      }
    } catch (e) { $('metaMsg').textContent = 'Không tải được siêu dữ liệu: ' + e.message; }
  }
  const canPublish = approvals.script_approved && approvals.metadata_approved && $('confirmPub').checked;
  $('publishBtn').disabled = !canPublish;
  $('approveBtn').disabled = !approvals.metadata_present;
  $('saveMetaBtn').disabled = !approvals.metadata_present;
}

$('createForm').onsubmit = async (e) => {
  e.preventDefault();
  const file = $('sourceFile').files[0];
  const linkUrl = String($('linkUrl') ? $('linkUrl').value : '').trim();
  const confirmRights = $('confirmDownloadRights') ? $('confirmDownloadRights').checked : false;
  if (linkUrl && file) { $('createMsg').textContent = 'Chọn tải file lên hoặc dán link video, không dùng cả hai.'; return; }
  if (linkUrl && !confirmRights) { $('createMsg').textContent = 'Vui lòng xác nhận bạn có quyền sử dụng video này để tải về.'; return; }
  const fd = new FormData(e.target);
  const payload = Object.fromEntries(fd.entries());
  payload.job_id = autoJobId(payload.movie_title, file);
  payload.creative_brief = {
    review_thesis: String(payload.review_thesis || '').trim(),
    tone: String(payload.tone || '').trim(),
    target_audience: String(payload.target_audience || '').trim(),
    spoiler_policy: payload.spoiler_policy || 'unspecified',
    forbidden_claims: String(payload.forbidden_claims || '').split('\\n').map(item => item.trim()).filter(Boolean)
  };
  for (const key of ['review_thesis', 'tone', 'target_audience', 'spoiler_policy', 'forbidden_claims']) delete payload[key];
  if (file && payload.source_video) { $('createMsg').textContent = 'Chọn file MP4 hoặc đường dẫn, không dùng cả hai.'; return; }
  if (file && (!/[.]mp4$/i.test(file.name) || !file.size)) { $('createMsg').textContent = 'Chọn video MP4 không rỗng.'; return; }
  const submit = e.target.querySelector('[type="submit"]');
  submit.disabled = true;
  let created = null;
  try {
    created = await api('POST', '/api/jobs', payload);
    if (file) {
      await uploadVideo(created.job_id, file);
      $('createMsg').textContent = 'Đã tạo project và import video.';
    } else if (linkUrl) {
      $('createMsg').textContent = 'Đang tải video từ liên kết (yt-dlp)...';
      await api('POST', '/api/jobs/' + encodeURIComponent(created.job_id) + '/download-link', {
        url: linkUrl,
        confirm_rights: confirmRights,
      });
      $('createMsg').textContent = 'Đã tạo project và tải video từ liên kết thành công.';
    } else {
      $('createMsg').textContent = 'Đã tạo project mới.';
    }
    e.target.reset();
    if ($('linkPreview')) $('linkPreview').style.display = 'none';
    await loadJobs();
    selectJob(created.job_id);
  } catch (err) {
    $('createMsg').textContent = (created ? 'Project đã tạo; có thể mở và thử import lại. ' : '') + err.message;
    if (created) { await loadJobs(); selectJob(created.job_id); }
  } finally { submit.disabled = false; }
};

if ($('probeLinkBtn')) {
  $('probeLinkBtn').onclick = async () => {
    const url = String($('linkUrl') ? $('linkUrl').value : '').trim();
    const btn = $('probeLinkBtn');
    const preview = $('linkPreview');
    if (!url) {
      if (preview) {
        preview.style.display = 'block';
        preview.className = 'notice warn';
        preview.textContent = 'Vui lòng nhập liên kết video trước khi kiểm tra.';
      }
      $('createMsg').textContent = 'Nhập liên kết video trước khi kiểm tra.';
      return;
    }
    btn.disabled = true;
    const origText = btn.textContent;
    btn.textContent = 'Đang kiểm tra…';
    if (preview) {
      preview.style.display = 'block';
      preview.className = 'notice info';
      preview.textContent = 'Đang kết nối và đọc thông tin video qua yt-dlp…';
    }
    $('createMsg').textContent = 'Đang kiểm tra thông tin liên kết…';
    try {
      const meta = await api('POST', '/api/link-probe', { url });
      if (meta.title && !$('createForm').elements.namedItem('movie_title').value) {
        $('createForm').elements.namedItem('movie_title').value = meta.title;
      }
      const mins = meta.duration ? Math.round(meta.duration / 60) : null;
      if (preview) {
        preview.style.display = 'block';
        preview.className = 'notice ok';
        preview.textContent = '✓ Tìm thấy: ' + (meta.title || 'Video') + (mins ? ' (~' + mins + ' phút)' : '') + (meta.subtitle_languages && meta.subtitle_languages.length ? ' · Phụ đề: ' + meta.subtitle_languages.join(', ') : '');
      }
      $('createMsg').textContent = 'Đã đọc thông tin liên kết: ' + (meta.title || url);
    } catch (err) {
      if (preview) {
        preview.style.display = 'block';
        preview.className = 'notice err';
        preview.textContent = 'Lỗi đọc liên kết: ' + err.message;
      }
      $('createMsg').textContent = 'Lỗi đọc liên kết: ' + err.message;
    } finally {
      btn.disabled = false;
      btn.textContent = origText;
    }
  };
}

const agyPoolSelect = document.querySelector('select[name="content_agent"]');
function setAgyBadge(state, text) {
  const badge = $('agyPoolBadge');
  badge.className = 'badge ' + state;
  badge.textContent = 'AGY: ' + text;
}
async function refreshAgyPool() {
  setAgyBadge('pending', 'đang kiểm tra…');
  try {
    const status = await api('GET', '/api/agy-pool');
    if (!status.configured) { setAgyBadge('skipped', 'chưa cấu hình'); return; }
    const state = status.total > 0 && status.reachable === status.total ? 'ready'
      : (status.reachable > 0 ? 'running' : 'failed');
    setAgyBadge(state, status.reachable + '/' + status.total + ' worker mở cổng');
  } catch (err) { setAgyBadge('failed', err.message); }
}
if (agyPoolSelect) {
  agyPoolSelect.addEventListener('change', () => { if (agyPoolSelect.value === 'agy') refreshAgyPool(); });
  if (agyPoolSelect.value === 'agy') refreshAgyPool();
}
$('agyProbeBtn').onclick = async () => {
  const btn = $('agyProbeBtn');
  btn.disabled = true;
  setAgyBadge('running', 'đang gọi thử…');
  try {
    const result = await api('POST', '/api/agy-pool/probe');
    if (result.ok) setAgyBadge('ready', 'gọi thử thành công');
    else setAgyBadge('failed', result.error || 'lỗi không rõ');
  } catch (err) { setAgyBadge('failed', err.message); }
  finally { btn.disabled = false; }
};

$('sourceRetryBtn').onclick = async () => {
  const file = $('sourceRetry').files[0];
  const id = current;
  $('sourceRetryBtn').disabled = true;
  try {
    await uploadVideo(id, file, true);
    $('sourceRetryMsg').textContent = 'Đã import video.';
    await loadStatus(); await loadJobs();
  } catch (err) { $('sourceRetryMsg').textContent = err.message; }
  finally { $('sourceRetryBtn').disabled = false; }
};

function updateSourceRetryFallback(customQuery) {
  const fallbackSec = $('sourceRetryFallbackSection');
  const fallbackBtn = $('sourceRetrySearchFallbackBtn');
  if (!fallbackSec || !fallbackBtn) return;
  let query = customQuery;
  if (!query) {
    const cfg = (currentStatus && currentStatus.config) || {};
    query = cfg.movie_title || (currentStatus && currentStatus.title) || '';
  }
  if (query) {
    const cleanQuery = query.replace(/\\s*\\(\\d{4}\\)\\s*$/, '').trim();
    const ytUrl = 'https://www.youtube.com/results?search_query=' + encodeURIComponent(cleanQuery + ' full movie');
    fallbackBtn.href = ytUrl;
    fallbackBtn.textContent = '🔍 Tìm "' + cleanQuery + '" trên YouTube';
    fallbackSec.hidden = false;
  } else {
    fallbackSec.hidden = true;
  }
}

if ($('sourceRetryUrlBtn')) {
  $('sourceRetryUrlBtn').onclick = async () => {
    if (!current) return;
    const url = $('sourceRetryUrl') ? $('sourceRetryUrl').value.trim() : '';
    if (!url) {
      alert('Vui lòng nhập liên kết video nguồn (YouTube / Web).');
      return;
    }
    const rights = $('sourceRetryRights') ? $('sourceRetryRights').checked : false;
    if (!rights) {
      alert('Vui lòng tích xác nhận quyền sử dụng video.');
      return;
    }
    const btn = $('sourceRetryUrlBtn');
    const msg = $('sourceRetryMsg');
    btn.disabled = true;
    if (msg) msg.textContent = 'Đang kết nối và tải video qua yt-dlp… Quá trình có thể mất vài phút.';
    try {
      await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/download-link', {
        url: url,
        confirm_rights: true
      });
      if (msg) msg.textContent = 'Đã tải và nạp video thành công!';
      await loadStatus();
      await loadJobs();
    } catch (err) {
      if (msg) msg.textContent = 'Lỗi tải video: ' + err.message + '. Bạn có thể tìm video thay thế trên YouTube bên dưới.';
      updateSourceRetryFallback();
    } finally {
      btn.disabled = false;
    }
  };
}

function openDeleteDialog(ids, single) {
  const dialog = $('deleteDialog');
  const phrase = single ? ids[0] : 'XOA ' + ids.length;
  dialog.dataset.ids = JSON.stringify(ids);
  dialog.dataset.single = single ? '1' : '0';
  dialog.dataset.confirm = phrase;
  $('deleteTitle').textContent = single ? 'Xóa project ' + ids[0] : 'Xóa ' + ids.length + ' project đã chọn';
  $('deleteTargets').textContent = ids.join(' · ');
  $('deleteConfirmLabel').textContent = 'Nhập ' + phrase + ' để xác nhận';
  $('deleteConfirm').value = '';
  $('deleteMsg').textContent = '';
  $('confirmDelete').textContent = single ? 'Xóa project' : 'Xóa ' + ids.length + ' project';
  $('confirmDelete').disabled = true;
  dialog.showModal();
}
function confirmDelete(id) { openDeleteDialog([id], true); }
$('deleteBtn').onclick = () => confirmDelete(current);
$('cancelDelete').onclick = () => $('deleteDialog').close();
$('deleteConfirm').oninput = () => {
  $('confirmDelete').disabled = $('deleteConfirm').value !== $('deleteDialog').dataset.confirm;
};
$('deleteForm').onsubmit = async event => {
  event.preventDefault();
  const dialog = $('deleteDialog');
  const ids = JSON.parse(dialog.dataset.ids);
  const phrase = dialog.dataset.confirm;
  if ($('deleteConfirm').value !== phrase) return;
  $('confirmDelete').disabled = true;
  try {
    const result = dialog.dataset.single === '1'
      ? await api('DELETE', '/api/jobs/' + encodeURIComponent(ids[0]), { confirm: phrase })
      : await api('DELETE', '/api/jobs', { job_ids: ids, confirm: phrase });
    const deleted = dialog.dataset.single === '1' ? ids : result.deleted;
    dialog.close();
    for (const id of deleted) selectedProjects.delete(id);
    if (deleted.includes(current)) {
      current = null; mediaLoaded = null; editorLoaded = null; scriptLoaded = null; metaLoaded = null;
      if (poller) { clearInterval(poller); poller = null; }
      clearThumbnailObjectUrls(); clearMediaObjectUrls();
      $('sourceVideo').removeAttribute('src'); $('sourceVideo').load();
      $('video').removeAttribute('src'); delete $('video').dataset.src; $('video').load();
      $('detail').style.display = 'none'; $('empty').style.display = ''; $('projectPanel').hidden = true; $('openCreateProject').hidden = true;
      if ($('openScoutHub')) $('openScoutHub').hidden = true;
      $('projectPanel').open = false;
      $('projectSummaryLabel').textContent = 'Chọn project';
    }
    $('bulkMsg').textContent = result.failed
      ? 'Đã xóa ' + deleted.length + ' project; không xóa được ' + result.failed + '. Kiểm tra project còn lại.'
      : 'Đã xóa ' + deleted.length + ' project.';
    await loadJobs();
  } catch (err) { $('deleteMsg').textContent = err.message; $('confirmDelete').disabled = false; }
};

$('stopBtn').onclick = async () => { await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/stop', {}); await loadStatus(); };
let savedVersions = [];
async function loadVersions() {
  if (!current) return;
  const project = current;
  const selected = $('versionSelect').value;
  const data = await api('GET', '/api/jobs/' + encodeURIComponent(project) + '/versions');
  if (current !== project) return;
  savedVersions = data.versions;
  $('versionSelect').replaceChildren();
  for (const item of savedVersions) {
    const option = document.createElement('option');
    option.value = item.id;
    option.textContent = item.name + ' · ' + new Date(item.created_at).toLocaleString('vi-VN');
    $('versionSelect').appendChild(option);
  }
  if (savedVersions.some(item => item.id === selected)) $('versionSelect').value = selected;
  updateVersionSelection();
}
function updateVersionSelection() {
  const item = savedVersions.find(entry => entry.id === $('versionSelect').value);
  $('restoreVersionBtn').disabled = !item;
  $('versionDownload').hidden = !item?.passing_final;
  if (item?.passing_final) $('versionDownload').href = '/api/jobs/' + encodeURIComponent(current) + '/versions/' + item.id + '/artifacts/final.mp4';
  for (const option of $('versionKind').options) {
    const required = {script: 'script.json', scene_plan: 'scene_plan.json', captions: 'aligned.srt', metadata: 'youtube_metadata.json', final: 'final.mp4'}[option.value];
    option.disabled = !item?.files?.includes(required) || (option.value === 'final' && !item.passing_final);
  }
  if ($('versionKind').selectedOptions[0]?.disabled) {
    const available = Array.from($('versionKind').options).find(option => !option.disabled);
    if (available) $('versionKind').value = available.value;
  }
}
$('versionsPanel').addEventListener('toggle', () => {
  if ($('versionsPanel').open) loadVersions().catch(error => { $('versionMsg').textContent = error.message; });
});
$('versionSelect').onchange = updateVersionSelection;
$('saveVersionBtn').onclick = async () => {
  const button = $('saveVersionBtn');
  button.disabled = true;
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/versions', {name: $('versionName').value.trim()});
    await loadVersions();
    $('versionName').value = '';
    $('versionMsg').textContent = 'Đã lưu phiên bản.';
  } catch (error) { $('versionMsg').textContent = error.message; }
  finally { button.disabled = false; }
};
$('restoreVersionBtn').onclick = async () => {
  const button = $('restoreVersionBtn');
  button.disabled = true;
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/versions/' + $('versionSelect').value + '/restore', {kind: $('versionKind').value});
    scriptLoaded = null; metaLoaded = null; editorLoaded = null;
    await Promise.all([loadVersions(), loadStatus()]);
    $('versionMsg').textContent = 'Đã khôi phục. Nếu nội dung thay đổi, duyệt lại trước khi dựng.';
  } catch (error) { $('versionMsg').textContent = error.message; }
  finally { button.disabled = false; }
};
$('shortMarkStart').onclick = () => { $('shortStart').value = $('video').currentTime.toFixed(1); };
$('shortMarkEnd').onclick = () => { $('shortEnd').value = $('video').currentTime.toFixed(1); };
$('shortExportBtn').onclick = async () => {
  const button = $('shortExportBtn');
  button.disabled = true;
  $('shortExportMsg').textContent = 'Đang bắt đầu xuất video ngắn…';
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/shorts', {
      start_seconds: Number($('shortStart').value),
      end_seconds: Number($('shortEnd').value),
    });
    await loadStatus();
  } catch (error) {
    $('shortExportMsg').textContent = error.message;
    button.disabled = false;
  }
};
function renderAdvice(advice) {
  const box = $('analyticsAdvice');
  box.replaceChildren();
  if (!advice || !advice.enabled || !(advice.suggestions || []).length) return;
  const title = document.createElement('div');
  title.textContent = 'Gợi ý cho video sau (advisory, không tự đổi kịch bản):';
  box.appendChild(title);
  for (const s of advice.suggestions) {
    const row = document.createElement('div');
    row.textContent = '• ' + (s.message_vi || '');
    box.appendChild(row);
  }
  if (advice.low_confidence) {
    const note = document.createElement('div');
    note.textContent = '(Mẫu còn ít — chỉ nên tham khảo.)';
    box.appendChild(note);
  }
}
function renderAnalytics(imports, advice) {
  const panel = $('analyticsResults');
  panel.replaceChildren();
  renderAdvice(advice);
  if (!imports.length) { panel.textContent = 'Chưa nhập số liệu cho project này.'; return; }
  const report = imports[0];
  const m = report.measurements || {};
  const shown = [
    'Bản xuất: ' + String(report.approved_revision_sha256 || '').slice(0, 12),
    'Impressions: ' + (report.impressions ?? 'Chưa có'),
    'CTR Studio: ' + (report.ctr_percent == null ? 'Chưa có' : report.ctr_percent + '%'),
    'Rơi ở intro: ' + (m.intro_drop_percentage_points == null ? 'Thiếu mẫu quanh giây 30' : m.intro_drop_percentage_points + ' điểm %'),
    'Rơi quanh CTA: ' + (m.cta_drop_percentage_points == null ? 'Thiếu mốc hoặc mẫu gần CTA' : m.cta_drop_percentage_points + ' điểm %'),
    'Ghi chú: ' + (report.notes || 'Chưa có'),
    'Kết quả thử thumbnail/tiêu đề: ' + (report.test_results || []).map(item => item.variant + ': ' + item.result).join('; '),
    'Số lần nhập: ' + imports.length,
  ];
  for (const line of shown) {
    const row = document.createElement('div');
    row.textContent = line;
    panel.appendChild(row);
  }
}
async function loadAnalytics() {
  if (!current) return;
  const project = current;
  const data = await api('GET', '/api/jobs/' + encodeURIComponent(project) + '/analytics');
  if (current === project) renderAnalytics(data.imports || [], data.advice);
}
$('analyticsPanel').addEventListener('toggle', () => {
  if ($('analyticsPanel').open) loadAnalytics().catch(error => { $('analyticsMsg').textContent = error.message; });
});
$('analyticsUploadBtn').onclick = async () => {
  const file = $('studioExport').files[0];
  if (!file || file.size < 1 || file.size > 2000000 || !/[.](csv|json)$/i.test(file.name)) {
    $('analyticsMsg').textContent = 'Chọn file CSV/JSON từ Studio, dung lượng 1 byte–2 MB.'; return;
  }
  const button = $('analyticsUploadBtn');
  button.disabled = true;
  try {
    const headers = {'X-Studio-Format': file.name.split('.').pop().toLowerCase(),
                     'X-Studio-Notes': encodeURIComponent($('analyticsNotes').value)};
    if ($('analyticsCTA').value.trim()) headers['X-Studio-CTA-Seconds'] = $('analyticsCTA').value.trim();
    if (_tok) headers['Authorization'] = 'Bearer ' + _tok;
    const response = await fetch('/api/jobs/' + encodeURIComponent(current) + '/analytics',
                                 {method: 'POST', headers, body: file});
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error_vi || payload.error || 'Không nhập được số liệu.');
    $('analyticsMsg').textContent = 'Đã lưu số liệu cho đúng phiên bản bàn giao.';
    await loadAnalytics();
  } catch (error) { $('analyticsMsg').textContent = error.message; }
  finally { button.disabled = false; }
};
$('studioGuideBtn').onclick = () => { const d = $('studioGuide'); if (d.showModal) { try { d.showModal(); } catch (e) { d.setAttribute('open', ''); } } else { d.setAttribute('open', ''); } };
$('studioGuideClose').onclick = () => { const d = $('studioGuide'); if (d.close) { try { d.close(); } catch (e) { d.removeAttribute('open'); } } else { d.removeAttribute('open'); } };
let batchPoller = null;
function renderBatch(state) {
  const list = $('batchList');
  list.replaceChildren();
  const label = {pending:'chờ', running:'đang chạy', ready:'xong (chờ duyệt)', failed:'lỗi', cancelled:'đã hủy'};
  for (const it of (state.items || [])) {
    const row = document.createElement('div');
    row.textContent = '#' + (it.index + 1) + ' · ' + (label[it.status] || it.status)
      + (it.job_id ? ' · ' + it.job_id : '') + (it.error ? ' · ' + it.error : '') + ' · ' + it.url;
    list.appendChild(row);
  }
  $('batchStopBtn').hidden = !state.running;
  $('batchStartBtn').disabled = !!state.running;
  if (state.running && !batchPoller) batchPoller = setInterval(loadBatch, 2000);
  if (!state.running && batchPoller) { clearInterval(batchPoller); batchPoller = null; }
}
async function loadBatch() {
  try { renderBatch(await api('GET', '/api/batch')); }
  catch (e) { if (batchPoller) { clearInterval(batchPoller); batchPoller = null; } }
}
$('batchStartBtn').onclick = async () => {
  $('batchMsg').textContent = 'Đang khởi động hàng đợi…';
  try {
    const state = await api('POST', '/api/batch', { links: $('batchLinks').value, confirm_rights: $('batchRights').checked });
    $('batchMsg').textContent = 'Hàng đợi đang chạy nền. Có thể đóng tab; tiến trình vẫn chạy.';
    renderBatch(state);
    await loadJobs();
  } catch (e) { $('batchMsg').textContent = e.message; }
};
$('batchStopBtn').onclick = async () => {
  try { renderBatch(await api('POST', '/api/batch/stop', {})); } catch (e) { $('batchMsg').textContent = e.message; }
};
loadBatch();
$('handoffBtn').onclick = async () => {
  const button = $('handoffBtn');
  button.disabled = true;
  $('handoffMsg').textContent = 'Đang kiểm tra và đóng gói…';
  try {
    const result = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/handoff', {});
    const link = document.createElement('a');
    link.href = result.href; link.download = result.name;
    document.body.appendChild(link); link.click(); link.remove();
    $('handoffMsg').textContent = 'Đã tạo gói video, phụ đề, ảnh bìa, thông tin đăng và mã kiểm tra.';
    await loadStatus();
  } catch (error) { $('handoffMsg').textContent = error.message; }
  finally { button.disabled = false; }
};
$('runBtn').onclick = async () => {
  try { await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/run', {}); loadStatus(); }
  catch (e) { $('runErr').textContent = e.message; }
};
$('refreshBtn').onclick = () => { mediaLoaded = null; editorLoaded = null; loadStatus(); };
$('mediaSearchBtn').onclick = () => renderMediaExplorer($('mediaSearch').value.trim());
$('mediaSearch').onkeydown = event => { if (event.key === 'Enter') renderMediaExplorer($('mediaSearch').value.trim()); };
document.querySelectorAll('[data-media-filter]').forEach(tab => tab.onclick = () => activateMediaFilter(tab.dataset.mediaFilter));
$('mediaResults').addEventListener('scroll', () => { manualTranscriptScrollUntil = Date.now() + 4000; }, {passive: true});
$('chatQuestion').addEventListener('keydown', event => { if (event.key === 'Enter') askVideo().catch(showError); });
$('sourceVideo').addEventListener('timeupdate', event => {
  const curTime = event.currentTarget.currentTime;
  $('playerTime').textContent = formatTime(curTime);
  syncActiveTranscript(curTime);
  if (activeTimelinePreviewIndex >= 0 && currentTimelineData && Array.isArray(currentTimelineData.clips)) {
    const curClip = currentTimelineData.clips.find(c => c.timeline_index === activeTimelinePreviewIndex);
    const endSec = Number(curClip && curClip.source_clip ? curClip.source_clip.end_seconds : 0);
    if (endSec > 0 && curTime >= endSec) {
      event.currentTarget.pause();
    }
  }
});
$('sourceVideo').addEventListener('pause', () => {
  activeTimelinePreviewIndex = -1;
  document.querySelectorAll('.timeline-preview-btn').forEach(btn => {
    btn.textContent = '▶ Xem trước';
  });
});
$('sourceVideo').addEventListener('ended', () => {
  activeTimelinePreviewIndex = -1;
  document.querySelectorAll('.timeline-preview-btn').forEach(btn => {
    btn.textContent = '▶ Xem trước';
  });
});

$('saveMetaBtn').onclick = async () => {
  const tags = $('metaTags').value.split(',').map(t => t.trim()).filter(Boolean);
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/metadata',
      { title: $('metaTitle').value, description: $('metaDesc').value, tags });
    $('metaMsg').innerHTML = '<span class="ok">Đã lưu. Cần duyệt lại trước khi xuất bản.</span>';
    metaLoaded = null; loadStatus();
  } catch (e) { $('metaMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('approveBtn').onclick = async () => {
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/metadata/approve', {});
    $('metaMsg').innerHTML = '<span class="ok">Đã duyệt siêu dữ liệu.</span>';
    loadStatus();
  } catch (e) { $('metaMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('confirmPub').onchange = () => loadStatus();

$('publishBtn').onclick = async () => {
  try {
    const r = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/publish', { confirm: true });
    $('pubMsg').innerHTML = r.published
      ? '<span class="ok">' + (r.message_vi || 'Đã tạo bản ghi xuất bản.') + '</span>'
      : '<span class="warn">' + (r.message_vi || 'Chưa thể xuất bản.') + '</span>';
    loadStatus();
  } catch (e) { $('pubMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('tagScriptBtn').onclick = async () => {
  const input = $('tagNarration');
  const start = Array.from(input.value.slice(0, input.selectionStart)).length;
  const end = Array.from(input.value.slice(0, input.selectionEnd)).length;
  if (start === end) { $('tagScriptMsg').textContent = 'Hãy bôi đen một đoạn lời dẫn.'; return; }
  const refs = $('tagEvidence').value.split(',').map(ref => ref.trim()).filter(Boolean);
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/script/tag', {
      section_index: Number($('tagSection').value), start, end,
      kind: $('tagKind').value, evidence_refs: refs
    });
    $('tagScriptMsg').textContent = 'Đã gắn nhãn. Hãy kiểm tra mốc nguồn rồi duyệt lại kịch bản.';
    scriptLoaded = null; await loadStatus();
  } catch (error) { $('tagScriptMsg').textContent = error.message; }
};

$('saveScriptBtn').onclick = async () => {
  try {
    const sections = JSON.parse($('scriptSections').value || '[]');
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/script', { sections });
    $('scriptMsg').innerHTML = '<span class="ok">Đã lưu. Cần duyệt lại trước khi xuất bản.</span>';
    scriptLoaded = null; loadStatus();
  } catch (e) { $('scriptMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};

$('approveScriptBtn').onclick = async () => {
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/script/approve', {});
    $('scriptMsg').innerHTML = '<span class="ok">Đã duyệt kịch bản.</span>';
    loadStatus();
  } catch (e) { $('scriptMsg').innerHTML = '<span class="err">' + e.message + '</span>'; }
};


function creatorBriefFields() {
  const fields = $('createForm').elements;
  return {
    review_thesis: fields.namedItem('review_thesis').value.trim(),
    tone: fields.namedItem('tone').value.trim(),
    target_audience: fields.namedItem('target_audience').value.trim(),
    spoiler_policy: fields.namedItem('spoiler_policy').value,
    forbidden_claims: fields.namedItem('forbidden_claims').value.split(String.fromCharCode(10)).map(value => value.trim()).filter(Boolean),
  };
}

async function loadCreatorBriefs() {
  const {briefs} = await api('GET', '/api/creator-library/briefs');
  const menu = $('briefTemplateSelect');
  menu.replaceChildren(new Option('Chọn mẫu để điền form…', ''));
  for (const brief of briefs) menu.add(new Option(brief.name, brief.name));
  return briefs;
}
$('applyBriefTemplate').onclick = async () => {
  try {
    const name = $('briefTemplateSelect').value;
    const brief = (await loadCreatorBriefs()).find(item => item.name === name);
    if (!brief) { $('briefTemplateMsg').textContent = 'Chọn một mẫu brief.'; return; }
    const fields = $('createForm').elements;
    for (const key of ['review_thesis', 'tone', 'target_audience', 'spoiler_policy']) {
      fields.namedItem(key).value = brief[key] || (key === 'spoiler_policy' ? 'unspecified' : '');
    }
    fields.namedItem('forbidden_claims').value = (brief.forbidden_claims || []).join(String.fromCharCode(10));
    $('briefTemplateMsg').textContent = 'Đã điền mẫu; bạn có thể sửa trước khi tạo project.';
  } catch (error) { $('briefTemplateMsg').textContent = error.message; }
};
$('saveBriefTemplate').onclick = async () => {
  try {
    await api('POST', '/api/creator-library/briefs', {
      name: $('briefTemplateName').value, brief: creatorBriefFields(),
    });
    await loadCreatorBriefs();
    $('briefTemplateSelect').value = $('briefTemplateName').value.trim();
    $('briefTemplateMsg').textContent = 'Đã lưu mẫu brief.';
  } catch (error) { $('briefTemplateMsg').textContent = error.message; }
};

async function loadCreatorSeries() {
  const {series} = await api('GET', '/api/creator-library/series');
  const panel = $('seriesList');
  panel.replaceChildren();
  if (!series.length) { panel.textContent = 'Chưa có kế hoạch series.'; return; }
  for (const plan of series) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = plan.title + ' · ' + plan.entries.length + ' phim';
    button.onclick = () => {
      $('seriesId').value = plan.series_id;
      $('seriesTitle').value = plan.title;
      $('seriesEntries').value = plan.entries.map(entry => entry.movie_title +
        (entry.job_id ? ' | ' + entry.job_id : '')).join(String.fromCharCode(10));
      $('seriesMsg').textContent = 'Đã tải kế hoạch; lưu lại để cập nhật.';
    };
    panel.append(button);
  }
}
$('saveSeriesBtn').onclick = async () => {
  try {
    const entries = $('seriesEntries').value.split(String.fromCharCode(10)).map(line => line.trim())
      .filter(Boolean).map(line => {
        const separator = line.lastIndexOf('|');
        return {movie_title: (separator < 0 ? line : line.slice(0, separator)).trim(),
          job_id: separator < 0 ? null : line.slice(separator + 1).trim() || null};
      });
    await api('POST', '/api/creator-library/series', {
      series_id: $('seriesId').value.trim(), title: $('seriesTitle').value.trim(), entries,
    });
    $('seriesMsg').textContent = 'Đã lưu kế hoạch series.';
    await loadCreatorSeries();
  } catch (error) { $('seriesMsg').textContent = error.message; }
};
$('creatorProjectSearchBtn').onclick = async () => {
  try {
    const query = $('creatorProjectSearch').value.trim();
    const {projects} = await api('GET', '/api/creator-library/search?q=' + encodeURIComponent(query));
    const panel = $('creatorProjectResults');
    panel.replaceChildren();
    if (!projects.length) { panel.textContent = 'Không tìm thấy project.'; return; }
    for (const project of projects) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = (project.movie_title || project.job_id) + ' · ' + project.job_id;
      button.onclick = () => selectJob(project.job_id);
      panel.append(button);
    }
  } catch (error) { $('creatorProjectResults').textContent = error.message; }
};
$('creatorProjectSearch').onkeydown = event => {
  if (event.key === 'Enter') { event.preventDefault(); $('creatorProjectSearchBtn').click(); }
};

async function loadRights() {
  const jobId = current;
  if (!jobId) return;
  try {
    const {rights} = await api('GET', '/api/jobs/' + encodeURIComponent(jobId) + '/rights');
    if (current !== jobId) return;
    const panel = $('rightsList');
    panel.replaceChildren();
    if (!rights.length) { panel.textContent = 'Chưa ghi nhận quyền sử dụng cho tài sản nào.'; return; }
    for (const entry of rights) {
      const line = document.createElement('div');
      line.className = 'project-card';
      line.textContent = entry.path + ' · ' + entry.permission_status + ' · ' +
        entry.source + ' · ' + entry.usage + (entry.evidence_note ? ' · ' + entry.evidence_note : '');
      panel.append(line);
    }
  } catch (error) { $('rightsMsg').textContent = error.message; }
}
$('saveRightsBtn').onclick = async () => {
  if (!current) return;
  try {
    await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/rights', {
      path: $('rightsAssetPath').value.trim(), source: $('rightsSource').value.trim(),
      usage: $('rightsUsage').value.trim(), permission_status: $('rightsStatus').value,
      evidence_note: $('rightsEvidence').value.trim(),
    });
    $('rightsMsg').textContent = 'Đã lưu ghi chú nguồn và quyền.';
    await loadRights();
  } catch (error) { $('rightsMsg').textContent = error.message; }
};
loadCreatorBriefs().catch(error => { $('briefTemplateMsg').textContent = error.message; });
loadCreatorSeries().catch(error => { $('seriesMsg').textContent = error.message; });

// --- Content Scout Engine (Săn phim tự động) ---
let _scoutLoaded = false;
// Curated and AGY-localized cards already carry a real Vietnamese film title.
// Unlocalized live finds only have a generic genre label, so lead with the
// real upload title (minus "| Full Movie | cast" noise) instead.
function scoutIsUnlocalizedLive(gem) {
  return /^(yt-live-|bili-)/.test(gem.id || '') && gem.localized !== 'agy';
}
function scoutDisplayTitle(gem) {
  if (!scoutIsUnlocalizedLive(gem)) return gem.vietnamese_title || gem.title || '';
  const raw = String(gem.title || '').replace(/\\s+/g, ' ').trim();
  const head = raw.split(/\\s+[|｜]\\s+/)[0]
    .replace(/\\s*[-–—:]?\\s*\\b(full(\\s+length)?\\s+(movie|film))\\b.*$/i, '').trim();
  return head || raw || gem.vietnamese_title || '';
}
function scoutSourceExcerpt(gem) {
  if (!scoutIsUnlocalizedLive(gem)) return '';
  const text = String(gem.summary || '').replace(/\\s+/g, ' ').trim();
  if (!text || text === String(gem.title || '').trim()) return '';
  return text.length > 220 ? text.slice(0, 217).trimEnd() + '…' : text;
}
function escapeScoutHtml(str) {
  return String(str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
function getScoutSourceBadge(gem) {
  if (gem.source === 'youtube_obscure' || (gem.source_url && (gem.source_url.includes('youtube.com') || gem.source_url.includes('youtu.be')) && gem.source !== 'tmdb_douban')) {
    return 'YouTube';
  }
  if (gem.source === 'douyin_bilibili' || (gem.source_url && gem.source_url.includes('bilibili.com'))) {
    return 'Bilibili';
  }
  if (gem.source === 'tmdb_douban') {
    return (['CN', 'HK', 'TW'].includes(gem.country)) ? 'Douban' : 'TMDb';
  }
  if (gem.source === 'tmdb') return 'TMDb';
  if (gem.source === 'douban') return 'Douban';
  if (gem.source === 'youtube') return 'YouTube';
  if (gem.source === 'bilibili') return 'Bilibili';
  return gem.source || 'TMDb';
}

function getScoutThumbnail(gem) {
  if (gem.thumbnail) return gem.thumbnail;
  const ytMatch = String(gem.source_url || '').match(/(?:v=|youtu\\.be\\/|embed\\/)([a-zA-Z0-9_-]{11})/);
  if (ytMatch && ytMatch[1]) {
    return 'https://i.ytimg.com/vi/' + ytMatch[1] + '/hqdefault.jpg';
  }
  return '';
}

async function loadScoutGems(forceRefresh) {
  const topic = $('scoutTopicSelect') ? $('scoutTopicSelect').value : 'all';
  const source = $('scoutSourceSelect') ? $('scoutSourceSelect').value : 'all';
  const grid = $('scoutGrid');
  const loading = $('scoutLoading');
  const empty = $('scoutEmpty');
  if (!grid) return;
  if (loading) loading.hidden = false;
  if (empty) empty.hidden = true;
  grid.innerHTML = '';
  try {
    const refreshQuery = forceRefresh ? '&refresh=1' : '';
    const res = await api('GET', '/api/scout/discover?topic=' + encodeURIComponent(topic) + '&source=' + encodeURIComponent(source) + refreshQuery);
    const candidates = res.candidates || [];
    if (loading) loading.hidden = true;
    if (candidates.length === 0) {
      if (empty) empty.hidden = false;
      return;
    }
    for (const gem of candidates) {
      const card = document.createElement('div');
      card.className = 'card scout-card';
      card.style.cssText = 'display:flex; flex-direction:column; justify-content:space-between; border:1px solid var(--line); border-radius:8px; padding:16px; background:var(--panel); color:var(--text)';

      const sourceBadge = getScoutSourceBadge(gem);
      const scoreFormatted = Number(gem.viral_score).toLocaleString();
      const posterUrl = getScoutThumbnail(gem);
      const isLive = gem.is_live !== false;
      const healthText = isLive ? 'Sống' : 'Cần kiểm tra';
      const healthColor = isLive ? '#059669' : '#d97706';
      const healthBg = isLive ? 'rgba(16,185,129,0.15)' : 'rgba(245,158,11,0.15)';
      const healthIcon = isLive ? '🟢' : '⚠️';
      const durationText = gem.duration_minutes ? (gem.duration_minutes + ' phút') : 'Chưa rõ';
      const fallbackUrl = gem.fallback_url || ('https://www.youtube.com/results?search_query=' + encodeURIComponent((gem.title || '') + ' full movie'));
      const displayTitle = scoutDisplayTitle(gem);
      const hookLabel = scoutIsUnlocalizedLive(gem) && gem.vietnamese_title !== displayTitle ? gem.vietnamese_title : '';
      const sourceExcerpt = scoutSourceExcerpt(gem);

      card.innerHTML = `
        <div>
          ${posterUrl ? `
          <div class="scout-poster-box" style="position:relative; width:100%; height:150px; border-radius:6px; overflow:hidden; background:var(--field-bg); margin-bottom:10px; display:flex; align-items:center; justify-content:center">
            <img referrerpolicy="no-referrer" src="${escapeScoutHtml(posterUrl)}" alt="${escapeScoutHtml(displayTitle)}" style="width:100%; height:100%; object-fit:cover" onerror="this.style.display='none'; if(this.nextElementSibling) this.nextElementSibling.style.display='flex'">
            <div class="scout-poster-placeholder" style="display:none; width:100%; height:100%; align-items:center; justify-content:center; font-size:2rem; color:var(--text-muted); background:var(--field-bg)">🎬</div>
          </div>
          ` : `
          <div class="scout-poster-box" style="position:relative; width:100%; height:100px; border-radius:6px; overflow:hidden; background:var(--field-bg); margin-bottom:10px; display:flex; align-items:center; justify-content:center; font-size:1.8rem; color:var(--text-muted)">🎬</div>
          `}
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:6px; margin-bottom:8px">
            <div style="display:flex; align-items:center; gap:6px; flex-wrap:wrap">
              <span class="badge scout-source-badge" style="font-size:0.75rem; font-weight:700; text-transform:uppercase; padding:2px 8px; border-radius:12px; background:rgba(59,130,246,0.18); color:var(--accent, #3b82f6)">${escapeScoutHtml(sourceBadge)}</span>
              <span class="badge scout-duration-badge" style="font-size:0.75rem; font-weight:600; padding:2px 8px; border-radius:12px; background:var(--field-bg); border:1px solid var(--line); color:var(--text-muted)">⏱️ ${escapeScoutHtml(durationText)}</span>
              <span class="badge scout-health-badge" style="font-size:0.75rem; font-weight:700; padding:2px 8px; border-radius:12px; background:${healthBg}; color:${healthColor}">${healthIcon} ${healthText}</span>
            </div>
            <span class="badge scout-viral-badge" style="font-size:0.85rem; font-weight:800; color:#d97706; background:rgba(245,158,11,0.15); padding:2px 8px; border-radius:12px">🔥 Tiềm năng lan truyền: ${scoreFormatted}</span>
          </div>
          <h3 class="scout-title" style="margin:4px 0 2px 0; font-size:1.05rem; line-height:1.3; color:var(--text)" title="${escapeScoutHtml(gem.title)}">${escapeScoutHtml(displayTitle)}</h3>
          ${hookLabel ? `<div class="scout-hook" style="font-size:0.8rem; font-weight:700; color:var(--accent, #d97706); margin-bottom:2px">${escapeScoutHtml(hookLabel)}</div>` : ''}
          <div style="font-size:0.8rem; font-weight:600; color:var(--text-muted); margin-bottom:8px">${gem.release_year} · ${escapeScoutHtml(gem.country)} · ★ ${gem.rating}/10 (${gem.vote_count.toLocaleString()} lượt bình chọn)</div>
          ${gem.vietnamese_summary ? `<div style="font-size:0.83rem; line-height:1.4; color:var(--text-dim, var(--text)); margin-bottom:12px">${escapeScoutHtml(gem.vietnamese_summary)}</div>` : ''}
          ${sourceExcerpt ? `<div class="scout-source-excerpt" style="font-size:0.8rem; line-height:1.4; color:var(--text-muted); margin-bottom:12px"><strong>Mô tả gốc:</strong> ${escapeScoutHtml(sourceExcerpt)}</div>` : ''}
          <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:12px">
            <span style="font-size:0.75rem; font-weight:600; padding:2px 6px; border-radius:4px; background:rgba(16,185,129,0.15); color:#059669">Điểm kịch tính: ${gem.story_twist_index}</span>
            <span style="font-size:0.75rem; font-weight:600; padding:2px 6px; border-radius:4px; background:rgba(99,102,241,0.15); color:#6366f1">Độ phủ VN: ${gem.popularity_index}</span>
            <span style="font-size:0.75rem; font-weight:600; padding:2px 6px; border-radius:4px; background:rgba(234,179,8,0.15); color:#d97706">Bản quyền: ${gem.copyright_risk}</span>
          </div>
          <div style="font-size:0.78rem; font-style:italic; color:var(--text-muted); margin-bottom:14px">💡 ${escapeScoutHtml(gem.reasoning)}</div>
        </div>
        <div style="display:flex; gap:8px; flex-wrap:wrap; border-top:1px solid var(--line); padding-top:12px; margin-top:8px">
          <button type="button" class="primary scout-enqueue-btn" style="flex:1">Dựng review phim này</button>
          <a href="${escapeScoutHtml(gem.source_url)}" target="_blank" rel="noopener noreferrer" class="button-link" style="padding:4px 10px; font-size:0.85rem; text-decoration:none; display:inline-flex; align-items:center" title="Mở liên kết gốc trong tab mới">Mở link ↗</a>
          <a href="${escapeScoutHtml(fallbackUrl)}" target="_blank" rel="noopener noreferrer" class="button-link scout-fallback-btn" style="padding:4px 10px; font-size:0.85rem; text-decoration:none; display:inline-flex; align-items:center; gap:4px" title="Tìm phim thay thế trên YouTube">🔍 Tìm YouTube</a>
          <button type="button" class="scout-copy-btn" title="Chép link gốc">Chép link</button>
        </div>
      `;
      card.querySelector('.scout-enqueue-btn').onclick = () => enqueueScoutGem(gem);
      card.querySelector('.scout-copy-btn').onclick = async () => {
        try {
          await navigator.clipboard.writeText(gem.source_url);
          alert('Đã chép link nguồn: ' + gem.source_url);
        } catch (_) {
          prompt('Link nguồn:', gem.source_url);
        }
      };
      grid.appendChild(card);
    }
    _scoutLoaded = true;
  } catch (err) {
    if (loading) loading.hidden = true;
    showError(err);
  }
}
function enqueueScoutGem(gem) {
  pendingScoutGem = gem;
  $('scoutMovieTitle').value = scoutDisplayTitle(gem);
  $('scoutWatermarkEnabled').checked = true;
  $('scoutWatermarkDetect').value = 'color';
  setWatermarkMethod('scoutWatermarkMethod', 'propainter');
  $('scoutContentAgent').value = 'agy';
  $('scoutCopyright').value = 'balanced';
  $('scoutConfigMsg').textContent = '';
  $('scoutConfigDialog').showModal();
}
$('closeScoutConfig').onclick = $('cancelScoutConfig').onclick = () => $('scoutConfigDialog').close();
$('scoutConfigForm').onsubmit = async event => {
  event.preventDefault();
  if (!pendingScoutGem) return;
  const submit = event.submitter;
  if (submit) submit.disabled = true;
  try {
    const gem = pendingScoutGem;
    const res = await api('POST', '/api/scout/enqueue', {
      candidate_id: gem.id, auto_create: true, movie_title: $('scoutMovieTitle').value.trim(),
      watermark_enabled: $('scoutWatermarkEnabled').checked, watermark_detect: $('scoutWatermarkDetect').value,
      watermark_method: $('scoutWatermarkMethod').value,
      content_agent: $('scoutContentAgent').value, visual_variety: $('scoutCopyright').value,
      tts_provider: $('scoutTtsProvider').value, tts_voice: $('scoutTtsVoice').value.trim(),
    });
    $('scoutConfigDialog').close(); closeToolDialog('scoutPanel'); await loadJobs();
    if (res.created_job && res.created_job.job_id) {
      current = res.created_job.job_id; await selectJob(current);
      if ($('sourceRetryUrl') && gem.source_url) $('sourceRetryUrl').value = gem.source_url;
      updateSourceRetryFallback(gem.title || gem.vietnamese_title);
      setWorkspaceView('explore');
    }
  } catch (err) { $('scoutConfigMsg').textContent = err.message; }
  finally { if (submit) submit.disabled = false; }
};

function configTag(text) { const tag = document.createElement('span'); tag.className = 'badge ready'; tag.textContent = text; return tag; }
function renderProjectConfig(status) {
  const cfg = status.config || {};
  const wm = cfg.watermark_removal || {};
  const detect = wm.detect || {};
  const tags = $('projectConfigTags'); tags.replaceChildren(
    configTag('🎬 Phim: ' + (cfg.movie_title || 'chưa đặt')),
    configTag('🤖 Kịch bản: ' + (cfg.content_agent || 'scaffold')),
    configTag('🎞️ Biến đổi hình ảnh: ' + (cfg.visual_variety || 'off')),
    configTag('🎙️ Giọng: ' + (cfg.tts_provider || 'edge') + ' (' + (cfg.tts_voice || 'mặc định') + ')'),
    configTag('🧼 Xoá watermark: ' + (wm.enabled ? 'Bật (' + watermarkMethodInfo(wm.method).label + ' · ' + (detect.method || 'color') + ')' : 'Tắt'))
  );
  $('editProjectConfig').disabled = !!status.running;
}
$('editProjectConfig').onclick = () => {
  const cfg = (currentStatus && currentStatus.config) || {}; const wm = cfg.watermark_removal || {}; const detect = wm.detect || {};
  $('projectMovieTitle').value = cfg.movie_title || ''; $('projectContentAgent').value = cfg.content_agent || 'scaffold';
  $('projectCopyright').value = cfg.visual_variety || 'off'; $('projectTtsProvider').value = cfg.tts_provider || 'edge';
  $('projectTtsVoice').value = cfg.tts_voice || ''; $('projectWatermarkEnabled').checked = !!wm.enabled; $('projectWatermarkDetect').value = detect.method || 'color';
  setWatermarkMethod('projectWatermarkMethod', wm.method);
  $('projectConfigMsg').textContent = ''; $('projectConfigDialog').showModal();
};
$('closeProjectConfig').onclick = $('cancelProjectConfig').onclick = () => $('projectConfigDialog').close();
$('projectConfigForm').onsubmit = async event => {
  event.preventDefault(); const save = $('saveProjectConfig'); save.disabled = true;
  try {
    const result = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/config', {
      movie_title: $('projectMovieTitle').value.trim(), content_agent: $('projectContentAgent').value,
      visual_variety: $('projectCopyright').value, tts_provider: $('projectTtsProvider').value,
      tts_voice: $('projectTtsVoice').value.trim(), watermark_removal: {
        enabled: $('projectWatermarkEnabled').checked, method: $('projectWatermarkMethod').value,
        detect: {method: $('projectWatermarkDetect').value}
      }
    });
    $('projectConfigDialog').close(); currentStatus = result; renderProjectConfig(result);
    $('projectConfigWarning').textContent = (result.config_warnings || []).join(' '); await loadStatus();
  } catch (err) { $('projectConfigMsg').textContent = err.message; }
  finally { save.disabled = false; }
};
if ($('scoutRefreshBtn')) $('scoutRefreshBtn').onclick = () => loadScoutGems(true);
if ($('scoutTopicSelect')) $('scoutTopicSelect').onchange = () => loadScoutGems(false);
if ($('scoutSourceSelect')) $('scoutSourceSelect').onchange = () => loadScoutGems(false);

loadJobs();
loadSavedSearches().catch(() => {});
$('chatAskBtn').addEventListener('click', () => askVideo().catch(showError));
</script>
</body>
</html>
""".replace("/*@ui-fonts*/", _UI_FONT_FACES)
