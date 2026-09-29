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
import queue
import re
import shutil
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import analytics, audio_mix, branding, cancellation, creative_brief, editor_ops, localization, midroll, pipeline, semantic_search, versions
from .content_agent import _terminate_process_tree
from .media_store import MediaStore
from .creator_library import CreatorLibrary
from . import media_intelligence
from .models import CONTENT_AGENT_MODES, JobConfig

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
}

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
        self._uploads: set[str] = set()
        self._deleting: set[str] = set()
        self._lock = threading.Lock()
        self._version_lock = threading.Lock()
        # Background indexing queue (roadmap #14): a single FIFO worker builds
        # media_index/embeddings/scene-memory for imported jobs one at a time so
        # import returns immediately and other projects stay usable meanwhile.
        self._indexing: dict[str, dict] = {}
        self._index_queue: "queue.Queue[str]" = queue.Queue()
        self._index_cv = threading.Condition(self._lock)
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
        info["stopping"] = bool(run_state.get("stopping"))
        info["cancelled"] = any(stage["status"] == "cancelled" for stage in info["stages"])
        info["can_stop"] = info["running"] and not info["stopping"]
        with self._lock:
            info["uploading"] = job_id in self._uploads
        cfg = pipeline.load_manifest(root).config
        info["has_source_video"] = bool(cfg.source_video)
        info["brand_top_band"] = cfg.brand_top_band
        info["brand_bottom_band"] = cfg.brand_bottom_band
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
        try:
            pipeline.subprocess.run([ffmpeg, "-y", "-ss", f"{start:.6f}", "-i", str(self.source_media_path(job_id)), "-t", f"{duration:.6f}", "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2", "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", "-movflags", "+faststart", str(temporary)], capture_output=True, text=True, check=True, shell=False)
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
        return {"imports": analytics.list_imports(self._require_job(job_id))}

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
        config = JobConfig(
            job_id=job_id,
            language=str(payload.get("language") or "vi"),
            target_minutes=float(payload.get("target_minutes") or 10),
            aspect_ratio=str(payload.get("aspect_ratio") or "16:9"),
            source_video=Path(source_video) if source_video else None,
            movie_title=str(payload.get("movie_title") or "").strip() or None,
            creative_brief=payload.get("creative_brief") or {},
            content_agent=content_agent,
            brand_top_band=float(payload.get("brand_top_band") or 0),
            brand_bottom_band=float(payload.get("brand_bottom_band") or 0),
        )
        pipeline.create_job(root, config)
        return self.status(job_id)

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

    def start_run(self, job_id: str, *, until: str | None = None) -> dict:
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
        prompt = (
            "Viết một lời thoại CTA bằng tiếng Việt, 25–40 từ, một hoặc hai câu, "
            "hài hước tự nhiên theo chi tiết của phim. Nhắc bấm thích và đăng ký "
            f"kênh {brand_name} để không bỏ lỡ phần tiếp theo. Chèn ở 50% video giữa "
            f"'{previous}' và '{following}' của '{manifest.config.movie_title}'. "
            "Không bịa sự kiện, không lặp nguyên văn thoại phim. Trả JSON đúng schema."
        )
        result = run_agy_json(stage="midroll", prompt=prompt, schema=schema)
        line = str(result.get("line") or "").strip()
        render = self._read_json(root, "render.json")
        _, _, at = midroll.prepare(script, plan, line, 10, render.get("narration_duration_seconds"))
        draft = {"line": line, "start_seconds": at, "duration_seconds": 10, "generator": "agy"}
        pipeline._write_json(root, "midroll-draft.json", draft)
        return draft

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


# --- HTTP layer --------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], service: JobsService):
        self.service = service
        super().__init__(address, MRFRequestHandler)


class MRFRequestHandler(BaseHTTPRequestHandler):
    server_version = "MovieReviewFactory/0.2"

    @property
    def service(self) -> JobsService:
        return self.server.service  # type: ignore[attr-defined]

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
        raw = self.rfile.read(length)
        if not raw:
            return {}
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
        except ValueError as exc:
            self._send_json(400, self._error_payload(exc))
        except FileNotFoundError as exc:
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
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    return  # Browser stopped reading after a seek or navigation.
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
            self._send_html(INDEX_HTML)
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
        if parts == ["api", "jobs"] or (len(parts) == 3 and parts[:2] == ["api", "jobs"]):
            self._dispatch(lambda: self._route_delete(parts))
            return
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def _route_delete(self, parts: list[str]) -> None:
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
        if parts == ["brand"]:
            self._send_json(200, {**branding.load_settings(self.service.jobs_root), "logo_url": "/api/brand/logo"})
            return
        if parts == ["brand", "logo"]:
            self._serve_file(branding.logo_path(self.service.jobs_root))
            return
        if parts == ["brand", "logo.svg"]:
            self._serve_file(branding.ASSETS / "man-ke.svg")
            return
        if parts == ["agy-pool"]:
            self._send_json(200, self.service.agy_pool_status())
            return
        if parts == ["jobs"]:
            self._send_json(200, {"jobs": self.service.list_jobs()})
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
        self._send_json(404, {"error": "not found", "error_vi": "không tìm thấy"})

    def _route_post(self, parts: list[str]) -> None:
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
        if parts == ["agy-pool", "probe"]:
            self._send_json(200, self.service.probe_agy())
            return
        if parts == ["jobs"]:
            self._send_json(201, self.service.create_job(self._read_body()))
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
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "stop":
            self._send_json(202, self.service.stop_run(parts[1]))
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
                  jobs_root: Path | str = "jobs") -> _Server:
    """Build (but do not start) the local web server."""
    return _Server((host, port), JobsService(Path(jobs_root)))


def run_server(host: str = "127.0.0.1", port: int = 8765,
               jobs_root: Path | str = "jobs") -> None:
    """Start the local web server and serve until interrupted."""
    server = create_server(host, port, jobs_root)
    bound_host, bound_port = server.server_address[:2]
    print(f"Movie Review Factory UI: http://{bound_host}:{bound_port}  (jobs: {Path(jobs_root)})")
    print("Nhấn Ctrl+C để dừng.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


# --- single-page UI ----------------------------------------------------------
# Vanilla HTML + JS (no framework, no build step). Kept as one inline document so the whole UI ships with the package and needs no static-file plumbing.

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
  .media-list { display: grid; align-content: start; gap: 6px; padding: 10px; max-height: 620px; overflow-y: auto; overscroll-behavior: contain; scroll-behavior: smooth; }
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
  .media-list { max-height: min(62dvh, 650px); }
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
<header>
  <div class="brand">
    <span class="brand-mark" aria-hidden="true">🎬</span>
    <div class="brand-text">
      <h1>Xưởng Review Phim</h1>
      <div class="sub">Import → khám phá → biên tập → duyệt và xuất</div>
    </div>
  </div>
  <button id="themeToggle" class="theme-toggle" type="button" title="Đổi giao diện sáng/tối" aria-label="Đổi giao diện sáng/tối" aria-pressed="false">
    <span class="theme-toggle-icon" aria-hidden="true">🌙</span>
    <span class="theme-toggle-label">Tối</span>
  </button>
</header>
<script>
  (function(){
    var btn = document.getElementById('themeToggle');
    if(!btn){ return; }
    function sync(){
      var cur = document.documentElement.getAttribute('data-theme') || 'dark';
      var icon = btn.querySelector('.theme-toggle-icon');
      var label = btn.querySelector('.theme-toggle-label');
      if(icon){ icon.textContent = cur === 'light' ? '☀️' : '🌙'; }
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
    <details class="card project-panel" id="projectPanel">
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
        <div id="jobList" class="project-list" aria-live="polite">Đang tải…</div>
      </div>
    </details>
    <button id="openCreateProject" class="primary toolbar-action" type="button">＋ Tạo project</button>
    <button id="openLibraryHub" class="toolbar-action" type="button">Thư viện</button>
    <button id="openBrandSettings" class="toolbar-action" type="button">Thương hiệu</button>
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
          <input id="newJobId" name="job_id" type="hidden">
          <label>Tên phim / truy vấn nghiên cứu</label>
          <input name="movie_title" placeholder="vd: The Matrix (1999)">
          <details class="advanced-fields"><summary>Định hướng review</summary>
            <label for="briefTemplateSelect">Mẫu brief dùng lại</label>
            <select id="briefTemplateSelect"><option value="">Chọn mẫu để điền form…</option></select>
            <button id="applyBriefTemplate" type="button">Áp dụng mẫu</button>
            <label for="briefTemplateName">Lưu các trường bên dưới thành mẫu mới</label>
            <input id="briefTemplateName" maxlength="200" placeholder="Tên mẫu brief">
            <button id="saveBriefTemplate" type="button">Lưu mẫu brief</button>
            <span id="briefTemplateMsg" class="notice" role="status"></span>
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
            <label>Che dải watermark phía trên (0–20% chiều cao)</label>
            <input name="brand_top_band" type="number" value="0" min="0" max="0.2" step="0.01">
            <label>Che dải tiêu đề cũ phía dưới (0–20% chiều cao)</label>
            <input name="brand_bottom_band" type="number" value="0" min="0" max="0.2" step="0.01">
          </details>
          <details><summary class="muted">Hoặc nhập đường dẫn cục bộ</summary>
            <label for="sourcePath">Đường dẫn video trên máy chạy ứng dụng</label>
            <input id="sourcePath" name="source_video" placeholder="data\\raw\\....mp4">
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
            <div class="settings-row">
              <div><label for="brandTopBand">Dải phía trên (0–0,2)</label><input id="brandTopBand" type="number" min="0" max="0.2" step="0.01" value="0"></div>
              <div><label for="brandBottomBand">Dải phía dưới (0–0,2)</label><input id="brandBottomBand" type="number" min="0" max="0.2" step="0.01" value="0"></div>
            </div>
            <button id="brandRenderBtn" type="button" disabled>Dựng lại video đang chọn</button>
          </section>
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
  <main>
    <div id="empty" class="card empty-state" role="status">
      <div class="empty-state-mark" aria-hidden="true">＋</div>
      <h2>Bắt đầu một video review</h2>
      <p>Chọn MP4 để tạo project mới. Mặc định đã đủ để bắt đầu; brief và tùy chọn dựng chỉ cần mở khi bạn muốn chỉnh sâu.</p>
      <div class="empty-actions">
        <button id="emptyCreateBtn" class="primary" type="button">Tạo project từ MP4</button>
        <button id="emptyProjectsBtn" type="button" hidden>Mở project có sẵn</button>
      </div>
      <div class="empty-help">Sau khi tạo, project mở thẳng vào workspace và giữ toàn bộ tiến trình ở một nơi.</div>
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
        <div id="sourceRetryCard" hidden>
          <label for="sourceRetry">Project chưa có video: chọn MP4 để import</label>
          <input id="sourceRetry" type="file" accept=".mp4,video/mp4">
          <button id="sourceRetryBtn" type="button">Import video</button>
          <div id="sourceRetryMsg" class="notice" role="status"></div>
        </div>
        <div style="margin-top:10px" class="progress"><div id="progBar"></div></div>
        <div id="progText" class="muted" style="margin-top:6px"></div>
        <div id="runErr" class="err" style="margin-top:6px"></div>
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
        <h2>Xuất video</h2>
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
          <div class="row">
            <button id="shortExportBtn" type="button">Xuất video ngắn</button>
            <a id="shortDownload" hidden download="short-review.mp4">Tải MP4 9:16</a>
            <a id="shortSrtDownload" hidden download="short-review.srt">Tải phụ đề SRT</a>
          </div>
          <span id="shortExportMsg" class="muted" role="status"></span>
        </details>
        <details id="analyticsPanel">
          <summary>Học từ số liệu YouTube Studio</summary>
          <p class="muted">Sau khi tải gói bàn giao và đăng video thủ công, nhập CSV/JSON retention đã đo. Số liệu gắn với đúng phiên bản xuất; không dự đoán lượt xem.</p>
          <label for="studioExport">File Studio (CSV/JSON, tối đa 2 MB)</label>
          <input id="studioExport" type="file" accept=".csv,.json,text/csv,application/json">
          <div class="filter-grid">
            <label for="analyticsCTA">Mốc bắt đầu CTA trong video cuối (giây, nếu có)<input id="analyticsCTA" type="number" min="0" step="0.1"></label>
            <label for="analyticsNotes">Ghi chú rút kinh nghiệm<textarea id="analyticsNotes" maxlength="2000" rows="2"></textarea></label>
          </div>
          <button id="analyticsUploadBtn" type="button">Nhập số liệu đã đo</button>
          <span id="analyticsMsg" role="status" class="muted"></span>
          <div id="analyticsResults" role="status" class="muted"></div>
        </details>
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
        <button id="saveRightsBtn" type="button">Lưu ghi chú quyền</button>
        <div id="rightsMsg" class="notice" role="status"></div>
        <div id="rightsList" class="project-list notice"></div>
      </details>
      <details id="versionsPanel" class="card">
        <summary>Phiên bản và khôi phục</summary>
        <div class="muted">Lưu mốc trước khi sửa. Bản MP4 đạt QA trước đó vẫn có thể tải về.</div>
        <label for="versionName">Tên phiên bản</label>
        <input id="versionName" type="text" maxlength="100" placeholder="Ví dụ: Trước khi sửa đoạn kết">
        <button id="saveVersionBtn" type="button">Lưu phiên bản</button>
        <label for="versionSelect">Phiên bản đã lưu</label>
        <select id="versionSelect" aria-label="Phiên bản đã lưu"></select>
        <label for="versionKind">Khôi phục phần</label>
        <select id="versionKind"><option value="script">Kịch bản</option><option value="scene_plan">Cảnh</option><option value="captions">Phụ đề</option><option value="metadata">Thông tin đăng</option><option value="final">Video đã QA</option></select>
        <button id="restoreVersionBtn" type="button">Khôi phục</button>
        <a id="versionDownload" hidden download="final.mp4">Tải bản MP4 cũ</a>
        <span id="versionMsg" class="muted" role="status"></span>
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
                <div id="highlightResults" class="highlight-grid"><div class="state-panel">Đang tải khoảnh khắc nổi bật…</div></div>
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
          <p class="muted">Chọn phần để dựng thử từ các cảnh đã chọn, giọng đọc và phụ đề có mốc thời gian. Không dựng lại toàn bộ video.</p>
          <div class="row"><label for="sectionPreviewSelect">Phần cần xem <select id="sectionPreviewSelect"></select></label>
            <button id="sectionPreviewBtn" type="button">Dựng nhanh phần này</button>
            <a id="sectionPreviewDownload" hidden download="section-preview.mp4">Tải MP4</a>
            <a id="sectionPreviewSrt" hidden download="section-preview.srt">Tải SRT</a></div>
          <video id="sectionPreviewVideo" controls preload="metadata" style="width:100%;max-height:440px" hidden aria-label="Xem trước phần đã chọn"></video>
          <div id="sectionPreviewMsg" class="notice" role="status"></div>
        </details>
        <div id="continuityTracks" class="continuity-grid"></div>
        <div id="timelineList" class="timeline-list"></div>
        <div id="editorMsg" class="notice" role="status"></div>
      </div>

      </section>
      <section id="view-review-content" class="workspace-view" hidden>
      <div class="card" id="thumbnailCard" style="display:none">
        <h2>Chọn ảnh bìa</h2>
        <div class="muted">Chọn một trong các frame đã tạo. Ảnh được chọn sẽ trở thành <code>thumbnail.jpg</code>.</div>
        <div id="thumbnailGrid" class="thumb-grid" style="margin-top:10px"></div>
        <details id="thumbnailEditor" style="margin-top:12px">
          <summary>Tạo ảnh bìa với chữ và thương hiệu</summary>
          <p class="muted">Ba phương án trên ảnh có sẵn. Chữ nằm trong vùng an toàn; xem bản thu nhỏ trước khi chọn. Thay ảnh sẽ cần duyệt lại metadata và gói xuất.</p>
          <div class="filter-grid">
            <label for="thumbnailHeadline">Dòng chính (tối đa 64 ký tự)<input id="thumbnailHeadline" maxlength="64" placeholder="BEN 10: AI ĐANG ĐIỀU KHIỂN THỜI GIAN?"></label>
            <label for="thumbnailChannel">Tên kênh trên ảnh<input id="thumbnailChannel" maxlength="40"></label>
          </div>
          <button id="thumbnailEditBtn" type="button">Tạo 3 phương án</button>
          <div id="thumbnailVariantGrid" class="thumb-grid" style="margin-top:10px"></div>
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
          <button id="tagScriptBtn" type="button" disabled>Gắn nhãn đoạn đã chọn</button>
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

      <div class="card" id="publishGateCard">
        <h2>Cổng xuất bản</h2>
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
let poller = null;
let editorLoaded = null;
// Which review block matches the current next action: 'script' | 'metadata'
// | 'thumbnail' | 'export'. loadStatus() keeps this in sync with nextAction so
// opening "Duyệt & xuất" jumps straight to the block that needs attention.
let reviewFocus = 'script';
// The card that matches each focus. Opening the review view scrolls to this
// block; the three FORM cards (script/metadata/publish) collapse so only the
// one called out by the next action is expanded. videoCard / exportCard /
// thumbnailCard keep whatever visibility their own state gave them (a ready
// preview or export must never disappear just because it is not the focus).
const REVIEW_FOCUS_TARGET = {
  script: 'scriptReviewCard',
  metadata: 'metadataReviewCard',
  thumbnail: 'thumbnailCard',
  export: 'exportCard',
};
// Only these three collapse by focus; the rest are governed by loadStatus.
const REVIEW_FORM_CARDS = ['scriptReviewCard', 'metadataReviewCard', 'publishGateCard'];
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
  for (const name of ['explore', 'edit', 'review', 'files']) $('view-' + name).hidden = name !== view;
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

// Read bearer token from URL fragment (#token=...) — fragment is never sent
// to the server so the token never appears in access logs.  Not persisted.
let _tok = '';
(function () {
  const m = location.hash.replace(/^#/, '').match(/(?:^|&)token=([^&]*)/);
  if (m) _tok = decodeURIComponent(m[1]);
})();

async function api(method, path, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) { opts.headers['Content-Type'] = 'application/json'; opts.body = JSON.stringify(body); }
  if (_tok) opts.headers['Authorization'] = 'Bearer ' + _tok;
  const res = await fetch(path, opts);
  const text = await res.text();
  const data = text ? JSON.parse(text) : {};
  if (!res.ok) { throw new Error(data.error_vi || data.error || ('HTTP ' + res.status)); }
  return data;
}

async function loadBrand() {
  const brand = await api('GET', '/api/brand');
  $('brandName').value = brand.name;
  $('brandPreviewName').textContent = brand.name;
  $('brandPreview').src = brand.logo_url + '?v=' + Date.now();
}
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
  box.innerHTML = '<div class="state-panel">Đang tải khoảnh khắc nổi bật…</div>';
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
    const preview = document.createElement('button'); preview.type = 'button'; preview.textContent = 'Xem trước';
    preview.onclick = () => seekSource(source.start_seconds);

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
    $('emptyProjectsBtn').hidden = !jobs.length;
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
  $('projectPanel').open = false;
  $('projectSummaryLabel').textContent = id;
  current = id;
  $('empty').style.display = 'none';
  $('detail').style.display = '';
  $('video').removeAttribute('src');
  delete $('video').dataset.src;
  $('video').load();
  scriptLoaded = null; metaLoaded = null;
  $('versionsPanel').open = false;
  $('analyticsPanel').open = false;
  $('analyticsResults').replaceChildren();
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
}

function closeToolDialog(id) {
  const dialog = $(id);
  if (dialog && dialog.open) dialog.close();
}

$('openCreateProject').onclick = () => openToolDialog('createPanel');
$('openBrandSettings').onclick = () => openToolDialog('brandPanel');
$('openLibraryHub').onclick = () => openToolDialog('libraryPanel');
$('closeCreateProject').onclick = () => closeToolDialog('createPanel');
$('closeBrandSettings').onclick = () => closeToolDialog('brandPanel');
$('closeLibraryHub').onclick = () => closeToolDialog('libraryPanel');
$('emptyCreateBtn').onclick = () => openToolDialog('createPanel');
$('emptyProjectsBtn').onclick = () => { $('projectPanel').open = true; };

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

async function loadStatus() {
  if (!current) return;
  let s;
  try { s = await api('GET', '/api/jobs/' + encodeURIComponent(current)); }
  catch (e) { $('progText').textContent = 'Lỗi: ' + e.message; return; }

  $('jobTitle').textContent = 'Job: ' + s.job_id;
  const total = s.stages.length;
  const ready = s.counts.ready || 0;
  $('progBar').style.width = Math.round(ready * 100 / total) + '%';
  $('progText').textContent = `${ready}/${total} bước hoàn tất · ${s.complete ? 'đã xong' : 'đang tiến hành'}`
    + (s.running ? ' · đang chạy…' : '')
    + (s.is_indexing ? ` · đang lập chỉ mục nền${s.indexing && s.indexing.stage ? ` (${s.indexing.stage} ${s.indexing.done}/${s.indexing.total})` : ''}…` : '');
  $('runErr').textContent = s.run_error_vi ? ('Lỗi chạy: ' + s.run_error_vi) : '';
  $('runBtn').disabled = !!s.running || !!s.uploading || !s.has_source_video;
  $('stopBtn').hidden = !s.running;
  $('stopBtn').disabled = !s.can_stop;
  $('deleteBtn').disabled = !!s.running || !!s.uploading;
  $('brandRenderBtn').disabled = !!s.running || !s.has_source_video;
  $('audioSaveBtn').disabled = !!s.running;
  if (document.activeElement !== $('brandTopBand')) $('brandTopBand').value = s.brand_top_band ?? 0;
  if (document.activeElement !== $('brandBottomBand')) $('brandBottomBand').value = s.brand_bottom_band ?? 0;
  $('sourceRetryCard').hidden = !!s.has_source_video;
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
  if (s.running || s.is_indexing || s.section_preview?.running) { poller = setInterval(loadStatus, 1500); }
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
    if (file) await uploadVideo(created.job_id, file);
    // The project code is auto-generated and kept hidden; confirm by name only.
    $('createMsg').textContent = file ? 'Đã tạo project và import video.' : 'Đã tạo project mới.';
    e.target.reset();
    await loadJobs();
    selectJob(created.job_id);
  } catch (err) {
    $('createMsg').textContent = (created ? 'Project đã tạo; có thể mở và thử import lại. ' : '') + err.message;
    if (created) { await loadJobs(); selectJob(created.job_id); }
  } finally { submit.disabled = false; }
};

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
      $('detail').style.display = 'none'; $('empty').style.display = '';
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
function renderAnalytics(imports) {
  const panel = $('analyticsResults');
  panel.replaceChildren();
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
  if (current === project) renderAnalytics(data.imports || []);
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
$('sourceVideo').addEventListener('timeupdate', event => { $('playerTime').textContent = formatTime(event.currentTarget.currentTime); syncActiveTranscript(event.currentTarget.currentTime); });
$('chatQuestion').addEventListener('keydown', event => { if (event.key === 'Enter') askVideo().catch(showError); });

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

loadJobs();
loadSavedSearches().catch(() => {});
$('chatAskBtn').addEventListener('click', () => askVideo().catch(showError));
</script>
</body>
</html>
"""
