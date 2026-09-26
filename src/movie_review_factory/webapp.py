"""Local web UI + JSON API for the movie-review-factory pipeline.

Python-native and dependency-free: it uses only the standard library
(``http.server`` + ``threading``) so it adds nothing to pyproject and needs no
JavaScript build step. The page is a single inline HTML document that talks to
a small JSON API. Bind to localhost - this is a single-operator tool, not a
public service.

Publishing safety is built into the shape of the API, not just the UI:

* The web "run" action stops the pipeline at the ``thumbnail`` stage
  (``until="thumbnail"``) and never reaches ``publish`` on its own.
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
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from . import cancellation, editor_ops, localization, pipeline, semantic_search
from .content_agent import _terminate_process_tree
from .media_store import MediaStore
from . import media_intelligence
from .models import CONTENT_AGENT_MODES, JobConfig

# A job id / artifact name must be a single safe path segment. This is the only thing standing between a URL and the filesystem, so it is deliberately strict.
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9._-]+$")

# The web "run" button intentionally stops here; publish is a separate action.
RUN_UNTIL_STAGE = "thumbnail"

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
        self._runs: dict[str, dict] = {}
        self._uploads: set[str] = set()
        self._deleting: set[str] = set()
        self._lock = threading.Lock()
        # Background indexing queue (roadmap #14): a single FIFO worker builds
        # media_index/embeddings/scene-memory for imported jobs one at a time so
        # import returns immediately and other projects stay usable meanwhile.
        self._indexing: dict[str, dict] = {}
        self._index_queue: "queue.Queue[str]" = queue.Queue()
        self._index_cv = threading.Condition(self._lock)
        self._index_auto = os.environ.get("MRF_AUTO_INDEX", "1").strip().lower() not in (
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

    # -- read ----------------------------------------------------------------

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
        info["stopping"] = bool(run_state.get("stopping"))
        info["cancelled"] = any(stage["status"] == "cancelled" for stage in info["stages"])
        info["can_stop"] = info["running"] and not info["stopping"]
        with self._lock:
            info["uploading"] = job_id in self._uploads
        info["has_source_video"] = bool(pipeline.load_manifest(root).config.source_video)
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
        return {"present": bool(thumbnails), "thumbnails": thumbnails}

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
        return editor_ops.edit_timeline(self._require_job(job_id), operation)

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
        return editor_ops.regenerate_section(
            self._require_job(job_id),
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
            content_agent=content_agent,
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
            if self._runs.get(job_id, {}).get("running") or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing:
                raise RuntimeError("job đang chạy hoặc đang tải video")
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
        # Import finished: kick off background indexing so the heavy transcript/
        # scene/visual/embedding work happens off the request while other
        # projects stay usable (roadmap #14). Opt out with MRF_AUTO_INDEX=0.
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
                if self._runs.get(job_id, {}).get("running") or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing:
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
                    deleted.append(job_id)
            finally:
                self._deleting.difference_update(job_ids)
        return {"deleted": deleted, "failed": failed}

    def start_run(self, job_id: str, *, until: str | None = RUN_UNTIL_STAGE) -> dict:
        root = self._require_job(job_id)
        # A background index for this job must yield before a full run starts so
        # the two never write the same media_index concurrently (roadmap #14).
        self._preempt_index(job_id)
        event = threading.Event()
        state = {"running": True, "stopping": False, "error": None, "until": until, "event": event, "process": None}
        with self._lock:
            if self._runs.get(job_id, {}).get("running") or job_id in self._uploads or job_id in self._deleting or job_id in self._indexing:
                raise RuntimeError("job đang chạy hoặc đang tải video")
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
        root = self._require_job(job_id)
        meta = pipeline.update_metadata(root, fields)
        return {"approved": bool(meta.get("approved")), "metadata": meta}

    def get_script(self, job_id: str) -> dict:
        root = self._require_job(job_id)
        script = self._read_json(root, "script.json")
        return {"present": bool(script), "script": script}

    def update_script(self, job_id: str, fields: dict) -> dict:
        root = self._require_job(job_id)
        script = pipeline.update_script(root, fields)
        return {"approved": False, "script": script}

    def select_thumbnail(self, job_id: str, candidate: str) -> dict:
        root = self._require_job(job_id)
        thumbnails = pipeline.select_thumbnail(root, candidate)
        return {
            "selected": thumbnails.get("primary_candidate", ""),
            "thumbnails": thumbnails,
        }

    def approve_script(self, job_id: str) -> dict:
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
                self.wfile.write(chunk)
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
            r"/api/jobs/[A-Za-z0-9._-]+/(?:media/source|artifacts/final\.mp4)", route
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
        if parts == ["jobs"]:
            self._send_json(200, {"jobs": self.service.list_jobs()})
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
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "artifacts":
            self._send_json(200, {"artifacts": self.service.list_artifacts(parts[1])})
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "artifacts":
            self._serve_file(self.service.artifact_path(parts[1], parts[3]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "metadata":
            self._send_json(200, self.service.get_metadata(parts[1]))
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
        if parts == ["jobs"]:
            self._send_json(201, self.service.create_job(self._read_body()))
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
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "run":
            self._send_json(202, self.service.start_run(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "index":
            self._send_json(202, self.service.enqueue_index(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "stop":
            self._send_json(202, self.service.stop_run(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "metadata":
            self._send_json(200, self.service.update_metadata(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "metadata" and parts[3] == "approve":
            self._send_json(200, self.service.approve_metadata(parts[1]))
            return
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "script":
            self._send_json(200, self.service.update_script(parts[1], self._read_body()))
            return
        if len(parts) == 4 and parts[0] == "jobs" and parts[2] == "script" and parts[3] == "approve":
            self._send_json(200, self.service.approve_script(parts[1]))
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
  :root { color-scheme: light dark; --gap: 16px; --accent: #4f7cff; }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, "Segoe UI", Roboto, sans-serif;
         line-height: 1.5; background: #0f1115; color: #e6e8ee; }
  header { padding: 14px 20px; background: #171a21; border-bottom: 1px solid #262b36; }
  header h1 { margin: 0; font-size: 18px; }
  header .sub { color: #9aa3b2; font-size: 13px; }
  .layout { display: grid; grid-template-columns: 320px 1fr; gap: var(--gap); padding: var(--gap); }
  @media (max-width: 820px) { .layout { grid-template-columns: 1fr; } }
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
  .project-list, .artifact-grid { display: grid; gap: 8px; }
  .project-toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 10px; }
  .project-toolbar label { display: flex; align-items: center; gap: 6px; margin: 0; }
  .project-toolbar input, .project-check input { width: auto; }
  .project-check { display: inline-flex; align-items: center; gap: 6px; margin: 0; }
  .delete-targets { max-height: 140px; overflow: auto; overflow-wrap: anywhere; color: var(--text-dim); }
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
  .timeline-controls { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 6px; margin-top: 8px; }
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
  @media (prefers-reduced-motion: reduce) { *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; animation-duration: .01ms !important; animation-iteration-count: 1 !important; } }
</style>
</head>
<body>
<header>
  <h1>Xưởng Review Phim — Bảng điều khiển</h1>
  <div class="sub">Chạy cục bộ · Không tự động xuất bản · Duyệt thủ công trước khi bàn giao</div>
</header>
<div class="layout">
  <aside>
    <div class="card">
      <h2>Tạo job mới</h2>
      <form id="createForm">
        <label>Mã job (job_id)</label>
        <input name="job_id" placeholder="vd: review-abc" required>
        <label>Tên phim / truy vấn nghiên cứu</label>
        <input name="movie_title" placeholder="vd: The Matrix (1999)">
        <label>Ngôn ngữ</label>
        <input name="language" value="vi">
        <label>Bộ tạo nội dung</label>
        <select name="content_agent">
          <option value="scaffold">Scaffold (offline)</option>
          <option value="claude">Claude Code (research → outline → script)</option>
          <option value="agy">AGY pool (research → outline → script)</option>
        </select>
        <label>Thời lượng mục tiêu (phút)</label>
        <input name="target_minutes" type="number" value="10" min="1" max="60" step="0.5">
        <label>Tỷ lệ khung hình</label>
        <select name="aspect_ratio"><option>16:9</option><option>9:16</option></select>
        <label for="sourceFile">Chọn video MP4 để import</label>
        <input id="sourceFile" type="file" accept=".mp4,video/mp4">
        <details><summary class="muted">Hoặc nhập đường dẫn cục bộ</summary>
          <label for="sourcePath">Đường dẫn video trên máy chạy ứng dụng</label>
          <input id="sourcePath" name="source_video" placeholder="data\\raw\\....mp4">
        </details>
        <div class="row" style="margin-top:10px">
          <button class="primary" type="submit">Tạo job</button>
        </div>
        <progress id="uploadProgress" class="upload-progress" max="100" value="0" hidden></progress>
        <div id="createMsg" class="notice" role="status"></div>
      </form>
    </div>
    <div class="card">
      <h2>Project</h2>
      <div class="project-toolbar">
        <label><input id="selectAllProjects" type="checkbox"> Chọn tất cả</label>
        <button id="deleteSelectedBtn" class="danger" type="button" disabled>Xóa đã chọn (0)</button>
      </div>
      <div id="bulkMsg" class="notice" role="status"></div>
      <div id="jobList" class="project-list" aria-live="polite">Đang tải…</div>
    </div>
    <div class="card">
      <h2>Tìm trong thư viện</h2>
      <div class="search-field" role="search">
        <label class="sr-only" for="librarySearch">Tìm lời thoại hoặc nội dung hình ảnh trong mọi project</label>
        <input id="librarySearch" type="search" maxlength="200" placeholder='vd: xe đỏ person:"Person 1" location:hospital'>
        <button id="librarySearchBtn" type="button">Tìm</button>
      </div>
      <details>
        <summary class="muted">Bộ lọc nâng cao</summary>
        <div class="filter-grid">
          <select id="libraryKind"><option value="">Mọi loại</option><option value="visual">Visual</option><option value="transcript">Transcript</option></select>
          <input id="libraryProject" placeholder="Project">
          <input id="libraryPerson" placeholder="Person / alias">
          <input id="libraryAction" placeholder="Action">
          <input id="libraryLocation" placeholder="Location">
          <input id="libraryObject" placeholder="Object">
          <input id="librarySceneType" placeholder="Scene type / shot label">
          <input id="librarySource" placeholder="Source (agy/transcript/...)">
          <input id="libraryDateFrom" type="date" aria-label="Project date from">
          <input id="libraryDateTo" type="date" aria-label="Project date to">
          <input id="libraryMinDuration" type="number" min="0" step="0.1" placeholder="Min seconds">
          <input id="libraryMaxDuration" type="number" min="0" step="0.1" placeholder="Max seconds">
          <input id="libraryMinConfidence" type="number" min="0" max="1" step="0.05" placeholder="Min confidence">
        </div>
      </details>
      <div class="row" style="margin-top:8px">
        <button id="saveLibrarySearchBtn" type="button">Lưu tìm kiếm</button>
        <select id="savedLibrarySearches" aria-label="Saved library searches"><option value="">Tìm kiếm đã lưu…</option></select>
      </div>
      <div id="libraryResults" class="project-list muted" aria-live="polite">Nhập từ khóa hoặc bộ lọc để tìm xuyên mọi project.</div>
    </div>
  </aside>
  <main>
    <div id="empty" class="card muted">Chọn video MP4 và tạo project để bắt đầu, hoặc mở một project đã có.</div>
    <div id="detail" style="display:none">
      <div class="card">
        <div class="row" style="justify-content:space-between">
          <h2 id="jobTitle" style="margin:0"></h2>
          <div class="row">
            <button id="runBtn" class="primary">Chạy pipeline</button>
            <button id="stopBtn" class="danger" style="display:none">Dừng project</button>
            <button id="refreshBtn">Làm mới</button>
            <button id="deleteBtn" class="danger" type="button">Xóa project</button>
          </div>
        </div>
        <div id="nextAction" class="action-note" role="status"></div>
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
      </div>

      <div class="card">
        <h2>Tiến trình các bước</h2>
        <div id="stages"></div>
      </div>

      <div class="card" id="videoCard" style="display:none">
        <h2>Xem trước bản dựng cuối</h2>
        <video id="video" controls preload="metadata"></video>
      </div>
      <div class="card" id="exportCard" hidden>
        <h2>Xuất video</h2>
        <div class="muted">Bản MP4 đã vượt qua bước kiểm tra chất lượng.</div>
        <a id="downloadFinal" class="button-link" download="final.mp4">Tải final.mp4</a>
      </div>

      <section class="card media-shell" id="mediaExplorerCard" style="display:none" aria-labelledby="mediaExplorerTitle">
        <div class="media-topbar">
          <div class="media-title">
            <h2 id="mediaExplorerTitle">Media Explorer</h2>
            <p>Review footage, follow the transcript, and pull clips from one workspace.</p>
          </div>
          <div class="export-tools" aria-label="Export controls">
            <button id="mediaExportBtn" type="button" style="display:none" aria-label="Download transcript as VTT">↓ VTT</button>
          </div>
        </div>
        <div class="media-workspace">
          <div class="player-pane">
            <div class="sticky-player">
              <div class="source-frame">
                <video id="sourceVideo" controls preload="metadata" aria-label="Source media player"><track id="sourceCaptions" kind="subtitles" srclang="vi" label="Vietnamese" default></video>
              </div>
              <div class="player-hint"><span>Click a transcript row or scene to seek</span><span id="playerTime" aria-live="off">00:00</span></div>
              <section class="highlight-section" aria-labelledby="highlightsHeading">
                <div class="section-heading"><h3 id="highlightsHeading">Smart highlights</h3><span class="muted">Vertical-ready picks</span></div>
                <div id="highlightResults" class="highlight-grid"><div class="state-panel">Loading highlights…</div></div>
              </section>
              <section class="chat-box" aria-labelledby="askHeading">
                <div class="section-heading"><h3 id="askHeading">Ask about this video</h3><span class="muted">Uses indexed transcript</span></div>
                <div class="search-field"><label class="sr-only" for="chatQuestion">Question about this video</label><input id="chatQuestion" maxlength="500" placeholder="What happens after the lighthouse scene?"><button id="chatAskBtn" type="button">Ask</button></div>
                <div id="chatAnswer" class="chat-answer notice" aria-live="polite">Ask a question to find grounded moments.</div>
              </section>
            </div>
          </div>
          <div class="browser-pane">
            <div class="media-searchbar">
              <div class="search-field" role="search"><label class="sr-only" for="mediaSearch">Search transcript and scenes</label><input id="mediaSearch" type="search" placeholder="Search words, speakers, or scenes…" autocomplete="off"><button id="mediaSearchBtn" type="button">Search</button></div>
              <div class="media-tabs" role="tablist" aria-label="Media result filter">
                <button class="media-tab" type="button" role="tab" aria-selected="true" data-media-filter="transcript">Transcript <span id="transcriptCount"></span></button>
                <button class="media-tab" type="button" role="tab" aria-selected="false" data-media-filter="scenes">Scenes <span id="sceneCount"></span></button>
                <button class="media-tab" type="button" role="tab" aria-selected="false" data-media-filter="highlights">Highlights <span id="highlightCount"></span></button>
              </div>
            </div>
            <div id="mediaResults" class="media-list" aria-live="polite" aria-busy="false"></div>
            <div id="mediaState" class="state-panel" role="status" hidden></div>
            <div id="mediaMsg" class="sr-only" aria-live="polite"></div>
          </div>
        </div>
      </section>

      <div class="card" id="editorCard" style="display:none">
        <div class="row" style="justify-content:space-between">
          <h2 style="margin:0">Timeline &amp; Continuity</h2>
          <button id="autoBrollBtn" type="button">Auto B-roll lặp cảnh</button>
        </div>
        <div class="muted">Khóa clip để giữ nguyên; trim/replace chỉ làm mất hiệu lực các bước sau scene plan.</div>
        <div id="continuityTracks" class="continuity-grid"></div>
        <div id="timelineList" class="timeline-list"></div>
        <div id="editorMsg" class="notice" role="status"></div>
      </div>

      <div class="card" id="thumbnailCard" style="display:none">
        <h2>Chọn ảnh bìa</h2>
        <div class="muted">Chọn một trong các frame đã tạo. Ảnh được chọn sẽ trở thành <code>thumbnail.jpg</code>.</div>
        <div id="thumbnailGrid" class="thumb-grid" style="margin-top:10px"></div>
        <div id="thumbnailMsg" class="notice"></div>
      </div>

      <div class="card">
        <h2>Tệp project</h2>
        <div id="artifacts" class="artifact-grid muted"></div>
      </div>

      <div class="card">
        <h2>Kịch bản &amp; Duyệt</h2>
        <div id="scriptState" class="muted"></div>
        <label>Nội dung các phần (JSON sections)</label>
        <textarea id="scriptSections" rows="8"></textarea>
        <div class="row" style="margin-top:10px">
          <button id="saveScriptBtn" disabled>Lưu kịch bản</button>
          <button id="approveScriptBtn" class="primary" disabled>Duyệt kịch bản</button>
        </div>
        <div class="notice">Lưu thay đổi sẽ đặt lại trạng thái duyệt (approved=false) — phải duyệt lại sau khi sửa.</div>
        <div id="scriptMsg" class="notice"></div>
      </div>

      <div class="card">
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

      <div class="card">
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
    <div class="row"><button type="button" id="cancelDelete">Hủy</button><button id="confirmDelete" class="danger" type="submit" disabled>Xóa project</button></div>
    <div id="deleteMsg" class="err" role="alert"></div>
  </form>
</dialog>
<script>
const $ = (id) => document.getElementById(id);
let current = null;
let poller = null;
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

function uploadVideo(id, file) {
  if (!file || !/[.]mp4$/i.test(file.name) || file.size <= 0) return Promise.reject(new Error('Chọn video MP4 không rỗng.'));
  const progress = $('uploadProgress');
  progress.hidden = false; progress.value = 0;
  $('createMsg').textContent = 'Đang import video ' + file.name + '…';
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/jobs/' + encodeURIComponent(id) + '/source');
    if (_tok) xhr.setRequestHeader('Authorization', 'Bearer ' + _tok);
    xhr.setRequestHeader('X-Source-Name', encodeURIComponent(file.name));
    xhr.setRequestHeader('Content-Type', 'video/mp4');
    xhr.upload.onprogress = event => {
      if (event.lengthComputable) progress.value = Math.round(event.loaded * 100 / event.total);
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
    setMediaState(searching ? 'No matching ' + mediaFilter + ' found. Try a broader search.' : 'No ' + mediaFilter + ' are available yet.', 'empty');
    return;
  }
  setMediaState('');
  for (const row of rows) {
    if (mediaFilter === 'highlights') {
      const card = document.createElement('article');
      card.className = 'highlight-card';
      const heading = document.createElement('strong');
      heading.textContent = row.title || 'Highlight';
      const meta = document.createElement('div'); meta.className = 'result-meta';
      meta.append(makeChip(formatTime(row.start_seconds ?? row.start) + '–' + formatTime(row.end_seconds ?? row.end), 'time-chip'), makeChip('Highlight', 'kind-chip'));
      const reason = document.createElement('p'); reason.textContent = row.reason || 'Suggested moment';
      const actions = document.createElement('div'); actions.className = 'row';
      const preview = document.createElement('button'); preview.type = 'button'; preview.textContent = '▶ Preview'; preview.onclick = makeSeekHandler(row.start_seconds ?? row.start);
      const download = document.createElement('button'); download.type = 'button'; download.textContent = '↓ 9:16 MP4'; download.setAttribute('aria-label', 'Export ' + heading.textContent + ' as vertical MP4'); download.onclick = () => downloadHighlight(row.export_href, row.id).catch(showError);
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
      img.alt = 'Scene thumbnail at ' + formatTime(row.start_seconds);
      img.loading = 'lazy';
      authFetch(row.thumbnail_href).then(resolved => { if (resolved.startsWith('blob:')) mediaObjectUrls.push(resolved); img.src = resolved; }).catch(error => { img.alt = 'Thumbnail unavailable: ' + error.message; });
      item.append(img);
    }
    const detail = document.createElement('div');
    const meta = document.createElement('div'); meta.className = 'result-meta';
    meta.append(makeChip(formatTime(row.start_seconds) + '–' + formatTime(row.end_seconds), 'time-chip'));
    if (isTranscript && row.speaker) meta.append(makeChip(row.speaker, 'speaker-chip'));
    meta.append(makeChip(isTranscript ? 'Transcript' : 'Scene', 'kind-chip'));
    if (row.semantic_score !== null && row.semantic_score !== undefined) {
      meta.append(makeChip('Semantic ' + Number(row.semantic_score).toFixed(2), 'kind-chip'));
    }
    const copy = document.createElement('div'); copy.className = 'result-copy';
    copy.textContent = isTranscript ? row.text : (row.visual_description || row.label || 'Indexed scene');
    if (!isTranscript) {
      for (const value of [
        ...(row.visual_tags || []).slice(0, 3),
        ...(row.visual_people || []).slice(0, 2),
        ...(row.visual_actions || []).slice(0, 2),
        ...(row.person_tracks || []).slice(0, 3),
      ]) meta.append(makeChip(value, 'kind-chip'));
    }
    const seek = document.createElement('button'); seek.type = 'button'; seek.className = 'seek-button'; seek.textContent = '▶ Jump to ' + formatTime(row.start_seconds); seek.setAttribute('aria-label', 'Seek source video to ' + formatTime(row.start_seconds)); seek.onclick = event => { event.stopPropagation(); seekSource(row.start_seconds); };
    detail.append(meta, copy, seek);
    if (!isTranscript && row.id) {
      const similar = document.createElement('button');
      similar.type = 'button';
      similar.textContent = 'Find similar';
      similar.onclick = event => { event.stopPropagation(); findSimilarScenes(row.id).catch(showError); };
      detail.append(similar);
    }
    item.append(detail);
    if (isTranscript) {
      item.tabIndex = 0; item.setAttribute('role', 'button'); item.setAttribute('aria-label', 'Play transcript from ' + formatTime(row.start_seconds) + ': ' + row.text);
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
  box.innerHTML = '<div class="state-panel">Loading highlights…</div>';
  try {
    const data = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/highlights');
    mediaExplorerData.highlights = Array.isArray(data.highlights) ? data.highlights : [];
    $('highlightCount').textContent = mediaExplorerData.highlights.length ? '(' + mediaExplorerData.highlights.length + ')' : '';
    box.replaceChildren();
    if (!mediaExplorerData.highlights.length) { box.innerHTML = '<div class="state-panel">No highlights detected yet.</div>'; }
    for (const item of mediaExplorerData.highlights.slice(0, 4)) {
      const card = document.createElement('article'); card.className = 'highlight-card';
      const title = document.createElement('strong'); title.textContent = item.title || 'Highlight';
      const meta = document.createElement('div'); meta.className = 'result-meta'; meta.append(makeChip(formatTime(item.start_seconds ?? item.start) + '–' + formatTime(item.end_seconds ?? item.end), 'time-chip'));
      const reason = document.createElement('p'); reason.textContent = item.reason || 'Suggested moment';
      const actions = document.createElement('div'); actions.className = 'row';
      const play = document.createElement('button'); play.type = 'button'; play.textContent = '▶ Preview'; play.onclick = makeSeekHandler(item.start_seconds ?? item.start);
      const download = document.createElement('button'); download.type = 'button'; download.textContent = '↓ Export'; download.onclick = () => downloadHighlight(item.export_href, item.id).catch(showError);
      actions.append(play, download); card.append(title, meta, reason, actions); box.append(card);
    }
    if (mediaFilter === 'highlights') renderMediaRows();
  } catch (error) {
    mediaExplorerData.highlights = [];
    box.innerHTML = '<div class="state-panel">Highlights unavailable: ' + error.message + '</div>';
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
  if (!question) { $('chatAnswer').textContent = 'Enter a question first.'; $('chatQuestion').focus(); return; }
  $('chatAskBtn').disabled = true; $('chatAnswer').textContent = 'Searching the indexed transcript…';
  try {
    const data = await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/chat', {question});
    const citations = Array.isArray(data.citations) ? data.citations.length : 0;
    $('chatAnswer').textContent = (data.answer || 'No answer available.') + (citations ? ' · ' + citations + ' cited moment' + (citations === 1 ? '' : 's') : '');
  } finally { $('chatAskBtn').disabled = false; }
}

async function renderMediaExplorer(query = '') {
  const card = $('mediaExplorerCard'); const results = $('mediaResults'); const message = $('mediaMsg');
  clearMediaObjectUrls(); card.style.display = ''; results.setAttribute('aria-busy', 'true'); setMediaState('Loading media index…', 'loading');
  let data;
  try {
    const suffix = query ? ('?q=' + encodeURIComponent(query)) : '';
    data = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/media-explorer' + suffix);
  } catch (error) {
    results.replaceChildren(); results.setAttribute('aria-busy', 'false'); setMediaState('Media Explorer could not load: ' + error.message, 'error'); message.textContent = 'Media Explorer load error'; return;
  }
  if (!data.present || !data.media_href) { card.style.display = 'none'; results.replaceChildren(); results.setAttribute('aria-busy', 'false'); return; }
  const exportBtn = $('mediaExportBtn');
  exportBtn.style.display = data.transcript_vtt_href ? '' : 'none'; exportBtn.onclick = data.transcript_vtt_href ? (() => downloadTranscript(data.transcript_vtt_href).catch(showError)) : null;
  const video = $('sourceVideo');
  if (video.dataset.src !== data.media_href) { video.dataset.src = data.media_href; video.src = data.media_href; }
  const track = $('sourceCaptions'); track.removeAttribute('src');
  if (data.transcript_vtt_href) { const captions = await authFetch(data.transcript_vtt_href); if (captions.startsWith('blob:')) mediaObjectUrls.push(captions); track.src = captions; track.default = true; track.addEventListener('load', () => { if (track.track) track.track.mode = 'showing'; }, {once:true}); }
  mediaExplorerData.transcript = Array.isArray(data.transcript) ? data.transcript : [];
  mediaExplorerData.shots = Array.isArray(data.shots) ? data.shots : [];
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

async function renderEditor() {
  const card = $('editorCard');
  let timelineData, tracksData;
  try {
    [timelineData, tracksData] = await Promise.all([
      api('GET', '/api/jobs/' + encodeURIComponent(current) + '/timeline'),
      api('GET', '/api/jobs/' + encodeURIComponent(current) + '/person-tracks'),
    ]);
  } catch (error) {
    card.style.display = 'none';
    $('timelineList').replaceChildren();
    $('continuityTracks').replaceChildren();
    return;
  }
  card.style.display = '';
  const trackBox = $('continuityTracks');
  trackBox.replaceChildren();
  for (const track of tracksData.tracks || []) {
    const item = document.createElement('article');
    item.className = 'track-card';
    const title = document.createElement('strong');
    title.textContent = (track.alias || track.label) + (track.alias ? ' · ' + track.label : '');
    const meta = document.createElement('div');
    meta.className = 'muted';
    meta.textContent = 'confidence ' + Number(track.mean_confidence || 0).toFixed(2)
      + (track.ambiguous ? ' · ambiguous' : '')
      + ' · ' + ((track.appearances || []).length) + ' scenes';
    const summary = document.createElement('div');
    summary.textContent = track.appearance_summary || track.description || '';
    const clothing = (track.traits || []).filter(x => x.trait_type === 'clothing').map(x => x.value);
    if (clothing.length) {
      const details = document.createElement('div');
      details.className = 'muted';
      details.textContent = 'Clothing: ' + clothing.join(' → ');
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
    heading.textContent = '#' + (clip.timeline_index + 1) + ' · ' + (clip.section || 'Section')
      + ' · ' + formatTime(source.start_seconds) + '–' + formatTime(source.end_seconds);
    const meta = document.createElement('div'); meta.className = 'muted';
    meta.textContent = 'Section ' + clip.section_index + ' · shot ' + clip.shot_index + '/' + clip.shot_count
      + ' · output ' + formatTime(clip.start_seconds) + ' +' + Number(clip.duration_seconds || 0).toFixed(1) + 's';
    const controls = document.createElement('div'); controls.className = 'timeline-controls';

    const lock = document.createElement('button'); lock.type = 'button';
    lock.textContent = clip.locked ? 'Unlock' : 'Lock';
    lock.onclick = () => timelineAction({action:'lock', clip_index:clip.timeline_index, locked:!clip.locked}).catch(showError);

    const trim = document.createElement('button'); trim.type = 'button'; trim.textContent = 'Trim';
    trim.onclick = () => {
      const start = prompt('Source start (seconds)', String(source.start_seconds ?? 0));
      if (start === null) return;
      const end = prompt('Source end (seconds)', String(source.end_seconds ?? 0));
      if (end === null) return;
      timelineAction({action:'trim', clip_index:clip.timeline_index, start_seconds:Number(start), end_seconds:Number(end)}).catch(showError);
    };

    const replace = document.createElement('button'); replace.type = 'button'; replace.textContent = 'Replace shot';
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

    const up = document.createElement('button'); up.type = 'button'; up.textContent = '↑';
    up.disabled = clip.timeline_index <= 0;
    up.onclick = () => timelineAction({action:'reorder', from_index:clip.timeline_index, to_index:clip.timeline_index - 1}).catch(showError);

    const down = document.createElement('button'); down.type = 'button'; down.textContent = '↓';
    down.disabled = clip.timeline_index >= (timelineData.clips || []).length - 1;
    down.onclick = () => timelineAction({action:'reorder', from_index:clip.timeline_index, to_index:clip.timeline_index + 1}).catch(showError);

    const regen = document.createElement('button'); regen.type = 'button'; regen.textContent = 'Regenerate section';
    regen.onclick = async () => {
      const instruction = prompt('Yêu cầu cho visual section này', 'more relevant visuals');
      if (instruction === null) return;
      await api(
        'POST',
        '/api/jobs/' + encodeURIComponent(current) + '/sections/' + clip.section_index + '/regenerate',
        {instruction}
      );
      $('editorMsg').textContent = 'Đã regenerate visual cho section ' + clip.section_index + '.';
      await renderEditor(); await loadStatus();
    };

    const preview = document.createElement('button'); preview.type = 'button'; preview.textContent = 'Preview';
    preview.onclick = () => seekSource(source.start_seconds);

    controls.append(lock, trim, replace, broll, up, down, regen, preview);
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
  clearThumbnailObjectUrls();

  let result;
  try {
    result = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/thumbnails');
  } catch (e) {
    card.style.display = '';
    grid.replaceChildren();
    message.innerHTML = '<span class="err">Lỗi tải ảnh bìa: ' + e.message + '</span>';
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
}

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
  if (!state.query) { $('libraryResults').textContent = 'Nhập query trước khi lưu.'; return; }
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
      meta.append(makeChip(item.kind === 'visual' ? 'Visual' : 'Transcript', 'kind-chip'));
      if (item.semantic_score !== null && item.semantic_score !== undefined) {
        meta.append(makeChip('Semantic ' + Number(item.semantic_score).toFixed(2), 'kind-chip'));
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
    if (!jobs.length) {
      el.textContent = 'Chưa có project. Chọn video MP4 để tạo project đầu tiên.';
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
  current = id;
  $('empty').style.display = 'none';
  $('detail').style.display = '';
  $('video').removeAttribute('src');
  delete $('video').dataset.src;
  $('video').load();
  scriptLoaded = null; metaLoaded = null;
  loadStatus();
  loadJobs();
}

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
  $('deleteBtn').disabled = !!s.running || !!s.uploading;
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
  $('exportCard').hidden = !(s.has_final_video && qaStage?.status === 'ready');
  $('downloadFinal').href = '/api/jobs/' + encodeURIComponent(current) + '/artifacts/final.mp4';

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

  const mediaState = current + ':' + Boolean(s.has_media_index);
  if (mediaLoaded !== mediaState) {
    mediaLoaded = mediaState;
    await renderMediaExplorer();
  }
  if (s.has_media_index) await renderEditor();
  else $('editorCard').style.display = 'none';

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
    if (_tok && item.kind !== 'video') link.onclick = async event => {
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
  if (poller) { clearInterval(poller); poller = null; }
  if (s.running || s.is_indexing) { poller = setInterval(loadStatus, 1500); }
}

let metaLoaded = null;
let scriptLoaded = null;
async function renderMeta(approvals) {
  $('scriptState').innerHTML = approvals.script_present
    ? (approvals.script_approved ? '<span class="ok">Kịch bản đã được duyệt.</span>'
        : '<span class="warn">Kịch bản chưa được duyệt.</span>')
    : '<span class="muted">Chưa có kịch bản (chạy pipeline tới bước Kịch bản).</span>';
  if (!approvals.script_present) { scriptLoaded = null; $('scriptSections').value = ''; }
  else if (scriptLoaded !== current) {
    try {
      const { present, script } = await api('GET', '/api/jobs/' + encodeURIComponent(current) + '/script');
      if (present && script.sections !== undefined) {
        $('scriptSections').value = JSON.stringify(script.sections, null, 2);
        scriptLoaded = current;
      }
    } catch (e) { $('scriptMsg').textContent = 'Không tải được kịch bản: ' + e.message; }
  }
  $('saveScriptBtn').disabled = !approvals.script_present;
  $('approveScriptBtn').disabled = !approvals.script_present;

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
  if (file && payload.source_video) { $('createMsg').textContent = 'Chọn file MP4 hoặc đường dẫn, không dùng cả hai.'; return; }
  if (file && (!/[.]mp4$/i.test(file.name) || !file.size)) { $('createMsg').textContent = 'Chọn video MP4 không rỗng.'; return; }
  const submit = e.target.querySelector('[type="submit"]');
  submit.disabled = true;
  let created = null;
  try {
    created = await api('POST', '/api/jobs', payload);
    if (file) await uploadVideo(created.job_id, file);
    $('createMsg').textContent = file ? 'Đã tạo và import video cho project ' + created.job_id + '.' : 'Đã tạo project ' + created.job_id + '.';
    e.target.reset();
    await loadJobs();
    selectJob(created.job_id);
  } catch (err) {
    $('createMsg').textContent = (created ? 'Project đã tạo; có thể mở và thử import lại. ' : '') + err.message;
    if (created) { await loadJobs(); selectJob(created.job_id); }
  } finally { submit.disabled = false; }
};

$('sourceRetryBtn').onclick = async () => {
  const file = $('sourceRetry').files[0];
  const id = current;
  $('sourceRetryBtn').disabled = true;
  try {
    await uploadVideo(id, file);
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
      current = null; mediaLoaded = null; scriptLoaded = null; metaLoaded = null;
      if (poller) { clearInterval(poller); poller = null; }
      clearThumbnailObjectUrls(); clearMediaObjectUrls();
      $('sourceVideo').removeAttribute('src'); $('sourceVideo').load();
      $('video').removeAttribute('src'); delete $('video').dataset.src; $('video').load();
      $('detail').style.display = 'none'; $('empty').style.display = '';
    }
    $('bulkMsg').textContent = result.failed
      ? 'Đã xóa ' + deleted.length + ' project; không xóa được ' + result.failed + '. Kiểm tra project còn lại.'
      : 'Đã xóa ' + deleted.length + ' project.';
    await loadJobs();
  } catch (err) { $('deleteMsg').textContent = err.message; $('confirmDelete').disabled = false; }
};

$('stopBtn').onclick = async () => { await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/stop', {}); await loadStatus(); };
$('runBtn').onclick = async () => {
  try { await api('POST', '/api/jobs/' + encodeURIComponent(current) + '/run', {}); loadStatus(); }
  catch (e) { $('runErr').textContent = e.message; }
};
$('refreshBtn').onclick = loadStatus;
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

loadJobs();
loadSavedSearches().catch(() => {});
$('chatAskBtn').addEventListener('click', () => askVideo().catch(showError));
</script>
</body>
</html>
"""
