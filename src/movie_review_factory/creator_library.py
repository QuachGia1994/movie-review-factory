"""Local, creator-maintained planning records across independent review jobs."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path

from .models import CreativeBrief, JobManifest

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _job_id(value: str) -> str:
    if not isinstance(value, str) or value in (".", "..") or not _SAFE_ID.fullmatch(value):
        raise ValueError("invalid job_id")
    return value


def _name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise ValueError(f"invalid {field}")
    return value.strip()


class CreatorLibrary:
    """Store editable series, brief templates, and rights notes beside a jobs directory."""

    def __init__(self, jobs_root: Path):
        self.jobs_root = Path(jobs_root)
        self.path = self.jobs_root / "_creator_library.json"
        self._lock = threading.RLock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"series": {}, "briefs": {}, "assets": {}}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not all(
            isinstance(data.get(key), dict) for key in ("series", "briefs", "assets")
        ):
            raise ValueError("invalid creator library")
        return data

    def _save(self, data: dict) -> None:
        self.jobs_root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f"{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def _manifest(self, job_id: str) -> JobManifest | None:
        path = self.jobs_root / _job_id(job_id) / "manifest.json"
        if not path.is_file():
            return None
        return JobManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def save_series(self, series_id: str, title: str, entries: list[dict]) -> dict:
        series_id = _job_id(series_id)
        title = _name(title, "series title")
        if not isinstance(entries, list):
            raise ValueError("entries must be a list")
        planned = []
        for entry in entries:
            movie_title = _name(entry.get("movie_title"), "movie title")
            job_id = entry.get("job_id")
            if job_id is not None and self._manifest(_job_id(job_id)) is None:
                raise ValueError(f"unknown job: {job_id}")
            planned.append({"movie_title": movie_title, "job_id": job_id})
        result = {"series_id": series_id, "title": title, "entries": planned}
        with self._lock:
            data = self._load()
            data["series"][series_id] = result
            self._save(data)
        return result

    def list_series(self) -> list[dict]:
        with self._lock:
            return sorted(self._load()["series"].values(), key=lambda item: item["title"].casefold())

    def save_brief(self, name: str, brief: CreativeBrief | dict) -> dict:
        name = _name(name, "brief name")
        result = {"name": name, **CreativeBrief.model_validate(brief).model_dump()}
        with self._lock:
            data = self._load()
            data["briefs"][name] = result
            self._save(data)
        return result

    def get_brief(self, name: str) -> dict:
        with self._lock:
            result = self._load()["briefs"].get(name)
        if result is None:
            raise KeyError(name)
        return dict(result)

    def list_briefs(self) -> list[dict]:
        with self._lock:
            return sorted(self._load()["briefs"].values(), key=lambda item: item["name"].casefold())

    def record_asset_rights(
        self,
        job_id: str,
        path: str,
        *,
        source: str,
        usage: str,
        permission_status: str = "unreviewed",
        evidence_note: str = "",
    ) -> dict:
        job_id = _job_id(job_id)
        if permission_status not in ("unreviewed", "permitted", "restricted"):
            raise ValueError("invalid permission status")
        note = evidence_note.strip()
        if permission_status == "permitted" and not note:
            raise ValueError("permission evidence is required")
        record = {
            "job_id": job_id,
            "path": _name(path, "asset path"),
            "source": _name(source, "asset source"),
            "usage": _name(usage, "asset usage"),
            "permission_status": permission_status,
            "evidence_note": note,
        }
        with self._lock:
            data = self._load()
            data["assets"].setdefault(job_id, {})[record["path"]] = record
            self._save(data)
        return record

    def list_asset_rights(self, job_id: str) -> list[dict]:
        with self._lock:
            return list(self._load()["assets"].get(_job_id(job_id), {}).values())

    def search_projects(self, query: str) -> list[dict]:
        """Search small project metadata on demand; do not rebuild per-job media indexes."""
        needle = query.strip().casefold()
        with self._lock:
            series = self._load()["series"].values()
            series_titles: dict[str, list[str]] = {}
            for item in series:
                for entry in item["entries"]:
                    if entry["job_id"] is not None:
                        series_titles.setdefault(entry["job_id"], []).append(item["title"])
        results = []
        if not self.jobs_root.exists():
            return results
        for child in sorted(self.jobs_root.iterdir()):
            if not child.is_dir() or child.is_symlink() or not _SAFE_ID.fullmatch(child.name):
                continue
            try:
                manifest = self._manifest(child.name)
            except (ValueError, OSError):
                continue
            if manifest is None:
                continue
            metadata_path = child / "youtube_metadata.json"
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
            except (ValueError, OSError):
                metadata = {}
            fields = [
                child.name, manifest.config.movie_title or "",
                manifest.config.creative_brief.review_thesis,
                manifest.config.creative_brief.tone,
                *series_titles.get(child.name, []),
            ]
            if isinstance(metadata, dict):
                fields.extend(str(metadata.get(key) or "") for key in ("title", "description"))
                tags = metadata.get("tags")
                if isinstance(tags, list):
                    fields.extend(str(tag) for tag in tags if isinstance(tag, str))
            if needle in "\n".join(fields).casefold():
                results.append({
                    "job_id": child.name,
                    "movie_title": manifest.config.movie_title or "",
                    "series": series_titles.get(child.name, []),
                })
        return results
