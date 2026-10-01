"""Local, creator-maintained planning records across independent review jobs."""

from __future__ import annotations

import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from . import branding
from .models import ChannelProfile, ChannelSfx, CreativeBrief, JobManifest

_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_CHANNEL_SFX_MAX = 8


def _job_id(value: str) -> str:
    if not isinstance(value, str) or value in (".", "..") or not _SAFE_ID.fullmatch(value):
        raise ValueError("invalid job_id")
    return value


def _name(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise ValueError(f"invalid {field}")
    return value.strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class CreatorLibrary:
    """Store editable series, brief templates, and rights notes beside a jobs directory."""

    def __init__(self, jobs_root: Path):
        self.jobs_root = Path(jobs_root)
        self.path = self.jobs_root / "_creator_library.json"
        self._lock = threading.RLock()

    def _load(self) -> dict:
        if not self.path.exists():
            return {"series": {}, "briefs": {}, "assets": {}, "channels": {}, "active_channel": None}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("invalid creator library")
        # Older libraries predate "channels"; default missing keys so they still load.
        for key in ("series", "briefs", "assets", "channels"):
            data.setdefault(key, {})
            if not isinstance(data[key], dict):
                raise ValueError("invalid creator library")
        data.setdefault("active_channel", None)
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

    # -- channel profiles (multi-channel switcher) ---------------------------

    _CHANNEL_DEFAULT_FIELDS = (
        "language", "aspect_ratio", "tts_provider", "tts_voice",
        "intro_seconds", "outro_seconds", "brand_top_band",
        "brand_bottom_band", "copyright_bypass",
    )

    def list_channels(self) -> dict:
        with self._lock:
            data = self._load()
        channels = sorted(data["channels"].values(), key=lambda item: item["name"].casefold())
        return {"channels": channels, "active": data.get("active_channel")}

    def get_channel(self, profile_id: str) -> dict:
        with self._lock:
            record = self._load()["channels"].get(_job_id(profile_id))
        if record is None:
            raise KeyError(profile_id)
        return dict(record)

    def save_channel(self, profile_id: str, profile: ChannelProfile | dict) -> dict:
        """Create or update a channel profile; activate it when none is active yet."""
        profile_id = _job_id(profile_id)
        try:
            model = ChannelProfile.model_validate(profile)
        except ValidationError as exc:
            detail = exc.errors()[0].get("msg", "hồ sơ kênh không hợp lệ")
            raise ValueError(f"hồ sơ kênh không hợp lệ: {detail}") from exc
        with self._lock:
            data = self._load()
            existing = data["channels"].get(profile_id, {})
            record = {
                "id": profile_id,
                **model.model_dump(),
                "has_logo": bool(existing.get("has_logo", False)),
                "sfx": list(existing.get("sfx", [])),
                "updated_at": _now(),
            }
            data["channels"][profile_id] = record
            if data.get("active_channel") is None:
                data["active_channel"] = profile_id
            active = data.get("active_channel")
            self._save(data)
        if active == profile_id:
            branding.save_name(self.jobs_root, model.name)
        return dict(record)

    def set_channel_logo(self, profile_id: str, data_bytes: bytes) -> dict:
        profile_id = _job_id(profile_id)
        with self._lock:
            if profile_id not in self._load()["channels"]:
                raise KeyError(profile_id)
        branding.save_channel_logo(self.jobs_root, profile_id, data_bytes)
        with self._lock:
            data = self._load()
            record = data["channels"].get(profile_id)
            if record is None:
                raise KeyError(profile_id)
            record["has_logo"] = True
            record["updated_at"] = _now()
            active = data.get("active_channel")
            self._save(data)
        if active == profile_id:
            branding.activate_channel(self.jobs_root, record["name"], profile_id)
        return dict(record)

    def activate_channel(self, profile_id: str) -> dict:
        """1-click switch: point the shared brand identity at this profile."""
        profile_id = _job_id(profile_id)
        with self._lock:
            data = self._load()
            record = data["channels"].get(profile_id)
            if record is None:
                raise KeyError(profile_id)
            data["active_channel"] = profile_id
            self._save(data)
        branding.activate_channel(self.jobs_root, record["name"], profile_id)
        return {"active": profile_id, "channel": dict(record)}

    def delete_channel(self, profile_id: str) -> None:
        profile_id = _job_id(profile_id)
        with self._lock:
            data = self._load()
            if profile_id not in data["channels"]:
                raise KeyError(profile_id)
            if data.get("active_channel") == profile_id:
                raise ValueError("không thể xoá kênh đang kích hoạt")
            del data["channels"][profile_id]
            self._save(data)
        branding.delete_channel_assets(self.jobs_root, profile_id)

    def active_channel_defaults(self) -> dict | None:
        """Job-creation defaults for the active profile (identity name excluded)."""
        with self._lock:
            data = self._load()
            active = data.get("active_channel")
            record = data["channels"].get(active) if active else None
        if record is None:
            return None
        return {key: record[key] for key in self._CHANNEL_DEFAULT_FIELDS if key in record}

    # -- transition SFX palette (reuses audio_mix effects) -------------------

    def save_channel_sfx(self, profile_id: str, sfx: ChannelSfx | dict) -> dict:
        """Upsert one transition-SFX entry's metadata (audio file uploaded separately)."""
        profile_id = _job_id(profile_id)
        try:
            model = ChannelSfx.model_validate(sfx)
        except ValidationError as exc:
            detail = exc.errors()[0].get("msg", "SFX không hợp lệ")
            raise ValueError(f"SFX không hợp lệ: {detail}") from exc
        with self._lock:
            data = self._load()
            record = data["channels"].get(profile_id)
            if record is None:
                raise KeyError(profile_id)
            items = record.setdefault("sfx", [])
            prev = next((s for s in items if s.get("slug") == model.slug), None)
            if prev is None and len(items) >= _CHANNEL_SFX_MAX:
                raise ValueError(f"tối đa {_CHANNEL_SFX_MAX} SFX mỗi kênh")
            entry = {**model.model_dump(), "has_file": bool(prev and prev.get("has_file"))}
            record["sfx"] = [s for s in items if s.get("slug") != model.slug] + [entry]
            record["updated_at"] = _now()
            self._save(data)
        return dict(entry)

    def set_channel_sfx_file(self, profile_id: str, slug: str, content_type: str, data_bytes: bytes) -> dict:
        profile_id = _job_id(profile_id)
        with self._lock:
            record = self._load()["channels"].get(profile_id)
            if record is None:
                raise KeyError(profile_id)
            if not any(s.get("slug") == slug for s in record.get("sfx", [])):
                raise KeyError(slug)
        branding.save_channel_sfx_file(self.jobs_root, profile_id, slug, content_type, data_bytes)
        with self._lock:
            data = self._load()
            record = data["channels"].get(profile_id)
            entry = next((s for s in record.get("sfx", []) if s.get("slug") == slug), None)
            if entry is None:
                raise KeyError(slug)
            entry["has_file"] = True
            record["updated_at"] = _now()
            self._save(data)
        return dict(entry)

    def delete_channel_sfx(self, profile_id: str, slug: str) -> dict:
        profile_id = _job_id(profile_id)
        with self._lock:
            data = self._load()
            record = data["channels"].get(profile_id)
            if record is None:
                raise KeyError(profile_id)
            record["sfx"] = [s for s in record.get("sfx", []) if s.get("slug") != slug]
            record["updated_at"] = _now()
            self._save(data)
        branding.delete_channel_sfx_file(self.jobs_root, profile_id, slug)
        return self.get_channel(profile_id)

    def active_channel_sfx(self) -> dict:
        """The active channel's playable SFX palette (only entries with an uploaded file)."""
        with self._lock:
            data = self._load()
            active = data.get("active_channel")
            record = data["channels"].get(active) if active else None
        if record is None:
            return {"channel": None, "sfx": []}
        palette = [
            {"slug": s["slug"], "label": s["label"], "gain_db": s.get("gain_db", -8)}
            for s in record.get("sfx", []) if s.get("has_file")
        ]
        return {"channel": record["name"], "sfx": palette}

    def resolve_active_sfx(self, slug: str) -> tuple[Path, dict]:
        """Resolve an active-channel SFX slug to its stored file path + metadata."""
        with self._lock:
            data = self._load()
            active = data.get("active_channel")
            record = data["channels"].get(active) if active else None
        if record is None:
            raise KeyError("no active channel")
        meta = next((s for s in record.get("sfx", []) if s.get("slug") == slug), None)
        if meta is None:
            raise KeyError(slug)
        path = branding.channel_sfx_path(self.jobs_root, active, slug)
        if path is None:
            raise FileNotFoundError(f"SFX chưa có tệp: {slug}")
        return path, meta
