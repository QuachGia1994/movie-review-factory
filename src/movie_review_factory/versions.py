"""Named, immutable job checkpoints and conservative artifact restore."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import pipeline

KINDS = {
    "script": ("script.json", "script.md"),
    "scene_plan": ("scene_plan.json",),
    "captions": ("alignment.json", "aligned.srt"),
    "metadata": ("youtube_metadata.json",),
    "final": ("final.mp4", "render.json", "qa.json"),
}
_VERSION_ID = re.compile(r"^[0-9a-f]{32}$")
_SNAPSHOT_FILES = ("script.json", "script.md", "scene_plan.json", "voice.json",
                   "narration.mp3", "alignment.json", "aligned.srt", "youtube_metadata.json",
                   "audio_mix.json", "final.mp4",
                   "render.json", "qa.json", "manifest.json")


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _versions(root: Path) -> Path:
    return root / "versions"


def _entry(root: Path, version_id: str) -> Path:
    if not _VERSION_ID.fullmatch(version_id):
        raise ValueError("invalid version id")
    parent = _versions(root)
    if parent.is_symlink():
        raise ValueError("versions folder cannot be a symlink")
    folder = parent / version_id
    if not folder.is_dir() or folder.is_symlink():
        raise FileNotFoundError("version not found")
    return folder


def _metadata(folder: Path) -> dict:
    record = json.loads((folder / "version.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
        raise ValueError("invalid version record")
    return record


def _checked_file(folder: Path, record: dict, name: str) -> Path:
    expected = record["files"].get(name)
    if not isinstance(expected, str):
        raise FileNotFoundError(f"{name} absent from version")
    path = folder / name
    if not path.is_file() or path.is_symlink() or _digest(path) != expected:
        raise ValueError(f"corrupt version artifact: {name}")
    return path


def create(root: Path, label: str) -> dict:
    """Copy available editor artifacts to a complete, atomically visible checkpoint."""
    root = Path(root)
    name = str(label).strip()
    if not name or len(name) > 100 or any(ord(char) < 32 for char in name):
        raise ValueError("version name must contain 1–100 printable characters")
    if not (root / "manifest.json").is_file():
        raise FileNotFoundError("job manifest missing")
    parent = _versions(root)
    if parent.is_symlink():
        raise ValueError("versions folder cannot be a symlink")
    parent.mkdir(exist_ok=True)
    version_id = uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix=".saving-", dir=parent) as temp:
        folder = Path(temp)
        files = {}
        for filename in _SNAPSHOT_FILES:
            source = root / filename
            if source.is_file() and not source.is_symlink():
                shutil.copyfile(source, folder / filename)
                files[filename] = _digest(folder / filename)
        record = {
            "id": version_id, "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": files,
            "passing_final": (
                "final.mp4" in files
                and "qa.json" in files
                and bool(json.loads((folder / "qa.json").read_text(encoding="utf-8")).get("passed"))
                and pipeline.load_manifest(root).stage("qa").status == "ready"
            ),
        }
        (folder / "version.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        folder.rename(parent / version_id)
    return record


def list_versions(root: Path) -> list[dict]:
    parent = _versions(Path(root))
    if not parent.is_dir() or parent.is_symlink():
        return []
    entries = []
    for folder in parent.iterdir():
        if folder.is_dir() and _VERSION_ID.fullmatch(folder.name):
            try:
                record = _metadata(folder)
                entries.append({**{key: record[key] for key in ("id", "name", "created_at", "passing_final")},
                                "files": sorted(record["files"])})
            except (OSError, ValueError, KeyError):
                continue
    return sorted(entries, key=lambda item: item["created_at"], reverse=True)


def artifact_path(root: Path, version_id: str, name: str) -> Path:
    if name not in _SNAPSHOT_FILES:
        raise ValueError("unsupported artifact")
    folder = _entry(Path(root), version_id)
    return _checked_file(folder, _metadata(folder), name)


def _content(doc: dict) -> dict:
    return {key: value for key, value in doc.items() if key != "approved"}


def restore(root: Path, version_id: str, kind: str) -> dict:
    """Restore one editable kind; preserve approval only for unchanged content.

    The previous passing MP4 is checkpointed first. Any derived live artifacts
    are invalidated through the pipeline's normal stage boundary.
    """
    root = Path(root)
    if kind not in KINDS:
        raise ValueError("unsupported version kind")
    folder = _entry(root, version_id)
    record = _metadata(folder)
    sources = {}
    for filename in KINDS[kind]:
        if filename in record["files"]:
            sources[filename] = _checked_file(folder, record, filename)
        elif filename == KINDS[kind][0]:
            raise FileNotFoundError(f"{filename} absent from version")
    if kind == "captions":
        if len(sources) != len(KINDS["captions"]):
            raise ValueError("captions require both alignment data and SRT")
        saved_script = _checked_file(folder, record, "script.json")
        live_script = root / "script.json"
        if not live_script.is_file() or live_script.is_symlink():
            raise ValueError("captions require the current script")
        saved_content = _content(json.loads(saved_script.read_text(encoding="utf-8")))
        live_doc = json.loads(live_script.read_text(encoding="utf-8"))
        if saved_content != _content(live_doc):
            raise ValueError("captions belong to a different script revision")
        if not live_doc.get("approved") or pipeline.load_manifest(root).stage("tts").status != "ready":
            raise ValueError("captions require approved script and ready voice")
        for name in ("voice.json", "narration.mp3"):
            _checked_file(folder, record, name)
            live = root / name
            if not live.is_file() or live.is_symlink() or _digest(live) != record["files"][name]:
                raise ValueError("captions belong to a different voice revision")
    if kind == "final":
        if (not record.get("passing_final") or len(sources) != len(KINDS[kind])
                or not json.loads(sources["qa.json"].read_text(encoding="utf-8")).get("passed")):
            raise ValueError("only a passing final can be restored")
        # A final from another editorial state must remain downloadable history,
        # never masquerade as QA for the current edit.
        for name in ("script.json", "scene_plan.json", "voice.json", "narration.mp3",
                     "alignment.json", "aligned.srt", "youtube_metadata.json", "audio_mix.json"):
            live = root / name
            if name in ("audio_mix.json", "voice.json", "narration.mp3") and (name in record["files"]) != live.exists():
                raise ValueError("final belongs to a different audio revision")
            if name in record["files"]:
                _checked_file(folder, record, name)
                if not live.is_file() or live.is_symlink() or _digest(live) != record["files"][name]:
                    raise ValueError("final belongs to a different editorial revision")
        mix_provenance = json.loads(sources["render.json"].read_text(encoding="utf-8")).get("audio_mix", {})
        if mix_provenance.get("config_sha256") != record["files"].get("audio_mix.json"):
            raise ValueError("render audio config provenance differs from saved audio settings")
        for asset in mix_provenance.get("provenance", []):
            if not isinstance(asset, dict) or not isinstance(asset.get("path"), str) or not isinstance(asset.get("sha256"), str):
                raise ValueError("audio provenance lacks verified source hashes")
            path = Path(asset["path"])
            if not path.is_file() or _digest(path) != asset["sha256"]:
                raise ValueError("final belongs to a different audio source revision")
        if not json.loads((root / "script.json").read_text(encoding="utf-8")).get("approved"):
            raise ValueError("current script is not approved")
        if not json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8")).get("approved"):
            raise ValueError("current metadata is not approved")
    previous = create(root, f"Trước khi khôi phục {record['name']}")
    for name, source in sources.items():
        target = root / name
        if kind in {"script", "metadata"} and name.endswith(".json"):
            incoming = json.loads(source.read_text(encoding="utf-8"))
            live = json.loads(target.read_text(encoding="utf-8")) if target.is_file() else {}
            incoming["approved"] = bool(live.get("approved")) if _content(live) == _content(incoming) else False
            pipeline._write_json(root, name, incoming)
        else:
            with tempfile.NamedTemporaryFile(prefix=f".{name}.", dir=root, delete=False) as temp:
                temporary = Path(temp.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, temp)
            temporary.replace(target)
    if kind == "script":
        script = json.loads((root / "script.json").read_text(encoding="utf-8"))
        pipeline._write_text(root, "script.md", pipeline._script_markdown(script))
    if kind != "final":
        pipeline.invalidate_downstream(root, "alignment" if kind == "captions" else kind)
        if kind == "captions":
            manifest = pipeline.load_manifest(root)
            manifest.stage("alignment").mark("ready", "restored captions")
            pipeline.save_manifest(root, manifest)
    else:
        manifest = pipeline.load_manifest(root)
        manifest.stage("render").mark("ready", "restored passing final")
        manifest.stage("qa").mark("ready", "restored passing QA")
        pipeline.save_manifest(root, manifest)
    return {"restored": kind, "version_id": version_id, "previous_version_id": previous["id"]}
