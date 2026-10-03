"""Create a verified, self-contained review handoff for manual platform upload."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from pathlib import Path

from .packaging import chapters as _chapters
from .pipeline import load_manifest

EXPORT_NAME = "review-handoff.zip"
ASSETS = (
    ("final.mp4", "video.mp4"),
    ("aligned.srt", "subtitles.srt"),
    ("thumbnail.jpg", "thumbnail.jpg"),
    ("youtube_metadata.json", "youtube_metadata.json"),
)


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_handoff(root: Path) -> Path:
    """Package only a QA-passed, approved render; replace the ZIP atomically.

    Run after metadata approval. This function never uploads to an external site.
    """
    root = Path(root)
    manifest = load_manifest(root)
    required = ("script", "scene_plan", "alignment", "render", "qa", "metadata", "thumbnail")
    pending = [name for name in required if not manifest.stage(name) or manifest.stage(name).status != "ready"]
    if pending:
        raise ValueError("handoff blocked: stages not ready: " + ", ".join(pending))

    script = _read_json(root / "script.json")
    metadata = _read_json(root / "youtube_metadata.json")
    qa = _read_json(root / "qa.json")
    plan = _read_json(root / "render.json")
    if not script.get("approved") or not metadata.get("approved"):
        raise ValueError("handoff blocked: script and metadata require approval")
    if not qa.get("passed") or qa.get("output_file") != "final.mp4":
        raise ValueError("handoff blocked: final.mp4 has no passing QA report")

    files = [(root / source, archive) for source, archive in ASSETS]
    missing = [path.name for path, _ in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError("handoff blocked: missing or empty assets: " + ", ".join(missing))

    title = str(metadata.get("title", "")).strip()
    description = str(metadata.get("description", "")).strip()
    if not title or not description:
        raise ValueError("handoff blocked: title and description must be filled in")
    chapters = _chapters(plan, script)
    pinned = str(metadata.get("pinned_comment") or "").strip()
    alternatives = [str(item) for item in metadata.get("title_options") or [] if str(item).strip() and str(item) != title]
    notes = "\n".join((
        title, "", description, "", "Chapters:", chapters.rstrip(), "",
        "Tags: " + ", ".join(str(tag) for tag in metadata.get("tags") or []), "",
        *(("Title options:", *alternatives, "") if alternatives else ()),
        *(("Pinned comment:", pinned, "") if pinned else ()),
        "Credits / source use:", str(metadata.get("credits") or "Creator review required before upload."), "",
    ))
    srt = (root / "aligned.srt").read_text(encoding="utf-8-sig")
    vtt = "WEBVTT\n\n" + re.sub(
        r"(\d{2}:\d{2}:\d{2}),(\d{3})", r"\1.\2", srt.strip()
    ) + "\n"
    generated = {
        "subtitles.vtt": vtt.encode("utf-8"),
        "chapters.txt": chapters.encode("utf-8"),
        "upload-notes.txt": notes.encode("utf-8"),
    }
    if pinned:
        generated["pinned-comment.txt"] = (pinned + "\n").encode("utf-8")
    checksums = {
        name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path, name in files
    }
    checksums.update({
        name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        for name, data in generated.items()
    })
    revision_files = ("script.json", "scene_plan.json", "alignment.json",
                      "render.json", "qa.json", "youtube_metadata.json")
    # The approved revision must be complete: record every file unconditionally so
    # the recorded set always matches the download-time integrity check
    # (analytics._current_receipt). A silently-dropped revision file would yield a
    # bundle that can never be served or bound to analytics.
    missing_revision = [name for name in revision_files if not (root / name).is_file()]
    if missing_revision:
        raise ValueError(
            "handoff blocked: missing approved-revision files: "
            + ", ".join(missing_revision)
        )
    receipt = {
        "job_id": manifest.config.job_id,
        "qa_passed": True,
        "approved_revision": {name: _sha256(root / name) for name in revision_files},
        "contents": checksums,
    }

    target = root / EXPORT_NAME
    temporary = root / (EXPORT_NAME + ".tmp")
    try:
        with zipfile.ZipFile(temporary, "w", allowZip64=True) as archive:
            for path, name in files:
                compression = zipfile.ZIP_STORED if name.endswith((".mp4", ".jpg")) else zipfile.ZIP_DEFLATED
                archive.write(path, name, compress_type=compression)
                if _sha256(path) != checksums[name]["sha256"]:
                    raise RuntimeError(f"{path.name} changed while packaging")
            for name, data in generated.items():
                archive.writestr(name, data, compress_type=zipfile.ZIP_DEFLATED)
            archive.writestr(
                "handoff-manifest.json",
                json.dumps(receipt, ensure_ascii=False, indent=2).encode("utf-8"),
                compress_type=zipfile.ZIP_DEFLATED,
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target
