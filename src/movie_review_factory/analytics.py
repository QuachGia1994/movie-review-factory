"""Local, measured YouTube Studio feedback bound to an approved handoff revision."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import tempfile
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from .pipeline import load_manifest

REVISION_FILES = ("script.json", "scene_plan.json", "alignment.json", "render.json",
                  "qa.json", "youtube_metadata.json")
MAX_EXPORT_BYTES = 2_000_000
MAX_POINTS = 5000
HANDOFF_ASSETS = {"video.mp4", "subtitles.srt", "subtitles.vtt", "thumbnail.jpg",
                  "youtube_metadata.json", "chapters.txt", "upload-notes.txt"}
LIVE_ASSETS = {"video.mp4": "final.mp4", "subtitles.srt": "aligned.srt",
               "thumbnail.jpg": "thumbnail.jpg", "youtube_metadata.json": "youtube_metadata.json"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _number(value: object, name: str, *, maximum: float | None = None) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or number < 0 or (maximum is not None and number > maximum):
        raise ValueError(f"{name} must be finite and in range")
    return number


def _current_receipt(root: Path) -> dict:
    archive_path = root / "review-handoff.zip"
    if not archive_path.is_file():
        raise ValueError("handoff bundle missing")
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            receipt = json.loads(bundle.read("handoff-manifest.json"))
    except (OSError, zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("invalid handoff bundle") from exc
    manifest = load_manifest(root)
    if not isinstance(receipt, dict) or receipt.get("job_id") != manifest.config.job_id or receipt.get("qa_passed") is not True:
        raise ValueError("handoff belongs to another job or failed QA")
    if any(not manifest.stage(stage) or manifest.stage(stage).status != "ready"
           for stage in ("script", "render", "qa", "metadata")):
        raise ValueError("handoff stages must be ready")
    revision = receipt.get("approved_revision")
    if not isinstance(revision, dict) or set(revision) != set(REVISION_FILES):
        raise ValueError("handoff revision is incomplete")
    for filename in REVISION_FILES:
        path = root / filename
        if not path.is_file() or path.is_symlink() or _sha256(path.read_bytes()) != revision[filename]:
            raise ValueError("approved handoff revision has changed")
    contents = receipt.get("contents")
    if not isinstance(contents, dict) or set(contents) != HANDOFF_ASSETS:
        raise ValueError("handoff asset receipt is incomplete")
    try:
        with zipfile.ZipFile(archive_path) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or set(names) != HANDOFF_ASSETS | {"handoff-manifest.json"}:
                raise ValueError("handoff archive has unexpected assets")
            for name in HANDOFF_ASSETS:
                expected = contents[name]
                if (not isinstance(expected, dict) or not isinstance(expected.get("bytes"), int)
                        or expected["bytes"] < 0 or not isinstance(expected.get("sha256"), str)):
                    raise ValueError("handoff asset receipt is malformed")
                checksum = hashlib.sha256()
                size = 0
                with bundle.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        checksum.update(block)
                        size += len(block)
                if size != expected["bytes"] or checksum.hexdigest() != expected["sha256"]:
                    raise ValueError("handoff archive asset differs from receipt")
                if name in LIVE_ASSETS:
                    live = root / LIVE_ASSETS[name]
                    if not live.is_file() or live.is_symlink() or live.stat().st_size != size:
                        raise ValueError("handoff live asset changed")
                    with live.open("rb") as stream:
                        live_digest = hashlib.sha256()
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            live_digest.update(block)
                    if live_digest.hexdigest() != expected["sha256"]:
                        raise ValueError("handoff live asset changed")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("handoff archive cannot be verified") from exc
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8"))
    qa = json.loads((root / "qa.json").read_text(encoding="utf-8"))
    if not script.get("approved") or not metadata.get("approved") or not qa.get("passed") or qa.get("output_file") != "final.mp4":
        raise ValueError("approved handoff gates changed")
    return receipt


def _parse_export(path: Path) -> tuple[dict, bytes]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_EXPORT_BYTES:
        raise ValueError("Studio export must be a regular file up to 2 MB")
    data = path.read_bytes()
    try:
        text = data.decode("utf-8-sig")
        if path.suffix.lower() == ".json":
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                raise ValueError("Studio JSON must be an object")
        elif path.suffix.lower() == ".csv":
            reader = csv.DictReader(io.StringIO(text, newline=""))
            required = {"time_seconds", "retention_percent"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("Studio CSV needs time_seconds and retention_percent columns")
            rows = list(reader)
            if len(rows) > MAX_POINTS:
                raise ValueError("too many retention rows")
            parsed = {"retention": rows}
            for field in ("impressions", "ctr_percent"):
                values = {row[field] for row in rows if row.get(field) not in (None, "")}
                if len(values) > 1:
                    raise ValueError(f"inconsistent {field} across CSV rows")
                if values:
                    parsed[field] = values.pop()
        else:
            raise ValueError("Studio export must be .csv or .json")
    except (UnicodeError, csv.Error, json.JSONDecodeError) as exc:
        raise ValueError("invalid Studio export encoding or structure") from exc
    return parsed, data


def _normalize(parsed: dict) -> tuple[list[dict], int | None, float | None, list[dict]]:
    raw = parsed.get("retention")
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_POINTS:
        raise ValueError("retention requires 1–5000 measured points")
    points = []
    last_time = -1.0
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError("retention rows must be objects")
        timestamp = _number(entry.get("time_seconds"), "time_seconds")
        retention = _number(entry.get("retention_percent"), "retention_percent", maximum=100)
        if timestamp <= last_time:
            raise ValueError("retention timestamps must be strictly increasing")
        points.append({"time_seconds": timestamp, "retention_percent": retention})
        last_time = timestamp
    impressions = parsed.get("impressions")
    if impressions is not None:
        count = _number(impressions, "impressions")
        if not count.is_integer():
            raise ValueError("impressions must be a whole number")
        impressions = int(count)
    ctr = parsed.get("ctr_percent")
    if ctr is not None:
        ctr = _number(ctr, "ctr_percent", maximum=100)
    tests = parsed.get("test_results", [])
    if not isinstance(tests, list) or len(tests) > 20:
        raise ValueError("test_results must contain at most 20 Studio observations")
    results = []
    for item in tests:
        if not isinstance(item, dict) or set(item) != {"variant", "result"}:
            raise ValueError("test result must include variant and result")
        variant, result = item["variant"], item["result"]
        if any(not isinstance(s, str) or not s.strip() or len(s) > 300
               for s in (variant, result)):
            raise ValueError("test result text must contain 1–300 characters")
        results.append({"variant": variant.strip(), "result": result.strip()})
    return points, impressions, ctr, results


def _measure(points: list[dict], cta_seconds: float | None) -> dict:
    origin = next((p for p in points if p["time_seconds"] <= 1), None)
    intro = min((p for p in points if 25 <= p["time_seconds"] <= 35),
                key=lambda p: abs(p["time_seconds"] - 30), default=None)
    result = {
        "intro_drop_percentage_points": round(origin["retention_percent"] - intro["retention_percent"], 4)
        if origin and intro else None,
        "intro_sample_seconds": [origin["time_seconds"], intro["time_seconds"]]
        if origin and intro else None,
        "cta_drop_percentage_points": None,
        "cta_sample_seconds": None,
    }
    if cta_seconds is not None:
        cta = _number(cta_seconds, "cta_seconds")
        before = max((p for p in points if cta - 10 <= p["time_seconds"] < cta),
                     key=lambda p: p["time_seconds"], default=None)
        after = min((p for p in points if cta < p["time_seconds"] <= cta + 10),
                    key=lambda p: p["time_seconds"], default=None)
        if before and after:
            result["cta_drop_percentage_points"] = round(
                before["retention_percent"] - after["retention_percent"], 4)
            result["cta_sample_seconds"] = [before["time_seconds"], after["time_seconds"]]
    return result


def import_studio_export(root: Path, export_path: Path, *,
                         cta_seconds: float | None = None, notes: str = "") -> dict:
    """Import measured Studio data to a local immutable record for the current approved ZIP.

    CSV uses time_seconds,retention_percent[,impressions,ctr_percent]. JSON uses
    retention:[{time_seconds,retention_percent}], optional impressions, ctr_percent,
    test_results:[{variant,result}]. Percent values are 0–100. CTA time is the
    actual output time supplied by the creator. Missing neighboring samples yield
    null measurements; this function does not interpolate or predict engagement.
    """
    root = Path(root)
    receipt = _current_receipt(root)
    parsed, source_bytes = _parse_export(Path(export_path))
    points, impressions, ctr, tests = _normalize(parsed)
    if cta_seconds is not None:
        cta_seconds = _number(cta_seconds, "cta_seconds")
    if not isinstance(notes, str) or len(notes) > 2000 or any(ord(c) < 32 and c not in "\n\t" for c in notes):
        raise ValueError("notes must contain at most 2000 printable characters")
    revision = receipt["approved_revision"]
    revision_digest = _sha256(json.dumps(revision, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    report = {
        "id": uuid.uuid4().hex,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_id": receipt["job_id"],
        "approved_revision_sha256": revision_digest,
        "source_sha256": _sha256(source_bytes),
        "retention": points,
        "impressions": impressions,
        "ctr_percent": ctr,
        "test_results": tests,
        "notes": notes.strip(),
        "cta_seconds": cta_seconds,
        "measurements": _measure(points, cta_seconds),
    }
    folder = root / "analytics"
    if folder.is_symlink():
        raise ValueError("analytics folder cannot be a symlink")
    folder.mkdir(exist_ok=True)
    destination = folder / (report["id"] + ".json")
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=folder,
                                     prefix=".import-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, ensure_ascii=False, indent=2)
    try:
        # Fail if the editorial state changed while the export was parsed.
        if _current_receipt(root)["approved_revision"] != revision:
            raise ValueError("approved handoff revision changed during import")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report


def list_imports(root: Path) -> list[dict]:
    """List local historical measurements without reassigning them to a newer revision."""
    folder = Path(root) / "analytics"
    if not folder.is_dir() or folder.is_symlink():
        return []
    reports = []
    for path in folder.glob("*.json"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(entry, dict) and path.stem == entry.get("id"):
                reports.append(entry)
        except (OSError, ValueError):
            continue
    return sorted(reports, key=lambda item: (item.get("created_at", ""), item["id"]), reverse=True)
