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


# --- advisory (read-only) ---------------------------------------------------
# Turn measured Studio retention into suggestions for the *next* video. Strictly
# ADVISORY: nothing here edits a script, brief, hook, or thumbnail — the creator /
# content agent decides. Off switch: pass ``enabled=False`` or set the env var
# ``MRF_RETENTION_ADVICE`` to 0/off/false/no.
INTRO_DROP_ADVICE_THRESHOLD = 35.0   # intro_drop (pp) above this -> opening hook too slow
CTA_DROP_ADVICE_THRESHOLD = 25.0     # cta_drop (pp) above this -> CTA placed badly
CTR_ADVICE_THRESHOLD = 4.0           # ctr_percent below this -> thumbnail underperforming
ADVICE_RECENT_DEFAULT = 5            # how many newest reports to average
ADVICE_MIN_CONFIDENT_SAMPLES = 3     # fewer than this -> flag low confidence
ADVICE_ENV_FLAG = "MRF_RETENTION_ADVICE"


def _advice_enabled(enabled: bool | None) -> bool:
    if enabled is not None:
        return bool(enabled)
    return os.environ.get(ADVICE_ENV_FLAG, "").strip().lower() not in ("0", "off", "false", "no")


def _avg(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def retention_advice(root: Path | None = None, *, reports: list[dict] | None = None,
                     enabled: bool | None = None, recent: int = ADVICE_RECENT_DEFAULT) -> dict:
    """Read-only suggestions for the next video from measured Studio retention.

    Averages the ``recent`` newest measurements (from ``reports`` when given, else
    ``list_imports(root)``) and flags the fixed thresholds. It never mutates any
    script/brief/hook/thumbnail; callers surface the suggestions and a human or the
    content agent decides. Small samples set ``low_confidence`` so one or two videos
    never drive a big change. Disable with ``enabled=False`` or
    ``MRF_RETENTION_ADVICE=0``.
    """
    thresholds = {"intro_drop": INTRO_DROP_ADVICE_THRESHOLD,
                  "cta_drop": CTA_DROP_ADVICE_THRESHOLD, "ctr": CTR_ADVICE_THRESHOLD}
    if not _advice_enabled(enabled):
        return {"enabled": False, "advisory_only": True, "sample_size": 0,
                "signals": {}, "thresholds": thresholds, "low_confidence": False,
                "suggestions": []}
    if reports is None:
        reports = list_imports(root) if root is not None else []
    considered = [r for r in reports if isinstance(r, dict)][:max(0, int(recent))]

    intro_values: list[float] = []
    cta_values: list[float] = []
    ctr_values: list[float] = []
    for report in considered:
        raw = report.get("measurements")
        measurements = raw if isinstance(raw, dict) else {}
        intro = measurements.get("intro_drop_percentage_points")
        cta = measurements.get("cta_drop_percentage_points")
        ctr = report.get("ctr_percent")
        if isinstance(intro, (int, float)) and not isinstance(intro, bool):
            intro_values.append(float(intro))
        if isinstance(cta, (int, float)) and not isinstance(cta, bool):
            cta_values.append(float(cta))
        if isinstance(ctr, (int, float)) and not isinstance(ctr, bool):
            ctr_values.append(float(ctr))

    signals = {"intro_drop_avg": _avg(intro_values), "cta_drop_avg": _avg(cta_values),
               "ctr_avg": _avg(ctr_values)}
    suggestions: list[dict] = []
    intro_avg = signals["intro_drop_avg"]
    if intro_avg is not None and intro_avg > INTRO_DROP_ADVICE_THRESHOLD:
        suggestions.append({
            "code": "intro_hook_too_slow", "target": "hook", "severity": "high",
            "metric": "intro_drop_percentage_points", "value": intro_avg,
            "threshold": INTRO_DROP_ADVICE_THRESHOLD,
            "message_vi": (f"Khán giả rời nhiều ở đoạn mở đầu (rơi ~{intro_avg:g} điểm % quanh giây 30). "
                           "Gợi ý: rút hook xuống dưới 15 giây, cắt cảnh nhanh hơn trong 60 giây đầu, "
                           "tăng nhịp đọc mở đầu. Chưa tự đổi kịch bản."),
            "message_en": (f"High early drop-off (~{intro_avg:g} pp around 0:30). Suggestion: cut the hook "
                           "below 15s, speed up the first 60s, lift the opening TTS pace. Advisory only."),
        })
    cta_avg = signals["cta_drop_avg"]
    if cta_avg is not None and cta_avg > CTA_DROP_ADVICE_THRESHOLD:
        suggestions.append({
            "code": "cta_placement", "target": "script", "severity": "medium",
            "metric": "cta_drop_percentage_points", "value": cta_avg,
            "threshold": CTA_DROP_ADVICE_THRESHOLD,
            "message_vi": (f"Khán giả rời ngay khi kêu gọi đăng ký (rơi ~{cta_avg:g} điểm % quanh CTA). "
                           "Gợi ý: đẩy CTA về cuối hơn hoặc lồng ghép tự nhiên trong lời dẫn. Chưa tự đổi kịch bản."),
            "message_en": (f"Viewers leave at the CTA (~{cta_avg:g} pp around the CTA). Suggestion: move the "
                           "CTA later or make it in-narrative. Advisory only."),
        })
    ctr_avg = signals["ctr_avg"]
    if ctr_avg is not None and ctr_avg < CTR_ADVICE_THRESHOLD:
        suggestions.append({
            "code": "thumbnail_ctr_low", "target": "thumbnail", "severity": "medium",
            "metric": "ctr_percent", "value": ctr_avg, "threshold": CTR_ADVICE_THRESHOLD,
            "message_vi": (f"CTR thumbnail thấp (~{ctr_avg:g}%). Gợi ý: tăng tương phản màu, cận mặt biểu cảm "
                           "mạnh, giới hạn 3–4 từ giật gân trên thumbnail. Chưa tự đổi kịch bản."),
            "message_en": (f"Low thumbnail CTR (~{ctr_avg:g}%). Suggestion: raise contrast, zoom on an "
                           "expressive face, keep to 3–4 punchy words. Advisory only."),
        })

    return {
        "enabled": True, "advisory_only": True, "sample_size": len(considered),
        "signals": signals, "thresholds": thresholds,
        "low_confidence": len(considered) < ADVICE_MIN_CONFIDENT_SAMPLES,
        "suggestions": suggestions,
    }
