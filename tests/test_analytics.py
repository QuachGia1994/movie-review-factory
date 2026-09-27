import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from movie_review_factory import analytics, pipeline
from movie_review_factory.models import JobConfig


REVISION_FILES = ("script.json", "scene_plan.json", "alignment.json", "render.json", "qa.json", "youtube_metadata.json")


@pytest.fixture
def handed_off(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    pipeline.create_job(root, JobConfig(job_id="review"))
    for name in REVISION_FILES:
        body = {"approved": True} if name in {"script.json", "youtube_metadata.json"} else {}
        if name == "qa.json":
            body = {"passed": True, "output_file": "final.mp4"}
        (root / name).write_text(json.dumps(body), encoding="utf-8")
    assets = {
        "video.mp4": b"approved video", "subtitles.srt": b"SRT", "thumbnail.jpg": b"JPG",
        "youtube_metadata.json": (root / "youtube_metadata.json").read_bytes(),
        "subtitles.vtt": b"VTT", "chapters.txt": b"chapter", "upload-notes.txt": b"notes",
    }
    for source, archive in (("final.mp4", "video.mp4"), ("aligned.srt", "subtitles.srt"),
                            ("thumbnail.jpg", "thumbnail.jpg")):
        (root / source).write_bytes(assets[archive])
    receipt = {"job_id": "review", "qa_passed": True, "approved_revision": {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in REVISION_FILES
    }, "contents": {name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                    for name, data in assets.items()}}
    with zipfile.ZipFile(root / "review-handoff.zip", "w") as bundle:
        for name, data in assets.items():
            bundle.writestr(name, data)
        bundle.writestr("handoff-manifest.json", json.dumps(receipt))
    manifest = pipeline.load_manifest(root)
    for stage in ("script", "render", "qa", "metadata"):
        manifest.stage(stage).mark("ready")
    pipeline.save_manifest(root, manifest)
    return root


def test_json_import_measures_intro_and_cta_and_links_exact_revision(handed_off: Path, tmp_path: Path):
    source = tmp_path / "studio.json"
    source.write_text(json.dumps({
        "retention": [
            {"time_seconds": 0, "retention_percent": 100},
            {"time_seconds": 30, "retention_percent": 72},
            {"time_seconds": 95, "retention_percent": 66},
            {"time_seconds": 105, "retention_percent": 58},
        ],
        "impressions": 10000,
        "ctr_percent": 5.4,
        "test_results": [{"variant": "Thumbnail B", "result": "Studio winner"}],
    }), encoding="utf-8")
    report = analytics.import_studio_export(handed_off, source, cta_seconds=100, notes="Intro cần ngắn lại")
    assert report["measurements"]["intro_drop_percentage_points"] == 28
    assert report["measurements"]["cta_drop_percentage_points"] == 8
    assert report["measurements"]["cta_sample_seconds"] == [95, 105]
    assert report["impressions"] == 10000
    assert report["ctr_percent"] == 5.4
    assert report["notes"] == "Intro cần ngắn lại"
    assert report["test_results"] == [{"variant": "Thumbnail B", "result": "Studio winner"}]
    assert len(report["approved_revision_sha256"]) == 64
    assert report["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert analytics.list_imports(handed_off)[0]["id"] == report["id"]


def test_csv_import_and_sparse_curve_leave_unmeasurable_windows_blank(handed_off: Path, tmp_path: Path):
    source = tmp_path / "studio.csv"
    source.write_text("time_seconds,retention_percent,impressions,ctr_percent\n0,100,1500,4.2\n60,65,,\n", encoding="utf-8-sig")
    report = analytics.import_studio_export(handed_off, source, cta_seconds=100)
    assert report["measurements"]["intro_drop_percentage_points"] is None
    assert report["measurements"]["cta_drop_percentage_points"] is None
    assert report["impressions"] == 1500
    assert report["ctr_percent"] == 4.2


def test_rejects_stale_or_unapproved_revision_and_preserves_prior_import(handed_off: Path, tmp_path: Path):
    source = tmp_path / "studio.json"
    source.write_text('{"retention": [{"time_seconds":0,"retention_percent":100}]}', encoding="utf-8")
    first = analytics.import_studio_export(handed_off, source)
    (handed_off / "script.json").write_text('{"approved": false}', encoding="utf-8")
    with pytest.raises(ValueError, match="approved|revision"):
        analytics.import_studio_export(handed_off, source)
    assert analytics.list_imports(handed_off)[0]["id"] == first["id"]


@pytest.mark.parametrize("payload", [
    {"retention": [{"time_seconds": 0, "retention_percent": 100}, {"time_seconds": 0, "retention_percent": 80}]},
    {"retention": [{"time_seconds": -1, "retention_percent": 90}]},
    {"retention": [{"time_seconds": 0, "retention_percent": 120}]},
    {"retention": [{"time_seconds": 0, "retention_percent": "NaN"}]},
    {"retention": [], "impressions": -1},
    {"retention": [], "ctr_percent": 101},
    {"retention": [], "test_results": [{"variant": "", "result": "winner"}]},
])
def test_rejects_invalid_studio_data(handed_off: Path, tmp_path: Path, payload):
    source = tmp_path / "bad.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        analytics.import_studio_export(handed_off, source)
    assert analytics.list_imports(handed_off) == []


def test_cta_time_is_normalized_before_persisting(handed_off: Path, tmp_path: Path):
    source = tmp_path / "studio.json"
    source.write_text('{"retention":[{"time_seconds":0,"retention_percent":100}]}', encoding="utf-8")
    report = analytics.import_studio_export(handed_off, source, cta_seconds="100")
    assert report["cta_seconds"] == 100.0
    assert isinstance(report["cta_seconds"], float)


def test_rejects_stale_video_and_tampered_zip(handed_off: Path, tmp_path: Path):
    source = tmp_path / "export.json"
    source.write_text('{"retention":[{"time_seconds":0,"retention_percent":100}]}', encoding="utf-8")
    prior = analytics.import_studio_export(handed_off, source)
    (handed_off / "final.mp4").write_bytes(b"replaced video")
    with pytest.raises(ValueError, match="handoff|revision"):
        analytics.import_studio_export(handed_off, source)
    (handed_off / "final.mp4").write_bytes(b"approved video")
    archive = handed_off / "review-handoff.zip"
    with zipfile.ZipFile(archive) as bundle:
        receipt = bundle.read("handoff-manifest.json")
        contents = {name: bundle.read(name) for name in bundle.namelist() if name != "handoff-manifest.json"}
    contents["video.mp4"] = b"tampered"
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, data in contents.items():
            bundle.writestr(name, data)
        bundle.writestr("handoff-manifest.json", receipt)
    with pytest.raises(ValueError, match="handoff|revision"):
        analytics.import_studio_export(handed_off, source)
    assert analytics.list_imports(handed_off)[0]["id"] == prior["id"]


def test_newest_import_is_first_for_result_card(handed_off: Path, tmp_path: Path, monkeypatch):
    from types import SimpleNamespace
    identifiers = iter(("f" * 32, "0" * 32))
    monkeypatch.setattr(analytics.uuid, "uuid4", lambda: SimpleNamespace(hex=next(identifiers)))
    source = tmp_path / "studio.json"
    source.write_text('{"retention":[{"time_seconds":0,"retention_percent":100}]}', encoding="utf-8")
    first = analytics.import_studio_export(handed_off, source, notes="first")
    second = analytics.import_studio_export(handed_off, source, notes="second")
    assert analytics.list_imports(handed_off)[0]["id"] == second["id"]
    assert first["id"] != second["id"]


def test_rejects_other_jobs_receipt(handed_off: Path, tmp_path: Path):
    path = handed_off / "review-handoff.zip"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("handoff-manifest.json", '{"job_id":"other","qa_passed":true,"approved_revision":{}}')
    source = tmp_path / "export.json"
    source.write_text('{"retention": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="handoff"):
        analytics.import_studio_export(handed_off, source)
