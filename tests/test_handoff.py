import hashlib
import json
import os
import zipfile
from pathlib import Path

import pytest

from movie_review_factory.handoff import build_handoff
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest


@pytest.fixture
def ready_job(tmp_path: Path) -> Path:
    create_job(tmp_path, JobConfig(job_id="review-sample"))
    artifacts = {
        "script.json": {"approved": True, "sections": [{"title": "Mở đầu"}, {"title": "Bình luận"}]},
        "scene_plan.json": {"clips": [
            {"section_index": 1, "start_seconds": 0.0},
            {"section_index": 2, "start_seconds": 61.2},
        ]},
        "alignment.json": {"cues": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "Xin chào"},
        ]},
        "render.json": {"clips": [
            {"section_index": 1, "start_seconds": 240, "duration_seconds": 61.2},
            {"section_index": 2, "start_seconds": 15, "duration_seconds": 8},
        ]},
        "qa.json": {"passed": True, "output_file": "final.mp4"},
        "youtube_metadata.json": {
            "approved": True, "title": "Review phim", "description": "Tóm tắt và bình luận",
            "tags": ["review", "phim"],
        },
    }
    for name, data in artifacts.items():
        (tmp_path / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for name, content in {
        "final.mp4": b"\x00sample-video\xff",
        "aligned.srt": "1\n00:00:00,000 --> 00:00:01,000\nXin chào\n".encode(),
        "thumbnail.jpg": b"\xff\xd8sample-jpeg\xff\xd9",
    }.items():
        (tmp_path / name).write_bytes(content)
    manifest = load_manifest(tmp_path)
    for name in ("script", "scene_plan", "alignment", "render", "qa", "metadata", "thumbnail"):
        manifest.stage(name).mark("ready")
    save_manifest(tmp_path, manifest)
    return tmp_path


def test_handoff_contains_review_assets_chapters_and_checksums(ready_job: Path) -> None:
    result = build_handoff(ready_job)
    assert result == ready_job / "review-handoff.zip"
    with zipfile.ZipFile(result) as archive:
        assert set(archive.namelist()) == {
            "video.mp4", "subtitles.srt", "subtitles.vtt", "thumbnail.jpg", "youtube_metadata.json",
            "chapters.txt", "upload-notes.txt", "handoff-manifest.json",
        }
        assert archive.read("chapters.txt").decode() == "00:00:00 Mở đầu\n00:01:01 Bình luận\n"
        assert archive.read("subtitles.srt") == (ready_job / "aligned.srt").read_bytes()
        assert b"00:00:00.000 --> 00:00:01.000" in archive.read("subtitles.vtt")
        receipt = json.loads(archive.read("handoff-manifest.json"))
        for name, details in receipt["contents"].items():
            data = archive.read(name)
            assert details == {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def test_handoff_rejects_pending_qa_even_if_old_video_exists(ready_job: Path) -> None:
    manifest = load_manifest(ready_job)
    manifest.stage("qa").mark("pending")
    save_manifest(ready_job, manifest)
    with pytest.raises(ValueError, match="qa"):
        build_handoff(ready_job)
    assert not (ready_job / "review-handoff.zip").exists()


@pytest.mark.parametrize("file,field", [("script.json", "approved"), ("youtube_metadata.json", "approved"), ("qa.json", "passed")])
def test_handoff_requires_gates(ready_job: Path, file: str, field: str) -> None:
    path = ready_job / file
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload[field] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="blocked"):
        build_handoff(ready_job)


def test_download_rejects_handoff_after_metadata_reapproved_without_rebuild(ready_job: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from movie_review_factory.webapp import JobsService

    build_handoff(ready_job)
    service = JobsService(ready_job.parent)
    monkeypatch.setattr(service, "_require_job", lambda _job_id: ready_job)
    assert service.artifact_path("review-sample", "review-handoff.zip").is_file()
    metadata = json.loads((ready_job / "youtube_metadata.json").read_text(encoding="utf-8"))
    metadata["title"] = "Một tiêu đề mới"
    path = ready_job / "youtube_metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    # A copied/restored file can retain an old timestamp; revision hashes must decide.
    zip_time = (ready_job / "review-handoff.zip").stat().st_mtime_ns
    os.utime(path, ns=(zip_time - 1_000_000, zip_time - 1_000_000))
    with pytest.raises(ValueError, match="hết hiệu lực|cũ|changed"):
        service.artifact_path("review-sample", "review-handoff.zip")


def test_handoff_keeps_previous_zip_if_asset_disappears(ready_job: Path) -> None:
    first = build_handoff(ready_job)
    previous = first.read_bytes()
    (ready_job / "aligned.srt").unlink()
    with pytest.raises(ValueError, match="aligned.srt"):
        build_handoff(ready_job)
    assert first.read_bytes() == previous
