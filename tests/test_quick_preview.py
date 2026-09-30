import json
import shutil
import subprocess
from pathlib import Path

import pytest

from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest
from movie_review_factory.quick_preview import build_section_preview, section_preview_artifact


@pytest.fixture
def section_job(tmp_path: Path) -> Path:
    create_job(tmp_path, JobConfig(job_id="preview", source_video=str(tmp_path / "source.mp4")))
    (tmp_path / "source.mp4").write_bytes(b"source")
    (tmp_path / "narration.mp3").write_bytes(b"voice")
    (tmp_path / "script.json").write_text(json.dumps({
        "approved": True, "sections": [
            {"title": "Mở đầu", "narration": "Lời giới thiệu"},
            {"title": "Phân tích", "narration": "Điều bất ngờ"},
        ],
    }), encoding="utf-8")
    (tmp_path / "scene_plan.json").write_text(json.dumps({
        "clips": [
            {"section_index": 1, "section": "Mở đầu", "source_clip": {"start_seconds": 0, "end_seconds": 1}, "duration_seconds": 1},
            {"section_index": 2, "section": "Phân tích", "source_clip": {"start_seconds": 2, "end_seconds": 3}, "duration_seconds": 1},
            {"section_index": 2, "section": "Phân tích", "source_clip": {"start_seconds": 4, "end_seconds": 5}, "duration_seconds": 1},
        ], "aspect_ratio": "16:9"
    }), encoding="utf-8")
    (tmp_path / "alignment.json").write_text(json.dumps({
        "section_bounds": [
            {"section_index": 1, "start_seconds": 0, "end_seconds": 2},
            {"section_index": 2, "start_seconds": 2, "end_seconds": 6},
        ],
        "cues": [
            {"start_seconds": 0, "end_seconds": 2, "text": "Lời giới thiệu"},
            {"start_seconds": 2.5, "end_seconds": 3.5, "text": "Điều bất ngờ"},
            {"start_seconds": 4, "end_seconds": 5, "text": "Có thể hiểu"},
        ],
    }), encoding="utf-8")
    manifest = load_manifest(tmp_path)
    for stage in ("script", "scene_plan", "tts", "alignment"):
        manifest.stage(stage).mark("ready")
    save_manifest(tmp_path, manifest)
    return tmp_path


def test_preview_selects_only_one_section_and_retimes_captions(section_job, monkeypatch):
    import movie_review_factory.quick_preview as preview
    monkeypatch.setattr(preview, "_probe_duration_seconds", lambda path, ffprobe: 10.0)
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"preview-video")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(preview.subprocess, "run", fake_run)
    output = build_section_preview(section_job, 2, ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert output.name == "section-2.mp4"
    srt = (section_job / "previews" / "section-2.srt").read_text(encoding="utf-8")
    assert "00:00:00,500 --> 00:00:01,500" in srt
    assert "00:00:02,000 --> 00:00:03,000" in srt
    assert "Lời giới thiệu" not in srt
    receipt = json.loads((section_job / "previews" / "section-2.json").read_text(encoding="utf-8"))
    assert receipt["section_index"] == 2
    assert receipt["duration_seconds"] == 4.0
    assert receipt["source_ranges"] == [[2.0, 3.0], [4.0, 5.0]]
    assert section_preview_artifact(section_job, 2, "mp4") == output


def test_preview_retries_single_threaded_on_x264_oom(section_job, monkeypatch):
    import movie_review_factory.quick_preview as preview
    monkeypatch.setattr(preview, "_probe_duration_seconds", lambda path, ffprobe: 10.0)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if len(calls) == 1:  # first attempt dies with the x264 allocation flake
            return subprocess.CompletedProcess(
                cmd, 1, "", "x264 [error]: malloc of size 5759552 failed\nCannot allocate memory",
            )
        Path(cmd[-1]).write_bytes(b"preview-video")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(preview.subprocess, "run", fake_run)
    output = build_section_preview(section_job, 2, ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert output.name == "section-2.mp4"
    assert len(calls) == 2  # exactly one bounded retry
    assert "-threads" in calls[0]  # first attempt is thread-capped
    assert calls[1][calls[1].index("-threads") + 1] == "1"  # retry forces single thread


def test_preview_does_not_retry_on_non_memory_failure(section_job, monkeypatch):
    import movie_review_factory.quick_preview as preview
    monkeypatch.setattr(preview, "_probe_duration_seconds", lambda path, ffprobe: 10.0)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1, "", "Invalid data found when processing input")

    monkeypatch.setattr(preview.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="section preview render failed"):
        build_section_preview(section_job, 2, ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert len(calls) == 1  # no retry for a non-allocation failure


def test_preview_stale_if_script_or_plan_changes(section_job, monkeypatch):
    import movie_review_factory.quick_preview as preview
    monkeypatch.setattr(preview, "_probe_duration_seconds", lambda path, ffprobe: 10.0)
    def fake_run(cmd, **kwargs):
        Path(cmd[-1]).write_bytes(b"preview")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(preview.subprocess, "run", fake_run)
    build_section_preview(section_job, 2, ffmpeg="ffmpeg", ffprobe="ffprobe")
    path = section_job / "scene_plan.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    plan["clips"][1]["source_clip"]["start_seconds"] = 2.2
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        section_preview_artifact(section_job, 2, "mp4")


def test_preview_requires_approval_and_precise_bounds(section_job):
    script = section_job / "script.json"
    data = json.loads(script.read_text(encoding="utf-8"))
    data["approved"] = False
    script.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="approved"):
        build_section_preview(section_job, 1)
    data["approved"] = True
    script.write_text(json.dumps(data), encoding="utf-8")
    alignment = section_job / "alignment.json"
    alignment.write_text('{"cues": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="section bounds"):
        build_section_preview(section_job, 1, ffmpeg="ffmpeg", ffprobe="ffprobe")


def test_preview_failure_preserves_previous_export(section_job, monkeypatch):
    import movie_review_factory.quick_preview as preview
    monkeypatch.setattr(preview, "_probe_duration_seconds", lambda path, ffprobe: 10.0)
    folder = section_job / "previews"
    folder.mkdir()
    target = folder / "section-2.mp4"
    target.write_bytes(b"old")
    monkeypatch.setattr(preview.subprocess, "run", lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, 1, "", "encoder failed"))
    with pytest.raises(RuntimeError, match="encoder failed"):
        build_section_preview(section_job, 2, ffmpeg="ffmpeg", ffprobe="ffprobe")
    assert target.read_bytes() == b"old"


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg unavailable")
def test_preview_540p_smoke(tmp_path):
    source = tmp_path / "source.mp4"
    audio = tmp_path / "narration.mp3"
    create_job(tmp_path, JobConfig(job_id="smoke", source_video=str(source)))
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=25",
                    "-t", "7", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                   check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
                    "-t", "7", "-c:a", "libmp3lame", str(audio)], check=True, capture_output=True)
    (tmp_path / "script.json").write_text('{"approved": true, "sections": [{"title": "Đoạn", "narration": "Một câu ngắn"}]}', encoding="utf-8")
    (tmp_path / "scene_plan.json").write_text(json.dumps({"aspect_ratio": "16:9", "clips": [
        {"section_index": 1, "section": "Đoạn", "duration_seconds": 3,
         "source_clip": {"start_seconds": 1, "end_seconds": 2}},
        {"section_index": 1, "section": "Đoạn", "duration_seconds": 3,
         "source_clip": {"start_seconds": 4, "end_seconds": 6}},
    ]}), encoding="utf-8")
    (tmp_path / "alignment.json").write_text(json.dumps({
        "section_bounds": [{"section_index": 1, "start_seconds": 0.5, "end_seconds": 4.5}],
        "cues": [{"start_seconds": 1.0, "end_seconds": 2.0, "text": "Một câu ngắn"}],
    }), encoding="utf-8")
    manifest = load_manifest(tmp_path)
    for stage in ("script", "scene_plan", "tts", "alignment"):
        manifest.stage(stage).mark("ready")
    save_manifest(tmp_path, manifest)
    output = build_section_preview(tmp_path, 1)
    probe = subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_streams", "-show_format", str(output)],
                           check=True, capture_output=True, text=True)
    doc = json.loads(probe.stdout)
    assert any(s.get("width") == 960 and s.get("height") == 540 for s in doc["streams"])
    assert any(s["codec_type"] == "audio" for s in doc["streams"])
    assert 3.7 <= float(doc["format"]["duration"]) <= 4.3
