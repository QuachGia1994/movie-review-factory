"""End-to-end evidence for the real user-facing MVP flow.

The rest of the suite exercises each stage in isolation and drives run_job in
text-only mode; every render/qa test replaces subprocess.run with a stub, so no
test actually invokes FFmpeg or produces a real final.mp4. This test closes that
gap: it runs the real pipeline on the real sample video through a real FFmpeg
render and a real FFprobe QA pass, then through the publish gate.

TTS is the one stage that needs the network (edge-tts is a cloud service), so it
stays out: this test supplies narration.mp3 + voice.json locally and lets the tts
stage skip honestly. No server is started and no network call is made.
"""

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import (
    approve_metadata,
    create_job,
    load_manifest,
    run_job,
    save_manifest,
    _probe_duration_seconds,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_VIDEO = REPO_ROOT / "data" / "raw" / "ultracode-smoke-sample.mp4"

pytestmark = pytest.mark.skipif(
    not (shutil.which("ffmpeg") and shutil.which("ffprobe")),
    reason="end-to-end render needs ffmpeg and ffprobe on PATH",
)


def _write_narration_fixture(job_dir: Path, seconds: float = 6.0) -> None:
    """Stand in for the network TTS stage with a local silent narration track."""
    audio = job_dir / "narration.mp3"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", "anullsrc=channel_layout=mono:sample_rate=44100",
         "-t", f"{seconds}", "-c:a", "libmp3lame", "-q:a", "9", str(audio)],
        capture_output=True, text=True, check=True,
    )
    (job_dir / "voice.json").write_text(
        json.dumps({
            "job_id": job_dir.name,
            "language": "vi",
            "engine": "test-fixture-silence",
            "voice": "offline",
            "audio_file": "narration.mp3",
            "section_count": 1,
            "sections": [],
        }, indent=2),
        encoding="utf-8",
    )
    (job_dir / "captions.srt").write_text("", encoding="utf-8")


def _write_captions_fixture(job_dir: Path, seconds: float = 6.0) -> None:
    (job_dir / "captions.srt").write_text(
        f"1\n00:00:00,000 --> 00:00:{seconds:06.3f}".replace(".", ",") + "\nSmoke test narration\n",
        encoding="utf-8",
    )


def _approve(job_dir: Path, filename: str) -> None:
    path = job_dir / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    data["approved"] = True
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason=f"sample video missing: {SAMPLE_VIDEO}")
def test_end_to_end_real_render_then_publish_gate(tmp_path: Path) -> None:
    cfg = JobConfig(
        job_id="e2e-smoke",
        language="vi",
        target_minutes=1,
        aspect_ratio="16:9",
        source_video=SAMPLE_VIDEO,
    )
    create_job(tmp_path, cfg)

    # Text stages first, so script.json exists to approve (the gate TTS/render
    # and publish require).
    run_job(tmp_path, until="script")
    _approve(tmp_path, "script.json")

    # Supply the network stage's output locally, then run the rest for real.
    _write_narration_fixture(tmp_path)
    _write_captions_fixture(tmp_path)
    manifest = run_job(tmp_path)
    by_stage = {s.stage: s for s in manifest.stages}

    # Real ffprobe ingest of the real sample video.
    assert by_stage["ingest"].status == "ready", by_stage["ingest"].message
    # Real FFmpeg render produced a real, non-empty final.mp4.
    assert by_stage["render"].status == "ready", by_stage["render"].message
    final_mp4 = tmp_path / "final.mp4"
    assert final_mp4.exists() and final_mp4.stat().st_size > 0
    # Real FFprobe QA validated the rendered file (qa raises if any check fails,
    # so a ready qa stage is proof every check passed).
    assert by_stage["qa"].status == "ready", by_stage["qa"].message
    assert (tmp_path / "qa.json").exists()

    # Publish gate is closed while metadata is unapproved: no record, no upload.
    assert by_stage["publish"].status == "skipped"
    assert "blocked" in by_stage["publish"].message
    assert not (tmp_path / "publish_record.json").exists()

    # Approve metadata; the gate now writes the handoff record (never uploads).
    approve_metadata(tmp_path)
    pub = next(s for s in run_job(tmp_path).stages if s.stage == "publish")
    assert pub.status == "ready", pub.message
    record = json.loads((tmp_path / "publish_record.json").read_text(encoding="utf-8"))
    assert record["publish_ready"] is True


@pytest.mark.skipif(
    os.environ.get("MRF_RUN_LIVE_TTS_TESTS") != "1",
    reason="live Edge-TTS network test is opt-in",
)
@pytest.mark.skipif(
    importlib.util.find_spec("edge_tts") is None,
    reason="edge-tts is not installed",
)
@pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason=f"sample video missing: {SAMPLE_VIDEO}")
def test_live_edge_tts_alignment_render_and_qa(tmp_path: Path) -> None:
    cfg = JobConfig(
        job_id="live-tts-e2e",
        language="vi",
        target_minutes=1,
        aspect_ratio="16:9",
        source_video=SAMPLE_VIDEO,
    )
    create_job(tmp_path, cfg)

    script = {
        "job_id": cfg.job_id,
        "language": "vi",
        "approved": True,
        "notes": "live Edge-TTS verification",
        "sections": [{
            "title": "Smoke",
            "duration_seconds": 6.0,
            "narration": "Xin chào. Đây là kiểm tra đường sản xuất thật của Movie Review Factory.",
        }],
    }
    (tmp_path / "script.json").write_text(
        json.dumps(script, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    source_duration = _probe_duration_seconds(SAMPLE_VIDEO)
    assert source_duration and source_duration > 0
    source_end = min(float(source_duration), 3.0)
    scene_plan = {
        "job_id": cfg.job_id,
        "language": "vi",
        "generator": "verification",
        "clips": [{
            "section": "Smoke",
            "section_index": 1,
            "shot_index": 1,
            "shot_count": 1,
            "type": "narration",
            "start_seconds": 0.0,
            "duration_seconds": 6.0,
            "source_clip": {"start_seconds": 0.0, "end_seconds": source_end},
            "notes": "owned verification sample",
        }],
        "notes": "live TTS verification plan",
    }
    (tmp_path / "scene_plan.json").write_text(
        json.dumps(scene_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = load_manifest(tmp_path)
    for name in (
        "ingest", "research", "transcript", "scenes", "outline", "script", "scene_plan"
    ):
        manifest.stage(name).status = "ready"
    save_manifest(tmp_path, manifest)

    after_tts = run_job(tmp_path, until="tts")
    assert after_tts.stage("tts").status == "ready", after_tts.stage("tts").message
    voice = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert voice["engine"] == "edge-tts"

    narration_duration = _probe_duration_seconds(tmp_path / "narration.mp3")
    assert narration_duration and narration_duration > 0
    scene_plan["clips"][0]["duration_seconds"] = narration_duration
    (tmp_path / "scene_plan.json").write_text(
        json.dumps(scene_plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_manifest = run_job(tmp_path, until="qa")
    for name in ("alignment", "render", "qa"):
        assert final_manifest.stage(name).status == "ready", final_manifest.stage(name).message

    alignment = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert alignment["cue_source"] == "voice.json"
    assert alignment["cue_count"] == 1
    final_mp4 = tmp_path / "final.mp4"
    assert final_mp4.exists() and final_mp4.stat().st_size > 0
    qa = json.loads((tmp_path / "qa.json").read_text(encoding="utf-8"))
    assert qa["passed"] is True


@pytest.mark.skipif(
    os.environ.get("MRF_RUN_LIVE_WHISPER_TESTS") != "1",
    reason="live faster-whisper network test is opt-in",
)
@pytest.mark.skipif(
    importlib.util.find_spec("faster_whisper") is None,
    reason="faster-whisper is not installed",
)
@pytest.mark.skipif(
    importlib.util.find_spec("edge_tts") is None,
    reason="edge-tts is not installed",
)
def test_live_whisper_download_cache_offline_and_scenes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import edge_tts

    source = tmp_path / "owned-vietnamese.mp3"
    edge_tts.Communicate(
        (
            "Xin chào. Đây là đoạn âm thanh tiếng Việt do hệ thống tự tạo để "
            "kiểm tra nhận dạng giọng nói và bộ nhớ đệm mô hình."
        ),
        "vi-VN-HoaiMyNeural",
    ).save_sync(str(source))
    assert source.exists() and source.stat().st_size > 0

    cache = tmp_path / "whisper-cache"
    monkeypatch.setenv("MRF_WHISPER_CACHE", str(cache))
    monkeypatch.delenv("MRF_WHISPER_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)

    online = tmp_path / "online"
    create_job(
        online,
        JobConfig(
            job_id="live-whisper-online",
            language="vi",
            target_minutes=1,
            source_video=source,
        ),
    )
    first = run_job(online, until="scenes")
    for name in ("ingest", "transcript", "scenes"):
        assert first.stage(name).status == "ready", first.stage(name).message

    duration = _probe_duration_seconds(source)
    assert duration and duration > 0

    transcript = json.loads((online / "transcript.json").read_text(encoding="utf-8"))
    segments = transcript["segments"]
    assert segments
    assert sum(len(segment["text"].strip()) for segment in segments) >= 10
    for segment in segments:
        start = float(segment["start_seconds"])
        end = float(segment["end_seconds"])
        assert 0 <= start < end <= float(duration) + 0.25
        assert segment["text"].strip()

    scenes_doc = json.loads((online / "scenes.json").read_text(encoding="utf-8"))
    scenes = scenes_doc["scenes"]
    assert scenes
    for scene in scenes:
        assert 0 <= float(scene["start_seconds"]) < float(scene["end_seconds"])
        assert float(scene["end_seconds"]) <= float(duration) + 0.25

    assert cache.exists()
    assert any(path.is_file() for path in cache.rglob("*"))

    monkeypatch.setenv("MRF_WHISPER_OFFLINE", "1")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    offline = tmp_path / "offline"
    create_job(
        offline,
        JobConfig(
            job_id="live-whisper-offline",
            language="vi",
            target_minutes=1,
            source_video=source,
        ),
    )
    second = run_job(offline, until="scenes")
    for name in ("ingest", "transcript", "scenes"):
        assert second.stage(name).status == "ready", second.stage(name).message

    offline_transcript = json.loads(
        (offline / "transcript.json").read_text(encoding="utf-8")
    )
    assert offline_transcript["segments"]
    assert [segment["text"] for segment in offline_transcript["segments"]] == [
        segment["text"] for segment in segments
    ]
