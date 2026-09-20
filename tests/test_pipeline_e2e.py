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

import json
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
