import json
import shutil
import subprocess
from pathlib import Path

import pytest

from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest
from movie_review_factory.short_variants import build_short, short_artifact


@pytest.fixture
def review(tmp_path: Path) -> Path:
    create_job(tmp_path, JobConfig(job_id="short-demo"))
    (tmp_path / "final.mp4").write_bytes(b"approved-review")
    (tmp_path / "script.json").write_text(
        json.dumps({"approved": True, "sections": [{"narration": "Lời bình riêng"}]}),
        encoding="utf-8",
    )
    (tmp_path / "qa.json").write_text(
        json.dumps({"passed": True, "output_file": "final.mp4",
                    "checks": [{"check": "positive_duration", "value": 70.0, "passed": True}]}),
        encoding="utf-8",
    )
    (tmp_path / "alignment.json").write_text(
        json.dumps({"cues": [
            {"start_seconds": 8.0, "end_seconds": 10.5, "text": "Lời ở trước"},
            {"start_seconds": 12.0, "end_seconds": 14.0, "text": "Bình luận chính"},
            {"start_seconds": 17.0, "end_seconds": 20.0, "text": "Kết phần bình luận"},
            {"start_seconds": 24.0, "end_seconds": 25.0, "text": "Lời ở sau"},
        ]}), encoding="utf-8",
    )
    manifest = load_manifest(tmp_path)
    for stage in ("script", "alignment", "render", "qa"):
        manifest.stage(stage).mark("ready")
    save_manifest(tmp_path, manifest)
    return tmp_path


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                    reason="FFmpeg tools unavailable")
def test_short_encodes_playable_portrait_with_audio_and_outro(review: Path) -> None:
    source = review / "final.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
        "-t", "15", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        str(source),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    exported = build_short(review, 10, 13)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries",
        "stream=codec_type,width,height:format=duration", "-of", "json", str(exported),
    ], check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    assert 4.5 <= float(info["format"]["duration"]) <= 5.5
    assert any(s.get("width") == 1080 and s.get("height") == 1920
               for s in info["streams"] if s["codec_type"] == "video")
    assert any(s["codec_type"] == "audio" for s in info["streams"])


def test_short_burns_captions_through_frame_sized_ass(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from movie_review_factory import short_variants

    source = review / "final.mp4"
    source.write_bytes(b"approved-review-body")
    captured: dict[str, str] = {}

    def fake_run(command, **kwargs):
        captured["filter"] = command[command.index("-filter_complex") + 1]
        # Read the generated ASS before build_short's finally deletes it.
        captured["ass"] = (review / "shorts" / "short-review-render.ass").read_text(
            encoding="utf-8"
        )
        (review / "shorts" / "short-review-render.tmp").write_bytes(b"portrait-mp4")

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    monkeypatch.setattr(short_variants.subprocess, "run", fake_run)
    output = build_short(review, 10, 13)
    assert output.name == "short-review.mp4"

    # Captions burn through the shared frame-sized ASS, never libass force_style.
    assert "ass='" in captured["filter"]
    assert "force_style" not in captured["filter"]
    assert "subtitles=" not in captured["filter"]

    ass = captured["ass"]
    # Frame-sized PlayRes is the guard against the 384x288 default that shoved
    # the portrait caption off its panel.
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "WrapStyle: 0" in ass
    # FontSize 45; Alignment=2 with MarginV 590 keeps the block inside the panel.
    assert "Style: Default,Arial,45," in ass
    assert "1,2,1,2,81,81,590,1" in ass


def test_short_requires_approved_qa_passing_review(review: Path) -> None:
    manifest = load_manifest(review)
    manifest.stage("qa").mark("pending")
    save_manifest(review, manifest)
    with pytest.raises(ValueError, match="qa"):
        build_short(review, 10, 20)
    assert not (review / "shorts" / "short-review.mp4").exists()


def test_short_rejects_unapproved_script(review: Path) -> None:
    script_path = review / "script.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    script["approved"] = False
    script_path.write_text(json.dumps(script), encoding="utf-8")
    with pytest.raises(ValueError, match="approved script"):
        build_short(review, 10, 20)


def test_short_requires_nonempty_commentary_highlight(review: Path) -> None:
    with pytest.raises(ValueError, match="commentary"):
        build_short(review, 25, 30)


def test_short_export_retimes_cues_and_uses_final_review_only(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"rendered-short")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    exported = build_short(review, 10, 20, ffmpeg="ffmpeg")
    assert exported == review / "shorts" / "short-review.mp4"
    subtitle = (review / "shorts" / "short-review.srt").read_text(encoding="utf-8")
    assert "00:00:02,000 --> 00:00:04,000" in subtitle
    assert "00:00:07,000 --> 00:00:10,000" in subtitle
    assert "00:00:00,000 --> 00:00:00,500" in subtitle
    assert "Lời ở trước" in subtitle
    assert "Lời ở sau" not in subtitle
    command = calls[0]
    assert str(review / "final.mp4") in command
    assert "-filter_complex" in command
    assert "1080:1920" in " ".join(command)
    assert "libx264" in command and "aac" in command
    receipt = json.loads((review / "shorts" / "short-review.json").read_text(encoding="utf-8"))
    assert receipt["source"] == "final.mp4"
    assert receipt["review_range_seconds"] == [10.0, 20.0]
    assert receipt["qa_passed"] is True


def test_short_retries_single_threaded_on_x264_oom(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        if len(calls) == 1:  # first attempt dies with the x264 allocation flake
            return subprocess.CompletedProcess(
                command, 1, "",
                "x264 [error]: malloc of size 7186688 failed\nCannot allocate memory",
            )
        Path(command[-1]).write_bytes(b"rendered-short")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    exported = build_short(review, 10, 20, ffmpeg="ffmpeg")

    assert exported == review / "shorts" / "short-review.mp4"
    assert len(calls) == 2  # exactly one bounded retry
    assert "-threads" in calls[0]  # first attempt is thread-capped
    # the retry forces single-threaded x264
    assert calls[1][calls[1].index("-threads") + 1] == "1"


def test_short_does_not_retry_on_non_memory_failure(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_run(command, **kwargs):
        calls.append(list(command))
        return subprocess.CompletedProcess(
            command, 1, "", "Invalid data found when processing input",
        )

    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="short FFmpeg render failed"):
        build_short(review, 10, 20, ffmpeg="ffmpeg")
    assert len(calls) == 1  # no retry for a non-allocation failure


def test_short_wraps_long_review_cue_for_portrait_safe_zone(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aligned = review / "alignment.json"
    aligned.write_text(json.dumps({"cues": [
        {"start_seconds": 10, "end_seconds": 16,
         "text": "Lời kể quá dài cần chia theo nhịp\nđể người xem đọc màn hình dọc thoải mái"}
    ]}), encoding="utf-8")
    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"portrait")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    build_short(review, 10, 20, ffmpeg="ffmpeg")
    srt = (review / "shorts" / "short-review.srt").read_text(encoding="utf-8")
    blocks = srt.strip().split("\n\n")
    assert len(blocks) >= 2
    for block in blocks:
        lines = block.splitlines()[2:]
        assert 1 <= len(lines) <= 2
        assert all(len(line) <= 25 for line in lines)


def test_short_download_rejects_stale_approval_or_source(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"portrait")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    build_short(review, 10, 20, ffmpeg="ffmpeg")
    assert short_artifact(review, "short-review.mp4").is_file()
    assert short_artifact(review, "short-review.srt").is_file()
    with pytest.raises(ValueError, match="artifact"):
        short_artifact(review, "../final.mp4")
    (review / "final.mp4").write_bytes(b"modified-review")
    with pytest.raises(ValueError, match="stale"):
        short_artifact(review, "short-review.mp4")


def test_short_download_rejects_unapproved_qa(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"portrait")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    build_short(review, 10, 20, ffmpeg="ffmpeg")
    qa_path = review / "qa.json"
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    qa["passed"] = False
    qa_path.write_text(json.dumps(qa), encoding="utf-8")
    with pytest.raises(ValueError, match="QA"):
        short_artifact(review, "short-review.mp4")


def test_failed_rerender_keeps_previous_video_subtitle_and_receipt(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def first(command, **kwargs):
        Path(command[-1]).write_bytes(b"first-video")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", first)
    build_short(review, 10, 20, ffmpeg="ffmpeg")
    folder = review / "shorts"
    original = {name: (folder / name).read_bytes() for name in
                ("short-review.mp4", "short-review.srt", "short-review.json")}

    def fail(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial-video")
        return subprocess.CompletedProcess(command, 1, "", "encode failed")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fail)
    with pytest.raises(RuntimeError, match="encode failed"):
        build_short(review, 12, 20, ffmpeg="ffmpeg")
    assert {name: (folder / name).read_bytes() for name in original} == original
    assert short_artifact(review, "short-review.mp4").read_bytes() == b"first-video"


@pytest.mark.parametrize("name", ["short-review.mp4", "short-review.srt"])
def test_short_download_rejects_modified_output(
    review: Path, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"portrait")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    build_short(review, 10, 20, ffmpeg="ffmpeg")
    (review / "shorts" / name).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="stale"):
        short_artifact(review, name)


def test_short_aborts_when_script_changes_during_render(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = review / "shorts" / "short-review.mp4"
    target.parent.mkdir()
    target.write_bytes(b"previous-export")
    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"new-export")
        (review / "script.json").write_text('{"approved": false}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="changed"):
        build_short(review, 10, 20, ffmpeg="ffmpeg")
    assert target.read_bytes() == b"previous-export"


def test_short_rejects_invalid_bounds_without_touching_previous_export(
    review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = review / "shorts" / "short-review.mp4"
    target.parent.mkdir()
    target.write_bytes(b"old-good-short")
    with pytest.raises(ValueError, match="duration|range"):
        build_short(review, 66, 73)
    assert target.read_bytes() == b"old-good-short"
