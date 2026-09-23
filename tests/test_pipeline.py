import json
import sys
import types
from pathlib import Path

import pytest

from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import (
    SkipStage,
    create_job,
    job_status,
    load_manifest,
    run_job,
    validate_job,
)


@pytest.fixture()
def job_dir(tmp_path: Path) -> Path:
    cfg = JobConfig(job_id="test-job", language="vi", target_minutes=5)
    create_job(tmp_path, cfg)
    return tmp_path


def test_create_job_produces_valid_manifest(tmp_path: Path) -> None:
    cfg = JobConfig(job_id="demo", language="vi", target_minutes=10)
    create_job(tmp_path, cfg)
    errors = validate_job(tmp_path)
    assert errors == []


def test_validate_job_missing_manifest(tmp_path: Path) -> None:
    errors = validate_job(tmp_path)
    assert any("missing" in e for e in errors)


def test_run_job_no_video_skips_ingest_completes_text_stages(job_dir: Path) -> None:
    manifest = run_job(job_dir)
    by_stage = {s.stage: s for s in manifest.stages}

    # ingest skips because no source_video
    assert by_stage["ingest"].status == "skipped"

    # text-only stages complete
    for stage in ("research", "outline", "script", "scene_plan", "metadata"):
        assert by_stage[stage].status == "ready", f"{stage} expected ready, got {by_stage[stage].status}"

    # media stages skipped honestly
    for stage in ("transcript", "scenes", "tts", "alignment", "render", "qa"):
        assert by_stage[stage].status == "skipped"

    # publish skips because approvals are missing
    assert by_stage["publish"].status == "skipped"
    assert "not approved" in by_stage["publish"].message or "missing" in by_stage["publish"].message


def test_run_job_produces_artifacts(job_dir: Path) -> None:
    run_job(job_dir)
    for name in ("research.json", "outline.json", "script.json", "script.md", "scene_plan.json", "youtube_metadata.json"):
        assert (job_dir / name).exists(), f"{name} missing"


def test_outline_time_budget(job_dir: Path) -> None:
    run_job(job_dir)
    outline = json.loads((job_dir / "outline.json").read_text(encoding="utf-8"))
    total = sum(s["budget_minutes"] for s in outline["sections"])
    assert abs(total - 5.0) < 0.1


def test_script_inherits_outline_sections(job_dir: Path) -> None:
    run_job(job_dir)
    outline = json.loads((job_dir / "outline.json").read_text(encoding="utf-8"))
    script = json.loads((job_dir / "script.json").read_text(encoding="utf-8"))
    outline_titles = [s["title"] for s in outline["sections"]]
    script_titles = [s["title"] for s in script["sections"]]
    assert script_titles == outline_titles
    assert script["approved"] is False
    assert all(section["narration"].strip() for section in script["sections"])


def test_scene_plan_clip_count_matches_script(job_dir: Path) -> None:
    run_job(job_dir)
    script = json.loads((job_dir / "script.json").read_text(encoding="utf-8"))
    scene_plan = json.loads((job_dir / "scene_plan.json").read_text(encoding="utf-8"))
    assert len(scene_plan["clips"]) == len(script["sections"])


def test_publish_blocked_without_approvals(job_dir: Path) -> None:
    run_job(job_dir)
    status = job_status(job_dir)
    pub = next(s for s in status["stages"] if s["stage"] == "publish")
    assert pub["status"] == "skipped"
    assert "blocked" in pub["message"]


def test_publish_ready_after_approvals(job_dir: Path) -> None:
    run_job(job_dir)

    # Manually approve script
    script_path = job_dir / "script.json"
    data = json.loads(script_path.read_text(encoding="utf-8"))
    data["approved"] = True
    script_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Manually approve metadata
    meta_path = job_dir / "youtube_metadata.json"
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    data["approved"] = True
    meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # Publish handoff also requires a verified final render.
    (job_dir / "final.mp4").write_bytes(b"verified-video")
    (job_dir / "qa.json").write_text(
        json.dumps({"passed": True}, indent=2),
        encoding="utf-8",
    )

    # Reset publish stage to pending so runner picks it up
    manifest = load_manifest(job_dir)
    for name in ("qa", "thumbnail"):
        stage = next(s for s in manifest.stages if s.stage == name)
        stage.status = "ready"
    pub = next(s for s in manifest.stages if s.stage == "publish")
    pub.status = "pending"
    from movie_review_factory.pipeline import save_manifest
    save_manifest(job_dir, manifest)

    run_job(job_dir)
    assert (job_dir / "publish_record.json").exists()
    record = json.loads((job_dir / "publish_record.json").read_text(encoding="utf-8"))
    assert record["publish_ready"] is True


def test_job_is_complete_after_full_run_with_approvals(job_dir: Path) -> None:
    run_job(job_dir)

    for fname in ("script.json", "youtube_metadata.json"):
        p = job_dir / fname
        d = json.loads(p.read_text(encoding="utf-8"))
        d["approved"] = True
        p.write_text(json.dumps(d, indent=2), encoding="utf-8")

    manifest = load_manifest(job_dir)
    pub = next(s for s in manifest.stages if s.stage == "publish")
    pub.status = "pending"
    from movie_review_factory.pipeline import save_manifest
    save_manifest(job_dir, manifest)

    manifest = run_job(job_dir)
    assert manifest.is_complete


def test_resumability(job_dir: Path) -> None:
    """Running run_job twice on a completed job is a no-op (stages stay ready/skipped)."""
    run_job(job_dir)
    m1 = load_manifest(job_dir)
    run_job(job_dir)
    m2 = load_manifest(job_dir)
    for s1, s2 in zip(m1.stages, m2.stages):
        assert s1.status == s2.status


def test_transcript_writes_timed_segments_and_srt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "owned-sample.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="transcript-job", source_video=source))

    class FakeModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str) -> None:
            assert model_name == "small"
            assert device == "cpu"
            assert compute_type == "int8"

        def transcribe(self, source_path: str, **kwargs: object) -> tuple[list[object], object]:
            assert source_path == str(source)
            assert kwargs == {"language": "vi", "vad_filter": True}
            return [
                types.SimpleNamespace(start=0.0, end=1.25, text=" Xin chào "),
                types.SimpleNamespace(start=1.25, end=2.5, text="thế giới"),
            ], object()

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(WhisperModel=FakeModel))
    from movie_review_factory.pipeline import _transcript

    artifacts, message = _transcript(tmp_path, load_manifest(tmp_path))

    assert [artifact.name for artifact in artifacts] == ["transcript.json", "captions.srt"]
    assert message == "transcribed 2 segments"
    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["source_video"] == str(source)
    assert transcript["language"] == "vi"
    assert transcript["segments"] == [
        {"start_seconds": 0.0, "end_seconds": 1.25, "text": "Xin chào"},
        {"start_seconds": 1.25, "end_seconds": 2.5, "text": "thế giới"},
    ]
    assert (tmp_path / "captions.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,250\nXin chào\n\n"
        "2\n00:00:01,250 --> 00:00:02,500\nthế giới\n"
    )


def test_whisper_model_options_use_app_cache_and_offline_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline

    cache = tmp_path / "whisper-cache"
    monkeypatch.setenv("MRF_WHISPER_CACHE", str(cache))
    monkeypatch.setenv("MRF_WHISPER_OFFLINE", "1")

    assert pipeline._whisper_model_options() == {
        "download_root": str(cache),
        "local_files_only": True,
    }


# --- scenes stage -----------------------------------------------------------


def _scenes_job(tmp_path: Path, *, with_video: bool = True) -> Path:
    """Create a scenes-ready job. Returns the (touched) owned source path."""
    source = tmp_path / "owned-sample.mp4"
    if with_video:
        source.touch()
    create_job(
        tmp_path,
        JobConfig(job_id="scenes-job", source_video=source if with_video else None),
    )
    return source


def test_scenes_skips_without_source_video(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="no-video"))
    from movie_review_factory.pipeline import _scenes

    with pytest.raises(SkipStage):
        _scenes(tmp_path, load_manifest(tmp_path))
    assert not (tmp_path / "scenes.json").exists()


def test_scenes_skips_when_ffprobe_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scenes_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: None)
    with pytest.raises(SkipStage) as excinfo:
        pipeline._scenes(tmp_path, load_manifest(tmp_path))
    assert "ffprobe" in str(excinfo.value)
    assert not (tmp_path / "scenes.json").exists()


def test_scenes_indexes_transcript_into_bounded_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _scenes_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    # transcript.json as produced by the transcript stage. Segment 4 ends past
    # the true video duration and must be clamped to the probed bound.
    transcript = {
        "job_id": "scenes-job",
        "source_video": str(source),
        "language": "vi",
        "segments": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "A"},
            {"start_seconds": 1.2, "end_seconds": 2.0, "text": "B"},
            {"start_seconds": 5.0, "end_seconds": 6.0, "text": "C"},
            {"start_seconds": 9.0, "end_seconds": 12.0, "text": "D"},
        ],
    }
    (tmp_path / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 10.0)

    artifacts, message = pipeline._scenes(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["scenes.json"]
    assert message == "indexed 3 scenes from transcript"
    doc = json.loads((tmp_path / "scenes.json").read_text(encoding="utf-8"))
    assert doc["source_video"] == str(source)
    assert doc["duration_seconds"] == 10.0
    assert doc["scene_count"] == 3
    assert doc["scenes"] == [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 2.0, "segment_count": 2, "text": "A B"},
        {"index": 2, "start_seconds": 5.0, "end_seconds": 6.0, "segment_count": 1, "text": "C"},
        {"index": 3, "start_seconds": 9.0, "end_seconds": 10.0, "segment_count": 1, "text": "D"},
    ]
    # Every timestamp stays within the probed video bounds.
    for scene in doc["scenes"]:
        assert 0.0 <= scene["start_seconds"] <= scene["end_seconds"] <= 10.0


def test_scenes_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _scenes_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    transcript = {
        "segments": [
            {"start_seconds": 0.0, "end_seconds": 1.0, "text": "A"},
            {"start_seconds": 5.0, "end_seconds": 6.0, "text": "B"},
        ]
    }
    (tmp_path / "transcript.json").write_text(json.dumps(transcript), encoding="utf-8")
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 8.0)

    pipeline._scenes(tmp_path, load_manifest(tmp_path))
    first = (tmp_path / "scenes.json").read_text(encoding="utf-8")
    pipeline._scenes(tmp_path, load_manifest(tmp_path))
    second = (tmp_path / "scenes.json").read_text(encoding="utf-8")
    assert first == second


def test_scenes_without_transcript_produces_single_bounded_scene(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _scenes_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 12.5)
    artifacts, _ = pipeline._scenes(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["scenes.json"]
    doc = json.loads((tmp_path / "scenes.json").read_text(encoding="utf-8"))
    assert doc["scene_count"] == 1
    assert doc["scenes"] == [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 12.5, "segment_count": 0, "text": ""},
    ]


def test_create_job_resets_scenes_artifact(tmp_path: Path) -> None:
    _scenes_job(tmp_path)
    stale = tmp_path / "scenes.json"
    stale.write_text("stale", encoding="utf-8")
    create_job(tmp_path, JobConfig(job_id="scenes-job"))
    assert not stale.exists()


# --- tts stage --------------------------------------------------------------


def _approved_script_job(
    tmp_path: Path, sections: list[dict], *, language: str = "vi"
) -> None:
    """Create a job and write an approved script.json with the given sections."""
    create_job(tmp_path, JobConfig(job_id="tts-job", language=language))
    script = {
        "job_id": "tts-job",
        "language": language,
        "approval_required": True,
        "approved": True,
        "sections": sections,
    }
    (tmp_path / "script.json").write_text(json.dumps(script), encoding="utf-8")


class _FakeCommunicate:
    """Stand-in for edge_tts.Communicate: records inputs, writes stub bytes.

    Mirrors the production call shape edge_tts.Communicate(text, voice).save_sync(path)
    without contacting any external service.
    """

    calls: list[tuple[str, str]] = []

    def __init__(self, text: str, voice: str) -> None:
        type(self).calls.append((text, voice))

    def save_sync(self, output_path: str) -> None:
        Path(output_path).write_bytes(b"ID3-fake-mp3")


def _fake_edge_tts() -> types.SimpleNamespace:
    _FakeCommunicate.calls = []
    return types.SimpleNamespace(Communicate=_FakeCommunicate)


def test_tts_skips_when_script_missing(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="tts-job"))
    from movie_review_factory.pipeline import _tts

    with pytest.raises(SkipStage) as excinfo:
        _tts(tmp_path, load_manifest(tmp_path))
    assert "script" in str(excinfo.value)
    assert not (tmp_path / "voice.json").exists()
    assert not (tmp_path / "narration.mp3").exists()


def test_tts_skips_when_script_not_approved(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="tts-job", language="vi"))
    script = {"approved": False, "sections": [{"title": "Hook", "narration": "Xin chào"}]}
    (tmp_path / "script.json").write_text(json.dumps(script), encoding="utf-8")
    from movie_review_factory.pipeline import _tts

    with pytest.raises(SkipStage) as excinfo:
        _tts(tmp_path, load_manifest(tmp_path))
    assert "approved" in str(excinfo.value)
    assert not (tmp_path / "narration.mp3").exists()


def test_tts_skips_when_narration_empty(tmp_path: Path) -> None:
    _approved_script_job(
        tmp_path,
        [{"title": "Hook", "narration": "   "}, {"title": "Body", "narration": ""}],
    )
    from movie_review_factory.pipeline import _tts

    with pytest.raises(SkipStage) as excinfo:
        _tts(tmp_path, load_manifest(tmp_path))
    assert "narration" in str(excinfo.value)
    assert not (tmp_path / "narration.mp3").exists()


def test_tts_skips_when_edge_tts_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(tmp_path, [{"title": "Hook", "narration": "Xin chào"}])
    # Force `import edge_tts` to raise ImportError even if it happens to be installed.
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    from movie_review_factory.pipeline import _tts

    with pytest.raises(SkipStage) as excinfo:
        _tts(tmp_path, load_manifest(tmp_path))
    assert "edge-tts" in str(excinfo.value)
    assert not (tmp_path / "narration.mp3").exists()


def test_tts_synthesizes_narration_and_writes_voice_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(
        tmp_path,
        [
            {"title": "Mở đầu", "narration": "Xin chào các bạn."},
            {"title": "Trống", "narration": "   "},
            {"title": "Kết", "narration": "Cảm ơn đã xem."},
        ],
    )
    monkeypatch.setitem(sys.modules, "edge_tts", _fake_edge_tts())
    from movie_review_factory.pipeline import _tts

    artifacts, message = _tts(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["voice.json", "narration.mp3"]
    audio = tmp_path / "narration.mp3"
    assert audio.exists()
    assert audio.read_bytes() == b"ID3-fake-mp3"

    # Only non-empty sections, joined in order, are sent to edge-tts.
    assert _FakeCommunicate.calls == [
        ("Xin chào các bạn.\n\nCảm ơn đã xem.", "vi-VN-HoaiMyNeural")
    ]

    voice = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert voice["voice"] == "vi-VN-HoaiMyNeural"
    assert voice["language"] == "vi"
    assert voice["engine"] == "edge-tts"
    assert voice["audio_file"] == "narration.mp3"
    assert voice["section_count"] == 2
    assert [s["title"] for s in voice["sections"]] == ["Mở đầu", "Kết"]
    assert "narration.mp3" in message


def test_tts_failure_cleans_partial_audio_and_keeps_metadata_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(
        tmp_path, [{"title": "Hook", "narration": "Xin chào các bạn."}]
    )

    class FailingCommunicate:
        def __init__(self, text: str, voice: str) -> None:
            self.text = text
            self.voice = voice

        def save_sync(self, output_path: str) -> None:
            Path(output_path).write_bytes(b"partial")
            raise RuntimeError("network interrupted")

    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        types.SimpleNamespace(Communicate=FailingCommunicate),
    )
    import movie_review_factory.pipeline as pipeline

    with pytest.raises(RuntimeError, match="network interrupted"):
        pipeline._tts(tmp_path, load_manifest(tmp_path))

    assert not (tmp_path / "narration.mp3").exists()
    assert not (tmp_path / "narration.synthesizing.mp3").exists()
    assert not (tmp_path / "voice.json").exists()


def test_tts_rejects_empty_synthesized_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(
        tmp_path, [{"title": "Hook", "narration": "Xin chào các bạn."}]
    )

    class EmptyCommunicate:
        def __init__(self, text: str, voice: str) -> None:
            pass

        def save_sync(self, output_path: str) -> None:
            Path(output_path).write_bytes(b"")

    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        types.SimpleNamespace(Communicate=EmptyCommunicate),
    )
    import movie_review_factory.pipeline as pipeline

    with pytest.raises(RuntimeError, match="without producing narration audio"):
        pipeline._tts(tmp_path, load_manifest(tmp_path))

    assert not (tmp_path / "narration.mp3").exists()
    assert not (tmp_path / "narration.synthesizing.mp3").exists()
    assert not (tmp_path / "voice.json").exists()


def test_tts_selects_default_voice_for_unknown_language(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(
        tmp_path, [{"title": "Intro", "narration": "Hello there."}], language="xx"
    )
    monkeypatch.setitem(sys.modules, "edge_tts", _fake_edge_tts())
    from movie_review_factory.pipeline import _tts

    _tts(tmp_path, load_manifest(tmp_path))
    voice = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert voice["voice"] == "en-US-AriaNeural"


def test_tts_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _approved_script_job(tmp_path, [{"title": "Hook", "narration": "Xin chào."}])
    monkeypatch.setitem(sys.modules, "edge_tts", _fake_edge_tts())
    from movie_review_factory.pipeline import _tts

    _tts(tmp_path, load_manifest(tmp_path))
    first = (tmp_path / "voice.json").read_text(encoding="utf-8")
    _tts(tmp_path, load_manifest(tmp_path))
    second = (tmp_path / "voice.json").read_text(encoding="utf-8")
    assert first == second


def test_create_job_resets_tts_artifacts(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="tts-job"))
    (tmp_path / "voice.json").write_text("stale", encoding="utf-8")
    (tmp_path / "narration.mp3").write_bytes(b"stale")
    create_job(tmp_path, JobConfig(job_id="tts-job"))
    assert not (tmp_path / "voice.json").exists()
    assert not (tmp_path / "narration.mp3").exists()


# --- alignment stage --------------------------------------------------------


# A captions.srt where cue 3 runs past the narration bound (5s) so alignment
# must clamp it, and cue 4 starts past the bound so it must be dropped.
_SAMPLE_SRT = (
    "1\n00:00:00,000 --> 00:00:01,000\nXin chào\n\n"
    "2\n00:00:01,000 --> 00:00:02,500\nthế giới\n\n"
    "3\n00:00:04,000 --> 00:00:06,000\nquá dài\n\n"
    "4\n00:00:07,000 --> 00:00:08,000\nngoài biên\n"
)


def _alignment_job(
    tmp_path: Path,
    *,
    srt: str = _SAMPLE_SRT,
    with_voice: bool = True,
    with_audio: bool = True,
    with_captions: bool = True,
) -> None:
    """Create a job with the artifacts alignment consumes (voice/audio/captions)."""
    create_job(tmp_path, JobConfig(job_id="align-job", language="vi"))
    if with_voice:
        voice = {
            "job_id": "align-job",
            "language": "vi",
            "engine": "edge-tts",
            "voice": "vi-VN-HoaiMyNeural",
            "audio_file": "narration.mp3",
        }
        (tmp_path / "voice.json").write_text(json.dumps(voice), encoding="utf-8")
    if with_audio:
        (tmp_path / "narration.mp3").write_bytes(b"ID3-fake-mp3")
    if with_captions:
        (tmp_path / "captions.srt").write_text(srt, encoding="utf-8")


def test_alignment_skips_when_voice_missing(tmp_path: Path) -> None:
    _alignment_job(tmp_path, with_voice=False)
    from movie_review_factory.pipeline import _alignment

    with pytest.raises(SkipStage) as excinfo:
        _alignment(tmp_path, load_manifest(tmp_path))
    assert "voice.json" in str(excinfo.value)
    assert not (tmp_path / "alignment.json").exists()
    assert not (tmp_path / "aligned.srt").exists()


def test_alignment_skips_when_narration_audio_missing(tmp_path: Path) -> None:
    _alignment_job(tmp_path, with_audio=False)
    from movie_review_factory.pipeline import _alignment

    with pytest.raises(SkipStage) as excinfo:
        _alignment(tmp_path, load_manifest(tmp_path))
    assert "narration.mp3" in str(excinfo.value)
    assert not (tmp_path / "alignment.json").exists()
    assert not (tmp_path / "aligned.srt").exists()


def test_alignment_skips_when_captions_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path, with_captions=False)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 5.0)
    with pytest.raises(SkipStage) as excinfo:
        pipeline._alignment(tmp_path, load_manifest(tmp_path))
    assert "captions.srt" in str(excinfo.value)
    assert not (tmp_path / "alignment.json").exists()


def test_alignment_skips_when_ffprobe_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: None)
    with pytest.raises(SkipStage) as excinfo:
        pipeline._alignment(tmp_path, load_manifest(tmp_path))
    assert "ffprobe" in str(excinfo.value)
    assert not (tmp_path / "alignment.json").exists()


def test_alignment_skips_when_duration_not_positive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 0.0)
    with pytest.raises(SkipStage) as excinfo:
        pipeline._alignment(tmp_path, load_manifest(tmp_path))
    assert "duration" in str(excinfo.value)
    assert not (tmp_path / "alignment.json").exists()


def test_alignment_clamps_and_drops_cues_within_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    # Narration is 5s long: cue 3 (4.0-6.0) clamps to 4.0-5.0, cue 4 (7.0-8.0)
    # is entirely out of bounds and is dropped.
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 5.0)

    artifacts, message = pipeline._alignment(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["alignment.json", "aligned.srt"]

    doc = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert doc["job_id"] == "align-job"
    assert doc["audio_file"] == "narration.mp3"
    assert doc["narration_seconds"] == 5.0
    assert doc["source_captions"] == "captions.srt"
    assert doc["cue_count"] == 3
    assert doc["dropped_cues"] == 1
    assert doc["cues"] == [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0, "text": "Xin chào"},
        {"index": 2, "start_seconds": 1.0, "end_seconds": 2.5, "text": "thế giới"},
        {"index": 3, "start_seconds": 4.0, "end_seconds": 5.0, "text": "quá dài"},
    ]

    # Every output cue stays inside the narration bounds.
    for cue in doc["cues"]:
        assert 0.0 <= cue["start_seconds"] <= cue["end_seconds"] <= 5.0

    assert (tmp_path / "aligned.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,000\nXin chào\n\n"
        "2\n00:00:01,000 --> 00:00:02,500\nthế giới\n\n"
        "3\n00:00:04,000 --> 00:00:05,000\nquá dài\n"
    )
    assert message == "aligned 3 cues within narration bounds (dropped 1)"


def test_alignment_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 5.0)

    pipeline._alignment(tmp_path, load_manifest(tmp_path))
    first_json = (tmp_path / "alignment.json").read_text(encoding="utf-8")
    first_srt = (tmp_path / "aligned.srt").read_text(encoding="utf-8")
    pipeline._alignment(tmp_path, load_manifest(tmp_path))
    second_json = (tmp_path / "alignment.json").read_text(encoding="utf-8")
    second_srt = (tmp_path / "aligned.srt").read_text(encoding="utf-8")
    assert first_json == second_json
    assert first_srt == second_srt


def test_create_job_resets_alignment_artifacts(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="align-job"))
    (tmp_path / "alignment.json").write_text("stale", encoding="utf-8")
    (tmp_path / "aligned.srt").write_text("stale", encoding="utf-8")
    create_job(tmp_path, JobConfig(job_id="align-job"))
    assert not (tmp_path / "alignment.json").exists()
    assert not (tmp_path / "aligned.srt").exists()


# --- job_status / CLI status contract ---------------------------------------


def test_job_status_reports_counts_covering_every_stage(job_dir: Path) -> None:
    run_job(job_dir)
    status = job_status(job_dir)
    # The CLI `status` command reads info['job_id'/'complete'/'counts'/'stages'].
    assert {"job_id", "complete", "stages", "counts"} <= set(status)
    counts = status["counts"]
    # Every stage is counted exactly once by its status.
    assert sum(counts.values()) == len(status["stages"])
    # No-video run: 5 text stages ready, the rest skipped.
    assert counts["ready"] == 5
    assert counts["skipped"] == 9


def test_cli_status_command_succeeds(job_dir: Path) -> None:
    from typer.testing import CliRunner

    from movie_review_factory.cli import app

    run_job(job_dir)
    result = CliRunner().invoke(app, ["status", str(job_dir)])
    assert result.exit_code == 0, result.output
    assert "counts:" in result.output


# --- alignment: parser + runner + edge-case coverage ------------------------


def test_parse_srt_skips_blocks_without_valid_timing() -> None:
    from movie_review_factory.pipeline import _parse_srt

    srt = (
        "1\nno timing line here\nstray text\n\n"            # no '-->' -> skipped
        "2\n00:00:01,000 --> not-a-timestamp\nbad end\n\n"  # unparseable end -> skipped
        "3\n00:00:02,000 --> 00:00:03,000\ngood cue\n"      # valid
    )
    assert _parse_srt(srt) == [
        {"start_seconds": 2.0, "end_seconds": 3.0, "text": "good cue"},
    ]


def test_align_cues_drops_inverted_and_zero_length_cues() -> None:
    from movie_review_factory.pipeline import _align_cues

    cues = [
        {"start_seconds": 3.0, "end_seconds": 1.0, "text": "inverted"},  # end < start
        {"start_seconds": 2.0, "end_seconds": 2.0, "text": "zero"},      # zero length
        {"start_seconds": 0.0, "end_seconds": 1.0, "text": "ok"},
    ]
    assert _align_cues(cues, 10.0) == [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 1.0, "text": "ok"},
    ]


def test_alignment_parses_dot_separator_short_ms_and_multiline_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The regex accepts '.' as the ms separator and 1-3 ms digits; multi-line
    # cue text must survive as a single cue with lines joined by newlines.
    srt = (
        "1\n00:00:00.5 --> 00:00:01.5\nline one\ncont\n\n"
        "2\n00:00:02.50 --> 00:00:03.500\nsecond\n"
    )
    _alignment_job(tmp_path, srt=srt)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 10.0)
    pipeline._alignment(tmp_path, load_manifest(tmp_path))

    doc = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert doc["cues"] == [
        {"index": 1, "start_seconds": 0.5, "end_seconds": 1.5, "text": "line one\ncont"},
        {"index": 2, "start_seconds": 2.5, "end_seconds": 3.5, "text": "second"},
    ]


def test_alignment_writes_empty_output_when_all_cues_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Narration (2s) is shorter than every cue's start, so all cues drop.
    srt = (
        "1\n00:00:05,000 --> 00:00:06,000\nlate\n\n"
        "2\n00:00:07,000 --> 00:00:08,000\nlater\n"
    )
    _alignment_job(tmp_path, srt=srt)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 2.0)
    artifacts, message = pipeline._alignment(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["alignment.json", "aligned.srt"]
    doc = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert doc["cue_count"] == 0
    assert doc["dropped_cues"] == 2
    assert doc["cues"] == []
    assert (tmp_path / "aligned.srt").read_text(encoding="utf-8") == ""
    assert message == "aligned 0 cues within narration bounds (dropped 2)"


def test_alignment_prefers_tts_narration_sections_over_source_captions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    voice_path = tmp_path / "voice.json"
    voice = json.loads(voice_path.read_text(encoding="utf-8"))
    voice["sections"] = [
        {"title": "Hook", "narration": "Đây là lời dẫn mở đầu."},
        {"title": "Kết", "narration": "Đây là phần kết của review."},
    ]
    voice_path.write_text(json.dumps(voice), encoding="utf-8")

    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 10.0)
    _, message = pipeline._alignment(tmp_path, load_manifest(tmp_path))

    doc = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert doc["cue_source"] == "voice.json"
    assert doc["source_captions"] == "voice.json"
    assert doc["dropped_cues"] == 0
    assert [cue["text"] for cue in doc["cues"]] == [
        "Đây là lời dẫn mở đầu.",
        "Đây là phần kết của review.",
    ]
    assert doc["cues"][0]["start_seconds"] == 0.0
    assert doc["cues"][-1]["end_seconds"] == 10.0
    assert "Xin chào" not in (tmp_path / "aligned.srt").read_text(encoding="utf-8")
    assert "from voice" in message


def test_alignment_falls_back_to_approved_script_when_transcript_captions_are_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path, srt="")
    (tmp_path / "script.json").write_text(
        json.dumps({
            "approved": True,
            "sections": [
                {"title": "Hook", "narration": "Xin chào các bạn."},
                {"title": "Kết", "narration": "Cảm ơn đã xem."},
            ],
        }),
        encoding="utf-8",
    )
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 10.0)
    _, message = pipeline._alignment(tmp_path, load_manifest(tmp_path))

    doc = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))
    assert doc["source_captions"] == "script.json"
    assert doc["cue_count"] == 2
    assert [cue["text"] for cue in doc["cues"]] == ["Xin chào các bạn.", "Cảm ơn đã xem."]
    assert doc["cues"][0]["start_seconds"] == 0.0
    assert doc["cues"][-1]["end_seconds"] == 10.0
    assert "from script" in message


def test_alignment_stage_completes_through_run_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 5.0)
    manifest = run_job(tmp_path, until="alignment")

    align = manifest.stage("alignment")
    assert align is not None
    assert align.status == "ready"
    assert align.message == "aligned 3 cues within narration bounds (dropped 1)"
    assert [a.name for a in align.artifacts] == ["alignment.json", "aligned.srt"]
    assert all(a.status == "ready" for a in align.artifacts)
    assert (tmp_path / "alignment.json").exists()
    assert (tmp_path / "aligned.srt").exists()

    # Manifest is persisted and the ready stage is resumable (a re-run no-ops it).
    assert load_manifest(tmp_path).stage("alignment").status == "ready"
    rerun = run_job(tmp_path, until="alignment")
    assert rerun.stage("alignment").status == "ready"


# --- render stage -----------------------------------------------------------


def _render_job(
    tmp_path: Path,
    *,
    clips: list[dict] | None = None,
    ratio: str = "16:9",
    with_narration: bool = True,
    with_subtitles: bool = True,
    with_plan: bool = True,
) -> Path:
    source = tmp_path / "owned-source.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="render-job", source_video=source, aspect_ratio=ratio))
    if with_narration:
        (tmp_path / "narration.mp3").write_bytes(b"ID3-fake-mp3")
    if with_subtitles:
        (tmp_path / "aligned.srt").write_text(
            "1\n00:00:00,000 --> 00:00:02,000\nXin chao\n", encoding="utf-8"
        )
    if with_plan:
        plan = {
            "job_id": "render-job",
            "aspect_ratio": ratio,
            "clips": clips if clips is not None else [
                {
                    "section": "Hook",
                    "type": "narration",
                    "source_clip": {"start_seconds": 1.0, "end_seconds": 3.0},
                }
            ],
        }
        (tmp_path / "scene_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    return source


def _render_duration(source: Path, root: Path) -> float:
    return 2.0 if source.name in {"narration.mp3", "final.rendering.mp4"} else 10.0


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"with_narration": False}, "narration.mp3"),
        ({"with_subtitles": False}, "aligned.srt"),
        ({"with_plan": False}, "scene_plan.json"),
    ],
)
def test_render_skips_when_prerequisite_missing(
    tmp_path: Path, kwargs: dict, expected: str
) -> None:
    _render_job(tmp_path, **kwargs)
    from movie_review_factory.pipeline import _render

    with pytest.raises(SkipStage, match=expected):
        _render(tmp_path, load_manifest(tmp_path))
    assert not (tmp_path / "final.mp4").exists()
    assert not (tmp_path / "render.json").exists()


def test_render_skips_unresolved_or_out_of_bounds_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _render_job(
        tmp_path,
        clips=[
            {"section": "Missing", "type": "narration", "source_clip": None},
            {"section": "Late", "type": "narration", "source_clip": {"start_seconds": 9, "end_seconds": 11}},
        ],
    )
    import movie_review_factory.pipeline as pipeline

    calls: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/ffmpeg" if name == "ffmpeg" else "/ffprobe")
    monkeypatch.setattr(pipeline.subprocess, "run", lambda command, **kwargs: calls.append(command))

    with pytest.raises(SkipStage, match="source_clip"):
        pipeline._render(tmp_path, load_manifest(tmp_path))
    assert calls == []
    assert source.exists()


def test_render_skips_when_ffmpeg_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    with pytest.raises(SkipStage, match="ffmpeg"):
        pipeline._render(tmp_path, load_manifest(tmp_path))


@pytest.mark.parametrize(("ratio", "dimensions"), [("16:9", (1920, 1080)), ("9:16", (1080, 1920))])
def test_render_produces_deterministic_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ratio: str, dimensions: tuple[int, int]
) -> None:
    source = _render_job(tmp_path, ratio=ratio)
    import movie_review_factory.pipeline as pipeline

    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        assert kwargs == {"capture_output": True, "text": True, "check": True}
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    artifacts, message = pipeline._render(tmp_path, load_manifest(tmp_path))

    assert [artifact.name for artifact in artifacts] == ["render.json", "final.mp4"]
    assert all(artifact.status == "ready" for artifact in artifacts)
    assert message == "rendered 1 clips to final.mp4"
    assert (tmp_path / "final.mp4").read_bytes() == b"fake-mp4"
    doc = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert (doc["width"], doc["height"]) == dimensions
    assert doc["clips"] == [{
        "index": 0, "section": "Hook", "type": "narration",
        "start_seconds": 1.0, "end_seconds": 3.0,
        "source_seconds": 2.0, "duration_seconds": 2.0,
    }]
    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    assert "trim=start=1.000000:end=3.000000" in filter_complex
    assert "subtitles='" in filter_complex
    assert pipeline._escape_ffmpeg_filter_path(tmp_path / "aligned.srt") in filter_complex
    assert str(source) in commands[0]
    # Source range equals playback length, so no loop filter is introduced.
    assert "loop=loop=" not in filter_complex


def test_render_loops_short_source_to_cover_clip_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(
        tmp_path,
        clips=[{
            "section": "Hook",
            "type": "narration",
            "duration_seconds": 30.0,
            "source_clip": {"start_seconds": 0.0, "end_seconds": 2.0},
        }],
    )
    import movie_review_factory.pipeline as pipeline

    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, load_manifest(tmp_path))

    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    # 2s source at 25 fps = 50 frames; 14 extra loops -> 15 * 2s = 30s, trimmed exact.
    assert "loop=loop=14:size=50:start=0" in filter_complex
    assert "trim=end=30.000000" in filter_complex
    doc = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert doc["clips"][0]["source_seconds"] == 2.0
    assert doc["clips"][0]["duration_seconds"] == 30.0


def test_render_trims_long_source_to_exact_shot_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(
        tmp_path,
        clips=[{
            "section": "Hook",
            "section_index": 1,
            "shot_index": 1,
            "shot_count": 2,
            "type": "narration",
            "duration_seconds": 1.0,
            "source_clip": {"start_seconds": 0.0, "end_seconds": 3.0},
        }],
    )
    import movie_review_factory.pipeline as pipeline

    commands: list[list[str]] = []
    monkeypatch.setattr(
        pipeline,
        "_probe_duration_seconds",
        lambda path: _render_duration(path, tmp_path),
    )
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, load_manifest(tmp_path))

    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    assert "trim=start=0.000000:end=3.000000" in filter_complex
    assert "trim=end=1.000000" in filter_complex
    assert "loop=loop=" not in filter_complex
    doc = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert doc["clips"][0]["source_seconds"] == 3.0
    assert doc["clips"][0]["duration_seconds"] == 1.0
    assert doc["clips"][0]["shot_index"] == 1
    assert doc["clips"][0]["shot_count"] == 2


def test_render_omits_subtitles_filter_for_empty_aligned_srt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    (tmp_path / "aligned.srt").write_text("", encoding="utf-8")
    pipeline._render(tmp_path, load_manifest(tmp_path))
    empty_filter = commands[-1][commands[-1].index("-filter_complex") + 1]
    assert "subtitles=" not in empty_filter

    (tmp_path / "aligned.srt").write_text(
        "1\\n00:00:00,000 --> 00:00:02,000\\nXin chao\\n", encoding="utf-8"
    )
    pipeline._render(tmp_path, load_manifest(tmp_path))
    non_empty_filter = commands[-1][commands[-1].index("-filter_complex") + 1]
    assert "subtitles=" in non_empty_filter


def test_render_failure_marks_stage_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _render_job(tmp_path)
    import subprocess as stdlib_subprocess
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fail(command: list[str], **kwargs: object) -> None:
        raise stdlib_subprocess.CalledProcessError(1, command, stderr="encoder failed")

    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    manifest = load_manifest(tmp_path)
    for stage in manifest.stages:
        if stage.stage != "render":
            stage.status = "ready"
    pipeline.save_manifest(tmp_path, manifest)

    result = run_job(tmp_path, until="render")
    assert result.stage("render").status == "failed"
    assert not (tmp_path / "final.mp4").exists()


def test_create_job_resets_render_artifacts(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="render-job"))
    (tmp_path / "render.json").write_text("stale", encoding="utf-8")
    (tmp_path / "final.mp4").write_bytes(b"stale")
    create_job(tmp_path, JobConfig(job_id="render-job"))
    assert not (tmp_path / "render.json").exists()
    assert not (tmp_path / "final.mp4").exists()


# --- qa stage ---------------------------------------------------------------


def _qa_job(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="qa-job", aspect_ratio="16:9"))
    (tmp_path / "final.mp4").write_bytes(b"fake-mp4")
    (tmp_path / "render.json").write_text(
        json.dumps({
            "job_id": "qa-job",
            "width": 1920,
            "height": 1080,
            "frame_rate": 25,
            "narration_duration_seconds": 30.0,
        }),
        encoding="utf-8",
    )


def _qa_probe_output(
    *,
    video_codec: str = "h264",
    audio_codec: str = "aac",
    width: int = 1920,
    height: int = 1080,
    r_frame_rate: str = "25/1",
    duration: str = "30.100000",
) -> str:
    return json.dumps({
        "format": {"duration": duration},
        "streams": [
            {"codec_type": "video", "codec_name": video_codec, "width": width, "height": height, "r_frame_rate": r_frame_rate},
            {"codec_type": "audio", "codec_name": audio_codec},
        ],
    })


def test_qa_skips_when_final_mp4_missing(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="qa-job"))
    from movie_review_factory.pipeline import _qa

    with pytest.raises(SkipStage, match="final.mp4"):
        _qa(tmp_path, load_manifest(tmp_path))
    assert not (tmp_path / "qa.json").exists()


def test_qa_skips_when_ffprobe_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _qa_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    with pytest.raises(SkipStage, match="ffprobe"):
        pipeline._qa(tmp_path, load_manifest(tmp_path))
    assert not (tmp_path / "qa.json").exists()


def test_qa_report_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _qa_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        return types.SimpleNamespace(returncode=0, stdout=_qa_probe_output(), stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    artifacts, message = pipeline._qa(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["qa.json"]
    assert "passed" in message
    doc = json.loads((tmp_path / "qa.json").read_text(encoding="utf-8"))
    assert doc["passed"] is True
    check_names = [c["check"] for c in doc["checks"]]
    assert "video_codec" in check_names
    assert "audio_codec" in check_names
    assert "canvas_dimensions" in check_names
    assert "frame_rate" in check_names
    assert "positive_duration" in check_names
    assert "duration_drift" in check_names
    assert all(c["passed"] for c in doc["checks"])


def test_qa_fails_and_records_wrong_codec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _qa_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        return types.SimpleNamespace(returncode=0, stdout=_qa_probe_output(video_codec="hevc"), stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="h264"):
        pipeline._qa(tmp_path, load_manifest(tmp_path))

    doc = json.loads((tmp_path / "qa.json").read_text(encoding="utf-8"))
    assert doc["passed"] is False
    codec_check = next(c for c in doc["checks"] if c["check"] == "video_codec")
    assert codec_check["passed"] is False
    assert "hevc" in codec_check["message"]


# --- scene_plan: _assign_source_clips + auto source-clip population ---------


def _make_scenes_doc(scenes: list[dict], duration: float) -> dict:
    return {
        "job_id": "test",
        "source_video": "owned.mp4",
        "duration_seconds": duration,
        "scene_count": len(scenes),
        "scenes": scenes,
    }


def _make_sections(*budgets: float) -> list[dict]:
    """Return script section dicts with the given duration_seconds budgets."""
    return [
        {
            "title": f"Section {i}",
            "budget_minutes": b / 60,
            "narration": "",
            "duration_seconds": b,
        }
        for i, b in enumerate(budgets, start=1)
    ]


def test_assign_source_clips_returns_nones_when_scenes_absent() -> None:
    from movie_review_factory.pipeline import _assign_source_clips

    sections = _make_sections(30.0, 60.0)
    assert _assign_source_clips(sections, {}) == [None, None]
    assert _assign_source_clips(
        sections, {"scenes": [], "duration_seconds": 10.0}
    ) == [None, None]


def test_assign_source_clips_returns_nones_when_video_duration_zero() -> None:
    from movie_review_factory.pipeline import _assign_source_clips

    scenes = [{"index": 1, "start_seconds": 0.0, "end_seconds": 2.0,
               "segment_count": 0, "text": ""}]
    assert _assign_source_clips(
        _make_sections(30.0), {"scenes": scenes, "duration_seconds": 0.0}
    ) == [None]


def test_assign_source_clips_single_section_covers_all_scenes() -> None:
    from movie_review_factory.pipeline import _assign_source_clips

    scenes = [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 5.0,
         "segment_count": 1, "text": "A"},
        {"index": 2, "start_seconds": 5.0, "end_seconds": 10.0,
         "segment_count": 1, "text": "B"},
    ]
    result = _assign_source_clips(_make_sections(60.0), _make_scenes_doc(scenes, 10.0))
    assert result == [{"start_seconds": 0.0, "end_seconds": 10.0}]


def test_assign_source_clips_proportional_two_equal_sections_four_scenes() -> None:
    """Two equal-duration sections -> each gets exactly half the scenes."""
    from movie_review_factory.pipeline import _assign_source_clips

    scenes = [
        {"index": 1, "start_seconds": 0.0,  "end_seconds": 5.0,  "segment_count": 1, "text": "A"},
        {"index": 2, "start_seconds": 5.0,  "end_seconds": 10.0, "segment_count": 1, "text": "B"},
        {"index": 3, "start_seconds": 10.0, "end_seconds": 15.0, "segment_count": 1, "text": "C"},
        {"index": 4, "start_seconds": 15.0, "end_seconds": 20.0, "segment_count": 1, "text": "D"},
    ]
    doc = _make_scenes_doc(scenes, 20.0)
    result = _assign_source_clips(_make_sections(30.0, 30.0), doc)
    assert result[0] == {"start_seconds": 0.0, "end_seconds": 10.0}
    assert result[1] == {"start_seconds": 10.0, "end_seconds": 20.0}


def test_assign_source_clips_more_sections_than_scenes_all_get_clips() -> None:
    """When N sections > M scenes every section still gets a non-None clip."""
    from movie_review_factory.pipeline import _assign_source_clips

    scenes = [{"index": 1, "start_seconds": 0.0, "end_seconds": 2.0,
               "segment_count": 0, "text": ""}]
    doc = _make_scenes_doc(scenes, 2.0)
    result = _assign_source_clips(_make_sections(30.0, 30.0, 30.0), doc)
    assert len(result) == 3
    assert all(c == {"start_seconds": 0.0, "end_seconds": 2.0} for c in result)


def test_assign_source_clips_zero_duration_sections_get_nearest_scene() -> None:
    """All-zero section budgets fall back to equal weights; clips are non-None."""
    from movie_review_factory.pipeline import _assign_source_clips

    scenes = [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 5.0,
         "segment_count": 1, "text": "A"},
        {"index": 2, "start_seconds": 5.0, "end_seconds": 10.0,
         "segment_count": 1, "text": "B"},
    ]
    doc = _make_scenes_doc(scenes, 10.0)
    result = _assign_source_clips(_make_sections(0.0, 0.0), doc)
    assert all(c is not None for c in result)
    for c in result:
        assert 0.0 <= c["start_seconds"] < c["end_seconds"] <= 10.0


def test_assign_source_clips_gap_between_scenes_snaps_to_nearest() -> None:
    """A section whose video range falls entirely in a gap gets the nearest scene."""
    from movie_review_factory.pipeline import _assign_source_clips

    scenes = [
        {"index": 1, "start_seconds": 0.0,  "end_seconds": 5.0,
         "segment_count": 1, "text": "A"},
        {"index": 2, "start_seconds": 15.0, "end_seconds": 20.0,
         "segment_count": 1, "text": "B"},
    ]
    doc = _make_scenes_doc(scenes, 20.0)
    result = _assign_source_clips(_make_sections(20.0, 20.0, 20.0), doc)
    assert all(c is not None for c in result)
    assert result[1] in (
        {"start_seconds": 0.0, "end_seconds": 5.0},
        {"start_seconds": 15.0, "end_seconds": 20.0},
    )


def test_assign_source_shots_spreads_long_section_across_distinct_scenes() -> None:
    from movie_review_factory.pipeline import _assign_source_shots

    scenes = [
        {
            "index": index + 1,
            "start_seconds": float(index * 4),
            "end_seconds": float((index + 1) * 4),
            "segment_count": 1,
            "text": str(index + 1),
        }
        for index in range(6)
    ]
    sections = _make_sections(24.0)
    shots = _assign_source_shots(sections, _make_scenes_doc(scenes, 24.0))

    assert len(shots) == 1
    assert [shot[0] for shot in shots[0]] == [
        {"start_seconds": 0.0, "end_seconds": 4.0},
        {"start_seconds": 8.0, "end_seconds": 12.0},
        {"start_seconds": 20.0, "end_seconds": 24.0},
    ]


def test_scene_plan_auto_populates_source_clip_when_scenes_present(
    tmp_path: Path,
) -> None:
    """scene_plan stage fills source_clip from scenes.json without manual edits."""
    cfg = JobConfig(job_id="sp-auto", language="vi", target_minutes=2)
    create_job(tmp_path, cfg)
    run_job(tmp_path, until="script")

    scenes_doc = {
        "job_id": "sp-auto",
        "source_video": "owned.mp4",
        "duration_seconds": 10.0,
        "scene_count": 2,
        "scenes": [
            {"index": 1, "start_seconds": 0.0, "end_seconds": 5.0,
             "segment_count": 1, "text": "A"},
            {"index": 2, "start_seconds": 5.0, "end_seconds": 10.0,
             "segment_count": 1, "text": "B"},
        ],
    }
    (tmp_path / "scenes.json").write_text(json.dumps(scenes_doc), encoding="utf-8")

    from movie_review_factory.pipeline import _scene_plan

    artifacts, message = _scene_plan(tmp_path, load_manifest(tmp_path))
    assert [a.name for a in artifacts] == ["scene_plan.json"]

    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    assert all(c["source_clip"] is not None for c in plan["clips"])
    for clip in plan["clips"]:
        sc = clip["source_clip"]
        assert 0.0 <= sc["start_seconds"] < sc["end_seconds"] <= 10.0
    assert "resolved" in message


def test_scene_plan_source_clip_is_none_when_scenes_absent(tmp_path: Path) -> None:
    """Without scenes.json source_clip stays None (graceful degradation)."""
    cfg = JobConfig(job_id="sp-none", language="vi", target_minutes=2)
    create_job(tmp_path, cfg)
    run_job(tmp_path, until="script")

    from movie_review_factory.pipeline import _scene_plan

    _scene_plan(tmp_path, load_manifest(tmp_path))
    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    assert all(c["source_clip"] is None for c in plan["clips"])


def test_scene_plan_is_deterministic_with_scenes(tmp_path: Path) -> None:
    """Running scene_plan twice with the same scenes.json produces identical output."""
    cfg = JobConfig(job_id="sp-det", language="vi", target_minutes=2)
    create_job(tmp_path, cfg)
    run_job(tmp_path, until="script")

    scenes_doc = {
        "job_id": "sp-det",
        "source_video": "owned.mp4",
        "duration_seconds": 8.0,
        "scene_count": 2,
        "scenes": [
            {"index": 1, "start_seconds": 0.0, "end_seconds": 4.0,
             "segment_count": 1, "text": "A"},
            {"index": 2, "start_seconds": 4.0, "end_seconds": 8.0,
             "segment_count": 1, "text": "B"},
        ],
    }
    (tmp_path / "scenes.json").write_text(json.dumps(scenes_doc), encoding="utf-8")

    from movie_review_factory.pipeline import _scene_plan

    _scene_plan(tmp_path, load_manifest(tmp_path))
    first = (tmp_path / "scene_plan.json").read_text(encoding="utf-8")
    _scene_plan(tmp_path, load_manifest(tmp_path))
    second = (tmp_path / "scene_plan.json").read_text(encoding="utf-8")
    assert first == second


def test_claude_scene_plan_uses_only_valid_indexed_scene_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    create_job(tmp_path, JobConfig(job_id="scene-agent", content_agent="claude"))
    (tmp_path / "script.json").write_text(json.dumps({
        "sections": [
            {"title": "Hook", "duration_seconds": 30, "narration": "A"},
            {"title": "Analysis", "duration_seconds": 45, "narration": "B"},
        ]
    }), encoding="utf-8")
    (tmp_path / "scenes.json").write_text(json.dumps({
        "duration_seconds": 40.0,
        "scenes": [
            {"index": 1, "start_seconds": 0.0, "end_seconds": 10.0, "text": "one"},
            {"index": 2, "start_seconds": 10.0, "end_seconds": 20.0, "text": "two"},
            {"index": 3, "start_seconds": 20.0, "end_seconds": 30.0, "text": "three"},
            {"index": 4, "start_seconds": 30.0, "end_seconds": 40.0, "text": "four"},
        ],
    }), encoding="utf-8")

    def fake_agent(**kwargs: object) -> dict:
        assert kwargs["stage"] == "scene_plan"
        return {
            "assignments": [
                {"shots": [
                    {"start_scene_index": 1, "end_scene_index": 1, "rationale": "setup-a"},
                    {"start_scene_index": 2, "end_scene_index": 2, "rationale": "setup-b"},
                ]},
                {"shots": [
                    {"start_scene_index": 3, "end_scene_index": 3, "rationale": "payoff-a"},
                    {"start_scene_index": 4, "end_scene_index": 4, "rationale": "payoff-b"},
                ]},
            ],
            "notes": "chosen from indexed scenes",
        }

    monkeypatch.setattr(pipeline_mod, "_run_reasoning_agent", fake_agent)
    artifacts, message = pipeline_mod._scene_plan(tmp_path, load_manifest(tmp_path))

    assert [artifact.name for artifact in artifacts] == ["scene_plan.json"]
    assert message.startswith("scene_plan generated by Claude")
    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    assert plan["generator"] == "claude"
    assert len(plan["clips"]) == 4
    assert [clip["source_clip"] for clip in plan["clips"]] == [
        {"start_seconds": 0.0, "end_seconds": 10.0},
        {"start_seconds": 10.0, "end_seconds": 20.0},
        {"start_seconds": 20.0, "end_seconds": 30.0},
        {"start_seconds": 30.0, "end_seconds": 40.0},
    ]
    assert [clip["duration_seconds"] for clip in plan["clips"]] == [15.0, 15.0, 22.5, 22.5]
    assert [clip["shot_index"] for clip in plan["clips"]] == [1, 2, 1, 2]
    assert [clip["notes"] for clip in plan["clips"]] == [
        "setup-a", "setup-b", "payoff-a", "payoff-b"
    ]


def test_claude_scene_plan_rejects_unknown_scene_index(tmp_path: Path) -> None:
    from movie_review_factory.pipeline import _agent_scene_assignments

    sections = [{"title": "A"}]
    scenes_doc = {
        "duration_seconds": 10.0,
        "scenes": [{"index": 1, "start_seconds": 0.0, "end_seconds": 10.0}],
    }
    with pytest.raises(ValueError, match="unknown scene index"):
        _agent_scene_assignments(
            sections,
            scenes_doc,
            [{"shots": [
                {"start_scene_index": 1, "end_scene_index": 99, "rationale": "bad"}
            ]}],
        )


def test_scene_plan_source_clips_bounded_to_video_duration(
    tmp_path: Path,
) -> None:
    """source_clip timestamps must never exceed scenes.json duration_seconds."""
    cfg = JobConfig(job_id="sp-bounds", language="vi", target_minutes=5)
    create_job(tmp_path, cfg)
    run_job(tmp_path, until="script")

    VIDEO_DURATION = 7.5
    scenes_doc = {
        "job_id": "sp-bounds",
        "source_video": "owned.mp4",
        "duration_seconds": VIDEO_DURATION,
        "scene_count": 3,
        "scenes": [
            {"index": 1, "start_seconds": 0.0, "end_seconds": 2.5,
             "segment_count": 1, "text": "A"},
            {"index": 2, "start_seconds": 2.5, "end_seconds": 5.0,
             "segment_count": 1, "text": "B"},
            {"index": 3, "start_seconds": 5.0, "end_seconds": 7.5,
             "segment_count": 1, "text": "C"},
        ],
    }
    (tmp_path / "scenes.json").write_text(json.dumps(scenes_doc), encoding="utf-8")

    from movie_review_factory.pipeline import _scene_plan

    _scene_plan(tmp_path, load_manifest(tmp_path))
    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    for clip in plan["clips"]:
        sc = clip["source_clip"]
        assert sc is not None
        assert sc["start_seconds"] >= 0.0
        assert sc["end_seconds"] <= VIDEO_DURATION
        assert sc["start_seconds"] < sc["end_seconds"]


def test_thumbnail_stage_generates_three_candidates_and_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    create_job(tmp_path, JobConfig(job_id="thumb", source_video=source))
    (tmp_path / "scene_plan.json").write_text(
        json.dumps({
            "clips": [
                {"source_clip": {"start_seconds": 0.0, "end_seconds": 20.0}},
                {"source_clip": {"start_seconds": 20.0, "end_seconds": 40.0}},
                {"source_clip": {"start_seconds": 40.0, "end_seconds": 60.0}},
                {"source_clip": {"start_seconds": 60.0, "end_seconds": 80.0}},
            ]
        }),
        encoding="utf-8",
    )
    (tmp_path / "youtube_metadata.json").write_text(
        json.dumps({"title": "Recap test"}), encoding="utf-8"
    )

    monkeypatch.setattr(pipeline_mod.shutil, "which", lambda name: name)
    monkeypatch.setattr(pipeline_mod, "_probe_duration_seconds", lambda _: 100.0)

    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> object:
        Path(command[-1]).write_bytes(b"jpeg")
        calls.append(command)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(pipeline_mod.subprocess, "run", fake_run)

    artifacts, message = pipeline_mod._thumbnail(tmp_path, load_manifest(tmp_path))

    assert message == "generated 3 thumbnail candidates"
    assert len(calls) == 3
    assert [Path(call[-1]).name for call in calls] == [
        "thumbnail-1.jpg", "thumbnail-2.jpg", "thumbnail-3.jpg"
    ]
    assert all("scale=1280:720" in call[call.index("-vf") + 1] for call in calls)
    for name in (
        "thumbnails.json", "thumbnail.jpg",
        "thumbnail-1.jpg", "thumbnail-2.jpg", "thumbnail-3.jpg",
    ):
        assert (tmp_path / name).exists()

    data = json.loads((tmp_path / "thumbnails.json").read_text(encoding="utf-8"))
    assert data["candidate_count"] == 3
    assert data["primary_candidate"] == "thumbnail-2.jpg"
    assert data["primary_thumbnail"] == "thumbnail.jpg"
    assert data["title_hint"] == "Recap test"
    assert [item["source_seconds"] for item in data["candidates"]] == [30.0, 50.0, 70.0]
    assert [artifact.name for artifact in artifacts] == [
        "thumbnails.json", "thumbnail-1.jpg", "thumbnail-2.jpg",
        "thumbnail-3.jpg", "thumbnail.jpg",
    ]


def test_select_thumbnail_replaces_primary_and_invalidates_publish(tmp_path: Path) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    create_job(tmp_path, JobConfig(job_id="select-thumb"))
    candidates = []
    for index in range(1, 4):
        name = f"thumbnail-{index}.jpg"
        (tmp_path / name).write_bytes(f"candidate-{index}".encode())
        candidates.append({
            "index": index,
            "file": name,
            "source_seconds": float(index * 10),
            "width": 1280,
            "height": 720,
        })
    (tmp_path / "thumbnail.jpg").write_bytes(b"candidate-2")
    (tmp_path / "thumbnails.json").write_text(
        json.dumps({
            "job_id": "select-thumb",
            "candidates": candidates,
            "primary_thumbnail": "thumbnail.jpg",
            "primary_candidate": "thumbnail-2.jpg",
        }),
        encoding="utf-8",
    )
    (tmp_path / "publish_record.json").write_text("{}", encoding="utf-8")

    manifest = load_manifest(tmp_path)
    manifest.stage("thumbnail").mark("ready", "generated 3 thumbnail candidates")
    manifest.stage("publish").mark("ready", "publish record written")
    pipeline_mod.save_manifest(tmp_path, manifest)

    result = pipeline_mod.select_thumbnail(tmp_path, "thumbnail-3.jpg")

    assert result["primary_candidate"] == "thumbnail-3.jpg"
    assert (tmp_path / "thumbnail.jpg").read_bytes() == b"candidate-3"
    persisted = json.loads((tmp_path / "thumbnails.json").read_text(encoding="utf-8"))
    assert persisted["primary_candidate"] == "thumbnail-3.jpg"
    refreshed = load_manifest(tmp_path)
    assert refreshed.stage("thumbnail").status == "ready"
    assert refreshed.stage("publish").status == "pending"
    assert not (tmp_path / "publish_record.json").exists()


def test_select_thumbnail_rejects_unknown_candidate_without_mutation(tmp_path: Path) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    create_job(tmp_path, JobConfig(job_id="select-invalid"))
    (tmp_path / "thumbnail-1.jpg").write_bytes(b"one")
    (tmp_path / "thumbnail.jpg").write_bytes(b"one")
    thumbnails = {
        "job_id": "select-invalid",
        "candidates": [{"index": 1, "file": "thumbnail-1.jpg"}],
        "primary_thumbnail": "thumbnail.jpg",
        "primary_candidate": "thumbnail-1.jpg",
    }
    (tmp_path / "thumbnails.json").write_text(json.dumps(thumbnails), encoding="utf-8")
    (tmp_path / "publish_record.json").write_text("{}", encoding="utf-8")
    manifest = load_manifest(tmp_path)
    manifest.stage("thumbnail").mark("ready")
    manifest.stage("publish").mark("ready")
    pipeline_mod.save_manifest(tmp_path, manifest)

    with pytest.raises(ValueError, match="not part"):
        pipeline_mod.select_thumbnail(tmp_path, "thumbnail-99.jpg")

    assert (tmp_path / "thumbnail.jpg").read_bytes() == b"one"
    assert json.loads((tmp_path / "thumbnails.json").read_text(encoding="utf-8")) == thumbnails
    assert (tmp_path / "publish_record.json").exists()
    assert load_manifest(tmp_path).stage("publish").status == "ready"


def test_skipped_thumbnail_is_retried_on_next_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    create_job(tmp_path, JobConfig(job_id="retry-thumb"))
    first = run_job(tmp_path, until="thumbnail")
    assert first.stage("thumbnail").status == "skipped"

    def ready_thumbnail(root: Path, manifest: object) -> tuple[list, str]:
        output = root / "thumbnail.jpg"
        output.write_bytes(b"jpeg")
        return [pipeline_mod.Artifact(name=output.name, path=output, status="ready")], "ready"

    monkeypatch.setitem(pipeline_mod.STAGE_HANDLERS, "thumbnail", ready_thumbnail)
    second = run_job(tmp_path, until="thumbnail")
    assert second.stage("thumbnail").status == "ready"
    assert (tmp_path / "thumbnail.jpg").exists()


def test_load_manifest_upgrades_pre_thumbnail_jobs(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="legacy"))
    raw = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    raw["stages"] = [stage for stage in raw["stages"] if stage["stage"] != "thumbnail"]
    (tmp_path / "manifest.json").write_text(json.dumps(raw), encoding="utf-8")

    upgraded = load_manifest(tmp_path)
    assert [stage.stage for stage in upgraded.stages] == list(__import__(
        "movie_review_factory.pipeline", fromlist=["STAGES"]
    ).STAGES)
    assert upgraded.stage("thumbnail").status == "pending"
    assert validate_job(tmp_path) == []

def test_claude_content_agent_drives_research_outline_and_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    calls: list[str] = []

    def fake_agent(**kwargs: object) -> dict:
        stage = str(kwargs["stage"])
        calls.append(stage)
        if stage == "research":
            return {
                "brief": "Bản nghiên cứu có nguồn.",
                "facts": ["Nhân vật chính đối mặt một lựa chọn quan trọng."],
                "sources": [
                    {
                        "title": "Reference",
                        "url": "https://example.test/movie",
                        "note": "metadata reference",
                    }
                ],
                "uncertainties": [],
            }
        if stage == "outline":
            return {
                "sections": [
                    {"title": "Hook", "budget_minutes": 1, "purpose": "Mở vấn đề"},
                    {"title": "Diễn biến", "budget_minutes": 2, "purpose": "Tóm tắt"},
                    {"title": "Phân tích", "budget_minutes": 1, "purpose": "Bình luận"},
                ],
                "notes": "agent outline",
            }
        if stage == "script":
            return {
                "sections": [
                    {"title": "Hook", "narration": "Mở đầu có nội dung thật."},
                    {"title": "Diễn biến", "narration": "Phần diễn biến có nội dung thật."},
                    {"title": "Phân tích", "narration": "Phần phân tích có nội dung thật."},
                ],
                "notes": "agent script",
            }
        raise AssertionError(stage)

    monkeypatch.setattr(pipeline_mod, "run_claude_json", fake_agent)

    create_job(
        tmp_path,
        JobConfig(
            job_id="claude-job",
            language="vi",
            target_minutes=5,
            movie_title="Example Movie",
            content_agent="claude",
        ),
    )
    manifest = run_job(tmp_path, until="script")

    assert calls == ["research", "outline", "script"]
    assert manifest.stage("research").status == "ready"
    assert manifest.stage("outline").status == "ready"
    assert manifest.stage("script").status == "ready"

    research = json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))
    assert research["movie_title"] == "Example Movie"
    assert research["generator"] == "claude"
    assert research["status"] == "ready"

    outline = json.loads((tmp_path / "outline.json").read_text(encoding="utf-8"))
    assert outline["generator"] == "claude"
    assert sum(section["budget_minutes"] for section in outline["sections"]) == 5.0

    script = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    assert script["generator"] == "claude"
    assert script["approved"] is False
    assert [section["title"] for section in script["sections"]] == [
        section["title"] for section in outline["sections"]
    ]
    assert all(section["narration"].strip() for section in script["sections"])
    assert "**approved: false**" in (tmp_path / "script.md").read_text(encoding="utf-8")


def test_scaffold_mode_never_calls_claude(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    def forbidden(**_: object) -> dict:
        raise AssertionError("Claude must not run for scaffold jobs")

    monkeypatch.setattr(pipeline_mod, "run_claude_json", forbidden)
    create_job(tmp_path, JobConfig(job_id="offline", content_agent="scaffold"))
    run_job(tmp_path, until="script")

    assert json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))[
        "generator"
    ] == "scaffold"
    assert json.loads((tmp_path / "outline.json").read_text(encoding="utf-8"))[
        "generator"
    ] == "scaffold"
    assert json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))[
        "generator"
    ] == "scaffold"


def test_claude_failure_does_not_fall_back_to_scaffold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod
    from movie_review_factory.content_agent import ContentAgentError

    def failing_agent(**_: object) -> dict:
        raise ContentAgentError("backend unavailable")

    monkeypatch.setattr(pipeline_mod, "run_claude_json", failing_agent)
    create_job(
        tmp_path,
        JobConfig(job_id="agent-fail", content_agent="claude"),
    )
    manifest = run_job(tmp_path, until="research")

    assert manifest.stage("research").status == "failed"
    assert "backend unavailable" in manifest.stage("research").message
    assert not (tmp_path / "research.json").exists()


def test_claude_script_section_mismatch_marks_stage_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    def fake_agent(**kwargs: object) -> dict:
        stage = str(kwargs["stage"])
        if stage == "research":
            return {"brief": "x", "facts": [], "sources": [], "uncertainties": []}
        if stage == "outline":
            return {
                "sections": [
                    {"title": "A", "budget_minutes": 1, "purpose": "a"},
                    {"title": "B", "budget_minutes": 1, "purpose": "b"},
                    {"title": "C", "budget_minutes": 1, "purpose": "c"},
                ],
                "notes": "",
            }
        return {
            "sections": [{"title": "A", "narration": "only one"}],
            "notes": "",
        }

    monkeypatch.setattr(pipeline_mod, "run_claude_json", fake_agent)
    create_job(
        tmp_path,
        JobConfig(job_id="bad-claude", target_minutes=3, content_agent="claude"),
    )
    manifest = run_job(tmp_path, until="script")

    assert manifest.stage("script").status == "failed"
    assert "section count" in manifest.stage("script").message
