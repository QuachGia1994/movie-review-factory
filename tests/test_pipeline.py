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

    # media stages skipped honestly (incl. the optional watermark stage)
    for stage in ("watermark", "transcript", "scenes", "tts", "alignment", "render", "qa"):
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


@pytest.mark.parametrize("batch_size", [1, 8])
def test_transcript_writes_timed_segments_and_srt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, batch_size: int) -> None:
    source = tmp_path / "owned-sample.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="transcript-job", source_video=source))

    class FakeModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str, cpu_threads: int) -> None:
            assert model_name == "small"
            assert device == "cpu"
            assert compute_type == "int8"
            assert cpu_threads == 8

        def transcribe(self, source_path: str, **kwargs: object) -> tuple[list[object], object]:
            assert source_path == str(source)
            assert kwargs == {"language": None, "vad_filter": True, "word_timestamps": True}
            return [
                types.SimpleNamespace(start=0.0, end=1.25, text=" Xin chào "),
                types.SimpleNamespace(start=1.25, end=2.5, text="thế giới"),
            ], types.SimpleNamespace(language="vi")

    class FakeBatched:
        def __init__(self, model: FakeModel) -> None:
            self.model = model

        def transcribe(self, source_path: str, **kwargs: object) -> tuple[list[object], object]:
            assert kwargs.pop("batch_size") == 8
            return self.model.transcribe(source_path, **kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(
        WhisperModel=FakeModel, BatchedInferencePipeline=FakeBatched,
    ))
    monkeypatch.setenv("MRF_WHISPER_CPU_THREADS", "8")
    monkeypatch.setenv("MRF_WHISPER_BATCH_SIZE", str(batch_size))
    # Pin the runtime so the assertion holds regardless of the test box's GPU, and
    # stub the WAV pre-extract so the unit test stays hermetic (no real ffmpeg).
    monkeypatch.setenv("MRF_WHISPER_DEVICE", "cpu")
    monkeypatch.setenv("MRF_WHISPER_COMPUTE_TYPE", "int8")
    monkeypatch.setattr(
        "movie_review_factory.pipeline._prepare_whisper_audio", lambda root, src: src
    )
    from movie_review_factory.pipeline import _transcript

    artifacts, message = _transcript(tmp_path, load_manifest(tmp_path))

    assert [artifact.name for artifact in artifacts] == ["transcript.json", "captions.srt"]
    assert message == "transcribed 2 segments"
    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["source_video"] == str(source)
    assert transcript["language"] == "vi"
    assert transcript["output_language"] == "vi"
    assert transcript["segments"] == [
        {"start_seconds": 0.0, "end_seconds": 1.25, "text": "Xin chào", "words": []},
        {"start_seconds": 1.25, "end_seconds": 2.5, "text": "thế giới", "words": []},
    ]
    assert (tmp_path / "captions.srt").read_text(encoding="utf-8") == (
        "1\n00:00:00,000 --> 00:00:01,250\nXin chào\n\n"
        "2\n00:00:01,250 --> 00:00:02,500\nthế giới\n"
    )


def test_transcript_splits_long_segment_into_word_anchored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coarse multi-sentence VAD block is cut into sentence-sized rows whose timing
    comes from the spoken words, so a mid-block pause never drifts a later caption
    ahead of the audio (the Khám phá tab bug)."""
    source = tmp_path / "owned-sample.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="split-job", source_video=source))

    def _w(word: str, start: float, end: float) -> object:
        return types.SimpleNamespace(word=word, start=start, end=end)

    # One 15.0–27.0s VAD block holding two utterances with a ~7s pause between them.
    long_segment = types.SimpleNamespace(
        start=15.0, end=27.0,
        text="Dạ em tới lấy hàng đi giao á chị ơi. Ê dạ dạ.",
        words=[
            _w(" Dạ", 15.0, 15.3), _w(" em", 15.3, 15.5), _w(" tới", 15.5, 15.8),
            _w(" lấy", 15.8, 16.1), _w(" hàng", 16.1, 16.5), _w(" đi", 16.5, 16.7),
            _w(" giao", 16.7, 17.0), _w(" á", 17.0, 17.2), _w(" chị", 17.2, 17.5),
            _w(" ơi.", 17.5, 19.0),
            _w(" Ê", 26.0, 26.3), _w(" dạ", 26.3, 26.6), _w(" dạ.", 26.6, 27.0),
        ],
    )

    class FakeModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def transcribe(self, source_path: str, **kwargs: object) -> tuple[list[object], object]:
            assert kwargs.get("word_timestamps") is True
            return [long_segment], types.SimpleNamespace(language="vi")

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(
        WhisperModel=FakeModel, BatchedInferencePipeline=lambda model: model,
    ))
    monkeypatch.setenv("MRF_WHISPER_BATCH_SIZE", "1")
    from movie_review_factory.pipeline import _transcript

    _, message = _transcript(tmp_path, load_manifest(tmp_path))
    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    seg = transcript["segments"]
    assert [(s["start_seconds"], s["end_seconds"], s["text"]) for s in seg] == [
        (15.0, 19.0, "Dạ em tới lấy hàng đi giao á chị ơi."),
        (26.0, 27.0, "Ê dạ dạ."),
    ]
    assert message == "transcribed 2 segments"
    # Each row now carries its own word timings (Lượt 4); the frontend times captions
    # from these instead of interpolating by character count.
    assert seg[0]["words"][0] == {"word": "Dạ", "start": 15.0, "end": 15.3}
    assert seg[0]["words"][-1] == {"word": "ơi.", "start": 17.5, "end": 19.0}
    assert [w["word"] for w in seg[1]["words"]] == ["Ê", "dạ", "dạ."]
    # The second utterance is anchored to its real spoken words at 26.0–27.0s, not
    # interpolated linearly across the 12s block (which would place it near ~22s).
    assert seg[1]["words"][0]["start"] == 26.0 and seg[1]["words"][-1]["end"] == 27.0


def test_transcript_skips_when_source_has_no_audio(tmp_path: Path) -> None:
    source = tmp_path / "silent-sample.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="silent-job", source_video=source))
    (tmp_path / "ingest.json").write_text(
        json.dumps({"job_id": "silent-job", "has_audio": False}), encoding="utf-8"
    )

    from movie_review_factory.pipeline import _transcript

    with pytest.raises(SkipStage, match="no audio track"):
        _transcript(tmp_path, load_manifest(tmp_path))
    assert not (tmp_path / "transcript.json").exists()


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


def test_whisper_runtime_auto_selects_cuda_then_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline

    for name in ("MRF_WHISPER_MODEL", "MRF_WHISPER_DEVICE", "MRF_WHISPER_COMPUTE_TYPE"):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(pipeline, "_cuda_available", lambda: True)
    assert pipeline._whisper_runtime() == ("small", "cuda", "float16")

    monkeypatch.setattr(pipeline, "_cuda_available", lambda: False)
    assert pipeline._whisper_runtime() == ("small", "cpu", "int8")


def test_whisper_runtime_honours_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_cuda_available", lambda: False)
    monkeypatch.setenv("MRF_WHISPER_MODEL", "distil-large-v3")
    monkeypatch.setenv("MRF_WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("MRF_WHISPER_COMPUTE_TYPE", "int8_float16")
    assert pipeline._whisper_runtime() == ("distil-large-v3", "cuda", "int8_float16")


def test_prepare_whisper_audio_prefers_wav_and_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import movie_review_factory.pipeline as pipeline

    source = tmp_path / "clip.mp4"
    source.touch()

    def fake_extract(src: Path, dest: Path) -> bool:
        dest.write_bytes(b"RIFFfake")
        return True

    monkeypatch.setattr(pipeline, "_extract_wav_16k_mono", fake_extract)
    prepared = pipeline._prepare_whisper_audio(tmp_path, source)
    assert prepared == tmp_path / ".mrf_whisper_audio.wav"
    assert prepared.exists()

    monkeypatch.setattr(pipeline, "_extract_wav_16k_mono", lambda src, dest: False)
    assert pipeline._prepare_whisper_audio(tmp_path, source) == source


def test_parse_srt_segments_reads_timings_and_strips_tags() -> None:
    import movie_review_factory.pipeline as pipeline

    raw = (
        "1\n00:00:01,000 --> 00:00:02,500\n<i>Hello</i> {\\an8}world\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nsecond line\n"
    )
    assert pipeline._parse_srt_segments(raw) == [
        {"start_seconds": 1.0, "end_seconds": 2.5, "text": "Hello world"},
        {"start_seconds": 3.0, "end_seconds": 4.0, "text": "second line"},
    ]


def test_transcript_falls_back_to_cpu_when_gpu_runtime_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "owned-sample.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="gpu-fallback-job", source_video=source))

    attempted: list[str] = []

    class FakeModel:
        def __init__(self, model_name: str, *, device: str, compute_type: str, cpu_threads: int) -> None:
            attempted.append(device)
            if device == "cuda":
                raise RuntimeError("CUDA driver not found")

        def transcribe(self, source_path: str, **kwargs: object) -> tuple[list[object], object]:
            return (
                [types.SimpleNamespace(start=0.0, end=1.0, text="hi", words=[])],
                types.SimpleNamespace(language="en"),
            )

    monkeypatch.setitem(sys.modules, "faster_whisper", types.SimpleNamespace(
        WhisperModel=FakeModel, BatchedInferencePipeline=lambda model: model,
    ))
    monkeypatch.setenv("MRF_WHISPER_DEVICE", "cuda")
    monkeypatch.setenv("MRF_WHISPER_BATCH_SIZE", "1")
    monkeypatch.setattr(
        "movie_review_factory.pipeline._prepare_whisper_audio", lambda root, src: src
    )
    from movie_review_factory.pipeline import _transcript

    artifacts, message = _transcript(tmp_path, load_manifest(tmp_path))

    assert attempted == ["cuda", "cpu"]
    assert message == "transcribed 1 segments"
    assert [artifact.name for artifact in artifacts] == ["transcript.json", "captions.srt"]


def test_transcript_reuses_embedded_subtitles_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "with-subs.mp4"
    source.touch()
    create_job(tmp_path, JobConfig(job_id="subs-job", language="vi", source_video=source))

    import movie_review_factory.pipeline as pipeline

    def fake_embedded(src: Path, want_langs: list, root: Path):
        assert want_langs == ["vi"]
        return (
            [
                {"start_seconds": 0.0, "end_seconds": 1.0, "text": "Xin ch\u00e0o"},
                {"start_seconds": 1.0, "end_seconds": 2.0, "text": "th\u1ebf gi\u1edbi"},
            ],
            "vi",
        )

    monkeypatch.setattr(pipeline, "_embedded_subtitle_segments", fake_embedded)
    monkeypatch.setenv("MRF_TRANSCRIPT_EMBEDDED_SUBS", "1")
    from movie_review_factory.pipeline import _transcript

    artifacts, message = _transcript(tmp_path, load_manifest(tmp_path))

    assert "embedded subtitle" in message
    transcript = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert transcript["source"] == "embedded-subtitles"
    assert transcript["language"] == "vi"
    assert len(transcript["segments"]) == 2
    assert [artifact.name for artifact in artifacts] == ["transcript.json", "captions.srt"]


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

    # transcript.json as produced by the transcript stage. Segment 4 ends past the true video duration and must be clamped to the probed bound.
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

    assert [a.name for a in artifacts] == ["scenes.json", "media_index.sqlite3"]
    assert message == "indexed 5 scenes from transcript"
    doc = json.loads((tmp_path / "scenes.json").read_text(encoding="utf-8"))
    assert doc["source_video"] == str(source)
    assert doc["duration_seconds"] == 10.0
    assert doc["scene_count"] == 5
    assert doc["scenes"] == [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 2.0, "segment_count": 2, "text": "A B"},
        {"index": 2, "start_seconds": 2.0, "end_seconds": 5.0, "segment_count": 0, "text": ""},
        {"index": 3, "start_seconds": 5.0, "end_seconds": 6.0, "segment_count": 1, "text": "C"},
        {"index": 4, "start_seconds": 6.0, "end_seconds": 9.0, "segment_count": 0, "text": ""},
        {"index": 5, "start_seconds": 9.0, "end_seconds": 10.0, "segment_count": 1, "text": "D"},
    ]
    # Every timestamp stays within the probed video bounds.
    for scene in doc["scenes"]:
        assert 0.0 <= scene["start_seconds"] <= scene["end_seconds"] <= 10.0
    from movie_review_factory.media_store import MediaStore
    with MediaStore(tmp_path / "media_index.sqlite3") as store:
        assert [shot.label for shot in store.list_shots()] == ["A B", "Scene 2", "C", "Scene 4", "D"]
        assert [segment.text for segment in store.search_transcript("B")] == ["B"]


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

    assert [a.name for a in artifacts] == ["scenes.json", "media_index.sqlite3"]
    doc = json.loads((tmp_path / "scenes.json").read_text(encoding="utf-8"))
    assert doc["scene_count"] == 1
    assert doc["scenes"] == [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 12.5, "segment_count": 0, "text": ""},
    ]


def test_long_silent_source_has_bounded_scenes_and_distinct_section_ranges() -> None:
    import movie_review_factory.pipeline as pipeline

    scenes = pipeline._build_scenes([], 95.0)
    assert [(item["start_seconds"], item["end_seconds"]) for item in scenes] == [
        (0.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, 95.0),
    ]
    sections = [{"duration_seconds": 30}, {"duration_seconds": 30}]
    ranges = pipeline._assign_source_clips(
        sections, {"scenes": scenes, "duration_seconds": 95.0},
    )
    assert ranges[0] != ranges[1]
    assert ranges[0]["start_seconds"] < ranges[1]["start_seconds"]


def test_ten_minute_silent_source_assigns_distinct_shots_to_sections() -> None:
    import movie_review_factory.pipeline as pipeline

    scenes = pipeline._build_scenes([], 600.0)
    assert len(scenes) == 20
    sections = [{"duration_seconds": 80} for _ in range(6)]
    assignments = pipeline._assign_source_shots(
        sections, {"scenes": scenes, "duration_seconds": 600.0},
    )
    first = [shot[0]["start_seconds"] for shot in assignments[0]]
    last = [shot[0]["start_seconds"] for shot in assignments[-1]]
    assert first and last
    assert max(first) < min(last)


def test_silent_gap_between_speech_remains_selectable() -> None:
    import movie_review_factory.pipeline as pipeline

    scenes = pipeline._build_scenes([
        {"start_seconds": 0, "end_seconds": 2, "text": "Opening"},
        {"start_seconds": 80, "end_seconds": 82, "text": "Return"},
    ], 95.0)
    assert any(item["segment_count"] == 0 and item["start_seconds"] >= 2
               and item["end_seconds"] <= 80 for item in scenes)
    assert max(item["end_seconds"] - item["start_seconds"] for item in scenes) <= 30


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

    Mirrors edge_tts.Communicate(text, voice, boundary=...).stream()
    without contacting any external service.
    """

    calls: list[tuple[str, str]] = []

    def __init__(self, text: str, voice: str, *, boundary: str = "WordBoundary") -> None:
        assert boundary == "WordBoundary"
        type(self).calls.append((text, voice))
        self.text = text

    async def stream(self):
        yield {"type": "audio", "data": b"ID3-fake-mp3"}
        for i, word in enumerate(self.text.split()):
            yield {"type": "WordBoundary", "text": word,
                   "offset": i * 3_000_000, "duration": 2_000_000}


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
        def __init__(self, text: str, voice: str, *, boundary: str) -> None:
            self.text = text
            self.voice = voice

        async def stream(self):
            yield {"type": "audio", "data": b"partial"}
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
        def __init__(self, text: str, voice: str, *, boundary: str) -> None:
            pass

        async def stream(self):
            yield {"type": "audio", "data": b""}

    monkeypatch.setitem(
        sys.modules,
        "edge_tts",
        types.SimpleNamespace(Communicate=EmptyCommunicate),
    )
    import movie_review_factory.pipeline as pipeline

    with pytest.raises(RuntimeError, match="no audio"):
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


def test_tts_prosody_resolves_presets_overrides_and_drops_noops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline

    for name in ("MRF_TTS_EMOTION", "MRF_TTS_RATE", "MRF_TTS_PITCH", "MRF_TTS_VOLUME"):
        monkeypatch.delenv(name, raising=False)
    assert pipeline._tts_prosody() == {}

    monkeypatch.setenv("MRF_TTS_EMOTION", "dramatic")
    assert pipeline._tts_prosody() == {"rate": "-6%", "pitch": "-4Hz"}

    monkeypatch.setenv("MRF_TTS_RATE", "+20%")   # explicit override wins
    monkeypatch.setenv("MRF_TTS_VOLUME", "+0%")  # no-op is dropped
    assert pipeline._tts_prosody() == {"rate": "+20%", "pitch": "-4Hz"}

    monkeypatch.setenv("MRF_TTS_EMOTION", "")
    monkeypatch.delenv("MRF_TTS_RATE", raising=False)
    monkeypatch.setenv("MRF_TTS_PITCH", "bogus")  # invalid is ignored
    assert pipeline._tts_prosody() == {}


def test_tts_applies_prosody_to_communicate_when_emotion_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(tmp_path, [{"title": "Hook", "narration": "Xin ch\u00e0o th\u1ebf gi\u1edbi"}])

    captured: dict[str, object] = {}

    class ProsodyCommunicate:
        def __init__(self, text: str, voice: str, *, boundary: str = "WordBoundary",
                     rate: str | None = None, pitch: str | None = None, volume: str | None = None) -> None:
            assert boundary == "WordBoundary"
            captured.update(rate=rate, pitch=pitch, volume=volume)
            self.text = text

        async def stream(self):
            yield {"type": "audio", "data": b"ID3-fake-mp3"}
            for i, word in enumerate(self.text.split()):
                yield {"type": "WordBoundary", "text": word,
                       "offset": i * 3_000_000, "duration": 2_000_000}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=ProsodyCommunicate))
    monkeypatch.setenv("MRF_TTS_EMOTION", "energetic")
    from movie_review_factory.pipeline import _tts

    _tts(tmp_path, load_manifest(tmp_path))

    assert (captured["rate"], captured["pitch"]) == ("+12%", "+8Hz")
    voice_meta = json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))
    assert voice_meta["prosody"] == {"rate": "+12%", "pitch": "+8Hz"}


def test_tts_is_deterministic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _approved_script_job(tmp_path, [{"title": "Hook", "narration": "Xin chào."}])
    monkeypatch.setitem(sys.modules, "edge_tts", _fake_edge_tts())
    from movie_review_factory.pipeline import _tts

    _tts(tmp_path, load_manifest(tmp_path))
    first = (tmp_path / "voice.json").read_text(encoding="utf-8")
    _tts(tmp_path, load_manifest(tmp_path))
    second = (tmp_path / "voice.json").read_text(encoding="utf-8")
    assert first == second


def test_tts_retries_transient_no_audio_without_installing_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _approved_script_job(tmp_path, [{"title": "Hook", "narration": "Xin chào."}])

    class NoAudioReceived(Exception):
        pass

    class IntermittentCommunicate:
        calls = 0

        def __init__(self, text, voice, *, boundary):
            type(self).calls += 1
            self.attempt = type(self).calls

        async def stream(self):
            yield {"type": "audio", "data": b"partial" if self.attempt == 1 else b"complete"}
            if self.attempt == 1:
                raise NoAudioReceived("No audio was received")
            yield {"type": "WordBoundary", "offset": 1000000,
                   "duration": 2000000, "text": "Xin"}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(
        Communicate=IntermittentCommunicate,
        exceptions=types.SimpleNamespace(NoAudioReceived=NoAudioReceived),
    ))
    import movie_review_factory.pipeline as pipeline
    monkeypatch.setattr(pipeline.time, "sleep", lambda _seconds: None)
    pipeline._tts(tmp_path, load_manifest(tmp_path))
    assert IntermittentCommunicate.calls == 2
    assert (tmp_path / "narration.mp3").read_bytes() == b"complete"
    assert json.loads((tmp_path / "voice.json").read_text(encoding="utf-8"))["word_boundaries"][0]["text"] == "Xin"


def test_create_job_resets_tts_artifacts(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="tts-job"))
    (tmp_path / "voice.json").write_text("stale", encoding="utf-8")
    (tmp_path / "narration.mp3").write_bytes(b"stale")
    create_job(tmp_path, JobConfig(job_id="tts-job"))
    assert not (tmp_path / "voice.json").exists()
    assert not (tmp_path / "narration.mp3").exists()


# --- alignment stage --------------------------------------------------------


# A captions.srt where cue 3 runs past the narration bound (5s) so alignment must clamp it, and cue 4 starts past the bound so it must be dropped.
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

    # Narration is 5s long: cue 3 (4.0-6.0) clamps to 4.0-5.0, cue 4 (7.0-8.0) is entirely out of bounds and is dropped.
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
    # No-video run: 5 text stages ready, the rest skipped (incl. optional watermark).
    assert counts["ready"] == 5
    assert counts["skipped"] == 10


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
    # The regex accepts '.' as the ms separator and 1-3 ms digits; multi-line cue text must survive as a single cue with lines joined by newlines.
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


def test_alignment_splits_long_narration_into_two_line_cues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _alignment_job(tmp_path)
    narration = " ".join(["Đây là một câu chuyện về người hùng và đồng đội."] * 25)
    voice_path = tmp_path / "voice.json"
    voice = json.loads(voice_path.read_text(encoding="utf-8"))
    voice["sections"] = [{"narration": narration}]
    voice_path.write_text(json.dumps(voice), encoding="utf-8")
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda src: 60.0)
    pipeline._alignment(tmp_path, load_manifest(tmp_path))

    cues = json.loads((tmp_path / "alignment.json").read_text(encoding="utf-8"))["cues"]
    assert len(cues) > 15
    assert " ".join(cue["text"].replace("\n", " ") for cue in cues) == narration
    assert all(len(cue["text"].splitlines()) <= 2 for cue in cues)
    assert all(all(len(line) <= 42 for line in cue["text"].splitlines()) for cue in cues)
    assert all(cue["end_seconds"] - cue["start_seconds"] <= 6.01 for cue in cues)
    assert cues[0]["start_seconds"] == 0
    assert cues[-1]["end_seconds"] == 60.0
    assert all(a["end_seconds"] == b["start_seconds"] for a, b in zip(cues, cues[1:]))


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


@pytest.mark.parametrize(("ratio", "dimensions", "bottom_band"), [
    ("16:9", (1920, 1080), 0), ("16:9", (1920, 1080), .1),
    ("9:16", (1080, 1920), 0),
])
def test_render_produces_deterministic_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ratio: str, dimensions: tuple[int, int], bottom_band: float
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
    manifest = load_manifest(tmp_path)
    manifest.config.brand_bottom_band = bottom_band
    artifacts, message = pipeline._render(tmp_path, manifest)

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
    # Each clip decodes from its own seeked source input (-ss/-t -i), not a trim
    # of a shared input, so the whole timeline is never buffered (the OOM fix).
    assert "trim=start=" not in filter_complex
    _ss = commands[0].index("-ss")
    assert commands[0][_ss:_ss + 4] == ["-ss", "1.000000", "-t", "2.000000"]
    assert commands[0][_ss + 4] == "-i"
    # Captions burn from a generated ASS whose PlayRes matches the frame, so the
    # filter references aligned.ass and carries no libass-default force_style.
    assert "subtitles='" not in filter_complex
    assert "original_size" not in filter_complex
    assert "force_style" not in filter_complex
    assert "ass='" in filter_complex
    assert pipeline._escape_ffmpeg_filter_path(tmp_path / "aligned.ass") in filter_complex

    band = max(0.10, bottom_band + 0.03)
    expected_v = round(dimensions[1] * band)
    expected_h = round(dimensions[0] * 0.075)
    expected_font = round(dimensions[1] * 0.042)
    caption = (tmp_path / "aligned.ass").read_text(encoding="utf-8")
    # PlayRes must equal the real frame; this is the guard against the 384x288
    # libass-default regression that floated captions into the upper half.
    assert f"PlayResX: {dimensions[0]}" in caption
    assert f"PlayResY: {dimensions[1]}" in caption
    assert "WrapStyle: 0" in caption
    # Style tail: ...BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
    assert f"Style: Default,Arial,{expected_font}," in caption
    assert f"1,2,1,2,{expected_h},{expected_h},{expected_v},1" in caption
    assert str(source) in commands[0]
    # Source range equals playback length, so no loop filter is introduced.
    assert "loop=loop=" not in filter_complex


def test_render_ken_burns_adds_zoompan_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setenv("MRF_KEN_BURNS", "1")
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, load_manifest(tmp_path))

    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    assert "zoompan=" in filter_complex


def test_render_visual_variety_adds_filters_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    manifest = load_manifest(tmp_path)
    manifest.config.visual_variety = "balanced"
    pipeline.save_manifest(tmp_path, manifest)

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, manifest)

    filter_complex = commands[0][commands[0].index("-filter_complex") + 1]
    assert "hflip" in filter_complex
    assert "crop=" in filter_complex
    assert "eq=" in filter_complex



def test_render_transition_sfx_mixes_whoosh_on_cuts_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    whoosh = tmp_path / "whoosh.wav"
    whoosh.write_bytes(b"RIFFfake")
    _render_job(tmp_path, clips=[
        {"section": "Hook", "type": "narration", "source_clip": {"start_seconds": 1.0, "end_seconds": 3.0}},
        {"section": "Body", "type": "narration", "source_clip": {"start_seconds": 4.0, "end_seconds": 6.0}},
    ])
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setenv("MRF_TRANSITION_SFX", str(whoosh))
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, load_manifest(tmp_path))

    cmd = commands[0]
    assert str(whoosh) in cmd  # whoosh SFX input appended after the clip inputs
    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    assert "amix=inputs=2" in filter_complex  # base audio + one whoosh at the single cut
    assert "adelay=" in filter_complex


def test_render_falls_back_to_libx264_when_hardware_encoder_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    monkeypatch.setenv("MRF_VIDEO_ENCODER", "nvenc")
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        if "h264_nvenc" in command:
            raise pipeline.subprocess.CalledProcessError(
                1, command, output="", stderr="No NVENC capable devices found",
            )
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    artifacts, _ = pipeline._render(tmp_path, load_manifest(tmp_path))

    assert [artifact.name for artifact in artifacts] == ["render.json", "final.mp4"]
    assert "h264_nvenc" in commands[0]        # hardware encoder attempted first
    assert "libx264" in commands[-1]          # transparent software fallback
    assert (tmp_path / "final.mp4").read_bytes() == b"fake-mp4"


def test_render_default_has_no_intro_outro_concat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard: with intro/outro at the 0s default the render must be the
    # plain body -- no card inputs, no concat, and the map stays on [rendered].
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
    pipeline._render(tmp_path, load_manifest(tmp_path))

    command = commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]
    # The intro/outro path is uniquely marked by the [showv]/[bodyv] labels; the
    # single-clip video concat (concat=n=1:...:a=0[video]) is a separate, expected
    # part of every render and must not be conflated with card bookending.
    assert "[showv]" not in filter_complex and "[showa]" not in filter_complex
    assert "[bodyv]" not in filter_complex
    assert "-loop" not in command
    # The body stays on [rendered] and is encoded as a single output. The QA
    # detect filters are no longer folded via split (that caused the render OOM),
    # so the first mapped stream is [rendered] and the QA video split is gone.
    assert "split=2[venc][vqa]" not in filter_complex
    assert command[command.index("-map") + 1] == "[rendered]"
    render = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert render["intro_seconds"] == 0.0 and render["outro_seconds"] == 0.0
    assert not (tmp_path / "intro-card.png").exists()


def test_render_encodes_single_output_without_qa_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression guard for the ffmpeg OOM (exit -12): the render must NOT fold the
    # QA detect filters onto a second (-f null) output. Splitting (split=2) the
    # encode and detect branches let the fast loop-filter frames queue unbounded
    # against the slow ebur128 branch and exhausted memory on long timelines. The
    # render now encodes a single output; the qa stage runs black/freeze/silence/
    # loudness as its own decode pass over final.mp4, so no render-signals.log is
    # written by the render pass.
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
    pipeline._render(tmp_path, load_manifest(tmp_path))

    assert len(commands) == 1
    command = commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]
    # No QA fold: the video split and the detect filters are absent.
    assert "split=2[venc][vqa]" not in filter_complex
    assert "asplit=2[aenc][aqa]" not in filter_complex
    assert "blackdetect=" not in filter_complex and "freezedetect=" not in filter_complex
    assert "silencedetect=" not in filter_complex and "ebur128=" not in filter_complex
    # A single encode output: [rendered] + audio, no discarded detect maps.
    map_targets = [command[i + 1] for i, tok in enumerate(command) if tok == "-map"]
    assert len(map_targets) == 2
    assert map_targets[0] == "[rendered]"
    assert "[vdet]" not in map_targets and "[adet]" not in map_targets
    # The command ends at the single temp output; there is no "-f null -".
    assert command[-1] == str(tmp_path / "final.rendering.mp4")
    assert "null" not in command
    # The render pass no longer writes a signal log; qa decodes final.mp4 itself.
    assert not (tmp_path / "render-signals.log").exists()


def test_render_bookends_body_with_branded_intro_outro_cards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    commands: list[list[str]] = []
    # Narration body is 2s; final output must be intro(2)+body(2)+outro(3)=7s, so
    # the probe of the muxed file returns 7 to satisfy the drift guard.
    def probe(path: Path) -> float:
        if path.name == "narration.mp3":
            return 2.0
        if path.name == "final.rendering.mp4":
            return 7.0
        return 10.0

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", probe)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    manifest = load_manifest(tmp_path)
    manifest.config.intro_seconds = 2.0
    manifest.config.outro_seconds = 3.0
    pipeline._render(tmp_path, manifest)

    command = commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]
    # Three exact-length segments (intro card, narration body, outro card) are
    # concatenated into one branded show; caption/section timing rides the body.
    assert "concat=n=3:v=1:a=1[showv][showa]" in filter_complex
    assert "[introv]" in filter_complex and "[bodyv]" in filter_complex and "[outrov]" in filter_complex
    assert f"trim=duration={2.0:.6f}" in filter_complex  # body trimmed to narration length
    # Both cards are looped stills fed as extra inputs at their exact durations.
    assert command.count("-loop") == 2
    assert "2.000" in command and "3.000" in command
    # Output maps switch to the concatenated show streams and are encoded
    # directly; the QA detect fold (split/asplit) was removed to fix the render
    # OOM, so the show streams are mapped straight to the encoder.
    assert "split=2[venc][vqa]" not in filter_complex
    assert "asplit=2[aenc][aqa]" not in filter_complex
    map_indices = [i for i, tok in enumerate(command) if tok == "-map"]
    assert command[map_indices[0] + 1] == "[showv]"
    assert command[map_indices[1] + 1] == "[loudnorm]"
    assert "[showa]loudnorm=" in filter_complex
    # Hard duration cap covers body + intro + outro.
    assert f"{7.0:.6f}" in command
    render = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert render["intro_seconds"] == 2.0
    assert render["outro_seconds"] == 3.0
    assert render["output_duration_seconds"] == 7.0
    assert render["duration_drift_seconds"] == 0.0
    # Transient card PNGs are cleaned up after the mux.
    assert not (tmp_path / "intro-card.png").exists()
    assert not (tmp_path / "outro-card.png").exists()


def test_caption_ass_pins_playres_to_frame_and_bottom_safe_geometry() -> None:
    from movie_review_factory import pipeline

    srt = "1\n00:00:00,200 --> 00:00:09,000\nDòng một dài\nDòng hai dài\n"
    ass = pipeline.caption_ass(srt, 1920, 1080, 0.10)
    # PlayRes equals the real frame -- the guard against libass's 384x288
    # default that mis-scaled pixel margins and floated captions to the top.
    assert "PlayResX: 1920" in ass
    assert "PlayResY: 1080" in ass
    assert "WrapStyle: 0" in ass
    # Bottom-centre (Alignment=2) with pixel margins/font from the frame size:
    # FontSize = 1080*0.042 = 45; MarginL/R = 1920*0.075 = 144; MarginV = 108.
    assert "Style: Default,Arial,45," in ass
    assert "1,2,1,2,144,144,108,1" in ass
    # SRT line breaks map to hard \N so a cue never overflows into stacked words.
    assert "0:00:00.20,0:00:09.00" in ass
    assert r"Dòng một dài\NDòng hai dài" in ass


def test_caption_ass_scales_geometry_with_resolution_and_band() -> None:
    from movie_review_factory import pipeline

    srt = "1\n00:00:00,000 --> 00:00:01,000\nHi\n"
    ass = pipeline.caption_ass(srt, 1080, 1920, 0.13)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    # FontSize = 1920*0.042 = 81; MarginL/R = 1080*0.075 = 81; MarginV = 250.
    assert "Style: Default,Arial,81," in ass
    assert "1,2,1,2,81,81,250,1" in ass


def test_caption_ass_neutralises_override_syntax() -> None:
    from movie_review_factory import pipeline

    srt = "1\n00:00:00,000 --> 00:00:01,000\n{\\an8}top hack\n"
    events = pipeline.caption_ass(srt, 960, 540, 0.10).split("[Events]", 1)[1]
    assert "{" not in events and "}" not in events
    assert "\\an8" not in events


def test_caption_ass_accepts_explicit_pixel_overrides_for_shorts_panel() -> None:
    from movie_review_factory import pipeline

    srt = "1\n00:00:00,000 --> 00:00:02,000\nBình luận\n"
    # Portrait Shorts panel geometry: frame-sized PlayRes, bespoke font/MarginV.
    ass = pipeline.caption_ass(srt, 1080, 1920, margin_v=590, font_size=45)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass
    assert "WrapStyle: 0" in ass
    assert "Style: Default,Arial,45," in ass
    # default side margin 1080*0.075=81; Alignment=2; MarginV=590 (in the panel).
    assert "1,2,1,2,81,81,590,1" in ass


def test_caption_ass_requires_band_or_margin_v() -> None:
    from movie_review_factory import pipeline

    with pytest.raises(ValueError, match="band or margin_v"):
        pipeline.caption_ass("1\n00:00:00,000 --> 00:00:01,000\nHi\n", 1920, 1080)


def test_render_uses_authorized_music_mix_only_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    music = tmp_path / "licensed-music.wav"
    music.write_bytes(b"fake-audio")
    (tmp_path / "audio_mix.json").write_text(json.dumps({
        "voice_gain_db": 1,
        "music": {"path": str(music), "rights_note": "licensed by creator", "gain_db": -18},
    }), encoding="utf-8")
    import movie_review_factory.pipeline as pipeline
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, load_manifest(tmp_path))
    command = commands[0]
    assert str(music) in command
    filter_complex = command[command.index("-filter_complex") + 1]
    # The mix output [audio] is mapped directly for encode; the QA detect fold
    # that consumed it via asplit was removed to fix the render OOM.
    assert "asplit=2[aenc][aqa]" not in filter_complex
    map_indices = [i for i, tok in enumerate(command) if tok == "-map"]
    assert command[map_indices[1] + 1] == "[loudnorm]"
    assert "[audio]loudnorm=" in filter_complex
    assert "sidechaincompress" in filter_complex
    render = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert render["audio_mix"]["provenance"][0]["rights_note"] == "licensed by creator"


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
    # The 2s clip grows into the free rest of the 10s source; the shortfall loops: 250 frames, 2 extra loops -> 30s.
    _ss = commands[0].index("-ss")
    assert commands[0][_ss:_ss + 4] == ["-ss", "0.000000", "-t", "10.000000"]
    assert "loop=loop=2:size=250:start=0" in filter_complex
    assert "trim=end=30.000000" in filter_complex
    doc = json.loads((tmp_path / "render.json").read_text(encoding="utf-8"))
    assert doc["clips"][0]["source_seconds"] == 2.0
    assert doc["clips"][0]["read_seconds"] == 10.0
    assert doc["clips"][0]["duration_seconds"] == 30.0


def test_extend_short_ranges_uses_free_neighbouring_footage_only() -> None:
    from movie_review_factory.pipeline import _extend_short_ranges

    ranges = [
        {"index": 0, "start_seconds": 10.0, "end_seconds": 12.0, "duration_seconds": 6.0},
        {"index": 1, "start_seconds": 14.0, "end_seconds": 20.0, "duration_seconds": 6.0},
        {"index": 2, "start_seconds": 21.0, "end_seconds": 22.0, "duration_seconds": 4.0},
    ]
    _extend_short_ranges(ranges, 23.0)

    # Clip 0 takes the 2s gap before clip 1, then 2s backwards.
    assert (ranges[0]["read_start_seconds"], ranges[0]["read_seconds"]) == (8.0, 6.0)
    # Clip 1 already fills its slot, so it keeps its planned range.
    assert "read_seconds" not in ranges[1]
    # Clip 2 gets 1s up to the source end and the 1s gap after clip 1.
    assert (ranges[2]["read_start_seconds"], ranges[2]["read_seconds"]) == (20.0, 3.0)


def test_render_normalizes_loudness_for_youtube(
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
    pipeline._render(tmp_path, load_manifest(tmp_path))

    command = commands[0]
    filter_complex = command[command.index("-filter_complex") + 1]
    assert "[1:a:0]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[loudnorm]" in filter_complex
    map_targets = [command[i + 1] for i, tok in enumerate(command) if tok == "-map"]
    assert map_targets[1] == "[loudnorm]"


def test_render_removes_stale_pre_midroll_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _render_job(tmp_path)
    import movie_review_factory.pipeline as pipeline

    (tmp_path / "final.pre-midroll.mp4").write_bytes(b"old-captions")
    (tmp_path / "script.pre-midroll.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def fake_run(command: list[str], **kwargs: object) -> object:
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    pipeline._render(tmp_path, load_manifest(tmp_path))

    assert not (tmp_path / "final.pre-midroll.mp4").exists()
    assert (tmp_path / "script.pre-midroll.json").exists()


def test_render_reads_the_middle_of_a_long_source_range(
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
    # Clip decodes from its own seeked input (-ss/-t): the middle second of the 3 s range.
    assert "trim=start=" not in filter_complex
    _ss = commands[0].index("-ss")
    assert commands[0][_ss:_ss + 4] == ["-ss", "1.000000", "-t", "1.000000"]
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
    # Empty captions -> no burn filter at all (neither the ASS nor SRT path).
    assert "ass=" not in empty_filter
    assert "subtitles=" not in empty_filter

    (tmp_path / "aligned.srt").write_text(
        "1\\n00:00:00,000 --> 00:00:02,000\\nXin chao\\n", encoding="utf-8"
    )
    pipeline._render(tmp_path, load_manifest(tmp_path))
    non_empty_filter = commands[-1][commands[-1].index("-filter_complex") + 1]
    # Non-empty captions burn through the frame-sized ASS (aligned.ass).
    assert "ass=" in non_empty_filter


def test_render_real_ffmpeg_extends_short_video_to_narration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess

    import movie_review_factory.pipeline as pipeline

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg toolchain unavailable")

    source = _render_job(tmp_path, clips=[{
        "section": "Hook",
        "type": "narration",
        "duration_seconds": 1.0,
        "source_clip": {"start_seconds": 0.0, "end_seconds": 1.0},
    }])
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-f", "lavfi",
        "-i", "color=c=black:s=320x180:r=25:d=1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source),
    ], capture_output=True, check=True)
    narration = tmp_path / "narration.mp3"
    subprocess.run([
        ffmpeg, "-v", "error", "-y", "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2",
        "-c:a", "libmp3lame", str(narration),
    ], capture_output=True, check=True)
    (tmp_path / "aligned.srt").write_text("", encoding="utf-8")
    monkeypatch.setenv("MRF_RENDER_MAX_HEIGHT", "180")

    artifacts, _ = pipeline._render(tmp_path, load_manifest(tmp_path))
    assert [artifact.name for artifact in artifacts] == ["render.json", "final.mp4"]
    output_duration = pipeline._probe_duration_seconds(tmp_path / "final.mp4")
    narration_duration = pipeline._probe_duration_seconds(narration)
    assert output_duration is not None and narration_duration is not None
    assert abs(output_duration - narration_duration) <= pipeline.RENDER_DURATION_DRIFT_SECONDS


def test_render_real_ffmpeg_changes_brand_between_chapters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess
    import movie_review_factory.pipeline as pipeline

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg toolchain unavailable")
    clips = [
        {"section": section, "type": "narration", "duration_seconds": 1.0,
         "source_clip": {"start_seconds": index, "end_seconds": index + 1}}
        for index, section in enumerate(("Mở đầu", "Race: hành trình", "Swarm: đối đầu"))
    ]
    source = _render_job(tmp_path, clips=clips)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=black:s=320x180:r=25:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                   capture_output=True, check=True)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=3",
                    "-c:a", "libmp3lame", str(tmp_path / "narration.mp3")],
                   capture_output=True, check=True)
    (tmp_path / "aligned.srt").write_text("", encoding="utf-8")
    manifest = load_manifest(tmp_path)
    manifest.config.movie_title = "BEN: Race + BEN: Swarm"
    manifest.config.brand_top_band = .1
    manifest.config.brand_bottom_band = .1
    monkeypatch.setenv("MRF_RENDER_MAX_HEIGHT", "180")
    artifacts, _ = pipeline._render(tmp_path, manifest)
    assert [artifact.name for artifact in artifacts] == ["render.json", "final.mp4"]
    assert (tmp_path / "final.mp4").stat().st_size > 0
    assert len(list(tmp_path.glob("brand-overlay-*.png"))) == 3
    # A top band keeps the lockup static inside it; only the faint ghost drifts.
    assert not (tmp_path / "brand-mark.png").exists()
    assert (tmp_path / "brand-ghost.png").exists()


def test_render_brand_guard_moves_corner_mark_without_bands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil
    import subprocess
    from PIL import Image
    import movie_review_factory.branding as branding
    import movie_review_factory.pipeline as pipeline

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg toolchain unavailable")
    source = _render_job(tmp_path, clips=[
        {"section": "Mở đầu", "type": "narration", "duration_seconds": 3.0,
         "source_clip": {"start_seconds": 0, "end_seconds": 3}},
    ])
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "color=c=black:s=320x180:r=25:d=3",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)],
                   capture_output=True, check=True)
    subprocess.run([ffmpeg, "-v", "error", "-y", "-f", "lavfi",
                    "-i", "sine=frequency=440:duration=3",
                    "-c:a", "libmp3lame", str(tmp_path / "narration.mp3")],
                   capture_output=True, check=True)
    (tmp_path / "aligned.srt").write_text("", encoding="utf-8")
    monkeypatch.setenv("MRF_RENDER_MAX_HEIGHT", "180")
    monkeypatch.delenv(branding.BRAND_GUARD_ENV, raising=False)
    monkeypatch.setattr(branding, "MARK_HOP_SECONDS", 1)
    pipeline._render(tmp_path, load_manifest(tmp_path))
    assert (tmp_path / "brand-mark.png").exists() and (tmp_path / "brand-ghost.png").exists()

    def corner_peaks(at: float) -> tuple[int, int]:
        frame = tmp_path / f"frame-{at}.png"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", str(at), "-i", str(tmp_path / "final.mp4"),
                        "-frames:v", "1", str(frame)], capture_output=True, check=True)
        with Image.open(frame) as image:
            gray = image.convert("L")
            return (gray.crop((0, 0, 60, 50)).getextrema()[1],
                    gray.crop((gray.width - 60, 0, gray.width, 50)).getextrema()[1])

    left, right = corner_peaks(0.5)
    assert left > 180 and right < 120  # first hop: top-left
    left, right = corner_peaks(1.5)
    assert left < 120 and right > 180  # next hop: top-right


@pytest.mark.parametrize(("planned", "padding"), [(1.5, 1.5), (2.0, 1.0)])
def test_render_pads_final_frame_when_narration_outlasts_scene_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, planned: float, padding: float
) -> None:
    import movie_review_factory.pipeline as pipeline

    _render_job(tmp_path, clips=[{
        "section": "Hook",
        "type": "narration",
        "duration_seconds": planned,
        "source_clip": {"start_seconds": 1.0, "end_seconds": 1.0 + planned},
    }])
    commands: list[list[str]] = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")

    def probe(path: Path) -> float:
        if path.name == "narration.mp3":
            return 2.0
        if path.name == "final.rendering.mp4":
            graph = commands[0][commands[0].index("-filter_complex") + 1]
            return 2.0 if "tpad=stop_mode=clone:" in graph else 1.5
        return 10.0

    def fake_run(command: list[str], **kwargs: object) -> object:
        commands.append(command)
        (tmp_path / "final.rendering.mp4").write_bytes(b"fake-mp4")
        return object()

    monkeypatch.setattr(pipeline, "_probe_duration_seconds", probe)
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)

    artifacts, _ = pipeline._render(tmp_path, load_manifest(tmp_path))
    graph = commands[0][commands[0].index("-filter_complex") + 1]
    assert f"tpad=stop_mode=clone:stop_duration={padding:.6f}" in graph
    assert [artifact.name for artifact in artifacts] == ["render.json", "final.mp4"]
    assert json.loads((tmp_path / "render.json").read_text())["duration_drift_seconds"] == 0.0


def test_render_in_cancellation_scope_completes_when_not_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys
    import threading

    import movie_review_factory.cancellation as cancellation
    import movie_review_factory.pipeline as pipeline

    _render_job(tmp_path)
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/ffmpeg")

    original_popen = subprocess.Popen
    registered: list[subprocess.Popen] = []
    unregistered: list[subprocess.Popen] = []

    def finish_ffmpeg(command: list[str], **kwargs: object) -> subprocess.Popen:
        return original_popen(
            [sys.executable, "-c",
             "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'video')",
             str(tmp_path / "final.rendering.mp4")],
            **kwargs,
        )

    monkeypatch.setattr(pipeline.subprocess, "Popen", finish_ffmpeg)
    context = cancellation.CancellationContext(
        threading.Event(), registered.append, unregistered.append,
    )
    with cancellation.cancellation_scope(context):
        artifacts, message = pipeline._render(tmp_path, load_manifest(tmp_path))

    assert message == "rendered 1 clips to final.mp4"
    assert [artifact.name for artifact in artifacts] == ["render.json", "final.mp4"]
    assert (tmp_path / "final.mp4").read_bytes() == b"video"
    assert len(registered) == 1
    assert unregistered == registered
    assert registered[0].returncode == 0


def test_render_stop_terminates_ffmpeg_and_cleans_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import subprocess
    import sys
    import threading

    import movie_review_factory.cancellation as cancellation
    import movie_review_factory.pipeline as pipeline

    _render_job(tmp_path)
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: _render_duration(path, tmp_path))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "/ffmpeg")

    original_popen = subprocess.Popen
    event = threading.Event()
    registered: list[subprocess.Popen] = []
    unregistered: list[subprocess.Popen] = []

    def start_slow_ffmpeg(command: list[str], **kwargs: object) -> subprocess.Popen:
        (tmp_path / "final.rendering.mp4").write_bytes(b"partial")
        process = original_popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            **kwargs,
        )
        event.set()
        return process

    monkeypatch.setattr(pipeline.subprocess, "Popen", start_slow_ffmpeg)
    def old_blocking_path(command: list[str], **kwargs: object) -> None:
        raise cancellation.RunCancelled("stop requested before process registration")

    monkeypatch.setattr(pipeline.subprocess, "run", old_blocking_path)
    manifest = load_manifest(tmp_path)
    for stage in manifest.stages:
        if stage.stage != "render":
            stage.status = "ready"
    pipeline.save_manifest(tmp_path, manifest)

    context = cancellation.CancellationContext(event, registered.append, unregistered.append)
    try:
        with cancellation.cancellation_scope(context):
            result = run_job(tmp_path, until="render")
        assert result.stage("render").status == "cancelled"
        assert len(registered) == 1
        assert unregistered == registered
        assert registered[0].poll() is not None
        assert not (tmp_path / "final.rendering.mp4").exists()
        assert not (tmp_path / "final.mp4").exists()
        assert not (tmp_path / "render.json").exists()
    finally:
        for process in registered:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


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
    from movie_review_factory import media_qa

    monkeypatch.setattr(media_qa, "inspect_rendered_media", lambda *_a, **_kw: [
        {"check": "decoded_media_scan", "value": {"video": True, "audio": True},
         "passed": True, "message": ""},
    ])
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


def test_qa_reuses_render_signal_log_without_second_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # When the render pass left a usable detect log, QA parses it instead of
    # decoding final.mp4 a second time.
    _qa_job(tmp_path)
    import movie_review_factory.pipeline as pipeline
    from movie_review_factory import media_qa

    (tmp_path / "render-signals.log").write_text(
        "[blackdetect @ 1] black_start:1 black_end:2 black_duration:1\n"
        "[Parsed_ebur128 @ 2] Integrated loudness:\n    I:  -15.0 LUFS\n"
        "[Parsed_ebur128 @ 2] True peak:\n    Peak:  -2.0 dBFS\n",
        encoding="utf-8",
    )

    def _no_decode(*_a: object, **_kw: object) -> list[dict]:
        raise AssertionError("QA must not decode final.mp4 again when a render log exists")

    monkeypatch.setattr(media_qa, "inspect_rendered_media", _no_decode)
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: f"/{name}")
    monkeypatch.setattr(pipeline.subprocess, "run",
                        lambda command, **kwargs: types.SimpleNamespace(
                            returncode=0, stdout=_qa_probe_output(), stderr=""))

    artifacts, message = pipeline._qa(tmp_path, load_manifest(tmp_path))

    assert [a.name for a in artifacts] == ["qa.json"]
    doc = json.loads((tmp_path / "qa.json").read_text(encoding="utf-8"))
    assert doc["passed"] is True
    scan = next(c for c in doc["checks"] if c["check"] == "decoded_media_scan")
    assert scan["value"]["source"] == "render_pass"
    check_names = [c["check"] for c in doc["checks"]]
    assert "black_intervals" in check_names and "decoded_audio_loudness" in check_names


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


def test_scene_candidates_keep_relevant_distant_scene_and_silent_anchor() -> None:
    import movie_review_factory.pipeline as pipeline_mod

    scenes = pipeline_mod._build_scenes([], 600.0)
    scenes[17]["text"] = "hidden lighthouse"
    sections = [
        {"title": "lighthouse", "narration": "hidden lighthouse", "duration_seconds": 30},
        {"title": "quiet ending", "narration": "silence", "duration_seconds": 30},
    ]
    result = pipeline_mod._retrieve_scene_candidates(sections, {"scenes": scenes, "duration_seconds": 600.0})
    assert len(result) == 2
    assert all(1 <= len(items) <= 12 for items in result)
    assert 18 in [item["index"] for item in result[0]]
    assert 1 in [item["index"] for item in result[0]]
    assert 20 in [item["index"] for item in result[1]]


def test_scene_candidates_can_retrieve_visual_match_without_dialogue() -> None:
    import movie_review_factory.pipeline as pipeline_mod

    scenes = pipeline_mod._build_scenes([], 600.0)
    scenes[15]["visual_description"] = "a red car races through a tunnel"
    scenes[15]["visual_tags"] = ["red car", "tunnel"]
    scenes[15]["visual_actions"] = ["racing"]
    sections = [{
        "title": "Red car chase",
        "narration": "The car races through the tunnel",
        "duration_seconds": 30,
    }]

    result = pipeline_mod._retrieve_scene_candidates(
        sections, {"scenes": scenes, "duration_seconds": 600.0}
    )

    assert 16 in [item["index"] for item in result[0]]


def test_semantic_vectors_can_promote_distant_scene_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod
    import movie_review_factory.semantic_search as semantic_search
    from movie_review_factory.media_store import MediaStore
    from movie_review_factory.models import MediaAsset, Shot

    scenes = [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 10.0, "text": ""},
        {"index": 2, "start_seconds": 10.0, "end_seconds": 20.0, "text": ""},
        {"index": 3, "start_seconds": 20.0, "end_seconds": 30.0, "text": ""},
    ]
    database = tmp_path / "media_index.sqlite3"
    with MediaStore(database) as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=tmp_path / "owned.mp4", duration_seconds=30),
            [
                Shot(media_asset_id=1, start_seconds=scene["start_seconds"], end_seconds=scene["end_seconds"], label=f"Scene {scene['index']}")
                for scene in scenes
            ],
            [],
        )
        shots = store.list_shots(1)
        _, unrelated = semantic_search._as_blob([0.0, 1.0])
        _, wanted = semantic_search._as_blob([1.0, 0.0])
        store.replace_shot_embeddings([
            (int(shots[0].id), "quiet room", "test-model", 2, unrelated),
            (int(shots[1].id), "empty corridor", "test-model", 2, unrelated),
            (int(shots[2].id), "fast automobile in storm", "test-model", 2, wanted),
        ])

    _, query = semantic_search._as_blob([1.0, 0.0])
    monkeypatch.setattr(semantic_search, "model_name", lambda: "test-model")
    monkeypatch.setattr(semantic_search, "embed_query", lambda text: (2, query))

    lexical = [[scenes[0]]]
    merged = pipeline_mod._merge_semantic_scene_candidates(
        [{"title": "Vehicle chase", "narration": "automobile racing during bad weather"}],
        scenes,
        database,
        lexical,
    )

    assert merged[0][0]["index"] == 3
    assert {item["index"] for item in merged[0]} == {1, 2, 3}
    assert all("candidate_score" in item for item in merged[0])


def test_claude_scene_plan_limits_context_without_fake_visual_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    source = tmp_path / "owned.mp4"
    source.write_bytes(b"owned-video")
    create_job(tmp_path, JobConfig(job_id="frame-agent", source_video=source, content_agent="claude"))
    (tmp_path / "script.json").write_text(json.dumps({"sections": [
        {"title": "Lighthouse", "narration": "lighthouse", "duration_seconds": 60},
    ]}), encoding="utf-8")
    scenes = pipeline_mod._build_scenes([], 600.0)
    scenes[17]["text"] = "lighthouse"
    (tmp_path / "scenes.json").write_text(json.dumps({
        "source_video": str(source), "duration_seconds": 600.0, "scenes": scenes,
    }), encoding="utf-8")
    def fake_agent(**kwargs: object) -> dict:
        groups = kwargs["context"]["scene_candidates"]
        assert len(groups) == 1 and len(groups[0]) <= 12
        assert 18 in [item["index"] for item in groups[0]]
        assert all("frame_path" not in item for item in groups[0])
        assert kwargs["allowed_tools"] == []
        return {"assignments": [{"shots": [
            {"start_scene_index": 18, "end_scene_index": 18, "rationale": "transcript match"},
        ]}], "notes": "transcript evidence only"}

    monkeypatch.setattr(pipeline_mod, "_run_reasoning_agent", fake_agent)
    pipeline_mod._scene_plan(tmp_path, load_manifest(tmp_path))
    plan = json.loads((tmp_path / "scene_plan.json").read_text(encoding="utf-8"))
    assert plan["clips"][0]["source_clip"] == {"start_seconds": 510.0, "end_seconds": 540.0}
    assert plan["clips"][0]["notes"] == "transcript match"
    assert not list(tmp_path.glob("scene-evidence-*.jpg"))


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


def test_claude_scene_plan_rejects_index_outside_section_candidates() -> None:
    from movie_review_factory.pipeline import _agent_scene_assignments

    scenes = [{"index": i, "start_seconds": (i - 1) * 10.0, "end_seconds": i * 10.0}
              for i in range(1, 4)]
    with pytest.raises(ValueError, match="outside.*candidates"):
        _agent_scene_assignments(
            [{"title": "first"}],
            {"duration_seconds": 30.0, "scenes": scenes},
            [{"shots": [{"start_scene_index": 2, "end_scene_index": 2, "rationale": "guess"}]}],
            candidates=[[scenes[0], scenes[2]]],
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


@pytest.mark.parametrize(("aspect_ratio", "dimensions"), [("16:9", (1280, 720)), ("9:16", (720, 1280))])
def test_thumbnail_stage_generates_three_candidates_and_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    aspect_ratio: str,
    dimensions: tuple[int, int],
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    source = tmp_path / "owned.mp4"
    source.write_bytes(b"video")
    create_job(tmp_path, JobConfig(job_id="thumb", source_video=source, aspect_ratio=aspect_ratio))
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
    width, height = dimensions
    assert all(f"scale={width}:{height}" in call[call.index("-vf") + 1] for call in calls)
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
    assert (data["width"], data["height"]) == dimensions
    assert all((item["width"], item["height"]) == dimensions for item in data["candidates"])
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
    persisted = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert [stage["stage"] for stage in persisted["stages"]] == list(__import__(
        "movie_review_factory.pipeline", fromlist=["STAGES"]
    ).STAGES)
    assert validate_job(tmp_path) == []


def test_every_pipeline_stage_has_a_registered_handler() -> None:
    import movie_review_factory.pipeline as pipeline_mod

    assert set(pipeline_mod.STAGE_HANDLERS) == set(pipeline_mod.STAGES)


def test_script_and_metadata_mutations_use_atomic_json_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    create_job(tmp_path, JobConfig(job_id="atomic-writes"))
    run_job(tmp_path, until="metadata")
    original_write_json = pipeline_mod._write_json
    writes: list[str] = []

    def tracked_write_json(root: Path, name: str, data: dict) -> object:
        writes.append(name)
        return original_write_json(root, name, data)

    monkeypatch.setattr(pipeline_mod, "_write_json", tracked_write_json)
    pipeline_mod.approve_script(tmp_path)
    assert writes == ["script.json"]

    writes.clear()
    pipeline_mod.update_script(tmp_path, {"notes": "revision"})
    assert writes == ["script.json"]

    run_job(tmp_path, until="metadata")
    writes.clear()
    pipeline_mod.approve_metadata(tmp_path)
    assert writes == [pipeline_mod.METADATA_NAME]

    writes.clear()
    pipeline_mod.update_metadata(tmp_path, {"title": "Updated"})
    assert writes == [pipeline_mod.METADATA_NAME]


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

    monkeypatch.setattr(pipeline_mod, "SCRIPT_EXPAND_ROUNDS", 0)
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


def test_agy_content_agent_drives_research_outline_and_script(
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
                "brief": "Nghiên cứu sinh qua AGY pool.",
                "facts": ["Một sự kiện quan trọng."],
                "sources": [],
                "uncertainties": [],
            }
        if stage == "outline":
            return {
                "sections": [
                    {"title": "Hook", "budget_minutes": 1, "purpose": "Mở vấn đề"},
                    {"title": "Diễn biến", "budget_minutes": 2, "purpose": "Tóm tắt"},
                    {"title": "Phân tích", "budget_minutes": 1, "purpose": "Bình luận"},
                ],
                "notes": "agy outline",
            }
        if stage == "script":
            return {
                "sections": [
                    {"title": "Hook", "narration": "Mở đầu có nội dung thật."},
                    {"title": "Diễn biến", "narration": "Phần diễn biến có nội dung thật."},
                    {"title": "Phân tích", "narration": "Phần phân tích có nội dung thật."},
                ],
                "notes": "agy script",
            }
        raise AssertionError(stage)

    def forbidden(**_: object) -> dict:
        raise AssertionError("claude runner must not run for agy jobs")

    monkeypatch.setattr(pipeline_mod, "SCRIPT_EXPAND_ROUNDS", 0)
    monkeypatch.setattr(pipeline_mod, "run_agy_json", fake_agent)
    monkeypatch.setattr(pipeline_mod, "run_claude_json", forbidden)

    create_job(
        tmp_path,
        JobConfig(
            job_id="agy-job",
            language="vi",
            target_minutes=5,
            movie_title="Example Movie",
            content_agent="agy",
        ),
    )
    manifest = run_job(tmp_path, until="script")

    assert calls == ["research", "outline", "script"]
    assert manifest.stage("research").message == "research generated by AGY"
    assert json.loads((tmp_path / "research.json").read_text(encoding="utf-8"))["generator"] == "agy"
    assert json.loads((tmp_path / "outline.json").read_text(encoding="utf-8"))["generator"] == "agy"
    script = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    assert script["generator"] == "agy"
    assert manifest.stage("script").message.startswith("script generated by AGY")


@pytest.mark.parametrize("agent", ["claude", "agy"])
def test_script_stage_inserts_agy_written_midroll_cta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent: str,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    monkeypatch.setenv("MRF_AUTO_CTA", "1")
    line = "Nếu bạn cũng muốn biết cái kết, bấm thích và đăng ký kênh để không bỏ lỡ phần tiếp theo nhé."
    calls: list[tuple[str, str]] = []

    def content(stage: str) -> dict:
        if stage == "research":
            return {"brief": "Nghiên cứu.", "facts": ["Một sự kiện."], "sources": [], "uncertainties": []}
        if stage == "outline":
            return {"sections": [{"title": t, "budget_minutes": 1, "purpose": "p"} for t in ("Hook", "Diễn biến", "Kết")], "notes": ""}
        if stage == "script":
            return {"sections": [
                {"title": "Hook", "narration": "Mở đầu có nội dung thật. Câu thứ hai cũng thật."},
                {"title": "Diễn biến", "narration": "Phần diễn biến có nội dung thật. Thêm một câu nữa."},
                {"title": "Kết", "narration": "Phần kết có nội dung thật. Câu cuối cùng."},
            ], "notes": ""}
        raise AssertionError(stage)

    def fake_agy(**kwargs: object) -> dict:
        stage = str(kwargs["stage"])
        calls.append(("agy", stage))
        if stage == "script_cta":
            assert "đăng ký" in str(kwargs["prompt"]) and "Example Movie" in str(kwargs["prompt"])
            return {"line": line}
        return content(stage)

    def fake_claude(**kwargs: object) -> dict:
        calls.append(("claude", str(kwargs["stage"])))
        return content(str(kwargs["stage"]))

    monkeypatch.setattr(pipeline_mod, "SCRIPT_EXPAND_ROUNDS", 0)
    monkeypatch.setattr(pipeline_mod, "run_agy_json", fake_agy)
    monkeypatch.setattr(pipeline_mod, "run_claude_json", fake_claude)
    create_job(tmp_path, JobConfig(job_id="cta-job", language="vi", target_minutes=5,
                                   movie_title="Example Movie", content_agent=agent))

    manifest = run_job(tmp_path, until="script")

    # The CTA is always AGY-written, whichever agent wrote the script.
    assert calls == [(agent, "research"), (agent, "outline"), (agent, "script"), ("agy", "script_cta")]
    script = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    midrolls = [section for section in script["sections"] if section.get("midroll")]
    assert len(midrolls) == 1 and midrolls[0]["narration"] == line
    assert 0 < script["sections"].index(midrolls[0]) < len(script["sections"]) - 1
    assert script["approved"] is False
    assert "CTA after" in manifest.stage("script").message
    assert line in (tmp_path / "script.md").read_text(encoding="utf-8")


def test_script_stage_cta_failure_never_blocks_the_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod
    from movie_review_factory.content_agent import ContentAgentError

    monkeypatch.setenv("MRF_AUTO_CTA", "1")

    def fake_agy(**kwargs: object) -> dict:
        stage = str(kwargs["stage"])
        if stage == "script_cta":
            raise ContentAgentError("pool offline")
        if stage == "research":
            return {"brief": "b", "facts": [], "sources": [], "uncertainties": []}
        if stage == "outline":
            return {"sections": [{"title": "A", "budget_minutes": 1}, {"title": "B", "budget_minutes": 1}], "notes": ""}
        return {"sections": [{"title": "A", "narration": "Một câu. Hai câu."},
                             {"title": "B", "narration": "Ba câu. Bốn câu."}], "notes": ""}

    monkeypatch.setattr(pipeline_mod, "SCRIPT_EXPAND_ROUNDS", 0)
    monkeypatch.setattr(pipeline_mod, "run_agy_json", fake_agy)
    create_job(tmp_path, JobConfig(job_id="cta-fail", language="vi", target_minutes=2,
                                   movie_title="Example Movie", content_agent="agy"))

    manifest = run_job(tmp_path, until="script")

    assert manifest.stage("script").status == "ready"
    assert "CTA skipped: pool offline" in manifest.stage("script").message
    script = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    assert not any(section.get("midroll") for section in script["sections"])


def test_insert_cta_lands_near_narration_midpoint() -> None:
    from movie_review_factory import midroll

    sections = [{"title": "Only", "budget_minutes": 2,
                 "narration": " ".join(f"Câu số {i} có năm từ." for i in range(20))}]
    line = "Bấm thích và đăng ký kênh nhé."

    out, index = midroll.insert_cta(sections, line, "vi")

    assert [s["title"] for s in out] == ["Only", midroll.section_title("vi"), "Only (2)"]
    assert index == 1 and out[1]["midroll"] is True and out[1]["narration"] == line
    assert out[2]["continued"] is True
    assert sections[0]["title"] == "Only" and len(sections) == 1  # input not mutated
    assert midroll.clean_line("x " * 60) == "" and midroll.clean_line("  a  b ") == "a b"


def _fit_context_scenes(count: int = 47, repeats: int = 30) -> list[dict]:
    return [
        {
            "index": index,
            "start_seconds": index * 10.0,
            "end_seconds": index * 10.0 + 10.0,
            "text": f"scene {index} " + "mo ta noi dung canh " * repeats,
        }
        for index in range(count)
    ]


def test_fit_agy_context_leaves_small_context_untouched(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    context = {"movie_title": "Example Movie", "scenes": [{"index": 0, "text": "ngan"}]}
    original = json.dumps(context, ensure_ascii=False)

    fitted, trimmed = pipeline_mod._fit_agy_context(context, 26000)

    assert fitted is context
    assert trimmed == 0
    assert json.dumps(fitted, ensure_ascii=False) == original
    assert "agy prompt trimmed" not in capsys.readouterr().out


def test_fit_agy_context_trims_scenes_before_script_sections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    context = {
        "movie_title": "Example Movie",
        "language": "vi",
        "target_minutes": 10,
        "research": {"brief": "research brief giu nguyen"},
        "scenes": _fit_context_scenes(),
        "script_sections": [
            {
                "title": f"Section {index}",
                "budget_minutes": 1.5,
                "narration": "ke chuyen " * 100,
                "duration_seconds": 90,
            }
            for index in range(6)
        ],
    }
    original = json.dumps(context, ensure_ascii=False)
    budget = len(original) - 8000

    fitted, trimmed = pipeline_mod._fit_agy_context(context, budget)

    assert len(json.dumps(fitted, ensure_ascii=False)) <= budget
    assert trimmed > 0
    # Tier A alone absorbed the cut: scenes keep every index, script and research stay whole.
    assert len(fitted["scenes"]) == 47
    assert [scene["index"] for scene in fitted["scenes"]] == [scene["index"] for scene in context["scenes"]]
    assert all(isinstance(scene["start_seconds"], float) for scene in fitted["scenes"])
    assert fitted["script_sections"] == context["script_sections"]
    assert fitted["research"] == context["research"]
    assert json.dumps(context, ensure_ascii=False) == original

    import re

    line = re.search(r"agy prompt trimmed: (\d+) -> (\d+) chars \((\d+) strings\)", capsys.readouterr().out)
    assert line is not None
    assert int(line.group(1)) == len(original)
    assert int(line.group(2)) == len(json.dumps(fitted, ensure_ascii=False))
    assert int(line.group(3)) == trimmed


def test_fit_agy_context_script_sections_fallback_to_lower_floor() -> None:
    import movie_review_factory.pipeline as pipeline_mod

    context = {
        "movie_title": "Example Movie",
        "script_sections": [
            {
                "title": "Section 0",
                "budget_minutes": 1.5,
                "narration": "x" * 4000,
                "duration_seconds": 90,
            }
        ],
        "scene_candidates": [],
    }
    original = json.dumps(context, ensure_ascii=False)

    fitted, trimmed = pipeline_mod._fit_agy_context(context, len(original) - 3600)
    assert len(json.dumps(fitted, ensure_ascii=False)) <= len(original) - 3600
    assert trimmed == 1
    assert fitted["script_sections"][0]["narration"] == "x" * 400
    assert list(fitted["script_sections"][0]) == ["title", "budget_minutes", "narration", "duration_seconds"]
    assert fitted["script_sections"][0]["title"] == "Section 0"
    assert fitted["script_sections"][0]["budget_minutes"] == 1.5
    assert fitted["script_sections"][0]["duration_seconds"] == 90
    assert fitted["scene_candidates"] == []
    assert fitted["movie_title"] == "Example Movie"

    deeper, deeper_trimmed = pipeline_mod._fit_agy_context(context, len(original) - 3800)
    # One floor relaxation (400 -> 300), then the loop stops even though it is still over budget.
    assert deeper_trimmed == 1
    assert deeper["script_sections"][0]["narration"] == "x" * 300
    assert len(json.dumps(deeper, ensure_ascii=False)) > len(original) - 3800
    assert context["script_sections"][0]["narration"] == "x" * 4000


def test_agy_prompt_budget_env_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    monkeypatch.setenv("MRF_AGY_PROMPT_MAX", "12000")
    assert pipeline_mod._agy_prompt_max() == 12000
    monkeypatch.setenv("MRF_AGY_PROMPT_MAX", "not-a-number")
    assert pipeline_mod._agy_prompt_max() == 26000
    monkeypatch.setenv("MRF_AGY_PROMPT_MAX", "1500")
    assert pipeline_mod._agy_prompt_max() == 26000
    monkeypatch.delenv("MRF_AGY_PROMPT_MAX")
    assert pipeline_mod._agy_prompt_max() == 26000

    monkeypatch.setenv("MRF_AGY_PROMPT_MAX", "8000")
    captured: dict = {}

    def fake_agent(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"sections": [], "notes": ""}

    monkeypatch.setattr(pipeline_mod, "run_agy_json", fake_agent)
    create_job(
        tmp_path,
        JobConfig(
            job_id="agy-budget",
            language="vi",
            target_minutes=5,
            movie_title="Example Movie",
            content_agent="agy",
        ),
    )
    context = {"movie_title": "Example Movie", "scenes": _fit_context_scenes(count=30, repeats=17)}
    untrimmed = len(json.dumps(context, ensure_ascii=False))
    pipeline_mod._run_reasoning_agent(
        root=tmp_path,
        manifest=pipeline_mod.load_manifest(tmp_path),
        stage="outline",
        instruction="outline instruction",
        context=context,
        schema={},
    )

    prompt = str(captured["prompt"])
    assert len(prompt) <= 8000
    assert len(prompt) < untrimmed


def test_claude_reasoning_agent_keeps_prompt_untrimmed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import movie_review_factory.pipeline as pipeline_mod

    captured: dict = {}

    def fake_agent(**kwargs: object) -> dict:
        captured.update(kwargs)
        return {"brief": "claude", "facts": [], "sources": [], "uncertainties": []}

    monkeypatch.setattr(pipeline_mod, "run_claude_json", fake_agent)
    monkeypatch.setenv("MRF_AGY_PROMPT_MAX", "4000")
    create_job(
        tmp_path,
        JobConfig(
            job_id="claude-budget",
            language="vi",
            target_minutes=5,
            movie_title="Example Movie",
            content_agent="claude",
        ),
    )
    context = {
        "movie_title": "Example Movie",
        "scenes": _fit_context_scenes(),
        "script_sections": [
            {"title": "Section 0", "budget_minutes": 1.5, "narration": "ke chuyen " * 100}
        ],
    }
    pipeline_mod._run_reasoning_agent(
        root=tmp_path,
        manifest=pipeline_mod.load_manifest(tmp_path),
        stage="scene_plan",
        instruction="scene plan instruction",
        context=context,
        schema={},
    )

    prompt = str(captured["prompt"])
    assert prompt.endswith(json.dumps(context, ensure_ascii=False))
    assert len(prompt) > 4000
    assert "agy prompt trimmed" not in capsys.readouterr().out


# --- roadmap #14: run_index (source-only background indexing) ----------------


def _seed_indexable_job(tmp_path: Path, job_id: str = "idx", content_agent: str = "scaffold") -> Path:
    """Create a job with a fake source and a pre-seeded (ready) transcript so
    run_index can build the scene index deterministically without whisper."""
    import movie_review_factory.pipeline as pipeline

    source = tmp_path / "source.mp4"
    source.write_bytes(b"fake-video-bytes")
    pipeline.create_job(
        tmp_path,
        JobConfig(job_id=job_id, source_video=source, content_agent=content_agent),
    )
    (tmp_path / "transcript.json").write_text(
        json.dumps({"segments": [
            {"start_seconds": 0.0, "end_seconds": 2.0, "text": "a mysterious lighthouse"},
            {"start_seconds": 2.0, "end_seconds": 4.0, "text": "waves crashing on rocks"},
        ]}),
        encoding="utf-8",
    )
    manifest = pipeline.load_manifest(tmp_path)
    # Mark ingest+transcript ready so run_index reaches the scenes stage without
    # a real video (ingest/whisper are covered by their own tests); run_index's
    # own orchestration is what these tests exercise.
    manifest.stage("ingest").mark("ready", "seeded ingest")
    manifest.stage("transcript").mark("ready", "seeded transcript")
    pipeline.save_manifest(tmp_path, manifest)
    return source


def test_run_index_builds_media_index_without_running_content(tmp_path: Path, monkeypatch) -> None:
    import movie_review_factory.pipeline as pipeline

    _seed_indexable_job(tmp_path)
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: 10.0)

    summary = pipeline.run_index(tmp_path)

    assert summary["stages"]["scenes"] == "ready"
    assert summary["stages"]["scene_memory"] == "skipped"  # scaffold: no AGY memory
    assert summary["scene_memory"] == {}
    assert summary["cancelled"] is False
    assert summary["media_index"] is True
    assert (tmp_path / "media_index.sqlite3").exists()
    assert (tmp_path / "scenes.json").exists()

    # Content stages must NOT run: they stay pending after indexing only.
    by_stage = {s["stage"]: s["status"] for s in pipeline.job_status(tmp_path)["stages"]}
    assert by_stage["research"] == "pending"
    assert by_stage["outline"] == "pending"
    assert by_stage["script"] == "pending"
    assert by_stage["scene_plan"] == "pending"


def test_run_index_reports_bounded_progress_and_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    import movie_review_factory.pipeline as pipeline

    _seed_indexable_job(tmp_path)
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: 10.0)

    events: list[dict] = []
    pipeline.run_index(tmp_path, progress=lambda ev: events.append(dict(ev)))

    stages = [e["stage"] for e in events]
    assert "scenes" in stages
    assert stages[-1] == "scene_memory"
    assert all(e["total"] == len(pipeline.INDEX_STAGES) + 1 for e in events)
    assert all(1 <= e["index"] <= len(pipeline.INDEX_STAGES) + 1 for e in events)

    # A second run is a no-op for already-ready stages (idempotent).
    again = pipeline.run_index(tmp_path)
    assert again["stages"]["scenes"] == "ready"
    assert again["cancelled"] is False


def test_run_index_honours_cancellation(tmp_path: Path, monkeypatch) -> None:
    import threading

    import movie_review_factory.cancellation as cancellation
    import movie_review_factory.pipeline as pipeline

    _seed_indexable_job(tmp_path)
    monkeypatch.setattr(pipeline, "_probe_duration_seconds", lambda path: 10.0)

    event = threading.Event()
    event.set()  # already cancelled: the first checkpoint must stop the run
    context = cancellation.CancellationContext(event)
    with cancellation.cancellation_scope(context):
        summary = pipeline.run_index(tmp_path)

    assert summary["cancelled"] is True


def test_scene_memory_accepts_index_built_from_watermark_clean_source(tmp_path: Path) -> None:
    import movie_review_factory.pipeline as pipeline

    source = tmp_path / "source.mp4"
    clean = tmp_path / "source_clean.mp4"
    source.write_bytes(b"raw")
    clean.write_bytes(b"clean")
    cfg = JobConfig(job_id="clean-job", source_video=str(source), content_agent="agy")

    for indexed in (clean, source):
        summary = pipeline.index_scene_memory(tmp_path, cfg, {"source_video": str(indexed)}, [])
        assert isinstance(summary, dict)
    with pytest.raises(ValueError, match="different source video"):
        pipeline.index_scene_memory(tmp_path, cfg, {"source_video": str(tmp_path / "other.mp4")}, [])
