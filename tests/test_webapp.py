"""Tests for the local web UI/API service, HTTP layer, and new CLI commands.

Every test operates in tmp_path - none of them touch the repository's
jobs/smoke-test job, whose approval gate must stay untouched.
"""

import importlib
import io
import json
import os
import threading
import time
import types
import urllib.error
from datetime import datetime, timezone
import urllib.request
from pathlib import Path

import pytest
from typer.testing import CliRunner

import movie_review_factory.pipeline as pipeline
import movie_review_factory.semantic_search as semantic_search
import movie_review_factory.webapp as webapp_mod
from movie_review_factory.cli import app
from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import CONTENT_AGENT_MODES, JobConfig, VisualObservation
from movie_review_factory.webapp import JobsService, create_server

runner = CliRunner()


def _job_with_metadata(jobs_root: Path, job_id: str = "demo") -> Path:
    """Create a media-less job and run it up to the metadata draft."""
    root = jobs_root / job_id
    pipeline.create_job(root, JobConfig(job_id=job_id))
    pipeline.run_job(root, until="metadata")
    return root


def test_status_exposes_tts_voice_info(tmp_path):
    root = tmp_path / "demo"
    pipeline.create_job(root, JobConfig(job_id="demo"))
    (root / "voice.json").write_text(json.dumps({
        "engine": "elevenlabs",
        "voice": "21m00Tcm4TlvDq8ikWAM",
        "timing_mode": "estimated_word_timing",
        "word_boundaries": [{"offset": 0, "duration": 1, "text": "x"}],
    }), encoding="utf-8")
    voice = JobsService(tmp_path).status("demo")["voice"]
    assert voice == {
        "engine": "elevenlabs",
        "voice": "21m00Tcm4TlvDq8ikWAM",
        "timing_mode": "estimated_word_timing",
    }


def test_status_voice_is_none_without_voice_json(tmp_path):
    root = tmp_path / "demo"
    pipeline.create_job(root, JobConfig(job_id="demo"))
    assert JobsService(tmp_path).status("demo")["voice"] is None


def test_check_tts_connection_edge_needs_no_key(tmp_path):
    result = JobsService(tmp_path).check_tts_connection("edge")
    assert result["provider"] == "edge"
    assert result["ok"] is True


def test_check_tts_connection_missing_key_is_not_ok(monkeypatch, tmp_path):
    monkeypatch.delenv("MRF_ELEVENLABS_API_KEY", raising=False)
    result = JobsService(tmp_path).check_tts_connection("elevenlabs")
    assert result["ok"] is False
    assert "MRF_ELEVENLABS_API_KEY" in result["detail"]


def test_check_tts_connection_reads_env_key_and_uses_injected_transport(monkeypatch, tmp_path):
    monkeypatch.setenv("MRF_ELEVENLABS_API_KEY", "secret")
    seen = {}

    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        seen["key"] = (headers or {}).get("xi-api-key")
        return 200, b'{"character_count": 0, "character_limit": 5000}'

    result = JobsService(tmp_path).check_tts_connection("elevenlabs", http_request=fake_request)
    assert result["ok"] is True
    assert seen["key"] == "secret"


def test_license_status_and_activate_via_service(monkeypatch, tmp_path):
    from movie_review_factory import licensing

    private, public = licensing.generate_keypair()
    monkeypatch.setattr(licensing, "LICENSE_PUBLIC_KEY_B64", licensing.b64encode(public))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv(licensing.LICENSE_ENV, raising=False)
    svc = JobsService(tmp_path)
    assert svc.license_status()["ok"] is False
    fingerprint = svc.license_status()["machine"]
    payload = licensing.build_payload(machine=fingerprint, name="Svc")
    key = licensing.encode_license(payload, licensing.ed25519_sign(private, licensing.payload_bytes(payload)))
    assert svc.activate_license(key)["ok"] is True
    assert svc.license_status()["ok"] is True


def test_require_license_gate_blocks_then_activation_opens(monkeypatch, tmp_path):
    from movie_review_factory import licensing

    private, public = licensing.generate_keypair()
    monkeypatch.setattr(licensing, "LICENSE_PUBLIC_KEY_B64", licensing.b64encode(public))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv(licensing.LICENSE_ENV, raising=False)

    server = create_server("127.0.0.1", 0, tmp_path, require_license=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert "Cần kích hoạt" in resp.read().decode("utf-8")
        with pytest.raises(urllib.error.HTTPError) as blocked:
            urllib.request.urlopen(base + "/api/jobs", timeout=5)
        assert blocked.value.code == 402
        fingerprint = licensing.machine_fingerprint()
        payload = licensing.build_payload(machine=fingerprint, name="UI")
        key = licensing.encode_license(payload, licensing.ed25519_sign(private, licensing.payload_bytes(payload)))
        activate = urllib.request.Request(
            base + "/api/license/activate", method="POST",
            data=json.dumps({"license": key}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(activate, timeout=5) as resp:
            assert json.loads(resp.read())["ok"] is True
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as resp:
            assert "jobs" in json.loads(resp.read())
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert "Bảng điều khiển" in resp.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()


def test_module_main_entry_exposes_cli_app():
    import movie_review_factory.__main__ as entry
    from movie_review_factory.cli import app as cli_app

    assert callable(entry.main)
    assert entry.app is cli_app


def test_start_batch_runs_sequentially_and_is_fault_tolerant(tmp_path):
    svc = JobsService(tmp_path)

    def fake_process(url, index):
        if "boom" in url:
            raise RuntimeError("nguồn hỏng")
        return f"job-{index}"

    svc.start_batch(
        {"links": "https://ok/a\nhttps://boom/b\nhttps://ok/c", "confirm_rights": True},
        process=fake_process,
    )
    deadline = time.time() + 10
    while time.time() < deadline and svc.batch_status()["running"]:
        time.sleep(0.05)
    status = svc.batch_status()
    assert status["running"] is False
    assert status["counts"].get("ready") == 2
    assert status["counts"].get("failed") == 1
    failed = [item for item in status["items"] if item["status"] == "failed"][0]
    assert "boom" in failed["url"] and failed["error"]


def test_start_batch_requires_rights_confirmation(tmp_path):
    from movie_review_factory import link_download

    svc = JobsService(tmp_path)
    with pytest.raises(link_download.RightsConfirmationRequired):
        svc.start_batch({"links": "https://ok/a", "confirm_rights": False})
    assert svc.batch_status()["running"] is False


def _job_with_thumbnails(jobs_root: Path, job_id: str = "thumbs") -> Path:
    root = jobs_root / job_id
    pipeline.create_job(root, JobConfig(job_id=job_id))
    candidates = []
    for index in range(1, 4):
        name = f"thumbnail-{index}.jpg"
        (root / name).write_bytes(f"candidate-{index}".encode())
        candidates.append({
            "index": index,
            "file": name,
            "source_seconds": float(index * 10),
            "width": 1280,
            "height": 720,
        })
    (root / "thumbnail.jpg").write_bytes(b"candidate-2")
    (root / "thumbnails.json").write_text(
        json.dumps({
            "job_id": job_id,
            "candidate_count": 3,
            "candidates": candidates,
            "primary_thumbnail": "thumbnail.jpg",
            "primary_candidate": "thumbnail-2.jpg",
        }),
        encoding="utf-8",
    )
    manifest = pipeline.load_manifest(root)
    manifest.stage("thumbnail").mark("ready", "generated 3 thumbnail candidates")
    pipeline.save_manifest(root, manifest)
    return root


def _job_with_media_index(jobs_root: Path, job_id: str = "media") -> Path:
    root = jobs_root / job_id
    source = root / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"source-video")
    pipeline.create_job(root, JobConfig(job_id=job_id, source_video=source))
    (root / "transcript.json").write_text(json.dumps({"segments": [{"start_seconds": 1.0, "end_seconds": 2.0, "text": "mysterious lighthouse"}]}), encoding="utf-8")
    original_probe = pipeline._probe_duration_seconds
    pipeline._probe_duration_seconds = lambda path: 10.0
    try:
        pipeline._scenes(root, pipeline.load_manifest(root))
    finally:
        pipeline._probe_duration_seconds = original_probe
    return root


def _mark_verified_render(root: Path, *, thumbnail_ready: bool = True) -> None:
    """Provide the publish gate with a verified final render without running media."""
    (root / "final.mp4").write_bytes(b"verified-video")
    (root / "qa.json").write_text(
        json.dumps({"passed": True}, indent=2),
        encoding="utf-8",
    )
    manifest = pipeline.load_manifest(root)
    manifest.stage("qa").mark("ready", "qa passed")
    if thumbnail_ready:
        manifest.stage("thumbnail").mark("ready", "thumbnail ready")
    pipeline.save_manifest(root, manifest)


# --- JobsService ------------------------------------------------------------


def test_create_job_status_is_localized(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    status = svc.create_job({"job_id": "demo", "language": "vi"})
    assert status["job_id"] == "demo"
    by_stage = {s["stage"]: s for s in status["stages"]}
    assert by_stage["publish"]["stage_label"] == "Xuất bản"
    assert by_stage["ingest"]["status_label"] == "Chờ xử lý"
    assert status["approvals"]["metadata_present"] is False
    assert status["has_media_index"] is False


def test_handoff_api_rejects_unapproved_job_with_existing_video(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "review"})
    (tmp_path / "review" / "final.mp4").write_bytes(b"old-video")
    with pytest.raises(ValueError, match="handoff blocked"):
        svc.build_handoff("review")
    assert not (tmp_path / "review" / "review-handoff.zip").exists()


def test_webapp_uses_model_content_agent_modes(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    for mode in CONTENT_AGENT_MODES:
        svc.create_job({"job_id": f"mode-{mode}", "content_agent": mode})
        assert pipeline.load_manifest(tmp_path / f"mode-{mode}").config.content_agent == mode
    with pytest.raises(ValueError, match="content_agent"):
        svc.create_job({"job_id": "mode-invalid", "content_agent": "invalid"})


def test_create_job_persists_movie_title_and_content_agent(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({
        "job_id": "agent-job",
        "movie_title": "Example Movie",
        "content_agent": "claude",
    })
    manifest = pipeline.load_manifest(tmp_path / "agent-job")
    assert manifest.config.movie_title == "Example Movie"
    assert manifest.config.content_agent == "claude"


def test_list_jobs(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "alpha"})
    svc.create_job({"job_id": "beta"})
    ids = {j["job_id"] for j in svc.list_jobs()}
    assert ids == {"alpha", "beta"}


def test_duplicate_job_rejected(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    with pytest.raises(FileExistsError):
        svc.create_job({"job_id": "demo"})


def test_invalid_job_id_rejected(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    for bad in ("../evil", "..", ".", "a/b", "space name"):
        with pytest.raises(ValueError):
            svc.create_job({"job_id": bad})


def test_artifact_path_blocks_traversal(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_metadata(tmp_path, "demo")
    # A legitimate artifact resolves.
    assert svc.artifact_path("demo", "research.json").name == "research.json"
    # Traversal / bad names are refused.
    for bad in ("..", "../manifest.json", "a/b", "..%2f"):
        with pytest.raises((ValueError, FileNotFoundError)):
            svc.artifact_path("demo", bad)


def test_status_remains_readable_while_run_persists_manifest(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    svc.start_run("demo")

    deadline = time.time() + 10
    while time.time() < deadline:
        # Every concurrent read must succeed; discard a potentially stale snapshot once the worker has finished.
        if not svc.status("demo")["running"]:
            break
        time.sleep(0.01)
    else:
        pytest.fail("background run did not finish")

    status = svc.status("demo")
    assert status["running"] is False
    # The run to script must produce the script draft while status stays readable.
    assert status["approvals"]["script_present"] is True


def test_import_does_not_auto_index_or_advance(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    data = b"\x00" * 64
    svc.import_video("demo", "source.mp4", len(data), io.BytesIO(data))
    # Loading the source must not auto-start indexing or any downstream stage.
    assert svc.index_state("demo") is None
    status = svc.status("demo")
    assert status["running"] is False
    stages = {s["stage"]: s["status"] for s in status["stages"]}
    for stage in ("research", "transcript", "scenes", "outline", "script", "scene_plan"):
        assert stages[stage] == "pending", (stage, stages[stage])


def test_run_stops_at_script_until_approved(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})

    # The run must stop at the script for review - never touching the scene plan
    # (AGY), metadata, or thumbnail, and never advancing publish.
    assert svc.start_run("demo")["until"] == "script"
    deadline = time.time() + 10
    while time.time() < deadline and svc.status("demo")["running"]:
        time.sleep(0.1)
    status = svc.status("demo")
    assert status["running"] is False
    assert status["approvals"]["script_present"] is True
    assert status["approvals"]["metadata_present"] is False
    stages = {s["stage"]: s["status"] for s in status["stages"]}
    assert stages["script"] == "ready"
    assert stages["scene_plan"] == "pending"
    assert stages["metadata"] == "pending"
    assert stages["thumbnail"] == "pending"
    assert stages["publish"] == "pending"
    assert not (tmp_path / "demo" / "publish_record.json").exists()


def test_run_targets_video_after_script_approved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    # An approved script must make the next run target the finished video.
    (tmp_path / "demo" / "script.json").write_text(
        json.dumps({"approved": True, "sections": [{"narration": "x"}]}), encoding="utf-8"
    )
    captured: dict = {}
    monkeypatch.setattr(
        pipeline, "run_job",
        lambda root, *, until=None, force=False: captured.setdefault("until", until),
    )
    assert svc.start_run("demo")["until"] == "thumbnail"
    deadline = time.time() + 5
    while time.time() < deadline and svc.status("demo")["running"]:
        time.sleep(0.02)
    assert captured["until"] == "thumbnail"


def test_reindex_transcript_resets_transcript_and_scenes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    root = tmp_path / "demo"
    # Give the job a source video and mark transcript/scenes/outline done so we
    # can prove the re-index flips exactly the first two back to pending.
    manifest = pipeline.load_manifest(root)
    manifest.config.source_video = root / "source.mp4"
    for stage in manifest.stages:
        if stage.stage in ("transcript", "scenes", "outline"):
            stage.status = "ready"
    pipeline.save_manifest(root, manifest)

    captured: dict = {}
    monkeypatch.setattr(
        pipeline, "run_job",
        lambda root, *, until=None, force=False: captured.setdefault("until", until),
    )
    assert svc.reindex_transcript("demo")["started"] is True
    deadline = time.time() + 5
    while time.time() < deadline and svc.status("demo").get("reindex", {}).get("running"):
        time.sleep(0.02)
    # It re-runs only up to the scene index, never toward the finished video.
    assert captured["until"] == "scenes"
    after = {stage.stage: stage.status for stage in pipeline.load_manifest(root).stages}
    assert after["transcript"] == "pending"
    assert after["scenes"] == "pending"
    # Downstream stages (already produced) are left untouched.
    assert after["outline"] == "ready"


def test_reindex_transcript_requires_source_video(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "demo"})
    with pytest.raises(ValueError):
        svc.reindex_transcript("demo")


# --- approval + publishing gate ---------------------------------------------


def test_publish_gate_requires_confirm(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_metadata(tmp_path, "demo")
    result = svc.prepare_publish("demo", confirm=False)
    assert result["confirmed"] is False
    assert result["published"] is False
    assert not (tmp_path / "demo" / "publish_record.json").exists()


def test_publish_blocked_until_approved_then_succeeds(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_metadata(tmp_path, "demo")

    # Nothing approved yet: confirmed publish stays blocked, writes no record.
    blocked = svc.prepare_publish("demo", confirm=True)
    assert blocked["published"] is False
    assert "Chưa thể xuất bản" in blocked["message_vi"]
    assert not (root / "publish_record.json").exists()

    # Approve the script (human step) and the metadata (explicit endpoint).
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    script["approved"] = True
    (root / "script.json").write_text(json.dumps(script), encoding="utf-8")
    approved = svc.approve_metadata("demo")
    assert approved["metadata"]["approved"] is True
    _mark_verified_render(root)

    ok = svc.prepare_publish("demo", confirm=True)
    assert ok["published"] is True
    assert (root / "publish_record.json").exists()


def test_update_metadata_clears_approval(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_metadata(tmp_path, "demo")
    svc.approve_metadata("demo")
    assert svc.status("demo")["approvals"]["metadata_approved"] is True
    svc.update_metadata("demo", {"title": "Tiêu đề mới"})
    status = svc.status("demo")
    assert status["approvals"]["metadata_approved"] is False
    assert svc.get_metadata("demo")["metadata"]["title"] == "Tiêu đề mới"


def test_metadata_edit_invalidates_ready_publish(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_metadata(tmp_path, "demo")
    svc.approve_script("demo")
    svc.approve_metadata("demo")
    manifest = pipeline.load_manifest(root)
    thumb = root / "thumbnail.jpg"
    thumb.write_bytes(b"jpeg")
    manifest.stage("thumbnail").mark(
        "ready", "generated 1 thumbnail candidates",
        [pipeline.Artifact(name=thumb.name, path=thumb, status="ready")],
    )
    pipeline.save_manifest(root, manifest)
    _mark_verified_render(root, thumbnail_ready=False)
    assert svc.prepare_publish("demo", confirm=True)["published"] is True
    assert (root / "publish_record.json").exists()

    svc.update_metadata("demo", {"title": "Tiêu đề sau publish"})

    manifest = pipeline.load_manifest(root)
    assert manifest.stage("metadata").status == "ready"
    assert manifest.stage("thumbnail").status == "pending"
    assert manifest.stage("publish").status == "pending"
    assert not thumb.exists()
    assert not (root / "publish_record.json").exists()


def test_script_get_update_approve(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_metadata(tmp_path, "demo")
    result = svc.get_script("demo")
    assert result["present"] is True
    assert result["script"]["approved"] is False

    updated = svc.update_script("demo", {"notes": "test note"})
    assert updated["approved"] is False
    assert updated["script"]["notes"] == "test note"

    approved = svc.approve_script("demo")
    assert approved["approved"] is True
    persisted = json.loads((root / "script.json").read_text(encoding="utf-8"))
    assert persisted["approved"] is True
    assert "**approved: true**" in (root / "script.md").read_text(encoding="utf-8")

    svc.update_script("demo", {"notes": "changed"})
    assert svc.status("demo")["approvals"]["script_approved"] is False


def test_script_edit_invalidates_all_derived_stages_and_artifacts(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_metadata(tmp_path, "demo")
    (root / "final.mp4").write_bytes(b"stale")
    (root / "publish_record.json").write_text("{}", encoding="utf-8")

    svc.update_script("demo", {"notes": "new script revision"})

    manifest = pipeline.load_manifest(root)
    assert manifest.stage("script").status == "ready"
    script_index = pipeline.STAGES.index("script")
    assert all(stage.status == "pending" for stage in manifest.stages[script_index + 1:])
    for name in (
        "scene_plan.json", "voice.json", "narration.mp3", "alignment.json",
        "aligned.srt", "render.json", "final.mp4", "qa.json",
        "youtube_metadata.json", "publish_record.json",
    ):
        assert not (root / name).exists()
    assert "new script revision" not in (root / "script.md").read_text(encoding="utf-8")
    assert "**approved: false**" in (root / "script.md").read_text(encoding="utf-8")


def test_thumbnail_artifacts_are_served_as_images() -> None:
    assert webapp_mod._content_type("thumbnail.jpg") == "image/jpeg"
    assert webapp_mod._artifact_kind("thumbnail.jpg") == "image"


def test_media_explorer_searches_transcript_and_lists_shots(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_media_index(tmp_path)
    listing = svc.media_explorer("media")
    assert listing["present"] is True
    assert listing["media_href"] == "/api/jobs/media/media/source"
    assert listing["shots"][0]["start_seconds"] == 1.0
    assert listing["shots"][0]["thumbnail_href"].endswith("/shots/1/thumbnail")
    search = svc.media_explorer("media", "lighthouse")
    assert [item["text"] for item in search["transcript"]] == ["mysterious lighthouse"]
    assert [item["label"] for item in search["shots"]] == ["mysterious lighthouse"]


def test_media_explorer_searches_persisted_visual_memory(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_media_index(tmp_path)
    with MediaStore(root / "media_index.sqlite3") as store:
        store.replace_visual_observations([
            VisualObservation(
                shot_id=1,
                description="A red car races through a wet tunnel",
                tags=["red car", "tunnel"],
                people=[],
                actions=["racing"],
            )
        ])

    result = svc.media_explorer("media", "tunnel")

    assert [item["id"] for item in result["shots"]] == [1]
    assert result["shots"][0]["visual_description"] == "A red car races through a wet tunnel"
    assert result["shots"][0]["visual_tags"] == ["red car", "tunnel"]
    assert result["visual_count"] == 1


def test_library_search_finds_visual_and_transcript_across_projects(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    first = _job_with_media_index(tmp_path, "first")
    second = _job_with_media_index(tmp_path, "second")
    with MediaStore(first / "media_index.sqlite3") as store:
        store.replace_visual_observations([
            VisualObservation(
                shot_id=1,
                description="A red car races through a tunnel",
                tags=["red car", "tunnel"],
                actions=["racing"],
            )
        ])
    with MediaStore(second / "media_index.sqlite3") as store:
        store.replace_visual_observations([
            VisualObservation(
                shot_id=1,
                description="A woman walks beside a lighthouse",
                tags=["lighthouse"],
                people=["woman"],
                actions=["walking"],
            )
        ])

    visual = svc.library_search("tunnel")
    transcript = svc.library_search("lighthouse")

    assert [(item["job_id"], item["kind"]) for item in visual["results"]] == [
        ("first", "visual")
    ]
    assert {item["job_id"] for item in transcript["results"]} == {"first", "second"}
    assert {item["kind"] for item in transcript["results"]} == {"visual", "transcript"}


def test_media_explorer_uses_true_semantic_vector_hits_and_person_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_media_index(tmp_path)
    with MediaStore(root / "media_index.sqlite3") as store:
        shots = store.list_shots(1)
        assert len(shots) >= 2 and shots[0].id is not None and shots[1].id is not None
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "red coat", "source": "agy"}],
            [{"label": "Person 1", "shot_id": int(shots[0].id), "confidence": 0.95, "evidence": "same red coat"}],
        )
        _, car_blob = semantic_search._as_blob([1.0, 0.0])
        _, other_blob = semantic_search._as_blob([0.0, 1.0])
        store.replace_shot_embeddings([
            (int(shots[0].id), "red vehicle in storm Person 1 red coat", "test-model", 2, car_blob),
            (int(shots[1].id), "quiet empty room", "test-model", 2, other_blob),
        ])

    _, query_blob = semantic_search._as_blob([1.0, 0.0])
    monkeypatch.setattr(semantic_search, "model_name", lambda: "test-model")
    monkeypatch.setattr(semantic_search, "embed_query", lambda query: (2, query_blob))

    result = svc.media_explorer("media", "automobile during bad weather")

    assert result["semantic_mode"] == "fastembed"
    assert result["shots"][0]["id"] == 1
    assert result["shots"][0]["semantic_score"] == pytest.approx(1.0)
    assert result["shots"][0]["person_tracks"] == ["Person 1"]


def test_scene_reindex_and_invalidation_remove_derived_media_caches(tmp_path: Path) -> None:
    root = _job_with_media_index(tmp_path)
    cached = [root / "shot-1.jpg", root / "highlight-h-1.mp4", root / "transcript.vtt"]
    for path in cached:
        path.write_bytes(b"old")
    original_probe = pipeline._probe_duration_seconds
    pipeline._probe_duration_seconds = lambda path: 10.0
    try:
        pipeline._scenes(root, pipeline.load_manifest(root))
    finally:
        pipeline._probe_duration_seconds = original_probe
    assert all(not path.exists() for path in cached)
    for path in cached:
        path.write_bytes(b"old")
    pipeline.invalidate_downstream(root, "transcript")
    assert all(not path.exists() for path in cached)


def test_shot_thumbnail_is_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)
    _job_with_media_index(tmp_path)
    calls = []
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: "ffmpeg.exe")
    def fake_run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_bytes(b"jpeg")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    first = svc.shot_thumbnail_path("media", 1)
    second = svc.shot_thumbnail_path("media", 1)
    assert first == second
    assert first.read_bytes() == b"jpeg"
    assert len(calls) == 1


def test_media_explorer_browses_full_transcript_without_query(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    _job_with_media_index(tmp_path)
    listing = svc.media_explorer("media")
    # With no query the whole transcript timeline is browsable (clipto-style).
    assert [item["text"] for item in listing["transcript"]] == ["mysterious lighthouse"]
    assert listing["transcript_vtt_href"] == "/api/jobs/media/transcript.vtt"


def test_transcript_export_renders_webvtt_with_speaker(tmp_path: Path) -> None:
    from movie_review_factory.media_store import MediaStore
    from movie_review_factory.models import MediaAsset, Shot, TranscriptSegment

    svc = JobsService(tmp_path)
    root = tmp_path / "vtt"
    source = root / "source.mp4"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"video")
    pipeline.create_job(root, JobConfig(job_id="vtt", source_video=source))
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=source, duration_seconds=30),
            [Shot(media_asset_id=1, start_seconds=0, end_seconds=5, label="Opening")],
            [
                TranscriptSegment(media_asset_id=1, start_seconds=1.5, end_seconds=3.25, text="Xin chào", speaker="Người dẫn"),
                TranscriptSegment(media_asset_id=1, start_seconds=3.25, end_seconds=5.0, text="Hẹn gặp lại"),
            ],
        )
    path = svc.transcript_export_path("vtt")
    assert path.name == "transcript.vtt"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("WEBVTT\n")
    assert "00:00:01.500 --> 00:00:03.250" in text
    assert "<v Người dẫn>Xin chào" in text
    assert "Hẹn gặp lại" in text


def test_vtt_is_served_as_subtitle_type() -> None:
    assert webapp_mod._content_type("transcript.vtt") == "text/vtt; charset=utf-8"
    assert webapp_mod._artifact_kind("transcript.vtt") == "subtitle"


def test_thumbnail_service_selects_primary_and_invalidates_publish(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    root = _job_with_thumbnails(tmp_path, "thumbs")
    (root / "publish_record.json").write_text("{}", encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("publish").mark("ready", "publish record written")
    pipeline.save_manifest(root, manifest)

    current = svc.get_thumbnails("thumbs")
    assert current["present"] is True
    assert current["thumbnails"]["primary_candidate"] == "thumbnail-2.jpg"

    result = svc.select_thumbnail("thumbs", "thumbnail-1.jpg")
    assert result["selected"] == "thumbnail-1.jpg"
    assert (root / "thumbnail.jpg").read_bytes() == b"candidate-1"
    assert pipeline.load_manifest(root).stage("publish").status == "pending"
    assert not (root / "publish_record.json").exists()


# --- HTTP layer -------------------------------------------------------------


def _serve(jobs_root: Path):
    server = create_server("127.0.0.1", 0, jobs_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, base


def _get_json(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_json(url: str, payload: dict):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        body = response.read().decode("utf-8")
        return response.status, (json.loads(body) if body else {})


def test_http_index_and_job_lifecycle(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert "Xưởng Review Phim" in html
        assert "Chọn ảnh bìa" in html
        assert "Trình khám phá tư liệu" in html
        assert 'class="media-workspace"' in html
        assert 'class="sticky-player"' in html
        assert 'role="tablist"' in html
        assert 'data-media-filter="transcript"' in html
        assert 'data-media-filter="scenes"' in html
        assert 'data-media-filter="highlights"' in html
        assert 'aria-label="Trình phát video nguồn"' in html
        assert "prefers-reduced-motion: reduce" in html
        assert "syncActiveTranscript" in html
        assert "manualTranscriptScrollUntil" in html
        assert "event.key === 'Enter' || event.key === ' '" in html
        assert "↓ VTT" in html
        assert 'id="sourceFile"' in html
        assert 'id="sourceRetry"' in html
        assert 'id="exportCard"' in html
        assert 'id="deleteDialog"' in html
        assert 'id="selectAllProjects"' in html
        assert 'id="deleteSelectedBtn"' in html
        assert 'id="librarySearch"' in html
        assert 'id="librarySearchBtn"' in html
        assert 'id="libraryPerson"' in html
        assert 'id="librarySceneType"' in html
        assert 'id="librarySource"' in html
        assert 'id="libraryDateFrom"' in html
        assert 'id="libraryDateTo"' in html
        assert 'id="savedLibrarySearches"' in html
        assert 'id="editorCard"' in html
        assert 'id="autoBrollBtn"' in html
        assert "/api/library-search?" in html
        assert "/timeline" in html
        assert "/regenerate" in html
        assert "Tìm cảnh tương tự" in html
        assert "pendingLibrarySeek" in html
        assert "row.visual_description || row.label" in html
        assert "className = 'project-card" in html

        status_code, created = _post_json(base + "/api/jobs", {"job_id": "demo"})
        assert status_code == 201
        assert created["job_id"] == "demo"

        _, listing = _get_json(base + "/api/jobs")
        assert any(j["job_id"] == "demo" for j in listing["jobs"])

        _, status = _get_json(base + "/api/jobs/demo")
        assert any(s["stage_label"] == "Xuất bản" for s in status["stages"])
    finally:
        server.shutdown()
        server.server_close()


def test_http_library_search_returns_cross_project_visual_matches(tmp_path: Path) -> None:
    first = _job_with_media_index(tmp_path, "first")
    _job_with_media_index(tmp_path, "second")
    with MediaStore(first / "media_index.sqlite3") as store:
        store.replace_visual_observations([
            VisualObservation(
                shot_id=1,
                description="A red car races through a wet tunnel",
                tags=["red car", "tunnel"],
                actions=["racing"],
            )
        ])

    server, base = _serve(tmp_path)
    try:
        _, data = _get_json(base + "/api/library-search?q=tunnel")
        assert data["query"] == "tunnel"
        assert [(item["job_id"], item["kind"]) for item in data["results"]] == [
            ("first", "visual")
        ]
        assert data["results"][0]["tags"] == ["red car", "tunnel"]
    finally:
        server.shutdown()
        server.server_close()


def test_library_filters_alias_story_facets_and_saved_searches(tmp_path: Path) -> None:
    root = _job_with_media_index(tmp_path, "facets")
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_visual_observations([
            VisualObservation(
                shot_id=1,
                description="Person waits beside a red bag in hospital",
                tags=["hospital"],
                actions=["waiting"],
            )
        ])
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "dark hair", "source": "agy"}],
            [{
                "label": "Person 1",
                "shot_id": 1,
                "description": "dark hair",
                "clothing": "red coat",
                "confidence": 0.91,
                "evidence": "visible",
            }],
        )
        store.set_person_alias("Person 1", "Nam")
        store.replace_story_graph(
            [
                {"type": "person", "label": "Person 1", "description": "", "source": "agy"},
                {"type": "location", "label": "Hospital", "description": "", "source": "agy"},
                {"type": "object", "label": "Red bag", "description": "", "source": "agy"},
            ],
            [
                {"type": "person", "label": "Person 1", "shot_id": 1, "confidence": 0.91, "evidence": ""},
                {"type": "location", "label": "Hospital", "shot_id": 1, "confidence": 0.95, "evidence": ""},
                {"type": "object", "label": "Red bag", "shot_id": 1, "confidence": 0.9, "evidence": ""},
            ],
            [],
        )

    svc = JobsService(tmp_path)
    project_date = datetime.fromtimestamp(
        pipeline.manifest_path(root).stat().st_mtime,
        tz=timezone.utc,
    ).date().isoformat()
    result = svc.library_search("", {
        "person": "Nam",
        "action": "waiting",
        "location": "hospital",
        "object": "red bag",
        "kind": "visual",
        "source": "agy",
        "scene_type": "Scene 1",
        "date_from": project_date,
        "date_to": project_date,
        "min_confidence": "0.9",
    })
    assert [(item["job_id"], item["kind"]) for item in result["results"]] == [("facets", "visual")]
    assert result["results"][0]["person_tracks"] == ["Nam"]
    assert result["results"][0]["locations"] == ["Hospital"]
    assert result["results"][0]["source"] == "agy"
    assert result["results"][0]["project_date"] == project_date

    transcript_only = svc.library_search("", {
        "source": "transcript",
        "date_from": project_date,
        "date_to": project_date,
    })
    assert transcript_only["results"]
    assert {item["kind"] for item in transcript_only["results"]} == {"transcript"}

    saved = svc.save_search({
        "name": "Hospital wait",
        "query": 'person:"Nam" location:hospital',
        "kind": "visual",
        "source": "agy",
        "date_from": project_date,
        "date_to": project_date,
    })
    assert saved["search"]["name"] == "Hospital wait"
    assert saved["search"]["source"] == "agy"
    assert saved["search"]["date_from"] == project_date
    assert svc.list_saved_searches()["searches"][0]["query"] == 'person:"Nam" location:hospital'

    with pytest.raises(ValueError, match="valid YYYY-MM-DD"):
        svc.library_search("", {"date_from": "2026-99-99"})


def test_http_saved_search_and_continuity_routes(tmp_path: Path) -> None:
    root = _job_with_media_index(tmp_path, "continuity")
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_person_tracks(
            [{"label": "Person 1", "description": "dark hair", "source": "agy"}],
            [{"label": "Person 1", "shot_id": 1, "confidence": 0.9, "evidence": "visible"}],
        )
    server, base = _serve(tmp_path)
    try:
        status, saved = _post_json(base + "/api/library-searches", {
            "name": "Lighthouse",
            "query": "lighthouse",
        })
        assert status == 201
        _, searches = _get_json(base + "/api/library-searches")
        assert searches["searches"][0]["name"] == "Lighthouse"

        _, tracks = _get_json(base + "/api/jobs/continuity/person-tracks")
        assert tracks["tracks"][0]["label"] == "Person 1"
        _, aliased = _post_json(
            base + "/api/jobs/continuity/person-tracks/Person%201/alias",
            {"alias": "Nam"},
        )
        assert aliased["track"]["alias"] == "Nam"
    finally:
        server.shutdown()
        server.server_close()


def test_http_timeline_editor_routes(tmp_path: Path) -> None:
    root = _job_with_media_index(tmp_path, "editor")
    (root / "script.json").write_text(json.dumps({
        "sections": [{"title": "Opening", "narration": "mysterious lighthouse", "duration_seconds": 10}]
    }), encoding="utf-8")
    (root / "scene_plan.json").write_text(json.dumps({
        "total_seconds": 10,
        "clips": [{
            "section": "Opening",
            "section_index": 1,
            "shot_index": 1,
            "shot_count": 1,
            "start_seconds": 0,
            "duration_seconds": 10,
            "source_clip": {"start_seconds": 0, "end_seconds": 10},
            "notes": "",
        }],
    }), encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("scene_plan").mark("ready", "ready")
    pipeline.save_manifest(root, manifest)

    server, base = _serve(tmp_path)
    try:
        _, current = _get_json(base + "/api/jobs/editor/timeline")
        assert current["clips"][0]["locked"] is False
        _, edited = _post_json(
            base + "/api/jobs/editor/timeline",
            {"action": "lock", "clip_index": 0, "locked": True},
        )
        assert edited["clips"][0]["locked"] is True
        _, regen = _post_json(
            base + "/api/jobs/editor/sections/1/regenerate",
            {"instruction": "lighthouse"},
        )
        assert regen["section_index"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_http_missing_job_returns_404(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _get_json(base + "/api/jobs/nope")
        assert excinfo.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("disconnect", [BrokenPipeError, ConnectionResetError, ConnectionAbortedError])
def test_streaming_stops_cleanly_when_browser_closes_connection(
    tmp_path: Path, disconnect: type[OSError]
) -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock

    video = tmp_path / "video.mp4"
    video.write_bytes(b"x" * 100)
    writer = Mock()
    writer.write.side_effect = disconnect()
    handler = SimpleNamespace(
        headers={}, wfile=writer, send_response=Mock(), send_header=Mock(), end_headers=Mock(),
    )
    webapp_mod.MRFRequestHandler._serve_file(handler, video)
    writer.write.assert_called_once()


def test_http_artifact_supports_range(tmp_path: Path) -> None:
    _job_with_metadata(tmp_path, "demo")
    server, base = _serve(tmp_path)
    try:
        request = urllib.request.Request(
            base + "/api/jobs/demo/artifacts/research.json",
            headers={"Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert len(response.read()) == 4
    finally:
        server.shutdown()
        server.server_close()


def test_http_invalid_range_rejected_before_streaming(tmp_path: Path) -> None:
    _job_with_media_index(tmp_path)
    server, base = _serve(tmp_path)
    try:
        for raw in ("bytes=-1-", "bytes=-5-10", "bytes=0--1", "bytes=-"):
            request = urllib.request.Request(
                base + "/api/jobs/media/media/source", headers={"Range": raw},
            )
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(request, timeout=5)
            assert exc.value.code == 416
            assert exc.value.headers["Content-Range"] == "bytes */12"
        tail = urllib.request.Request(
            base + "/api/jobs/media/media/source", headers={"Range": "bytes=-4"},
        )
        with urllib.request.urlopen(tail, timeout=5) as response:
            assert response.status == 206
            assert response.read() == b"ideo"
    finally:
        server.shutdown()
        server.server_close()


def test_http_media_explorer_and_source_routes(tmp_path: Path) -> None:
    _job_with_media_index(tmp_path)
    server, base = _serve(tmp_path)
    try:
        status, listing = _get_json(base + "/api/jobs/media/media-explorer?q=lighthouse")
        assert status == 200
        assert listing["transcript"][0]["text"] == "mysterious lighthouse"
        with urllib.request.urlopen(base + "/api/jobs/media/media/source", timeout=5) as response:
            assert response.status == 200
            assert response.read() == b"source-video"
    finally:
        server.shutdown()
        server.server_close()


def test_http_transcript_vtt_route(tmp_path: Path) -> None:
    _job_with_media_index(tmp_path)
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/api/jobs/media/transcript.vtt", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "text/vtt; charset=utf-8"
            body = response.read().decode("utf-8")
        assert body.startswith("WEBVTT")
        assert "mysterious lighthouse" in body
    finally:
        server.shutdown()
        server.server_close()


def test_http_thumbnail_get_and_select_routes(tmp_path: Path) -> None:
    root = _job_with_thumbnails(tmp_path, "thumbs")
    server, base = _serve(tmp_path)
    try:
        status, current = _get_json(base + "/api/jobs/thumbs/thumbnails")
        assert status == 200
        assert current["thumbnails"]["primary_candidate"] == "thumbnail-2.jpg"

        status, selected = _post_json(
            base + "/api/jobs/thumbs/thumbnails/select",
            {"candidate": "thumbnail-3.jpg"},
        )
        assert status == 200
        assert selected["selected"] == "thumbnail-3.jpg"
        assert (root / "thumbnail.jpg").read_bytes() == b"candidate-3"

        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post_json(
                base + "/api/jobs/thumbs/thumbnails/select",
                {"candidate": "thumbnail-99.jpg"},
            )
        assert excinfo.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_http_script_routes(tmp_path: Path) -> None:
    _job_with_metadata(tmp_path, "demo")
    server, base = _serve(tmp_path)
    try:
        _, result = _get_json(base + "/api/jobs/demo/script")
        assert result["present"] is True

        _, updated = _post_json(base + "/api/jobs/demo/script", {"notes": "http note"})
        assert updated["approved"] is False

        _, approved = _post_json(base + "/api/jobs/demo/script/approve", {})
        assert approved["approved"] is True
        persisted = json.loads((tmp_path / "demo" / "script.json").read_text(encoding="utf-8"))
        assert persisted["approved"] is True
    finally:
        server.shutdown()
        server.server_close()


# --- new CLI commands -------------------------------------------------------


def test_cli_init_job_accepts_content_agent_options(tmp_path: Path) -> None:
    root = tmp_path / "agent-cli"
    result = runner.invoke(app, [
        "init-job", str(root),
        "--movie-title", "Example Movie",
        "--content-agent", "claude",
    ])
    assert result.exit_code == 0
    manifest = pipeline.load_manifest(root)
    assert manifest.config.movie_title == "Example Movie"
    assert manifest.config.content_agent == "claude"


def test_cli_init_job_enables_external_watermark_detect(tmp_path: Path) -> None:
    root = tmp_path / "wm-cli"
    result = runner.invoke(app, [
        "init-job", str(root),
        "--watermark-detect", "external",
        "--detector-cmd", "python detect.py --in {video} --out {out}",
    ])
    assert result.exit_code == 0, result.output
    wm = pipeline.load_manifest(root).config.watermark_removal
    assert wm.enabled is True
    assert wm.detect is not None
    assert wm.detect.method == "external"
    assert wm.detect.external_cmd == "python detect.py --in {video} --out {out}"


def test_cli_init_job_sets_cheap_watermark_method(tmp_path: Path) -> None:
    root = tmp_path / "wm-blur"
    result = runner.invoke(app, ["init-job", str(root), "--watermark-detect", "color", "--watermark-method", "blur"])
    assert result.exit_code == 0, result.output
    wm = pipeline.load_manifest(root).config.watermark_removal
    assert wm.enabled is True
    assert wm.method == "blur"


def test_cli_init_job_rejects_unknown_watermark_method(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init-job", str(tmp_path / "wm-bad"), "--watermark-method", "magic"])
    assert result.exit_code != 0


def test_cli_status_prints_unicode_on_legacy_console(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    root = tmp_path / "wm-status"
    assert runner.invoke(app, ["init-job", str(root)]).exit_code == 0
    raw = io.BytesIO()
    legacy = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    monkeypatch.setattr("sys.stdout", legacy)
    app(["status", str(root)], standalone_mode=False)
    legacy.flush()
    assert legacy.encoding.lower().replace("-", "") == "utf8"
    print("Xo\u00e1 watermark \u2014 ok")
    legacy.flush()
    assert "Xo\u00e1 watermark \u2014 ok".encode("utf-8") in raw.getvalue()
    assert b"wm-status" in raw.getvalue()


def test_cli_init_job_watermark_detect_defaults_off(tmp_path: Path) -> None:
    root = tmp_path / "wm-off"
    assert runner.invoke(app, ["init-job", str(root)]).exit_code == 0
    wm = pipeline.load_manifest(root).config.watermark_removal
    assert wm.enabled is False
    assert wm.detect is None


def test_cli_init_job_rejects_bad_watermark_detect(tmp_path: Path) -> None:
    result = runner.invoke(app, ["init-job", str(tmp_path / "wm-bad"), "--watermark-detect", "bogus"])
    assert result.exit_code != 0


def test_create_job_enables_external_watermark_detect(tmp_path: Path) -> None:
    service = JobsService(tmp_path)
    service.create_job({
        "job_id": "wm-web",
        "watermark_detect": "external",
        "detector_cmd": "python detect.py --in {video} --out {out}",
    })
    wm = pipeline.load_manifest(tmp_path / "wm-web").config.watermark_removal
    assert wm.enabled is True
    assert wm.detect.method == "external"
    assert wm.detect.external_cmd == "python detect.py --in {video} --out {out}"


def test_create_job_rejects_bad_watermark_detect(tmp_path: Path) -> None:
    service = JobsService(tmp_path)
    with pytest.raises(ValueError):
        service.create_job({"job_id": "wm-web-bad", "watermark_detect": "bogus"})


def _fake_detector_run(cmd, **kwargs):
    """Fake ffmpeg (1-frame extract) + external detector (writes one mask)."""
    if "-frames:v" in cmd:
        Path(cmd[-1]).write_bytes(b"clip")
    else:
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "00000.png").write_bytes(b"mask")
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_cli_probe_detector_reports_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    monkeypatch.setattr("movie_review_factory.mask_detection.subprocess.run", _fake_detector_run)
    monkeypatch.setattr("movie_review_factory.mask_detection.shutil.which", lambda name: "ffmpeg")
    result = runner.invoke(app, [
        "probe-detector", str(video), "--detector-cmd", "detect --in {video} --out {out}",
    ])
    assert result.exit_code == 0, result.output
    assert "ok" in result.output.lower()


def test_cli_probe_detector_without_command_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRF_MASK_DETECTOR_CMD", raising=False)
    monkeypatch.setattr("movie_review_factory.mask_detection.shutil.which", lambda name: "ffmpeg")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    result = runner.invoke(app, ["probe-detector", str(video)])
    assert result.exit_code == 1


def test_service_probe_detector_reports_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    root = tmp_path / "wm-probe"
    pipeline.create_job(root, JobConfig(
        job_id="wm-probe",
        source_video=video,
        watermark_removal={
            "enabled": True,
            "detect": {"method": "external", "external_cmd": "detect --in {video} --out {out}"},
        },
    ))
    monkeypatch.setattr("movie_review_factory.mask_detection.subprocess.run", _fake_detector_run)
    monkeypatch.setattr("movie_review_factory.mask_detection.shutil.which", lambda name: "ffmpeg")
    result = JobsService(tmp_path).probe_detector("wm-probe")
    assert result["ok"] is True
    assert result["masks"] >= 1


def test_service_probe_detector_requires_source(tmp_path: Path) -> None:
    pipeline.create_job(tmp_path / "no-src", JobConfig(job_id="no-src"))
    with pytest.raises(ValueError):
        JobsService(tmp_path).probe_detector("no-src")


def test_cli_approve_script_requires_confirm_and_syncs_markdown(tmp_path: Path) -> None:
    root = _job_with_metadata(tmp_path, "demo")
    script_path = root / "script.json"

    result = runner.invoke(app, ["approve-script", str(root)])
    assert result.exit_code == 2
    assert json.loads(script_path.read_text(encoding="utf-8"))["approved"] is False

    result = runner.invoke(app, ["approve-script", str(root), "--confirm"])
    assert result.exit_code == 0
    assert json.loads(script_path.read_text(encoding="utf-8"))["approved"] is True
    assert "**approved: true**" in (root / "script.md").read_text(encoding="utf-8")


def test_cli_approve_metadata_requires_confirm(tmp_path: Path) -> None:
    root = _job_with_metadata(tmp_path, "demo")
    meta_path = root / "youtube_metadata.json"

    result = runner.invoke(app, ["approve-metadata", str(root)])
    assert result.exit_code == 2
    assert json.loads(meta_path.read_text(encoding="utf-8"))["approved"] is False

    result = runner.invoke(app, ["approve-metadata", str(root), "--confirm"])
    assert result.exit_code == 0
    assert json.loads(meta_path.read_text(encoding="utf-8"))["approved"] is True


def test_cli_publish_requires_confirm_and_respects_gate(tmp_path: Path) -> None:
    root = _job_with_metadata(tmp_path, "demo")

    # No --confirm: refuses.
    result = runner.invoke(app, ["publish", str(root)])
    assert result.exit_code == 2
    assert not (root / "publish_record.json").exists()

    # Confirmed but unapproved: gate blocks, exit 1, still no record.
    result = runner.invoke(app, ["publish", str(root), "--confirm"])
    assert result.exit_code == 1
    assert not (root / "publish_record.json").exists()


# --- security: auth, headers, oversized body --------------------------------


def _serve_with_token(jobs_root: Path, token: str):
    """Start a server with DASHBOARD_TOKEN set via module-level reload."""
    orig = webapp_mod._DASHBOARD_TOKEN
    webapp_mod._DASHBOARD_TOKEN = token
    server = create_server("127.0.0.1", 0, jobs_root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, base, orig


def _restore_token(server, orig):
    server.shutdown()
    server.server_close()
    webapp_mod._DASHBOARD_TOKEN = orig


def test_security_headers_present_no_auth(tmp_path: Path) -> None:
    """Security headers appear on every response even without auth."""
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
            assert resp.headers["X-Frame-Options"] == "DENY"
            assert "strict-origin" in resp.headers["Referrer-Policy"]
            assert resp.headers["Content-Security-Policy"]
    finally:
        server.shutdown()
        server.server_close()


def test_security_headers_present_on_json(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        _, _ = _post_json(base + "/api/jobs", {"job_id": "sec-test"})
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as resp:
            assert resp.headers["X-Content-Type-Options"] == "nosniff"
    finally:
        server.shutdown()
        server.server_close()


def test_auth_required_when_token_set(tmp_path: Path) -> None:
    """401 is returned when DASHBOARD_TOKEN is set and no header provided."""
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/jobs", timeout=5)
        assert exc.value.code == 401
        assert "Bearer" in exc.value.headers.get("WWW-Authenticate", "")
    finally:
        _restore_token(server, orig)


def test_token_mode_serves_public_shell_then_protects_api(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        with urllib.request.urlopen(base + "/#token=s3cr3t", timeout=5) as response:
            html = response.read().decode("utf-8")
            assert response.status == 200
            assert "Xưởng Review Phim" in html
            assert "s3cr3t" not in html
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base + "/api/jobs", timeout=5)
        assert exc.value.code == 401
    finally:
        _restore_token(server, orig)


def test_ui_fonts_are_public_but_confined_to_the_fonts_folder(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        with urllib.request.urlopen(base + "/assets/fonts/montserrat-latin.woff2", timeout=5) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "font/woff2"
            assert response.read(4) == b"wOF2"
        for bad in ("/assets/fonts/missing.woff2", "/assets/fonts/..%2Fman-ke.svg", "/assets/man-ke.svg"):
            with pytest.raises(urllib.error.HTTPError) as exc:
                urllib.request.urlopen(base + bad, timeout=5)
            assert exc.value.code == 401
    finally:
        _restore_token(server, orig)


def test_dashboard_shell_ships_bounded_gold_shimmer() -> None:
    from movie_review_factory.webapp import INDEX_HTML

    assert "/assets/fonts/montserrat-vietnamese.woff2" in INDEX_HTML
    assert ':root[data-busy="1"] #mrfBusyBar' in INDEX_HTML
    assert "prefers-reduced-motion:reduce" in INDEX_HTML
    assert "setBusy(1);" in INDEX_HTML and "setBusy(-1);" in INDEX_HTML
    assert '<span class="mrf-loading">' in INDEX_HTML
    assert "/*@ui-fonts*/" not in INDEX_HTML


def test_activation_page_shares_the_noir_skin() -> None:
    from movie_review_factory.webapp import ACTIVATION_HTML

    assert "/assets/fonts/cormorant-vietnamese.woff2" in ACTIVATION_HTML
    assert "/*@ui-fonts*/" not in ACTIVATION_HTML
    assert "#338ef7" not in ACTIVATION_HTML
    assert '.msg[data-kind="busy"]' in ACTIVATION_HTML
    assert "\u2014" not in ACTIVATION_HTML


def test_auth_passes_with_correct_token(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        req = urllib.request.Request(
            base + "/api/jobs",
            headers={"Authorization": "Bearer s3cr3t"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
    finally:
        _restore_token(server, orig)


def test_authenticated_video_uses_range_streaming_cookie(tmp_path: Path) -> None:
    _job_with_media_index(tmp_path)
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        index_request = urllib.request.Request(
            base + "/api/jobs/media/media-explorer",
            headers={"Authorization": "Bearer s3cr3t"},
        )
        with urllib.request.urlopen(index_request, timeout=5) as response:
            cookie = response.headers.get("Set-Cookie")
            assert cookie and "HttpOnly" in cookie and "SameSite=Strict" in cookie
        source = base + "/api/jobs/media/media/source"
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(source, timeout=5)
        assert exc.value.code == 401
        stream = urllib.request.Request(
            source, headers={"Cookie": cookie.split(";", 1)[0], "Range": "bytes=0-3"},
        )
        with urllib.request.urlopen(stream, timeout=5) as response:
            assert response.status == 206
            assert response.headers["Content-Range"] == "bytes 0-3/12"
            assert response.read() == b"sour"
        jobs = urllib.request.Request(base + "/api/jobs", headers={"Cookie": cookie.split(";", 1)[0]})
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(jobs, timeout=5)
        assert exc.value.code == 401
    finally:
        _restore_token(server, orig)


def test_auth_fails_with_wrong_token(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        req = urllib.request.Request(
            base + "/api/jobs",
            headers={"Authorization": "Bearer wrongtoken"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 401
    finally:
        _restore_token(server, orig)


def test_no_auth_when_token_absent(tmp_path: Path) -> None:
    """When DASHBOARD_TOKEN is empty every request succeeds without a header."""
    orig = webapp_mod._DASHBOARD_TOKEN
    webapp_mod._DASHBOARD_TOKEN = ""
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/api/jobs", timeout=5) as resp:
            assert resp.status == 200
    finally:
        server.shutdown()
        server.server_close()
        webapp_mod._DASHBOARD_TOKEN = orig


def test_import_video_streams_into_owned_job_and_rejects_invalid_input(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "imported"})
    data = b"clip" * 300_000
    status = svc.import_video("imported", "scene.mp4", len(data), io.BytesIO(data))
    source = tmp_path / "imported" / "source.mp4"
    assert source.read_bytes() == data
    assert pipeline.load_manifest(source.parent).config.source_video == source
    assert status["has_source_video"] is True
    assert "source.mp4" in {a["name"] for a in status["artifacts"]}
    with pytest.raises(FileExistsError):
        svc.import_video("imported", "again.mp4", 4, io.BytesIO(b"new!"))
    assert source.read_bytes() == data

    svc.create_job({"job_id": "empty"})
    with pytest.raises(ValueError):
        svc.import_video("empty", "scene.mp4", 0, io.BytesIO())
    with pytest.raises(ValueError):
        svc.import_video("empty", "scene.txt", 4, io.BytesIO(b"nope"))
    with pytest.raises(ValueError):
        svc.import_video("empty", "scene.mp4", 5, io.BytesIO(b"tiny"))
    assert not (tmp_path / "empty" / "source.mp4").exists()
    assert pipeline.load_manifest(tmp_path / "empty").config.source_video is None


def test_import_can_retry_when_manifest_save_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "retry"})
    original = pipeline.save_manifest
    def fail_once(root, manifest):
        monkeypatch.setattr(pipeline, "save_manifest", original)
        raise OSError("disk error")
    monkeypatch.setattr(pipeline, "save_manifest", fail_once)
    with pytest.raises(OSError):
        svc.import_video("retry", "source.mp4", 4, io.BytesIO(b"data"))
    assert not (tmp_path / "retry" / "source.mp4").exists()
    assert svc.import_video("retry", "source.mp4", 4, io.BytesIO(b"data"))["has_source_video"]


def test_create_job_stores_copyright_bypass_setting(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "bypass-test", "copyright_bypass": "balanced"})
    manifest = pipeline.load_manifest(tmp_path / "bypass-test")
    assert manifest.config.copyright_bypass == "balanced"


def test_probe_link_and_download_link_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "dl-job"})

    from movie_review_factory import link_download

    fake_meta = {
        "title": "Online Review Clip",
        "duration": 600.0,
        "thumbnail": "https://example.com/thumb.jpg",
        "subtitle_languages": ["vi", "en"],
        "webpage_url": "https://example.com/watch?v=123",
    }
    monkeypatch.setattr(link_download, "fetch_metadata", lambda url, runner=None: fake_meta)

    meta = svc.probe_link("https://example.com/watch?v=123")
    assert meta["title"] == "Online Review Clip"
    assert meta["duration"] == 600.0

    with pytest.raises(link_download.RightsConfirmationRequired):
        svc.download_link_video("dl-job", "https://example.com/watch?v=123", confirm_rights=False)

    def fake_download(url, dest_dir, *, confirm_rights=None, sub_langs="vi,en", runner=None):
        out = Path(dest_dir) / "source.mp4"
        out.write_bytes(b"downloaded-mp4")
        return {"source_video": str(out), "subtitles": [], "url": url}

    monkeypatch.setattr(link_download, "download_video", fake_download)

    status = svc.download_link_video("dl-job", "https://example.com/watch?v=123", confirm_rights=True)
    assert status["has_source_video"] is True
    source = tmp_path / "dl-job" / "source.mp4"
    assert source.read_bytes() == b"downloaded-mp4"
    assert pipeline.load_manifest(source.parent).config.source_video == source


def test_generate_hook_and_hook_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "hook-job"})

    with pytest.raises(ValueError, match="scenes.json"):
        svc.generate_hook("hook-job")

    (tmp_path / "hook-job" / "scenes.json").write_text('{"scenes": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="video nguồn"):
        svc.generate_hook("hook-job")

    source = tmp_path / "hook-job" / "source.mp4"
    source.write_bytes(b"source-bytes")
    manifest = pipeline.load_manifest(tmp_path / "hook-job")
    manifest.config.source_video = source
    pipeline.save_manifest(tmp_path / "hook-job", manifest)

    from movie_review_factory import hook_crafter

    def fake_build(root, **kwargs):
        dest = Path(root) / "hook.mp4"
        dest.write_bytes(b"hook-video-bytes")
        (Path(root) / "hook.json").write_text('{"scene_index": 1, "duration_seconds": 4.0, "dramatic_score": 0.8}', encoding="utf-8")
        return dest

    monkeypatch.setattr(hook_crafter, "build_hook_teaser", fake_build)

    result = svc.generate_hook("hook-job")
    assert result["present"] is True
    assert result["href"] == "/api/jobs/hook-job/hook.mp4"
    assert result["meta"]["scene_index"] == 1

    path = svc.hook_path("hook-job")
    assert path.read_bytes() == b"hook-video-bytes"

    info = svc.get_hook_info("hook-job")
    assert info["present"] is True
    assert info["meta"]["duration_seconds"] == 4.0




def test_http_import_and_delete_project_are_authenticated(tmp_path: Path) -> None:
    server, base, orig = _serve_with_token(tmp_path, "s3cr3t")
    try:
        headers = {"Authorization": "Bearer s3cr3t"}
        create = urllib.request.Request(
            base + "/api/jobs", data=b'{"job_id":"browser"}', method="POST",
            headers={**headers, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(create, timeout=5) as response:
            assert response.status == 201
        data = b"owned-movie" * 200_000
        upload = urllib.request.Request(
            base + "/api/jobs/browser/source", data=data, method="POST",
            headers={**headers, "X-Source-Name": "movie.mp4", "Content-Type": "video/mp4"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(urllib.request.Request(
                base + "/api/jobs/browser/source", data=b"tiny", method="POST",
                headers={"X-Source-Name": "movie.mp4"},
            ), timeout=5)
        assert exc.value.code == 401
        with urllib.request.urlopen(upload, timeout=10) as response:
            assert response.status == 200
            assert json.load(response)["has_source_video"] is True
        assert (tmp_path / "browser" / "source.mp4").read_bytes() == data

        wrong = urllib.request.Request(
            base + "/api/jobs/browser", data=b'{"confirm":"other"}', method="DELETE",
            headers={**headers, "Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(wrong, timeout=5)
        assert exc.value.code == 400
        assert (tmp_path / "browser" / "manifest.json").exists()
        delete = urllib.request.Request(
            base + "/api/jobs/browser", data=b'{"confirm":"browser"}', method="DELETE",
            headers={**headers, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(delete, timeout=5) as response:
            assert response.status == 200
            assert json.load(response)["deleted"] is True
        assert not (tmp_path / "browser").exists()
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(delete, timeout=5)
        assert exc.value.code == 404
    finally:
        _restore_token(server, orig)


def test_delete_refuses_running_project(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "busy"})
    with svc._lock:
        svc._runs["busy"] = {"running": True}
    with pytest.raises(RuntimeError, match="đang chạy"):
        svc.delete_job("busy", "busy")
    assert (tmp_path / "busy" / "manifest.json").exists()


def test_delete_selected_projects_preflights_all_and_preserves_unselected(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    for name in ("alpha", "beta", "keep"):
        svc.create_job({"job_id": name})
    (tmp_path / "alpha" / "source.mp4").write_bytes(b"owned")
    result = svc.delete_jobs(["alpha", "beta"], "XOA 2")
    assert result == {"deleted": ["alpha", "beta"], "failed": None}
    assert not (tmp_path / "alpha").exists()
    assert not (tmp_path / "beta").exists()
    assert (tmp_path / "keep" / "manifest.json").exists()


def test_delete_selected_rejects_bad_confirmation_duplicate_missing_or_busy_before_deletion(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    for name in ("alpha", "beta"):
        svc.create_job({"job_id": name})
    with pytest.raises(ValueError):
        svc.delete_jobs(["alpha", "beta"], "XOA 1")
    with pytest.raises(ValueError):
        svc.delete_jobs(["alpha", "alpha"], "XOA 2")
    with pytest.raises(ValueError):
        svc.delete_jobs([], "XOA 0")
    with pytest.raises(FileNotFoundError):
        svc.delete_jobs(["alpha", "missing"], "XOA 2")
    with svc._lock:
        svc._runs["beta"] = {"running": True}
    with pytest.raises(RuntimeError, match="đang chạy"):
        svc.delete_jobs(["alpha", "beta"], "XOA 2")
    assert all((tmp_path / name / "manifest.json").exists() for name in ("alpha", "beta"))


def test_bulk_delete_reports_partial_io_failure_and_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    svc = JobsService(tmp_path)
    for name in ("alpha", "beta", "gamma"):
        svc.create_job({"job_id": name})
    original = webapp_mod.shutil.rmtree
    def remove(root):
        if root.name == "beta":
            raise PermissionError("locked")
        return original(root)
    monkeypatch.setattr(webapp_mod.shutil, "rmtree", remove)
    result = svc.delete_jobs(["alpha", "beta", "gamma"], "XOA 3")
    assert result == {"deleted": ["alpha"], "failed": "beta"}
    assert (tmp_path / "beta" / "manifest.json").exists()
    assert (tmp_path / "gamma" / "manifest.json").exists()


def test_http_bulk_delete_requires_token_and_count_confirmation(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "alpha"})
    svc.create_job({"job_id": "beta"})
    server, base, orig = _serve_with_token(tmp_path, "bulk-token")
    try:
        body = json.dumps({"job_ids": ["alpha", "beta"], "confirm": "XOA 2"}).encode()
        unauthorized = urllib.request.Request(
            base + "/api/jobs", data=body, method="DELETE",
            headers={"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(unauthorized, timeout=5)
        assert exc.value.code == 401
        authorized = urllib.request.Request(
            base + "/api/jobs", data=body, method="DELETE",
            headers={"Content-Type": "application/json", "Authorization": "Bearer bulk-token"},
        )
        with urllib.request.urlopen(authorized, timeout=5) as response:
            assert response.status == 200
            assert json.load(response)["deleted"] == ["alpha", "beta"]
        assert not (tmp_path / "alpha").exists()
        assert not (tmp_path / "beta").exists()
    finally:
        _restore_token(server, orig)


def test_delete_refuses_symlink(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    outside = tmp_path.parent / (tmp_path.name + "-external")
    outside.mkdir()
    (outside / "manifest.json").write_text("sentinel", encoding="utf-8")
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("Windows symlink privilege unavailable")
    with pytest.raises(ValueError):
        svc.delete_job("link", "link")
    assert (outside / "manifest.json").read_text(encoding="utf-8") == "sentinel"


def test_oversized_body_returns_413(tmp_path: Path) -> None:
    server, base = _serve(tmp_path)
    try:
        big = json.dumps({"job_id": "x", "pad": "A" * (1_048_576 + 1)}).encode()
        req = urllib.request.Request(
            base + "/api/jobs",
            data=big,
            headers={"Content-Type": "application/json", "Content-Length": str(len(big))},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 413
    finally:
        server.shutdown()
        server.server_close()


def test_unexpected_exception_hides_details(tmp_path: Path) -> None:
    """When auth is on, unhandled exceptions return generic 500 without stack text."""
    server, base, orig = _serve_with_token(tmp_path, "tok")
    try:
        # Trigger a 500 by pointing at a job that exists structurally but whose jobs_root we corrupt right after creation.
        req_create = urllib.request.Request(
            base + "/api/jobs",
            data=json.dumps({"job_id": "boom"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer tok"},
            method="POST",
        )
        urllib.request.urlopen(req_create, timeout=5)
        # Delete the manifest so status() raises unexpectedly deep inside.
        manifest = tmp_path / "boom" / pipeline.MANIFEST_NAME
        manifest.unlink()
        req_status = urllib.request.Request(
            base + "/api/jobs/boom",
            headers={"Authorization": "Bearer tok"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req_status, timeout=5)
        assert exc.value.code in (404, 500)
        body = json.loads(exc.value.read().decode())
        assert "Traceback" not in body.get("error", "")
        assert "Traceback" not in body.get("error_vi", "")
    finally:
        _restore_token(server, orig)


def test_auth_header_not_required_on_html_page_without_token(tmp_path: Path) -> None:
    """The HTML page is accessible without auth when no token is configured."""
    orig = webapp_mod._DASHBOARD_TOKEN
    webapp_mod._DASHBOARD_TOKEN = ""
    server, base = _serve(tmp_path)
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as resp:
            assert resp.status == 200
            assert "Xưởng Review Phim" in resp.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        webapp_mod._DASHBOARD_TOKEN = orig


def test_content_agent_select_offers_agy_pool() -> None:
    assert '<option value="agy">AGY pool (nghiên cứu → dàn ý → kịch bản)</option>' in webapp_mod.INDEX_HTML


def test_agy_pool_badge_and_probe_button_are_wired_into_the_dashboard() -> None:
    html = webapp_mod.INDEX_HTML
    assert 'id="agyPoolBadge"' in html
    assert 'id="agyProbeBtn"' in html
    assert "api('GET', '/api/agy-pool')" in html
    assert "api('POST', '/api/agy-pool/probe')" in html


def test_showerror_rejection_handler_is_defined() -> None:
    # Regression: showError is passed as the rejection handler in many
    # `.catch(showError)` sites (highlight/transcript downloads, section preview,
    # timeline edits, save-search, chat, and the mid-roll "Chèn vào kịch bản"
    # flow via loadStatus). It was referenced but never defined, so any failing
    # action threw "showError is not defined". Guard that a definition ships.
    html = webapp_mod.INDEX_HTML
    assert "function showError(" in html
    assert ".catch(showError)" in html


def test_dashboard_root_layout_dialog_tools_and_project_switcher_are_wired() -> None:
    html = webapp_mod.INDEX_HTML
    assert 'class="layout dashboard-shell"' in html
    assert 'class="top-toolbar"' in html
    assert 'id="projectSummaryLabel"' in html
    assert 'class="project-popover"' in html
    assert '.project-popover{ position:absolute' in html
    assert '<dialog id="createPanel" class="tool-dialog"' in html
    assert '<dialog id="brandPanel" class="tool-dialog"' in html
    assert '<dialog id="libraryPanel" class="tool-dialog tool-dialog-wide"' in html
    assert 'id="openCreateProject"' in html
    assert 'id="openLibraryHub"' in html
    assert 'id="openBrandSettings"' in html
    assert "dialog.showModal()" in html
    assert "document.addEventListener('pointerdown'" in html
    assert 'id="emptyCreateBtn"' in html
    assert 'id="emptyScoutBtn"' in html
    assert "$('createPanel').open = true" not in html
    assert 'id="midrollStatus"' in html
    assert 'insertion.location_label' in html


def test_midroll_state_exposes_insertion_location(tmp_path: Path) -> None:
    root = _job_with_metadata(tmp_path, "cta-location")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    script["sections"][0]["midroll"] = True
    script["sections"][0]["start_seconds"] = 42.4
    (root / "script.json").write_text(json.dumps(script), encoding="utf-8")
    state = JobsService(tmp_path).get_midroll("cta-location")
    assert state["insertion"] == {
        "status": "inserted",
        "start_seconds": 42.4,
        "location_label": "Khoảng 42 giây · gần giữa video",
    }


def test_create_form_normalizes_job_id_to_safe_segment() -> None:
    html = webapp_mod.INDEX_HTML
    assert "function slugifyJobId(" in html
    # The project code is auto-generated and hidden: no visible code field, and
    # the id input is a hidden field the operator never edits.
    assert 'id="newJobId" name="job_id" type="hidden"' in html
    assert "function autoJobId(" in html
    # Submit auto-generates a safe slug, so an accented or spaced title can never
    # be rejected by _is_safe_segment while leaving the project list blank.
    assert "payload.job_id = autoJobId(" in html


# --- roadmap #14: background indexing queue ---------------------------------


def _wait_until(predicate, timeout: float = 5.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def _fake_index_summary(job_id: str) -> dict:
    return {
        "job_id": job_id,
        "stages": {"transcript": "ready", "scenes": "ready", "scene_memory": "skipped"},
        "cancelled": False,
        "media_index": True,
        "scene_memory": {},
    }


def _sourced_job(svc: JobsService, jobs_root: Path, job_id: str) -> Path:
    root = jobs_root / job_id
    pipeline.create_job(root, JobConfig(job_id=job_id, source_video=(root / "source.mp4")))
    (root / "source.mp4").write_bytes(b"video-bytes")
    return root


def test_enqueue_index_runs_in_background_and_reports_progress(tmp_path: Path, monkeypatch) -> None:
    calls: list[str] = []
    finished = threading.Event()

    def fake_run_index(root, *, force=False, progress=None):
        calls.append(Path(root).name)
        if progress is not None:
            progress({"stage": "scenes", "status": "running", "index": 3, "total": 4})
        finished.set()
        return _fake_index_summary(Path(root).name)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    _sourced_job(svc, tmp_path, "job1")

    result = svc.enqueue_index("job1")
    assert result["queued"] is True and result["already"] is False

    assert finished.wait(5.0)
    assert calls == ["job1"]
    # The worker releases the job from the registry once indexing completes.
    assert _wait_until(lambda: svc.index_state("job1") is None)
    assert svc.status("job1")["is_indexing"] is False


def test_enqueue_index_is_idempotent_while_queued(tmp_path: Path, monkeypatch) -> None:
    release = threading.Event()
    started = threading.Event()

    def fake_run_index(root, *, force=False, progress=None):
        started.set()
        release.wait(5.0)
        return _fake_index_summary(Path(root).name)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    _sourced_job(svc, tmp_path, "dup")
    try:
        first = svc.enqueue_index("dup")
        assert started.wait(5.0)
        second = svc.enqueue_index("dup")
        assert first["already"] is False
        assert second["already"] is True
    finally:
        release.set()
    assert _wait_until(lambda: svc.index_state("dup") is None)


def test_background_index_processes_jobs_fifo(tmp_path: Path, monkeypatch) -> None:
    order: list[str] = []

    def fake_run_index(root, *, force=False, progress=None):
        order.append(Path(root).name)
        return _fake_index_summary(Path(root).name)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    for name in ("a", "b", "c"):
        _sourced_job(svc, tmp_path, name)
    svc.enqueue_index("a")
    svc.enqueue_index("b")
    svc.enqueue_index("c")

    assert _wait_until(lambda: len(order) == 3, timeout=5.0)
    assert order == ["a", "b", "c"]


def test_enqueue_index_requires_source_video(tmp_path: Path) -> None:
    svc = JobsService(tmp_path)
    pipeline.create_job(tmp_path / "nosrc", JobConfig(job_id="nosrc"))
    with pytest.raises(RuntimeError):
        svc.enqueue_index("nosrc")


def test_import_auto_enqueues_index_when_enabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MRF_AUTO_INDEX", "1")
    calls: list[str] = []
    finished = threading.Event()

    def fake_run_index(root, *, force=False, progress=None):
        calls.append(Path(root).name)
        finished.set()
        return _fake_index_summary(Path(root).name)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "auto"})
    data = b"clip" * 1000
    svc.import_video("auto", "scene.mp4", len(data), io.BytesIO(data))

    assert finished.wait(5.0)
    assert calls == ["auto"]


def test_import_does_not_auto_enqueue_when_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MRF_AUTO_INDEX", "0")
    calls: list[str] = []

    def fake_run_index(root, *, force=False, progress=None):
        calls.append(Path(root).name)
        return _fake_index_summary(Path(root).name)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    svc.create_job({"job_id": "manual"})
    data = b"clip" * 1000
    svc.import_video("manual", "scene.mp4", len(data), io.BytesIO(data))

    assert not _wait_until(lambda: bool(calls), timeout=0.5)
    assert svc.index_state("manual") is None


def test_active_index_blocks_reimport_and_status_reports_progress(tmp_path: Path, monkeypatch) -> None:
    release = threading.Event()
    started = threading.Event()

    def fake_run_index(root, *, force=False, progress=None):
        started.set()
        if progress is not None:
            progress({"stage": "transcript", "status": "running", "index": 2, "total": 4})
        release.wait(5.0)
        return _fake_index_summary(Path(root).name)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    _sourced_job(svc, tmp_path, "busy")
    svc.enqueue_index("busy")
    try:
        assert started.wait(5.0)
        info = svc.status("busy")
        assert info["is_indexing"] is True
        assert info["indexing"]["running"] is True
        assert info["indexing"]["stage"] == "transcript"
        # A second import of the same (already-indexing) job is refused.
        with pytest.raises((RuntimeError, FileExistsError)):
            svc.import_video("busy", "again.mp4", 4, io.BytesIO(b"new!"))
    finally:
        release.set()
    assert _wait_until(lambda: svc.index_state("busy") is None)


def test_stop_run_cancels_active_index(tmp_path: Path, monkeypatch) -> None:
    import movie_review_factory.cancellation as cancellation

    started = threading.Event()

    def fake_run_index(root, *, force=False, progress=None):
        started.set()
        while True:
            cancellation.checkpoint()
            time.sleep(0.01)

    monkeypatch.setattr(pipeline, "run_index", fake_run_index)
    svc = JobsService(tmp_path)
    _sourced_job(svc, tmp_path, "stopme")
    svc.enqueue_index("stopme")

    assert started.wait(5.0)
    result = svc.stop_run("stopme")
    assert result["stopping"] is True
    assert result.get("indexing") is True
    assert _wait_until(lambda: svc.index_state("stopme") is None)
