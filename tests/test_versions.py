import hashlib
import json
import threading
import urllib.request
from pathlib import Path

import pytest

from movie_review_factory import pipeline, versions
from movie_review_factory.models import JobConfig
from movie_review_factory.webapp import JobsService, create_server


def _job(tmp_path: Path) -> Path:
    root = tmp_path / "demo"
    pipeline.create_job(root, JobConfig(job_id="demo"))
    pipeline.run_job(root, until="metadata")
    return root


def test_snapshot_restore_script_rejects_stale_approval_and_keeps_previous(tmp_path):
    root = _job(tmp_path)
    pipeline.approve_script(root)
    first = versions.create(root, "Bản trước")
    changed = pipeline.update_script(root, {"notes": "new commentary"})
    assert changed["approved"] is False
    result = versions.restore(root, first["id"], "script")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    assert script["approved"] is False
    assert "new commentary" not in str(script)
    assert any(item["id"] == result["previous_version_id"] for item in versions.list_versions(root))
    assert pipeline.load_manifest(root).stage("tts").status == "pending"


def test_restore_same_script_content_keeps_current_approval(tmp_path):
    root = _job(tmp_path)
    snap = versions.create(root, "Bản gốc")
    pipeline.approve_script(root)
    versions.restore(root, snap["id"], "script")
    assert json.loads((root / "script.json").read_text(encoding="utf-8"))["approved"] is True


def test_snapshot_copies_final_and_survives_edit_invalidation(tmp_path):
    root = _job(tmp_path)
    (root / "final.mp4").write_bytes(b"previous passing export")
    (root / "qa.json").write_text('{"passed": true}', encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("qa").mark("ready", "passed")
    pipeline.save_manifest(root, manifest)
    snap = versions.create(root, "Xuất trước")
    assert snap["passing_final"] is True
    pipeline.update_script(root, {"notes": "different revision"})
    assert not (root / "final.mp4").exists()
    assert versions.artifact_path(root, snap["id"], "final.mp4").read_bytes() == b"previous passing export"


def test_passing_final_restores_only_matching_revision(tmp_path):
    root = _job(tmp_path)
    pipeline.approve_script(root)
    pipeline.approve_metadata(root)
    (root / "voice.json").write_text('{"duration_seconds": 4}', encoding="utf-8")
    (root / "narration.mp3").write_bytes(b"old voice")
    (root / "final.mp4").write_bytes(b"old passing")
    (root / "render.json").write_text('{"duration_seconds": 4}', encoding="utf-8")
    (root / "qa.json").write_text('{"passed": true}', encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("qa").mark("ready", "passed")
    pipeline.save_manifest(root, manifest)
    first = versions.create(root, "Passed")
    (root / "final.mp4").write_bytes(b"new output")
    versions.restore(root, first["id"], "final")
    assert (root / "final.mp4").read_bytes() == b"old passing"
    assert pipeline.load_manifest(root).stage("qa").status == "ready"
    (root / "narration.mp3").write_bytes(b"new voice")
    with pytest.raises(ValueError, match="different editorial revision"):
        versions.restore(root, first["id"], "final")
    (root / "narration.mp3").write_bytes(b"old voice")
    pipeline.update_script(root, {"notes": "changed"})
    with pytest.raises(ValueError, match="different editorial revision"):
        versions.restore(root, first["id"], "final")


def test_audio_config_and_source_revision_gate_historical_final(tmp_path):
    root = _job(tmp_path)
    pipeline.approve_script(root)
    pipeline.approve_metadata(root)
    music = tmp_path / "licensed-music.wav"
    music.write_bytes(b"licensed sample v1")
    config = {"voice_gain_db": 1, "music": {"path": str(music), "rights_note": "licensed"}}
    (root / "audio_mix.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "final.mp4").write_bytes(b"passing version")
    provenance = [{"kind": "music", "path": str(music), "sha256": hashlib.sha256(music.read_bytes()).hexdigest()}]
    config_hash = hashlib.sha256((root / "audio_mix.json").read_bytes()).hexdigest()
    (root / "render.json").write_text(json.dumps({"audio_mix": {"provenance": provenance, "config_sha256": config_hash}}), encoding="utf-8")
    (root / "qa.json").write_text('{"passed": true}', encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("qa").mark("ready", "passed")
    pipeline.save_manifest(root, manifest)
    snap = versions.create(root, "Bản âm thanh")
    assert versions.artifact_path(root, snap["id"], "audio_mix.json").is_file()
    versions.restore(root, snap["id"], "final")
    music.write_bytes(b"replaced in-place")
    with pytest.raises(ValueError, match="audio source revision"):
        versions.restore(root, snap["id"], "final")
    music.write_bytes(b"licensed sample v1")
    config["voice_gain_db"] = 2
    (root / "audio_mix.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="editorial revision"):
        versions.restore(root, snap["id"], "final")
    (root / "audio_mix.json").unlink()
    with pytest.raises(ValueError, match="audio revision"):
        versions.restore(root, snap["id"], "final")


def test_snapshot_cannot_promote_final_from_stale_audio_render(tmp_path):
    root = _job(tmp_path)
    pipeline.approve_script(root)
    pipeline.approve_metadata(root)
    (root / "audio_mix.json").write_text('{"voice_gain_db": 1}', encoding="utf-8")
    (root / "final.mp4").write_bytes(b"old render")
    (root / "render.json").write_text('{"audio_mix": {"provenance": [], "config_sha256": "stale"}}', encoding="utf-8")
    (root / "qa.json").write_text('{"passed": true}', encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("qa").mark("ready", "passed")
    pipeline.save_manifest(root, manifest)
    snap = versions.create(root, "Wrong revision")
    with pytest.raises(ValueError, match="render audio config provenance"):
        versions.restore(root, snap["id"], "final")


def test_audio_enabled_after_silent_snapshot_cannot_restore_old_final(tmp_path):
    root = _job(tmp_path)
    pipeline.approve_script(root)
    pipeline.approve_metadata(root)
    (root / "final.mp4").write_bytes(b"silent version")
    (root / "render.json").write_text('{"audio_mix": {"provenance": []}}', encoding="utf-8")
    (root / "qa.json").write_text('{"passed": true}', encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("qa").mark("ready", "passed")
    pipeline.save_manifest(root, manifest)
    snap = versions.create(root, "Không nhạc")
    (root / "audio_mix.json").write_text('{"voice_gain_db": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="audio revision"):
        versions.restore(root, snap["id"], "final")


def test_service_auto_snapshot_before_edit(tmp_path):
    root = _job(tmp_path)
    service = JobsService(tmp_path)
    service.update_script("demo", {"notes": "reworked"})
    entries = service.list_versions("demo")["versions"]
    assert len(entries) == 1
    original = entries[0]["id"]
    assert service.version_artifact("demo", original, "script.json").is_file()
    service.restore_version("demo", original, "script")
    assert len(service.list_versions("demo")["versions"]) == 2


def test_http_versions_and_historical_artifact(tmp_path):
    root = _job(tmp_path)
    server = create_server("127.0.0.1", 0, tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}/api/jobs/demo/versions"
    try:
        request = urllib.request.Request(base, data=json.dumps({"name": "Bản đầu"}).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 201
            version_id = json.load(response)["id"]
        with urllib.request.urlopen(base, timeout=5) as response:
            assert json.load(response)["versions"][0]["id"] == version_id
        with urllib.request.urlopen(base + f"/{version_id}/artifacts/script.json", timeout=5) as response:
            assert json.load(response)["approved"] is False
        with urllib.request.urlopen(base.removesuffix('/api/jobs/demo/versions'), timeout=5) as response:
            html = response.read().decode('utf-8')
            assert 'id="versionsPanel"' in html
            assert 'id="restoreVersionBtn"' in html
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_metadata_restore_revokes_only_on_changed_content(tmp_path):
    root = _job(tmp_path)
    snap = versions.create(root, "Metadata before")
    pipeline.approve_metadata(root)
    versions.restore(root, snap["id"], "metadata")
    meta = json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8"))
    assert meta["approved"] is True
    pipeline.update_metadata(root, {"title": "Different review"})
    versions.restore(root, snap["id"], "metadata")
    meta = json.loads((root / "youtube_metadata.json").read_text(encoding="utf-8"))
    assert meta["approved"] is False


def test_reject_bad_id_and_tampered_snapshot(tmp_path):
    root = _job(tmp_path)
    snap = versions.create(root, "Snapshot")
    with pytest.raises(ValueError):
        versions.restore(root, "../../etc", "script")
    (root / "versions" / snap["id"] / "script.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupt"):
        versions.restore(root, snap["id"], "script")


def _caption_job(tmp_path):
    root = _job(tmp_path)
    pipeline.approve_script(root)
    (root / "voice.json").write_text('{"duration_seconds": 2}', encoding="utf-8")
    (root / "narration.mp3").write_bytes(b"original narration")
    manifest = pipeline.load_manifest(root)
    manifest.stage("tts").mark("ready", "voice recorded")
    pipeline.save_manifest(root, manifest)
    (root / "alignment.json").write_text('{"cues": []}', encoding="utf-8")
    (root / "aligned.srt").write_text("1\\n00:00:00,000 --> 00:00:01,000\\nHi\\n", encoding="utf-8")
    return root


def test_captions_restore_invalidates_render(tmp_path):
    root = _caption_job(tmp_path)
    snap = versions.create(root, "Captions")
    (root / "aligned.srt").write_text("changed", encoding="utf-8")
    versions.restore(root, snap["id"], "captions")
    assert (root / "aligned.srt").read_text(encoding="utf-8").startswith("1")
    assert pipeline.load_manifest(root).stage("render").status == "pending"


def test_captions_restore_rejects_changed_script_before_writing(tmp_path):
    root = _caption_job(tmp_path)
    snap = versions.create(root, "Original captions")
    (root / "aligned.srt").write_text("current captions", encoding="utf-8")
    script = json.loads((root / "script.json").read_text(encoding="utf-8"))
    script["notes"] = "new commentary"
    (root / "script.json").write_text(json.dumps(script), encoding="utf-8")
    with pytest.raises(ValueError, match="different script revision"):
        versions.restore(root, snap["id"], "captions")
    assert (root / "aligned.srt").read_text(encoding="utf-8") == "current captions"
    assert len(versions.list_versions(root)) == 1


def test_captions_restore_rejects_changed_narration_before_writing(tmp_path):
    root = _caption_job(tmp_path)
    snap = versions.create(root, "Original captions")
    (root / "aligned.srt").write_text("current captions", encoding="utf-8")
    (root / "narration.mp3").write_bytes(b"new narration")
    with pytest.raises(ValueError, match="different voice revision"):
        versions.restore(root, snap["id"], "captions")
    assert (root / "aligned.srt").read_text(encoding="utf-8") == "current captions"
    assert len(versions.list_versions(root)) == 1
