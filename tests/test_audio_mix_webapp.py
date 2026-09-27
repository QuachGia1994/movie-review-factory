"""Audio settings are project-local and invalidate stale final artifacts."""

import json
import threading
import urllib.request
from pathlib import Path

import pytest

from movie_review_factory import pipeline
from movie_review_factory.webapp import JobsService, create_server


def test_audio_change_keeps_approved_script_but_invalidates_final(tmp_path: Path) -> None:
    service = JobsService(tmp_path)
    service.create_job({"job_id": "film"})
    root = tmp_path / "film"
    song = tmp_path / "licensed.wav"
    song.write_bytes(b"test")
    (root / "script.json").write_text(json.dumps({"approved": True}), encoding="utf-8")
    (root / "final.mp4").write_bytes(b"old final")
    (root / "qa.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("render").mark("ready")
    manifest.stage("qa").mark("ready")
    pipeline.save_manifest(root, manifest)

    new = {"voice_gain_db": 1,
           "music": {"path": str(song), "rights_note": "owner license", "gain_db": -18},
           "effects": []}
    result = service.update_audio_mix("film", new)
    assert result["changed"] is True
    assert service.get_audio_mix("film")["audio_mix"] == new
    assert (root / "script.json").exists()
    assert json.loads((root / "script.json").read_text(encoding="utf-8"))["approved"] is True
    assert not (root / "final.mp4").exists()
    assert not (root / "qa.json").exists()
    manifest = pipeline.load_manifest(root)
    assert manifest.stage("render").status == "pending"
    assert manifest.stage("qa").status == "pending"
    versions = service.list_versions("film")["versions"]
    assert versions[0]["passing_final"] is True
    assert "final.mp4" in versions[0]["files"]
    assert service.update_audio_mix("film", new)["changed"] is False
    assert len(service.list_versions("film")["versions"]) == 1


def test_invalid_or_unlicensed_audio_does_not_invalidate(tmp_path: Path) -> None:
    service = JobsService(tmp_path)
    service.create_job({"job_id": "film"})
    root = tmp_path / "film"
    (root / "final.mp4").write_bytes(b"old")
    with pytest.raises(ValueError, match="rights_note"):
        service.update_audio_mix("film", {
            "music": {"path": str(tmp_path / "absent.wav"), "rights_note": ""},
        })
    assert (root / "final.mp4").exists()
    assert not (root / "audio_mix.json").exists()


def test_audio_http_roundtrip_and_invalid_license(tmp_path: Path) -> None:
    server = create_server("127.0.0.1", 0, tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        service = server.service
        service.create_job({"job_id": "film"})
        base = f"http://127.0.0.1:{server.server_address[1]}/api/jobs/film/audio-mix"
        with urllib.request.urlopen(base, timeout=5) as response:
            assert json.load(response)["audio_mix"]["music"] is None
        request = urllib.request.Request(base, data=json.dumps({"voice_gain_db": 2}).encode(),
                                         method="POST", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.load(response)["changed"] is True
        with urllib.request.urlopen(base, timeout=5) as response:
            assert json.load(response)["audio_mix"]["voice_gain_db"] == 2
        with urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/", timeout=5) as response:
            page = response.read().decode()
        assert 'id="audioPanel"' in page
        assert 'id="audioMusicRights"' in page
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
