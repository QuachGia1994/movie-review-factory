import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest
from movie_review_factory.webapp import JobsService, create_server, INDEX_HTML


def preview_job(jobs_root: Path) -> Path:
    root = jobs_root / "demo"
    create_job(root, JobConfig(job_id="demo", source_video=root / "source.mp4"))
    (root / "source.mp4").write_bytes(b"footage")
    (root / "narration.mp3").write_bytes(b"voice")
    (root / "script.json").write_text(json.dumps({
        "approved": True, "sections": [{"title": "Đoạn 1", "narration": "Lời kể"}],
    }), encoding="utf-8")
    (root / "scene_plan.json").write_text(json.dumps({
        "aspect_ratio": "16:9", "clips": [
            {"section_index": 1, "section": "Đoạn 1",
             "source_clip": {"start_seconds": 1, "end_seconds": 2},
             "duration_seconds": 2},
        ],
    }), encoding="utf-8")
    (root / "alignment.json").write_text(json.dumps({
        "section_bounds": [{"section_index": 1, "start_seconds": 0, "end_seconds": 2}],
        "cues": [{"start_seconds": 0, "end_seconds": 2, "text": "Lời kể"}],
    }), encoding="utf-8")
    manifest = load_manifest(root)
    for stage in ("script", "scene_plan", "tts", "alignment"):
        manifest.stage(stage).mark("ready")
    save_manifest(root, manifest)
    return root


def _eventually(predicate, timeout=3):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_preview_service_background_guards_and_revision(tmp_path, monkeypatch):
    import movie_review_factory.quick_preview as preview
    root = preview_job(tmp_path)
    svc = JobsService(tmp_path)
    started = threading.Event()
    release = threading.Event()
    def slow_build(path, section):
        started.set()
        release.wait(2)
        raise RuntimeError("encoder stopped")
    monkeypatch.setattr(preview, "build_section_preview", slow_build)
    assert svc.start_section_preview("demo", 1)["started"]
    assert started.wait(2)
    assert svc.section_preview_state("demo", 1)["running"]
    with pytest.raises(RuntimeError, match="preview"):
        svc.edit_timeline("demo", {"action": "trim"})
    with pytest.raises(RuntimeError):
        svc.start_run("demo")
    with pytest.raises(RuntimeError):
        svc.delete_job("demo", "demo")
    with pytest.raises(RuntimeError):
        svc.start_section_preview("demo", 1)
    release.set()
    assert _eventually(lambda: not svc.section_preview_state("demo", 1)["running"])
    assert "encoder stopped" in svc.section_preview_state("demo", 1)["error"]
    assert (root / "manifest.json").exists()


def test_preview_http_202_poll_range_and_stale(tmp_path, monkeypatch):
    import movie_review_factory.quick_preview as preview
    root = preview_job(tmp_path)
    monkeypatch.setattr(preview, "_probe_duration_seconds", lambda path, ffprobe: 10.0)
    monkeypatch.setattr(preview.shutil, "which", lambda tool: tool)
    def fake_ffmpeg(command, **kwargs):
        Path(command[-1]).write_bytes(b"pretend encoded mp4")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(preview.subprocess, "run", fake_ffmpeg)
    server = create_server("127.0.0.1", 0, tmp_path)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(base + "/api/jobs/demo/previews/1", method="POST",
                                         data=b"{}", headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 202
        def ready():
            with urllib.request.urlopen(base + "/api/jobs/demo/previews/1", timeout=3) as response:
                return json.load(response)
        assert _eventually(lambda: bool(ready()["href"]))
        state = ready()
        assert state["srt_href"].endswith("srt?v=" + state["href"].split("?v=")[1])
        request = urllib.request.Request(base + state["href"], headers={"Range": "bytes=0-6"})
        with urllib.request.urlopen(request, timeout=3) as response:
            assert response.status == 206
            assert response.read() == b"pretend"
        with urllib.request.urlopen(base + state["srt_href"], timeout=3) as response:
            assert "Lời kể" in response.read().decode("utf-8")
        plan = root / "scene_plan.json"
        data = json.loads(plan.read_text(encoding="utf-8"))
        data["clips"][0]["source_clip"]["start_seconds"] = 1.2
        plan.write_text(json.dumps(data), encoding="utf-8")
        assert ready()["href"] is None
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + state["href"], timeout=3)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()


def test_preview_ui_controls_exist():
    for identifier in ("sectionPreviewSelect", "sectionPreviewBtn", "sectionPreviewVideo",
                       "sectionPreviewDownload", "sectionPreviewSrt", "sectionPreviewMsg"):
        assert f'id="{identifier}"' in INDEX_HTML
    assert "refreshSectionPreview()" in INDEX_HTML
