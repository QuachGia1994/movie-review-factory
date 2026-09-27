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
from movie_review_factory.webapp import JobsService, create_server


@pytest.fixture
def approved_review(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    create_job(root, JobConfig(job_id="review"))
    (root / "final.mp4").write_bytes(b"approved-review")
    (root / "script.json").write_text('{"approved":true,"sections":[{"narration":"Lời bình"}]}', encoding="utf-8")
    (root / "qa.json").write_text(json.dumps({
        "passed": True, "output_file": "final.mp4",
        "checks": [{"check": "positive_duration", "passed": True, "value": 30.0}],
    }), encoding="utf-8")
    (root / "alignment.json").write_text(json.dumps({
        "cues": [{"start_seconds": 5.0, "end_seconds": 8.0, "text": "Lời bình"}],
    }), encoding="utf-8")
    manifest = load_manifest(root)
    for name in ("script", "alignment", "render", "qa"):
        manifest.stage(name).mark("ready")
    save_manifest(root, manifest)
    return root


def test_short_export_is_background_and_blocks_pipeline(
    approved_review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started, resume = threading.Event(), threading.Event()
    def slow_run(command, **kwargs):
        started.set()
        assert resume.wait(timeout=5)
        Path(command[-1]).write_bytes(b"portrait")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", slow_run)
    svc = JobsService(approved_review.parent)
    try:
        assert svc.start_short("review", {"start_seconds": 5, "end_seconds": 8})["started"]
        assert started.wait(timeout=5)
        assert svc.status("review")["short_export"]["running"] is True
        with pytest.raises(RuntimeError, match="short|xuất"):
            svc.start_run("review")
    finally:
        resume.set()
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline and svc.status("review")["short_export"]["running"]:
        time.sleep(.02)
    assert svc.status("review")["short_export"]["error"] is None
    assert svc.short_path("review", "short-review.mp4").read_bytes() == b"portrait"
    assert svc.short_path("review", "short-review.srt").is_file()


def test_short_http_route_denies_stale_download(
    approved_review: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def render(command, **kwargs):
        Path(command[-1]).write_bytes(b"portrait")
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("movie_review_factory.short_variants.subprocess.run", render)
    server = create_server("127.0.0.1", 0, approved_review.parent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(
            base + "/api/jobs/review/shorts",
            data=b'{"start_seconds":5,"end_seconds":8}',
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 202
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline and server.service.status("review")["short_export"]["running"]:
            time.sleep(.02)
        path = base + "/api/jobs/review/shorts/short-review.mp4"
        with urllib.request.urlopen(path, timeout=5) as response:
            assert response.read() == b"portrait"
            assert response.headers["Content-Type"] == "video/mp4"
        reloaded = JobsService(approved_review.parent)
        assert reloaded.status("review")["short_export"]["href"] == "/api/jobs/review/shorts/short-review.mp4"
        qa_path = approved_review / "qa.json"
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        qa["passed"] = False
        qa_path.write_text(json.dumps(qa), encoding="utf-8")
        assert server.service.status("review")["short_export"]["href"] is None
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(path, timeout=5)
        assert exc.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
