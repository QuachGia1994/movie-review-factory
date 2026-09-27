import http.client
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from movie_review_factory.handoff import build_handoff
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest
from movie_review_factory.webapp import JobsService, create_server


@pytest.fixture
def handoff_job(tmp_path: Path) -> Path:
    root = tmp_path / "review"
    create_job(root, JobConfig(job_id="review"))
    docs = {
        "script.json": {"approved": True, "sections": [{"title": "Intro"}]},
        "scene_plan.json": {"clips": []},
        "alignment.json": {"cues": []},
        "render.json": {"clips": []},
        "qa.json": {"passed": True, "output_file": "final.mp4"},
        "youtube_metadata.json": {"approved": True, "title": "Review", "description": "Description"},
    }
    for name, value in docs.items():
        (root / name).write_text(json.dumps(value), encoding="utf-8")
    for name, value in {"final.mp4": b"video", "aligned.srt": b"1\n00:00:00,000 --> 00:00:01,000\nHi\n",
                        "thumbnail.jpg": b"image"}.items():
        (root / name).write_bytes(value)
    manifest = load_manifest(root)
    for stage in ("script", "scene_plan", "alignment", "render", "qa", "metadata", "thumbnail"):
        manifest.stage(stage).mark("ready")
    save_manifest(root, manifest)
    build_handoff(root)
    return root


def test_service_imports_bytes_to_current_handoff_and_keeps_no_temp(handoff_job: Path):
    service = JobsService(handoff_job.parent)
    raw = b"time_seconds,retention_percent,impressions,ctr_percent\n0,100,1800,4.1\n30,73,,\n95,67,,\n105,61,,\n"
    result = service.import_analytics("review", "csv", raw, "100", "Cắt intro nhanh hơn")
    assert result["measurements"]["intro_drop_percentage_points"] == 27
    assert result["measurements"]["cta_drop_percentage_points"] == 6
    assert service.list_analytics("review")["imports"][0]["approved_revision_sha256"] == result["approved_revision_sha256"]
    assert not list(handoff_job.glob(".studio-*"))


def test_service_rejects_bad_format_and_overlimit(handoff_job: Path):
    service = JobsService(handoff_job.parent)
    with pytest.raises(ValueError):
        service.import_analytics("review", "../../studio.csv", b"bad", "", "")
    with pytest.raises(OverflowError):
        service.import_analytics("review", "csv", b"x" * 2_000_001, "", "")
    assert service.list_analytics("review")["imports"] == []


def test_http_upload_and_history_with_request_limit_and_ui(handoff_job: Path):
    server = create_server("127.0.0.1", 0, handoff_job.parent)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        request = urllib.request.Request(
            url + "/api/jobs/review/analytics",
            data=b'{"retention":[{"time_seconds":0,"retention_percent":100},{"time_seconds":30,"retention_percent":70}],"ctr_percent":5.2}',
            headers={"Content-Type": "application/json", "X-Studio-Format": "json",
                     "X-Studio-Notes": "Shorter%20intro", "X-Studio-CTA-Seconds": "100"},
            method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.load(response)
            assert response.status == 201
            assert result["notes"] == "Shorter intro"
            assert result["measurements"]["intro_drop_percentage_points"] == 30
        with urllib.request.urlopen(url + "/api/jobs/review/analytics", timeout=5) as response:
            assert json.load(response)["imports"][0]["id"] == result["id"]
        connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
        connection.putrequest("POST", "/api/jobs/review/analytics")
        connection.putheader("Content-Length", "2000001")
        connection.putheader("X-Studio-Format", "csv")
        connection.endheaders()
        assert connection.getresponse().status == 413
        connection.close()
        with urllib.request.urlopen(url, timeout=5) as response:
            html = response.read().decode("utf-8")
            assert 'id="studioExport"' in html
            assert 'id="analyticsResults"' in html
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)
