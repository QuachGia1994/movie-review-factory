from __future__ import annotations

import json
import threading
from http.client import HTTPConnection

import pytest

from movie_review_factory import pipeline
from movie_review_factory.models import JobConfig
from movie_review_factory.webapp import JobsService, _Server


def test_update_job_config_persists_allowed_fields(tmp_path):
    service = JobsService(tmp_path)
    service.create_job({"job_id": "demo"})

    status = service.update_job_config("demo", {
        "movie_title": "Hidden Gem",
        "content_agent": "agy",
        "visual_variety": "balanced",
        "watermark_removal": {"enabled": True, "detect": {"method": "temporal"}},
        "tts_provider": "vieneu",
        "tts_voice": "studio-vi",
        "target_minutes": 12,
        "aspect_ratio": "9:16",
    })

    cfg = pipeline.load_manifest(tmp_path / "demo").config
    assert cfg.movie_title == "Hidden Gem"
    assert cfg.content_agent == "agy"
    assert cfg.visual_variety == "balanced"
    assert cfg.watermark_removal.enabled is True
    assert cfg.watermark_removal.detect.method == "temporal"
    assert cfg.tts_provider == "vieneu"
    assert cfg.tts_voice == "studio-vi"
    assert cfg.target_minutes == 12
    assert cfg.aspect_ratio == "9:16"
    assert status["config"]["movie_title"] == "Hidden Gem"


def test_update_job_config_rejects_running_and_unknown_fields(tmp_path):
    service = JobsService(tmp_path)
    service.create_job({"job_id": "demo"})
    service._runs["demo"] = {"running": True}
    with pytest.raises(RuntimeError, match="pipeline"):
        service.update_job_config("demo", {"content_agent": "agy"})
    service._runs["demo"] = {"running": False}
    with pytest.raises(ValueError, match="không được hỗ trợ"):
        service.update_job_config("demo", {"source_video": "other.mp4"})


def test_scout_enqueue_uses_candidate_and_active_channel_defaults(tmp_path, monkeypatch):
    service = JobsService(tmp_path)
    monkeypatch.setattr(service.creator_library, "active_channel_defaults", lambda: {
        "tts_provider": "fptai", "tts_voice": "banmai", "visual_variety": "light"
    })
    monkeypatch.setattr(
        "movie_review_factory.webapp.content_scout.enqueue_gem_for_review",
        lambda candidate_id: {"candidate": {
            "id": candidate_id, "title": "Original Title", "vietnamese_title": "Tên Việt",
        }},
    )

    result = service.scout_enqueue({"candidate_id": "gem-1", "auto_create": True})
    cfg = pipeline.load_manifest(tmp_path / result["created_job"]["job_id"]).config
    assert cfg.movie_title == "Tên Việt"
    assert cfg.content_agent == "agy"
    assert cfg.visual_variety == "light"
    assert cfg.tts_provider == "fptai"
    assert cfg.tts_voice == "banmai"
    assert cfg.watermark_removal.enabled is True
    assert cfg.watermark_removal.detect.method == "color"


def test_job_config_http_route(tmp_path):
    service = JobsService(tmp_path)
    pipeline.create_job(tmp_path / "demo", JobConfig(job_id="demo"))
    server = _Server(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        body = json.dumps({"content_agent": "claude"})
        conn.request("POST", "/api/jobs/demo/config", body, {"Content-Type": "application/json"})
        response = conn.getresponse()
        payload = json.loads(response.read())
        assert response.status == 200
        assert payload["config"]["content_agent"] == "claude"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_create_job_accepts_cheap_watermark_method(tmp_path):
    service = JobsService(tmp_path)
    service.create_job({"job_id": "cheap", "watermark_detect": "temporal", "watermark_method": "delogo"})
    wm = pipeline.load_manifest(tmp_path / "cheap").config.watermark_removal
    assert wm.enabled is True
    assert wm.method == "delogo"
    assert wm.detect.method == "temporal"


def test_create_job_rejects_unknown_watermark_method(tmp_path):
    service = JobsService(tmp_path)
    with pytest.raises(ValueError, match="watermark_method"):
        service.create_job({"job_id": "bad", "watermark_detect": "color", "watermark_method": "magic"})


def test_update_job_config_keeps_mask_and_detector_knobs(tmp_path):
    service = JobsService(tmp_path)
    service.create_job({"job_id": "demo"})
    root = tmp_path / "demo"
    manifest = pipeline.load_manifest(root)
    manifest.config.watermark_removal = manifest.config.watermark_removal.model_validate({
        "enabled": True, "mask": "masks/wm.png", "boxes": [[0.1, 0.1, 0.2, 0.2]],
        "detect": {"method": "external", "external_cmd": "detect {video} {out}"},
    })
    pipeline.save_manifest(root, manifest)

    service.update_job_config("demo", {"watermark_removal": {
        "enabled": True, "method": "blur", "detect": {"method": "external"},
    }})

    wm = pipeline.load_manifest(root).config.watermark_removal
    assert wm.method == "blur"
    assert wm.mask.as_posix() == "masks/wm.png"
    assert wm.boxes == [[0.1, 0.1, 0.2, 0.2]]
    assert wm.detect.external_cmd == "detect {video} {out}"


def test_dashboard_offers_and_explains_watermark_methods():
    from movie_review_factory.webapp import INDEX_HTML

    assert INDEX_HTML.count('class="wm-method"') == 3
    assert 'name="watermark_method"' in INDEX_HTML
    for method in ("propainter", "delogo", "blur"):
        assert f"value: '{method}'" in INDEX_HTML
