import json
import threading
import urllib.request

import pytest

from movie_review_factory.creative_brief import prompt_creative_brief, script_evidence_issues, stale_script_tags, tag_script_span
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import approve_script, create_job, load_manifest, run_job, update_script
from movie_review_factory.webapp import JobsService, create_server


def test_creative_brief_persists_in_manifest(tmp_path):
    config = JobConfig(
        job_id="creative",
        target_minutes=9,
        creative_brief={
            "review_thesis": "Quyền lực không thay thế tình bạn",
            "tone": "hài hước nhưng có phân tích",
            "target_audience": "người đã xem phim",
            "spoiler_policy": "full",
            "forbidden_claims": ["Không nói nhân vật đã chết"],
        },
    )
    create_job(tmp_path, config)

    stored = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert stored["config"]["creative_brief"]["review_thesis"] == "Quyền lực không thay thế tình bạn"
    restored = load_manifest(tmp_path).config
    assert restored.creative_brief.spoiler_policy == "full"
    assert prompt_creative_brief(restored)["desired_length_minutes"] == 9


def test_creative_brief_legacy_default_and_prompt_shape():
    legacy = JobConfig.model_validate({"job_id": "legacy"})
    assert legacy.creative_brief.review_thesis == ""
    assert prompt_creative_brief(legacy) == {"desired_length_minutes": 10}
    with_brief = JobConfig(
        job_id="review",
        creative_brief={
            "review_thesis": "  Chủ đề chính  ",
            "spoiler_policy": "none",
            "forbidden_claims": ["  Không đoán kết thúc  ", "  "],
        },
    )
    assert prompt_creative_brief(with_brief) == {
        "desired_length_minutes": 10,
        "review_thesis": "Chủ đề chính",
        "spoiler_policy": "none",
        "forbidden_claims": ["Không đoán kết thúc"],
    }


def test_prompt_creative_brief_surfaces_retention_advisory():
    from movie_review_factory import analytics

    advice = analytics.retention_advice(reports=[
        {"id": "a", "measurements": {"intro_drop_percentage_points": 42.0,
                                     "cta_drop_percentage_points": None}, "ctr_percent": None},
    ])
    config = JobConfig(job_id="advice")
    prompt = prompt_creative_brief(config, retention_advice=advice)
    assert "retention_advisory" in prompt
    hook_rows = [row for row in prompt["retention_advisory"] if row["target"] == "hook"]
    assert hook_rows
    assert "mở đầu" in hook_rows[0]["message"]
    # advisory is additive only: it must not mutate the editorial fields
    assert prompt["desired_length_minutes"] == config.target_minutes


def test_prompt_creative_brief_without_advice_keeps_legacy_shape():
    config = JobConfig(job_id="advice")
    assert "retention_advisory" not in prompt_creative_brief(config)
    assert prompt_creative_brief(config, retention_advice=None) == prompt_creative_brief(config)


def test_prompt_creative_brief_disabled_advice_adds_nothing():
    from movie_review_factory import analytics

    advice = analytics.retention_advice(
        reports=[{"id": "a", "measurements": {"intro_drop_percentage_points": 90.0}}],
        enabled=False,
    )
    assert "retention_advisory" not in prompt_creative_brief(JobConfig(job_id="advice"), retention_advice=advice)


def test_tag_script_span_records_quote_and_resets_approval():
    script = {"approved": True, "sections": [{"narration": "Ben chạy. Đây là điểm hay."}]}
    tagged = tag_script_span(script, 0, 0, 9, "plot_recap", ["scene:2"])
    assert script["approved"] is True
    assert "annotations" not in script["sections"][0]
    assert tagged["approved"] is False
    assert tagged["sections"][0]["annotations"] == [
        {"start": 0, "end": 9, "quote": "Ben chạy.", "kind": "plot_recap", "evidence_refs": ["scene:2"]}
    ]
    assert stale_script_tags(tagged) == []
    assert tag_script_span(tagged, 0, 0, 9, "plot_recap", ["scene:2"]) == tagged


def test_tag_script_span_rejects_overlap_and_detects_stale_quote():
    script = {"approved": False, "sections": [{"narration": "Ben chạy. Đây là điểm hay."}]}
    tagged = tag_script_span(script, 0, 0, 9, "plot_recap")
    with pytest.raises(ValueError, match="overlap"):
        tag_script_span(tagged, 0, 4, 12, "opinion")
    with pytest.raises(ValueError, match="invalid narration span"):
        tag_script_span(tagged, 0, 99, 100, "opinion")
    tagged["sections"][0]["narration"] = "Ben đứng. Đây là điểm hay."
    assert stale_script_tags(tagged) == [(0, 0)]


def test_script_editor_preserves_tags_and_approval_checks_stale_quote(tmp_path):
    create_job(tmp_path, JobConfig(job_id="tagged"))
    run_job(tmp_path, until="script")
    path = tmp_path / "script.json"
    original = json.loads(path.read_text(encoding="utf-8"))
    original_text = original["sections"][0]["narration"]
    tagged = tag_script_span(original, 0, 0, 5, "plot_recap", ["scene:0"])
    path.write_text(json.dumps(tagged, ensure_ascii=False), encoding="utf-8")
    fields = [{key: value for key, value in section.items() if key != "annotations"} for section in tagged["sections"]]
    fields[0]["narration"] = "Sửa!" + original_text[5:]
    saved = update_script(tmp_path, {"sections": fields})
    assert saved["sections"][0]["annotations"] == tagged["sections"][0]["annotations"]
    with pytest.raises(ValueError, match="stale"):
        approve_script(tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["approved"] is False
    fields[0]["annotations"] = []
    update_script(tmp_path, {"sections": fields})
    assert approve_script(tmp_path)["approved"] is True


def test_web_create_brief_and_tag_script_with_version(tmp_path):
    service = JobsService(tmp_path)
    service.create_job({
        "job_id": "web-brief", "creative_brief": {
            "review_thesis": "Phân tích tình bạn", "spoiler_policy": "limited",
        },
    })
    root = tmp_path / "web-brief"
    assert load_manifest(root).config.creative_brief.review_thesis == "Phân tích tình bạn"
    run_job(root, until="script")
    result = service.tag_script("web-brief", {
        "section_index": 0, "start": 0, "end": 5, "kind": "opinion", "evidence_refs": [],
    })
    assert result["script"]["sections"][0]["annotations"][0]["kind"] == "opinion"
    assert result["approved"] is False
    assert service.list_versions("web-brief")["versions"]


def test_web_tag_route_and_create_form_controls(tmp_path):
    root = tmp_path / "web-route"
    create_job(root, JobConfig(job_id="web-route"))
    run_job(root, until="script")
    server = create_server("127.0.0.1", 0, tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        assert 'name="review_thesis"' in html
        assert 'id="tagScriptBtn"' in html
        body = json.dumps({"section_index": 0, "start": 0, "end": 5, "kind": "opinion"}).encode("utf-8")
        request = urllib.request.Request(
            base + "/api/jobs/web-route/script/tag", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            tagged = json.loads(response.read().decode("utf-8"))
        assert tagged["script"]["sections"][0]["annotations"][0]["quote"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_tts_rejects_manually_approved_stale_tags(tmp_path):
    create_job(tmp_path, JobConfig(job_id="tag-bypass"))
    run_job(tmp_path, until="script")
    path = tmp_path / "script.json"
    script = json.loads(path.read_text(encoding="utf-8"))
    tagged = tag_script_span(script, 0, 0, 5, "plot_recap")
    tagged["sections"][0]["narration"] = "Sửa!" + tagged["sections"][0]["narration"][5:]
    tagged["approved"] = True
    path.write_text(json.dumps(tagged, ensure_ascii=False), encoding="utf-8")
    result = run_job(tmp_path, until="tts")
    stage = result.stage("tts")
    assert stage.status == "skipped"
    assert "stale" in stage.message


def test_plot_recap_requires_existing_internal_evidence_only(tmp_path):
    create_job(tmp_path, JobConfig(job_id="evidence"))
    run_job(tmp_path, until="script")
    script_path = tmp_path / "script.json"
    script = json.loads(script_path.read_text(encoding="utf-8"))
    tagged = tag_script_span(script, 0, 0, 5, "plot_recap")
    script_path.write_text(json.dumps(tagged, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="needs source evidence"):
        approve_script(tmp_path)

    tagged = tag_script_span(script, 0, 0, 5, "plot_recap", ["scene:1"])
    script_path.write_text(json.dumps(tagged, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="missing source evidence"):
        approve_script(tmp_path)

    (tmp_path / "scenes.json").write_text(json.dumps({"scenes": [{"index": 1, "text": "a real indexed scene"}]}), encoding="utf-8")
    assert script_evidence_issues(tagged, {"scenes": [{"index": 1}]}, {}) == []
    assert approve_script(tmp_path)["approved"] is True
