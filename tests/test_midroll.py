"""AGY CTA must occupy the midpoint without disturbing the original shots."""
import copy
import pytest

from movie_review_factory import midroll


def fixtures():
    script = {"approved": True, "target_minutes": 10, "sections": [
        {"title": f"Phần {i}", "duration_seconds": duration, "budget_minutes": duration / 60,
         "narration": "Lời dẫn nội dung phim."}
        for i, duration in enumerate((60, 150, 90, 150, 120, 30), 1)
    ]}
    plan = {"total_seconds": 600, "clips": [
        {"section": f"Phần {i}", "section_index": i, "start_seconds": start,
         "duration_seconds": duration, "source_clip": {"start_seconds": start + 10,
                                                       "end_seconds": start + duration + 10}}
        for i, (start, duration) in enumerate(
            ((0, 60), (60, 150), (210, 90), (300, 150), (450, 120), (570, 30)), 1
        )
    ]}
    return script, plan


def test_inserts_at_midpoint_and_preserves_source_ranges():
    script, plan = fixtures()
    original = copy.deepcopy(plan)
    amended_script, amended_plan, midpoint = midroll.prepare(
        script, plan, "Eon tua thời gian, bạn bấm thích và đăng ký Màn Kể nhé!", 15,
    )
    assert midpoint == 300
    assert amended_script["approved"] is False
    assert amended_script["sections"][3]["midroll"] is True
    assert amended_script["sections"][3]["duration_seconds"] == 15
    assert amended_plan["total_seconds"] == 615
    assert amended_plan["clips"][3]["type"] == "midroll"
    assert amended_plan["clips"][3]["start_seconds"] == 300
    assert amended_plan["clips"][4]["start_seconds"] == 315
    assert amended_plan["clips"][4]["source_clip"] == original["clips"][3]["source_clip"]
    assert script["approved"] is True
    assert plan == original


def test_adjusts_visual_midpoint_to_match_existing_narration():
    script, plan = fixtures()
    script["sections"][0]["narration"] = "một " * 150
    script["sections"][1]["narration"] = "hai " * 150
    script["sections"][2]["narration"] = "ba " * 150
    for section in script["sections"][3:]:
        section["narration"] = "bốn " * 145
    _, amended, start = midroll.prepare(script, plan, "Bấm thích và đăng ký kênh nhé!", 10, 610)
    assert start > 309
    assert amended["clips"][2]["duration_seconds"] == pytest.approx(start - 210)
    assert amended["clips"][3]["start_seconds"] == start
    assert amended["clips"][4]["start_seconds"] == start + 10


def test_places_cta_by_narration_when_visual_budget_is_a_different_scale():
    """Nominal 10-minute visual budget, but the spoken narration is ~3.6 min and
    front-loaded. The nearest *visual* boundary to 50% lands the CTA past 60% of
    actual narration (the QA failure); the placer must instead pick the boundary
    that keeps the CTA inside the central 40-60% of narration."""
    visual = (90, 120, 150, 150, 90)   # sums to 600s (the 10-minute target budget)
    words = (165, 180, 181, 174, 143)  # measured narration ~3.6 min, front-loaded
    script = {"approved": True, "target_minutes": 10, "sections": [
        {"title": f"Phần {i}", "duration_seconds": d, "budget_minutes": d / 60,
         "narration": "từ " * w}
        for i, (d, w) in enumerate(zip(visual, words), 1)
    ]}
    clips = []
    start = 0.0
    for i, d in enumerate(visual, 1):
        clips.append({"section": f"Phần {i}", "section_index": i, "start_seconds": start,
                      "duration_seconds": d,
                      "source_clip": {"start_seconds": start, "end_seconds": start + d}})
        start += d
    plan = {"total_seconds": 600, "clips": clips}
    _, amended, _ = midroll.prepare(
        script, plan, "Bấm thích và đăng ký kênh Màn Kể nhé!", 10, 215.856,
    )
    cta = next(clip for clip in amended["clips"] if clip.get("type") == "midroll")
    # Section-2/3 boundary (script section 2) keeps narration fraction ~0.41; the
    # section-3/4 boundary the visual budget would have chosen sits at ~0.62.
    assert cta["section_index"] == 3
    words_before = sum(words[:2])
    voice_before = 215.856 * words_before / sum(words)
    fraction = (voice_before + 5) / (215.856 + 10)
    assert 0.4 <= fraction <= 0.6


def test_rejects_duplicate_and_far_from_middle():
    script, plan = fixtures()
    first, modified, _ = midroll.prepare(script, plan, "Hãy bấm thích và đăng ký Màn Kể.", 15)
    with pytest.raises(ValueError, match="đã có"):
        midroll.prepare(first, modified, "Hãy bấm thích và đăng ký Màn Kể.", 15)
    plan["clips"][2]["duration_seconds"] = 250
    plan["clips"][3]["start_seconds"] = 460
    with pytest.raises(ValueError, match="50%"):
        midroll.prepare(script, plan, "Hãy bấm thích và đăng ký Màn Kể.", 15)


def test_stage_keeps_last_video_and_requires_script_review(tmp_path):
    import json
    from movie_review_factory import pipeline
    from movie_review_factory.models import JobConfig
    script, plan = fixtures()
    pipeline.create_job(tmp_path, JobConfig(job_id="cta"))
    manifest = pipeline.load_manifest(tmp_path)
    for stage in manifest.stages:
        if stage.stage == "tts":
            break
        stage.status = "ready"
    pipeline.save_manifest(tmp_path, manifest)
    (tmp_path / "script.json").write_text(json.dumps(script), encoding="utf-8")
    (tmp_path / "scene_plan.json").write_text(json.dumps(plan), encoding="utf-8")
    (tmp_path / "final.mp4").write_bytes(b"old-final")
    result = midroll.stage(tmp_path, "Eon tua giờ, hãy bấm thích và đăng ký Màn Kể.", 15)
    assert result["start_seconds"] == 300
    assert (tmp_path / "final.pre-midroll.mp4").read_bytes() == b"old-final"
    assert (tmp_path / "final.mp4").read_bytes() == b"old-final"
    staged = json.loads((tmp_path / "script.json").read_text(encoding="utf-8"))
    assert staged["approved"] is False
    assert staged["sections"][3]["midroll"] is True
    assert pipeline.load_manifest(tmp_path).stage("tts").status == "pending"
    assert pipeline.load_manifest(tmp_path).stage("scene_plan").status == "ready"
    guarded = pipeline.run_job(tmp_path)
    assert guarded.stage("tts").status == "pending"
    assert (tmp_path / "final.mp4").read_bytes() == b"old-final"
