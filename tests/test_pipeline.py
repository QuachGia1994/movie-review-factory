import json
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

    # media stages skipped honestly
    for stage in ("transcript", "scenes", "tts", "alignment", "render", "qa"):
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

    # Reset publish stage to pending so runner picks it up
    manifest = load_manifest(job_dir)
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
