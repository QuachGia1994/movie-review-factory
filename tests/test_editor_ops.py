from __future__ import annotations

import json
from pathlib import Path

import pytest

from movie_review_factory import editor_ops, pipeline, semantic_search
from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import JobConfig, MediaAsset, Shot, VisualObservation


def _editor_job(tmp_path: Path) -> Path:
    root = tmp_path / "edit"
    source = root / "source.mp4"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"owned")
    pipeline.create_job(root, JobConfig(job_id="edit", source_video=source))
    (root / "script.json").write_text(json.dumps({
        "sections": [
            {"title": "Chase", "narration": "red car racing rain", "duration_seconds": 20},
            {"title": "Hospital", "narration": "Person 1 waits at hospital", "duration_seconds": 10},
        ]
    }), encoding="utf-8")
    scenes = [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 10.0, "text": "talk"},
        {"index": 2, "start_seconds": 10.0, "end_seconds": 20.0, "text": ""},
        {"index": 3, "start_seconds": 20.0, "end_seconds": 30.0, "text": ""},
        {"index": 4, "start_seconds": 30.0, "end_seconds": 40.0, "text": ""},
    ]
    (root / "scenes.json").write_text(json.dumps({
        "source_video": str(source), "duration_seconds": 40.0, "scenes": scenes,
    }), encoding="utf-8")
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=source, duration_seconds=40),
            [
                Shot(media_asset_id=1, start_seconds=item["start_seconds"], end_seconds=item["end_seconds"], label=f"Scene {item['index']}")
                for item in scenes
            ],
            [],
        )
        shots = store.list_shots(1)
        store.replace_visual_observations([
            VisualObservation(shot_id=int(shots[0].id), description="indoor talk"),
            VisualObservation(shot_id=int(shots[1].id), description="red car racing in rain", actions=["racing"]),
            VisualObservation(shot_id=int(shots[2].id), description="red car turns onto another wet street", actions=["driving"]),
            VisualObservation(shot_id=int(shots[3].id), description="hospital waiting room", tags=["hospital"]),
        ])
        _, talk = semantic_search._as_blob([0.0, 1.0])
        _, chase = semantic_search._as_blob([1.0, 0.0])
        _, chase2 = semantic_search._as_blob([0.95, 0.05])
        _, hospital = semantic_search._as_blob([0.2, 0.8])
        store.replace_shot_embeddings([
            (int(shots[0].id), "talk", "test-model", 2, talk),
            (int(shots[1].id), "red car rain", "test-model", 2, chase),
            (int(shots[2].id), "red car wet street", "test-model", 2, chase2),
            (int(shots[3].id), "hospital waiting", "test-model", 2, hospital),
        ])
    (root / "scene_plan.json").write_text(json.dumps({
        "job_id": "edit",
        "total_seconds": 30,
        "clips": [
            {"section": "Chase", "section_index": 1, "shot_index": 1, "shot_count": 2, "start_seconds": 0, "duration_seconds": 10, "source_clip": {"start_seconds": 10, "end_seconds": 20}, "notes": ""},
            {"section": "Chase", "section_index": 1, "shot_index": 2, "shot_count": 2, "start_seconds": 10, "duration_seconds": 10, "source_clip": {"start_seconds": 10, "end_seconds": 20}, "notes": ""},
            {"section": "Hospital", "section_index": 2, "shot_index": 1, "shot_count": 1, "start_seconds": 20, "duration_seconds": 10, "source_clip": {"start_seconds": 30, "end_seconds": 40}, "notes": ""},
        ],
    }), encoding="utf-8")
    manifest = pipeline.load_manifest(root)
    manifest.stage("scene_plan").mark("ready", "manual plan")
    manifest.stage("render").mark("ready", "stale")
    pipeline.save_manifest(root, manifest)
    return root


@pytest.fixture()
def semantic_stub(monkeypatch: pytest.MonkeyPatch):
    _, chase = semantic_search._as_blob([1.0, 0.0])
    _, hospital = semantic_search._as_blob([0.2, 0.8])
    monkeypatch.setattr(semantic_search, "model_name", lambda: "test-model")
    monkeypatch.setattr(
        semantic_search,
        "embed_query",
        lambda text: (2, hospital if "hospital" in text.casefold() else chase),
    )


def test_timeline_edit_supports_lock_trim_replace_and_same_section_reorder(tmp_path: Path) -> None:
    root = _editor_job(tmp_path)
    locked = editor_ops.edit_timeline(root, {"action": "lock", "clip_index": 0, "locked": True})
    assert locked["clips"][0]["locked"] is True
    trimmed = editor_ops.edit_timeline(root, {"action": "trim", "clip_index": 0, "start_seconds": 11, "end_seconds": 18})
    assert trimmed["clips"][0]["source_clip"] == {"start_seconds": 11.0, "end_seconds": 18.0}
    replaced = editor_ops.edit_timeline(root, {"action": "replace", "clip_index": 1, "shot_id": 3})
    assert replaced["clips"][1]["source_clip"] == {"start_seconds": 20.0, "end_seconds": 30.0}
    reordered = editor_ops.edit_timeline(root, {"action": "reorder", "from_index": 1, "to_index": 0})
    assert [clip["shot_index"] for clip in reordered["clips"][:2]] == [1, 2]
    with pytest.raises(ValueError, match="same narration section"):
        editor_ops.edit_timeline(root, {"action": "reorder", "from_index": 0, "to_index": 2})
    assert pipeline.load_manifest(root).stage("render").status == "pending"


def test_find_similar_and_auto_broll_replace_repeated_clip(
    tmp_path: Path, semantic_stub: None,
) -> None:
    root = _editor_job(tmp_path)
    similar = editor_ops.similar_scenes(root, 2, limit=2)
    assert similar[0]["shot_id"] == 3

    preview = editor_ops.auto_replace_broll(root, apply=False)
    assert preview["replacements"][0]["clip_index"] == 1
    assert preview["replacements"][0]["shot_id"] == 3

    applied = editor_ops.auto_replace_broll(root, apply=True)
    assert applied["replacements"][0]["shot_id"] == 3
    plan = json.loads((root / "scene_plan.json").read_text(encoding="utf-8"))
    assert plan["clips"][1]["source_clip"] == {"start_seconds": 20.0, "end_seconds": 30.0}


def test_auto_broll_replaces_unique_but_semantically_weak_clip(
    tmp_path: Path, semantic_stub: None,
) -> None:
    root = _editor_job(tmp_path)
    plan_path = root / "scene_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["clips"][0]["source_clip"] = {"start_seconds": 0.0, "end_seconds": 10.0}
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    preview = editor_ops.auto_replace_broll(root, apply=False)

    first = next(item for item in preview["replacements"] if item["clip_index"] == 0)
    assert first["reason"] == "weak_semantic_match"
    assert first["current_semantic_score"] == pytest.approx(0.0)
    assert first["shot_id"] == 3


def test_section_regenerate_preserves_locked_clip(
    tmp_path: Path, semantic_stub: None,
) -> None:
    root = _editor_job(tmp_path)
    editor_ops.edit_timeline(root, {"action": "lock", "clip_index": 0, "locked": True})
    before = editor_ops.timeline(root)["clips"][0]["source_clip"]
    result = editor_ops.regenerate_section(root, 1, instruction="more action")
    clips = result["timeline"]["clips"]
    assert clips[0]["source_clip"] == before
    assert clips[0]["locked"] is True
    assert clips[1]["source_clip"] != {"start_seconds": 10.0, "end_seconds": 20.0}
