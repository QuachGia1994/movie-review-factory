from __future__ import annotations

import json
from pathlib import Path

import pytest

from movie_review_factory import scene_scoring, scene_validation, semantic_search
from movie_review_factory.media_store import MediaStore
from movie_review_factory.models import MediaAsset, Shot, VisualObservation


def test_scene_scoring_exposes_components_and_semantic_can_win() -> None:
    section = {"title": "Car chase", "narration": "A vehicle escapes through the storm"}
    scenes = [
        {
            "index": 1,
            "start_seconds": 0,
            "end_seconds": 10,
            "text": "car chase",
            "visual_description": "two people indoors",
        },
        {
            "index": 2,
            "start_seconds": 10,
            "end_seconds": 20,
            "text": "",
            "visual_description": "a red automobile speeds through rain",
            "visual_tags": ["wet road", "vehicle"],
            "visual_actions": ["speeding"],
        },
    ]

    ranked = scene_scoring.rank_scenes(
        section,
        scenes,
        anchor=0,
        semantic_scores={1: 0.05, 2: 0.95},
        limit=2,
    )

    assert ranked[0]["index"] == 2
    score = ranked[0]["candidate_score"]
    assert set(score) == {
        "scene_index",
        "dialogue",
        "visual",
        "semantic",
        "person",
        "chronology",
        "diversity_penalty",
        "total",
    }
    assert score["semantic"] > ranked[1]["candidate_score"]["semantic"]


def test_scene_scoring_penalizes_repeat_and_adjacent_scene() -> None:
    section = {"title": "Escape", "narration": "escape"}
    scenes = [
        {"index": 1, "text": "escape", "start_seconds": 0, "end_seconds": 10},
        {"index": 2, "text": "escape", "start_seconds": 10, "end_seconds": 20},
        {"index": 8, "text": "escape", "start_seconds": 70, "end_seconds": 80},
    ]

    ranked = scene_scoring.rank_scenes(
        section,
        scenes,
        anchor=0,
        semantic_scores={1: 0.8, 2: 0.8, 8: 0.8},
        previous_scene_indexes={1},
        limit=3,
    )

    by_index = {item["index"]: item["candidate_score"] for item in ranked}
    assert by_index[1]["diversity_penalty"] == 1.0
    assert by_index[2]["diversity_penalty"] == 0.45
    assert by_index[8]["diversity_penalty"] == 0.0
    assert ranked[0]["index"] == 8


def test_validation_report_compares_lexical_and_semantic_strategies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    scenes = [
        {"index": 1, "start_seconds": 0.0, "end_seconds": 10.0, "text": "indoor talk"},
        {"index": 2, "start_seconds": 10.0, "end_seconds": 20.0, "text": ""},
        {"index": 3, "start_seconds": 20.0, "end_seconds": 30.0, "text": ""},
    ]
    (job / "script.json").write_text(json.dumps({
        "sections": [{"title": "Storm chase", "narration": "A car escapes in heavy rain"}]
    }), encoding="utf-8")
    (job / "scenes.json").write_text(json.dumps({"scenes": scenes}), encoding="utf-8")

    with MediaStore(job / "media_index.sqlite3") as store:
        store.migrate()
        store.replace_index(
            MediaAsset(path=tmp_path / "owned.mp4", duration_seconds=30),
            [
                Shot(
                    media_asset_id=1,
                    start_seconds=item["start_seconds"],
                    end_seconds=item["end_seconds"],
                    label=f"Scene {item['index']}",
                )
                for item in scenes
            ],
            [],
        )
        shots = store.list_shots(1)
        store.replace_visual_observations([
            VisualObservation(
                shot_id=int(shots[2].id),
                description="A red automobile speeds through a rainstorm",
                tags=["vehicle", "rain"],
                actions=["speeding"],
            )
        ])
        _, other = semantic_search._as_blob([0.0, 1.0])
        _, wanted = semantic_search._as_blob([1.0, 0.0])
        store.replace_shot_embeddings([
            (int(shots[0].id), "indoor talk", "test-model", 2, other),
            (int(shots[1].id), "empty hall", "test-model", 2, other),
            (int(shots[2].id), "red automobile rainstorm", "test-model", 2, wanted),
        ])

    _, query_blob = semantic_search._as_blob([1.0, 0.0])
    monkeypatch.setattr(semantic_search, "model_name", lambda: "test-model")
    monkeypatch.setattr(semantic_search, "embed_query", lambda _: (2, query_blob))

    truth = job / "truth.json"
    truth.write_text(json.dumps({
        "sections": [{"expected_scene_indexes": [3]}]
    }), encoding="utf-8")

    report = scene_validation.compare_scene_strategies(job, truth_path=truth)

    assert report["semantic_identity"]["top_scene_indexes"] == [3]
    assert report["semantic_identity"]["top1_hit_rate"] == 1.0
    assert report["semantic_identity"]["candidate_hit_rate"] == 1.0
    assert report["sections"][0]["semantic_top"]["candidate_score"]["semantic"] == 1.0
