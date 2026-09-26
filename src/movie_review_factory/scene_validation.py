from __future__ import annotations

import json
from pathlib import Path

from . import scene_scoring, semantic_search
from .media_store import MediaStore


def _range_key(item: dict) -> tuple[float, float]:
    return (
        round(float(item.get("start_seconds") or 0.0), 6),
        round(float(item.get("end_seconds") or 0.0), 6),
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _truth_metrics(groups: list[list[dict]], truth: dict | None) -> dict:
    if not truth:
        return {"truth_sections": 0, "top1_hits": 0, "candidate_hits": 0}
    expected = truth.get("sections") or []
    top1_hits = 0
    candidate_hits = 0
    counted = 0
    for group, section_truth in zip(groups, expected):
        wanted = {
            int(value)
            for value in section_truth.get("expected_scene_indexes", [])
            if isinstance(value, int)
        }
        if not wanted:
            continue
        counted += 1
        indexes = [int(item["index"]) for item in group if isinstance(item.get("index"), int)]
        if indexes and indexes[0] in wanted:
            top1_hits += 1
        if wanted.intersection(indexes):
            candidate_hits += 1
    return {
        "truth_sections": counted,
        "top1_hits": top1_hits,
        "candidate_hits": candidate_hits,
        "top1_hit_rate": round(top1_hits / counted, 6) if counted else None,
        "candidate_hit_rate": round(candidate_hits / counted, 6) if counted else None,
    }


def compare_scene_strategies(
    job_root: Path,
    *,
    truth_path: Path | None = None,
    limit: int = 12,
) -> dict:
    script = _load_json(job_root / "script.json")
    scenes_doc = _load_json(job_root / "scenes.json")
    sections = [item for item in script.get("sections", []) if isinstance(item, dict)]
    scenes = [item for item in scenes_doc.get("scenes", []) if isinstance(item, dict)]
    database = job_root / "media_index.sqlite3"
    if not sections or not scenes or not database.is_file():
        raise ValueError("job requires script sections, scenes, and media_index.sqlite3")

    lexical_groups: list[list[dict]] = []
    previous_top: set[int] = set()
    for section_index, section in enumerate(sections):
        anchor = round(section_index * (len(scenes) - 1) / max(len(sections) - 1, 1))
        group = scene_scoring.rank_scenes(
            section,
            scenes,
            anchor=anchor,
            previous_scene_indexes=previous_top,
            limit=limit,
        )
        lexical_groups.append(group)
        if group:
            previous_top.add(int(group[0]["index"]))

    semantic_groups: list[list[dict]] = []
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
        range_to_scene = {
            _range_key(scene): int(scene["index"])
            for scene in scenes if isinstance(scene.get("index"), int)
        }
        shot_to_scene = {
            int(shot.id): range_to_scene[_range_key(shot.model_dump())]
            for shot in shots
            if shot.id and _range_key(shot.model_dump()) in range_to_scene
        }
        previous_top = set()
        for section_index, section in enumerate(sections):
            query = " ".join([
                str(section.get("title") or ""),
                str(section.get("narration") or ""),
            ]).strip()
            semantic_scores: dict[int, float] = {}
            if query:
                for hit in semantic_search.search_store(
                    store, query, limit=max(limit * 3, len(scenes))
                ):
                    if hit["kind"] != "visual":
                        continue
                    scene_index = shot_to_scene.get(int(hit["shot_id"]))
                    if scene_index is not None:
                        semantic_scores[scene_index] = float(hit["score"])
            anchor = round(section_index * (len(scenes) - 1) / max(len(sections) - 1, 1))
            group = scene_scoring.rank_scenes(
                section,
                scenes,
                anchor=anchor,
                semantic_scores=semantic_scores,
                previous_scene_indexes=previous_top,
                limit=limit,
            )
            semantic_groups.append(group)
            if group:
                previous_top.add(int(group[0]["index"]))

    truth = _load_json(truth_path) if truth_path and truth_path.is_file() else None
    return {
        "job": job_root.name,
        "semantic_model": semantic_search.model_name(),
        "section_count": len(sections),
        "scene_count": len(scenes),
        "lexical_visual": {
            **scene_scoring.summarize_selection(lexical_groups),
            **_truth_metrics(lexical_groups, truth),
        },
        "semantic_identity": {
            **scene_scoring.summarize_selection(semantic_groups),
            **_truth_metrics(semantic_groups, truth),
        },
        "sections": [
            {
                "section_index": index + 1,
                "title": str(section.get("title") or ""),
                "lexical_top": lexical_groups[index][0] if lexical_groups[index] else None,
                "semantic_top": semantic_groups[index][0] if semantic_groups[index] else None,
            }
            for index, section in enumerate(sections)
        ],
    }


def write_validation_report(
    job_root: Path,
    *,
    truth_path: Path | None = None,
    output: Path | None = None,
) -> Path:
    report = compare_scene_strategies(job_root, truth_path=truth_path)
    target = output or (job_root / "scene_validation.json")
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
