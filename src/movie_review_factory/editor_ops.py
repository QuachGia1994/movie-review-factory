from __future__ import annotations

import json
from pathlib import Path

from . import pipeline, scene_scoring, semantic_search
from .media_store import MediaStore


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path.name)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _range_key(item: dict) -> tuple[float, float]:
    return (
        round(float(item.get("start_seconds") or 0.0), 6),
        round(float(item.get("end_seconds") or 0.0), 6),
    )


def _load_index(root: Path) -> tuple[dict, list[dict], dict[int, int], dict[int, dict]]:
    scenes_doc = _read_json(root / "scenes.json")
    scenes = [item for item in scenes_doc.get("scenes", []) if isinstance(item, dict)]
    database = root / "media_index.sqlite3"
    if not database.is_file():
        raise FileNotFoundError("media_index.sqlite3")
    with MediaStore(database) as store:
        store.migrate()
        shots = store.list_shots(1)
    range_to_scene = {
        _range_key(scene): int(scene["index"])
        for scene in scenes if isinstance(scene.get("index"), int)
    }
    shot_to_scene: dict[int, int] = {}
    shot_by_id: dict[int, dict] = {}
    for shot in shots:
        if not shot.id:
            continue
        payload = shot.model_dump(mode="json")
        shot_by_id[int(shot.id)] = payload
        scene_index = range_to_scene.get(_range_key(payload))
        if scene_index is not None:
            shot_to_scene[int(shot.id)] = scene_index
    return scenes_doc, scenes, shot_to_scene, shot_by_id


def timeline(root: Path) -> dict:
    plan = _read_json(root / "scene_plan.json")
    clips = [dict(item) for item in plan.get("clips", []) if isinstance(item, dict)]
    for index, clip in enumerate(clips):
        clip["timeline_index"] = index
        clip["locked"] = bool(clip.get("locked", False))
    return {
        "clips": clips,
        "total_seconds": float(plan.get("total_seconds") or 0.0),
        "generator": plan.get("generator"),
    }


def _recompute_clip_positions(clips: list[dict]) -> None:
    cursor = 0.0
    per_section: dict[int, list[dict]] = {}
    for clip in clips:
        clip["start_seconds"] = cursor
        cursor += max(0.0, float(clip.get("duration_seconds") or 0.0))
        per_section.setdefault(int(clip.get("section_index") or 0), []).append(clip)
    for group in per_section.values():
        count = len(group)
        for index, clip in enumerate(group, start=1):
            clip["shot_index"] = index
            clip["shot_count"] = count


def edit_timeline(root: Path, operation: dict) -> dict:
    plan_path = root / "scene_plan.json"
    plan = _read_json(plan_path)
    clips = [dict(item) for item in plan.get("clips", []) if isinstance(item, dict)]
    action = str(operation.get("action") or "").strip().casefold()
    index = int(operation.get("clip_index", -1))
    if action != "reorder" and not 0 <= index < len(clips):
        raise ValueError("clip_index is out of range")

    if action == "lock":
        clips[index]["locked"] = bool(operation.get("locked", True))
    elif action == "trim":
        source = clips[index].get("source_clip")
        if not isinstance(source, dict):
            raise ValueError("clip has no source range")
        start = float(operation.get("start_seconds"))
        end = float(operation.get("end_seconds"))
        duration = float(_read_json(root / "scenes.json").get("duration_seconds") or 0.0)
        if start < 0 or end <= start or (duration > 0 and end > duration):
            raise ValueError("invalid source trim range")
        clips[index]["source_clip"] = {"start_seconds": start, "end_seconds": end}
    elif action == "replace":
        shot_id = int(operation.get("shot_id", 0))
        with MediaStore(root / "media_index.sqlite3") as store:
            store.migrate()
            shot = store.get_shot(shot_id)
        if shot is None:
            raise ValueError("replacement shot was not found")
        clips[index]["source_clip"] = {
            "start_seconds": float(shot.start_seconds),
            "end_seconds": float(shot.end_seconds),
        }
        clips[index]["notes"] = str(operation.get("notes") or "manual replacement")
    elif action == "reorder":
        source_index = int(operation.get("from_index", -1))
        target_index = int(operation.get("to_index", -1))
        if not (0 <= source_index < len(clips) and 0 <= target_index < len(clips)):
            raise ValueError("timeline reorder index is out of range")
        if clips[source_index].get("section_index") != clips[target_index].get("section_index"):
            raise ValueError("clips can only be reordered inside the same narration section")
        clip = clips.pop(source_index)
        clips.insert(target_index, clip)
    else:
        raise ValueError("unsupported timeline action")

    _recompute_clip_positions(clips)
    plan["clips"] = clips
    plan["total_seconds"] = sum(float(item.get("duration_seconds") or 0.0) for item in clips)
    _write_json(plan_path, plan)
    pipeline.invalidate_downstream(root, "scene_plan")
    return timeline(root)


def similar_scenes(root: Path, shot_id: int, *, limit: int = 8) -> list[dict]:
    database = root / "media_index.sqlite3"
    with MediaStore(database) as store:
        store.migrate()
        hits = semantic_search.similar_shots(store, shot_id, limit=limit)
        visual = {int(item["shot_id"]): item for item in store.list_visual_observations()}
        people = {
            int(shot.id): store.person_labels_for_shot(int(shot.id))
            for shot in store.list_shots(1) if shot.id
        }
    return [
        {
            **item,
            "semantic_score": item["score"],
            "visual_description": visual.get(int(item["shot_id"]), {}).get("description"),
            "person_tracks": people.get(int(item["shot_id"]), []),
        }
        for item in hits
    ]


def _section_query(root: Path, section_index: int, extra: str = "") -> tuple[dict, str]:
    script = _read_json(root / "script.json")
    sections = [item for item in script.get("sections", []) if isinstance(item, dict)]
    if not 1 <= section_index <= len(sections):
        raise ValueError("section_index is out of range")
    section = sections[section_index - 1]
    query = " ".join([
        str(section.get("title") or ""),
        str(section.get("narration") or ""),
        str(extra or ""),
    ]).strip()
    return section, query


def broll_suggestions(root: Path, clip_index: int, *, limit: int = 6) -> list[dict]:
    plan = _read_json(root / "scene_plan.json")
    clips = [item for item in plan.get("clips", []) if isinstance(item, dict)]
    if not 0 <= clip_index < len(clips):
        raise ValueError("clip_index is out of range")
    clip = clips[clip_index]
    section_index = int(clip.get("section_index") or 0)
    section, query = _section_query(root, section_index)
    database = root / "media_index.sqlite3"
    scenes_doc, scenes, shot_to_scene, shot_by_id = _load_index(root)
    del scenes_doc, section
    used_ranges = {
        _range_key(item["source_clip"])
        for item in clips
        if isinstance(item.get("source_clip"), dict)
    }
    current_range = _range_key(clip.get("source_clip") or {})
    with MediaStore(database) as store:
        store.migrate()
        has_embeddings = bool(store.list_shot_embeddings(semantic_search.model_name()))
        hits = (
            semantic_search.search_store(store, query, limit=max(limit * 4, 24))
            if has_embeddings else []
        )
        visual = {int(item["shot_id"]): item for item in store.list_visual_observations()}
    suggestions: list[dict] = []
    for hit in hits:
        if hit["kind"] != "visual":
            continue
        shot_id = int(hit["shot_id"])
        shot = shot_by_id.get(shot_id)
        if not shot:
            continue
        candidate_range = _range_key(shot)
        if candidate_range == current_range or candidate_range in used_ranges:
            continue
        scene_index = shot_to_scene.get(shot_id)
        scene = next((item for item in scenes if int(item.get("index") or 0) == scene_index), {})
        suggestions.append({
            "shot_id": shot_id,
            "scene_index": scene_index,
            "start_seconds": shot["start_seconds"],
            "end_seconds": shot["end_seconds"],
            "semantic_score": float(hit["score"]),
            "visual_description": visual.get(shot_id, {}).get("description"),
            "candidate_score": scene.get("candidate_score"),
        })
        if len(suggestions) >= limit:
            break
    return suggestions


def auto_replace_broll(root: Path, *, apply: bool = False) -> dict:
    plan = _read_json(root / "scene_plan.json")
    clips = [dict(item) for item in plan.get("clips", []) if isinstance(item, dict)]
    _, _, _, shot_by_id = _load_index(root)
    range_to_shot = {_range_key(shot): shot_id for shot_id, shot in shot_by_id.items()}
    section_scores: dict[int, dict[int, float]] = {}

    def scores_for(section_index: int) -> dict[int, float]:
        if section_index in section_scores:
            return section_scores[section_index]
        _, query = _section_query(root, section_index)
        with MediaStore(root / "media_index.sqlite3") as store:
            store.migrate()
            has_embeddings = bool(store.list_shot_embeddings(semantic_search.model_name()))
            hits = (
                semantic_search.search_store(
                    store,
                    query,
                    limit=max(40, len(shot_by_id)),
                    min_score=-1.0,
                )
                if has_embeddings else []
            )
        section_scores[section_index] = {
            int(hit["shot_id"]): float(hit["score"])
            for hit in hits if hit["kind"] == "visual"
        }
        return section_scores[section_index]

    seen: set[tuple[float, float]] = set()
    replacements: list[dict] = []
    for index, clip in enumerate(clips):
        if bool(clip.get("locked")):
            source = clip.get("source_clip")
            if isinstance(source, dict):
                seen.add(_range_key(source))
            continue
        source = clip.get("source_clip")
        key = _range_key(source) if isinstance(source, dict) else None
        repeated_or_missing = key is None or key in seen
        if key is not None:
            seen.add(key)
        suggestions = broll_suggestions(root, index, limit=3)
        if not suggestions:
            continue
        chosen = suggestions[0]
        current_shot_id = range_to_shot.get(key) if key is not None else None
        current_score = (
            scores_for(int(clip.get("section_index") or 0)).get(current_shot_id)
            if current_shot_id is not None else None
        )
        weak_semantic = (
            current_score is not None
            and current_score < 0.55
            and float(chosen["semantic_score"]) >= current_score + 0.18
        )
        if not repeated_or_missing and not weak_semantic:
            continue
        reason = "repeated_or_missing" if repeated_or_missing else "weak_semantic_match"
        replacements.append({
            "clip_index": index,
            "reason": reason,
            "current_semantic_score": current_score,
            **chosen,
        })
        if apply:
            clips[index]["source_clip"] = {
                "start_seconds": chosen["start_seconds"],
                "end_seconds": chosen["end_seconds"],
            }
            clips[index]["notes"] = f"automatic B-roll replacement: {reason}"
    if apply and replacements:
        plan["clips"] = clips
        _write_json(root / "scene_plan.json", plan)
        pipeline.invalidate_downstream(root, "scene_plan")
    return {"apply": apply, "replacements": replacements}


def regenerate_section(root: Path, section_index: int, *, instruction: str = "") -> dict:
    plan_path = root / "scene_plan.json"
    plan = _read_json(plan_path)
    clips = [dict(item) for item in plan.get("clips", []) if isinstance(item, dict)]
    section_clips = [
        (index, clip)
        for index, clip in enumerate(clips)
        if int(clip.get("section_index") or 0) == section_index
    ]
    if not section_clips:
        raise ValueError("section has no timeline clips")
    section, query = _section_query(root, section_index, instruction)
    scenes_doc, scenes, shot_to_scene, shot_by_id = _load_index(root)
    del scenes_doc
    range_to_shot = {_range_key(shot): shot_id for shot_id, shot in shot_by_id.items()}
    locked_scene_indexes: set[int] = set()
    for _, clip in section_clips:
        if not clip.get("locked") or not isinstance(clip.get("source_clip"), dict):
            continue
        shot_id = range_to_shot.get(_range_key(clip["source_clip"]))
        if shot_id in shot_to_scene:
            locked_scene_indexes.add(shot_to_scene[shot_id])
    semantic_scores: dict[int, float] = {}
    with MediaStore(root / "media_index.sqlite3") as store:
        store.migrate()
        has_embeddings = bool(store.list_shot_embeddings(semantic_search.model_name()))
        hits = (
            semantic_search.search_store(store, query, limit=max(24, len(scenes)))
            if has_embeddings else []
        )
        for hit in hits:
            if hit["kind"] == "visual" and int(hit["shot_id"]) in shot_to_scene:
                semantic_scores[shot_to_scene[int(hit["shot_id"])]] = float(hit["score"])
    anchor = max(0, min(len(scenes) - 1, round((section_index - 1) * (len(scenes) - 1) / max(1, len(_read_json(root / "script.json").get("sections", [])) - 1))))
    ranked = scene_scoring.rank_scenes(
        {**section, "narration": query},
        scenes,
        anchor=anchor,
        semantic_scores=semantic_scores,
        previous_scene_indexes=locked_scene_indexes,
        limit=max(len(section_clips) * 3, 12),
    )
    candidates = [item for item in ranked if int(item["index"]) not in locked_scene_indexes]
    replacement_cursor = 0
    for clip_index, clip in section_clips:
        if clip.get("locked"):
            continue
        if replacement_cursor >= len(candidates):
            break
        scene = candidates[replacement_cursor]
        replacement_cursor += 1
        clip["source_clip"] = {
            "start_seconds": float(scene["start_seconds"]),
            "end_seconds": float(scene["end_seconds"]),
        }
        clip["notes"] = (
            "section regenerate"
            + (f": {instruction.strip()}" if instruction.strip() else "")
        )
        clips[clip_index] = clip
    plan["clips"] = clips
    _write_json(plan_path, plan)
    pipeline.invalidate_downstream(root, "scene_plan")
    return {
        "section_index": section_index,
        "instruction": instruction,
        "timeline": timeline(root),
    }
