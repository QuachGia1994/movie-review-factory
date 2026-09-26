from __future__ import annotations

import re
from dataclasses import asdict, dataclass


_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _tokens(value: object) -> set[str]:
    return set(_TOKEN_RE.findall(str(value or "").casefold()))


def _bounded(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass(frozen=True)
class SceneScore:
    scene_index: int
    dialogue: float
    visual: float
    semantic: float
    person: float
    chronology: float
    diversity_penalty: float
    total: float

    def model_dump(self) -> dict:
        return asdict(self)


def score_scene(
    section: dict,
    scene: dict,
    *,
    position: int,
    scene_count: int,
    anchor: int,
    semantic_score: float | None = None,
    previous_scene_indexes: set[int] | None = None,
) -> SceneScore:
    title = _tokens(section.get("title"))
    narration = _tokens(section.get("narration"))
    section_words = title | narration

    dialogue_words = _tokens(scene.get("text"))
    story_terms = [
        str(value)
        for item in scene.get("story_entities") or []
        if isinstance(item, dict)
        for value in (item.get("type"), item.get("label"))
        if value
    ]
    story_terms.extend(
        str(value)
        for item in scene.get("story_relations") or []
        if isinstance(item, dict)
        for value in (
            item.get("subject_label"),
            item.get("predicate"),
            item.get("object_label"),
        )
        if value
    )
    visual_words = _tokens(" ".join([
        str(scene.get("visual_description") or ""),
        *[str(value) for value in scene.get("visual_tags") or []],
        *[str(value) for value in scene.get("visual_people") or []],
        *[str(value) for value in scene.get("visual_actions") or []],
        *story_terms,
    ]))
    person_words = _tokens(" ".join(str(value) for value in scene.get("person_tracks") or []))

    dialogue = _bounded(
        (2.0 * len(title & dialogue_words) + len(narration & dialogue_words))
        / max(1.0, 2.0 * len(title) + len(narration))
    )
    visual = _bounded(
        (2.5 * len(title & visual_words) + len(narration & visual_words))
        / max(1.0, 2.5 * len(title) + len(narration))
    )
    person = _bounded(
        len(section_words & person_words) / max(1.0, len(section_words))
    )
    semantic = _bounded(((semantic_score or -1.0) + 1.0) / 2.0) if semantic_score is not None else 0.0
    chronology = 1.0 - _bounded(abs(position - anchor) / max(1, scene_count - 1))

    previous = previous_scene_indexes or set()
    diversity_penalty = 0.0
    scene_index = int(scene.get("index") or position + 1)
    if scene_index in previous:
        diversity_penalty += 1.0
    elif any(abs(scene_index - prior) <= 1 for prior in previous):
        diversity_penalty += 0.45

    total = (
        0.38 * semantic
        + 0.27 * visual
        + 0.16 * dialogue
        + 0.07 * person
        + 0.12 * chronology
        - 0.35 * diversity_penalty
    )
    return SceneScore(
        scene_index=scene_index,
        dialogue=round(dialogue, 6),
        visual=round(visual, 6),
        semantic=round(semantic, 6),
        person=round(person, 6),
        chronology=round(chronology, 6),
        diversity_penalty=round(diversity_penalty, 6),
        total=round(total, 6),
    )


def rank_scenes(
    section: dict,
    scenes: list[dict],
    *,
    anchor: int,
    semantic_scores: dict[int, float] | None = None,
    previous_scene_indexes: set[int] | None = None,
    limit: int = 12,
) -> list[dict]:
    semantic_scores = semantic_scores or {}
    scored: list[tuple[SceneScore, int, dict]] = []
    for position, scene in enumerate(scenes):
        index = int(scene.get("index") or position + 1)
        score = score_scene(
            section,
            scene,
            position=position,
            scene_count=len(scenes),
            anchor=anchor,
            semantic_score=semantic_scores.get(index),
            previous_scene_indexes=previous_scene_indexes,
        )
        enriched = dict(scene)
        enriched["candidate_score"] = score.model_dump()
        scored.append((score, position, enriched))

    scored.sort(key=lambda item: (-item[0].total, abs(item[1] - anchor), item[1]))
    return [item[2] for item in scored[:max(1, limit)]]


def summarize_selection(groups: list[list[dict]]) -> dict:
    selected = [
        int(group[0]["index"])
        for group in groups
        if group and isinstance(group[0].get("index"), int)
    ]
    repeats = len(selected) - len(set(selected))
    adjacent = sum(1 for left, right in zip(selected, selected[1:]) if abs(left - right) <= 1)
    totals = [
        float(group[0].get("candidate_score", {}).get("total", 0.0))
        for group in groups if group
    ]
    return {
        "sections": len(groups),
        "top_scene_indexes": selected,
        "repeat_count": repeats,
        "adjacent_count": adjacent,
        "mean_top_score": round(sum(totals) / len(totals), 6) if totals else 0.0,
    }
