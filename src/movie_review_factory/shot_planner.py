"""Beat-level shot planning: map every narration beat to the source scene it describes.

Two AGY passes per script section, sections in parallel:
1. locate - the beat's timeline window, from a compact whole-film summary of AGY frame descriptions;
2. pick   - one scene for each beat among the scenes of the located windows.
"""
from __future__ import annotations

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

AgentRunner = Callable[[str, str, dict, dict], "dict | None"]

BEAT_MIN_SECONDS = 4.0
WINDOW_TARGET_COUNT = 48
WINDOW_MIN_SECONDS = 60.0
WINDOW_SUMMARY_CHARS = 280
SCENE_SUMMARY_CHARS = 70
PICK_CANDIDATE_LIMIT = 110
PARALLEL_SECTIONS = 4
# Pacing (see pace_shots): cuts land every ~3 s and the picture advances through film time at the recap's compression rate.
SHOT_TARGET_SECONDS = 3.0
SHOT_SECONDS_ENV = "MRF_SHOT_SECONDS"
MIN_MONTAGE_SCENE_SECONDS = 0.8
MAX_ANCHOR_GAP_SECONDS = 300.0
COMPRESSION_LIMITS = (1.5, 12.0)

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")

_LOCATE_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"beat_id": {"type": "integer"}, "window_id": {"type": "integer"}},
                "required": ["beat_id", "window_id"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["beats"],
    "additionalProperties": False,
}

_PICK_SCHEMA = {
    "type": "object",
    "properties": {
        "shots": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "beat_id": {"type": "integer"},
                    "scene_index": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["beat_id", "scene_index", "rationale"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["shots"],
    "additionalProperties": False,
}


class ShotPlanError(ValueError):
    """The beat planner could not produce a usable plan; callers fall back to section shots."""


def split_beats(text: str, words_per_minute: int) -> list[str]:
    """Sentences merged until each beat lasts at least ``BEAT_MIN_SECONDS`` when spoken."""
    min_words = max(1, round(words_per_minute * BEAT_MIN_SECONDS / 60))
    beats: list[str] = []
    for sentence in (part.strip() for part in _SENTENCE_END.split(str(text or "").strip())):
        if not sentence:
            continue
        if beats and len(beats[-1].split()) < min_words:
            beats[-1] = f"{beats[-1]} {sentence}"
        else:
            beats.append(sentence)
    if len(beats) > 1 and len(beats[-1].split()) < min_words / 2:
        tail = beats.pop()
        beats[-1] = f"{beats[-1]} {tail}"
    return beats


def _clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _scene_summary(scene: dict, limit: int) -> str:
    text = str(scene.get("visual_description") or scene.get("text") or "").strip()
    return text[:limit]


def _scene_people(scene: dict) -> list[str]:
    names = [str(item.get("label")) for item in scene.get("story_entities") or []
             if isinstance(item, dict) and item.get("type") == "person" and item.get("label")]
    names.extend(str(label) for label in scene.get("person_tracks") or [] if label)
    return list(dict.fromkeys(names))


def timeline_windows(scenes: list[dict]) -> list[dict]:
    """Group scenes into at most ~``WINDOW_TARGET_COUNT`` consecutive time windows."""
    ordered = sorted(scenes, key=lambda scene: float(scene.get("start_seconds") or 0.0))
    if not ordered:
        return []
    duration = float(ordered[-1].get("end_seconds") or 0.0)
    span = max(WINDOW_MIN_SECONDS, duration / WINDOW_TARGET_COUNT)
    windows: list[dict] = []
    for scene in ordered:
        slot = int(float(scene.get("start_seconds") or 0.0) // span)
        if not windows or windows[-1]["slot"] != slot:
            windows.append({"slot": slot, "scenes": []})
        windows[-1]["scenes"].append(scene)
    result = []
    for number, window in enumerate(windows, start=1):
        members = window["scenes"]
        people = list(dict.fromkeys(name for scene in members for name in _scene_people(scene)))
        summary = "; ".join(filter(None, (_scene_summary(scene, SCENE_SUMMARY_CHARS) for scene in members)))
        result.append({
            "window_id": number,
            "time": f"{_clock(members[0].get('start_seconds') or 0)}-{_clock(members[-1].get('end_seconds') or 0)}",
            "people": people[:6],
            "summary": summary[:WINDOW_SUMMARY_CHARS],
            "scene_indexes": [int(scene["index"]) for scene in members],
        })
    return result


def _locate(run_agent: AgentRunner, movie_title: str, title: str, beats: list[str],
            windows: list[dict]) -> dict[int, int]:
    context = {
        "movie_title": movie_title,
        "section_title": title,
        "narration_beats": [{"beat_id": number, "text": text} for number, text in enumerate(beats, start=1)],
        "scenes_timeline": [{key: window[key] for key in ("window_id", "time", "people", "summary")}
                             for window in windows],
    }
    agent = run_agent(
        "scene_locate",
        "For every narration beat, return the scenes_timeline window whose footage best shows what that beat "
        "says. Recap beats map to the window where the event happens in the film; analysis or opinion "
        "beats map to a window that illustrates the point (the character, place or moment discussed). "
        "Window summaries are AGY frame descriptions of the footage, in film order.",
        context,
        _LOCATE_SCHEMA,
    )
    valid = {window["window_id"] for window in windows}
    located: dict[int, int] = {}
    for item in (agent or {}).get("beats") or []:
        if isinstance(item, dict) and item.get("window_id") in valid and isinstance(item.get("beat_id"), int):
            located.setdefault(int(item["beat_id"]), int(item["window_id"]))
    if not located:
        raise ShotPlanError(f"AGY could not locate the footage for section {title!r}")
    previous = min(located.values())
    for number in range(1, len(beats) + 1):
        previous = located.setdefault(number, previous)
    return located


def _candidate(scene: dict, window_id: int) -> dict:
    item = {
        "index": int(scene["index"]),
        "window_id": window_id,
        "time": _clock(scene.get("start_seconds") or 0),
        "visual": str(scene.get("visual_description") or "")[:220],
        "dialogue": str(scene.get("text") or "")[:90],
    }
    people = _scene_people(scene)
    if people:
        item["people"] = people[:4]
    actions = [str(action) for action in scene.get("visual_actions") or []][:3]
    if actions:
        item["actions"] = actions
    return item


def _pick(run_agent: AgentRunner, movie_title: str, title: str, beats: list[str],
          located: dict[int, int], windows: dict[int, dict], scenes: dict[int, dict]) -> dict[int, tuple[int, str]]:
    window_ids = sorted(set(located.values()))
    pool: list[dict] = []
    for window_id in window_ids:
        pool.extend(_candidate(scenes[index], window_id) for index in windows[window_id]["scene_indexes"])
    if len(pool) > PICK_CANDIDATE_LIMIT:
        step = len(pool) / PICK_CANDIDATE_LIMIT
        pool = [pool[int(position * step)] for position in range(PICK_CANDIDATE_LIMIT)]
    allowed = {item["index"] for item in pool}
    context = {
        "movie_title": movie_title,
        "section_title": title,
        "narration_beats": [
            {"beat_id": number, "window_id": located[number], "text": text}
            for number, text in enumerate(beats, start=1)
        ],
        "scene_candidates": pool,
    }
    agent = run_agent(
        "scene_pick",
        "Pick exactly one source scene for every narration beat so the picture shows what the beat says "
        "at that moment. Start from the beat's window_id; judge by the visual description, people and "
        "actions (dialogue is only supporting evidence). Use a different scene for each beat unless two "
        "beats talk about the very same moment. Rationale: one short English sentence naming what is on screen.",
        context,
        _PICK_SCHEMA,
    )
    picked: dict[int, tuple[int, str]] = {}
    for item in (agent or {}).get("shots") or []:
        if not isinstance(item, dict) or item.get("scene_index") not in allowed:
            continue
        beat = item.get("beat_id")
        if isinstance(beat, int) and 1 <= beat <= len(beats):
            picked.setdefault(beat, (int(item["scene_index"]), str(item.get("rationale") or "").strip()))
    for number in range(1, len(beats) + 1):
        if number not in picked:
            members = [index for index in windows[located[number]]["scene_indexes"] if index in allowed]
            members = members or windows[located[number]]["scene_indexes"]
            picked[number] = (members[len(members) // 2], "")
    return picked


def _spread_repeats(plan: list[list[list]], order: list[int], positions: dict[int, int]) -> None:
    """Move a reused scene to its nearest unused neighbour so the video does not show one moment twice."""
    used: set[int] = set()
    for shots in plan:
        for shot in shots:
            index = shot[0]
            if index in used:
                position = positions[index]
                for distance in range(1, 4):
                    swap = next((order[candidate] for candidate in (position + distance, position - distance)
                                 if 0 <= candidate < len(order) and order[candidate] not in used), None)
                    if swap is not None:
                        shot[0] = swap
                        break
            used.add(shot[0])


def shot_target_seconds() -> float:
    try:
        value = float(os.environ.get(SHOT_SECONDS_ENV, "") or SHOT_TARGET_SECONDS)
    except ValueError:
        return SHOT_TARGET_SECONDS
    return min(max(value, 1.5), 12.0)


def _spoken_seconds(section: dict, words_per_minute: int) -> float:
    words = len(str(section.get("narration") or "").split())
    if words:
        return words * 60.0 / max(words_per_minute, 1)
    return max(float(section.get("duration_seconds") or 0.0), 0.0)


def pace_shots(
    sections: list[dict],
    assignments: list[list[tuple]],
    scenes: list[dict],
    *,
    words_per_minute: int,
    shot_seconds: float | None = None,
) -> list[list[tuple]]:
    """Split every shot longer than ~``shot_seconds`` into a montage that tracks the narration.

    A recap tells ~90 film minutes in ~10, so one scene played in real time falls behind
    the voice. Each long shot keeps its matched scene first, then samples unused scenes
    between it and the next shot's scene (film time advances with the narration); when the
    next shot jumps back or far ahead it samples the stretch right after the scene, sized by
    the recap's compression rate. Sub-shots share the original weight and a ``beat`` key so
    the render can time the whole beat to its spoken words.
    """
    target = shot_seconds or shot_target_seconds()
    ordered = sorted(
        (scene for scene in scenes
         if isinstance(scene.get("index"), int)
         and float(scene.get("end_seconds") or 0) - float(scene.get("start_seconds") or 0) >= MIN_MONTAGE_SCENE_SECONDS),
        key=lambda scene: float(scene.get("start_seconds") or 0.0),
    )
    if not ordered:
        return assignments
    film_span = float(ordered[-1]["end_seconds"]) - float(ordered[0]["start_seconds"])
    spoken = [_spoken_seconds(section, words_per_minute) for section in sections]
    low, high = COMPRESSION_LIMITS
    compression = min(max(film_span / max(sum(spoken), 1.0), low), high)

    anchors: list[tuple[int, int, float, float, float]] = []
    for position, shots in enumerate(assignments):
        entries = [tuple(shot) + ({},) * (3 - len(shot)) for shot in shots]
        weights = [float(extras.get("weight") or 1.0) for _, _, extras in entries]
        for shot_position, (clip, _, _) in enumerate(entries):
            if not isinstance(clip, dict):
                continue
            slot = spoken[position] * weights[shot_position] / max(sum(weights), 1e-9)
            anchors.append((position, shot_position, float(clip["start_seconds"]),
                            float(clip["end_seconds"]), slot))
    used = {round(start, 3) for _, _, start, _, _ in anchors}
    montage: dict[tuple[int, int], list[dict]] = {}
    for number, (position, shot_position, start, end, slot) in enumerate(anchors):
        extra = math.ceil(slot / target - 0.25) - 1
        if extra <= 0:
            continue
        following = anchors[number + 1][2] if number + 1 < len(anchors) else None
        if following is not None and end < following <= end + MAX_ANCHOR_GAP_SECONDS:
            window_end = following
        else:
            window_end = end + slot * compression
        pool = [scene for scene in ordered
                if end - 0.01 <= float(scene["start_seconds"]) < window_end
                and round(float(scene["start_seconds"]), 3) not in used]
        picks: list[dict] = []
        for step in range(extra):
            if not pool:
                break
            goal = end + (window_end - end) * (step + 0.5) / extra
            choice = min(pool, key=lambda scene: abs(float(scene["start_seconds"]) - goal))
            pool.remove(choice)
            picks.append(choice)
        if picks:
            used.update(round(float(scene["start_seconds"]), 3) for scene in picks)
            montage[(position, shot_position)] = sorted(picks, key=lambda scene: float(scene["start_seconds"]))

    paced: list[list[tuple]] = []
    for position, shots in enumerate(assignments):
        out: list[tuple] = []
        for shot_position, shot in enumerate(shots):
            clip, rationale, extras = tuple(shot) + ({},) * (3 - len(shot))
            picks = montage.get((position, shot_position))
            if not picks:
                out.append(tuple(shot))
                continue
            share = float(extras.get("weight") or 1.0) / (len(picks) + 1)
            beat = f"{position + 1}:{shot_position + 1}"
            out.append((clip, rationale, {**extras, "weight": share, "beat": beat}))
            for scene in picks:
                out.append((
                    {"start_seconds": float(scene["start_seconds"]), "end_seconds": float(scene["end_seconds"])},
                    "Montage: follows the narration through the film",
                    {"weight": share, "beat": beat, "scene_index": int(scene["index"]), "montage": True},
                ))
        paced.append(out)
    return paced


def plan_beat_shots(
    sections: list[dict],
    scenes: list[dict],
    *,
    run_agent: AgentRunner,
    words_per_minute: int,
    movie_title: str = "",
) -> list[list[tuple[dict, str, dict]]]:
    """Per section: ``(source_clip, rationale, {"weight", "narration", "scene_index"})`` for each beat."""
    by_index = {int(scene["index"]): scene for scene in scenes if isinstance(scene.get("index"), int)}
    windows = timeline_windows(list(by_index.values()))
    if not windows:
        raise ShotPlanError("no indexed scenes to plan shots from")
    window_map = {window["window_id"]: window for window in windows}
    beats_by_section = [split_beats(str(section.get("narration") or ""), words_per_minute) for section in sections]

    def plan_section(position: int) -> list[list]:
        beats = beats_by_section[position]
        if not beats or sections[position].get("midroll"):
            return []
        title = str(sections[position].get("title") or "")
        located = _locate(run_agent, movie_title, title, beats, windows)
        picked = _pick(run_agent, movie_title, title, beats, located, window_map, by_index)
        return [[picked[number][0], picked[number][1], text] for number, text in enumerate(beats, start=1)]

    with ThreadPoolExecutor(max_workers=PARALLEL_SECTIONS) as executor:
        plan = list(executor.map(plan_section, range(len(sections))))
    order = [int(scene["index"]) for scene in sorted(
        by_index.values(), key=lambda scene: float(scene.get("start_seconds") or 0.0))]
    _spread_repeats(plan, order, {index: position for position, index in enumerate(order)})
    result = []
    for shots in plan:
        result.append([
            ({"start_seconds": float(by_index[index]["start_seconds"]),
              "end_seconds": float(by_index[index]["end_seconds"])},
             rationale,
             {"weight": max(1, len(text.split())), "narration": text, "scene_index": index})
            for index, rationale, text in shots
        ])
    return result
