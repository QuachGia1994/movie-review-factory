import pytest

from movie_review_factory import shot_planner
from movie_review_factory.pipeline import (
    _beat_timed_durations,
    _expand_scene_plan_clips,
    _extend_short_ranges,
    _spoken_word_starts,
)


def _scenes(count: int = 60) -> list[dict]:
    return [
        {"index": index, "start_seconds": (index - 1) * 10.0, "end_seconds": index * 10.0,
         "text": f"line {index}", "visual_description": f"frame {index}"}
        for index in range(1, count + 1)
    ]


def test_split_beats_merges_short_sentences_into_spoken_beats() -> None:
    text = "One two. Three four five six. Seven eight nine ten eleven. End."

    assert shot_planner.split_beats(text, 60) == [
        "One two. Three four five six.",
        "Seven eight nine ten eleven. End.",
    ]
    assert shot_planner.split_beats("   ", 60) == []


def test_timeline_windows_cover_the_film_in_order() -> None:
    windows = shot_planner.timeline_windows(_scenes())

    assert len(windows) == 10
    assert windows[0]["scene_indexes"] == [1, 2, 3, 4, 5, 6]
    assert windows[-1]["time"] == "09:00-10:00"
    assert windows[0]["summary"].startswith("frame 1; frame 2")


def test_plan_beat_shots_locates_then_picks_one_scene_per_beat() -> None:
    calls: list[tuple[str, dict]] = []

    def run_agent(stage: str, instruction: str, context: dict, schema: dict) -> dict:
        calls.append((stage, context))
        alpha = context["section_title"] == "A"
        if stage == "scene_locate":
            assert len(context["scenes_timeline"]) == 10
            return {"beats": [{"beat_id": 1, "window_id": 1}, {"beat_id": 2, "window_id": 2}]} if alpha \
                else {"beats": [{"beat_id": 1, "window_id": 10}]}
        if alpha:
            return {"shots": [{"beat_id": 1, "scene_index": 3, "rationale": "r1"},
                              {"beat_id": 2, "scene_index": 8, "rationale": "r2"}]}
        return {"shots": [{"beat_id": 1, "scene_index": 3, "rationale": "outside"},
                          {"beat_id": 2, "scene_index": 58, "rationale": "r4"}]}

    sections = [
        {"title": "A", "narration": "Alpha beat one here. Alpha beat two here."},
        {"title": "B", "narration": "Beta beat one here. Beta beat two here."},
    ]
    plan = shot_planner.plan_beat_shots(sections, _scenes(), run_agent=run_agent, words_per_minute=60)

    assert sorted(stage for stage, _ in calls) == ["scene_locate", "scene_locate", "scene_pick", "scene_pick"]
    assert plan[0] == [
        ({"start_seconds": 20.0, "end_seconds": 30.0}, "r1",
         {"weight": 4, "narration": "Alpha beat one here.", "scene_index": 3}),
        ({"start_seconds": 70.0, "end_seconds": 80.0}, "r2",
         {"weight": 4, "narration": "Alpha beat two here.", "scene_index": 8}),
    ]
    # Beat 1 of B picked a scene outside its window -> window middle (58); beat 2 then reuses 58 -> neighbour 59.
    assert [shot[2]["scene_index"] for shot in plan[1]] == [58, 59]
    assert plan[1][0][1] == ""


def test_plan_beat_shots_raises_when_footage_cannot_be_located() -> None:
    with pytest.raises(shot_planner.ShotPlanError, match="could not locate"):
        shot_planner.plan_beat_shots(
            [{"title": "A", "narration": "Something happens."}], _scenes(),
            run_agent=lambda *args: None, words_per_minute=60,
        )


def test_weighted_shots_split_section_time_by_beat_length() -> None:
    clip = {"start_seconds": 0.0, "end_seconds": 5.0}
    clips, total = _expand_scene_plan_clips(
        [{"title": "A", "duration_seconds": 30}],
        [[(clip, "a", {"weight": 1}), (clip, "b", {"weight": 2, "narration": "x"})]],
    )

    assert [item["duration_seconds"] for item in clips] == [10.0, 20.0]
    assert clips[1]["narration"] == "x" and total == 30.0


def test_long_source_range_reads_its_middle() -> None:
    ranges = [{"index": 0, "start_seconds": 100.0, "end_seconds": 190.0, "duration_seconds": 10.0}]

    _extend_short_ranges(ranges, 600.0)

    assert (ranges[0]["read_start_seconds"], ranges[0]["read_seconds"]) == (140.0, 10.0)


def _clip(index: int) -> dict:
    return {"start_seconds": (index - 1) * 10.0, "end_seconds": index * 10.0}


def test_pace_shots_turns_a_long_beat_into_a_forward_montage() -> None:
    sections = [{"title": "A", "narration": " ".join(["w"] * 60)}]  # 30 s at 120 wpm
    assignments = [[(_clip(1), "open", {"weight": 3.0}), (_clip(11), "later", {"weight": 1.0})]]

    paced = shot_planner.pace_shots(sections, assignments, _scenes(), words_per_minute=120, shot_seconds=3.0)

    shots = paced[0]
    first_beat = [shot for shot in shots if shot[2].get("beat") == "1:1"]
    assert shots[0][0] == _clip(1) and shots[0][1] == "open"
    assert len(first_beat) == 8  # 22.5 s slot -> anchor + 7 montage scenes
    starts = [shot[0]["start_seconds"] for shot in first_beat]
    assert starts == sorted(starts) and all(start < 100.0 for start in starts)
    assert all(shot[2]["montage"] for shot in first_beat[1:])
    second_beat = [shot for shot in shots if shot[2].get("beat") == "1:2"]
    assert second_beat[0][0] == _clip(11) and len(second_beat) == 3
    assert all(110.0 <= shot[0]["start_seconds"] < 200.0 for shot in second_beat[1:])
    assert sum(shot[2]["weight"] for shot in shots) == pytest.approx(4.0)
    all_starts = [shot[0]["start_seconds"] for shot in shots]
    assert len(all_starts) == len(set(all_starts))


def test_pace_shots_keeps_short_shots_and_falls_back_when_the_film_jumps_back() -> None:
    sections = [{"title": "A", "narration": " ".join(["w"] * 60)}, {"title": "B", "narration": "w w w w"}]
    assignments = [[(_clip(30), "late", {"weight": 1.0})], [(_clip(5), "flashback", {"weight": 1.0})]]

    unchanged = shot_planner.pace_shots(sections, assignments, _scenes(), words_per_minute=120, shot_seconds=40.0)
    assert unchanged == assignments

    paced = shot_planner.pace_shots(sections, assignments, _scenes(), words_per_minute=120, shot_seconds=3.0)
    assert all(shot[0]["start_seconds"] >= 300.0 for shot in paced[0][1:]) and len(paced[0]) > 1
    assert paced[1] == assignments[1]  # 2 s slot stays a single shot


def test_beat_timed_durations_start_each_beat_on_its_first_spoken_word() -> None:
    items = [
        {"beat": "1:1", "narration": "a b c d", "weight": 2.0},
        {"beat": "1:1", "weight": 2.0, "montage": True},
        {"narration": "e f", "weight": 2.0},
    ]
    boundaries = [{"offset": int(moment * 10_000_000), "duration": 5_000_000, "text": word}
                  for moment, word in [(10, "a"), (11, "b"), (12, "c"), (13, "d"), (15, "e"), (16, "f")]]
    spoken = _spoken_word_starts(boundaries)

    assert spoken == [10.0, 11.0, 12.0, 13.0, 15.0, 16.0]
    assert _beat_timed_durations(items, 10.0, 20.0, spoken) == pytest.approx([2.5, 2.5, 5.0])
    items[2].pop("narration")
    assert _beat_timed_durations(items, 10.0, 20.0, spoken) is None
