"""Unit tests for the Hook Crafter teaser builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from movie_review_factory import hook_crafter
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job


def test_dramatic_score_rewards_action_over_calm() -> None:
    calm = {"text": "Họ ngồi uống trà và trò chuyện", "visual_actions": ["sitting", "talking"]}
    intense = {"text": "Hắn rút súng bắn rồi bỏ chạy", "visual_actions": ["fight", "explosion"]}
    assert hook_crafter.dramatic_score(intense) > hook_crafter.dramatic_score(calm)
    assert hook_crafter.dramatic_score(calm) == 0.0


def test_select_dramatic_scene_picks_highest_scoring() -> None:
    scenes = [
        {"index": 1, "start_seconds": 0, "end_seconds": 10, "text": "quiet conversation"},
        {"index": 2, "start_seconds": 10, "end_seconds": 20, "text": "gun fight explosion",
         "visual_actions": ["shoot", "run"]},
        {"index": 3, "start_seconds": 20, "end_seconds": 25, "text": "another chase attack"},
    ]
    assert hook_crafter.select_dramatic_scene(scenes)["index"] == 2


def test_select_dramatic_scene_falls_back_to_longest_when_no_drama() -> None:
    scenes = [
        {"index": 1, "start_seconds": 0, "end_seconds": 5, "text": "tea time"},
        {"index": 2, "start_seconds": 5, "end_seconds": 20, "text": "a walk in the park"},
    ]
    assert hook_crafter.select_dramatic_scene(scenes)["index"] == 2


def test_select_dramatic_scene_returns_none_when_empty() -> None:
    assert hook_crafter.select_dramatic_scene([]) is None
    assert hook_crafter.select_dramatic_scene([{"index": 1, "start_seconds": 3, "end_seconds": 3}]) is None


def test_hook_teaser_range_clamps_long_scene_and_keeps_short_scene() -> None:
    long_scene = hook_crafter.hook_teaser_range({"start_seconds": 100.0, "end_seconds": 140.0})
    assert long_scene["duration_seconds"] == pytest.approx(4.0)
    assert 100.0 <= long_scene["start_seconds"] and long_scene["end_seconds"] <= 140.0

    short_scene = hook_crafter.hook_teaser_range({"start_seconds": 10.0, "end_seconds": 12.0})
    assert short_scene["duration_seconds"] == pytest.approx(2.0)
    assert (short_scene["start_seconds"], short_scene["end_seconds"]) == (10.0, 12.0)


def test_build_teaser_command_has_seek_zoom_and_output(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    dest = tmp_path / "hook.mp4"
    plan = {"start_seconds": 12.0, "end_seconds": 16.0, "duration_seconds": 4.0}
    cmd = hook_crafter.build_teaser_command("ffmpeg", src, plan, dest, width=1920, height=1080)
    assert cmd[0] == "ffmpeg"
    assert "-ss" in cmd and "12.000000" in cmd
    assert "zoompan" in " ".join(cmd)
    assert "libx264" in cmd
    assert cmd[-1] == str(dest)
    assert "amix" not in " ".join(cmd)  # no SFX -> no mix


def test_build_teaser_command_mixes_sfx_when_provided(tmp_path: Path) -> None:
    src = tmp_path / "src.mp4"
    dest = tmp_path / "hook.mp4"
    sfx = tmp_path / "whoosh.wav"
    plan = {"start_seconds": 0.0, "end_seconds": 3.0, "duration_seconds": 3.0}
    cmd = hook_crafter.build_teaser_command(
        "ffmpeg", src, plan, dest, width=1080, height=1920, sfx_path=sfx,
    )
    assert "amix=inputs=2:duration=first:dropout_transition=0" in " ".join(cmd)
    assert str(sfx) in cmd


def test_build_hook_teaser_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "src.mp4"
    source.write_bytes(b"x")
    create_job(tmp_path, JobConfig(job_id="hook-job", source_video=source))
    (tmp_path / "scenes.json").write_text(
        json.dumps({"scenes": [
            {"index": 1, "start_seconds": 0, "end_seconds": 30,
             "text": "gun fight explosion chase", "visual_actions": ["shoot", "run"]},
        ]}),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        (tmp_path / "hook.mp4").write_bytes(b"fake-teaser")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(hook_crafter.shutil, "which", lambda name: f"/{name}")
    monkeypatch.setattr(hook_crafter.subprocess, "run", fake_run)
    monkeypatch.delenv("MRF_HOOK_SFX", raising=False)

    dest = hook_crafter.build_hook_teaser(tmp_path, ratio="16:9")

    assert dest == tmp_path / "hook.mp4"
    assert (tmp_path / "hook.mp4").read_bytes() == b"fake-teaser"
    doc = json.loads((tmp_path / "hook.json").read_text(encoding="utf-8"))
    assert doc["scene_index"] == 1
    assert (doc["width"], doc["height"]) == (1920, 1080)
    assert doc["duration_seconds"] > 0
    assert doc["dramatic_score"] > 0
    assert calls and calls[0][0] == "/ffmpeg"


def test_build_hook_teaser_requires_scenes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="scenes.json"):
        hook_crafter.build_hook_teaser(tmp_path)
