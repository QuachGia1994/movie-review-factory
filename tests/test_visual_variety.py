"""Unit tests for the per-clip visual variety module."""
from __future__ import annotations

import json
from pathlib import Path

from movie_review_factory import visual_variety
from movie_review_factory.creator_library import CreatorLibrary
from movie_review_factory.models import ChannelProfile, JobConfig


def test_resolve_profile_defaults_to_off() -> None:
    assert visual_variety.resolve_profile(None).name == "off"
    assert visual_variety.resolve_profile("").name == "off"
    assert visual_variety.resolve_profile("off").is_active is False


def test_resolve_profile_standard_names() -> None:
    balanced = visual_variety.resolve_profile("balanced")
    assert balanced.hflip is True
    assert balanced.zoom_ratio == 0.05
    assert balanced.color_grade is True
    assert balanced.is_active is True

    light = visual_variety.resolve_profile("light")
    assert light.hflip is False
    assert light.zoom_ratio == 0.03
    assert light.is_active is True

    agg = visual_variety.resolve_profile("aggressive")
    assert agg.zoom_ratio == 0.08
    assert agg.hflip is True


def test_resolve_profile_custom_dict() -> None:
    prof = visual_variety.resolve_profile({"hflip": True, "zoom_ratio": 0.07})
    assert prof.hflip is True
    assert prof.zoom_ratio == 0.07
    assert prof.is_active is True


def test_hflip_filter() -> None:
    assert visual_variety.hflip_filter() == "hflip"


def test_zoom_crop_filter() -> None:
    filter_str = visual_variety.zoom_crop_filter(1920, 1080, 0.05)
    assert "crop=" in filter_str
    assert "scale=1920:1080" in filter_str


def test_color_grade_filter() -> None:
    eq = visual_variety.color_grade_filter(contrast=1.05, brightness=0.02, saturation=1.08)
    assert eq == "eq=contrast=1.050:brightness=0.020:saturation=1.080"


def test_build_clip_variety_filters() -> None:
    off_filters = visual_variety.build_clip_variety_filters(1920, 1080, "off")
    assert off_filters == []

    balanced = visual_variety.build_clip_variety_filters(1920, 1080, "balanced")
    assert len(balanced) == 3
    assert balanced[0] == "hflip"
    assert "crop=" in balanced[1]
    assert "eq=" in balanced[2]


def test_legacy_key_loads_into_job_config_and_channel_profile() -> None:
    cfg = JobConfig.model_validate({"job_id": "old", "copyright_bypass": "light"})
    assert cfg.visual_variety == "light"
    assert "copyright_bypass" not in cfg.model_dump()
    assert JobConfig(job_id="new", visual_variety="balanced").visual_variety == "balanced"
    assert ChannelProfile.model_validate({"name": "Old", "copyright_bypass": "aggressive"}).visual_variety == "aggressive"


def test_legacy_channel_record_exposes_visual_variety(tmp_path: Path) -> None:
    library = CreatorLibrary(tmp_path)
    library.path.write_text(json.dumps({
        "channels": {"old": {"id": "old", "name": "Old", "copyright_bypass": "light"}},
        "active_channel": "old",
    }), encoding="utf-8")
    assert library.get_channel("old")["visual_variety"] == "light"
    assert library.active_channel_defaults() == {"visual_variety": "light"}
