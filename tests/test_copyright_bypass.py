"""Unit tests for the copyright bypass transformation module."""
from __future__ import annotations

import pytest

from movie_review_factory import copyright_bypass


def test_resolve_profile_defaults_to_off() -> None:
    assert copyright_bypass.resolve_profile(None).name == "off"
    assert copyright_bypass.resolve_profile("").name == "off"
    assert copyright_bypass.resolve_profile("off").is_active is False


def test_resolve_profile_standard_names() -> None:
    balanced = copyright_bypass.resolve_profile("balanced")
    assert balanced.hflip is True
    assert balanced.zoom_ratio == 0.05
    assert balanced.color_grade is True
    assert balanced.is_active is True

    light = copyright_bypass.resolve_profile("light")
    assert light.hflip is False
    assert light.zoom_ratio == 0.03
    assert light.is_active is True

    agg = copyright_bypass.resolve_profile("aggressive")
    assert agg.zoom_ratio == 0.08
    assert agg.hflip is True


def test_resolve_profile_custom_dict() -> None:
    prof = copyright_bypass.resolve_profile({"hflip": True, "zoom_ratio": 0.07})
    assert prof.hflip is True
    assert prof.zoom_ratio == 0.07
    assert prof.is_active is True


def test_hflip_filter() -> None:
    assert copyright_bypass.hflip_filter() == "hflip"


def test_zoom_crop_filter() -> None:
    filter_str = copyright_bypass.zoom_crop_filter(1920, 1080, 0.05)
    assert "crop=" in filter_str
    assert "scale=1920:1080" in filter_str


def test_color_grade_filter() -> None:
    eq = copyright_bypass.color_grade_filter(contrast=1.05, brightness=0.02, saturation=1.08)
    assert eq == "eq=contrast=1.050:brightness=0.020:saturation=1.080"


def test_build_clip_bypass_filters() -> None:
    off_filters = copyright_bypass.build_clip_bypass_filters(1920, 1080, "off")
    assert off_filters == []

    balanced = copyright_bypass.build_clip_bypass_filters(1920, 1080, "balanced")
    assert len(balanced) == 3
    assert balanced[0] == "hflip"
    assert "crop=" in balanced[1]
    assert "eq=" in balanced[2]
