"""Unit tests for the dynamic visual rhythm helpers (Ken Burns + transition SFX)."""
from __future__ import annotations

from movie_review_factory import visual_rhythm


def test_ken_burns_filter_produces_zoompan_and_alternates_pan() -> None:
    even = visual_rhythm.ken_burns_filter(1920, 1080, 25, 100, index=0)
    odd = visual_rhythm.ken_burns_filter(1920, 1080, 25, 100, index=1)
    assert "zoompan=" in even
    assert "s=1920x1080" in even
    assert even != odd  # pan direction alternates by clip index
    # 2x upscale before the crop so the pan never reveals padding.
    assert "scale=3840:2160" in even


def test_ken_burns_filter_clamps_intensity_and_frames() -> None:
    frag = visual_rhythm.ken_burns_filter(1080, 1920, 30, 0, index=0, intensity=5.0)
    assert "d=1:" in frag  # frames floored to >=1
    assert "1.4" in frag   # intensity clamped to the 0.4 ceiling -> max_zoom 1.4


def test_transition_cut_times_only_between_clips() -> None:
    ranges = [
        {"duration_seconds": 2.0},
        {"duration_seconds": 3.0},
        {"duration_seconds": 1.5},
    ]
    # Cuts after clip 1 and clip 2, none at t=0 or after the last clip.
    assert visual_rhythm.transition_cut_times(ranges) == [2.0, 5.0]
    assert visual_rhythm.transition_cut_times(ranges, intro_seconds=4.0) == [6.0, 9.0]
    assert visual_rhythm.transition_cut_times([{"duration_seconds": 5.0}]) == []


def test_transition_sfx_filtergraph_delays_and_mixes() -> None:
    fragment, label = visual_rhythm.transition_sfx_filtergraph("[showa]", 7, [2.0, 5.0])
    assert label == "[audiosfx]"
    assert "[7:a]asplit=2[s0][s1]" in fragment
    assert "adelay=2000|2000" in fragment
    assert "adelay=5000|5000" in fragment
    assert "amix=inputs=3:duration=first:dropout_transition=0:normalize=0[audiosfx]" in fragment


def test_transition_sfx_filtergraph_noop_without_cuts() -> None:
    fragment, label = visual_rhythm.transition_sfx_filtergraph("[showa]", 7, [])
    assert fragment == ""
    assert label == "[showa]"


def test_transition_sfx_filtergraph_normalises_raw_audio_map() -> None:
    fragment, _ = visual_rhythm.transition_sfx_filtergraph("1:a:0", 7, [1.0])
    assert "[1:a][d0]amix=inputs=2" in fragment
