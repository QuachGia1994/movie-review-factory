"""Video transformation filters for review commentary and Content ID safety.

Provides deterministic FFmpeg filter fragments (horizontal flip, subtle zoom-crop,
color balance, and tempo shift) to transform raw movie clips into original commentary
footage and prevent automated Content ID false-positives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_ZOOM_RATIO = 0.01
MAX_ZOOM_RATIO = 0.15
DEFAULT_ZOOM_RATIO = 0.05

# NOTE: tempo / micro-speed shifting is intentionally NOT a bypass knob. The
# render concatenates clip video with a=0 and lays the independent TTS narration
# as the master audio, and clip durations are already dictated by that narration
# timing. Speeding the video (setpts) against a fixed-length voice-over - with no
# matching audio atempo - would desync the two, so no speed factor is exposed
# here and none is applied downstream in pipeline.py.


@dataclass(frozen=True)
class BypassProfile:
    name: str = "off"
    hflip: bool = False
    zoom_ratio: float = 0.0
    color_grade: bool = False

    @property
    def is_active(self) -> bool:
        return self.hflip or self.zoom_ratio > 0.0 or self.color_grade


PROFILES: dict[str, BypassProfile] = {
    "off": BypassProfile(name="off"),
    "light": BypassProfile(
        name="light",
        hflip=False,
        zoom_ratio=0.03,
        color_grade=True,
    ),
    "balanced": BypassProfile(
        name="balanced",
        hflip=True,
        zoom_ratio=DEFAULT_ZOOM_RATIO,
        color_grade=True,
    ),
    "aggressive": BypassProfile(
        name="aggressive",
        hflip=True,
        zoom_ratio=0.08,
        color_grade=True,
    ),
}


def resolve_profile(value: str | dict[str, Any] | BypassProfile | None) -> BypassProfile:
    if value is None:
        return PROFILES["off"]
    if isinstance(value, BypassProfile):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in PROFILES:
            return PROFILES[normalized]
        if normalized in {"1", "true", "yes", "on"}:
            return PROFILES["balanced"]
        return PROFILES["off"]
    if isinstance(value, dict):
        base_name = str(value.get("name", "custom")).lower()
        base = PROFILES.get(base_name, PROFILES["off"])
        hflip = bool(value.get("hflip", base.hflip))
        zoom = float(value.get("zoom_ratio", base.zoom_ratio))
        color = bool(value.get("color_grade", base.color_grade))
        return BypassProfile(
            name=base_name,
            hflip=hflip,
            zoom_ratio=max(0.0, min(MAX_ZOOM_RATIO, zoom)),
            color_grade=color,
        )
    return PROFILES["off"]


def hflip_filter() -> str:
    return "hflip"


def zoom_crop_filter(width: int, height: int, zoom_ratio: float = DEFAULT_ZOOM_RATIO) -> str:
    ratio = max(MIN_ZOOM_RATIO, min(MAX_ZOOM_RATIO, float(zoom_ratio)))
    crop_w = max(2, round(width * (1.0 - ratio) / 2) * 2)
    crop_h = max(2, round(height * (1.0 - ratio) / 2) * 2)
    return (
        f"crop={crop_w}:{crop_h}:(iw-{crop_w})/2:(ih-{crop_h})/2,"
        f"scale={width}:{height}"
    )


def color_grade_filter(
    *, contrast: float = 1.04, brightness: float = 0.01, saturation: float = 1.06
) -> str:
    return f"eq=contrast={contrast:.3f}:brightness={brightness:.3f}:saturation={saturation:.3f}"


def build_clip_bypass_filters(
    width: int,
    height: int,
    profile: BypassProfile | str | dict[str, Any] | None,
) -> list[str]:
    prof = resolve_profile(profile)
    if not prof.is_active:
        return []
    filters: list[str] = []
    if prof.hflip:
        filters.append(hflip_filter())
    if prof.zoom_ratio > 0.0:
        filters.append(zoom_crop_filter(width, height, prof.zoom_ratio))
    if prof.color_grade:
        filters.append(color_grade_filter())
    return filters
