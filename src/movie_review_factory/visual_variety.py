"""Per-clip visual variety filters for review commentary footage.

Provides deterministic FFmpeg filter fragments (horizontal flip, subtle zoom-crop,
colour grade) applied to each clip in the commentary render.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MIN_ZOOM_RATIO = 0.01
MAX_ZOOM_RATIO = 0.15
DEFAULT_ZOOM_RATIO = 0.05

# No speed knob: clip video is silent under fixed-length TTS narration, so a video-only tempo shift would desync them.


@dataclass(frozen=True)
class VarietyProfile:
    name: str = "off"
    hflip: bool = False
    zoom_ratio: float = 0.0
    color_grade: bool = False

    @property
    def is_active(self) -> bool:
        return self.hflip or self.zoom_ratio > 0.0 or self.color_grade


PROFILES: dict[str, VarietyProfile] = {
    "off": VarietyProfile(name="off"),
    "light": VarietyProfile(
        name="light",
        hflip=False,
        zoom_ratio=0.03,
        color_grade=True,
    ),
    "balanced": VarietyProfile(
        name="balanced",
        hflip=True,
        zoom_ratio=DEFAULT_ZOOM_RATIO,
        color_grade=True,
    ),
    "aggressive": VarietyProfile(
        name="aggressive",
        hflip=True,
        zoom_ratio=0.08,
        color_grade=True,
    ),
}


def resolve_profile(value: str | dict[str, Any] | VarietyProfile | None) -> VarietyProfile:
    if value is None:
        return PROFILES["off"]
    if isinstance(value, VarietyProfile):
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
        return VarietyProfile(
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


def build_clip_variety_filters(
    width: int,
    height: int,
    profile: VarietyProfile | str | dict[str, Any] | None,
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
