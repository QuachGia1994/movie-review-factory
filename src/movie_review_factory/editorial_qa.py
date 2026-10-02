"""Deterministic editorial checks from the artifacts that produced a review video.

These checks identify concrete timing and provenance errors. Reused footage is
reported for human review because deliberate callbacks cannot be judged here.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _check(name: str, value: object, passed: bool, message: str = "") -> dict:
    return {"check": name, "value": value, "passed": passed, "message": message}


def caption_checks(alignment: dict) -> list[dict]:
    """Check the actual aligned cue timing and on-screen text geometry."""
    cues = alignment.get("cues") or []
    if not isinstance(cues, list):
        cues = []
    narration = _number(alignment.get("narration_seconds"))
    invalid: list[object] = []
    overflow: list[object] = []
    unreadable: list[object] = []
    previous_end = 0.0
    for position, cue in enumerate(cues, 1):
        if not isinstance(cue, dict):
            invalid.append(position)
            continue
        index = cue.get("index", position)
        start, end = _number(cue.get("start_seconds")), _number(cue.get("end_seconds"))
        lines = str(cue.get("text") or "").splitlines()
        if not lines or any(not line.strip() for line in lines) or len(lines) > 2 or any(len(line) > 42 for line in lines):
            overflow.append(index)
        if start is not None and end is not None and end > start and (end - start > 6.1 or end - start < 0.25):
            unreadable.append(index)
        if (start is None or end is None or start < 0 or end <= start
                or start < previous_end - 0.02
                or narration is not None and end > narration + 0.02):
            invalid.append(index)
            continue
        previous_end = end
    return [
        _check("caption_timing", {"cue_count": len(cues), "invalid": invalid},
               not invalid, f"Subtitle timing invalid or overlapping: {invalid}" if invalid else ""),
        _check("caption_two_line_safe", {"overflow": overflow, "max_lines": 2, "max_characters_per_line": 42},
               not overflow, f"Subtitle exceeds two short lines: {overflow}" if overflow else ""),
        {"check": "caption_readable_duration", "value": {"outside_0.25_to_6.1_seconds": unreadable},
         "passed": True, "review_required": bool(unreadable),
         "message": f"Subtitle duration requires review: {unreadable}" if unreadable else ""},
    ]


def clip_checks(plan: dict, render: dict) -> list[dict]:
    """Check repeated footage and that the renderer used the selected clips."""
    clips = plan.get("clips") or []
    used = render.get("clips") or []
    if not isinstance(clips, list):
        clips = []
    if not isinstance(used, list):
        used = []
    mismatch: list[int] = []
    repeats: list[dict] = []
    loops: list[dict] = []
    history: list[tuple[int, float, float]] = []
    for position, clip in enumerate(clips):
        if not isinstance(clip, dict):
            mismatch.append(position + 1)
            continue
        source = clip.get("source_clip")
        if not isinstance(source, dict):
            mismatch.append(position + 1)
            continue
        start, end = _number(source.get("start_seconds")), _number(source.get("end_seconds"))
        if start is None or end is None or end <= start:
            mismatch.append(position + 1)
            continue
        if position < len(used) and isinstance(used[position], dict):
            item = used[position]
            actual_start = _number(item.get("start_seconds"))
            actual_end = _number(item.get("end_seconds"))
            if actual_start is None or actual_end is None or abs(start - actual_start) > .01 or abs(end - actual_end) > .01:
                mismatch.append(position + 1)
        elif render:
            mismatch.append(position + 1)
        target = _number(clip.get("duration_seconds"))
        window = end - start
        rendered = used[position] if position < len(used) and isinstance(used[position], dict) else {}
        read = _number(rendered.get("read_seconds"))
        if read is not None:
            window = read
            target = _number(rendered.get("duration_seconds"))
        if target is not None and target > window * 1.5 and target - window > 3:
            loops.append({"clip": position + 1, "source_seconds": round(window, 2),
                          "timeline_seconds": round(target, 2)})
        if end - start < 3:
            history.append((position + 1, start, end))
            continue
        for former, prev_start, prev_end in history:
            overlap = max(0.0, min(end, prev_end) - max(start, prev_start))
            if overlap >= .8 * min(end - start, prev_end - prev_start):
                repeats.append({"first_clip": former, "repeated_clip": position + 1,
                                "source_overlap_seconds": round(overlap, 2)})
                break
        history.append((position + 1, start, end))
    if len(used) != len(clips) and render:
        mismatch.append(0)
    return [
        _check("render_clip_provenance", {"plan_count": len(clips), "render_count": len(used),
                                         "mismatch": sorted(set(mismatch))}, not mismatch,
               f"Rendered clip sources diverge from the plan: {sorted(set(mismatch))}" if mismatch else ""),
        {"check": "source_footage_reuse", "value": {"repeated_pairs": repeats},
         "passed": True, "review_required": bool(repeats),
         "message": f"{len(repeats)} reused source ranges need editorial review" if repeats else ""},
        {"check": "stretched_footage", "value": {"looped_clips": loops},
         "passed": True, "review_required": bool(loops),
         "message": f"{len(loops)} clips loop source frames to fill the timeline" if loops else ""},
    ]


def provenance_checks(render: dict, source_video: str) -> list[dict]:
    rendered = str(render.get("source_video") or "")
    # abspath: a CLI run records job-relative paths, the web app absolute ones.
    canonical = lambda value: os.path.abspath(value.strip()).replace("/", "\\").casefold()
    ok = bool(rendered and source_video and canonical(rendered) == canonical(source_video))
    return [_check("render_source_provenance", {"expected": source_video, "rendered": rendered}, ok,
                   "" if ok else "The rendered source differs from the job source")]


def section_sync_checks(alignment: dict, render: dict) -> list[dict]:
    """Compare rendered chapter cuts against the measured spoken transitions."""
    bounds = alignment.get("section_bounds") or []
    if not bounds:
        if alignment.get("timing_mode") == "tts_word_boundary":
            return [_check("section_voice_sync", {"drift": [], "bounds": 0}, False,
                           "TTS word timing could not be matched to script sections")]
        return []  # Legacy projects have no measured section boundaries.
    expected = {item["section_index"]: float(item["start_seconds"]) for item in bounds}
    actual: dict[int, float] = {}
    cursor = 0.0
    for clip in render.get("clips") or []:
        index = clip.get("section_index")
        if index not in actual:
            actual[index] = cursor
        cursor += float(clip.get("duration_seconds") or 0)
    drift = [
        {"section_index": index, "voice_seconds": start,
         "video_seconds": actual.get(index),
         "drift_seconds": round(abs(start - actual[index]), 3) if index in actual else None}
        for index, start in expected.items()
        if index not in actual or abs(start - actual[index]) > 0.5
    ]
    return [_check("section_voice_sync", {"drift": drift, "tolerance_seconds": 0.5},
                   not drift, f"Visual chapter cuts differ from voice at {drift}" if drift else "")]


def midroll_checks(render: dict) -> list[dict]:
    """Require a staged midroll to remain within 10% of the measured midpoint."""
    clips = render.get("clips") or []
    midpoint = None
    cursor = 0.0
    for clip in clips:
        if clip.get("type") == "midroll":
            midpoint = cursor + float(clip.get("duration_seconds") or 0) / 2
        cursor += float(clip.get("duration_seconds") or 0)
    if midpoint is None:
        return []
    fraction = midpoint / cursor if cursor > 0 else 0.0
    ok = cursor > 0 and abs(fraction - 0.5) <= 0.1
    return [_check("midroll_at_midpoint", {"fraction": round(fraction, 3)}, ok,
                   "" if ok else "The CTA moved outside the central 40–60% of actual narration")]


def _ass_number(value: object) -> float | None:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_ass_geometry(ass_text: str) -> tuple[float | None, float | None, list[str] | None]:
    """Return (PlayResX, PlayResY, first Default style fields) from an ASS script."""
    play_x = play_y = None
    style: list[str] | None = None
    for raw in ass_text.splitlines():
        line = raw.strip()
        if line.startswith("PlayResX:"):
            play_x = _ass_number(line.split(":", 1)[1])
        elif line.startswith("PlayResY:"):
            play_y = _ass_number(line.split(":", 1)[1])
        elif style is None and line.startswith("Style:"):
            style = [part.strip() for part in line[len("Style:"):].split(",")]
    return play_x, play_y, style


def caption_geometry_checks(root: Path, render: dict) -> list[dict]:
    """Guard that burned captions used a frame-sized ASS in the bottom safe area.

    Regression guard for the libass 384x288-default bug: pixel margins and font
    are only accurate when the caption ASS PlayRes equals the real frame. When
    PlayRes is wrong the block floats up as a narrow, word-per-line column. Skips
    cleanly when no caption ASS was produced (an empty subtitle track).

    ASS style fields (0-indexed after ``Style:``): 18=Alignment, 19=MarginL,
    20=MarginR, 21=MarginV.
    """
    ass_path = root / "aligned.ass"
    width = _number(render.get("width"))
    height = _number(render.get("height"))
    if not ass_path.is_file() or not width or not height:
        return []
    try:
        play_x, play_y, style = _parse_ass_geometry(ass_path.read_text(encoding="utf-8"))
    except OSError:
        return [_check("caption_ass_playres", {"parsed": False}, False,
                       "Caption ASS could not be read")]
    if style is None or len(style) < 22:
        return [_check("caption_ass_playres", {"parsed": False}, False,
                       "Caption ASS has no parseable style line")]
    alignment_field = style[18]
    margin_l = _ass_number(style[19])
    margin_r = _ass_number(style[20])
    margin_v = _ass_number(style[21])
    playres_ok = play_x == width and play_y == height
    bottom_gap = (margin_v / height) if margin_v is not None else None
    text_width = ((width - (margin_l or 0) - (margin_r or 0)) / width)
    anchored = alignment_field == "2" and bottom_gap is not None and 0.05 <= bottom_gap <= 0.30
    width_ok = margin_l is not None and margin_r is not None and 0 < text_width <= 0.90
    return [
        _check("caption_ass_playres",
               {"play_res": [play_x, play_y], "frame": [width, height]},
               bool(playres_ok),
               "" if playres_ok else "Caption ASS PlayRes does not match the frame; pixel margins mis-scale"),
        _check("caption_bottom_safe_area",
               {"alignment": alignment_field,
                "bottom_gap_fraction": round(bottom_gap, 3) if bottom_gap is not None else None,
                "text_width_fraction": round(text_width, 3)},
               bool(anchored and width_ok),
               "" if (anchored and width_ok) else "Burned caption is not bottom-anchored within the 90% safe width"),
    ]


def source_text_review_checks(top_band: float = 0, bottom_band: float = 0) -> list[dict]:
    """Flag source text covered by configured bands for explicit rights review."""
    bands = {"top_fraction": float(top_band), "bottom_fraction": float(bottom_band)}
    configured = any(value > 0 for value in bands.values())
    return [{
        "check": "source_watermark_text_review",
        "value": bands,
        "passed": True,
        "review_required": configured,
        "message": "Configured source-text bands need manual watermark and rights review" if configured else "",
    }]


def editorial_checks(
    alignment: dict,
    plan: dict,
    render: dict,
    source_video: str,
    *,
    brand_top_band: float = 0,
    brand_bottom_band: float = 0,
) -> list[dict]:
    """Return QA-compatible check records without modifying any job artifact."""
    return (caption_checks(alignment) + clip_checks(plan, render)
            + section_sync_checks(alignment, render) + midroll_checks(render)
            + provenance_checks(render, source_video)
            + source_text_review_checks(brand_top_band, brand_bottom_band))


def inspect_job(
    root: Path,
    source_video: str,
    *,
    brand_top_band: float = 0,
    brand_bottom_band: float = 0,
) -> list[dict]:
    """Read current job artifacts; suitable for use inside the QA stage."""
    def read(name: str) -> dict:
        path = root / name
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    render = read("render.json")
    return (editorial_checks(
                read("alignment.json"), read("scene_plan.json"), render, source_video,
                brand_top_band=brand_top_band, brand_bottom_band=brand_bottom_band,
            )
            + caption_geometry_checks(root, render))
