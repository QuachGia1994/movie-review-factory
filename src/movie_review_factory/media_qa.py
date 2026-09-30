"""Decoded signal checks for a rendered review; findings include output timeline seconds."""
from __future__ import annotations

import math
import re
import shutil
import subprocess
from pathlib import Path

_NUMBER = r"(-?(?:\d+(?:\.\d*)?|\.\d+|inf))"
_BLACK = re.compile(r"black_start:\s*" + _NUMBER + r"\s+black_end:\s*" + _NUMBER + r"\s+black_duration:\s*" + _NUMBER)
_FREEZE_START = re.compile(r"freeze_start:\s*" + _NUMBER)
_FREEZE_END = re.compile(r"freeze_end:\s*" + _NUMBER + r"\s*\|\s*freeze_duration:\s*" + _NUMBER)
_SILENCE_START = re.compile(r"silence_start:\s*" + _NUMBER)
_SILENCE_END = re.compile(r"silence_end:\s*" + _NUMBER + r"\s*\|\s*silence_duration:\s*" + _NUMBER)
_LOUDNESS = re.compile(r"Integrated loudness:\s*I:\s*" + _NUMBER + r"\s*LUFS")
_TRUE_PEAK = re.compile(r"True peak:\s*Peak:\s*" + _NUMBER + r"\s*dBFS")


def _intervals(log: str, start: re.Pattern, end: re.Pattern, duration: float | None) -> list[dict]:
    events: list[tuple[int, str, re.Match]] = [
        (match.start(), "start", match) for match in start.finditer(log)
    ] + [(match.start(), "end", match) for match in end.finditer(log)]
    events.sort(key=lambda event: event[0])
    intervals: list[dict] = []
    opened: float | None = None
    for _, kind, match in events:
        value = float(match.group(1))
        if kind == "start":
            opened = value
        elif opened is not None and value >= opened:
            intervals.append({"start_seconds": round(opened, 3),
                              "end_seconds": round(value, 3),
                              "duration_seconds": round(value - opened, 3)})
            opened = None
    if opened is not None and duration is not None and duration >= opened:
        intervals.append({"start_seconds": round(opened, 3),
                          "end_seconds": round(duration, 3),
                          "duration_seconds": round(duration - opened, 3)})
    return intervals


def parse_signal_log(log: str, duration_seconds: float | None = None) -> list[dict]:
    """Parse one FFmpeg decode pass into QA-compatible checks.

    Content findings are review cues because black, held frames and speech
    pauses can all be intentional. Failed decode and absent measurable speech
    remain blocking checks.
    """
    black = [
        {"start_seconds": round(float(match.group(1)), 3),
         "end_seconds": round(float(match.group(2)), 3),
         "duration_seconds": round(float(match.group(3)), 3)}
        for match in _BLACK.finditer(log)
    ]
    freeze = _intervals(log, _FREEZE_START, _FREEZE_END, duration_seconds)
    silence = _intervals(log, _SILENCE_START, _SILENCE_END, duration_seconds)
    loudness_matches = list(_LOUDNESS.finditer(log))
    peak_matches = list(_TRUE_PEAK.finditer(log))
    integrated = float(loudness_matches[-1].group(1)) if loudness_matches else None
    peak = float(peak_matches[-1].group(1)) if peak_matches else None
    level_available = integrated is not None and math.isfinite(integrated)
    level_review = bool(level_available and (integrated < -20 or integrated > -10
                                            or peak is not None and peak > -1))
    checks = []
    for name, intervals in (("black_intervals", black), ("freeze_intervals", freeze),
                            ("silence_intervals", silence)):
        checks.append({"check": name, "value": {"intervals": intervals},
                       "passed": True, "review_required": bool(intervals),
                       "message": f"{len(intervals)} output timeline intervals need editorial review" if intervals else ""})
    checks.append({"check": "decoded_audio_loudness",
                   "value": {"integrated_lufs": integrated, "true_peak_dbfs": peak,
                             "review_range_lufs": [-20, -10], "review_peak_dbfs": -1},
                   "passed": level_available,
                   "review_required": level_review,
                   "message": ("No measurable decoded audio loudness" if not level_available else
                               "Loudness or true peak needs an editorial listening check" if level_review else "")})
    return checks


def inspect_rendered_media(
    video_path: Path, *, ffmpeg_bin: str | None = None,
    duration_seconds: float | None = None,
) -> list[dict]:
    """Decode video and audio once with FFmpeg and report timecoded findings."""
    ffmpeg = ffmpeg_bin or shutil.which("ffmpeg")
    if not ffmpeg:
        return [{"check": "decoded_media_scan", "value": None, "passed": False,
                 "message": "FFmpeg is unavailable for decoded media QA"}]
    command = [
        ffmpeg, "-hide_banner", "-nostdin", "-nostats", "-v", "info",
        "-i", str(video_path),
        "-filter_complex",
        "[0:v]blackdetect=d=2.0:pic_th=0.98:pix_th=0.10,"
        "freezedetect=n=-60dB:d=2.5[v];"
        "[0:a]silencedetect=noise=-50dB:d=2.5,"
        "ebur128=peak=true[a]",
        "-map", "[v]", "-map", "[a]", "-f", "null", "-",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=1800)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [{"check": "decoded_media_scan", "value": None, "passed": False,
                 "message": f"Decoded media scan failed: {type(exc).__name__}"}]
    if result.returncode:
        detail = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "unknown FFmpeg error"
        return [{"check": "decoded_media_scan", "value": None, "passed": False,
                 "message": f"Decoded media scan failed: {detail[:200]}"}]
    scan = {"check": "decoded_media_scan", "value": {"video": True, "audio": True},
            "passed": True, "message": ""}
    return [scan, *parse_signal_log(result.stderr, duration_seconds)]


def signals_from_render_log(
    log: str, duration_seconds: float | None = None
) -> list[dict] | None:
    """Build QA signal checks from the detect log captured during the render.

    The render pass runs the same detect filters on the frames FFmpeg already
    decoded for the encode, so QA reuses them instead of decoding final.mp4 a
    second time. Returns ``None`` when the log carries no measurable loudness
    summary - the detect pass did not run or the log is unusable - so the caller
    can fall back to a fresh decode pass.
    """
    if "Integrated loudness" not in log:
        return None
    scan = {"check": "decoded_media_scan",
            "value": {"video": True, "audio": True, "source": "render_pass"},
            "passed": True, "message": ""}
    return [scan, *parse_signal_log(log, duration_seconds)]
