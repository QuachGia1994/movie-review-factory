"""Subtitle cues from boundary metadata emitted with the narration audio."""

from __future__ import annotations

import asyncio
import math
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

TICKS_PER_SECOND = 10_000_000
MAX_LINE_CHARS = 42
MAX_CUE_SECONDS = 6.0
MAX_GAP_SECONDS = 0.8


def synthesize_with_boundaries(
    narration: str,
    voice: str,
    output: Path,
    communicate_factory: Callable | None = None,
) -> list[dict]:
    """Write one MP3 and collect timestamps from that same synthesis stream."""
    if communicate_factory is None:
        from edge_tts import Communicate

        communicate_factory = Communicate
    boundaries: list[dict] = []
    communicator = communicate_factory(narration, voice, boundary="WordBoundary")
    async def collect() -> None:
        with output.open("wb") as audio:
            async for chunk in communicator.stream():
                kind = chunk.get("type")
                if kind == "audio":
                    audio.write(chunk["data"])
                elif kind == "WordBoundary":
                    boundaries.append({
                        "offset": int(chunk["offset"]),
                        "duration": int(chunk["duration"]),
                        "text": str(chunk["text"]),
                    })

    asyncio.run(collect())
    if not output.stat().st_size:
        raise RuntimeError("TTS returned no audio")
    if not boundaries:
        raise RuntimeError("TTS returned audio without word timestamps")
    return boundaries


def section_bounds_from_boundaries(
    boundaries: Iterable[dict], sections: list[dict], duration: float
) -> list[dict]:
    """Match TTS words to script sections; reject ambiguous chapter transitions."""
    def tokens(value: str) -> list[str]:
        return re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)

    source = [tokens(str(section.get("narration", ""))) for section in sections]
    spoken: list[tuple[str, float, float]] = []
    for item in boundaries:
        parts = tokens(str(item.get("text", "")))
        start = float(item["offset"]) / TICKS_PER_SECOND
        end = (float(item["offset"]) + float(item["duration"])) / TICKS_PER_SECOND
        if not parts or not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        for index, word in enumerate(parts):
            spoken.append((word, start + (end - start) * index / len(parts),
                           start + (end - start) * (index + 1) / len(parts)))
    if not source or any(not section for section in source) or not spoken:
        return []
    original = [word for section in source for word in section]
    matcher = SequenceMatcher(None, original, [word for word, _, _ in spoken], autojunk=False)
    matches = {
        source_index: spoken_index
        for source_start, spoken_start, size in matcher.get_matching_blocks()
        for source_index, spoken_index in (
            (source_start + offset, spoken_start + offset) for offset in range(size)
        )
    }
    if len(matches) < max(1, math.ceil(len(original) * 0.65)):
        return []
    edges = [0.0]
    cursor = 0
    for section in source[:-1]:
        cursor += len(section)
        left = next((matches[index] for index in range(cursor - 1, max(-1, cursor - 5), -1)
                     if index in matches), None)
        right = next((matches[index] for index in range(cursor, min(len(original), cursor + 4))
                      if index in matches), None)
        if left is None or right is None or left >= right:
            return []
        boundary = (spoken[left][2] + spoken[right][1]) / 2
        if not math.isfinite(boundary) or boundary <= edges[-1] or boundary >= duration:
            return []
        edges.append(boundary)
    edges.append(duration)
    return [
        {"section_index": index + 1, "start_seconds": edges[index],
         "end_seconds": edges[index + 1]}
        for index in range(len(source))
    ]


def cues_from_boundaries(boundaries: Iterable[dict], duration: float) -> list[dict]:
    """Group spoken words into cues of at most two 42-character lines."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("narration duration must be positive")
    tokens: list[tuple[str, float, float]] = []
    for item in boundaries:
        offset = item.get("offset")
        length = item.get("duration")
        text = str(item.get("text", "")).strip()
        if not isinstance(offset, (int, float)) or not isinstance(length, (int, float)):
            raise ValueError("word boundary has nonnumeric timestamps")
        start = float(offset) / TICKS_PER_SECOND
        end = (float(offset) + float(length)) / TICKS_PER_SECOND
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start:
            raise ValueError("word boundary has invalid timestamps")
        if not text:
            continue
        words = text.split()
        for index, word in enumerate(words):
            word_start = start + (end - start) * index / len(words)
            word_end = start + (end - start) * (index + 1) / len(words)
            tokens.append((word, word_start, word_end))
    if not tokens:
        raise ValueError("no spoken words in TTS boundaries")

    cues: list[dict] = []
    current: list[str] = []
    lines: list[str] = []
    cue_start = 0.0
    cue_end = 0.0

    def flush() -> None:
        if current:
            cues.append({
                "index": len(cues) + 1,
                "start_seconds": cue_start,
                "end_seconds": cue_end,
                "text": "\n".join(lines),
            })

    for word, start, end in tokens:
        if start >= duration:
            break
        end = min(end, duration)
        if end <= start:
            continue
        if len(word) > MAX_LINE_CHARS:
            raise ValueError("word boundary exceeds caption line width")
        if current and start < cue_end - 0.05:
            raise ValueError("TTS word boundaries overlap")
        candidate = list(lines)
        if not candidate:
            candidate = [word]
        elif len(candidate[-1]) + 1 + len(word) <= MAX_LINE_CHARS:
            candidate[-1] += " " + word
        else:
            candidate.append(word)
        if current and (
            len(candidate) > 2
            or end - cue_start > MAX_CUE_SECONDS
            or start - cue_end > MAX_GAP_SECONDS
        ):
            flush()
            current, lines, cue_start = [], [word], start
        else:
            lines = candidate
            if not current:
                cue_start = start
        current.append(word)
        cue_end = end
    flush()
    return cues
