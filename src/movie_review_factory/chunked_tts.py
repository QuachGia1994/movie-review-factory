"""Bounded Edge TTS batches with word timing tied to concatenated MP3 audio."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .narration_alignment import TICKS_PER_SECOND, synthesize_with_boundaries


FFPROBE_TIMEOUT_SECONDS = 45
FFMPEG_CONCAT_TIMEOUT_SECONDS = 240


def _run_media_command(
    command: list[str], *, timeout: float, cleanup: tuple[Path, ...] = (), **kwargs
) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        for path in cleanup:
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"media command timed out after {timeout:g}s: {subprocess.list2cmdline(command)}"
        ) from exc


def split_narration(narration: str, *, max_chars: int = 500) -> list[str]:
    """Split at sentence endings or whitespace, preserving the original characters."""
    if max_chars < 2:
        raise ValueError("max_chars must be at least 2")
    if not narration or not narration.strip():
        raise ValueError("narration must contain words")
    chunks: list[str] = []
    cursor = 0
    while cursor < len(narration):
        limit = min(cursor + max_chars, len(narration))
        if limit == len(narration):
            chunks.append(narration[cursor:])
            break
        candidates = [match.end() for match in re.finditer(r"\s+", narration[cursor:limit])]
        candidates = [cursor + candidate for candidate in candidates if cursor + candidate < len(narration)]
        if not candidates:
            raise ValueError("narration contains a word longer than max_chars")
        near_end = [p for p in candidates if p >= cursor + max_chars // 2]
        sentence = [p for p in near_end if re.search(r"[.!?…][\s]*$", narration[cursor:p])]
        end = sentence[-1] if sentence else candidates[-1]
        chunks.append(narration[cursor:end])
        cursor = end
    if any(not chunk.strip() for chunk in chunks):
        raise ValueError("narration has a whitespace-only chunk")
    return chunks


def _bisect_failed_chunk(chunk: str, *, min_chars: int = 60) -> tuple[str, str] | None:
    """Find a near-midpoint word boundary that leaves two useful TTS requests."""
    middle = len(chunk) / 2
    candidates = (
        match.end() for match in re.finditer(r"\s+", chunk)
    )
    safe = [
        position for position in candidates
        if min_chars <= position <= len(chunk) - min_chars
        and chunk[:position].strip() and chunk[position:].strip()
    ]
    if not safe:
        return None
    position = min(safe, key=lambda value: abs(value - middle))
    return chunk[:position], chunk[position:]


def probe_mp3_duration(path: Path, *, ffprobe_bin: str = "ffprobe") -> float:
    """Read each actual MP3 duration; a synthetic duration would drift across batches."""
    result = _run_media_command(
        [ffprobe_bin, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        timeout=FFPROBE_TIMEOUT_SECONDS, capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(result.stdout)["format"]["duration"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"invalid MP3 duration for {path.name}")
    return duration


def concat_mp3(parts: list[Path], output: Path, *, ffmpeg_bin: str = "ffmpeg") -> None:
    """Concatenate MP3 streams without another encode pass or transformed voice."""
    if not parts:
        raise ValueError("cannot concatenate zero MP3 parts")
    listing = output.with_suffix(".concat.txt")
    # All parts come from our own temporary directory and have simple names.
    listing.write_text(
        "".join(f"file '{part.name}'\n" for part in parts), encoding="utf-8",
    )
    _run_media_command(
        [ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-f", "concat",
         "-safe", "0", "-i", str(listing), "-c:a", "copy", str(output)],
        timeout=FFMPEG_CONCAT_TIMEOUT_SECONDS, cleanup=(output, listing),
        check=True, capture_output=True, text=True,
    )
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("FFmpeg returned no concatenated audio")


def synthesize_chunked(
    narration: str,
    voice: str,
    output: Path,
    *,
    max_chars: int = 500,
    communicate_factory: Callable | None = None,
    synthesize: Callable = synthesize_with_boundaries,
    probe_duration: Callable[[Path], float] | None = None,
    concat_audio: Callable[[list[Path], Path], None] | None = None,
    ffmpeg_bin: str = "ffmpeg",
    ffprobe_bin: str = "ffprobe",
    no_audio_error: type[Exception] | None = None,
    retries: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict]:
    """Create one atomic MP3 with boundaries shifted by probed part durations."""
    if retries < 1:
        raise ValueError("retries must be positive")
    if communicate_factory is None:
        from edge_tts import Communicate

        communicate_factory = Communicate
    if no_audio_error is None:
        from edge_tts.exceptions import NoAudioReceived

        no_audio_error = NoAudioReceived
    probe = probe_duration or (lambda path: probe_mp3_duration(path, ffprobe_bin=ffprobe_bin))
    concat = concat_audio or (
        lambda parts, target: concat_mp3(parts, target, ffmpeg_bin=ffmpeg_bin)
    )
    chunks = split_narration(narration, max_chars=max_chars)
    output.parent.mkdir(parents=True, exist_ok=True)
    shifted: list[dict] = []
    running_ticks = 0
    with tempfile.TemporaryDirectory(prefix=f".{output.stem}.synthesizing-", dir=output.parent) as temp:
        temp_root = Path(temp)
        parts: list[Path] = []
        index = 0
        while index < len(chunks):
            chunk = chunks[index]
            part = temp_root / f"part-{index:04d}.mp3"
            for attempt in range(retries):
                try:
                    boundaries = synthesize(chunk, voice, part, communicate_factory)
                    break
                except no_audio_error:
                    part.unlink(missing_ok=True)
                    if attempt + 1 < retries:
                        sleep(attempt + 1)
                        continue
                    halves = _bisect_failed_chunk(chunk)
                    if halves is None:
                        digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:12]
                        raise RuntimeError(
                            f"TTS failed at chunk {index}, length {len(chunk)}, sha256 {digest}"
                        ) from None
                    chunks[index:index + 1] = halves
                    break
            if chunks[index] != chunk:
                continue
            if len(chunks) == 1:
                if not boundaries or not part.is_file() or part.stat().st_size == 0:
                    raise RuntimeError("TTS returned no audio or word timestamps")
                part.replace(output)
                return boundaries
            duration = probe(part)
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"invalid duration for chunk {index}")
            if not boundaries:
                raise RuntimeError(f"no word timestamps for chunk {index}")
            for boundary in boundaries:
                shifted.append({
                    **boundary,
                    "offset": int(boundary["offset"]) + running_ticks,
                })
            running_ticks += round(duration * TICKS_PER_SECOND)
            parts.append(part)
            index += 1
        combined = temp_root / "combined.mp3"
        concat(parts, combined)
        if not combined.is_file() or combined.stat().st_size == 0:
            raise RuntimeError("concatenated audio is empty")
        combined.replace(output)
    return shifted
