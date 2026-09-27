from pathlib import Path

import pytest

from movie_review_factory.narration_alignment import (
    TICKS_PER_SECOND,
    cues_from_boundaries,
    section_bounds_from_boundaries,
    synthesize_with_boundaries,
)


def boundary(word, start, end):
    return {
        "text": word,
        "offset": round(start * TICKS_PER_SECOND),
        "duration": round((end - start) * TICKS_PER_SECOND),
    }


def test_same_tts_stream_produces_audio_and_word_timestamps(tmp_path: Path):
    class FakeCommunicate:
        def __init__(self, text, voice, *, boundary):
            assert (text, voice, boundary) == ("Xin chào", "vi-VN-HoaiMyNeural", "WordBoundary")

        async def stream(self):
            yield {"type": "audio", "data": b"mp3"}
            yield {"type": "WordBoundary", **boundary("Xin", 0.3, 0.5)}
            yield {"type": "audio", "data": b"-bytes"}
            yield {"type": "WordBoundary", **boundary("chào", 0.6, 0.9)}

    output = tmp_path / "voice.mp3"
    timed = synthesize_with_boundaries(
        "Xin chào", "vi-VN-HoaiMyNeural", output, FakeCommunicate,
    )
    assert output.read_bytes() == b"mp3-bytes"
    assert timed == [boundary("Xin", 0.3, 0.5), boundary("chào", 0.6, 0.9)]
    assert cues_from_boundaries(timed, 1.0) == [{
        "index": 1, "start_seconds": 0.3, "end_seconds": 0.9, "text": "Xin chào",
    }]


def test_cues_follow_speech_pauses_and_never_exceed_two_lines():
    spoken = [boundary("Ben", 0.2, 0.4), boundary("đã", 0.5, 0.7)]
    spoken += [boundary("Omnitrix", 2.0 + i * 0.5, 2.2 + i * 0.5) for i in range(25)]
    cues = cues_from_boundaries(spoken, 15.0)
    assert cues[0]["text"] == "Ben đã"
    assert cues[0]["end_seconds"] == 0.7
    assert cues[1]["start_seconds"] == 2.0
    assert all(len(cue["text"].splitlines()) <= 2 for cue in cues)
    assert all(
        len(line) <= 42
        for cue in cues for line in cue["text"].splitlines()
    )
    assert all(cue["end_seconds"] - cue["start_seconds"] <= 6 for cue in cues)
    assert all(a["end_seconds"] <= b["start_seconds"] for a, b in zip(cues, cues[1:]))


def test_section_bounds_place_midroll_at_measured_voice_transition():
    spoken = [boundary("Mở", 0.1, 0.5), boundary("đầu", 0.6, 1.0),
              boundary("Like", 3.0, 3.4), boundary("nhé", 3.5, 4.0)]
    sections = [{"narration": "Mở đầu"}, {"narration": "Like nhé"}]
    assert section_bounds_from_boundaries(spoken, sections, 5.0) == [
        {"section_index": 1, "start_seconds": 0.0, "end_seconds": 2.0},
        {"section_index": 2, "start_seconds": 2.0, "end_seconds": 5.0},
    ]
    assert section_bounds_from_boundaries(spoken[:2] + [boundary("khác", 3, 3.4)], sections, 5.0) == []
    assert section_bounds_from_boundaries(
        [boundary("Cảnh", 0.1, 0.5), boundary("và", 0.6, 0.9),
         boundary("nhạc", 1.0, 1.3), boundary("Like", 3.0, 3.4),
         boundary("nhé", 3.5, 4.0)],
        [{"narration": "Cảnh & nhạc"}, {"narration": "Like nhé"}], 5.0,
    )[1]["start_seconds"] == 2.15


def test_invalid_timestamps_do_not_silently_shift_subtitles():
    with pytest.raises(ValueError, match="overlap"):
        cues_from_boundaries([boundary("A", 0.1, 1.0), boundary("B", 0.5, 1.2)], 2)
    with pytest.raises(ValueError, match="nonnumeric"):
        cues_from_boundaries([{"text": "A", "offset": "0", "duration": 1}], 2)
    with pytest.raises(ValueError, match="spoken"):
        cues_from_boundaries([], 2)


def test_audio_without_timing_fails_before_installing_artifact(tmp_path: Path):
    class NoTimestamps:
        def __init__(self, *_args, **_kwargs):
            pass

        async def stream(self):
            yield {"type": "audio", "data": b"mp3"}

    with pytest.raises(RuntimeError, match="without word timestamps"):
        synthesize_with_boundaries("test", "voice", tmp_path / "audio.mp3", NoTimestamps)
