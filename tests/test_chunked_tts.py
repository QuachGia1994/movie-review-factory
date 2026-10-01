from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

import movie_review_factory.chunked_tts as chunked_tts
from movie_review_factory.chunked_tts import split_narration, synthesize_chunked


def test_split_preserves_vietnamese_words_and_original_narration():
    narration = ("Ben 10 gặp Eon. Gwen giúp cả nhóm! " * 33) + "\n\n" + ("Omnitrix vẫn hoạt động? " * 14)
    chunks = split_narration(narration, max_chars=120)
    assert len(chunks) > 4
    assert all(0 < len(chunk) <= 120 for chunk in chunks)
    assert "".join(chunks) == narration
    assert [word.casefold() for chunk in chunks for word in re.findall(r"[^\W_]+", chunk, re.UNICODE)] == [
        word.casefold() for word in re.findall(r"[^\W_]+", narration, re.UNICODE)
    ]


def test_network_provider_tts_flows_through_chunked_pipeline(tmp_path: Path):
    """A network provider (HTTP mocked) drops into synthesize_chunked offline."""
    from movie_review_factory import tts_providers

    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        return 200, b"MP3BYTES"

    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="k", http_request=fake_request, probe_duration=lambda _p: 1.0,
    )
    output = tmp_path / "narration.mp3"
    boundaries = synthesize_chunked(
        "Xin chào các bạn yêu điện ảnh", "voice", output, synthesize=synth,
        communicate_factory=None, no_audio_error=RuntimeError,
        probe_duration=lambda _p: 1.0,
        concat_audio=lambda *_: pytest.fail("single chunk must not remux"),
    )
    assert output.read_bytes() == b"MP3BYTES"
    assert boundaries


def test_short_narration_preserves_single_tts_stream_without_remuxing(tmp_path: Path):
    calls = []

    def synthesize(text, voice, path, factory):
        calls.append((text, voice, factory))
        path.write_bytes(b"existing exact stream")
        return [{"text": "Xin", "offset": 100_000, "duration": 200_000}]

    output = tmp_path / "narration.mp3"
    boundaries = synthesize_chunked(
        "Xin chào", "voice", output, synthesize=synthesize,
        communicate_factory="factory", no_audio_error=RuntimeError,
        probe_duration=lambda _: pytest.fail("single MP3 should not be probed"),
        concat_audio=lambda *_: pytest.fail("single MP3 should not be remuxed"),
    )
    assert output.read_bytes() == b"existing exact stream"
    assert boundaries == [{"text": "Xin", "offset": 100_000, "duration": 200_000}]
    assert calls == [("Xin chào", "voice", "factory")]


def test_chunk_offsets_use_measured_audio_duration_and_retry_only_no_audio(tmp_path: Path):
    class NoAudioReceived(Exception):
        pass

    calls = []
    durations = [1.12, 1.35, 0.94, 1.2, 1.11, 0.99]

    def synthesize(text, voice, path, factory):
        calls.append(text)
        if len(calls) == 2:
            path.write_bytes(b"partial")
            raise NoAudioReceived("temporary failure")
        assert factory == "factory"
        assert voice == "vi-VN-HoaiMyNeural"
        path.write_bytes(b"fake MP3")
        return [{"text": "Ben", "offset": 100_000, "duration": 200_000}]

    def probe(path):
        assert path.read_bytes() == b"fake MP3"
        return durations.pop(0)

    def concat(paths, target):
        target.write_bytes(b"joined:" + b"|".join(p.read_bytes() for p in paths))

    narration = "Ben 10! " * 110
    chunks = split_narration(narration, max_chars=180)
    durations[:] = [1.12 + i * .01 for i in range(len(chunks))]
    output = tmp_path / "narration.mp3"
    output.write_bytes(b"previous approved result")
    boundaries = synthesize_chunked(
        narration, "vi-VN-HoaiMyNeural", output,
        max_chars=180, synthesize=synthesize, communicate_factory="factory",
        probe_duration=probe, concat_audio=concat,
        no_audio_error=NoAudioReceived, sleep=lambda _: None,
    )
    assert output.read_bytes().startswith(b"joined:")
    assert len(boundaries) == len(chunks)
    assert len(calls) == len(chunks) + 1
    assert boundaries[0]["offset"] == 100_000
    assert boundaries[1]["offset"] == 100_000 + round(1.12 * 10_000_000)
    assert not list(tmp_path.glob("*.synthesizing*"))


def test_failed_chunk_splits_and_keeps_exact_words_and_offsets(tmp_path: Path):
    class NoAudioReceived(Exception):
        pass

    narration = "Ben 10 cứu Gwen rồi nhìn Omnitrix. " * 7
    calls = []
    synthesized = []

    def synthesize(text, _voice, path, _factory):
        calls.append(text)
        if len(text) > 150:
            path.write_bytes(b"partial")
            raise NoAudioReceived("transient")
        synthesized.append(text)
        path.write_bytes(text.encode("utf-8"))
        return [{"text": "Ben", "offset": 50_000, "duration": 100_000}]

    def concat(paths, target):
        target.write_bytes(b"".join(path.read_bytes() for path in paths))

    output = tmp_path / "narration.mp3"
    boundaries = synthesize_chunked(
        narration, "voice", output, max_chars=500, synthesize=synthesize,
        communicate_factory="factory", no_audio_error=NoAudioReceived,
        probe_duration=lambda _: 1.0, concat_audio=concat, sleep=lambda _: None,
    )
    assert len(calls) >= 5
    assert len(synthesized) == 2
    assert "".join(synthesized) == narration
    assert output.read_bytes() == narration.encode("utf-8")
    assert [b["offset"] for b in boundaries] == [50_000, 10_050_000]


def test_irreducible_failure_reports_safe_diagnostic_and_preserves_audio(tmp_path: Path):
    class NoAudioReceived(Exception):
        pass

    narration = "Bí mật của Ben." * 4

    def fail(_text, _voice, part, _factory):
        part.write_bytes(b"partial")
        raise NoAudioReceived("source diagnostic")

    output = tmp_path / "narration.mp3"
    output.write_bytes(b"approved")
    with pytest.raises(RuntimeError, match=r"chunk 0.*length 6[0-9].*sha256 [0-9a-f]{12}") as error:
        synthesize_chunked(
            narration, "voice", output, synthesize=fail,
            communicate_factory="factory", no_audio_error=NoAudioReceived,
            probe_duration=lambda _: 1.0, concat_audio=lambda *_: None,
            sleep=lambda _: None,
        )
    assert "Bí mật" not in str(error.value)
    assert output.read_bytes() == b"approved"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["narration.mp3"]


def test_chunk_failure_does_not_replace_approved_audio(tmp_path: Path):
    class NoAudioReceived(Exception):
        pass

    calls = 0

    def always_fail(_text, _voice, path, _factory):
        nonlocal calls
        calls += 1
        path.write_bytes(b"partial")
        raise NoAudioReceived("no audio")

    output = tmp_path / "narration.mp3"
    output.write_bytes(b"previous approved result")
    with pytest.raises(RuntimeError, match="TTS failed at chunk"):
        synthesize_chunked(
            "Lời thoại " * 50, "voice", output, max_chars=120,
            synthesize=always_fail, communicate_factory="factory", probe_duration=lambda _: 1.0,
            concat_audio=lambda *_: None, no_audio_error=NoAudioReceived,
            retries=3, sleep=lambda _: None,
        )
    assert calls >= 3
    assert output.read_bytes() == b"previous approved result"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["narration.mp3"]


def test_concat_timeout_maps_context_and_cleans_partial_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    part = tmp_path / "part.mp3"
    part.write_bytes(b"audio")
    output = tmp_path / "merged.mp3"

    def timeout(command, **_kwargs):
        output.write_bytes(b"partial")
        raise subprocess.TimeoutExpired(command, chunked_tts.FFMPEG_CONCAT_TIMEOUT_SECONDS)

    monkeypatch.setattr(chunked_tts.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match=r"timed out.*ffmpeg.*merged\.mp3"):
        chunked_tts.concat_mp3([part], output)
    assert not output.exists()
    assert not output.with_suffix(".concat.txt").exists()


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg binaries unavailable")
def test_real_ffmpeg_concatenates_mp3_without_truncating_duration(tmp_path: Path):
    from movie_review_factory.chunked_tts import concat_mp3, probe_mp3_duration

    parts = []
    for idx, freq in enumerate((420, 620, 850)):
        path = tmp_path / f"part-{idx}.mp3"
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi",
                        "-i", f"sine=frequency={freq}:duration=0.5", "-c:a", "libmp3lame",
                        "-q:a", "4", str(path)], check=True)
        parts.append(path)
    merged = tmp_path / "merged.mp3"
    concat_mp3(parts, merged, ffmpeg_bin="ffmpeg")
    assert merged.stat().st_size > max(p.stat().st_size for p in parts)
    assert 1.45 <= probe_mp3_duration(merged, ffprobe_bin="ffprobe") <= 1.7
