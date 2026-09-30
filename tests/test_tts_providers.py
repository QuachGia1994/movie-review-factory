"""Offline unit tests for the pluggable studio-grade TTS providers.

Zero network: the HTTP transport (``http_request``) or the whole ``fetch_audio``
seam is injected, and duration probing is stubbed, so these run fully offline like
the rest of the suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from movie_review_factory import tts_providers
from movie_review_factory.chunked_tts import synthesize_chunked
from movie_review_factory.narration_alignment import TICKS_PER_SECOND


def test_resolve_provider_defaults_to_edge(monkeypatch):
    monkeypatch.delenv(tts_providers.PROVIDER_ENV, raising=False)
    assert tts_providers.resolve_provider() == "edge"


def test_resolve_provider_env_then_config_wins(monkeypatch):
    monkeypatch.setenv(tts_providers.PROVIDER_ENV, "fptai")
    assert tts_providers.resolve_provider() == "fptai"

    class Cfg:
        tts_provider = "elevenlabs"

    assert tts_providers.resolve_provider(Cfg()) == "elevenlabs"


def test_resolve_provider_rejects_unknown(monkeypatch):
    monkeypatch.setenv(tts_providers.PROVIDER_ENV, "bogus")
    with pytest.raises(tts_providers.TTSProviderError):
        tts_providers.resolve_provider()


def test_provider_api_key_reads_env_only(monkeypatch):
    monkeypatch.setenv("MRF_ELEVENLABS_API_KEY", " secret ")
    assert tts_providers.provider_api_key("elevenlabs") == "secret"
    monkeypatch.delenv("MRF_FPTAI_API_KEY", raising=False)
    assert tts_providers.provider_api_key("fptai") is None


def test_resolve_voice_precedence(monkeypatch):
    monkeypatch.delenv(tts_providers.VOICE_ENV, raising=False)
    # per-provider default when nothing configured
    assert tts_providers.resolve_voice("fptai") == tts_providers.DEFAULT_VOICE["fptai"]
    # env override beats the default
    monkeypatch.setenv(tts_providers.VOICE_ENV, "env-voice")
    assert tts_providers.resolve_voice("elevenlabs") == "env-voice"

    # per-project config beats the env override
    class Cfg:
        tts_voice = "cfg-voice"

    assert tts_providers.resolve_voice("elevenlabs", cfg=Cfg()) == "cfg-voice"


def test_resolve_voice_reads_jobconfig_field(monkeypatch):
    from movie_review_factory.models import JobConfig

    monkeypatch.delenv(tts_providers.VOICE_ENV, raising=False)
    cfg = JobConfig(job_id="voice-job", tts_provider="fptai", tts_voice="leminh")
    assert tts_providers.resolve_voice("fptai", cfg=cfg) == "leminh"
    # blank config field falls through to the per-provider default
    blank = JobConfig(job_id="voice-job-2", tts_provider="fptai")
    assert tts_providers.resolve_voice("fptai", cfg=blank) == tts_providers.DEFAULT_VOICE["fptai"]


def test_estimate_word_boundaries_are_monotonic_and_cover_duration():
    boundaries = tts_providers.estimate_word_boundaries("Ben chạy rất nhanh", 4.0)
    assert [b["text"] for b in boundaries] == ["Ben", "chạy", "rất", "nhanh"]
    offsets = [b["offset"] for b in boundaries]
    assert offsets == sorted(offsets)
    assert offsets[0] == 0
    last = boundaries[-1]
    total_ticks = round(4.0 * TICKS_PER_SECOND)
    assert abs((last["offset"] + last["duration"]) - total_ticks) <= 100


def test_estimate_word_boundaries_rejects_nonpositive_duration():
    with pytest.raises(tts_providers.TTSProviderError):
        tts_providers.estimate_word_boundaries("x", 0)


def test_build_synthesize_requires_api_key_without_fetch():
    with pytest.raises(tts_providers.TTSProviderError):
        tts_providers.build_synthesize("elevenlabs", api_key=None)


def test_build_synthesize_rejects_non_network_provider():
    with pytest.raises(tts_providers.TTSProviderError):
        tts_providers.build_synthesize("edge", api_key="k")


def test_provider_synthesize_writes_audio_and_estimates_boundaries(tmp_path: Path):
    def fake_fetch(text, voice):
        assert voice == "voice-x"
        return b"FAKEMP3-" + text.encode("utf-8")

    synth = tts_providers.build_synthesize(
        "elevenlabs", fetch_audio=fake_fetch, probe_duration=lambda _p: 2.0,
    )
    part = tmp_path / "part.mp3"
    boundaries = synth("Ben chạy nhanh", "voice-x", part, None)
    assert part.read_bytes().startswith(b"FAKEMP3-")
    assert [b["text"] for b in boundaries] == ["Ben", "chạy", "nhanh"]


def test_empty_audio_raises(tmp_path: Path):
    synth = tts_providers.build_synthesize(
        "fptai", fetch_audio=lambda _t, _v: b"", probe_duration=lambda _p: 1.0,
    )
    with pytest.raises(tts_providers.TTSProviderError):
        synth("hi", "banmai", tmp_path / "p.mp3", None)


def test_elevenlabs_default_fetcher_posts_to_voice_url(tmp_path: Path):
    seen = {}

    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        seen["url"] = url
        seen["method"] = method
        seen["api_key"] = (headers or {}).get("xi-api-key")
        return 200, b"AUDIO"

    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="secret", http_request=fake_request, probe_duration=lambda _p: 1.0,
    )
    part = tmp_path / "p.mp3"
    synth("Hello world", "VOICEID", part, None)
    assert seen["method"] == "POST"
    assert seen["url"].endswith("/VOICEID")
    assert seen["api_key"] == "secret"
    assert part.read_bytes() == b"AUDIO"


def test_fptai_default_fetcher_polls_async_url(tmp_path: Path):
    responses = iter([
        (200, b'{"async": "https://fpt.example/audio.mp3", "error": 0}'),  # POST
        (404, b""),          # first poll: not ready yet
        (200, b"FPTAUDIO"),  # second poll: ready
    ])

    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        return next(responses)

    synth = tts_providers.build_synthesize(
        "fptai", api_key="k", http_request=fake_request,
        probe_duration=lambda _p: 1.0, sleep=lambda _s: None,
    )
    part = tmp_path / "p.mp3"
    synth("Xin chào", "banmai", part, None)
    assert part.read_bytes() == b"FPTAUDIO"


def test_provider_integrates_with_synthesize_chunked_single_chunk(tmp_path: Path):
    synth = tts_providers.build_synthesize(
        "fptai", fetch_audio=lambda text, _v: b"MP3:" + text.encode("utf-8"),
        probe_duration=lambda _p: 1.5,
    )
    out = tmp_path / "narration.mp3"
    boundaries = synthesize_chunked(
        "Xin chào các bạn", "banmai", out, synthesize=synth,
        communicate_factory=None, no_audio_error=RuntimeError,
        probe_duration=lambda _p: 1.5,
        concat_audio=lambda *_: pytest.fail("single chunk must not remux"),
    )
    assert out.read_bytes().startswith(b"MP3:")
    assert boundaries and all("offset" in b for b in boundaries)
