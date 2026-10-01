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


def test_elevenlabs_retries_503_then_succeeds(tmp_path: Path):
    responses = iter([(503, b"busy"), (200, b"AUDIO")])
    sleeps = []

    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="secret",
        http_request=lambda *_args, **_kwargs: next(responses),
        probe_duration=lambda _p: 1.0, sleep=sleeps.append,
    )
    part = tmp_path / "p.mp3"
    synth("Hello", "VOICEID", part, None)
    assert part.read_bytes() == b"AUDIO"
    assert sleeps == [0.5]


def test_elevenlabs_429_honors_retry_after(tmp_path: Path):
    responses = iter([(429, b"limited", {"Retry-After": "2.5"}), (200, b"AUDIO")])
    sleeps = []
    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="secret",
        http_request=lambda *_args, **_kwargs: next(responses),
        probe_duration=lambda _p: 1.0, sleep=sleeps.append,
    )
    synth("Hello", "VOICEID", tmp_path / "p.mp3", None)
    assert sleeps == [2.5]


def test_elevenlabs_retries_timeout_then_succeeds(tmp_path: Path):
    calls = 0

    def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("slow")
        return 200, b"AUDIO"

    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="secret", http_request=fake_request,
        probe_duration=lambda _p: 1.0, sleep=lambda _seconds: None,
    )
    synth("Hello", "VOICEID", tmp_path / "p.mp3", None)
    assert calls == 2


def test_elevenlabs_401_fails_without_retry(tmp_path: Path):
    calls = 0

    def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return 401, b"unauthorized"

    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="bad", http_request=fake_request,
        probe_duration=lambda _p: 1.0,
        sleep=lambda _seconds: pytest.fail("must not retry"),
    )
    with pytest.raises(tts_providers.TTSProviderError, match="HTTP 401"):
        synth("Hello", "VOICEID", tmp_path / "p.mp3", None)
    assert calls == 1


def test_elevenlabs_exhausted_retries_raises(tmp_path: Path):
    calls = 0

    def fake_request(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return 503, b"busy"

    synth = tts_providers.build_synthesize(
        "elevenlabs", api_key="secret", http_request=fake_request,
        probe_duration=lambda _p: 1.0, sleep=lambda _seconds: None,
    )
    with pytest.raises(tts_providers.TTSProviderError, match="HTTP 503"):
        synth("Hello", "VOICEID", tmp_path / "p.mp3", None)
    assert calls == 3


def test_fptai_default_fetcher_polls_404_without_reposting(tmp_path: Path):
    responses = iter([
        (200, b'{"async": "https://fpt.example/audio.mp3", "error": 0}'),
        (404, b""),
        (200, b"FPTAUDIO"),
    ])
    methods = []

    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        methods.append(method)
        return next(responses)

    synth = tts_providers.build_synthesize(
        "fptai", api_key="k", http_request=fake_request,
        probe_duration=lambda _p: 1.0, sleep=lambda _s: None,
    )
    part = tmp_path / "p.mp3"
    synth("Xin chào", "banmai", part, None)
    assert part.read_bytes() == b"FPTAUDIO"
    assert methods == ["POST", "GET", "GET"]


def test_fptai_poll_401_fails_fast_without_reposting(tmp_path: Path):
    responses = iter([
        (200, b'{"async": "https://fpt.example/audio.mp3", "error": 0}'),
        (401, b"unauthorized"),
    ])
    methods = []

    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        methods.append(method)
        return next(responses)

    synth = tts_providers.build_synthesize(
        "fptai", api_key="k", http_request=fake_request,
        probe_duration=lambda _p: 1.0,
        sleep=lambda _seconds: pytest.fail("401 must fail fast"),
    )
    with pytest.raises(tts_providers.TTSProviderError, match="HTTP 401"):
        synth("Xin chào", "banmai", tmp_path / "p.mp3", None)
    assert methods == ["POST", "GET"]


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


def test_check_provider_edge_needs_no_key():
    result = tts_providers.check_provider("edge")
    assert result["provider"] == "edge"
    assert result["ok"] is True


def test_check_provider_missing_key_is_not_ok():
    result = tts_providers.check_provider("elevenlabs", api_key=None)
    assert result["ok"] is False
    assert "MRF_ELEVENLABS_API_KEY" in result["detail"]


def test_check_provider_rejects_unknown_provider():
    result = tts_providers.check_provider("bogus", api_key="k")
    assert result["ok"] is False


def test_check_provider_elevenlabs_valid_reports_quota():
    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        assert method == "GET"
        assert url == tts_providers.ELEVENLABS_SUBSCRIPTION_URL
        assert (headers or {}).get("xi-api-key") == "secret"
        return 200, b'{"character_count": 1000, "character_limit": 10000}'

    result = tts_providers.check_provider("elevenlabs", api_key="secret", http_request=fake_request)
    assert result["ok"] is True
    assert "9,000" in result["detail"]


def test_check_provider_elevenlabs_invalid_key_is_not_ok():
    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        return 401, b'{"detail": "unauthorized"}'

    result = tts_providers.check_provider("elevenlabs", api_key="bad", http_request=fake_request)
    assert result["ok"] is False
    assert "401" in result["detail"]


def test_check_provider_network_error_is_reported_not_raised():
    def fake_request(url, *, method="GET", headers=None, data=None, timeout=60.0):
        raise tts_providers.TTSProviderError("mạng lỗi")

    result = tts_providers.check_provider("elevenlabs", api_key="secret", http_request=fake_request)
    assert result["ok"] is False
    assert "mạng lỗi" in result["detail"]


def test_check_provider_fptai_present_key_is_ok_without_network():
    calls = []

    def fake_request(*args, **kwargs):
        calls.append(1)
        return 200, b""

    result = tts_providers.check_provider("fptai", api_key="k", http_request=fake_request)
    assert result["ok"] is True
    assert calls == []  # FPT.AI presence check must not hit the network


def test_resolve_provider_accepts_vieneu(monkeypatch):
    monkeypatch.setenv(tts_providers.PROVIDER_ENV, "vieneu")
    assert tts_providers.resolve_provider() == "vieneu"
    assert "vieneu" in tts_providers.SUPPORTED_PROVIDERS
    # VieNeu is local/offline, not a network provider that needs an API key.
    assert "vieneu" not in tts_providers.NETWORK_PROVIDERS
    assert "vieneu" in tts_providers.LOCAL_PROVIDERS


def test_vieneu_synthesize_needs_no_key_and_estimates_boundaries(tmp_path: Path):
    def fake_engine(text, voice):
        assert voice == ""  # no default voice configured -> engine default
        return b"MP3:" + text.encode("utf-8")

    synth = tts_providers.build_synthesize(
        "vieneu", synthesize_audio=fake_engine, probe_duration=lambda _p: 2.0,
    )
    part = tmp_path / "part.mp3"
    boundaries = synth("Ben chạy nhanh", "", part, None)
    assert part.read_bytes().startswith(b"MP3:")
    assert [b["text"] for b in boundaries] == ["Ben", "chạy", "nhanh"]


def test_vieneu_integrates_with_synthesize_chunked_single_chunk(tmp_path: Path):
    synth = tts_providers.build_synthesize(
        "vieneu", synthesize_audio=lambda text, _v: b"VIENEU:" + text.encode("utf-8"),
        probe_duration=lambda _p: 1.5,
    )
    out = tmp_path / "narration.mp3"
    boundaries = synthesize_chunked(
        "Xin chào các bạn", "", out, synthesize=synth,
        communicate_factory=None, no_audio_error=RuntimeError,
        probe_duration=lambda _p: 1.5,
        concat_audio=lambda *_: pytest.fail("single chunk must not remux"),
    )
    assert out.read_bytes().startswith(b"VIENEU:")
    assert boundaries and all("offset" in b for b in boundaries)


def test_vieneu_available_uses_injected_importer():
    assert tts_providers.vieneu_available(import_module=lambda name: object()) is True

    def boom(name):
        raise ImportError(name)

    assert tts_providers.vieneu_available(
        import_module=boom, isolated_check=lambda: False,
    ) is False


def test_check_provider_vieneu_available_needs_no_key(monkeypatch):
    monkeypatch.setattr(tts_providers, "vieneu_available", lambda: True)
    result = tts_providers.check_provider("vieneu")
    assert result["provider"] == "vieneu"
    assert result["ok"] is True
    assert "không cần API key" in result["detail"]


def test_check_provider_vieneu_missing_package_reports_install_hint(monkeypatch):
    monkeypatch.setattr(tts_providers, "vieneu_available", lambda: False)
    result = tts_providers.check_provider("vieneu")
    assert result["ok"] is False
    assert "pip install vieneu" in result["detail"]


def test_vieneu_infer_kwargs_routes_preset_default_and_clone(tmp_path: Path):
    # empty -> model default voice
    assert tts_providers._vieneu_infer_kwargs("") == {}
    assert tts_providers._vieneu_infer_kwargs("   ") == {}
    # a plain name -> preset voice
    assert tts_providers._vieneu_infer_kwargs("Mai Anh") == {"voice": "Mai Anh"}
    # an existing audio file -> zero-shot clone reference
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF0000WAVE")
    assert tts_providers._vieneu_infer_kwargs(str(ref)) == {"ref_audio": str(ref)}
