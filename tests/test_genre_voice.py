import json

import pytest

from movie_review_factory import genre_tone, narration_style, pipeline, tts_providers
from movie_review_factory.models import JobConfig

_ENV = ("MRF_TTS_EMOTION", "MRF_TTS_RATE", "MRF_TTS_PITCH", "MRF_TTS_VOLUME", "MRF_TTS_PROVIDER", "MRF_TTS_VOICE", "MRF_GENRE_VOICE")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


def _cfg(**kwargs):
    return JobConfig(job_id="j1", language="vi", **kwargs)


def test_voice_follows_genre_mood():
    assert genre_tone.voice("edge", "vi", "Kinh dị") == "vi-VN-NamMinhNeural"
    assert genre_tone.voice("edge", "vi", "Tình cảm") == "vi-VN-HoaiMyNeural"
    assert genre_tone.voice("edge", "en", "horror") == "en-US-GuyNeural"
    assert genre_tone.voice("fptai", "vi", "hành động") == "leminh"
    assert genre_tone.voice("elevenlabs", "vi", "kinh dị") == ""
    assert genre_tone.voice("edge", "ja", "horror") == ""
    assert genre_tone.voice("edge", "vi", "") == ""


def test_rate_factor():
    assert genre_tone.rate_factor("-10%") == pytest.approx(0.9)
    assert genre_tone.rate_factor("+8%") == pytest.approx(1.08)
    assert genre_tone.rate_factor("") == 1.0
    assert genre_tone.rate_factor("fast") == 1.0


def test_every_genre_prosody_is_valid_edge_syntax():
    for key, values in genre_tone._PROSODY.items():
        assert key in genre_tone._ALIASES
        assert pipeline._TTS_PERCENT_RE.match(values["rate"])
        if "pitch" in values:
            assert pipeline._TTS_HZ_RE.match(values["pitch"])


def test_prosody_genre_then_emotion_then_env(monkeypatch):
    assert pipeline._tts_prosody("kinh dị") == {"rate": "-10%", "pitch": "-6Hz"}
    monkeypatch.setenv("MRF_TTS_EMOTION", "hype")
    assert pipeline._tts_prosody("kinh dị")["rate"] == "+18%"
    monkeypatch.delenv("MRF_TTS_EMOTION")
    monkeypatch.setenv("MRF_TTS_RATE", "-20%")
    assert pipeline._tts_prosody("kinh dị") == {"rate": "-20%", "pitch": "-6Hz"}


def test_speech_factor_scales_word_budget(monkeypatch):
    horror = _cfg(genre="Kinh dị")
    assert pipeline._speech_factor(horror) == pytest.approx(0.9)
    assert pipeline._words_per_minute(horror) == 216
    assert pipeline._target_words({"budget_minutes": 2}, horror) == 432
    assert pipeline._words_per_minute(_cfg()) == narration_style.words_per_minute("vi")
    assert pipeline._speech_factor(_cfg(genre="Kinh dị", tts_provider="vieneu")) == 1.0
    monkeypatch.setenv("MRF_GENRE_VOICE", "0")
    assert pipeline._speech_factor(horror) == 1.0
    assert pipeline._voice_genre(horror) == ""


def test_prompt_style_factor():
    style = narration_style.prompt_style("vi", [{"title": "A", "budget_minutes": 1}], 0.9)
    assert style["words_per_minute"] == 216
    assert style["section_word_targets"][0]["target_words"] == 216


def _capture(store):
    def request(url, method="GET", headers=None, data=None, timeout=None):
        store.append({"url": url, "headers": headers or {}, "data": data})
        if url.startswith("https://api.fpt.ai"):
            return 200, json.dumps({"async": "https://cdn.example/a.mp3"}).encode()
        return 200, b"ID3audio"
    return request


def test_elevenlabs_speed_only_when_changed():
    calls = []
    fetch = tts_providers._elevenlabs_fetcher("k", request=_capture(calls), sleep=lambda s: None, timeout=1, speed=0.9)
    fetch("xin chào", "v")
    assert json.loads(calls[0]["data"])["voice_settings"]["speed"] == 0.9
    calls.clear()
    tts_providers._elevenlabs_fetcher("k", request=_capture(calls), sleep=lambda s: None, timeout=1)("a", "v")
    assert "speed" not in json.loads(calls[0]["data"])["voice_settings"]


def test_fptai_speed_header_steps():
    calls = []
    tts_providers._fptai_fetcher("k", request=_capture(calls), sleep=lambda s: None, timeout=1, speed=0.9)("a", "leminh")
    assert calls[0]["headers"]["speed"] == "-1"
    calls.clear()
    tts_providers._fptai_fetcher("k", request=_capture(calls), sleep=lambda s: None, timeout=1)("a", "banmai")
    assert calls[0]["headers"]["speed"] == ""
