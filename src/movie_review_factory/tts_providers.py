"""Pluggable studio-grade TTS providers (FPT.AI / ElevenLabs) alongside Edge-TTS.

Edge-TTS stays the free, offline default and keeps its real word-boundary timing.
FPT.AI and ElevenLabs return audio only, so this module estimates word timing from
the probed chunk duration and exposes a ``synthesize``-compatible callable that
drops straight into :func:`chunked_tts.synthesize_chunked`.

Design notes:
- API keys are read from the environment only (never persisted in a manifest).
- The HTTP layer is injectable (``fetch_audio`` / ``http_request``) so the whole
  module is unit-testable offline with zero network calls.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .chunked_tts import probe_mp3_duration
from .narration_alignment import TICKS_PER_SECOND

SUPPORTED_PROVIDERS = ("edge", "fptai", "elevenlabs")
NETWORK_PROVIDERS = ("fptai", "elevenlabs")

PROVIDER_ENV = "MRF_TTS_PROVIDER"
VOICE_ENV = "MRF_TTS_VOICE"
KEY_ENV = {"fptai": "MRF_FPTAI_API_KEY", "elevenlabs": "MRF_ELEVENLABS_API_KEY"}

# Sensible defaults; a user picking a network provider normally sets MRF_TTS_VOICE.
DEFAULT_VOICE = {"fptai": "banmai", "elevenlabs": "21m00Tcm4TlvDq8ikWAM"}

FPTAI_TTS_URL = "https://api.fpt.ai/hmi/tts/v5"
ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
ELEVENLABS_SUBSCRIPTION_URL = "https://api.elevenlabs.io/v1/user/subscription"
ELEVENLABS_MODEL = "eleven_multilingual_v2"


class TTSProviderError(RuntimeError):
    """A studio-grade TTS provider could not produce audio."""


def resolve_provider(cfg: object | None = None) -> str:
    """Pick the TTS provider from job config then env, defaulting to 'edge'."""
    value = (getattr(cfg, "tts_provider", "") or os.environ.get(PROVIDER_ENV, "") or "edge")
    value = str(value).strip().lower()
    if value not in SUPPORTED_PROVIDERS:
        raise TTSProviderError(f"{PROVIDER_ENV} phải là {', '.join(SUPPORTED_PROVIDERS)}")
    return value


def provider_api_key(provider: str) -> str | None:
    """Read the provider API key from its env var only; never from the manifest."""
    return os.environ.get(KEY_ENV.get((provider or "").lower(), ""), "").strip() or None


def missing_key_message(provider: str) -> str:
    env = KEY_ENV.get((provider or "").lower(), "MRF_TTS_API_KEY")
    return f"{provider} TTS cần API key qua biến môi trường {env} (không lưu vào manifest)"


def resolve_voice(provider: str, *, cfg: object | None = None) -> str:
    """Voice id/name from job config, then MRF_TTS_VOICE, else a per-provider default.

    Config wins over the env override (per-project beats machine-wide), matching how
    the provider itself is resolved. Not used by the edge provider.
    """
    configured = str(getattr(cfg, "tts_voice", "") or "").strip()
    if configured:
        return configured
    return os.environ.get(VOICE_ENV, "").strip() or DEFAULT_VOICE.get((provider or "").lower(), "")


def estimate_word_boundaries(text: str, duration_seconds: float) -> list[dict]:
    """Distribute word timings across the probed chunk duration, weighted by length.

    Providers that return audio-only give no word events, so this keeps the same
    ``{offset, duration, text}`` (ticks) shape edge-tts emits, letting the existing
    caption / section-boundary alignment run unchanged (approximate, not exact).
    """
    total = float(duration_seconds)
    if not math.isfinite(total) or total <= 0:
        raise TTSProviderError("estimated word timing needs a positive audio duration")
    tokens = re.findall(r"\S+", text or "")
    if not tokens:
        return [{"offset": 0, "duration": round(total * TICKS_PER_SECOND), "text": (text or "").strip()}]
    weights = [max(1, len(token)) for token in tokens]
    weight_sum = sum(weights)
    boundaries: list[dict] = []
    cursor = 0.0
    for token, weight in zip(tokens, weights):
        span = total * weight / weight_sum
        boundaries.append({
            "offset": round(cursor * TICKS_PER_SECOND),
            "duration": round(span * TICKS_PER_SECOND),
            "text": token,
        })
        cursor += span
    return boundaries


def _http_request(url: str, *, method: str = "GET", headers: dict | None = None,
                  data: bytes | None = None, timeout: float = 60.0) -> tuple[int, bytes]:
    """Minimal stdlib HTTP; returns (status, body) and maps transport errors."""
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(getattr(response, "status", None) or response.getcode()), response.read()
    except urllib.error.HTTPError as exc:  # non-2xx: surface the code to the caller
        return int(exc.code), exc.read()
    except urllib.error.URLError as exc:
        raise TTSProviderError(f"TTS request tới {url} thất bại: {exc.reason}") from exc


def _elevenlabs_fetcher(api_key: str, *, request: Callable, timeout: float) -> Callable[[str, str], bytes]:
    def fetch(text: str, voice: str) -> bytes:
        url = ELEVENLABS_TTS_URL.format(voice=voice)
        body = json.dumps({
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "style": 0.35},
        }).encode("utf-8")
        headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
        status, payload = request(url, method="POST", headers=headers, data=body, timeout=timeout)
        if status != 200 or not payload:
            raise TTSProviderError(f"ElevenLabs TTS lỗi HTTP {status}")
        return payload
    return fetch


def _fptai_fetcher(api_key: str, *, request: Callable, sleep: Callable[[float], None],
                   timeout: float, poll_attempts: int = 8, poll_delay: float = 1.0) -> Callable[[str, str], bytes]:
    def fetch(text: str, voice: str) -> bytes:
        headers = {"api-key": api_key, "voice": voice, "speed": ""}
        status, payload = request(FPTAI_TTS_URL, method="POST", headers=headers,
                                  data=text.encode("utf-8"), timeout=timeout)
        if status != 200 or not payload:
            raise TTSProviderError(f"FPT.AI TTS lỗi HTTP {status}")
        try:
            info = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise TTSProviderError("FPT.AI trả về phản hồi không hợp lệ") from exc
        audio_url = info.get("async") if isinstance(info, dict) else None
        if not audio_url:
            detail = info.get("message") if isinstance(info, dict) else ""
            raise TTSProviderError(f"FPT.AI không trả về link audio: {detail}")
        # FPT.AI renders asynchronously; poll the returned URL until the MP3 is ready.
        for _ in range(max(1, poll_attempts)):
            audio_status, audio = request(str(audio_url), method="GET", timeout=timeout)
            if audio_status == 200 and audio:
                return audio
            sleep(poll_delay)
        raise TTSProviderError("FPT.AI chưa tạo xong audio sau nhiều lần thử")
    return fetch


def build_synthesize(
    provider: str,
    *,
    api_key: str | None = None,
    ffprobe_bin: str = "ffprobe",
    fetch_audio: Callable[[str, str], bytes] | None = None,
    probe_duration: Callable[[Path], float] | None = None,
    http_request: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 60.0,
) -> Callable:
    """Build a ``synthesize(text, voice, part, communicate_factory)`` for a network provider.

    Matches the callable :func:`chunked_tts.synthesize_chunked` expects. ``fetch_audio``
    and ``probe_duration`` are injectable for offline tests; otherwise the real HTTP
    provider and ffprobe are used. ``communicate_factory`` is ignored (edge-only).
    """
    provider = (provider or "").lower()
    if provider not in NETWORK_PROVIDERS:
        raise TTSProviderError(f"build_synthesize chỉ hỗ trợ {', '.join(NETWORK_PROVIDERS)}")
    if fetch_audio is None:
        if not api_key:
            raise TTSProviderError(missing_key_message(provider))
        request = http_request or _http_request
        if provider == "fptai":
            fetch_audio = _fptai_fetcher(api_key, request=request, sleep=sleep, timeout=timeout)
        else:
            fetch_audio = _elevenlabs_fetcher(api_key, request=request, timeout=timeout)
    probe = probe_duration or (lambda path: probe_mp3_duration(path, ffprobe_bin=ffprobe_bin))

    def synthesize(text: str, voice: str, part: Path, communicate_factory: object = None) -> list[dict]:
        audio = fetch_audio(text, voice)
        if not audio:
            raise TTSProviderError(f"{provider} không trả về audio")
        part = Path(part)
        part.write_bytes(audio)
        duration = probe(part)
        return estimate_word_boundaries(text, duration)

    return synthesize


def check_provider(
    provider: str,
    *,
    api_key: str | None = None,
    http_request: Callable | None = None,
    timeout: float = 15.0,
) -> dict:
    """Cheaply check a provider's key/connectivity without spending synthesis credits.

    Returns ``{"provider", "ok", "detail"}`` and never raises for an ordinary failure
    (a transport or auth error is reported as ``ok=False`` so the UI can show it).
    - ``edge``: no key required.
    - ``elevenlabs``: GET the subscription endpoint - validates the key and reports the
      remaining character quota at zero synthesis cost.
    - ``fptai``: no free balance endpoint exists, so only key presence is verified; a
      real (billable) synthesis probe is intentionally not run here.
    """
    provider = (provider or "").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        return {"provider": provider, "ok": False,
                "detail": f"Nhà cung cấp không hỗ trợ (chọn {', '.join(SUPPORTED_PROVIDERS)})."}
    if provider == "edge":
        return {"provider": provider, "ok": True, "detail": "Edge-TTS miễn phí, không cần API key."}
    if not api_key:
        return {"provider": provider, "ok": False, "detail": missing_key_message(provider)}
    if provider == "fptai":
        return {"provider": provider, "ok": True,
                "detail": "Đã có API key. FPT.AI không có endpoint kiểm tra số dư miễn phí, "
                          "nên chỉ xác nhận key tồn tại (chưa gọi tổng hợp tính phí)."}
    request = http_request or _http_request
    try:
        status, payload = request(
            ELEVENLABS_SUBSCRIPTION_URL, method="GET",
            headers={"xi-api-key": api_key, "accept": "application/json"}, timeout=timeout,
        )
    except TTSProviderError as exc:
        return {"provider": provider, "ok": False, "detail": str(exc)}
    if status in (401, 403):
        return {"provider": provider, "ok": False, "detail": f"API key không hợp lệ (HTTP {status})."}
    if status != 200 or not payload:
        return {"provider": provider, "ok": False, "detail": f"ElevenLabs trả về HTTP {status}."}
    try:
        info = json.loads(payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        info = {}
    used = int(info.get("character_count") or 0)
    limit = int(info.get("character_limit") or 0)
    remaining = max(0, limit - used)
    return {"provider": provider, "ok": True,
            "detail": f"Kết nối OK. Còn {remaining:,}/{limit:,} ký tự trong hạn mức."}
