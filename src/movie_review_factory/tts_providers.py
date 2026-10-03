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

import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from .chunked_tts import probe_mp3_duration
from .narration_alignment import TICKS_PER_SECOND

# 'vieneu' is a free local/offline voice: no API key, needs the optional package; output is transcoded to MP3 for the shared chunk pipeline.
SUPPORTED_PROVIDERS = ("edge", "fptai", "elevenlabs", "vieneu")
NETWORK_PROVIDERS = ("fptai", "elevenlabs")
LOCAL_PROVIDERS = ("vieneu",)

PROVIDER_ENV = "MRF_TTS_PROVIDER"
VOICE_ENV = "MRF_TTS_VOICE"
KEY_ENV = {"fptai": "MRF_FPTAI_API_KEY", "elevenlabs": "MRF_ELEVENLABS_API_KEY"}

# Always transcode VieNeu to 24 kHz mono MP3 so concat gets one format regardless of model rate.
VIENEU_SAMPLE_RATE = 24000
VIENEU_MODE_ENV = "MRF_VIENEU_MODE"
VIENEU_PRECISION_ENV = "MRF_VIENEU_PRECISION"
VIENEU_ENV_ENV = "MRF_VIENEU_ENV"
VIENEU_BRIDGE_TIMEOUT = 900
VIENEU_INSTALL_HINT = (
    "VieNeu-TTS (giọng miễn phí, offline 24kHz) chưa sẵn sàng. "
    "Dùng nút cài tự động hoặc chạy pip install vieneu; trên Windows/Python 3.14+ "
    "ứng dụng sẽ tạo môi trường Python 3.12 riêng — không cần API key hay bộ biên dịch C++."
)

# Inline so an isolated interpreter needs only VieNeu; input is JSON in a private temp dir, never shell syntax.
_VIENEU_BRIDGE_CODE = r'''import json, sys
from pathlib import Path
from vieneu import Vieneu
request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
kwargs = {key: request[key] for key in ("mode", "precision") if request.get(key)}
tts = Vieneu(**kwargs)
voice = str(request.get("voice") or "").strip()
infer_kwargs = {}
if voice:
    infer_kwargs["ref_audio" if Path(voice).is_file() else "voice"] = voice
audio = tts.infer(request["text"], **infer_kwargs)
tts.save(audio, sys.argv[2])
'''

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


def _response_parts(response: tuple) -> tuple[int, bytes, dict[str, str]]:
    """Accept legacy two-item transports and optional response headers."""
    if len(response) == 2:
        status, payload = response
        return int(status), payload, {}
    status, payload, headers = response
    return int(status), payload, {str(k).lower(): str(v) for k, v in dict(headers or {}).items()}


def _retry_delay(headers: dict[str, str], attempt: int, base_delay: float) -> float:
    try:
        retry_after = float(headers.get("retry-after", ""))
    except ValueError:
        retry_after = 0.0
    return min(30.0, retry_after if retry_after >= 0 else 0.0) or min(8.0, base_delay * (2 ** attempt))


def _request_with_retry(request: Callable, url: str, *, sleep: Callable[[float], None],
                        attempts: int, base_delay: float, **kwargs) -> tuple[int, bytes, dict[str, str]]:
    attempts = min(4, max(1, int(attempts)))
    for attempt in range(attempts):
        try:
            status, payload, headers = _response_parts(request(url, **kwargs))
        except (OSError, TimeoutError, TTSProviderError) as exc:
            if attempt + 1 >= attempts:
                raise TTSProviderError(f"TTS request tới {url} thất bại sau {attempts} lần thử: {exc}") from exc
            sleep(min(8.0, base_delay * (2 ** attempt)))
            continue
        if status != 429 and status < 500:
            return status, payload, headers
        if attempt + 1 >= attempts:
            return status, payload, headers
        sleep(_retry_delay(headers, attempt, base_delay))
    raise AssertionError("unreachable")


def _elevenlabs_fetcher(api_key: str, *, request: Callable, sleep: Callable[[float], None],
                         timeout: float, retry_attempts: int = 3,
                         retry_delay: float = 0.5, speed: float = 1.0) -> Callable[[str, str], bytes]:
    settings = {"stability": 0.5, "similarity_boost": 0.75, "style": 0.35}
    if abs(speed - 1.0) >= 0.01:
        settings["speed"] = round(max(0.7, min(1.2, speed)), 2)

    def fetch(text: str, voice: str) -> bytes:
        url = ELEVENLABS_TTS_URL.format(voice=voice)
        body = json.dumps({
            "text": text,
            "model_id": ELEVENLABS_MODEL,
            "voice_settings": settings,
        }).encode("utf-8")
        headers = {"xi-api-key": api_key, "accept": "audio/mpeg", "content-type": "application/json"}
        status, payload, _ = _request_with_retry(
            request, url, method="POST", headers=headers, data=body, timeout=timeout,
            sleep=sleep, attempts=retry_attempts, base_delay=retry_delay,
        )
        if status != 200 or not payload:
            raise TTSProviderError(f"ElevenLabs TTS lỗi HTTP {status}")
        return payload
    return fetch


def _fptai_fetcher(api_key: str, *, request: Callable, sleep: Callable[[float], None],
                   timeout: float, poll_attempts: int = 8, poll_delay: float = 1.0,
                   retry_attempts: int = 3, retry_delay: float = 0.5, speed: float = 1.0) -> Callable[[str, str], bytes]:
    step = max(-3, min(3, round((speed - 1.0) * 10)))

    def fetch(text: str, voice: str) -> bytes:
        headers = {"api-key": api_key, "voice": voice, "speed": str(step) if step else ""}
        status, payload, _ = _request_with_retry(
            request, FPTAI_TTS_URL, method="POST", headers=headers,
            data=text.encode("utf-8"), timeout=timeout, sleep=sleep,
            attempts=retry_attempts, base_delay=retry_delay,
        )
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
        for attempt in range(max(1, poll_attempts)):
            audio_status, audio, response_headers = _response_parts(
                request(str(audio_url), method="GET", timeout=timeout)
            )
            if audio_status == 200 and audio:
                return audio
            if 400 <= audio_status < 500 and audio_status not in {404, 429}:
                raise TTSProviderError(f"FPT.AI audio lỗi HTTP {audio_status}")
            if attempt + 1 < max(1, poll_attempts):
                sleep(_retry_delay(response_headers, attempt, poll_delay))
        raise TTSProviderError("FPT.AI chưa tạo xong audio sau nhiều lần thử")
    return fetch


def vieneu_env_dir() -> Path:
    """Return the portable per-user VieNeu environment path.

    An explicit override supports managed/offline deployments. The default follows
    the platform cache convention and never depends on the repository or current
    working directory.
    """
    override = os.environ.get(VIENEU_ENV_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    cache_root = os.environ.get("LOCALAPPDATA", "").strip()
    return (Path(cache_root) if cache_root else Path.home() / ".cache") / "movie-review-factory" / "vieneu-py312"


def vieneu_env_python(env_dir: Path | None = None) -> Path:
    root = Path(env_dir) if env_dir is not None else vieneu_env_dir()
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _isolated_vieneu_available(
    python: Path | None = None,
    *,
    run: Callable[..., object] = subprocess.run,
) -> bool:
    executable = Path(python) if python is not None else vieneu_env_python()
    if not executable.is_file():
        return False
    try:
        proc = run(
            [str(executable), "-c", "import vieneu"],
            capture_output=True, text=True, shell=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return getattr(proc, "returncode", 1) == 0


def vieneu_available(
    *,
    import_module: Callable[[str], object] | None = None,
    isolated_check: Callable[[], bool] | None = None,
) -> bool:
    """True when VieNeu imports in-process or in the isolated environment."""
    importer = import_module or importlib.import_module
    try:
        importer("vieneu")
        return True
    except Exception:  # noqa: BLE001 - any import failure means direct mode is unusable
        check = isolated_check or _isolated_vieneu_available
        return check()


def _vieneu_infer_kwargs(voice: str) -> dict:
    """Map a configured voice to VieNeu ``infer`` kwargs (per the VieNeu SDK).

    An existing audio-file path is treated as a zero-shot clone reference
    (``ref_audio``); any other non-empty value is a preset voice name (``voice``);
    empty falls back to the model's default voice.
    """
    voice = (voice or "").strip()
    if voice and Path(voice).is_file():
        return {"ref_audio": voice}
    if voice:
        return {"voice": voice}
    return {}


def _vieneu_synthesizer(
    *,
    sample_rate: int = VIENEU_SAMPLE_RATE,
    ffmpeg_bin: str = "ffmpeg",
    mode: str | None = None,
    precision: str | None = None,
) -> Callable[[str, str], bytes]:
    """Build a local VieNeu-TTS producer returning mono MP3 bytes at ``sample_rate``.

    The heavy ``vieneu`` package + model is imported lazily and instantiated once, then
    reused across chunks. Each chunk is rendered to a temp WAV via the VieNeu SDK
    (``Vieneu().infer()`` / ``.save()``, with a preset ``voice`` name or an audio-file
    path used as a zero-shot ``ref_audio`` clone) and transcoded to MP3 with ffmpeg so it slots
    into the existing MP3 concat pipeline unchanged. Raises :class:`TTSProviderError`
    with an install hint when the optional package is missing. This whole seam is
    injectable via ``build_synthesize(..., synthesize_audio=)`` so unit tests stay
    fully offline and never load the model.
    """
    engine: dict = {}

    def _load():
        if "tts" not in engine:
            try:
                from vieneu import Vieneu  # optional heavy dep, never bundled
            except ImportError:
                engine["tts"] = None
                return None
            kwargs = {}
            if mode:
                kwargs["mode"] = mode
            if precision:
                kwargs["precision"] = precision
            engine["tts"] = Vieneu(**kwargs)
        return engine["tts"]

    def produce(text: str, voice: str) -> bytes:
        tts = _load()
        with tempfile.TemporaryDirectory(prefix=".vieneu-") as tmp:
            root = Path(tmp)
            wav_path = root / "chunk.wav"
            if tts is not None:
                audio = tts.infer(text, **_vieneu_infer_kwargs(voice))
                tts.save(audio, str(wav_path))
            else:
                python = vieneu_env_python()
                if not _isolated_vieneu_available(python):
                    raise TTSProviderError(VIENEU_INSTALL_HINT)
                request_path = root / "request.json"
                request_path.write_text(json.dumps({
                    "text": text, "voice": voice, "mode": mode, "precision": precision,
                }, ensure_ascii=False), encoding="utf-8")
                try:
                    proc = subprocess.run(
                        [str(python), "-c", _VIENEU_BRIDGE_CODE, str(request_path), str(wav_path)],
                        capture_output=True, text=True, shell=False, timeout=VIENEU_BRIDGE_TIMEOUT,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise TTSProviderError("VieNeu-TTS quá thời gian tổng hợp trong môi trường Python 3.12") from exc
                if proc.returncode != 0 or not wav_path.is_file():
                    detail = (proc.stderr or proc.stdout or "bridge không tạo WAV").strip()[-500:]
                    raise TTSProviderError(f"VieNeu-TTS (Python 3.12) thất bại: {detail}")
            mp3_path = root / "chunk.mp3"
            try:
                subprocess.run(
                    [ffmpeg_bin, "-nostdin", "-y", "-v", "error", "-i", str(wav_path),
                     "-ac", "1", "-ar", str(sample_rate), str(mp3_path)],
                    check=True, capture_output=True, text=True, timeout=120,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise TTSProviderError(f"Không thể chuyển audio VieNeu sang MP3: {exc}") from exc
            data = mp3_path.read_bytes()
        if not data:
            raise TTSProviderError("VieNeu-TTS không tạo được audio")
        return data

    return produce


def build_synthesize(
    provider: str,
    *,
    api_key: str | None = None,
    ffprobe_bin: str = "ffprobe",
    ffmpeg_bin: str = "ffmpeg",
    fetch_audio: Callable[[str, str], bytes] | None = None,
    synthesize_audio: Callable[[str, str], bytes] | None = None,
    probe_duration: Callable[[Path], float] | None = None,
    http_request: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
    timeout: float = 60.0,
    speed: float = 1.0,
) -> Callable:
    """Build a ``synthesize(text, voice, part, communicate_factory)`` for a non-edge provider.

    Matches the callable :func:`chunked_tts.synthesize_chunked` expects, for the network
    providers (FPT.AI / ElevenLabs) and the local ``vieneu`` engine alike - each yields
    audio bytes that are written to the MP3 part, probed for duration, and turned into
    estimated word boundaries. Injectable seams keep tests offline: ``fetch_audio``
    (network audio), ``synthesize_audio`` (local vieneu audio) and ``probe_duration``.
    ``communicate_factory`` is ignored (edge-only).
    """
    provider = (provider or "").lower()
    if provider not in NETWORK_PROVIDERS and provider not in LOCAL_PROVIDERS:
        supported = ", ".join(NETWORK_PROVIDERS + LOCAL_PROVIDERS)
        raise TTSProviderError(f"build_synthesize chỉ hỗ trợ {supported}")
    if provider in LOCAL_PROVIDERS:
        # Local, network-free engine: no API key; the model call is injectable.
        producer = synthesize_audio or _vieneu_synthesizer(
            ffmpeg_bin=ffmpeg_bin,
            mode=os.environ.get(VIENEU_MODE_ENV, "").strip() or None,
            precision=os.environ.get(VIENEU_PRECISION_ENV, "").strip() or None,
        )
    elif fetch_audio is not None:
        producer = fetch_audio
    else:
        if not api_key:
            raise TTSProviderError(missing_key_message(provider))
        request = http_request or _http_request
        if provider == "fptai":
            producer = _fptai_fetcher(api_key, request=request, sleep=sleep, timeout=timeout, speed=speed)
        else:
            producer = _elevenlabs_fetcher(api_key, request=request, sleep=sleep, timeout=timeout, speed=speed)
    probe = probe_duration or (lambda path: probe_mp3_duration(path, ffprobe_bin=ffprobe_bin))

    def synthesize(text: str, voice: str, part: Path, communicate_factory: object = None) -> list[dict]:
        audio = producer(text, voice)
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
    if provider == "vieneu":
        if vieneu_available():
            return {"provider": provider, "ok": True,
                    "detail": "VieNeu-TTS cục bộ đã sẵn sàng (miễn phí, offline 24kHz, không cần API key)."}
        return {"provider": provider, "ok": False, "detail": VIENEU_INSTALL_HINT}
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
