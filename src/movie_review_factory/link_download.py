"""Download-from-link ingest via yt-dlp, gated on an explicit rights confirmation.

Use this only for footage you are authorized to download - your own uploads,
public-domain / Creative-Commons works, or material you hold a licence for. Per
the project guardrail (see PLAN.md / CLAUDE.md) the caller MUST confirm usage
rights before anything is fetched, and this module deliberately ships NO
IP-block / cookie / proxy evasion tooling.

The command builders are pure so they are fully unit-testable; the network calls
go through an injectable ``runner`` (defaults to ``subprocess.run``).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from importlib.util import find_spec
from pathlib import Path
from typing import Callable
from urllib.parse import urlsplit, urlunsplit

# H.264 first: Bilibili ranks HEVC/AV1 higher, but H.264 decodes faster and more reliably in render.
DEFAULT_FORMAT = (
    "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height<=1080][ext=mp4]/best"
)
# Second try after a broken stream: another rendition (often another CDN file).
FALLBACK_FORMAT = (
    "bestvideo[height<=720][vcodec^=avc1]+bestaudio[ext=m4a]/"
    "bestvideo[height<=720]+bestaudio/best[height<=720]/best"
)
# yt-dlp errors where the stream broke mid-transfer, not where the video is unavailable.
_STREAM_FAILURES = ("Got error:", "more expected", "timed out", "Connection reset", "HTTP Error 5")

# Bilibili's overseas Akamai mirror drops after ~2 MB; its own upos mirrors serve the same signed path.
_BILIBILI_HOSTS = ("bilibili.com", "b23.tv")
_AKAMAI_MIRROR_RE = re.compile(r"^upos-[a-z0-9]+-mirrorakam\.akamaized\.net$", re.IGNORECASE)
BILIBILI_MIRROR_HOST = "upos-sz-mirrorcos.bilivideo.com"


def _is_bilibili(url: str) -> bool:
    host = (urlsplit(url).hostname or "").lower()
    return any(host == name or host.endswith("." + name) for name in _BILIBILI_HOSTS)


def _swap_bilibili_mirrors(info: dict) -> int:
    """Point Akamai-mirrored stream URLs at Bilibili's own mirror; returns how many changed."""
    changed = 0
    for key in ("formats", "requested_formats", "requested_downloads"):
        for item in info.get(key) or []:
            if not isinstance(item, dict):
                continue
            if key == "requested_downloads":
                changed += _swap_bilibili_mirrors(item)
            if not isinstance(item.get("url"), str):
                continue
            parts = urlsplit(item["url"])
            if parts.scheme in {"http", "https"} and _AKAMAI_MIRROR_RE.match(parts.netloc):
                item["url"] = urlunsplit(parts._replace(netloc=BILIBILI_MIRROR_HOST))
                changed += 1
    return changed


DEFAULT_SUB_LANGS = "vi,en"
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

# Bounded timeouts stop a stalled connection or endless livestream from hanging yt-dlp; config may lower, never disable or raise, the ceiling.
METADATA_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 1800.0
HARD_DOWNLOAD_TIMEOUT_SECONDS = 3600.0
HARD_MAX_FILESIZE_BYTES = 20 * 1024 * 1024 * 1024
HARD_MAX_DURATION_SECONDS = 6 * 60 * 60
PROBE_TIMEOUT_SECONDS = 30.0
# yt-dlp per-socket read timeout: abort a stalled chunk fast so a mid-transfer
# TCP stall is caught in seconds rather than waiting out the whole-process cap.
SOCKET_TIMEOUT_SECONDS = 30

# Only real web links are ever handed to yt-dlp. A value starting with "-" could
# otherwise be parsed by yt-dlp as an OPTION instead of a URL - e.g. --exec,
# which runs arbitrary commands - so anything that is not http(s) is rejected up
# front. The command builders additionally place a "--" guard before the URL.
_URL_SCHEME_RE = re.compile(r"^https?://", re.IGNORECASE)


def _require_web_url(url: str) -> str:
    """Return a trimmed http(s) URL or raise ``ValueError`` for anything else."""
    cleaned = str(url or "").strip()
    if not _URL_SCHEME_RE.match(cleaned):
        raise ValueError("only http:// or https:// links are supported")
    return cleaned


def _download_timeout_seconds() -> float:
    """Resolve a positive timeout capped by the non-disableable hard ceiling."""
    raw = os.environ.get("MRF_DOWNLOAD_TIMEOUT", "").strip()
    try:
        value = float(raw) if raw else DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    except ValueError:
        value = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    if value <= 0:
        value = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    return min(value, HARD_DOWNLOAD_TIMEOUT_SECONDS)


def _configured_limit(name: str, hard_limit: int) -> int:
    try:
        configured = int(os.environ.get(name, "") or hard_limit)
    except ValueError:
        configured = hard_limit
    return min(max(1, configured), hard_limit)


def _validate_video(path: Path, *, runner: Callable[..., object]) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("downloaded source video is empty")
    if path.stat().st_size > _configured_limit("MRF_DOWNLOAD_MAX_FILESIZE", HARD_MAX_FILESIZE_BYTES):
        raise RuntimeError("downloaded source video exceeds the maximum file size")
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:  # yt-dlp may run without ffprobe; retain basic non-empty validation.
        return
    result = runner(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, timeout=PROBE_TIMEOUT_SECONDS,
    )
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError("downloaded source video failed ffprobe validation")
    try:
        duration = float(json.loads(getattr(result, "stdout", "") or "{}")["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("downloaded source video has no valid duration") from exc
    if duration <= 0 or duration > _configured_limit("MRF_DOWNLOAD_MAX_DURATION", HARD_MAX_DURATION_SECONDS):
        raise RuntimeError("downloaded source video duration is outside allowed limits")


class RightsConfirmationRequired(RuntimeError):
    """Raised when a download is attempted without confirming usage rights."""


def _find_ytdlp(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    found = shutil.which("yt-dlp") or shutil.which("yt_dlp")
    if found:
        return found
    exe_name = "yt-dlp.exe" if os.name == "nt" else "yt-dlp"
    parent = Path(sys.executable).parent
    candidates = [
        parent / exe_name,
        parent / "Scripts" / exe_name,
        parent / "bin" / exe_name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


YTDLP_REQUIREMENT = "yt-dlp>=2025.1"
_install_lock = threading.Lock()
_last_install_error = ""


def _install_commands() -> list[list[str]]:
    """pip first, then uv: the launcher's uv-made venv ships without pip."""
    commands = []
    if find_spec("pip") is not None:
        commands.append([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", YTDLP_REQUIREMENT])
    uv = os.environ.get("MRF_UV") or shutil.which("uv")
    if uv:
        commands.append([uv, "pip", "install", "--python", sys.executable, YTDLP_REQUIREMENT])
    return commands


def _ensure_ytdlp() -> str | None:
    """Install yt-dlp into the running interpreter on first use; None plus a recorded reason on failure."""
    global _last_install_error
    with _install_lock:
        found = _find_ytdlp()
        if found:
            return found
        commands = _install_commands()
        errors = [] if commands else ["no pip or uv available"]
        for command in commands:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=300)
            except (OSError, subprocess.TimeoutExpired) as exc:
                errors.append(f"{Path(command[0]).name}: {exc}")
                continue
            if getattr(result, "returncode", 1) == 0:
                found = _find_ytdlp()
                if found:
                    _last_install_error = ""
                    return found
                errors.append(f"{Path(command[0]).name}: installed but yt-dlp executable not found")
            else:
                tail = (getattr(result, "stderr", "") or getattr(result, "stdout", "") or "").strip()[-300:]
                errors.append(f"{Path(command[0]).name}: {tail or 'install failed'}")
        _last_install_error = "; ".join(errors)
        return None


def _missing_ytdlp_message(action: str) -> str:
    reason = f" (auto-install failed: {_last_install_error})" if _last_install_error else ""
    return f"yt-dlp is not installed - install it to {action}{reason}"


def _ytdlp_binary(explicit: str | None = None, *, auto_install: bool = True) -> str | None:
    if explicit:
        return explicit
    found = _find_ytdlp()
    if found:
        return found
    if auto_install:
        return _ensure_ytdlp()
    return None


def rights_confirmed(explicit: bool | None = None) -> bool:
    """Whether the caller has confirmed they may download this content.

    An explicit boolean wins; otherwise ``MRF_DOWNLOAD_RIGHTS_CONFIRMED`` is read.
    """
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("MRF_DOWNLOAD_RIGHTS_CONFIRMED", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


def build_metadata_command(ytdlp: str, url: str) -> list[str]:
    """`yt-dlp` command that dumps title/duration/subtitle info as JSON only."""
    return [ytdlp, "--no-playlist", "--skip-download", "--dump-single-json", "--", url]


def build_download_command(
    ytdlp: str,
    url: str,
    output_template: str,
    *,
    sub_langs: str = DEFAULT_SUB_LANGS,
    include_subs: bool = True,
    video_format: str = DEFAULT_FORMAT,
    info_json: str | None = None,
) -> list[str]:
    """`yt-dlp` command that fetches <=1080p mp4 and, optionally, matching subtitles.

    ``info_json`` downloads from a pre-extracted (mirror-adjusted) info file instead of
    re-extracting ``url``.
    """
    source = ["--load-info-json", info_json] if info_json else ["--", url]
    subs = ["--write-subs", "--write-auto-subs", "--sub-langs", sub_langs, "--convert-subs", "srt"]
    return [
        ytdlp,
        "--no-playlist",
        "--socket-timeout", str(SOCKET_TIMEOUT_SECONDS),
        "-f", video_format,
        "--merge-output-format", "mp4",
        *(subs if include_subs else []),
        "-o", output_template,
        *source,
    ]


# yt-dlp aborts the whole download when a subtitle fetch fails (YouTube often answers 429).
_SUBTITLE_FAILURE = "Unable to download video subtitles"


def _ytdlp_error(stderr: str) -> str:
    """The ERROR lines of yt-dlp stderr, without the WARNING noise; tail as fallback.

    yt-dlp writes download errors as ``ERROR: \\r[download] Got error: ...``; text-mode
    pipes turn that carriage return into a line break, so an empty ``ERROR:`` line takes
    the next non-empty line as its message.
    """
    lines = [" ".join(line.split()) for line in (stderr or "").splitlines()]
    errors: list[str] = []
    for number, line in enumerate(lines):
        if not line.startswith("ERROR:"):
            continue
        if line == "ERROR:":
            following = next((text for text in lines[number + 1:] if text), "")
            if not following or following.startswith(("ERROR:", "WARNING:")):
                continue
            line = f"ERROR: {following}"
        errors.append(line)
    return (" ".join(errors) or (stderr or "").strip())[-1000:]


def fetch_metadata(
    url: str,
    *,
    ytdlp: str | None = None,
    runner: Callable[..., object] | None = None,
) -> dict:
    """Return {title, duration, thumbnail, subtitle_languages, webpage_url}."""
    url = _require_web_url(url)
    binary = _ytdlp_binary(ytdlp)
    if not binary:
        raise RuntimeError(_missing_ytdlp_message("read a link's metadata"))
    runner = runner or subprocess.run
    try:
        result = runner(
            build_metadata_command(binary, url),
            capture_output=True, text=True, timeout=METADATA_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"yt-dlp metadata fetch timed out after {METADATA_TIMEOUT_SECONDS:.0f}s"
        ) from exc
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError("yt-dlp metadata fetch failed: " + (getattr(result, "stderr", "") or "")[-1000:])
    data = json.loads(getattr(result, "stdout", "") or "{}")
    subtitles = sorted((data.get("subtitles") or {}).keys())
    return {
        "title": data.get("title"),
        "duration": data.get("duration"),
        "thumbnail": data.get("thumbnail"),
        "subtitle_languages": subtitles,
        "webpage_url": data.get("webpage_url") or url,
    }


def _find_source_video(dest_dir: Path) -> Path | None:
    mp4 = dest_dir / "source.mp4"
    if mp4.is_file():
        return mp4
    for candidate in sorted(dest_dir.glob("source.*")):
        if candidate.suffix.lower() in _VIDEO_SUFFIXES:
            return candidate
    return None


def _bilibili_info_json(binary: str, url: str, staging: Path, runner: Callable[..., object]) -> str | None:
    """Extract a Bilibili video once and save its info with stream URLs on Bilibili's mirror.

    None (plain URL download) when extraction fails or no Akamai URL needs swapping.
    """
    try:
        result = runner(build_metadata_command(binary, url), capture_output=True, text=True,
                        timeout=METADATA_TIMEOUT_SECONDS)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if getattr(result, "returncode", 1) != 0:
        return None
    try:
        info = json.loads(getattr(result, "stdout", "") or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(info, dict) or not _swap_bilibili_mirrors(info):
        return None
    path = staging / "info.json"
    path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return str(path)


def download_video(
    url: str,
    dest_dir: Path,
    *,
    confirm_rights: bool | None = None,
    ytdlp: str | None = None,
    sub_langs: str = DEFAULT_SUB_LANGS,
    runner: Callable[..., object] | None = None,
) -> dict:
    """Download ``url`` into ``dest_dir`` as ``source.mp4`` (+ any subtitles).

    Raises :class:`RightsConfirmationRequired` unless the caller confirms rights
    (``confirm_rights=True`` or ``MRF_DOWNLOAD_RIGHTS_CONFIRMED=1``). Returns
    {source_video, subtitles, url}; pulled subtitles can feed the transcript
    stage's embedded-subtitle bypass instead of running Whisper.
    """
    url = _require_web_url(url)
    if not rights_confirmed(confirm_rights):
        raise RightsConfirmationRequired(
            "Confirm you are authorized to download this video before fetching it "
            "(pass confirm_rights=True or set MRF_DOWNLOAD_RIGHTS_CONFIRMED=1). "
            "This tool does not bypass access controls or copyright."
        )
    binary = _ytdlp_binary(ytdlp)
    if not binary:
        raise RuntimeError(_missing_ytdlp_message("download from a link"))
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    runner = runner or subprocess.run
    timeout = _download_timeout_seconds()
    try:
        with tempfile.TemporaryDirectory(prefix=".download-", dir=dest_dir) as staging_name:
            staging = Path(staging_name)
            template = str(staging / "source.%(ext)s")
            info_json = _bilibili_info_json(binary, url, staging, runner) if _is_bilibili(url) else None
            subtitle_warning = None
            stream_warning = None
            include_subs = True
            video_format = DEFAULT_FORMAT
            while True:
                command = build_download_command(
                    binary, url, template, sub_langs=sub_langs, include_subs=include_subs,
                    video_format=video_format, info_json=info_json,
                )
                try:
                    result = runner(command, capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"yt-dlp download timed out after {timeout:.0f}s") from exc
                if getattr(result, "returncode", 1) == 0:
                    break
                error = _ytdlp_error(getattr(result, "stderr", "") or "")
                if include_subs and _SUBTITLE_FAILURE in error:
                    # Subtitles only save a Whisper pass; retry the video without them.
                    subtitle_warning = error
                    include_subs = False
                    continue
                if video_format == DEFAULT_FORMAT and any(mark in error for mark in _STREAM_FAILURES):
                    # Rendition broke mid-transfer: retry once at a lower rendition.
                    for partial in staging.glob("source.*"):
                        if partial.is_file():
                            partial.unlink()
                    stream_warning = error
                    video_format = FALLBACK_FORMAT
                    continue
                raise RuntimeError("yt-dlp download failed: " + error)
            source = _find_source_video(staging)
            if source is None:
                raise RuntimeError("yt-dlp reported success but no source video was produced")
            _validate_video(source, runner=runner)
            promoted_source = dest_dir / "source.mp4"
            os.replace(source, promoted_source)
            promoted_subtitles: list[str] = []
            for subtitle in sorted(staging.glob("source*.srt")):
                target = dest_dir / subtitle.name
                os.replace(subtitle, target)
                promoted_subtitles.append(str(target))
            return {
                "source_video": str(promoted_source), "subtitles": promoted_subtitles, "url": url,
                "subtitle_warning": subtitle_warning, "stream_warning": stream_warning,
            }
    except OSError as exc:
        raise RuntimeError(f"could not stage/promote downloaded video: {exc}") from exc
