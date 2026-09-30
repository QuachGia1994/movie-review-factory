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
from pathlib import Path
from typing import Callable

# 1080p mp4 video + m4a audio, merged to mp4; graceful fallbacks for sites that
# do not expose separate streams.
DEFAULT_FORMAT = (
    "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
    "best[height<=1080][ext=mp4]/best"
)
DEFAULT_SUB_LANGS = "vi,en"
_VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}

# Bounded timeouts stop a stalled connection or a never-ending livestream from
# hanging the yt-dlp child (and the request thread) forever. Metadata is a quick
# --skip-download probe; the download cap is generous and overridable via
# MRF_DOWNLOAD_TIMEOUT (set it to 0 to disable the cap entirely).
METADATA_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 1800.0
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


def _download_timeout_seconds() -> float | None:
    """Resolve the download timeout cap; ``MRF_DOWNLOAD_TIMEOUT=0`` disables it."""
    raw = os.environ.get("MRF_DOWNLOAD_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
    return value if value > 0 else None


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


def _ensure_ytdlp() -> str | None:
    found = _find_ytdlp()
    if found:
        return found
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "yt-dlp"],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        return _find_ytdlp()
    except Exception:
        return None


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
) -> list[str]:
    """`yt-dlp` command that fetches <=1080p mp4 and pulls matching subtitles."""
    return [
        ytdlp,
        "--no-playlist",
        "--socket-timeout", str(SOCKET_TIMEOUT_SECONDS),
        "-f", DEFAULT_FORMAT,
        "--merge-output-format", "mp4",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", sub_langs,
        "--convert-subs", "srt",
        "-o", output_template,
        "--", url,
    ]


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
        raise RuntimeError("yt-dlp is not installed - install it to read a link's metadata")
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
        raise RuntimeError("yt-dlp is not installed - install it to download from a link")
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    runner = runner or subprocess.run
    template = str(dest_dir / "source.%(ext)s")
    command = build_download_command(binary, url, template, sub_langs=sub_langs)
    timeout = _download_timeout_seconds()
    try:
        result = runner(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"yt-dlp download timed out after {timeout:.0f}s "
            "(raise or disable via MRF_DOWNLOAD_TIMEOUT)"
        ) from exc
    if getattr(result, "returncode", 1) != 0:
        raise RuntimeError("yt-dlp download failed: " + (getattr(result, "stderr", "") or "")[-1000:])
    source = _find_source_video(dest_dir)
    if source is None:
        raise RuntimeError("yt-dlp reported success but no source video was produced")
    subtitles = sorted(str(path) for path in dest_dir.glob("source*.srt"))
    return {"source_video": str(source), "subtitles": subtitles, "url": url}
