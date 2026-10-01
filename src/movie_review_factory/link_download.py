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
    include_subs: bool = True,
) -> list[str]:
    """`yt-dlp` command that fetches <=1080p mp4 and, optionally, matching subtitles."""
    subs = ["--write-subs", "--write-auto-subs", "--sub-langs", sub_langs, "--convert-subs", "srt"]
    return [
        ytdlp,
        "--no-playlist",
        "--socket-timeout", str(SOCKET_TIMEOUT_SECONDS),
        "-f", DEFAULT_FORMAT,
        "--merge-output-format", "mp4",
        *(subs if include_subs else []),
        "-o", output_template,
        "--", url,
    ]


# yt-dlp aborts the whole download when a subtitle fetch fails (YouTube often answers 429).
_SUBTITLE_FAILURE = "Unable to download video subtitles"


def _ytdlp_error(stderr: str) -> str:
    """The ERROR lines of yt-dlp stderr, without the WARNING noise; tail as fallback."""
    errors = [line.strip() for line in (stderr or "").splitlines() if line.strip().startswith("ERROR:")]
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
    timeout = _download_timeout_seconds()
    try:
        with tempfile.TemporaryDirectory(prefix=".download-", dir=dest_dir) as staging_name:
            staging = Path(staging_name)
            template = str(staging / "source.%(ext)s")
            subtitle_warning = None
            for include_subs in (True, False):
                command = build_download_command(
                    binary, url, template, sub_langs=sub_langs, include_subs=include_subs,
                )
                try:
                    result = runner(command, capture_output=True, text=True, timeout=timeout)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f"yt-dlp download timed out after {timeout:.0f}s") from exc
                if getattr(result, "returncode", 1) == 0:
                    break
                error = _ytdlp_error(getattr(result, "stderr", "") or "")
                if not (include_subs and _SUBTITLE_FAILURE in error):
                    raise RuntimeError("yt-dlp download failed: " + error)
                # Subtitles only save a Whisper pass; retry the video without them.
                subtitle_warning = error
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
                "subtitle_warning": subtitle_warning,
            }
    except OSError as exc:
        raise RuntimeError(f"could not stage/promote downloaded video: {exc}") from exc
