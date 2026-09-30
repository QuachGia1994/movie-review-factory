"""Unit tests for the yt-dlp download-from-link engine (rights-gated)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from movie_review_factory import link_download


def test_rights_confirmed_reads_flag_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MRF_DOWNLOAD_RIGHTS_CONFIRMED", raising=False)
    assert link_download.rights_confirmed() is False
    assert link_download.rights_confirmed(True) is True
    monkeypatch.setenv("MRF_DOWNLOAD_RIGHTS_CONFIRMED", "1")
    assert link_download.rights_confirmed() is True
    assert link_download.rights_confirmed(False) is False  # explicit wins


def test_download_requires_rights_confirmation(tmp_path: Path) -> None:
    with pytest.raises(link_download.RightsConfirmationRequired):
        link_download.download_video("https://example.com/v", tmp_path, ytdlp="/yt-dlp")


def test_build_download_command_shape() -> None:
    cmd = link_download.build_download_command("/yt-dlp", "https://x/v", "/out/source.%(ext)s", sub_langs="vi")
    assert cmd[0] == "/yt-dlp"
    assert cmd[-1] == "https://x/v"
    assert "--merge-output-format" in cmd and "mp4" in cmd
    assert "--write-subs" in cmd
    assert cmd[cmd.index("--sub-langs") + 1] == "vi"
    assert cmd[cmd.index("-o") + 1] == "/out/source.%(ext)s"
    # No cookie / proxy / IP-evasion flags by design.
    assert not any(flag in cmd for flag in ("--cookies", "--proxy", "--source-address"))


def test_build_metadata_command_skips_download() -> None:
    cmd = link_download.build_metadata_command("/yt-dlp", "https://x/v")
    assert "--skip-download" in cmd and "--dump-single-json" in cmd


def test_download_video_returns_source_and_subtitles(tmp_path: Path) -> None:
    def fake_runner(command: list[str], **kwargs: object) -> object:
        (tmp_path / "source.mp4").write_bytes(b"fake-mp4")
        (tmp_path / "source.vi.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    result = link_download.download_video(
        "https://x/v", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=fake_runner,
    )
    assert result["source_video"] == str(tmp_path / "source.mp4")
    assert result["subtitles"] == [str(tmp_path / "source.vi.srt")]


def test_download_video_raises_when_binary_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(link_download, "_ytdlp_binary", lambda explicit=None: None)
    with pytest.raises(RuntimeError, match="yt-dlp is not installed"):
        link_download.download_video("https://x/v", tmp_path, confirm_rights=True)


def test_download_video_raises_when_no_output_produced(tmp_path: Path) -> None:
    def empty_runner(command: list[str], **kwargs: object) -> object:
        return type("R", (), {"returncode": 0, "stderr": ""})()

    with pytest.raises(RuntimeError, match="no source video"):
        link_download.download_video(
            "https://x/v", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=empty_runner,
        )


def test_fetch_metadata_parses_json(tmp_path: Path) -> None:
    payload = {
        "title": "My Own Clip",
        "duration": 123.4,
        "thumbnail": "https://x/t.jpg",
        "subtitles": {"vi": [{}], "en": [{}]},
        "webpage_url": "https://x/v",
    }

    def fake_runner(command: list[str], **kwargs: object) -> object:
        return type("R", (), {"returncode": 0, "stdout": json.dumps(payload), "stderr": ""})()

    meta = link_download.fetch_metadata("https://x/v", ytdlp="/yt-dlp", runner=fake_runner)
    assert meta["title"] == "My Own Clip"
    assert meta["duration"] == 123.4
    assert meta["subtitle_languages"] == ["en", "vi"]


def test_find_ytdlp_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(link_download.shutil, "which", lambda cmd: None)
    dummy_exe = tmp_path / "yt-dlp.exe"
    dummy_exe.write_text("dummy", encoding="utf-8")
    monkeypatch.setattr(link_download.sys, "executable", str(tmp_path / "python.exe"))
    assert link_download._find_ytdlp() == str(dummy_exe)


def test_ytdlp_binary_auto_installs_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(link_download, "_find_ytdlp", lambda explicit=None: None)
    installed = False

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        nonlocal installed
        installed = True
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(link_download.subprocess, "run", fake_run)
    # _ensure_ytdlp will run fake_run, but _find_ytdlp returns None
    result = link_download._ensure_ytdlp()
    assert installed is True
    assert result is None

