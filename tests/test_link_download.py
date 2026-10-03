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
    assert cmd[cmd.index("--socket-timeout") + 1] == "30"
    assert cmd[cmd.index("--sub-langs") + 1] == "vi"
    assert cmd[cmd.index("-o") + 1] == "/out/source.%(ext)s"
    # No cookie / proxy / IP-evasion flags by design.
    assert not any(flag in cmd for flag in ("--cookies", "--proxy", "--source-address"))


def test_build_metadata_command_skips_download() -> None:
    cmd = link_download.build_metadata_command("/yt-dlp", "https://x/v")
    assert "--skip-download" in cmd and "--dump-single-json" in cmd


def test_download_video_returns_source_and_subtitles(tmp_path: Path) -> None:
    def fake_runner(command: list[str], **kwargs: object) -> object:
        if "-o" not in command:
            return type("R", (), {"returncode": 0, "stdout": '{"format":{"duration":"1.0"}}', "stderr": ""})()
        staging = Path(command[command.index("-o") + 1]).parent
        (staging / "source.mp4").write_bytes(b"fake-mp4")
        (staging / "source.vi.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chao\n", encoding="utf-8")
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


_SUB_429 = (
    "WARNING: [youtube] No supported JavaScript runtime could be found.\n"
    "ERROR: Unable to download video subtitles for 'vi': HTTP Error 429: Too Many Requests\n"
)


def test_download_retries_without_subtitles_when_subtitles_fail(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> object:
        if "-o" not in command:
            return type("R", (), {"returncode": 0, "stdout": '{"format":{"duration":"1.0"}}', "stderr": ""})()
        calls.append(command)
        if "--write-subs" in command:
            return type("R", (), {"returncode": 1, "stderr": _SUB_429})()
        (Path(command[command.index("-o") + 1]).parent / "source.mp4").write_bytes(b"fake-mp4")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    result = link_download.download_video(
        "https://x/v", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=runner,
    )
    assert len(calls) == 2 and "--write-subs" not in calls[1]
    assert result["source_video"] == str(tmp_path / "source.mp4")
    assert result["subtitles"] == []
    assert "429" in result["subtitle_warning"]


def test_download_failure_reports_only_error_lines(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        return type("R", (), {"returncode": 1, "stderr": "WARNING: noise\nERROR: [youtube] abc: Video unavailable\n"})()

    with pytest.raises(RuntimeError) as excinfo:
        link_download.download_video(
            "https://x/v", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=runner,
        )
    assert str(excinfo.value) == "yt-dlp download failed: ERROR: [youtube] abc: Video unavailable"
    assert len(calls) == 1


# Real yt-dlp stderr from a Bilibili HEVC stream the CDN cut short (\r became \n in the pipe).
_BILI_CUT = (
    "WARNING: [BiliBili] Subtitles are only available when logged in.\n"
    "ERROR: \n[download] Got error: 161 bytes read, 250765220 more expected. Giving up after 10 retries\n"
)


def test_download_error_keeps_the_message_after_a_carriage_return() -> None:
    assert link_download._ytdlp_error(_BILI_CUT) == (
        "ERROR: [download] Got error: 161 bytes read, 250765220 more expected. Giving up after 10 retries"
    )
    assert link_download._ytdlp_error("ERROR: \r[download] Got error: boom\n") == "ERROR: [download] Got error: boom"


def test_default_format_prefers_h264_and_cut_stream_falls_back_once(tmp_path: Path) -> None:
    formats: list[str] = []

    def runner(command: list[str], **kwargs: object) -> object:
        if "-o" not in command:
            return type("R", (), {"returncode": 0, "stdout": '{"format":{"duration":"1.0"}}', "stderr": ""})()
        formats.append(command[command.index("-f") + 1])
        staging = Path(command[command.index("-o") + 1]).parent
        if len(formats) == 1:
            (staging / "source.f30077.mp4.part").write_bytes(b"x" * 10)
            return type("R", (), {"returncode": 1, "stderr": _BILI_CUT})()
        assert not list(staging.glob("*.part"))
        (staging / "source.mp4").write_bytes(b"fake-mp4")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    result = link_download.download_video(
        "https://x/v", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=runner,
    )
    assert formats[0].startswith("bestvideo[height<=1080][vcodec^=avc1]")
    assert formats == [link_download.DEFAULT_FORMAT, link_download.FALLBACK_FORMAT]
    assert "161 bytes read" in result["stream_warning"]
    assert Path(result["source_video"]).read_bytes() == b"fake-mp4"


def test_cut_stream_on_the_fallback_still_fails(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        return type("R", (), {"returncode": 1, "stderr": _BILI_CUT})()

    with pytest.raises(RuntimeError, match="161 bytes read"):
        link_download.download_video(
            "https://x/v", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=runner,
        )
    assert len(calls) == 2


def test_bilibili_download_moves_akamai_streams_to_bilibili_mirror(tmp_path: Path) -> None:
    akamai = "https://upos-hz-mirrorakam.akamaized.net/upgcxcode/32/46/1-1-30080.m4s?e=sig&os=akam"
    info = {"id": "BV1", "formats": [{"format_id": "30080", "url": akamai},
                                       {"format_id": "x", "url": "https://other.example/x"}],
            "requested_formats": [{"format_id": "30080", "url": akamai}],
            "requested_downloads": [{"requested_formats": [{"format_id": "30080", "url": akamai}]}]}
    downloads: list[list[str]] = []

    def runner(command: list[str], **kwargs: object) -> object:
        if "--dump-single-json" in command:
            assert command[-1] == "https://www.bilibili.com/video/BV1"
            return type("R", (), {"returncode": 0, "stdout": json.dumps(info), "stderr": ""})()
        if "-o" not in command:
            return type("R", (), {"returncode": 0, "stdout": '{"format":{"duration":"1.0"}}', "stderr": ""})()
        downloads.append(command)
        saved = json.loads(Path(command[command.index("--load-info-json") + 1]).read_text(encoding="utf-8"))
        assert saved["formats"][0]["url"].startswith("https://upos-sz-mirrorcos.bilivideo.com/upgcxcode/32/46/")
        assert saved["formats"][0]["url"].endswith("?e=sig&os=akam")
        assert saved["formats"][1]["url"] == "https://other.example/x"
        assert saved["requested_formats"][0]["url"] == saved["formats"][0]["url"]
        assert saved["requested_downloads"][0]["requested_formats"][0]["url"] == saved["formats"][0]["url"]
        (Path(command[command.index("-o") + 1]).parent / "source.mp4").write_bytes(b"fake-mp4")
        return type("R", (), {"returncode": 0, "stderr": ""})()

    result = link_download.download_video(
        "https://www.bilibili.com/video/BV1", tmp_path, confirm_rights=True, ytdlp="/yt-dlp", runner=runner,
    )
    assert len(downloads) == 1 and "--" not in downloads[0]
    assert Path(result["source_video"]).read_bytes() == b"fake-mp4"
    assert not (tmp_path / "info.json").exists()


def test_non_bilibili_and_unswappable_links_download_by_url() -> None:
    assert link_download._is_bilibili("https://m.bilibili.com/video/BV1")
    assert link_download._is_bilibili("https://b23.tv/abc")
    assert not link_download._is_bilibili("https://notbilibili.com/video")
    info = {"formats": [{"url": "https://upos-sz-mirrorcos.bilivideo.com/a.m4s"}]}
    assert link_download._swap_bilibili_mirrors(info) == 0
    command = link_download.build_download_command("/yt-dlp", "https://x/v", "/o/s.%(ext)s", info_json="/o/i.json")
    assert command[-2:] == ["--load-info-json", "/o/i.json"] and "https://x/v" not in command


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


def test_ensure_ytdlp_falls_back_to_uv_when_pip_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    found: list[str | None] = [None]
    calls: list[list[str]] = []
    monkeypatch.setattr(link_download, "_find_ytdlp", lambda explicit=None: found[0])
    monkeypatch.setattr(link_download, "find_spec", lambda name: None)
    monkeypatch.setenv("MRF_UV", "C:/mrf/uv.exe")

    def fake_run(cmd: list[str], **kwargs: object) -> object:
        calls.append(cmd)
        found[0] = "C:/venv/Scripts/yt-dlp.exe"
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr(link_download.subprocess, "run", fake_run)
    assert link_download._ensure_ytdlp() == "C:/venv/Scripts/yt-dlp.exe"
    assert calls == [["C:/mrf/uv.exe", "pip", "install", "--python", link_download.sys.executable, link_download.YTDLP_REQUIREMENT]]


def test_missing_ytdlp_error_explains_the_failed_auto_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(link_download, "_find_ytdlp", lambda explicit=None: None)
    monkeypatch.setattr(link_download, "find_spec", lambda name: None)
    monkeypatch.delenv("MRF_UV", raising=False)
    monkeypatch.setattr(link_download.shutil, "which", lambda cmd: None)
    monkeypatch.setattr(link_download, "_last_install_error", "")
    with pytest.raises(RuntimeError, match=r"yt-dlp is not installed.*no pip or uv available"):
        link_download.download_video("https://example.com/v", tmp_path, confirm_rights=True)

