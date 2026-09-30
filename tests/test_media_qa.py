from pathlib import Path
from types import SimpleNamespace

from movie_review_factory import media_qa


LOG = """
[blackdetect @ 0001] black_start:2 black_end:4.1 black_duration:2.1
[freezedetect @ 0002] lavfi.freezedetect.freeze_start: 5.200
[freezedetect @ 0002] lavfi.freezedetect.freeze_end: 8.000 | freeze_duration: 2.800
[silencedetect @ 0003] silence_start: 9.000
[silencedetect @ 0003] silence_end: 12.200 | silence_duration: 3.200
[Parsed_ebur128_4 @ 0004] Integrated loudness:
    I:         -16.3 LUFS
[Parsed_ebur128_4 @ 0004] True peak:
    Peak:       -1.7 dBFS
"""


def _checks(log, duration=None):
    return {check["check"]: check for check in media_qa.parse_signal_log(log, duration)}


def test_signal_intervals_have_output_timestamps_and_are_advisory():
    checks = _checks(LOG)
    assert checks["black_intervals"]["value"]["intervals"] == [
        {"start_seconds": 2.0, "end_seconds": 4.1, "duration_seconds": 2.1}
    ]
    assert checks["freeze_intervals"]["value"]["intervals"][0]["start_seconds"] == 5.2
    assert checks["silence_intervals"]["value"]["intervals"][0]["duration_seconds"] == 3.2
    assert all(checks[key]["passed"] and checks[key]["review_required"]
               for key in ("black_intervals", "freeze_intervals", "silence_intervals"))
    assert checks["decoded_audio_loudness"]["passed"]
    assert checks["decoded_audio_loudness"]["review_required"] is False


def test_open_interval_reaches_duration_and_loudness_outlier_requires_review():
    log = "freeze_start: 7.4\nsilence_start: 8.0\nIntegrated loudness:\n I: -27.2 LUFS\nTrue peak:\n Peak: -0.3 dBFS"
    checks = _checks(log, 10)
    assert checks["freeze_intervals"]["value"]["intervals"][0]["end_seconds"] == 10
    assert checks["silence_intervals"]["value"]["intervals"][0]["duration_seconds"] == 2
    assert checks["decoded_audio_loudness"]["review_required"]


def test_no_measurable_audio_is_a_failure():
    checks = _checks("Integrated loudness:\n I: -inf LUFS\nTrue peak:\n Peak: -inf dBFS")
    assert not checks["decoded_audio_loudness"]["passed"]
    assert not checks["decoded_audio_loudness"]["review_required"]


def test_inspect_rendered_media_decodes_audio_video_in_single_pass(monkeypatch, tmp_path):
    command_seen = []

    def run(command, **kwargs):
        command_seen.extend(command)
        return SimpleNamespace(returncode=0, stderr=LOG)

    monkeypatch.setattr(media_qa.subprocess, "run", run)
    checks = media_qa.inspect_rendered_media(tmp_path / "final.mp4", ffmpeg_bin="ffmpeg",
                                              duration_seconds=20)
    assert checks[0]["check"] == "decoded_media_scan"
    assert checks[0]["passed"]
    assert "blackdetect" in command_seen[command_seen.index("-filter_complex") + 1]
    assert "ebur128" in command_seen[command_seen.index("-filter_complex") + 1]
    assert Path(command_seen[command_seen.index("-i") + 1]).name == "final.mp4"


def test_inspect_rendered_media_fails_on_decode_error(monkeypatch, tmp_path):
    monkeypatch.setattr(media_qa.subprocess, "run",
                        lambda *_args, **_kwargs: SimpleNamespace(returncode=1, stderr="Invalid data"))
    checks = media_qa.inspect_rendered_media(tmp_path / "bad.mp4", ffmpeg_bin="ffmpeg")
    assert len(checks) == 1 and not checks[0]["passed"]
    assert "Invalid data" in checks[0]["message"]


def test_real_ffmpeg_decodes_black_tone_and_loudness(tmp_path):
    import shutil
    import subprocess
    import pytest
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg not installed")
    path = tmp_path / "black-tone.mp4"
    subprocess.run([
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x90:r=10:d=3",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=3",
        "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(path),
    ], capture_output=True, check=True)
    checks = {check["check"]: check for check in media_qa.inspect_rendered_media(
        path, ffmpeg_bin=ffmpeg, duration_seconds=3
    )}
    assert checks["decoded_media_scan"]["passed"]
    assert checks["black_intervals"]["review_required"]
    assert checks["black_intervals"]["value"]["intervals"][0]["start_seconds"] == 0
    assert checks["decoded_audio_loudness"]["passed"]


def test_signals_from_render_log_reuses_detect_log_and_marks_scan_source():
    checks = {c["check"]: c
              for c in media_qa.signals_from_render_log(LOG, duration_seconds=20)}
    # The render pass already decoded the frames, so the scan says so.
    assert checks["decoded_media_scan"]["passed"]
    assert checks["decoded_media_scan"]["value"]["source"] == "render_pass"
    # Same parser as the decode-pass path, so the signal checks match.
    assert checks["black_intervals"]["value"]["intervals"][0]["start_seconds"] == 2.0
    assert checks["decoded_audio_loudness"]["passed"]


def test_signals_from_render_log_returns_none_without_loudness_summary():
    # No ebur128 summary -> the detect pass did not run or the log is unusable,
    # so the caller must fall back to a fresh decode pass, not fail QA.
    assert media_qa.signals_from_render_log(
        "black_start:1 black_end:2 black_duration:1") is None
    assert media_qa.signals_from_render_log("") is None
