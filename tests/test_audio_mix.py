import hashlib
from pathlib import Path

import pytest

from movie_review_factory.audio_mix import build_audio_mix


def test_default_preserves_original_narration() -> None:
    mix = build_audio_mix(None, overlay_count=2, duration_seconds=60)
    assert mix.output_map == "1:a:0"
    assert not mix.filters and not mix.input_args and not mix.provenance


def test_music_and_effect_mix_have_voice_first_and_provenance(tmp_path: Path) -> None:
    music = tmp_path / "licensed music.wav"
    effect = tmp_path / "effect.wav"
    music.write_bytes(b"stub")
    effect.write_bytes(b"stub")
    config = {
        "voice_gain_db": 1,
        "music": {"path": str(music), "rights_note": "licensed for this video"},
        "effects": [{"path": str(effect), "rights_note": "own recording", "at_seconds": 12.5}],
    }
    mix = build_audio_mix(config, overlay_count=3, duration_seconds=30)
    assert mix.input_args[:3] == ("-stream_loop", "-1", "-i")
    assert mix.output_map == "[audio]"
    assert "[5:a:0]" in mix.filters[1]
    assert "[6:a:0]" in mix.filters[2]
    assert "asplit=2" in ";".join(mix.filters)
    assert "[music][duck_key]sidechaincompress" in ";".join(mix.filters)
    assert "[voice_mix][ducked_music][effect0]amix=inputs=3:duration=first" in ";".join(mix.filters)
    assert "alimiter=limit=0.95:level=false" in mix.filters[-1]
    assert "adelay=12500:all=1" in ";".join(mix.filters)
    assert mix.provenance[0]["rights_note"] == "licensed for this video"
    assert mix.provenance[0]["sha256"] == hashlib.sha256(b"stub").hexdigest()
    assert mix.provenance[1]["at_seconds"] == 12.5
    assert "[0:a" not in ";".join(mix.filters)


def test_provenance_hash_changes_when_audio_file_changes(tmp_path: Path) -> None:
    music = tmp_path / "music.wav"
    music.write_bytes(b"first take")
    config = {"music": {"path": str(music), "rights_note": "self recorded"}}
    first = build_audio_mix(config, overlay_count=0, duration_seconds=15)
    music.write_bytes(b"second take")
    second = build_audio_mix(config, overlay_count=0, duration_seconds=15)
    assert first.provenance[0]["path"] == second.provenance[0]["path"]
    assert first.provenance[0]["sha256"] != second.provenance[0]["sha256"]


def test_voice_gain_only_builds_single_track_filter() -> None:
    mix = build_audio_mix({"voice_gain_db": -3}, overlay_count=0, duration_seconds=10)
    assert mix.input_args == ()
    assert mix.output_map == "[audio]"
    assert "volume=-3dB" in mix.filters[0]
    assert len(mix.filters) == 2


@pytest.mark.parametrize("spec", [
    {"music": {"path": "missing.mp3", "rights_note": "licensed"}},
    {"music": {"path": "missing.mp3", "rights_note": ""}},
    {"effects": [{"path": "missing.mp3", "rights_note": "own recording", "at_seconds": -1}]},
    {"voice_gain_db": float("nan")},
    {"effects": "sound.mp3"},
])
def test_invalid_audio_config_is_rejected(spec: dict) -> None:
    with pytest.raises(ValueError):
        build_audio_mix(spec, overlay_count=0, duration_seconds=10)


def test_effect_outside_narration_is_rejected(tmp_path: Path) -> None:
    effect = tmp_path / "effect.wav"
    effect.write_bytes(b"stub")
    with pytest.raises(ValueError, match="outside narration"):
        build_audio_mix(
            {"effects": [{"path": str(effect), "rights_note": "own recording", "at_seconds": 10}]},
            overlay_count=0, duration_seconds=10,
        )


def test_ffmpeg_renders_opt_in_music_and_effect(tmp_path: Path) -> None:
    import math
    import shutil
    import struct
    import subprocess
    import wave

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg binaries unavailable")

    def tone(path: Path, frequency: float, seconds: float) -> None:
        samples = int(seconds * 48000)
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(48000)
            stream.writeframes(b"".join(
                struct.pack("<h", int(5000 * math.sin(i * frequency * math.tau / 48000)))
                for i in range(samples)
            ))

    voice, music, effect = (tmp_path / name for name in ("voice.wav", "music.wav", "effect.wav"))
    tone(voice, 200, 1.25)
    tone(music, 300, 1.25)
    tone(effect, 400, 0.15)
    mix = build_audio_mix({
        "music": {"path": str(music), "rights_note": "generated test wave"},
        "effects": [{"path": str(effect), "rights_note": "generated test wave", "at_seconds": 0.5}],
    }, overlay_count=0, duration_seconds=1.25)
    output = tmp_path / "mixed.mp4"
    command = [
        ffmpeg, "-v", "error", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x90:r=25:d=1.25",
        "-i", str(voice), *mix.input_args,
        "-filter_complex", ";".join(mix.filters),
        "-map", "0:v:0", "-map", mix.output_map, "-t", "1.25",
        "-c:v", "mpeg4", "-c:a", "aac", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr[-1000:]
    probe = subprocess.run([ffprobe, "-v", "error", "-show_entries", "stream=codec_type",
                            "-of", "default=noprint_wrappers=1", str(output)],
                           capture_output=True, text=True, timeout=10)
    assert probe.returncode == 0
    assert "codec_type=audio" in probe.stdout
