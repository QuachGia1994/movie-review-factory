"""Unit tests for automatic per-frame watermark mask detection.

Hermetic: mask maths uses real Pillow; FFmpeg and any external detector are
faked, so no ffmpeg/GPU/model is required.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

from movie_review_factory import mask_detection as md


def _pixel(path: Path, xy: tuple[int, int]) -> int:
    with Image.open(path) as image:
        return image.convert("L").getpixel(xy)


# --- pure detectors ---------------------------------------------------------

def test_color_mask_marks_target_colour(tmp_path: Path) -> None:
    img = Image.new("RGB", (4, 4), (0, 0, 0))
    for x in range(2):
        for y in range(2):
            img.putpixel((x, y), (255, 255, 255))  # white 2x2 block, top-left
    mask = md.color_mask(img, (255, 255, 255), tolerance=30)
    assert mask.getpixel((0, 0)) == 255  # inside white block
    assert mask.getpixel((3, 3)) == 0    # black background


def test_temporal_static_mask_flags_unchanging_pixels() -> None:
    frame_a = Image.new("L", (4, 2), 100)
    frame_b = Image.new("L", (4, 2), 100)
    for y in range(2):  # right half changes between frames, left half is static
        frame_b.putpixel((2, y), 200)
        frame_b.putpixel((3, y), 200)
    mask = md.temporal_static_mask([frame_a, frame_b], threshold=12)
    assert mask.getpixel((0, 0)) == 255  # static -> flagged
    assert mask.getpixel((3, 0)) == 0    # changed -> not flagged


# --- external detector hook -------------------------------------------------

def test_resolve_external_command_requires_env() -> None:
    with pytest.raises(md.MaskDetectorUnavailable):
        md.resolve_external_command(env={})


def test_build_external_command_expands_placeholders(tmp_path: Path) -> None:
    argv = md.build_external_command("detect --in {video} --out {out}", Path("v.mp4"), tmp_path)
    assert "v.mp4" in " ".join(argv)
    assert str(tmp_path) in " ".join(argv)


# --- orchestration (faked FFmpeg) -------------------------------------------

def test_generate_frame_masks_color_writes_numbered_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"video")
    out_dir = tmp_path / "masks"

    def fake_extract(video, out_dir, **kwargs):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(3):
            frame = out_dir / f"{index:05d}.png"
            Image.new("RGB", (4, 4), (255, 255, 255)).save(frame)
            paths.append(frame)
        return paths

    monkeypatch.setattr(md, "extract_frames", fake_extract)
    result = md.generate_frame_masks(video, out_dir, md.DetectSettings(method="color"))

    assert result == out_dir
    names = sorted(p.name for p in out_dir.glob("*.png"))
    assert names == ["00000.png", "00001.png", "00002.png"]
    # white frames -> fully white masks
    assert _pixel(out_dir / "00000.png", (0, 0)) == 255


def test_generate_frame_masks_rejects_unknown_method(tmp_path: Path) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"v")
    with pytest.raises(md.MaskDetectorError):
        md.generate_frame_masks(video, tmp_path / "m", md.DetectSettings(method="bogus"))


def test_generate_frame_masks_external(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "in.mp4"
    video.write_bytes(b"v")
    out_dir = tmp_path / "masks"

    def fake_run(cmd, **kwargs):
        target = Path(cmd[cmd.index("--out") + 1])
        target.mkdir(parents=True, exist_ok=True)
        (target / "00000.png").write_bytes(b"\x89PNG")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(md.subprocess, "run", fake_run)
    settings = md.DetectSettings(method="external", external_cmd="detect --in {video} --out {out}")
    result = md.generate_frame_masks(video, out_dir, settings)
    assert result == out_dir
    assert (out_dir / "00000.png").exists()


# --- external detector probe (single-frame self-test) -----------------------

def _fake_probe_run(cmd, **kwargs):
    if "-frames:v" in cmd:  # ffmpeg extract of one probe frame
        Path(cmd[-1]).write_bytes(b"clip")
    else:  # the external detector
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "00000.png").write_bytes(b"mask")
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def test_probe_external_detector_reports_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    monkeypatch.setattr(md.subprocess, "run", _fake_probe_run)
    probe = md.probe_external_detector(
        video,
        md.DetectSettings(method="external", external_cmd="detect --in {video} --out {out}"),
        ffmpeg="ffmpeg",
    )
    assert probe.ok is True
    assert probe.masks >= 1


def test_probe_external_detector_without_command(tmp_path: Path) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")
    probe = md.probe_external_detector(video, md.DetectSettings(method="external"), ffmpeg="ffmpeg", env={})
    assert probe.ok is False
    assert "no detector command" in probe.message


def test_probe_external_detector_reports_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "v.mp4"
    video.write_bytes(b"v")

    def fake_run(cmd, **kwargs):
        if "-frames:v" in cmd:
            Path(cmd[-1]).write_bytes(b"clip")
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")
        return types.SimpleNamespace(returncode=3, stdout="", stderr="boom")

    monkeypatch.setattr(md.subprocess, "run", fake_run)
    probe = md.probe_external_detector(
        video,
        md.DetectSettings(method="external", external_cmd="detect --in {video} --out {out}"),
        ffmpeg="ffmpeg",
    )
    assert probe.ok is False
    assert "exited 3" in probe.message
