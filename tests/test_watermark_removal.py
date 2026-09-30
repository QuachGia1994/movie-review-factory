"""Unit tests for the optional ProPainter watermark-removal wrapper.

These are fast and hermetic: mask generation uses real Pillow, and the
ProPainter subprocess is faked, so no GPU/model/ffmpeg is required.
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

from movie_review_factory import watermark_removal as wr


def _pixel(path: Path, xy: tuple[int, int]) -> int:
    with Image.open(path) as image:
        return image.convert("L").getpixel(xy)


def _config(home: Path, **overrides) -> wr.ProPainterConfig:
    return wr.ProPainterConfig(home=home, python_bin="python", **overrides)


# --- mask helpers -----------------------------------------------------------

def test_rect_mask_from_bands_marks_top_and_bottom(tmp_path: Path) -> None:
    out = wr.rect_mask_from_bands(
        100, 100, tmp_path / "m.png", top_fraction=0.1, bottom_fraction=0.2
    )
    assert _pixel(out, (50, 2)) == 255   # inside top band
    assert _pixel(out, (50, 50)) == 0    # untouched middle
    assert _pixel(out, (50, 95)) == 255  # inside bottom band


def test_rect_mask_from_boxes_marks_region(tmp_path: Path) -> None:
    out = wr.rect_mask_from_boxes(100, 100, [[0.25, 0.25, 0.5, 0.5]], tmp_path / "b.png")
    assert _pixel(out, (50, 50)) == 255  # inside box
    assert _pixel(out, (5, 5)) == 0      # outside box


def test_rect_mask_from_boxes_rejects_empty(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        wr.rect_mask_from_boxes(100, 100, [[0, 0, 0, 0]], tmp_path / "e.png")


# --- per-frame mask folders -------------------------------------------------

def test_frame_mask_paths_sorted_and_filtered(tmp_path: Path) -> None:
    for name in ("00002.png", "00000.png", "00001.png"):
        (tmp_path / name).write_bytes(b"\x89PNG")
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    names = [p.name for p in wr.frame_mask_paths(tmp_path)]
    assert names == ["00000.png", "00001.png", "00002.png"]


def test_remove_watermark_accepts_mask_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    mask_dir = tmp_path / "masks"
    mask_dir.mkdir()
    (mask_dir / "00000.png").write_bytes(b"\x89PNG")
    output = tmp_path / "clean.mp4"
    seen: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        seen["mask"] = cmd[cmd.index("--mask") + 1]
        work = Path(cmd[cmd.index("--output") + 1])
        produced = work / "source" / wr._OUTPUT_VIDEO_NAME
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(b"clean-video")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wr.subprocess, "run", fake_run)
    result = wr.remove_watermark(source, mask_dir, output, _config(tmp_path))
    assert result == output
    assert seen["mask"] == str(mask_dir)  # the folder is passed straight through


def test_remove_watermark_rejects_empty_mask_folder(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"v")
    empty = tmp_path / "masks"
    empty.mkdir()
    with pytest.raises(wr.ProPainterError):
        wr.remove_watermark(source, empty, tmp_path / "clean.mp4", _config(tmp_path))


# --- config resolution ------------------------------------------------------

def test_resolve_config_requires_home() -> None:
    with pytest.raises(wr.ProPainterUnavailable):
        wr.resolve_config(env={})


def test_resolve_config_reads_home(tmp_path: Path) -> None:
    (tmp_path / wr.INFERENCE_SCRIPT).write_text("# stub", encoding="utf-8")
    cfg = wr.resolve_config(env={"MRF_PROPAINTER_DIR": str(tmp_path)})
    assert cfg.home == tmp_path


def test_resolve_config_missing_script_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(wr.ProPainterUnavailable):
        wr.resolve_config(env={"MRF_PROPAINTER_DIR": str(tmp_path)})


# --- command building -------------------------------------------------------

def test_build_command_includes_core_flags(tmp_path: Path) -> None:
    cmd = wr.build_command(_config(tmp_path), Path("in.mp4"), Path("m.png"), tmp_path / "w")
    assert wr.INFERENCE_SCRIPT in cmd
    for flag in ("--video", "--mask", "--output", "--mask_dilation"):
        assert flag in cmd
    assert "--fp16" in cmd  # cuda + fp16 default


def test_build_command_omits_fp16_on_cpu(tmp_path: Path) -> None:
    cmd = wr.build_command(
        _config(tmp_path, device="cpu"), Path("in.mp4"), Path("m.png"), tmp_path / "w"
    )
    assert "--fp16" not in cmd


def test_build_command_adds_resize_ratio(tmp_path: Path) -> None:
    cmd = wr.build_command(
        _config(tmp_path, resize_ratio=0.5), Path("in.mp4"), Path("m.png"), tmp_path / "w"
    )
    assert "--resize_ratio" in cmd


# --- end-to-end wrapper (faked subprocess) ----------------------------------

def test_remove_watermark_relocates_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"\x89PNG")
    output = tmp_path / "clean.mp4"

    def fake_run(cmd, **kwargs):
        work = Path(cmd[cmd.index("--output") + 1])
        produced = work / "source" / wr._OUTPUT_VIDEO_NAME
        produced.parent.mkdir(parents=True, exist_ok=True)
        produced.write_bytes(b"clean-video")
        return types.SimpleNamespace(returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(wr.subprocess, "run", fake_run)
    result = wr.remove_watermark(source, mask, output, _config(tmp_path))

    assert result == output
    assert output.read_bytes() == b"clean-video"
    # the scratch work dir is cleaned up on success
    assert not (output.parent / f"{output.stem}.propainter").exists()


def test_remove_watermark_raises_on_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"v")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"m")

    def fake_run(cmd, **kwargs):
        return types.SimpleNamespace(returncode=1, stdout="", stderr="boom")

    monkeypatch.setattr(wr.subprocess, "run", fake_run)
    with pytest.raises(wr.ProPainterError):
        wr.remove_watermark(source, mask, tmp_path / "clean.mp4", _config(tmp_path))


def test_remove_watermark_missing_source_errors(tmp_path: Path) -> None:
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"m")
    with pytest.raises(wr.ProPainterError):
        wr.remove_watermark(tmp_path / "nope.mp4", mask, tmp_path / "clean.mp4", _config(tmp_path))
