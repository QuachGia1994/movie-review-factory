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
    overrides.setdefault("chunk_seconds", 0)
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

def test_resolve_config_requires_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(wr.propainter_setup, "install_root", lambda: tmp_path / "missing")
    with pytest.raises(wr.ProPainterUnavailable):
        wr.resolve_config(env={})


def test_resolve_config_reads_home(tmp_path: Path) -> None:
    (tmp_path / wr.INFERENCE_SCRIPT).write_text("# stub", encoding="utf-8")
    cfg = wr.resolve_config(env={"MRF_PROPAINTER_DIR": str(tmp_path)})
    assert cfg.home == tmp_path


def test_resolve_config_reads_chunk_limits(tmp_path: Path) -> None:
    (tmp_path / wr.INFERENCE_SCRIPT).write_text("# stub", encoding="utf-8")
    cfg = wr.resolve_config(env={"MRF_PROPAINTER_DIR": str(tmp_path), "MRF_PROPAINTER_CHUNK_SECONDS": "7.5", "MRF_PROPAINTER_CHUNK_MAX_HEIGHT": "360"})
    assert cfg.chunk_seconds == 7.5
    assert cfg.chunk_max_height == 360


def test_chunked_remove_bounds_each_propainter_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask")
    output = tmp_path / "clean.mp4"
    seen: list[Path] = []
    monkeypatch.setattr(wr, "_probe_duration", lambda _source: 21.0)

    def fake_extract(_source, target, _start, _duration, _height):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"chunk")

    def fake_run(command, **_kwargs):
        if wr.INFERENCE_SCRIPT in command:
            chunk = Path(command[command.index("--video") + 1])
            seen.append(chunk)
            work = Path(command[command.index("--output") + 1]) / chunk.stem
            work.mkdir(parents=True, exist_ok=True)
            (work / wr._OUTPUT_VIDEO_NAME).write_bytes(b"clean")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_concat(chunks, _source, target, _work):
        assert len(chunks) == 3
        target.write_bytes(b"joined")

    monkeypatch.setattr(wr, "_extract_chunk", fake_extract)
    monkeypatch.setattr(wr, "_run_checked", fake_run)
    monkeypatch.setattr(wr, "_concat_chunks", fake_concat)
    result = wr.remove_watermark(source, mask, output, _config(tmp_path, chunk_seconds=10, chunk_max_height=360))
    assert result == output
    assert len(seen) == 3


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


# --- FFmpeg methods (delogo / blur) ----------------------------------------

def _box_mask(path: Path, size: tuple[int, int], box: tuple[int, int, int, int]) -> Path:
    image = Image.new("L", size, 0)
    image.paste(255, box)
    image.save(path)
    return path


def test_static_mask_returns_single_image_unchanged(tmp_path: Path) -> None:
    mask = _box_mask(tmp_path / "mask.png", (40, 20), (5, 5, 10, 10))
    assert wr.static_mask(mask, tmp_path / "static.png") == mask
    assert not (tmp_path / "static.png").exists()


def test_static_mask_unions_per_frame_folder(tmp_path: Path) -> None:
    frames = tmp_path / "masks"
    frames.mkdir()
    _box_mask(frames / "00000.png", (40, 20), (0, 0, 10, 10))
    _box_mask(frames / "00001.png", (40, 20), (30, 10, 40, 20))
    union = wr.static_mask(frames, tmp_path / "static.png")
    assert _pixel(union, (2, 2)) == 255
    assert _pixel(union, (35, 15)) == 255
    assert _pixel(union, (20, 5)) == 0
    coverage, bbox, size = wr.mask_stats(union)
    assert size == (40, 20)
    assert bbox == (0, 0, 40, 20)
    assert coverage == pytest.approx(200 / 800)


def test_static_mask_drops_pixels_seen_in_too_few_frames(tmp_path: Path) -> None:
    frames = tmp_path / "masks"
    frames.mkdir()
    for index in range(40):
        image = Image.new("L", (40, 20), 0)
        image.paste(255, (0, 0, 10, 10))  # the real, persistent watermark
        if index == 7:
            image.paste(255, (20, 0, 40, 20))  # one-frame false positive
        image.save(frames / f"{index:05d}.png")
    persistent = wr.static_mask(frames, tmp_path / "static.png")
    assert _pixel(persistent, (2, 2)) == 255
    assert _pixel(persistent, (30, 10)) == 0


def test_static_mask_falls_back_to_union_when_nothing_persists(tmp_path: Path) -> None:
    frames = tmp_path / "masks"
    frames.mkdir()
    for index in range(40):
        image = Image.new("L", (40, 20), 0)
        image.paste(255, (index, 0, index + 1, 1))
        image.save(frames / f"{index:05d}.png")
    union = wr.static_mask(frames, tmp_path / "static.png", min_fraction=0.5)
    assert _pixel(union, (0, 0)) == 255 and _pixel(union, (39, 0)) == 255


def test_delogo_command_scales_mask_bbox_to_video(tmp_path: Path) -> None:
    mask = _box_mask(tmp_path / "mask.png", (320, 180), (192, 9, 288, 36))
    command = wr.build_ffmpeg_command("ffmpeg", tmp_path / "in.mp4", mask, tmp_path / "out.mp4", "delogo", (640, 360))
    assert command[command.index("-vf") + 1] == "delogo=x=384:y=18:w=192:h=54"
    assert command[-1] == str(tmp_path / "out.mp4")
    assert "0:a?" in command


def test_delogo_rect_stays_inside_frame(tmp_path: Path) -> None:
    mask = _box_mask(tmp_path / "mask.png", (100, 50), (0, 0, 100, 50))
    command = wr.build_ffmpeg_command("ffmpeg", tmp_path / "in.mp4", mask, tmp_path / "out.mp4", "delogo", (100, 50))
    assert command[command.index("-vf") + 1] == "delogo=x=1:y=1:w=97:h=47"


def test_blur_command_composites_through_mask(tmp_path: Path) -> None:
    mask = _box_mask(tmp_path / "mask.png", (64, 36), (10, 10, 20, 20))
    command = wr.build_ffmpeg_command("ffmpeg", tmp_path / "in.mp4", mask, tmp_path / "out.mp4", "blur", (640, 360))
    graph = command[command.index("-filter_complex") + 1]
    assert "scale=640:360" in graph and "alphamerge" in graph and "overlay" in graph
    assert command[command.index("-loop") + 3] == str(mask)


def test_ffmpeg_command_rejects_empty_mask(tmp_path: Path) -> None:
    mask = tmp_path / "mask.png"
    Image.new("L", (20, 20), 0).save(mask)
    with pytest.raises(wr.WatermarkRemovalError, match="empty"):
        wr.build_ffmpeg_command("ffmpeg", tmp_path / "in.mp4", mask, tmp_path / "out.mp4", "blur", (20, 20))


def test_remove_watermark_ffmpeg_without_ffmpeg_is_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wr.shutil, "which", lambda name: None)
    with pytest.raises(wr.WatermarkToolUnavailable):
        wr.remove_watermark_ffmpeg(tmp_path / "in.mp4", tmp_path / "m.png", tmp_path / "out.mp4", "delogo", (10, 10))


def test_remove_watermark_ffmpeg_replaces_output_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "in.mp4"
    source.write_bytes(b"v")
    mask = _box_mask(tmp_path / "mask.png", (20, 20), (5, 5, 10, 10))
    monkeypatch.setattr(wr.shutil, "which", lambda name: "ffmpeg")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"clean")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(wr.subprocess, "run", fake_run)
    output = wr.remove_watermark_ffmpeg(source, mask, tmp_path / "clean.mp4", "blur", (20, 20))
    assert output.read_bytes() == b"clean"
    assert not (tmp_path / "clean.tmp.mp4").exists()


def test_remove_watermark_ffmpeg_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "in.mp4"
    source.write_bytes(b"v")
    mask = _box_mask(tmp_path / "mask.png", (20, 20), (5, 5, 10, 10))
    monkeypatch.setattr(wr.shutil, "which", lambda name: "ffmpeg")
    monkeypatch.setattr(wr.subprocess, "run", lambda command, **kwargs: types.SimpleNamespace(returncode=1, stdout="", stderr="bad filter"))
    with pytest.raises(wr.WatermarkRemovalError, match="bad filter"):
        wr.remove_watermark_ffmpeg(source, mask, tmp_path / "clean.mp4", "delogo", (20, 20))
    assert not (tmp_path / "clean.mp4").exists()


@pytest.mark.skipif(wr.shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
@pytest.mark.parametrize("method", wr.FFMPEG_METHODS)
def test_ffmpeg_methods_only_touch_masked_region(tmp_path: Path, method: str) -> None:
    import subprocess

    from PIL import ImageChops, ImageStat

    source = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=10:duration=1",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True, capture_output=True)
    mask = _box_mask(tmp_path / "mask.png", (320, 180), (200, 10, 300, 50))
    output = wr.remove_watermark_ffmpeg(source, mask, tmp_path / "clean.mp4", method, (320, 180))
    frames = {}
    for name, video in (("src", source), ("out", output)):
        frame = tmp_path / f"{name}.png"
        subprocess.run(["ffmpeg", "-y", "-i", str(video), "-frames:v", "1", str(frame)], check=True, capture_output=True)
        frames[name] = Image.open(frame).convert("L")
    diff = ImageChops.difference(frames["src"], frames["out"])
    assert ImageStat.Stat(diff.crop((210, 15, 290, 45))).mean[0] > 5
    assert ImageStat.Stat(diff.crop((0, 100, 320, 180))).mean[0] < 2


def _stage_job(tmp_path: Path, method: str):
    from movie_review_factory import pipeline
    from movie_review_factory.models import JobConfig, WatermarkRemoval

    root = tmp_path / "job"
    source = tmp_path / "src.mp4"
    source.write_bytes(b"v")
    frames = tmp_path / "masks"
    frames.mkdir()
    _box_mask(frames / "00000.png", (64, 36), (0, 0, 16, 9))
    _box_mask(frames / "00001.png", (64, 36), (48, 27, 64, 36))
    config = JobConfig(job_id="job", source_video=source,
                       watermark_removal=WatermarkRemoval(enabled=True, method=method, mask=str(frames)))
    pipeline.create_job(root, config)
    (root / "ingest.json").write_text('{"width": 640, "height": 360}', encoding="utf-8")
    return pipeline, root, pipeline.load_manifest(root)


@pytest.mark.parametrize("method", ["delogo", "blur"])
def test_watermark_stage_routes_ffmpeg_methods(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str) -> None:
    pipeline, root, manifest = _stage_job(tmp_path, method)
    calls = []

    def fake_ffmpeg(source, mask, output, chosen, frame_size, **kwargs):
        calls.append((Path(mask).name, chosen, frame_size))
        Path(output).write_bytes(b"clean")
        return Path(output)

    monkeypatch.setattr(pipeline.watermark_removal, "remove_watermark_ffmpeg", fake_ffmpeg)
    monkeypatch.setattr(pipeline.watermark_removal, "resolve_config", lambda: pytest.fail("ProPainter must not run"))
    _artifacts, message = pipeline._watermark(root, manifest)
    assert calls == [("watermark_mask_static.png", method, (640, 360))]
    assert "mask covers 12% of the frame" in message
    data = (root / "watermark.json").read_text(encoding="utf-8")
    assert f'"method": "{method}"' in data and '"mask_coverage": 0.125' in data


def test_watermark_stage_skips_when_ffmpeg_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline, root, manifest = _stage_job(tmp_path, "delogo")
    monkeypatch.setattr(wr.shutil, "which", lambda name: None)
    with pytest.raises(pipeline.SkipStage, match="ffmpeg"):
        pipeline._watermark(root, manifest)
