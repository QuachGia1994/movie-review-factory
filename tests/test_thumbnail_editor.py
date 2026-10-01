from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image, ImageChops

from movie_review_factory.thumbnail_editor import render_thumbnail_variants
from movie_review_factory.models import JobConfig
from movie_review_factory.pipeline import create_job, load_manifest, save_manifest, select_thumbnail


def _candidates(root: Path) -> None:
    records = []
    for index, color in enumerate(((24, 80, 120), (94, 65, 101), (125, 75, 41)), start=1):
        name = f"thumbnail-{index}.jpg"
        Image.new("RGB", (1280, 720), color).save(root / name)
        records.append({"index": index, "file": name})
    (root / "thumbnails.json").write_text(json.dumps({"candidates": records}), encoding="utf-8")
    (root / "thumbnail.jpg").write_bytes((root / "thumbnail-1.jpg").read_bytes())


def test_creates_three_editable_safe_area_variants_and_small_previews(tmp_path: Path) -> None:
    _candidates(tmp_path)
    original = (tmp_path / "thumbnail.jpg").read_bytes()
    manifest = render_thumbnail_variants(
        tmp_path, headline="BEN 10: AI ĐANG ĐIỀU KHIỂN THỜI GIAN?", channel_name="Màn Kể"
    )
    assert manifest["headline"] == "BEN 10: AI ĐANG ĐIỀU KHIỂN THỜI GIAN?"
    assert manifest["channel_name"] == "Màn Kể"
    assert len(manifest["variants"]) == 3
    assert len({v["layout"] for v in manifest["variants"]}) == 3
    assert (tmp_path / "thumbnail.jpg").read_bytes() == original
    for variant in manifest["variants"]:
        full = tmp_path / variant["file"]
        preview = tmp_path / variant["preview_file"]
        assert full.exists() and preview.exists()
        with Image.open(full) as canvas, Image.open(preview) as small:
            assert canvas.size == (1280, 720)
            assert small.size == (320, 180)
            assert ImageChops.difference(
                canvas, Image.open(tmp_path / variant["source_file"])
            ).getbbox() is not None
        for layer in variant["layers"]:
            x0, y0, x1, y1 = layer["box"]
            assert 90 <= x0 < x1 <= 1190
            assert 70 <= y0 < y1 <= 650
            assert layer["text"]
    saved = json.loads((tmp_path / "thumbnail_edits.json").read_text(encoding="utf-8"))
    assert saved == manifest


def test_reediting_preserves_previous_variant_bytes_for_approved_selection(tmp_path: Path) -> None:
    _candidates(tmp_path)
    first = render_thumbnail_variants(tmp_path, headline="BEN 10 BÍ ẨN", channel_name="Màn Kể")
    selected = tmp_path / first["variants"][0]["file"]
    original = selected.read_bytes()
    second = render_thumbnail_variants(tmp_path, headline="BEN 10 TRỞ LẠI", channel_name="Màn Kể")
    assert second["variants"][0]["file"] != first["variants"][0]["file"]
    assert selected.read_bytes() == original


def test_rejects_unreadable_headline_without_changing_selected_thumbnail(tmp_path: Path) -> None:
    _candidates(tmp_path)
    original = (tmp_path / "thumbnail.jpg").read_bytes()
    with pytest.raises(ValueError, match="headline"):
        render_thumbnail_variants(tmp_path, headline="A" * 90, channel_name="Màn Kể")
    assert (tmp_path / "thumbnail.jpg").read_bytes() == original
    assert not (tmp_path / "thumbnail_edits.json").exists()


def test_selecting_edited_artwork_resets_metadata_approval_and_export(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="thumbnail-edited"))
    _candidates(tmp_path)
    edit = render_thumbnail_variants(tmp_path, headline="BEN 10 BÍ ẨN", channel_name="Màn Kể")
    selected = edit["variants"][0]["file"]
    (tmp_path / "youtube_metadata.json").write_text(json.dumps({"approved": True}), encoding="utf-8")
    (tmp_path / "publish_record.json").write_text("{}", encoding="utf-8")
    manifest = load_manifest(tmp_path)
    manifest.stage("thumbnail").mark("ready")
    manifest.stage("publish").mark("ready")
    save_manifest(tmp_path, manifest)

    result = select_thumbnail(tmp_path, selected)
    assert result["primary_candidate"] == selected
    assert (tmp_path / "thumbnail.jpg").read_bytes() == (tmp_path / selected).read_bytes()
    assert json.loads((tmp_path / "youtube_metadata.json").read_text(encoding="utf-8"))["approved"] is False
    assert load_manifest(tmp_path).stage("publish").status == "pending"
    assert not (tmp_path / "publish_record.json").exists()


def test_reselecting_identical_artwork_keeps_metadata_approval(tmp_path: Path) -> None:
    create_job(tmp_path, JobConfig(job_id="thumbnail-same"))
    _candidates(tmp_path)
    source = (tmp_path / "thumbnail-1.jpg").read_bytes()
    (tmp_path / "thumbnail.jpg").write_bytes(source)
    (tmp_path / "youtube_metadata.json").write_text(json.dumps({"approved": True}), encoding="utf-8")
    (tmp_path / "publish_record.json").write_text("{}", encoding="utf-8")
    manifest = load_manifest(tmp_path)
    manifest.stage("thumbnail").mark("ready")
    manifest.stage("publish").mark("ready")
    save_manifest(tmp_path, manifest)
    result = select_thumbnail(tmp_path, "thumbnail-1.jpg")
    assert result["primary_candidate"] == "thumbnail-1.jpg"
    assert json.loads((tmp_path / "youtube_metadata.json").read_text(encoding="utf-8"))["approved"] is True
    assert load_manifest(tmp_path).stage("publish").status == "ready"
    assert (tmp_path / "publish_record.json").exists()


def test_rejects_candidate_path_traversal(tmp_path: Path) -> None:
    _candidates(tmp_path)
    data = json.loads((tmp_path / "thumbnails.json").read_text(encoding="utf-8"))
    data["candidates"][0]["file"] = "../outside.jpg"
    (tmp_path / "thumbnails.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate"):
        render_thumbnail_variants(tmp_path, headline="BEN 10", channel_name="Màn Kể")


def test_force_cover_paints_watermark_bands_over_every_variant(tmp_path: Path) -> None:
    _candidates(tmp_path)
    manifest = render_thumbnail_variants(
        tmp_path, headline="BEN 10 BÍ ẨN", channel_name="Màn Kể", force_cover=True
    )
    for variant in manifest["variants"]:
        with Image.open(tmp_path / variant["file"]) as canvas:
            rgb = canvas.convert("RGB")
            # Top (>=72px) and bottom (>=100px) bands are black at a right-edge corner clear of the plates.
            assert max(rgb.getpixel((1240, 8))) <= 8
            assert max(rgb.getpixel((1240, 712))) <= 8


def test_without_force_cover_leaves_source_corner_untouched(tmp_path: Path) -> None:
    _candidates(tmp_path)
    manifest = render_thumbnail_variants(tmp_path, headline="BEN 10 BÍ ẨN", channel_name="Màn Kể")
    with Image.open(tmp_path / manifest["variants"][0]["file"]) as canvas:
        # No manifest bands + no force_cover => corner keeps the (non-black) source colour.
        assert max(canvas.convert("RGB").getpixel((1240, 8))) > 30


def test_headlines_list_prints_a_distinct_headline_per_variant(tmp_path: Path) -> None:
    _candidates(tmp_path)
    manifest = render_thumbnail_variants(
        tmp_path,
        headline="CÂU MỘT",
        channel_name="Màn Kể",
        headlines=["CÂU MỘT", "CÂU HAI", "CÂU BA"],
    )
    assert [v["headline"] for v in manifest["variants"]] == ["CÂU MỘT", "CÂU HAI", "CÂU BA"]
    assert manifest["headline"] == "CÂU MỘT"
