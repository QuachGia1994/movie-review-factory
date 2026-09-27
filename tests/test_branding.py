"""Brand settings and render overlays keep the user-facing identity editable."""
from io import BytesIO

import pytest
from PIL import Image

from movie_review_factory import branding


def test_channel_name_persists_and_rejects_control_characters(tmp_path):
    assert branding.load_settings(tmp_path)["name"] == "Màn Kể"
    assert branding.save_name(tmp_path, "  Góc Phim  ")["name"] == "Góc Phim"
    assert branding.load_settings(tmp_path)["name"] == "Góc Phim"
    with pytest.raises(ValueError):
        branding.save_name(tmp_path, "Tên\nSai")
    assert branding.load_settings(tmp_path)["name"] == "Góc Phim"


def test_logo_upload_keeps_only_valid_transparent_png(tmp_path):
    picture = Image.new("RGBA", (100, 100), (255, 255, 255, 0))
    picture.putpixel((50, 50), (255, 255, 255, 255))
    buffer = BytesIO()
    picture.save(buffer, format="PNG")
    path = branding.save_logo(tmp_path, buffer.getvalue())
    assert path.exists()
    assert Image.open(path).mode == "RGBA"
    with pytest.raises(ValueError):
        branding.save_logo(tmp_path, b"<svg onload='x'>")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_brand_overlay_has_source_bands_and_readable_title(tmp_path):
    target = tmp_path / "overlay.png"
    branding.render_overlay(tmp_path, target, 1920, 1080,
                            "BEN 10: RACE AGAINST TIME", top_band=0.09, bottom_band=0.09)
    with Image.open(target) as image:
        assert image.size == (1920, 1080)
        assert image.getpixel((1000, 20))[3] == 255
        assert image.getpixel((1000, 1060))[3] == 255
        assert image.getpixel((1000, 540))[3] == 0


@pytest.mark.parametrize("width,height", [(1920, 1080), (1080, 1920)])
def test_render_card_is_frame_sized_and_branded(tmp_path, width, height):
    target = tmp_path / "intro-card.png"
    branding.render_card(tmp_path, target, width, height,
                         headline="Màn Kể", sublines=["Review phim · Ben 10", "Theo dõi kênh"])
    with Image.open(target) as image:
        assert image.size == (width, height)
        # Card is not a flat fill: logo + bright headline/support text render on
        # the dark (15,18,27) canvas, so the brightest red channel is near white.
        assert image.convert("RGB").getextrema()[0][1] > 200
        # Top of frame is the dark background (content starts ~24% down).
        assert image.convert("RGB").getpixel((width // 2, 4)) == (15, 18, 27)


def test_render_card_rejects_degenerate_frame(tmp_path):
    with pytest.raises(ValueError):
        branding.render_card(tmp_path, tmp_path / "x.png", 32, 32, headline="Màn Kể")


def test_chapter_titles_follow_explicit_movie_names():
    title = "Ben 10: Race Against Time + Ben 10: Alien Swarm"
    sections = ["Mở đầu", "Race Against Time: tuổi thơ", "Bí mật Eon",
                "Alien Swarm: đại dịch", "Đột nhập Hive", "Tổng kết"]
    assert branding.chapter_titles(sections, title) == [
        title, "Ben 10: Race Against Time", "Ben 10: Race Against Time",
        "Ben 10: Alien Swarm", "Ben 10: Alien Swarm", title,
    ]
