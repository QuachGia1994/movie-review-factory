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


def test_bundled_logo_is_transparent_without_background_tile():
    with Image.open(branding.ASSETS / "man-ke.png") as image:
        logo = image.convert("RGBA")
    alpha = logo.getchannel("A")
    assert alpha.getextrema() == (0, 255)
    for corner in ((0, 0), (logo.width - 1, 0), (0, logo.height - 1), (logo.width - 1, logo.height - 1)):
        assert alpha.getpixel(corner) == 0
    # Mostly see-through: the old dark tile covered the whole square.
    assert sum(alpha.histogram()[201:]) < logo.width * logo.height * .45
    assert "#11151d" not in (branding.ASSETS / "man-ke.svg").read_text(encoding="utf-8").lower()


def test_overlay_without_bands_has_no_dark_plates(tmp_path):
    target = tmp_path / "overlay.png"
    branding.render_overlay(tmp_path, target, 1920, 1080, "BEN 10")
    with Image.open(target) as image:
        alpha = image.getchannel("A")
        # Lockup + caption keep only glyph/halo pixels; most of each area stays clear.
        for box in ((0, 0, 520, 140), (0, 960, 1100, 1080)):
            region = alpha.crop(box)
            assert region.getextrema()[1] == 255
            assert sum(region.histogram()[151:]) < region.width * region.height * .5
        assert alpha.getpixel((960, 540)) == 0
    branding.render_overlay(tmp_path, target, 1920, 1080, "BEN 10", include_mark=False)
    with Image.open(target) as image:
        assert image.getchannel("A").crop((0, 0, 520, 140)).getextrema()[1] == 0


def test_brand_guard_layers_are_transparent_and_ghost_is_faint(tmp_path, monkeypatch):
    mark = branding.render_mark(tmp_path, tmp_path / "mark.png", 1080)
    ghost = branding.render_ghost(tmp_path, tmp_path / "ghost.png", 1080)
    with Image.open(mark) as image:
        alpha = image.getchannel("A")
        assert alpha.getpixel((0, 0)) == 0 and alpha.getextrema()[1] == 255
    with Image.open(ghost) as image:
        assert image.getchannel("A").getextrema()[1] <= round(255 * branding.GHOST_OPACITY) + 1
    x, y = branding.mark_overlay_xy(30)
    assert "W-w-30" in x and f"t/{branding.MARK_HOP_SECONDS}" in x and y == "30"
    gx, gy = branding.ghost_overlay_xy(30, 1.5)
    assert f"floor(t/{branding.GHOST_HOP_SECONDS})" in gx and "H*0.16" in gy
    assert branding.ghost_overlay_xy(30, 2.5) != (gx, gy)
    monkeypatch.setenv(branding.BRAND_GUARD_ENV, "0")
    assert not branding.brand_guard_enabled()
    monkeypatch.setenv(branding.BRAND_GUARD_ENV, "1")
    assert branding.brand_guard_enabled()


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
