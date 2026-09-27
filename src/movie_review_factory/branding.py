"""Reusable channel identity and video band artwork."""
from __future__ import annotations

import io
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DEFAULT_NAME = "Màn Kể"
ASSETS = Path(__file__).resolve().parent / "assets"


def _brand_dir(jobs_root: Path) -> Path:
    return Path(jobs_root) / ".brand"


def load_settings(jobs_root: Path) -> dict:
    path = _brand_dir(jobs_root) / "settings.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        name = str(data.get("name") or DEFAULT_NAME)
        if not name.strip() or len(name) > 40 or any(ord(c) < 32 for c in name):
            raise ValueError("invalid saved name")
    except (OSError, ValueError, TypeError, AttributeError):
        name = DEFAULT_NAME
    return {"name": name}


def save_name(jobs_root: Path, name: str) -> dict:
    raw = str(name)
    name = raw.strip()
    if not name or len(name) > 40 or any(ord(c) < 32 or ord(c) == 127 for c in raw):
        raise ValueError("Tên kênh cần 1–40 ký tự và không chứa dấu xuống dòng.")
    folder = _brand_dir(jobs_root)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "settings.json.tmp"
    tmp.write_text(json.dumps({"name": name}, ensure_ascii=False), encoding="utf-8")
    tmp.replace(folder / "settings.json")
    return {"name": name}


def logo_path(jobs_root: Path) -> Path:
    custom = _brand_dir(jobs_root) / "logo.png"
    return custom if custom.is_file() else ASSETS / "man-ke.png"


def save_logo(jobs_root: Path, data: bytes) -> Path:
    if len(data) > 2_000_000 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Logo phải là PNG hợp lệ, tối đa 2 MB.")
    try:
        with Image.open(io.BytesIO(data)) as opened:
            opened.verify()
        with Image.open(io.BytesIO(data)) as opened:
            if opened.width < 64 or opened.height < 64 or opened.width > 4096 or opened.height > 4096:
                raise ValueError("Logo cần kích thước 64–4096 px.")
            image = opened.convert("RGBA")
            if image.getchannel("A").getextrema()[0] == 255:
                raise ValueError("Logo PNG cần nền trong suốt.")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            folder = _brand_dir(jobs_root)
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / "logo.png.tmp"
            image.save(temporary, format="PNG", optimize=True)
            temporary.replace(folder / "logo.png")
            return folder / "logo.png"
    except (OSError, SyntaxError) as exc:
        raise ValueError("Logo PNG không đọc được.") from exc


def chapter_titles(sections: list[str], combined_title: str) -> list[str]:
    films = [part.strip() for part in (combined_title or "").split(" + ") if part.strip()]
    if len(films) < 2:
        return [combined_title or "TÓM TẮT PHIM"] * len(sections)
    current = combined_title
    titles = []
    for index, section in enumerate(sections):
        if index == len(sections) - 1 and re.search(r"tổng kết|kết luận|outro|ending", section, re.I):
            current = combined_title
        else:
            for film in films:
                key = film.split(":")[-1].strip().casefold()
                if key and key in section.casefold():
                    current = film
                    break
        titles.append(current)
    return titles


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for filename in ("C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(filename, max(size, 8))
        except OSError:
            pass
    return ImageFont.load_default()


def _fitted(draw: ImageDraw.ImageDraw, label: str, max_width: int, size: int):
    font = _font(size)
    while size > 9 and draw.textbbox((0, 0), label, font=font)[2] > max_width:
        size -= 1
        font = _font(size)
    return font


def render_overlay(jobs_root: Path, target: Path, width: int, height: int, title: str,
                   top_band: float = 0, bottom_band: float = 0) -> Path:
    """Compose an RGBA branding layer; explicit band coverage is opt-in per job."""
    if width < 64 or height < 64 or not (0 <= top_band <= .2 and 0 <= bottom_band <= .2):
        raise ValueError("Kích thước video hoặc tỷ lệ dải nền không hợp lệ.")
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    top = round(height * top_band)
    bottom = round(height * bottom_band)
    margin = max(12, round(width * .028))
    name = load_settings(jobs_root)["name"]
    logo = Image.open(logo_path(jobs_root)).convert("RGBA")
    if top:
        draw.rectangle((0, 0, width, top - 1), fill=(10, 12, 17, 255))
    if bottom:
        draw.rectangle((0, height - bottom, width, height), fill=(10, 12, 17, 255))
    icon_size = min(max(26, round(height * .048)), max(20, top - 14) if top else 54)
    x = margin
    y = (top - icon_size) // 2 if top else margin
    if not top:
        label_width = round(width * .27)
        draw.rounded_rectangle((x - 8, y - 6, x + label_width, y + icon_size + 6),
                               radius=10, fill=(10, 12, 17, 205))
    logo.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
    image.alpha_composite(logo, (x, y))
    font = _fitted(draw, name, round(width * .42), round(icon_size * .48))
    name_height = draw.textbbox((0, 0), name, font=font)[3]
    draw.text((x + icon_size + 12, y + (icon_size - name_height) / 2),
              name, font=font, fill=(247, 248, 250, 255))
    caption = "REVIEW PHIM  /  " + (title or "TÓM TẮT PHIM").upper()
    title_font = _fitted(draw, caption, width - 2 * margin - 18, round(height * .026))
    bbox = draw.textbbox((0, 0), caption, font=title_font)
    text_height = bbox[3] - bbox[1]
    title_y = height - bottom + (bottom - text_height) // 2 - bbox[1] if bottom else height - margin - text_height - 18
    if not bottom:
        draw.rounded_rectangle((margin - 8, title_y - 7, margin + bbox[2] + 8, title_y + text_height + 14),
                               radius=9, fill=(10, 12, 17, 205))
    draw.text((margin, title_y), caption, font=title_font, fill=(245, 246, 248, 255))
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    return target


def render_card(jobs_root: Path, target: Path, width: int, height: int, *,
                headline: str, sublines: tuple[str, ...] | list[str] = ()) -> Path:
    """Render a full-frame intro/outro card: brand logo, headline and support lines.

    Frame-size agnostic (works for 16:9 and 9:16); positions are fractions of the
    frame so the card matches the render canvas. Used by the pipeline to bookend
    final.mp4 with a branded intro/outro when the project enables it.
    """
    if width < 64 or height < 64:
        raise ValueError("Kích thước thẻ mở đầu/kết thúc không hợp lệ.")
    image = Image.new("RGB", (width, height), (15, 18, 27))
    draw = ImageDraw.Draw(image)
    logo = Image.open(logo_path(jobs_root)).convert("RGBA")
    box = max(96, round(min(width, height) * 0.26))
    logo.thumbnail((box, box), Image.Resampling.LANCZOS)
    logo_y = round(height * 0.24)
    image.paste(logo, ((width - logo.width) // 2, logo_y), logo)
    cursor = logo_y + logo.height + round(height * 0.055)
    headline = " ".join(str(headline).split()) or load_settings(jobs_root)["name"]
    head_font = _fitted(draw, headline, round(width * 0.82), round(height * 0.072))
    hb = draw.textbbox((0, 0), headline, font=head_font)
    draw.text(((width - (hb[2] - hb[0])) / 2, cursor), headline,
              font=head_font, fill=(248, 249, 251))
    cursor += (hb[3] - hb[1]) + round(height * 0.045)
    for raw in sublines:
        line = " ".join(str(raw).split())
        if not line:
            continue
        font = _fitted(draw, line, round(width * 0.82), round(height * 0.038))
        box_line = draw.textbbox((0, 0), line, font=font)
        draw.text(((width - (box_line[2] - box_line[0])) / 2, cursor), line,
                  font=font, fill=(203, 213, 225))
        cursor += (box_line[3] - box_line[1]) + round(height * 0.022)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    return target
