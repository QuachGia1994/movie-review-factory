"""Reusable channel identity and video band artwork."""
from __future__ import annotations

import io
import json
import os
import re
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

DEFAULT_NAME = "Màn Kể"
ASSETS = Path(__file__).resolve().parent / "assets"

# Brand guard (anti-removal watermark): docs/research/brand-guard.md.
BRAND_GUARD_ENV = "MRF_BRAND_GUARD"
MARK_HOP_SECONDS = 45
GHOST_HOP_SECONDS = 11
GHOST_OPACITY = 0.16
_INK = (8, 10, 14)


def brand_guard_enabled() -> bool:
    return os.environ.get(BRAND_GUARD_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


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


def _store_png(folder: Path, data: bytes) -> Path:
    """Validate a transparent PNG and store it atomically as ``folder/logo.png``."""
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
            folder.mkdir(parents=True, exist_ok=True)
            temporary = folder / "logo.png.tmp"
            image.save(temporary, format="PNG", optimize=True)
            temporary.replace(folder / "logo.png")
            return folder / "logo.png"
    except (OSError, SyntaxError) as exc:
        raise ValueError("Logo PNG không đọc được.") from exc


def save_logo(jobs_root: Path, data: bytes) -> Path:
    return _store_png(_brand_dir(jobs_root), data)


# -- multi-channel profiles: per-profile logos + activation ------------------

def _channels_dir(jobs_root: Path) -> Path:
    return _brand_dir(jobs_root) / "channels"


def channel_logo_path(jobs_root: Path, profile_id: str) -> Path | None:
    """Return a channel profile's own logo file, or None when it has none."""
    candidate = _channels_dir(jobs_root) / profile_id / "logo.png"
    return candidate if candidate.is_file() else None


def save_channel_logo(jobs_root: Path, profile_id: str, data: bytes) -> Path:
    return _store_png(_channels_dir(jobs_root) / profile_id, data)


def delete_channel_assets(jobs_root: Path, profile_id: str) -> None:
    shutil.rmtree(_channels_dir(jobs_root) / profile_id, ignore_errors=True)


def activate_channel(jobs_root: Path, name: str, profile_id: str) -> dict:
    """Materialise a channel profile into the shared brand identity (name + logo).

    Copies the profile's own logo onto the active ``logo.png`` (or clears it so the
    bundled default applies) and saves the name. Every render/thumbnail stage reads
    the identity from here, so switching channels needs no pipeline changes.
    """
    result = save_name(jobs_root, name)
    source = channel_logo_path(jobs_root, profile_id)
    dest = _brand_dir(jobs_root) / "logo.png"
    if source is not None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    else:
        dest.unlink(missing_ok=True)
    return result


# -- per-channel transition SFX files ----------------------------------------

# Upload Content-Type -> storage extension; FFmpeg probes the real format at render, unknown types are rejected.
_SFX_CONTENT_EXT = {
    "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
    "audio/wav": ".wav", "audio/x-wav": ".wav", "audio/wave": ".wav",
    "audio/mp4": ".m4a", "audio/aac": ".m4a", "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg", "application/ogg": ".ogg", "audio/webm": ".webm",
}


def _sfx_dir(jobs_root: Path, profile_id: str) -> Path:
    return _channels_dir(jobs_root) / profile_id / "sfx"


def channel_sfx_path(jobs_root: Path, profile_id: str, slug: str) -> Path | None:
    """Return the stored audio file for a channel SFX slug, or None if absent."""
    folder = _sfx_dir(jobs_root, profile_id)
    if not folder.is_dir():
        return None
    for item in sorted(folder.glob(slug + ".*")):
        if item.is_file():
            return item
    return None


def save_channel_sfx_file(jobs_root: Path, profile_id: str, slug: str, content_type: str, data: bytes) -> Path:
    ext = _SFX_CONTENT_EXT.get((content_type or "").split(";")[0].strip().lower())
    if ext is None:
        raise ValueError("Định dạng âm thanh không hỗ trợ (dùng mp3/wav/m4a/ogg).")
    if len(data) <= 0 or len(data) > 3_000_000:
        raise ValueError("Tệp SFX cần 1 byte–3 MB.")
    folder = _sfx_dir(jobs_root, profile_id)
    folder.mkdir(parents=True, exist_ok=True)
    for item in folder.glob(slug + ".*"):  # replace any prior file for this slug
        item.unlink(missing_ok=True)
    target = folder / (slug + ext)
    temporary = folder / (slug + ext + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(target)
    return target


def delete_channel_sfx_file(jobs_root: Path, profile_id: str, slug: str) -> None:
    folder = _sfx_dir(jobs_root, profile_id)
    if folder.is_dir():
        for item in folder.glob(slug + ".*"):
            item.unlink(missing_ok=True)


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


def _with_halo(layer: Image.Image, spread: int, blur: float, strength: float = .7) -> Image.Image:
    """Put a soft dark halo under a transparent layer: legible on any footage, no box."""
    halo = layer.getchannel("A").filter(ImageFilter.MaxFilter(spread * 2 + 1)).filter(ImageFilter.GaussianBlur(blur))
    out = Image.new("RGBA", layer.size, _INK + (0,))
    out.putalpha(halo.point(lambda value: round(value * strength)))
    out.alpha_composite(layer)
    return out


def brand_lockup(jobs_root: Path, icon_size: int) -> Image.Image:
    """Logo + channel name on a tight transparent canvas (no background tile)."""
    name = load_settings(jobs_root)["name"]
    logo = Image.open(logo_path(jobs_root)).convert("RGBA")
    logo.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
    font = _font(round(icon_size * .5))
    stroke = max(2, icon_size // 16)
    pad = stroke * 3 + 4
    left, top, right, bottom = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox(
        (0, 0), name, font=font, stroke_width=stroke)
    gap = max(8, icon_size // 4)
    width = pad * 2 + logo.width + gap + (right - left)
    height = pad * 2 + max(logo.height, bottom - top)
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    layer.alpha_composite(logo, (pad, (height - logo.height) // 2))
    ImageDraw.Draw(layer).text(
        (pad + logo.width + gap - left, (height - (bottom - top)) // 2 - top), name, font=font,
        fill=(247, 248, 250, 255), stroke_width=stroke, stroke_fill=_INK + (220,))
    return _with_halo(layer, 1, max(2, stroke))


def _corner_icon(height: int) -> int:
    return min(max(26, round(height * .048)), 54)


def render_mark(jobs_root: Path, target: Path, height: int) -> Path:
    """Corner brand lockup for the brand guard; the render moves it between top corners."""
    target.parent.mkdir(parents=True, exist_ok=True)
    brand_lockup(jobs_root, _corner_icon(height)).save(target, format="PNG")
    return target


def render_ghost(jobs_root: Path, target: Path, height: int, opacity: float = GHOST_OPACITY) -> Path:
    """Faint lockup the render drifts over the picture: visible to people, hard to mask."""
    layer = brand_lockup(jobs_root, max(24, round(height * .04)))
    layer.putalpha(layer.getchannel("A").point(lambda value: round(value * opacity)))
    target.parent.mkdir(parents=True, exist_ok=True)
    layer.save(target, format="PNG")
    return target


def _hash01(slot: str, factor: float, seed: float) -> str:
    inner = f"abs(sin({slot}*{factor}+{seed:.4f}))*43758.5453"
    return f"({inner}-floor({inner}))"


def mark_overlay_xy(margin: int) -> tuple[str, str]:
    """FFmpeg overlay x/y: the corner lockup swaps top-left/top-right every hop."""
    return f"if(eq(mod(floor(t/{MARK_HOP_SECONDS}),2),0),{margin},W-w-{margin})", str(margin)


def ghost_overlay_xy(margin: int, seed: float) -> tuple[str, str]:
    """FFmpeg overlay x/y: a per-job pseudo-random spot every hop, between 16% and 64% of
    the frame height so it stays clear of the corner lockup and the captions."""
    slot = f"floor(t/{GHOST_HOP_SECONDS})"
    return (f"{margin}+(W-w-{2 * margin})*{_hash01(slot, 12.9898, seed)}",
            f"H*0.16+(H*0.48-h)*{_hash01(slot, 78.233, seed)}")


def render_overlay(jobs_root: Path, target: Path, width: int, height: int, title: str,
                   top_band: float = 0, bottom_band: float = 0, include_mark: bool = True) -> Path:
    """Compose an RGBA branding layer; explicit band coverage is opt-in per job.

    Outside the opt-in bands nothing is boxed: the lockup and the chapter caption get
    a stroke and a soft halo instead of a dark plate, so no footage is hidden.
    ``include_mark=False`` leaves the corner lockup to the moving brand-guard layer.
    """
    if width < 64 or height < 64 or not (0 <= top_band <= .2 and 0 <= bottom_band <= .2):
        raise ValueError("Kích thước video hoặc tỷ lệ dải nền không hợp lệ.")
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    top = round(height * top_band)
    bottom = round(height * bottom_band)
    margin = max(12, round(width * .028))
    if top:
        draw.rectangle((0, 0, width, top - 1), fill=(10, 12, 17, 255))
    if bottom:
        draw.rectangle((0, height - bottom, width, height), fill=(10, 12, 17, 255))
    if top:
        name = load_settings(jobs_root)["name"]
        logo = Image.open(logo_path(jobs_root)).convert("RGBA")
        icon_size = min(max(26, round(height * .048)), max(20, top - 14))
        x, y = margin, (top - icon_size) // 2
        logo.thumbnail((icon_size, icon_size), Image.Resampling.LANCZOS)
        image.alpha_composite(logo, (x, y))
        font = _fitted(draw, name, round(width * .42), round(icon_size * .48))
        name_height = draw.textbbox((0, 0), name, font=font)[3]
        draw.text((x + icon_size + 12, y + (icon_size - name_height) / 2),
                  name, font=font, fill=(247, 248, 250, 255))
    elif include_mark:
        lockup = brand_lockup(jobs_root, _corner_icon(height))
        image.alpha_composite(lockup, (max(0, margin - 10), max(0, margin - 10)))
    caption = "REVIEW PHIM  /  " + (title or "TÓM TẮT PHIM").upper()
    title_font = _fitted(draw, caption, width - 2 * margin - 18, round(height * .026))
    bbox = draw.textbbox((0, 0), caption, font=title_font)
    text_height = bbox[3] - bbox[1]
    if bottom:
        title_y = height - bottom + (bottom - text_height) // 2 - bbox[1]
        draw.text((margin, title_y), caption, font=title_font, fill=(245, 246, 248, 255))
    else:
        stroke = max(2, round(height * .0025))
        pad = stroke * 3 + 4
        strip = Image.new("RGBA", (bbox[2] + 2 * pad, bbox[3] + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(strip).text((pad, pad), caption, font=title_font, fill=(245, 246, 248, 255),
                                   stroke_width=stroke, stroke_fill=_INK + (220,))
        strip = _with_halo(strip, 1, stroke + 1)
        image.alpha_composite(strip, (margin - pad, height - margin - text_height - 18 - pad))
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
