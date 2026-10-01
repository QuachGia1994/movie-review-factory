from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

_LAYOUTS = ("bottom-left", "top-left", "bottom-center")
_SAFE = (96, 72, 1184, 648)


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    raise RuntimeError("No Unicode TrueType font available for thumbnail text")


def _fit_headline(headline: str) -> tuple[ImageFont.FreeTypeFont, list[str]]:
    words = headline.split()
    for size in range(94, 47, -2):
        font = _font(size)
        probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
        lines: list[str] = []
        for word in words:
            candidate = (lines[-1] + " " + word) if lines else word
            if probe.textbbox((0, 0), candidate, font=font)[2] <= 1020:
                if lines:
                    lines[-1] = candidate
                else:
                    lines.append(candidate)
            else:
                if probe.textbbox((0, 0), word, font=font)[2] > 1020:
                    break
                lines.append(word)
        if len(lines) <= 2 and " ".join(lines) == headline:
            return font, lines
    raise ValueError("headline cannot fit legibly within two lines")


def headline_fits(headline: str) -> bool:
    """True when ``headline`` lays out within the two-line safe area.

    A non-raising wrapper around :func:`_fit_headline` so callers that generate
    headlines automatically (the AGY auto-thumbnail flow) can trim to a
    guaranteed-legible string instead of letting :func:`render_thumbnail_variants`
    raise on an over-long line.
    """
    text = " ".join((headline or "").split())
    if not text or len(text) > 64:
        return False
    try:
        _fit_headline(text)
    except ValueError:
        return False
    return True


def _draw_variant(source: Path, headline: str, channel: str, layout: str, top_band: float = 0.0, bottom_band: float = 0.0) -> tuple[Image.Image, list[dict]]:
    with Image.open(source) as original:
        base = original.convert("RGB").resize((1280, 720), Image.Resampling.LANCZOS)
    # Cover any residual top/bottom brand bands carried over from the source
    # channel before printing the new branding, matching the render/thumbnail
    # band cover (fractions of frame height, per branding.render_overlay).
    if top_band > 0 or bottom_band > 0:
        cover = ImageDraw.Draw(base)
        top_px = round(720 * top_band)
        bottom_px = round(720 * bottom_band)
        if top_px > 0:
            cover.rectangle((0, 0, 1280, top_px), fill=(0, 0, 0))
        if bottom_px > 0:
            cover.rectangle((0, 720 - bottom_px, 1280, 720), fill=(0, 0, 0))
    headline_font, lines = _fit_headline(headline)
    brand_font = _font(34)
    canvas = Image.new("RGBA", base.size)
    draw = ImageDraw.Draw(canvas)
    layers: list[dict] = []

    brand_width = int(draw.textbbox((0, 0), channel, font=brand_font)[2])
    if brand_width > 780:
        raise ValueError("channel_name exceeds safe area")
    brand_y = 99 if layout != "top-left" else 579
    brand_x = 110
    brand_box = (brand_x - 12, brand_y - 8, brand_x + brand_width + 14, brand_y + 50)
    draw.rounded_rectangle(brand_box, radius=12, fill=(6, 16, 31, 210))
    draw.text((brand_x, brand_y), channel, font=brand_font, fill=(255, 226, 87, 255), stroke_width=1, stroke_fill=(13, 17, 21, 255))
    layers.append({"kind": "brand", "text": channel, "box": list(brand_box)})

    line_height = headline_font.size + 15
    top = 195 if layout == "top-left" else 603 - line_height * len(lines)
    for line in lines:
        width = int(draw.textbbox((0, 0), line, font=headline_font)[2])
        x = (1280 - width) // 2 if layout == "bottom-center" else 110
        plate = (x - 12, top - 8, x + width + 16, top + headline_font.size + 13)
        draw.rounded_rectangle(plate, radius=15, fill=(6, 16, 31, 219))
        draw.text((x, top), line, font=headline_font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(6, 16, 31, 255))
        layers.append({"kind": "headline", "text": line, "box": list(plate)})
        top += line_height

    for layer in layers:
        x0, y0, x1, y1 = layer["box"]
        if not (_SAFE[0] <= x0 < x1 <= _SAFE[2] and _SAFE[1] <= y0 < y1 <= _SAFE[3]):
            raise ValueError("headline or channel_name exceeds thumbnail safe area")
    return Image.alpha_composite(base.convert("RGBA"), canvas).convert("RGB"), layers


def render_thumbnail_variants(
    job_root: Path,
    *,
    headline: str,
    channel_name: str,
    headlines: list[str] | None = None,
    force_cover: bool = False,
) -> dict:
    """Prepare editable overlays on three existing candidate images; selection remains manual.

    This does not modify thumbnail.jpg, metadata, script, approvals, or publication state.
    The caller must invalidate metadata/export approval if a chosen variant replaces the
    approved selected thumbnail.

    ``headlines`` (exactly three) prints a different headline on each variant
    instead of repeating ``headline`` on all three - used by the AGY
    auto-thumbnail flow. ``force_cover`` raises the brand band floors
    (top >= 0.10, bottom >= 0.14) so a source channel's residual watermark is
    always painted over even when the job never configured brand bands.
    """
    root = Path(job_root)
    headline = " ".join(headline.split())
    channel_name = " ".join(channel_name.split())
    if headlines is None:
        variant_headlines = [headline, headline, headline]
    else:
        variant_headlines = [" ".join(str(text).split()) for text in headlines]
        if len(variant_headlines) != 3:
            raise ValueError("headlines must provide exactly three entries")
        headline = variant_headlines[0]
    if not channel_name or len(channel_name) > 40:
        raise ValueError("channel_name must be between 1 and 40 characters")
    for text in variant_headlines:
        if not text or len(text) > 64:
            raise ValueError("headline must be between 1 and 64 characters")
        _fit_headline(text)
    records = json.loads((root / "thumbnails.json").read_text(encoding="utf-8"))["candidates"]
    if len(records) < 3:
        raise ValueError("at least three thumbnail candidates are required")
    source_names = []
    for record in records[:3]:
        name = record["file"]
        if not isinstance(name, str) or not name or Path(name).name != name or "/" in name or "\\" in name:
            raise ValueError("invalid thumbnail candidate path")
        if not (root / name).is_file():
            raise ValueError("missing thumbnail candidate")
        source_names.append(name)

    # Brand bands to cover on each variant, read from the job manifest when
    # present (default 0 = no cover, e.g. unit fixtures without a manifest).
    top_band = bottom_band = 0.0
    manifest_file = root / "manifest.json"
    if manifest_file.is_file():
        try:
            job_cfg = json.loads(manifest_file.read_text(encoding="utf-8")).get("config", {})
            top_band = float(job_cfg.get("brand_top_band") or 0)
            bottom_band = float(job_cfg.get("brand_bottom_band") or 0)
        except (ValueError, OSError, TypeError):
            top_band = bottom_band = 0.0
    if force_cover:
        # Auto flow always covers a source's residual band watermark; the manual flow keeps the job's own bands.
        top_band = max(top_band, 0.10)
        bottom_band = max(bottom_band, 0.14)

    variants = []
    with tempfile.TemporaryDirectory(prefix=".thumbnail-edits-", dir=root) as directory:
        staged = Path(directory)
        for i, (source_name, layout) in enumerate(zip(source_names, _LAYOUTS), start=1):
            variant_headline = variant_headlines[i - 1]
            revision = hashlib.sha256(
                (variant_headline + "\0" + channel_name).encode("utf-8")
                + (root / source_name).read_bytes()
            ).hexdigest()[:12]
            full_name = f"thumbnail-edit-{i}-{revision}.jpg"
            preview_name = f"thumbnail-edit-{i}-{revision}-small.jpg"
            image, layers = _draw_variant(root / source_name, variant_headline, channel_name, layout, top_band, bottom_band)
            image.save(staged / full_name, quality=92, subsampling=0)
            image.resize((320, 180), Image.Resampling.LANCZOS).save(staged / preview_name, quality=90)
            variants.append({
                "index": i,
                "source_file": source_name,
                "layout": layout,
                "headline": variant_headline,
                "file": full_name,
                "preview_file": preview_name,
                "layers": layers,
                "safe_area": list(_SAFE),
            })
        result = {
            "headline": headline,
            "channel_name": channel_name,
            "width": 1280,
            "height": 720,
            "preview_width": 320,
            "preview_height": 180,
            "variants": variants,
        }
        (staged / "thumbnail_edits.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        for variant in variants:
            for key in ("file", "preview_file"):
                os.replace(staged / variant[key], root / variant[key])
        os.replace(staged / "thumbnail_edits.json", root / "thumbnail_edits.json")
    return result
