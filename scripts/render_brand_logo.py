"""Render the bundled Màn Kể logo (transparent PNG + SVG) from one geometry.

Run: py -3 scripts/render_brand_logo.py  -> src/movie_review_factory/assets/man-ke.{png,svg}
The mark is a speech-bubble screen (màn = screen, kể = telling) with a play
triangle, drawn in warm white + teal with a dark halo so it stays legible on
bright or dark footage without any background tile.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFilter

SIZE = 512
SCALE = 4
WHITE = (246, 246, 244)
TEAL = (102, 219, 199)
HALO = (8, 10, 14)
# Geometry in 512-space.
BUBBLE = (64, 84, 448, 352)
BUBBLE_R = 56
STROKE = 30
TAIL = [(150, 340), (122, 436), (238, 340)]
PLAY = [(222, 176), (322, 226), (222, 276)]


def _bezier(p0, p1, p2, steps=24):
    return [((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
             (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
            for t in (i / steps for i in range(steps + 1))]


# Curtain drapes (màn) tied back in the top corners of the screen.
_IX0, _IY0, _IX1 = BUBBLE[0] + STROKE - 2, BUBBLE[1] + STROKE - 2, BUBBLE[2] - STROKE + 2
DRAPES = [
    [(_IX0, _IY0)] + _bezier((_IX0 + 108, _IY0), (_IX0 + 26, _IY0 + 26), (_IX0, _IY0 + 136)),
    [(_IX1, _IY0)] + _bezier((_IX1 - 108, _IY0), (_IX1 - 26, _IY0 + 26), (_IX1, _IY0 + 136)),
]
PLAY_R = 18

def _svg() -> str:
    inset = STROKE / 2
    x, y = BUBBLE[0] + inset, BUBBLE[1] + inset
    w, h = BUBBLE[2] - BUBBLE[0] - STROKE, BUBBLE[3] - BUBBLE[1] - STROKE
    tail = "M" + "L".join(f"{px} {py}" for px, py in TAIL) + "Z"
    play = "M" + "L".join(f"{px} {py}" for px, py in PLAY) + "Z"
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIZE} {SIZE}" fill="none" aria-label="Màn Kể - màn hình kể chuyện">
<defs><filter id="halo" x="-10%" y="-10%" width="120%" height="120%">
<feMorphology in="SourceAlpha" operator="dilate" radius="8" result="grown"/>
<feGaussianBlur in="grown" stdDeviation="3" result="soft"/>
<feFlood flood-color="#080A0E" flood-opacity=".6"/>
<feComposite in2="soft" operator="in" result="shadow"/>
<feMerge><feMergeNode in="shadow"/><feMergeNode in="SourceGraphic"/></feMerge>
</filter></defs>
<g filter="url(#halo)">
<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{BUBBLE_R - inset}" stroke="#F6F6F4" stroke-width="{STROKE}"/>
<path d="{tail}" fill="#F6F6F4"/>
{_DRAPES_SVG}
<path d="{play}" fill="#66DBC7" stroke="#66DBC7" stroke-width="{PLAY_R * 2}" stroke-linejoin="round"/>
</g>
</svg>
"""


def _drape_path(points) -> str:
    head, *rest = points
    return f"M{head[0]:.1f} {head[1]:.1f}" + "".join(f"L{x:.1f} {y:.1f}" for x, y in rest) + "Z"


_DRAPES_SVG = ""


def _s(values):
    return [round(v * SCALE) for v in values]


def _mask() -> tuple[Image.Image, Image.Image]:
    big = SIZE * SCALE
    ring = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(ring)
    draw.rounded_rectangle(_s(BUBBLE), radius=BUBBLE_R * SCALE, fill=255)
    inner = [BUBBLE[0] + STROKE, BUBBLE[1] + STROKE, BUBBLE[2] - STROKE, BUBBLE[3] - STROKE]
    draw.rounded_rectangle(_s(inner), radius=(BUBBLE_R - STROKE) * SCALE, fill=0)
    draw.polygon([(x * SCALE, y * SCALE) for x, y in TAIL], fill=255)
    for drape in DRAPES:
        draw.polygon([(x * SCALE, y * SCALE) for x, y in drape], fill=255)
    play = _round_poly(PLAY, PLAY_R)
    return ring, play


def _round_poly(points, radius):
    big = SIZE * SCALE
    img = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(img)
    pts = [(x * SCALE, y * SCALE) for x, y in points]
    draw.polygon(pts, fill=255)
    draw.line(pts + [pts[0]], fill=255, width=radius * 2 * SCALE, joint="curve")
    for x, y in pts:
        r = radius * SCALE
        draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
    return img


def render(size: int = SIZE) -> Image.Image:
    ring, play = _mask()
    shape = ImageChops.lighter(ring, play)
    halo = shape.filter(ImageFilter.MaxFilter(10 * SCALE + 1)).filter(ImageFilter.GaussianBlur(3 * SCALE))
    big = SIZE * SCALE
    image = Image.new("RGBA", (big, big), HALO + (0,))
    image.putalpha(halo.point(lambda v: round(v * 0.6)))
    white = Image.new("RGBA", (big, big), WHITE + (255,))
    white.putalpha(ring)
    teal = Image.new("RGBA", (big, big), TEAL + (255,))
    teal.putalpha(play)
    image.alpha_composite(white)
    image.alpha_composite(teal)
    return image.resize((size, size), Image.Resampling.LANCZOS)


_DRAPES_SVG = "\n".join(f'<path d="{_drape_path(d)}" fill="#F6F6F4"/>' for d in DRAPES)


def main(out_dir: str) -> None:
    folder = Path(out_dir)
    render().save(folder / "man-ke.png", format="PNG", optimize=True)
    (folder / "man-ke.svg").write_text(_svg(), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[1] / "src" / "movie_review_factory" / "assets"))
