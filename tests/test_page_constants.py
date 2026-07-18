"""Guard against JS<->Python constant drift. The reference page (index.html) and the
Easel (browser_canvas.py) independently hardcode the fiducial colors and the palette;
if they ever disagree, localization or color selection silently breaks. These tests
parse the page and assert the two sources still agree."""

import os
import re

from easels import browser_canvas as BC

PAGE = os.path.join(
    os.path.dirname(__file__), "..", "easels", "canvas_page", "index.html"
)


def _html():
    with open(PAGE, encoding="utf-8") as f:
        return f.read()


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def test_fiducial_colors_match_page():
    """Each `#fid-XX { ... background: #rrggbb }` rule must match FIDUCIAL_COLORS."""
    html = _html()
    parsed = {}
    for corner, hex_color in re.findall(
        r"#fid-(tl|tr|bl|br)\s*\{[^}]*background:\s*(#[0-9a-fA-F]{6})", html
    ):
        parsed[corner] = _hex_to_rgb(hex_color)
    assert parsed == BC.FIDUCIAL_COLORS


def test_palette_colors_match_page():
    """The JS `COLORS` array must match PALETTE, in order."""
    html = _html()
    m = re.search(r"const\s+COLORS\s*=\s*\[([^\]]*)\]", html)
    assert m, "COLORS array not found in page"
    hexes = re.findall(r"#[0-9a-fA-F]{6}", m.group(1))
    parsed = tuple(_hex_to_rgb(h) for h in hexes)
    assert parsed == BC.PALETTE


def test_width_presets_match_page():
    """The JS `WIDTHS` array must match WIDTH_PRESETS, in order."""
    html = _html()
    m = re.search(r"const\s+WIDTHS\s*=\s*\[([^\]]*)\]", html, re.DOTALL)
    assert m, "WIDTHS array not found in page"
    parsed = []
    for name, width, color in re.findall(
        r"\{name:\s*'([^']+)',\s*width:\s*([0-9.]+),\s*color:\s*'(#[0-9a-fA-F]{6})'\}",
        m.group(1),
    ):
        parsed.append((name, float(width), _hex_to_rgb(color)))
    expected = [(p.name, p.width, p.locator_color) for p in BC.WIDTH_PRESETS]
    assert parsed == expected


def test_pointerdown_paints_single_point_dab():
    """A one-point Stroke produces a visible mark: the page paints a brush-radius dab
    on pointerup only when no pointermove events arrived. Drag strokes must not get
    circular endpoint dabs, because those would reintroduce lateral bleed."""
    html = _html()
    assert "function dab(p)" in html
    assert "ctx.arc(p.x, p.y, currentWidth / 2" in html
    assert "if (drawing && !moved && last) dab(last);" in html
    pointerdown = html[html.index("canvas.addEventListener('pointerdown'"):]
    pointerdown = pointerdown[:pointerdown.index("canvas.addEventListener('pointermove'")]
    assert "dab(" not in pointerdown


def test_fiducial_geometry_matches_inset():
    """Guard FIDUCIAL_INSET against silent drift from the page's layout (the F2
    geometry). FIDUCIAL_INSET = -15.0 is correct ONLY because it was hand-derived from
    the page's `#frame` padding, the `.fid` size, and each fiducial's `left/top`. If
    any of those CSS values change without re-deriving the inset, localization breaks
    (or fiducials could re-overlap the canvas, reintroducing the exact bug F2 fixed).

    We re-derive each fiducial's centroid in canvas coordinates straight from the CSS
    and assert it equals the destination quad the homography expects for
    BC.FIDUCIAL_INSET / BC.CANVAS_SIZE:
        canvas_coord(centroid) = (left + size/2 - pad,  top + size/2 - pad)
    (the canvas's internal origin sits at frame pixel (pad, pad))."""
    html = _html()

    pad_m = re.search(r"#frame\s*\{[^}]*\bpadding:\s*(\d+)px", html)
    size_m = re.search(r"\.fid\s*\{[^}]*\bwidth:\s*(\d+)px", html)
    assert pad_m and size_m, "could not parse #frame padding / .fid size from page"
    pad = int(pad_m.group(1))
    size = int(size_m.group(1))

    centroids = {}
    for corner, left, top in re.findall(
        r"#fid-(tl|tr|bl|br)\s*\{[^}]*\bleft:\s*(\d+)px[^}]*\btop:\s*(\d+)px", html
    ):
        cx = int(left) + size / 2 - pad
        cy = int(top) + size / 2 - pad
        centroids[corner] = (cx, cy)
    assert set(centroids) == {"tl", "tr", "bl", "br"}, "missing a fiducial rule"

    inset = BC.FIDUCIAL_INSET
    w, h = BC.CANVAS_SIZE
    expected = {
        "tl": (inset, inset),
        "tr": (w - inset, inset),
        "bl": (inset, h - inset),
        "br": (w - inset, h - inset),
    }
    assert centroids == expected
