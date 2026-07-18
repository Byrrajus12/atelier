"""Headless tests for easels/browser_canvas.py. The live capture/input paths (mss +
pydirectinput) are stubbed; we test the glue that the core actually depends on:
capability declaration, nearest-palette color realization, canvas localization in
capture, and canvas->screen path mapping in apply_stroke."""

import numpy as np
import pytest

from core.adapter import BrushSpec, Point, Stroke
from easels import _geometry as G
from easels import browser_canvas as BC

FIDUCIALS = BC.FIDUCIAL_COLORS
CENTROIDS = {"tl": (100, 50), "tr": (500, 50), "bl": (100, 450), "br": (500, 450)}
# The swatch must sit where the REAL palette strip maps to for this homography (just
# right of the canvas), since swatch search is now restricted to that strip. This is the
# red swatch's canvas-space center (~679, -11) mapped through the CENTROIDS homography.
RED_SWATCH_XY = (541, 53)
# Width buttons live in the next projected control strip to the right of the palette.
WIDTH_BUTTON_XY = {
    "thin": (584, 53),
    "medium": (584, 87),
    "thick": (584, 121),
}


def make_screen(paint_canvas_red=False):
    img = np.zeros((520, 640, 3), dtype=np.uint8)
    half = 8
    for name, (cx, cy) in CENTROIDS.items():
        img[cy - half:cy + half + 1, cx - half:cx + half + 1] = FIDUCIALS[name]
    # Red palette swatch OUTSIDE the canvas (to the right).
    sx, sy = RED_SWATCH_XY
    img[sy - 12:sy + 13, sx - 12:sx + 13] = (255, 0, 0)
    # Width preset buttons OUTSIDE the canvas, in their own restricted search strip.
    for preset in BC.WIDTH_PRESETS:
        wx, wy = WIDTH_BUTTON_XY[preset.name]
        img[wy - 12:wy + 13, wx - 12:wx + 13] = preset.locator_color
    if paint_canvas_red:
        # A big red blob INSIDE the canvas — must not be mistaken for the swatch.
        img[200:300, 200:400] = (255, 0, 0)
    return img


def make_easel():
    return BC.BrowserCanvasEasel(launch=False)


# --- pure helpers ----------------------------------------------------------------
@pytest.mark.parametrize("requested,expected", [
    ((250, 10, 10), (255, 0, 0)),     # near red
    ((10, 10, 240), (0, 0, 255)),     # near blue
    ((30, 30, 30), (17, 17, 17)),     # near black
])
def test_nearest_palette_color(requested, expected):
    assert BC.nearest_palette_color(requested) == expected


@pytest.mark.parametrize("requested,expected", [
    (1.0, "thin"),
    (10.0, "medium"),
    (18.1, "thick"),
])
def test_nearest_width_preset(requested, expected):
    assert BC.nearest_width_preset(requested).name == expected


def test_capabilities_declare_no_undo():
    caps = make_easel().capabilities()
    assert caps.reversible is False
    assert caps.has_undo is False
    assert caps.stroke_cost == 1.0


def test_canvas_size_default():
    assert make_easel().canvas_size() == (600, 600)


def test_realizable_width_uses_nearest_preset():
    assert make_easel().realizable_width(18.1) == 24.0


# --- capture localizes the canvas ------------------------------------------------
def test_capture_returns_canvas_space_frame():
    e = make_easel()
    screen = make_screen()
    screen[240:260, 290:310] = (0, 200, 0)  # a mark inside the canvas
    e._grab_screen = lambda: screen  # stub the live capture

    frame = e.capture()
    assert frame.size == (600, 600)
    # The interior mark should appear somewhere near the canvas center after warping.
    corners = G.find_fiducials(screen, FIDUCIALS)
    h_s2c, _ = G.canvas_homographies(corners, (600, 600), BC.FIDUCIAL_INSET)
    cx, cy = G.apply_homography((300.0, 250.0), h_s2c)
    patch = frame.image[int(cy) - 5:int(cy) + 5, int(cx) - 5:int(cx) + 5]
    assert patch[:, :, 1].mean() > 120  # green channel present


# --- apply_stroke: width/color selection + path mapping --------------------------
def test_apply_stroke_selects_width_and_swatch_and_maps_path():
    e = make_easel()
    screen = make_screen()
    e._grab_screen = lambda: screen

    clicks, drags = [], []
    e._click = lambda x, y: clicks.append((x, y))
    e._drag = lambda pts: drags.append(pts)

    path = (Point(300, 300), Point(400, 350))
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(240, 5, 5), size=10)))

    # Clicked the medium width button (nearest preset to the requested size).
    assert len(clicks) == 2
    assert clicks[0][0] == pytest.approx(WIDTH_BUTTON_XY["medium"][0], abs=2)
    assert clicks[0][1] == pytest.approx(WIDTH_BUTTON_XY["medium"][1], abs=2)

    # Clicked the red swatch (nearest palette color to the requested near-red).
    assert clicks[1][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert clicks[1][1] == pytest.approx(RED_SWATCH_XY[1], abs=2)

    # Dragged the canvas path mapped through the canvas->screen homography.
    corners = G.find_fiducials(screen, FIDUCIALS)
    _, h_c2s = G.canvas_homographies(corners, (600, 600), BC.FIDUCIAL_INSET)
    expected = [tuple(int(v) for v in G.apply_homography((p.x, p.y), h_c2s)) for p in path]
    assert drags == [expected]


def test_apply_stroke_maps_non_rectilinear_path_sequence():
    """The Easel accepts arbitrary sampled polylines; scanlines are only what the
    current executor emits."""
    e = make_easel()
    screen = make_screen()
    e._grab_screen = lambda: screen
    e._click = lambda x, y: None
    drags = []
    e._drag = lambda pts: drags.append(pts)

    path = (
        Point(120, 180),
        Point(180, 225),
        Point(145, 310),
        Point(260, 285),
    )
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(255, 0, 0), size=12)))

    corners = G.find_fiducials(screen, FIDUCIALS)
    _, h_c2s = G.canvas_homographies(corners, (600, 600), BC.FIDUCIAL_INSET)
    expected = [tuple(int(v) for v in G.apply_homography((p.x, p.y), h_c2s)) for p in path]
    assert drags == [expected]


def test_apply_strokes_selects_controls_once_then_drags_all_paths():
    e = make_easel()
    screen = make_screen()
    e._grab_screen = lambda: screen

    clicks, drags = [], []
    e._click = lambda x, y: clicks.append((x, y))
    e._drag = lambda pts: drags.append(pts)

    strokes = (
        Stroke(path=(Point(200, 200), Point(400, 200)), brush=BrushSpec(color=(255, 0, 0), size=12)),
        Stroke(path=(Point(400, 212), Point(200, 212)), brush=BrushSpec(color=(255, 0, 0), size=12)),
    )
    e.apply_strokes(strokes)

    assert len(clicks) == 2
    assert clicks[0][0] == pytest.approx(WIDTH_BUTTON_XY["medium"][0], abs=2)
    assert clicks[1][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert len(drags) == 2


def test_apply_strokes_rejects_mismatched_brush_batch():
    e = make_easel()
    strokes = (
        Stroke(path=(Point(200, 200), Point(400, 200)), brush=BrushSpec(color=(255, 0, 0), size=12)),
        Stroke(path=(Point(400, 212), Point(200, 212)), brush=BrushSpec(color=(0, 0, 255), size=12)),
    )

    with pytest.raises(ValueError):
        e.apply_strokes(strokes)


def test_apply_stroke_ignores_painted_pixels_when_locating_swatch():
    """Even with a big red blob painted on the canvas, the swatch outside the canvas
    is the one clicked (canvas region is masked during swatch search)."""
    e = make_easel()
    screen = make_screen(paint_canvas_red=True)
    e._grab_screen = lambda: screen
    clicks = []
    e._click = lambda x, y: clicks.append((x, y))
    e._drag = lambda pts: None

    e.apply_stroke(Stroke(path=(Point(300, 300),), brush=BrushSpec(color=(255, 0, 0))))
    assert clicks[1][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert clicks[1][1] == pytest.approx(RED_SWATCH_XY[1], abs=2)


def test_locate_swatch_restricted_to_palette_strip_ignores_dark_distractor():
    """The near-black swatch failure mode: a large near-black region OUTSIDE the palette
    strip (a dark editor, window chrome, the taskbar) used to win the largest-blob
    contest and steal the click. Restricting the search to the palette strip (derived
    from the canvas location) must ignore it and find the small black swatch."""
    e = make_easel()
    # Grey page background (NOT near-black), fiducials, a small black swatch in the strip.
    screen = np.full((520, 640, 3), 128, dtype=np.uint8)
    half = 8
    for name, (cx, cy) in CENTROIDS.items():
        screen[cy - half:cy + half + 1, cx - half:cx + half + 1] = FIDUCIALS[name]
    black_xy = (541, 120)  # in the palette strip (black is the third swatch, lower down)
    bx, by = black_xy
    screen[by - 12:by + 13, bx - 12:bx + 13] = (17, 17, 17)
    # A big near-black distractor far outside the palette strip (e.g. a dark editor),
    # clear of the corner fiducials.
    screen[240:360, 150:400] = (10, 10, 10)

    corners = G.find_fiducials(screen, FIDUCIALS)
    _, h_c2s = G.canvas_homographies(corners, (600, 600), BC.FIDUCIAL_INSET)
    loc = e._locate_swatch(screen, h_c2s, (17, 17, 17))
    assert loc is not None
    assert loc[0] == pytest.approx(bx, abs=3)  # the swatch, not the distractor (~x=160)
    assert loc[1] == pytest.approx(by, abs=3)


def test_apply_stroke_raises_when_swatch_missing():
    """A missing swatch is a hard failure (F4): the Easel must not silently paint on
    with an unknown active color. It raises rather than guessing."""
    e = make_easel()
    screen = make_screen()
    sx, sy = RED_SWATCH_XY
    screen[sy - 14:sy + 15, sx - 14:sx + 15] = 0  # erase the only red swatch
    e._grab_screen = lambda: screen
    e._click = lambda x, y: None
    e._drag = lambda pts: None

    with pytest.raises(LookupError):
        e.apply_stroke(Stroke(path=(Point(300, 300),), brush=BrushSpec(color=(255, 0, 0))))


def test_apply_stroke_raises_when_width_button_missing():
    """A missing width button is a hard failure: the Easel must not silently paint on
    with an unknown active width."""
    e = make_easel()
    screen = make_screen()
    wx, wy = WIDTH_BUTTON_XY["thick"]
    screen[wy - 14:wy + 15, wx - 14:wx + 15] = 0  # erase the thick width button
    e._grab_screen = lambda: screen
    e._click = lambda x, y: None
    e._drag = lambda pts: None

    with pytest.raises(LookupError):
        e.apply_stroke(
            Stroke(path=(Point(300, 300),), brush=BrushSpec(color=(255, 0, 0), size=24))
        )


def test_brush_size_selects_different_width_presets_without_changing_path():
    """BrushSpec.size now selects the nearest width preset before the same canvas path
    is dragged."""
    e = make_easel()
    screen = make_screen()
    e._grab_screen = lambda: screen
    clicks = []
    e._click = lambda x, y: clicks.append((x, y))
    drags = []
    e._drag = lambda pts: drags.append(pts)

    path = (Point(200, 200), Point(400, 400))
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(255, 0, 0), size=4)))
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(255, 0, 0), size=24)))

    assert clicks[0][0] == pytest.approx(WIDTH_BUTTON_XY["thin"][0], abs=2)
    assert clicks[0][1] == pytest.approx(WIDTH_BUTTON_XY["thin"][1], abs=2)
    assert clicks[2][0] == pytest.approx(WIDTH_BUTTON_XY["thick"][0], abs=2)
    assert clicks[2][1] == pytest.approx(WIDTH_BUTTON_XY["thick"][1], abs=2)
    assert clicks[1][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert clicks[3][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert drags[0] == drags[1]
