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
RED_SWATCH_XY = (560, 90)


def make_screen(paint_canvas_red=False):
    img = np.zeros((520, 640, 3), dtype=np.uint8)
    half = 8
    for name, (cx, cy) in CENTROIDS.items():
        img[cy - half:cy + half + 1, cx - half:cx + half + 1] = FIDUCIALS[name]
    # Red palette swatch OUTSIDE the canvas (to the right).
    sx, sy = RED_SWATCH_XY
    img[sy - 12:sy + 13, sx - 12:sx + 13] = (255, 0, 0)
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


def test_capabilities_declare_no_undo():
    caps = make_easel().capabilities()
    assert caps.reversible is False
    assert caps.has_undo is False
    assert caps.stroke_cost == 1.0


def test_canvas_size_default():
    assert make_easel().canvas_size() == (600, 600)


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


# --- apply_stroke: color selection + path mapping --------------------------------
def test_apply_stroke_selects_swatch_and_maps_path():
    e = make_easel()
    screen = make_screen()
    e._grab_screen = lambda: screen

    clicks, drags = [], []
    e._click = lambda x, y: clicks.append((x, y))
    e._drag = lambda pts: drags.append(pts)

    path = (Point(300, 300), Point(400, 350))
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(240, 5, 5), size=10)))

    # Clicked the red swatch (nearest palette color to the requested near-red).
    assert len(clicks) == 1
    assert clicks[0][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert clicks[0][1] == pytest.approx(RED_SWATCH_XY[1], abs=2)

    # Dragged the canvas path mapped through the canvas->screen homography.
    corners = G.find_fiducials(screen, FIDUCIALS)
    _, h_c2s = G.canvas_homographies(corners, (600, 600), BC.FIDUCIAL_INSET)
    expected = [tuple(int(v) for v in G.apply_homography((p.x, p.y), h_c2s)) for p in path]
    assert drags == [expected]


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
    assert clicks[0][0] == pytest.approx(RED_SWATCH_XY[0], abs=2)
    assert clicks[0][1] == pytest.approx(RED_SWATCH_XY[1], abs=2)


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


def test_brush_size_not_yet_realized():
    """BrushSpec.size is defined in the contract but the reference page hardcodes
    lineWidth=12; realizing variable width is deferred to M5 (F1). Pin that no realized
    stroke depends on brush size, so no core code can assume size takes effect yet."""
    e = make_easel()
    screen = make_screen()
    e._grab_screen = lambda: screen
    e._click = lambda x, y: None
    drags = []
    e._drag = lambda pts: drags.append(pts)

    path = (Point(200, 200), Point(400, 400))
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(255, 0, 0), size=4)))
    e.apply_stroke(Stroke(path=path, brush=BrushSpec(color=(255, 0, 0), size=40)))
    assert drags[0] == drags[1]  # brush size has no effect on the realized stroke (yet)
