"""Headless tests for easels/_geometry.py — fiducial localization and the
screen<->canvas homography, exercised on a synthetic screenshot (no browser)."""

import numpy as np
import pytest

from easels import _geometry as G

FIDUCIALS = {
    "tl": (255, 0, 255),   # magenta
    "tr": (0, 255, 255),   # cyan
    "bl": (255, 255, 0),   # yellow
    "br": (0, 255, 0),     # green
}

# Screen-pixel centroids we place the fiducials at (axis-aligned rectangle).
CENTROIDS = {"tl": (100, 50), "tr": (500, 50), "bl": (100, 450), "br": (500, 450)}
CANVAS = (600, 600)
INSET = 9.0


def make_screen():
    """A black 'screen' with a filled fiducial square at each centroid."""
    img = np.zeros((520, 640, 3), dtype=np.uint8)
    half = 8
    for name, (cx, cy) in CENTROIDS.items():
        color = FIDUCIALS[name]
        # Symmetric (odd-width) square so the centroid is exactly (cx, cy).
        img[cy - half:cy + half + 1, cx - half:cx + half + 1] = color
    return img


def test_find_color_centroid_locates_square():
    img = make_screen()
    hit = G.find_color_centroid(img, (255, 0, 255))  # magenta at TL
    assert hit is not None
    x, y, area = hit
    assert x == pytest.approx(100, abs=1) and y == pytest.approx(50, abs=1)
    assert area > 100


def test_find_color_centroid_missing_returns_none():
    img = make_screen()
    assert G.find_color_centroid(img, (123, 45, 67)) is None


def test_find_fiducials_all_present():
    corners = G.find_fiducials(make_screen(), FIDUCIALS)
    for name, (cx, cy) in CENTROIDS.items():
        assert corners[name][0] == pytest.approx(cx, abs=1)
        assert corners[name][1] == pytest.approx(cy, abs=1)


def test_find_fiducials_raises_when_missing():
    img = make_screen()
    img[img[:, :, 1] == 255] = 0  # wipe green/cyan/yellow channels' pixels crudely
    with pytest.raises(LookupError):
        G.find_fiducials(img, FIDUCIALS)


def test_canvas_to_screen_maps_inset_corners_to_centroids():
    corners = G.find_fiducials(make_screen(), FIDUCIALS)
    _, h_c2s = G.canvas_homographies(corners, CANVAS, INSET)
    w, h = CANVAS
    expect = {
        (INSET, INSET): CENTROIDS["tl"],
        (w - INSET, INSET): CENTROIDS["tr"],
        (INSET, h - INSET): CENTROIDS["bl"],
        (w - INSET, h - INSET): CENTROIDS["br"],
    }
    for canvas_pt, screen_pt in expect.items():
        sx, sy = G.apply_homography(canvas_pt, h_c2s)
        assert sx == pytest.approx(screen_pt[0], abs=0.5)
        assert sy == pytest.approx(screen_pt[1], abs=0.5)


def test_screen_canvas_roundtrip_is_identity():
    corners = G.find_fiducials(make_screen(), FIDUCIALS)
    h_s2c, h_c2s = G.canvas_homographies(corners, CANVAS, INSET)
    for pt in [(60.0, 60.0), (300.0, 123.0), (590.0, 400.0)]:
        screen = G.apply_homography(pt, h_c2s)
        back = G.apply_homography(screen, h_s2c)
        assert back[0] == pytest.approx(pt[0], abs=0.5)
        assert back[1] == pytest.approx(pt[1], abs=0.5)


def test_warp_places_inner_mark_at_expected_canvas_location():
    img = make_screen()
    # A red block on screen, inside the fiducial rectangle.
    img[240:260, 290:310] = (255, 0, 0)
    corners = G.find_fiducials(img, FIDUCIALS)
    h_s2c, _ = G.canvas_homographies(corners, CANVAS, INSET)
    canvas_img = G.warp_to_canvas(img, h_s2c, CANVAS)
    assert canvas_img.shape == (600, 600, 3)
    # Where should screen (300,250) land in canvas space?
    cx, cy = G.apply_homography((300.0, 250.0), h_s2c)
    patch = canvas_img[int(cy) - 5:int(cy) + 5, int(cx) - 5:int(cx) + 5]
    assert patch[:, :, 0].mean() > 180  # red channel high
    assert patch[:, :, 1].mean() < 80 and patch[:, :, 2].mean() < 80


def test_apply_homography_rejects_degenerate():
    h = np.zeros((3, 3), dtype=np.float64)  # maps everything to w=0
    with pytest.raises(ValueError):
        G.apply_homography((1.0, 2.0), h)
