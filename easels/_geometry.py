"""Pure geometry for the browser-canvas Easel: locate the canvas in a screen capture
by its corner fiducials, and map between screen pixels and canvas pixels.

This is the messy coordinate math the M1 spike surfaced (fiducial centroids, the
half-fiducial inset, device-pixel vs canvas-pixel scaling), factored into pure
functions that need only numpy / scipy / cv2 — so they are unit-testable headlessly,
with no screen, cursor, or browser. Environment constants (which colors mark which
corner, the fiducial inset, the canvas size) are supplied by the caller
(``browser_canvas.py``); nothing here is specific to a particular canvas.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage

Color = Tuple[int, int, int]
PointXY = Tuple[float, float]


def find_color_centroid(
    image: np.ndarray, target: Color, tol: int = 60
) -> Optional[Tuple[float, float, int]]:
    """Centroid ``(x, y, area)`` of the largest connected region whose RGB is within
    L1 distance ``tol`` of ``target``, in image-pixel coords. ``None`` if not found.

    (The spike found fiducials and the palette swatch exactly this way.)
    """
    diff = np.abs(image.astype(np.int16) - np.array(target, dtype=np.int16)).sum(axis=2)
    mask = diff < tol
    if not mask.any():
        return None
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == biggest)
    return float(xs.mean()), float(ys.mean()), int(sizes[biggest - 1])


def find_fiducials(
    image: np.ndarray, fiducial_colors: Dict[str, Color], tol: int = 60
) -> Dict[str, PointXY]:
    """Locate each named fiducial's centroid in ``image``. Raises ``LookupError`` if
    any is missing (the canvas cannot be localized without all four corners)."""
    found: Dict[str, PointXY] = {}
    for name, color in fiducial_colors.items():
        hit = find_color_centroid(image, color, tol=tol)
        if hit is None:
            raise LookupError(f"fiducial {name!r} {color} not found in capture")
        found[name] = (hit[0], hit[1])
    return found


def _quad_arrays(
    corners: Dict[str, PointXY],
    canvas_size: Tuple[int, int],
    inset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build matched source (screen) and destination (canvas) point quads in a fixed
    TL, TR, BL, BR order.

    The destination is inset by ``inset`` canvas pixels on every side, because a
    fiducial's *centroid* sits inside the true canvas corner by half the fiducial's
    size — the exact reason the spike's naive ``span/600`` read 1.457 instead of 1.5.
    """
    w, h = canvas_size
    src = np.array(
        [corners["tl"], corners["tr"], corners["bl"], corners["br"]],
        dtype=np.float32,
    )
    dst = np.array(
        [
            (inset, inset),
            (w - inset, inset),
            (inset, h - inset),
            (w - inset, h - inset),
        ],
        dtype=np.float32,
    )
    return src, dst


def canvas_homographies(
    corners: Dict[str, PointXY],
    canvas_size: Tuple[int, int],
    inset: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(H_screen_to_canvas, H_canvas_to_screen)`` 3x3 homographies from the
    four fiducial centroids. ``corners`` must have keys ``tl, tr, bl, br`` in screen
    pixels."""
    src, dst = _quad_arrays(corners, canvas_size, inset)
    h_s2c = cv2.getPerspectiveTransform(src, dst)
    h_c2s = cv2.getPerspectiveTransform(dst, src)
    return h_s2c, h_c2s


def warp_to_canvas(
    screen_image: np.ndarray,
    h_screen_to_canvas: np.ndarray,
    canvas_size: Tuple[int, int],
) -> np.ndarray:
    """Rectify the canvas region out of a full screen capture into a canvas-space
    ``HxWx3`` image of ``canvas_size`` (width, height)."""
    w, h = canvas_size
    return cv2.warpPerspective(screen_image, h_screen_to_canvas, (w, h))


def apply_homography(point: PointXY, h: np.ndarray) -> PointXY:
    """Map a single point through a 3x3 homography (perspective divide included)."""
    x, y = point
    v = h @ np.array([x, y, 1.0], dtype=np.float64)
    if v[2] == 0:
        raise ValueError("degenerate homography mapping (w == 0)")
    return float(v[0] / v[2]), float(v[1] / v[2])
