"""Headless executor fill proxy: quantify visible coverage and neighbor bleed.

This is a proxy, not the decision gate. The ground truth is the live browser render
(``scripts/live_run.py``, run by hand). This script exists to give a quick, faithful-ish
read on the executor geometry before live confirmation.

Raster model:
  * The reference page paints opaque HTML canvas strokes with ``lineCap='butt'``.
  * The Easel ultimately sends integer screen cursor coordinates. Near the reference
    page's 1:1 canvas scale, that means each independent scanline center can realize
    about half a canvas pixel away from the requested float coordinate. Adjacent
    scanlines can drift in opposite directions, so the proxy applies a worst-case
    +/-0.5 px vertical realization offset to alternating scanlines.
  * Browser antialiasing means a pixel can be partly painted and still look like a
    pale seam over the white canvas. This proxy supersamples stroke geometry and
    downsamples to per-pixel paint alpha instead of using a boolean touched/not-touched
    test.
  * ``VISIBLE_ALPHA`` is 0.95: below that, an opaque red/blue/black stroke over white
    leaves at least about 13/255 levels of white contamination in a saturated channel,
    which is visible as a light seam. This threshold is a visibility rule, not a tuning
    knob for making the test pass.

Both the CLI report and the tests use production scanline paths plus ``densify``. The
current executor emits independent horizontal strokes, so there are no vertical edge
connectors. Under the reference page's butt cap, lateral bleed should be zero while
top/bottom footprint bleed is bounded by roughly half the realized brush width.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adapter import Point
from core.executor import DEFAULT_BRUSH_WIDTH, DEFAULT_SPACING, scanline_fill
from core.motion import DEFAULT_MAX_STEP_PX, densify

Box = Tuple[int, int, int, int]
VISIBLE_ALPHA = 0.95
SUPERSAMPLE = 8
REALIZATION_TOLERANCE_PX = 0.5


def _rasterize_path_highres(
    canvas: np.ndarray,
    path: Tuple[Point, ...],
    brush_width: float,
    cap: str,
    box: Box,
    pad: int,
    supersample: int,
) -> None:
    """Paint one path into a high-resolution boolean coverage canvas."""
    x0, y0, x1, y1 = box
    r = brush_width / 2.0
    gx0, gy0 = x0 - pad, y0 - pad
    pts = [(p.x, p.y) for p in path]

    def paint_disk(cx: float, cy: float) -> None:
        ix0 = max(0, int(np.floor((cx - r - gx0) * supersample)))
        ix1 = min(canvas.shape[1], int(np.ceil((cx + r - gx0) * supersample)))
        iy0 = max(0, int(np.floor((cy - r - gy0) * supersample)))
        iy1 = min(canvas.shape[0], int(np.ceil((cy + r - gy0) * supersample)))
        if ix1 <= ix0 or iy1 <= iy0:
            return
        xs = gx0 + (np.arange(ix0, ix1) + 0.5) / supersample
        ys = gy0 + (np.arange(iy0, iy1) + 0.5) / supersample
        gx, gy = np.meshgrid(xs, ys)
        canvas[iy0:iy1, ix0:ix1] |= ((gx - cx) ** 2 + (gy - cy) ** 2) <= r * r

    if len(pts) == 1:
        paint_disk(*pts[0])
        return

    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0.0:
            if cap == "round":
                paint_disk(ax, ay)
            continue

        sx0 = min(ax, bx) - r
        sx1 = max(ax, bx) + r
        sy0 = min(ay, by) - r
        sy1 = max(ay, by) + r
        ix0 = max(0, int(np.floor((sx0 - gx0) * supersample)))
        ix1 = min(canvas.shape[1], int(np.ceil((sx1 - gx0) * supersample)))
        iy0 = max(0, int(np.floor((sy0 - gy0) * supersample)))
        iy1 = min(canvas.shape[0], int(np.ceil((sy1 - gy0) * supersample)))
        if ix1 <= ix0 or iy1 <= iy0:
            continue

        xs = gx0 + (np.arange(ix0, ix1) + 0.5) / supersample
        ys = gy0 + (np.arange(iy0, iy1) + 0.5) / supersample
        gx, gy = np.meshgrid(xs, ys)
        t = ((gx - ax) * dx + (gy - ay) * dy) / seg2
        if cap == "butt":
            inside = (t >= 0.0) & (t <= 1.0)
        else:
            inside = np.ones_like(t, dtype=bool)
        tc = np.clip(t, 0.0, 1.0)
        px = ax + tc * dx
        py = ay + tc * dy
        dist2 = (gx - px) ** 2 + (gy - py) ** 2
        canvas[iy0:iy1, ix0:ix1] |= inside & (dist2 <= r * r)


def _rasterize_alpha(
    paths: Iterable[Tuple[Point, ...]],
    brush_width: float,
    cap: str,
    box: Box,
    pad: int,
    supersample: int = SUPERSAMPLE,
) -> np.ndarray:
    """Return per-pixel paint alpha for all stroked ``paths`` over expanded ``box``."""
    x0, y0, x1, y1 = box
    h = (y1 + pad) - (y0 - pad)
    w = (x1 + pad) - (x0 - pad)
    high = np.zeros((h * supersample, w * supersample), dtype=bool)
    for path in paths:
        _rasterize_path_highres(high, path, brush_width, cap, box, pad, supersample)
    return high.reshape(h, supersample, w, supersample).mean(axis=(1, 3))


def _realized_paths(
    paths: Iterable[Tuple[Point, ...]],
    tolerance_px: float = REALIZATION_TOLERANCE_PX,
) -> Tuple[Tuple[Point, ...], ...]:
    """Approximate worst-case browser/input realization of independent scanlines.

    The browser sees integer cursor positions, not the executor's ideal float canvas
    centers. Alternating +/- tolerance widens the gap between every other adjacent
    scanline by about one pixel, matching the visible seam risk seen on the live page.
    """
    realized = []
    for i, path in enumerate(paths):
        dy = -tolerance_px if i % 2 == 0 else tolerance_px
        realized.append(tuple(Point(p.x, p.y + dy) for p in path))
    return tuple(realized)


def _visible_mask(alpha: np.ndarray, threshold: float = VISIBLE_ALPHA) -> np.ndarray:
    """Pixels painted strongly enough not to read as pale seams against white."""
    return alpha >= threshold


def _bleed_report(visible: np.ndarray, box: Box, pad: int) -> dict:
    """Outward visible over-paint past each box edge, as pixel count and max depth."""
    x0, y0, x1, y1 = box
    gx0, gy0 = x0 - pad, y0 - pad
    ys_idx, xs_idx = np.nonzero(visible)
    xs = xs_idx + gx0
    ys = ys_idx + gy0

    right = xs >= x1
    left = xs < x0
    bottom = ys >= y1
    top = ys < y0

    def depth(mask, coord, boundary, sign):
        if not mask.any():
            return 0
        return int(np.max(sign * (coord[mask] - boundary)))

    return {
        "right": {"pixels": int(right.sum()),
                  "max_depth": depth(right, xs, x1 - 1, +1)},
        "left": {"pixels": int(left.sum()),
                 "max_depth": depth(left, xs, x0, -1)},
        "bottom": {"pixels": int(bottom.sum()),
                   "max_depth": depth(bottom, ys, y1 - 1, +1)},
        "top": {"pixels": int(top.sum()),
                "max_depth": depth(top, ys, y0, -1)},
    }


def _interior_uncovered(visible: np.ndarray, box: Box, pad: int) -> int:
    x0, y0, x1, y1 = box
    interior = visible[pad:pad + (y1 - y0), pad:pad + (x1 - x0)]
    return int((~interior).sum())


def _min_interior_alpha(alpha: np.ndarray, box: Box, pad: int) -> float:
    x0, y0, x1, y1 = box
    interior = alpha[pad:pad + (y1 - y0), pad:pad + (x1 - x0)]
    return float(interior.min()) if interior.size else 1.0


def run(box: Box, brush_width: float, spacing: float, max_step: float) -> None:
    raw_paths = scanline_fill(box, spacing, brush_width)
    paths = _realized_paths(tuple(densify(path, max_step) for path in raw_paths))
    pad = int(brush_width)

    print(f"box={box}  brush_width={brush_width}  spacing={spacing}  "
          f"visible_alpha={VISIBLE_ALPHA}  supersample={SUPERSAMPLE}  "
          f"realization_tolerance={REALIZATION_TOLERANCE_PX}  "
          f"strokes={len(paths)}  path_points={sum(len(p) for p in paths)}")
    print("visible interior gaps and outward over-paint (pixels / max depth px):")
    print(f"{'cap':>6} | {'gaps':>10} | {'min_alpha':>9} | {'left':>14} | "
          f"{'right':>14} | {'top':>14} | {'bottom':>14}")
    for cap in ("round", "butt"):
        alpha = _rasterize_alpha(paths, brush_width, cap, box, pad)
        visible = _visible_mask(alpha)
        r = _bleed_report(visible, box, pad)

        def cell(side):
            return f"{r[side]['pixels']:>6} / {r[side]['max_depth']:>2}px"

        print(f"{cap:>6} | {_interior_uncovered(visible, box, pad):>10} | "
              f"{_min_interior_alpha(alpha, box, pad):>9.3f} | "
              f"{cell('left'):>14} | {cell('right'):>14} | "
              f"{cell('top'):>14} | {cell('bottom'):>14}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--box", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    default=[200, 200, 400, 400],
                    help="cell box in canvas px (half-open)")
    ap.add_argument("--brush-width", type=float, default=DEFAULT_BRUSH_WIDTH)
    ap.add_argument("--spacing", type=float, default=DEFAULT_SPACING)
    ap.add_argument("--max-step", type=float, default=DEFAULT_MAX_STEP_PX)
    args = ap.parse_args()
    run(tuple(args.box), args.brush_width, args.spacing, args.max_step)


if __name__ == "__main__":
    main()
