"""Headless edge-cleanliness proxy for M7.5 — quantify how much a filled cell
OVER-paints into its neighbors, round brush cap vs butt.

This is a PROXY, not the decision gate. The ground truth is the live browser render
(scripts/live_run.py, run by hand). This script exists only to give a quick, faithful-
ish before/after read on the one quantity tied to the boundary-cell stall: outward
over-paint past the cell box into the adjacent cell.

Why outward, not inward: a filled cell can have perfectly straight edges and still
contaminate its neighbor. The reference page strokes each cursor-sample segment as its
own `beginPath();...;stroke()` (index.html), so the brush cap applies at EVERY segment
end. With a round cap each horizontal scanline end adds a semicircle that bleeds ~T/2
past the box's left/right edge; a butt cap ends those flat. The serpentine's vertical
connectors sit centered on the box edge, so they bleed ~T/2 regardless of cap. This
script measures exactly that bleed, so we can see how much the cap fix removes and what
the connectors leave behind.

Raster model (faithful to per-segment stroking, lineWidth=12):
  * round: union of capsules (disk radius T/2 swept along each densified segment) —
    i.e. round cap + the disk overlap that canvas produces between samples.
  * butt : union of flat-ended rectangles (half-width T/2) per densified segment.
Both use the SAME production path (core scanline_fill + densify), so only the cap model
differs. Distances are computed on an integer canvas grid padded by the brush radius.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.executor import DEFAULT_BRUSH_WIDTH, DEFAULT_SPACING, scanline_fill
from core.motion import DEFAULT_MAX_STEP_PX, densify

Box = Tuple[int, int, int, int]


def _rasterize(path, brush_width: float, cap: str, box: Box, pad: int) -> np.ndarray:
    """Return a boolean coverage grid for the stroked ``path`` under ``cap`` in
    ('round','butt'). The grid spans the box expanded by ``pad`` on every side, so all
    outward bleed is captured. Grid[j, i] is canvas pixel (x0-pad + i, y0-pad + j)."""
    x0, y0, x1, y1 = box
    R = brush_width / 2.0
    gx0, gy0 = x0 - pad, y0 - pad
    W = (x1 + pad) - gx0
    H = (y1 + pad) - gy0

    # Pixel-center coordinate grids in canvas space.
    xs = gx0 + np.arange(W) + 0.5
    ys = gy0 + np.arange(H) + 0.5
    gx, gy = np.meshgrid(xs, ys)  # (H, W)

    covered = np.zeros((H, W), dtype=bool)
    pts = [(p.x, p.y) for p in path]
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        if seg2 == 0.0:
            # Degenerate segment: a single dab. Round -> disk, butt -> nothing extra.
            if cap == "round":
                covered |= ((gx - ax) ** 2 + (gy - ay) ** 2) <= R * R
            continue
        # Projection parameter t of every pixel onto the segment.
        t = ((gx - ax) * dx + (gy - ay) * dy) / seg2
        if cap == "butt":
            # Flat ends: only pixels whose projection lands within the segment.
            inside = (t >= 0.0) & (t <= 1.0)
        else:  # round: clamp the projection so ends round off into disks.
            inside = np.ones_like(t, dtype=bool)
        tc = np.clip(t, 0.0, 1.0)
        px = ax + tc * dx
        py = ay + tc * dy
        dist2 = (gx - px) ** 2 + (gy - py) ** 2
        covered |= inside & (dist2 <= R * R)
    return covered


def _bleed_report(covered: np.ndarray, box: Box, pad: int) -> dict:
    """Outward over-paint past each box edge (into the neighbor cell). Returns painted-
    pixel counts and max bleed depth (px past the boundary) for each side."""
    x0, y0, x1, y1 = box
    gx0, gy0 = x0 - pad, y0 - pad
    ys_idx, xs_idx = np.nonzero(covered)
    xs = xs_idx + gx0
    ys = ys_idx + gy0

    right = xs >= x1          # neighbor to the right starts at x1 (box is half-open)
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


def run(box: Box, brush_width: float, spacing: float, max_step: float) -> None:
    raw = scanline_fill(box, spacing, brush_width)
    path = densify(raw, max_step)
    pad = int(brush_width)  # comfortably larger than the max possible bleed (T/2)

    print(f"box={box}  brush_width={brush_width}  spacing={spacing}  "
          f"path_points={len(path)}")
    print("outward over-paint into the neighbor cell (pixels / max depth px):")
    print(f"{'cap':>6} | {'left':>14} | {'right':>14} | "
          f"{'top':>14} | {'bottom':>14}")
    for cap in ("round", "butt"):
        cov = _rasterize(path, brush_width, cap, box, pad)
        r = _bleed_report(cov, box, pad)
        def cell(side):
            return f"{r[side]['pixels']:>6} / {r[side]['max_depth']:>2}px"
        print(f"{cap:>6} | {cell('left'):>14} | {cell('right'):>14} | "
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
