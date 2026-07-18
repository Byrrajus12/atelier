"""Stroke pacing — a small, deterministic geometry utility (Milestone 5).

This is emphatically NOT a human-motion model. atelier is a vision-only agent closing
the perceive->paint loop; painting *like a human* would be counterproductive here
(jitter/overshoot inject error into the very signal the verifier drives to zero, and
muddy the honest "operates through pixels" story). So there is no overshoot, jitter,
micro-pause, or velocity shaping anywhere in this module.

Its only job is pacing by point placement. The easel drags at a fixed ~30 ms per path
sample (unchanged, and the ``Stroke`` contract is not extended), so a path's *density*
alone controls how the browser renders it:

  * enough points that consecutive samples are close => a connected stroke, no gaps;
  * ``max_step_px`` is the single watchability knob — smaller means more points, which
    on the fixed cadence means a slower, more watchable stroke (points-per-canvas-pixel
    expressed as pixels-between-points).

``densify`` inserts points on the straight segments between the path's existing
vertices; every original vertex (and both endpoints) is preserved exactly, so the
geometric intent is unchanged — only the sampling gets denser.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from core.adapter import Point

DEFAULT_MAX_STEP_PX = 7.0  # max canvas px between consecutive samples (watchability knob)


def densify(
    path: Sequence[Point], max_step_px: float = DEFAULT_MAX_STEP_PX
) -> Tuple[Point, ...]:
    """Return ``path`` with points inserted so no two consecutive points are more than
    ``max_step_px`` apart, by linear interpolation along each segment.

    Endpoints and every original vertex are preserved *exactly* (no drift); inserted
    points lie exactly on the original polyline. A path of 0 or 1 points is returned
    unchanged (a single point is a dab)."""
    if max_step_px <= 0:
        raise ValueError("max_step_px must be > 0")
    pts = [Point(float(p.x), float(p.y)) for p in path]
    if len(pts) <= 1:
        return tuple(pts)

    out = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        dx, dy = b.x - a.x, b.y - a.y
        dist = math.hypot(dx, dy)
        if dist == 0.0:
            continue  # duplicate vertex: nothing to interpolate, stay connected
        segments = max(1, math.ceil(dist / max_step_px))
        for i in range(1, segments + 1):
            if i == segments:
                out.append(Point(b.x, b.y))  # land the vertex exactly — no drift
            else:
                t = i / segments
                out.append(Point(a.x + dx * t, a.y + dy * t))
    return tuple(out)
