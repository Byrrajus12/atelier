"""The executor - turns a region-level ``PaintIntent`` into real brush strokes and
paints them through the Easel (Milestone 5).

The planner decides *what/where* (a ``PaintIntent``: a region box + desired color); the
executor decides *how to lay paint* to fill that region and hands finished ``Stroke``s
to the Easel. It fills with simple **scanlines**: independent horizontal passes spaced
from the Easel's realized brush width. There is no cleverness, no human-motion mimicry.

Executor fill contract:
  * The executor emits scanline fills only. Each scanline is a separate horizontal
    ``Stroke``; the broader ``Stroke.path`` contract still accepts arbitrary sampled
    polylines for future planners and for Easels that receive strokes directly.
  * Supported widths are whatever positive widths an Easel realizes. The reference
    browser Easel realizes the discrete presets 4, 12, and 24 canvas px.
  * For a half-open box ``[x0, x1) x [y0, y1)``, scanline endpoints run from ``x0`` to
    ``x1`` so the page's butt line cap covers the full interior with no last-column
    sliver. Consecutive scanline centers are at most 77.7% of one realized brush width
    apart, the loosest tested global spacing that leaves enough overlap for
    antialiasing and integer cursor realization at 4/12/24 px.
  * There are no vertical edge connectors, so the fill geometry adds no lateral
    connector bleed. The finite brush footprint may still paint beyond the top/bottom
    box edges by at most roughly ``realized_width / 2``.

Boundary: the executor depends ONLY on the ``Easel`` interface (Principles 1 & 4). It
never touches ``pydirectinput`` or the screen directly - synthetic input lives entirely
behind ``apply_stroke``. Everything here is domain-agnostic geometry over a box
(Principle 2). It paints ONE intent; the repeating plan->execute->re-observe loop is the
orchestrator's job (M7), not the executor's.

Brush size is realized through the ``Easel`` interface. The executor asks the Easel what
width will actually land, then spaces scanlines as a fixed ratio of that realized width.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from core.adapter import BrushSpec, Easel, Point, Stroke
from core.planner import PaintIntent

DEFAULT_BRUSH_WIDTH = 12.0  # assumed realized brush width in canvas px (page lineWidth)
DEFAULT_SPACING = 9.324     # gap between scanlines; 77.7% of default width

Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open canvas pixels


def scanline_fill(
    box: Box, spacing: float, brush_width: float
) -> Tuple[Tuple[Point, ...], ...]:
    """Build independent horizontal scanline paths that fill ``box``.

    Scanlines are distributed so consecutive centers are at most ``spacing`` apart, and
    since ``spacing <= brush_width`` the ~brush-wide passes overlap: every canvas row in
    the box is within ``brush_width / 2`` of a scanline (no gap wider than the brush).
    Each path runs from ``x0`` to ``x1``: the box is half-open in pixel ownership, but
    the browser page uses a butt line cap, so ending at the far boundary covers the last
    interior pixel column without creating a round-cap lobe.
    """
    if spacing <= 0 or brush_width <= 0:
        raise ValueError("spacing and brush_width must be > 0")
    if spacing > brush_width:
        # The coverage guarantee above would break: passes would not overlap. Enforce
        # it here so the pure function owns its own contract, not just its callers.
        raise ValueError("spacing must be <= brush_width so passes overlap")
    x0, y0, x1, y1 = box
    xL, xR = x0, max(x0, x1)              # half-open boundary for butt-cap coverage
    top, bottom = y0, max(y0, y1 - 1)     # cover the box's pixel rows

    span = bottom - top
    if span <= 0:
        ys = [float(top)]
    else:
        gaps = math.ceil(span / spacing)
        ys = [top + span * k / gaps for k in range(gaps + 1)]

    paths: List[Tuple[Point, ...]] = []
    for r, y in enumerate(ys):
        left_to_right = (r % 2 == 0)
        a, b = (xL, xR) if left_to_right else (xR, xL)
        paths.append((Point(float(a), float(y)), Point(float(b), float(y))))
    return tuple(paths)


class Executor:
    """Fills a ``PaintIntent``'s region by scanlining its box and painting independent
    horizontal ``Stroke``s through the Easel. Returns the strokes it applied (for the
    orchestrator's event stream and for tests)."""

    def __init__(
        self,
        easel: Easel,
        spacing_ratio: float = DEFAULT_SPACING / DEFAULT_BRUSH_WIDTH,
    ):
        if spacing_ratio <= 0 or spacing_ratio > 1:
            raise ValueError("spacing_ratio must be > 0 and <= 1")
        self._easel = easel
        self.spacing_ratio = spacing_ratio

    def execute(self, intent: PaintIntent) -> List[Stroke]:
        """Paint ``intent``'s region and return the ``Stroke``(s) applied."""
        brush_width = self._easel.realizable_width(intent.size)
        spacing = brush_width * self.spacing_ratio
        raw_paths = scanline_fill(intent.box, spacing, brush_width)
        brush = BrushSpec(color=intent.color, size=intent.size)
        strokes: List[Stroke] = []
        for raw in raw_paths:
            stroke = Stroke(path=raw, brush=brush)
            strokes.append(stroke)
        self._easel.apply_strokes(strokes)
        return strokes
