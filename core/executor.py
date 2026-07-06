"""The executor — turns a region-level ``PaintIntent`` into real brush strokes and
paints them through the Easel (Milestone 5).

The planner decides *what/where* (a ``PaintIntent``: a region box + desired color); the
executor decides *how to lay paint* to fill that region and hands finished ``Stroke``s
to the Easel. It fills with a simple **scanline** (back-and-forth horizontal passes) —
no cleverness, no human-motion mimicry — spaced so the real ~12 px brush overlaps and
leaves no gap. Passes deliberately run edge-to-edge and let the round brush bleed ~half
a brush-width past the box: regions tile the whole canvas, so insetting would leave an
unpaintable seam at every border, whereas bleed just overpaints a neighbor that the
planner repaints later (recoverable, unlike seams).

Boundary: the executor depends ONLY on the ``Easel`` interface (Principles 1 & 4). It
never touches ``pydirectinput`` or the screen directly — synthetic input lives entirely
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
from core.motion import DEFAULT_MAX_STEP_PX, densify
from core.planner import PaintIntent

DEFAULT_BRUSH_WIDTH = 12.0  # assumed realized brush width in canvas px (page lineWidth)
DEFAULT_SPACING = 10.0      # gap between scanlines; < brush width so passes overlap

Box = Tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open canvas pixels


def scanline_fill(box: Box, spacing: float, brush_width: float) -> Tuple[Point, ...]:
    """Build one connected serpentine path that fills ``box`` with horizontal passes
    ``spacing`` apart. Even rows go left->right, odd rows right->left, joined by short
    vertical connectors at the box edges — a single continuous stroke.

    Scanlines are distributed so consecutive centers are at most ``spacing`` apart, and
    since ``spacing <= brush_width`` the ~brush-wide passes overlap: every canvas row in
    the box is within ``brush_width / 2`` of a scanline (no gap wider than the brush)."""
    if spacing <= 0 or brush_width <= 0:
        raise ValueError("spacing and brush_width must be > 0")
    if spacing > brush_width:
        # The coverage guarantee above would break — passes wouldn't overlap. Enforce
        # it here so the pure function owns its own contract, not just its callers.
        raise ValueError("spacing must be <= brush_width so passes overlap")
    x0, y0, x1, y1 = box
    xL, xR = x0, max(x0, x1 - 1)          # cover the box's pixel columns
    top, bottom = y0, max(y0, y1 - 1)     # ...and its pixel rows

    span = bottom - top
    if span <= 0:
        ys = [float(top)]
    else:
        gaps = math.ceil(span / spacing)
        ys = [top + span * k / gaps for k in range(gaps + 1)]

    path: List[Point] = []
    for r, y in enumerate(ys):
        left_to_right = (r % 2 == 0)
        a, b = (xL, xR) if left_to_right else (xR, xL)
        path.append(Point(float(a), float(y)))
        path.append(Point(float(b), float(y)))
    return tuple(path)


class Executor:
    """Fills a ``PaintIntent``'s region by scanlining its box and painting one serpentine
    ``Stroke`` through the Easel. Returns the strokes it applied (for the orchestrator's
    event stream and for tests)."""

    def __init__(
        self,
        easel: Easel,
        spacing_ratio: float = DEFAULT_SPACING / DEFAULT_BRUSH_WIDTH,
        max_step_px: float = DEFAULT_MAX_STEP_PX,
    ):
        if spacing_ratio <= 0 or spacing_ratio > 1:
            raise ValueError("spacing_ratio must be > 0 and <= 1")
        if max_step_px <= 0:
            raise ValueError("max_step_px must be > 0")
        self._easel = easel
        self.spacing_ratio = spacing_ratio
        self.max_step_px = max_step_px

    def execute(self, intent: PaintIntent) -> List[Stroke]:
        """Paint ``intent``'s region and return the ``Stroke``(s) applied."""
        brush_width = self._easel.realizable_width(intent.size)
        spacing = brush_width * self.spacing_ratio
        raw = scanline_fill(intent.box, spacing, brush_width)
        path = densify(raw, self.max_step_px)
        stroke = Stroke(path=path, brush=BrushSpec(color=intent.color, size=intent.size))
        self._easel.apply_stroke(stroke)
        return [stroke]
