"""The planner — decides WHAT to paint and WHERE, one region at a time (Milestone 4).

The planner is a pure *decider* (CLAUDE.md Principle 6): it reads an ``Observation`` and
emits a single region-level ``PaintIntent`` (which region, what color), or ``None`` when
the canvas has converged. It deliberately does **not** emit ``Stroke`` objects — turning
an intent into strokes is the executor's job (M5). Keeping the planner free of stroke
geometry is what lets a model-backed planner drop into the same seat later without
re-learning how to draw.

It is also stateless and emits **one** intent per call: the orchestrator loops
plan -> execute -> re-observe, handing the planner a fresh ``Observation`` each time, so
every decision is made against freshly perceived pixels (Principle 3).

``GreedyPlanner`` is the classical, model-free baseline: paint the single highest-error
region toward the target's color there. It is dumb on purpose — it exists to prove the
loop closes before any model takes the planner seat.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from core.adapter import Color
from core.perception import Observation, cell_box, color_error, DELTA_E_REF

DEFAULT_ERROR_THRESHOLD = 0.02  # provisional; region error is mean-per-pixel in [0,1]


@dataclass
class PaintIntent:
    """One region-level decision: fill region ``cell`` (bounded by ``box``) toward
    ``color``. This is *what/where*, not *how* — the executor (M5) realizes it as
    strokes.

    cell:  ``(i, j)`` region index — ``i`` = row (canvas y), ``j`` = col (canvas x),
           the same convention as ``Observation.region_error``.
    box:   ``(x0, y0, x1, y1)`` canvas-pixel bounding box, half-open, from
           ``perception.cell_box`` (so it matches the cell the error was measured over).
    color: desired RGB — the target's mean color in this region. The environment
           realizes it as faithfully as it can (nearest palette swatch) and the true
           result is confirmed by re-capture, so it is a request, not a guarantee.
    error: the region error that drove this pick, in ``[0, 1]`` — carried for
           observability (events/dashboard) and to keep a future model planner's stated
           reasoning tied to the value it actually acted on.
    size:  brush-size hint in canvas pixels. Carried for contract completeness; the
           reference executor does not honor it yet (brush size is pinned-unrealized
           until M5), so coverage reasoning should assume the real ~12px fixed width.
    """

    cell: Tuple[int, int]
    box: Tuple[int, int, int, int]
    color: Color
    error: float
    size: float = 12.0


def region_mean_color(image: np.ndarray, box: Tuple[int, int, int, int]) -> Color:
    """Mean RGB of ``image`` inside ``box = (x0, y0, x1, y1)`` (half-open), rounded to
    an integer ``Color``. Used to read the target's color in a region straight from the
    ``Observation`` — perception exposes no target-color grid, so the planner computes
    it itself."""
    x0, y0, x1, y1 = box
    region = image[y0:y1, x0:x1]
    if region.size == 0:
        raise ValueError(f"empty region box {box}")
    mean = region.reshape(-1, 3).mean(axis=0)
    return (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))


class Planner(ABC):
    """The pluggable planner seat (Principle 6). Implementations decide the next paint
    action from a perceived ``Observation`` and nothing else."""

    @abstractmethod
    def plan(self, observation: Observation) -> Optional[PaintIntent]:
        """Return the next ``PaintIntent``, or ``None`` if the canvas has converged
        (no region worth acting on)."""


class GreedyPlanner(Planner):
    """Model-free baseline: each call, pick the single highest-error region and fill it
    toward the target's mean color there. Returns ``None`` once no region's error
    exceeds ``error_threshold``.

    ``palette`` opts in a no-undo safety guard (Principle 7). When the available swatches
    are supplied, the planner skips any region that no swatch can improve — the target
    there is closer to what is already on the canvas than to any reachable color, so
    painting it would only make the canvas worse and cannot be undone. The classic
    failure this prevents: a region whose target is the (unpaintable) white background;
    without white in the palette, blindly filling it with the nearest swatch raises the
    error irreversibly.

    Limits of the guard, stated honestly: it compares only the perceptual **color** term
    of the error metric (a uniform swatch fill has no interior edges, and the metric's
    structural term can't be predicted per-cell without border artifacts). So it removes
    self-damage from color-unpaintable cells like the white gap — completely — but it does
    NOT guarantee the verifier never rejects an edge-dominated region. It is a color-space
    safeguard, not a promise of no rejections.

    With ``palette=None`` (the default) the guard is off and behavior is the plain argmax
    baseline, unchanged."""

    def __init__(
        self,
        error_threshold: float = DEFAULT_ERROR_THRESHOLD,
        palette: Optional[Tuple[Color, ...]] = None,
    ):
        if error_threshold < 0:
            raise ValueError("error_threshold must be >= 0")
        if palette is not None and len(palette) == 0:
            raise ValueError("palette, if given, must be non-empty")
        self.error_threshold = error_threshold
        self.palette = tuple(palette) if palette is not None else None

    def plan(self, observation: Observation) -> Optional[PaintIntent]:
        grid = observation.region_error
        n = grid.shape[0]
        # Walk regions from most to least wrong. With no palette this returns the argmax
        # region (legacy behavior, first-in-row-major tie-break); with a palette it skips
        # regions no swatch can improve and takes the worst improvable one instead.
        # Descending error, stable so an exact tie still breaks first-in-row-major (the
        # orchestrator relies on that determinism). Negate + stable sort, not a reversed
        # ascending sort, which would flip ties to last-in-row-major.
        order = np.argsort(-grid, axis=None, kind="stable")
        for flat in order:
            i, j = np.unravel_index(int(flat), grid.shape)
            i, j = int(i), int(j)
            error = float(grid[i, j])
            if error <= self.error_threshold:
                return None  # this and every remaining region is below threshold
            box = cell_box(i, j, n, observation.frame.size)
            color = region_mean_color(observation.target.image, box)
            if self.palette is not None and not self._swatch_improves(observation, box):
                continue  # no reachable color helps here; painting would self-damage
            return PaintIntent(cell=(i, j), box=box, color=color, error=error)
        return None  # nothing above threshold is worth (or safe) painting

    def _swatch_improves(self, observation: Observation, box: Tuple[int, int, int, int]) -> bool:
        """True if the swatch the easel would actually pick brings this region closer, in
        perceptual color, than the canvas already is. Mirrors the easel's Euclidean-RGB
        nearest-swatch selection so the predicted color is the one that would really land,
        then compares mean CIEDE2000-to-target for the current canvas vs a uniform swatch
        fill (same normalization as the error metric)."""
        x0, y0, x1, y1 = box
        target_patch = observation.target.image[y0:y1, x0:x1]
        canvas_patch = observation.frame.image[y0:y1, x0:x1]
        requested = region_mean_color(observation.target.image, box)
        swatch = self._nearest_swatch(requested)
        fill = np.full(target_patch.shape, swatch, dtype=np.uint8)
        current = float(color_error(canvas_patch, target_patch).mean()) / DELTA_E_REF
        predicted = float(color_error(fill, target_patch).mean()) / DELTA_E_REF
        return predicted < current

    def _nearest_swatch(self, requested: Color) -> Color:
        """Nearest palette color to ``requested`` in Euclidean RGB — the same rule the
        easel uses to realize a requested color, so the guard predicts the color that
        would truly be painted."""
        r = np.array(requested, dtype=float)
        return min(
            self.palette,  # type: ignore[arg-type]  # guarded by caller (palette set)
            key=lambda c: float(np.sum((np.array(c, dtype=float) - r) ** 2)),
        )
