"""Pure pre-paint validation for proposed paint intents.

The validator is a no-undo safety gate: it decides whether a proposed cell/color should be
allowed to reach the executor. It is deliberately pure core code: it reads only the current
``Observation``, the ``Target``, and core color/error helpers.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from core.adapter import Color
from core.perception import Observation, cell_box, color_error
from core.planner import (
    DEFAULT_ERROR_THRESHOLD,
    region_mean_color,
    swatch_would_improve,
)
from core.target import Target

COLOR_MISMATCH_THRESHOLD = 30.0


def validate_intent(
    observation: Observation,
    target: Target,
    cell: Tuple[int, int],
    color: Color,
    n: int,
    palette: Tuple[Color, ...],
) -> Tuple[bool, str]:
    """Return whether painting ``cell`` with ``color`` is safe to attempt.

    Checks short-circuit in safety order: avoid already-finished cells, avoid no-undo
    swatch damage, then ensure the requested color actually matches the target cell.
    """
    i, j = cell
    error = float(observation.region_error[i, j])
    if error <= DEFAULT_ERROR_THRESHOLD:
        return (
            False,
            f"cell ({i},{j}) already converged (error={error:.4f}), pick a cell "
            f"that still differs from target",
        )

    box = cell_box(i, j, n, observation.frame.size)
    if not swatch_would_improve(observation, box, palette, color):
        return (
            False,
            f"painting color {tuple(color)} at cell ({i},{j}) would not reduce error - "
            f"check that this color matches the target and is achievable with the "
            f"available palette",
        )

    target_box = cell_box(i, j, n, target.size)
    target_color = region_mean_color(target.image, target_box)
    proposed = np.array([[color]], dtype=np.uint8)
    actual = np.array([[target_color]], dtype=np.uint8)
    distance = float(color_error(proposed, actual)[0, 0])
    if distance > COLOR_MISMATCH_THRESHOLD:
        return (
            False,
            f"proposed color {tuple(color)} doesn't match target color {target_color} "
            f"at cell ({i},{j}) - re-examine the target image",
        )

    return True, ""