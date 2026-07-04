"""The Target contract — what the agent is trying to reproduce on the canvas.

A ``Target`` is simply a canvas-sized RGB image: the reference the perception diff
measures the current canvas against. It is one of the three things the core is
defined to reason about (a canvas, a target, and the action interface — CLAUDE.md
Principle 2), so it lives in ``core/`` and carries no environment specifics.

Deliberately minimal: this module defines the *contract* (a uint8 RGB ``HxWx3``
image) plus a trivial loader for tests and demos. Target *generation* and any
file-format handling beyond a single convenience loader are out of scope for the
core (see CLAUDE.md Scope).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Size = Tuple[int, int]  # (width, height), matching Frame.size / Easel.canvas_size()


@dataclass(frozen=True)
class Target:
    """The reference image to reproduce. ``image`` is an ``HxWx3`` uint8 RGB array,
    the same size the canvas is captured at. Perception compares a captured ``Frame``
    against this, so the two must share dimensions (checked at the perception
    boundary, since a ``Target`` on its own does not know the canvas size)."""

    image: np.ndarray

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Target.image must be HxWx3 RGB")
        if self.image.dtype != np.uint8:
            raise ValueError(
                f"Target.image must be uint8 (0..255), got {self.image.dtype}"
            )

    @property
    def size(self) -> Size:
        """(width, height) in canvas pixels."""
        return (self.image.shape[1], self.image.shape[0])


def load_target(path: str, size: Size = (600, 600)) -> Target:
    """Load an image file and fit it to the canvas as a ``Target`` (uint8 RGB, resized
    to ``size``). A convenience for tests and demos — not the core's job to source
    targets. Uses OpenCV for decode/resize, converting BGR->RGB.

    ``size`` is (width, height), matching the canvas convention.
    """
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)  # 3-channel BGR uint8, or None on failure
    if bgr is None:
        raise FileNotFoundError(f"could not read target image: {path!r}")
    w, h = size
    resized = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    rgb = resized[:, :, ::-1].copy()  # BGR -> RGB, contiguous
    return Target(image=np.ascontiguousarray(rgb, dtype=np.uint8))
