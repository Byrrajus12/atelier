"""The Easel interface — the sole contract between the domain-agnostic core and any
painting environment (CLAUDE.md Principle 4).

This interface was *extracted from the Milestone 1 spike* (see ``spike/FINDINGS.md``),
not designed up front. Every element here traces to something the spike concretely
needed to drive a real browser canvas vision-only:

  - it captured the screen and had to locate/crop the canvas      -> ``capture`` + a
    canvas-space coordinate frame,
  - it moved the real cursor and dragged a stroke                 -> ``apply_stroke``,
  - it selected a paint color by clicking a swatch               -> folded into
    ``apply_stroke`` (the brush carries a *desired* color; realizing it is the
    environment's job),
  - it could only observe results by re-capturing                -> ``apply_stroke``
    returns nothing; the only signal is a fresh ``capture`` (Principle 3),
  - the browser canvas had no undo                                -> ``Capabilities``.

Coordinate frame contract: the core reasons **only** in canvas-space pixels — the
image returned by ``capture`` is the canvas alone, of size ``canvas_size()``, and every
``Point`` in a ``Stroke`` is in that same 0..W × 0..H space. Screen pixels, DPI, display
scaling, window position, and fiducials live entirely behind this interface (they were
the messy part of the spike, and per Principle 2 they must never leak into the core).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import numpy as np

Color = Tuple[int, int, int]  # RGB, each channel 0..255


class Point(NamedTuple):
    """A location in canvas space: canvas pixels, origin at the canvas top-left,
    x rightward and y downward. Floats are allowed (sub-pixel path samples)."""

    x: float
    y: float


class UnsupportedOperation(RuntimeError):
    """Raised when the core asks an Easel for a capability it does not declare
    (e.g. ``undo()`` on an easel whose ``Capabilities.has_undo`` is False)."""


@dataclass(frozen=True)
class Capabilities:
    """What an environment allows, so the core can act more conservatively where
    actions are irreversible or expensive (Principle 7).

    reversible:  whether a laid stroke can be taken back at all (by any means).
    has_undo:    whether the environment exposes an explicit undo the core may call.
    stroke_cost: relative cost of one stroke, ``>= 0``. Higher means the core should
                 be more certain before committing (e.g. verify more, plan larger,
                 lower-risk strokes). ``1.0`` is the neutral reference cost.
    """

    reversible: bool
    has_undo: bool
    stroke_cost: float = 1.0

    def __post_init__(self) -> None:
        if self.stroke_cost < 0:
            raise ValueError("stroke_cost must be non-negative")
        if self.has_undo and not self.reversible:
            raise ValueError("has_undo=True implies reversible=True")


@dataclass(frozen=True)
class BrushSpec:
    """A requested brush. ``color`` is the color the core *wants*; the environment
    realizes it as faithfully as it can (nearest palette swatch, a picker, ...). The
    core does not assume it got exactly this color — it verifies by re-capture.

    ``size`` is the intended stroke width in canvas pixels.
    """

    color: Color
    size: float = 12.0

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("brush size must be positive")
        if len(self.color) != 3 or not all(0 <= c <= 255 for c in self.color):
            raise ValueError("color must be an RGB triple in 0..255")


@dataclass(frozen=True)
class Stroke:
    """One continuous stroke: press at ``path[0]``, drag through the rest with the
    brush down, release at ``path[-1]``. Points are canvas-space (see ``Point``).

    The path is a geometric intent. How it is timed into synthetic cursor samples
    (the browser needed ~30 ms between samples — see FINDINGS) is the environment's
    responsibility, not encoded here.
    """

    path: Tuple[Point, ...]
    brush: BrushSpec

    def __post_init__(self) -> None:
        if len(self.path) < 1:
            raise ValueError("stroke path needs at least one point")


@dataclass
class Frame:
    """A canvas-space observation: ``image`` is an ``HxWx3`` uint8 RGB array showing
    the canvas alone, already localized/rectified out of the raw screen capture.
    ``timestamp`` is a monotonic seconds reading from when it was captured.

    The ``uint8``/RGB/3-channel shape is a *guarantee* every consumer may rely on
    (perception, the planner, the verifier, the dashboard), not merely a convention —
    so it is validated here rather than assumed downstream. The image's *size* is not
    checked against the canvas here (a ``Frame`` does not know the canvas dimensions);
    that agreement is enforced where a frame meets a target, in perception.
    """

    image: np.ndarray
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError("Frame.image must be HxWx3 RGB")
        if self.image.dtype != np.uint8:
            raise ValueError(
                f"Frame.image must be uint8 (0..255), got {self.image.dtype}"
            )

    @property
    def size(self) -> Tuple[int, int]:
        """(width, height) in canvas pixels."""
        return (self.image.shape[1], self.image.shape[0])


class Easel(ABC):
    """The one boundary the core is allowed to cross to reach an environment.

    Implementations drive their environment **only** through screen capture and
    synthetic input (Principle 1). The core depends on this interface and nothing
    else about any environment (Principle 4).
    """

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Declare reversibility / undo / stroke cost. Should be cheap and stable."""

    @abstractmethod
    def canvas_size(self) -> Tuple[int, int]:
        """(width, height) of the canvas coordinate frame, in canvas pixels. All
        captures and all stroke coordinates live in this frame."""

    @abstractmethod
    def capture(self) -> Frame:
        """Grab the current canvas as a canvas-space ``Frame``. This is the only way
        the core learns what is on the canvas; there is no readback of prior actions
        (Principle 3)."""

    @abstractmethod
    def apply_stroke(self, stroke: Stroke) -> None:
        """Realize ``stroke`` on the canvas: select its brush's color as faithfully as
        possible, then lay the stroke via synthetic input. Returns nothing — the core
        observes the effect by calling ``capture`` again."""

    def undo(self) -> None:
        """Reverse the most recent stroke, if supported. Default: unsupported. Only
        call when ``capabilities().has_undo`` is True."""
        raise UnsupportedOperation(
            f"{type(self).__name__} does not support undo"
        )

    def close(self) -> None:
        """Release environment resources (capture handles, launched processes, ...).
        Default: no-op."""

    def __enter__(self) -> "Easel":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
