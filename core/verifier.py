"""The verifier — judges whether a stroke HELPED (CLAUDE.md Principle 3: every action
is verified against the target before the next is chosen).

Shape: a **pure judge**. Given the ``Observation`` from *before* a stroke and the one
from *after*, it decides whether that stroke reduced the error of the region it was
aimed at, and returns a ``Verdict``. It does **not** capture or observe — the
orchestrator (M7) owns all capture and feeds both Observations in — and it does **not**
act on its own verdict: undoing or re-planning around a bad stroke is the orchestrator's
job at the Easel boundary. On the reference easel (``reversible=False, has_undo=False``)
a bad stroke can be *detected* here but not taken back; detection still lets the loop
re-plan rather than paint on blindly.

Because it only reads two Observations and a ``PaintIntent``, the verifier is pure,
domain-agnostic core (Principle 2) and unit-testable with hand-built error grids — no
easel, no screen capture, no timing. It mirrors the planner's structure: a small class
holding one knob, with a single method that reads a perceived state and decides.

What it judges: the **targeted region only**. A stroke's job is to reduce the error of
its own region (``intent.cell``), so it is judged against exactly that region's error,
read from the freshly-observed grids — not against ``intent.error`` (the planner's prior
read) and not against global error. The executor's fill deliberately bleeds ~half a
brush-width into neighbors; that cross-region effect is accepted as self-correcting (the
neighbor is repainted with its own color when the planner next picks it), so it must not
count against this stroke. Global before/after IS carried in the ``Verdict`` for
observability (so the orchestrator/dashboard can watch overall convergence), but it does
not gate the accept — whole-canvas progress is a loop-level concern (M7), not a
per-stroke judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from core.perception import Observation
from core.planner import PaintIntent

# Provisional, like the planner's seam-tolerant DEFAULT_ERROR_THRESHOLD (0.08): region error is
# mean-per-pixel in [0, 1], so this is the smallest drop that reads as a real
# improvement rather than capture / anti-aliasing noise. Tunable via the constructor.
DEFAULT_IMPROVEMENT_THRESHOLD = 0.005


@dataclass(frozen=True)
class Verdict:
    """The result of judging one stroke. Pure, immutable data (no narration) — enough
    for the orchestrator to re-plan and for the dashboard to show *why* a stroke was
    kept or rejected.

    accepted:      did the stroke help — did the targeted region's error drop by at
                   least the verifier's improvement threshold. This is the only field
                   the accept rule is computed from; everything below is context.
    cell:          ``(i, j)`` region that was judged (``intent.cell``) — i = row
                   (canvas y), j = col (canvas x), the ``Observation.region_error``
                   convention. Says *what* was judged.
    region_before / region_after:
                   the targeted region's mean error before and after the stroke, each
                   in ``[0, 1]``.
    region_delta:  ``region_before - region_after`` — positive means improvement. The
                   accept rule is ``region_delta >= improvement_threshold``.
    global_before / global_after / global_delta:
                   whole-canvas mean error before/after and their difference (same
                   positive-means-improvement sign). Carried for observability only —
                   it does NOT gate ``accepted`` (see module docstring).
    """

    accepted: bool
    cell: Tuple[int, int]
    region_before: float
    region_after: float
    region_delta: float
    global_before: float
    global_after: float
    global_delta: float


class Verifier:
    """Judges whether a stroke helped its targeted region. Stateless apart from the
    improvement threshold; one ``verify`` call per stroke, reading only the before/after
    Observations and the intent."""

    def __init__(self, improvement_threshold: float = DEFAULT_IMPROVEMENT_THRESHOLD):
        if improvement_threshold < 0:
            raise ValueError("improvement_threshold must be >= 0")
        self.improvement_threshold = improvement_threshold

    def verify(
        self,
        before: Observation,
        after: Observation,
        intent: PaintIntent,
    ) -> Verdict:
        """Judge the stroke that turned ``before`` into ``after`` while aiming at
        ``intent``'s region. Returns a ``Verdict``.

        The judgment reads the targeted region's error from BOTH freshly-observed grids
        at ``intent.cell`` (the Observation is the source of truth, not ``intent.error``)
        and accepts when that region dropped by at least the improvement threshold.
        Global error is carried through but does not affect the accept.
        """
        if before.region_error.shape != after.region_error.shape:
            raise ValueError(
                f"before/after region grids must match: "
                f"{before.region_error.shape} != {after.region_error.shape}"
            )
        i, j = intent.cell
        n_rows, n_cols = before.region_error.shape
        if not (0 <= i < n_rows and 0 <= j < n_cols):
            raise IndexError(
                f"intent.cell {intent.cell} out of range for a "
                f"{n_rows}x{n_cols} region grid"
            )

        region_before = float(before.region_error[i, j])
        region_after = float(after.region_error[i, j])
        region_delta = region_before - region_after  # positive = improvement

        global_before = float(before.global_error)
        global_after = float(after.global_error)

        return Verdict(
            accepted=region_delta >= self.improvement_threshold,
            cell=(i, j),
            region_before=region_before,
            region_after=region_after,
            region_delta=region_delta,
            global_before=global_before,
            global_after=global_after,
            global_delta=global_before - global_after,
        )
