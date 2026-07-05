"""The orchestrator — the closed loop itself (CLAUDE.md Principle 3). It strings the four
pieces together and drives them until the canvas converges on the target, emitting a
typed event stream as it goes.

One iteration, and the single load-bearing dataflow decision (ONE capture+observe per
stroke, not two):

    before = observe(capture())          # baseline, done once before the loop
    intent = planner.plan(before)        # what/where, or None -> done
    executor.execute(intent)             # lay the strokes through the easel
    after  = observe(capture())          # perceive the result
    verdict = verifier.verify(before, after, intent)
    ... emit events ...
    before = after                       # the result becomes the next baseline

The orchestrator knows nothing about any UI: it emits to an abstract ``EventSink`` and
lets consumers listen (Principle 5). It depends on the pluggable ``Planner`` interface,
not any concrete planner (Principle 6), and it reads the easel's declared
``Capabilities`` for observability. ``emit`` is best-effort — a sink/transport error is
swallowed so a flaky listener can never kill a paint run.

Two policies beyond the bare loop:

  * **Rejected strokes / blacklisting.** A rejected verdict does NOT trigger an in-place
    retry; the loop simply moves on and the greedy planner re-picks whatever region is
    highest next time (possibly the same one). A per-region failure counter guards
    against hammering an unpaintable region forever: after ``max_region_failures``
    rejects, the region is blacklisted. Blacklisting is the orchestrator's policy, kept
    OUT of the pure planner — it is realized by handing the planner a *masked* copy of
    the Observation with blacklisted cells zeroed, so the planner stays unchanged and
    threshold-agnostic (Principle 6).

  * **Termination**, whichever hits first, each ending with a ``RunDone`` stating why:
    ``converged`` (planner returns None on the true, unmasked grid — nothing worth
    painting), ``stalled_no_progress`` (global error has not dropped by more than
    ``progress_epsilon`` for ``max_stall_iterations`` consecutive strokes), ``stalled``
    (the planner would still act, but only on blacklisted regions), ``budget``
    (max-iterations safety cap), or ``canvas_lost`` (capture/stroke kept failing to
    locate the canvas). The unmasked-vs-masked plan comparison distinguishes converged
    from stalled using the planner itself as the threshold oracle, with no new interface.

    ``stalled_no_progress`` is the robust, target-independent stop. Threshold- and
    blacklist-based termination can grind for a long time when the painting is already as
    close as the tools allow: an inherent error floor (round-brush scalloping +
    nearest-palette color approximation) leaves border cells hovering just over the
    planner's error threshold, so the planner keeps re-picking them and each has to
    exhaust its per-region reject quota before it is blacklisted. Watching whole-canvas
    error directly catches that plateau the moment it flattens, on any image, without
    tuning the planner's threshold down onto the floor (which would blind it to genuine
    small errors on other targets).
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Set, Tuple

from core.adapter import Easel
from core.events import (
    REASON_BUDGET,
    REASON_CANVAS_LOST,
    REASON_CONVERGED,
    REASON_STALLED,
    REASON_STALLED_NO_PROGRESS,
    STATUS_DONE,
    STATUS_RUNNING,
    EventSink,
    ExecuteDone,
    FrameCaptured,
    ObserveDone,
    PlanDone,
    RunDone,
    RunStart,
    StateUpdate,
    VerifyDone,
)
from core.executor import Executor
from core.perception import (
    DEFAULT_COLOR_WEIGHT,
    DEFAULT_GRID_N,
    Observation,
    observe,
)
from core.planner import PaintIntent, Planner
from core.target import Target
from core.verifier import Verifier

DEFAULT_MAX_ITERATIONS = 500       # safety cap on strokes; a full loop rarely nears this
DEFAULT_MAX_REGION_FAILURES = 3    # rejects on one region before it is blacklisted
DEFAULT_CAPTURE_RETRIES = 3        # extra attempts if capture/stroke can't find the canvas
DEFAULT_CAPTURE_RETRY_DELAY = 0.5  # seconds between those attempts
# Stall detector (target-independent stop). A genuine full-cell paint drops global error
# by far more than this; floor-grinding after the picture is right leaves it essentially
# flat. So a per-stroke global improvement at or below this epsilon counts as "no
# progress", and this many consecutive such strokes ends the run.
DEFAULT_PROGRESS_EPSILON = 0.001
DEFAULT_MAX_STALL_ITERATIONS = 3

Cell = Tuple[int, int]


class _CanvasLost(Exception):
    """Internal: capture or stroke could not locate the canvas after bounded retries.
    Caught in ``run`` and turned into a clean ``canvas_lost`` termination — never
    propagated to the caller."""


class Orchestrator:
    """Runs the perceive -> plan -> paint -> verify loop to convergence, emitting events.

    The ``executor`` must be built over the SAME ``easel`` passed here (the orchestrator
    captures through ``easel`` and paints through ``executor``); they share one
    environment. Everything else is pure core.
    """

    def __init__(
        self,
        easel: Easel,
        target: Target,
        planner: Planner,
        executor: Executor,
        verifier: Verifier,
        sink: EventSink,
        *,
        grid_n: int = DEFAULT_GRID_N,
        color_weight: float = DEFAULT_COLOR_WEIGHT,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        max_region_failures: int = DEFAULT_MAX_REGION_FAILURES,
        capture_retries: int = DEFAULT_CAPTURE_RETRIES,
        capture_retry_delay: float = DEFAULT_CAPTURE_RETRY_DELAY,
        progress_epsilon: float = DEFAULT_PROGRESS_EPSILON,
        max_stall_iterations: int = DEFAULT_MAX_STALL_ITERATIONS,
    ) -> None:
        if grid_n < 1:
            raise ValueError("grid_n must be >= 1")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if max_region_failures < 1:
            raise ValueError("max_region_failures must be >= 1")
        if capture_retries < 0:
            raise ValueError("capture_retries must be >= 0")
        if capture_retry_delay < 0:
            raise ValueError("capture_retry_delay must be >= 0")
        if progress_epsilon < 0:
            raise ValueError("progress_epsilon must be >= 0")
        if max_stall_iterations < 1:
            raise ValueError("max_stall_iterations must be >= 1")

        self._easel = easel
        self._target = target
        self._planner = planner
        self._executor = executor
        self._verifier = verifier
        self._sink = sink
        self._grid_n = grid_n
        self._color_weight = color_weight
        self._max_iterations = max_iterations
        self._max_region_failures = max_region_failures
        self._capture_retries = capture_retries
        self._capture_retry_delay = capture_retry_delay
        self._progress_epsilon = progress_epsilon
        self._max_stall_iterations = max_stall_iterations

        self._failures: Dict[Cell, int] = {}
        self._blacklist: Set[Cell] = set()
        self._last_global: Optional[float] = None
        self._no_progress = 0  # consecutive strokes with global improvement <= epsilon

    # --- public entry point ------------------------------------------------------
    def run(self) -> RunDone:
        """Drive the loop to termination. Returns the final ``RunDone`` (also emitted),
        whose ``reason`` says why it stopped."""
        caps = self._easel.capabilities()
        self._emit(RunStart(
            canvas_size=self._easel.canvas_size(),
            grid_n=self._grid_n,
            max_iterations=self._max_iterations,
            max_region_failures=self._max_region_failures,
            error_threshold=float(getattr(self._planner, "error_threshold", float("nan"))),
            improvement_threshold=float(self._verifier.improvement_threshold),
            reversible=caps.reversible,
            has_undo=caps.has_undo,
            stroke_cost=caps.stroke_cost,
        ))

        # Baseline perception (iteration 0), before any stroke.
        try:
            before = self._observe(self._capture(), iteration=0)
        except _CanvasLost:
            return self._finish(0, REASON_CANVAS_LOST, converged=False)
        self._emit(StateUpdate(iteration=0, global_error=before.global_error))

        iteration = 0  # strokes completed
        while True:
            # Check "nothing left to do" BEFORE the budget cap. If the canvas converged
            # (or stalled) on exactly the last allowed stroke, that is the true outcome;
            # budget is the safety net for a run that still has work when its allowance
            # runs out, not a pre-emption of an already-finished run.
            intent, stop_reason = self._next_intent(before)
            if intent is None:
                return self._finish(
                    iteration, stop_reason, converged=(stop_reason == REASON_CONVERGED)
                )
            if iteration >= self._max_iterations:
                return self._finish(iteration, REASON_BUDGET, converged=False)

            stroke_no = iteration + 1
            self._emit(PlanDone(iteration=stroke_no, intent=intent))

            try:
                strokes = self._execute(intent)
            except _CanvasLost:
                return self._finish(iteration, REASON_CANVAS_LOST, converged=False)
            self._emit(ExecuteDone(
                iteration=stroke_no, cell=tuple(intent.cell), stroke_count=len(strokes)
            ))

            try:
                after = self._observe(self._capture(), iteration=stroke_no)
            except _CanvasLost:
                return self._finish(iteration, REASON_CANVAS_LOST, converged=False)

            verdict = self._verifier.verify(before, after, intent)
            self._emit(VerifyDone(iteration=stroke_no, verdict=verdict))
            self._record_outcome(tuple(intent.cell), verdict.accepted)
            self._emit(StateUpdate(iteration=stroke_no, global_error=after.global_error))

            # Stall detector: measure whole-canvas progress this stroke (while `before`
            # still holds the prior observation), then advance. `converged` and `stalled`
            # are checked at the top of the loop; this is the plateau stop that fires
            # regardless of thresholds or per-region reject quotas.
            global_improved = before.global_error - after.global_error
            if global_improved <= self._progress_epsilon:
                self._no_progress += 1
            else:
                self._no_progress = 0

            before = after
            iteration = stroke_no

            if self._no_progress >= self._max_stall_iterations:
                return self._finish(iteration, REASON_STALLED_NO_PROGRESS, converged=False)

    # --- intent selection with blacklist masking ---------------------------------
    def _next_intent(
        self, before: Observation
    ) -> Tuple[Optional[PaintIntent], Optional[str]]:
        """Pick the next intent, honoring the blacklist. Returns ``(intent, None)`` to
        act, or ``(None, reason)`` to stop (``converged`` or ``stalled``).

        The unmasked plan is the threshold oracle: if it is None, nothing anywhere is
        worth painting -> converged. If it wants a non-blacklisted region, act on it. If
        it wants a blacklisted region, re-plan on a masked grid; None there means the
        only actionable regions left are blacklisted -> stalled."""
        unmasked = self._planner.plan(before)
        if unmasked is None:
            return None, REASON_CONVERGED
        if tuple(unmasked.cell) not in self._blacklist:
            return unmasked, None
        masked = self._planner.plan(self._masked(before))
        if masked is None:
            return None, REASON_STALLED
        return masked, None

    def _masked(self, before: Observation) -> Observation:
        """A copy of ``before`` with blacklisted cells' error zeroed, so the pure planner
        treats them as already done and never re-picks them."""
        grid = before.region_error.copy()
        for (i, j) in self._blacklist:
            grid[i, j] = 0.0
        return Observation(
            frame=before.frame,
            target=before.target,
            global_error=before.global_error,
            region_error=grid,
            heatmap=before.heatmap,
        )

    def _record_outcome(self, cell: Cell, accepted: bool) -> None:
        """Update the per-region failure counter and blacklist. Success resets the
        counter (a region that improved has earned a clean slate)."""
        if accepted:
            self._failures.pop(cell, None)
            return
        self._failures[cell] = self._failures.get(cell, 0) + 1
        if self._failures[cell] >= self._max_region_failures:
            self._blacklist.add(cell)

    # --- perception + retrying capture/stroke ------------------------------------
    def _observe(self, frame, iteration: int) -> Observation:
        obs = observe(frame, self._target, n=self._grid_n, color_weight=self._color_weight)
        self._last_global = obs.global_error
        self._emit(ObserveDone(
            iteration=iteration,
            global_error=obs.global_error,
            region_error=obs.region_error,
            heatmap=obs.heatmap,
        ))
        self._emit(FrameCaptured(iteration=iteration, frame=obs.frame.image))
        return obs

    def _capture(self):
        """Capture with bounded retry; raise ``_CanvasLost`` if the canvas stays
        unlocatable. The easel does its own internal fiducial retries; this is a second,
        coarser layer so a longer occlusion still aborts cleanly rather than crashing."""
        return self._with_retry(self._easel.capture)

    def _execute(self, intent: PaintIntent):
        """Paint with bounded retry. Re-executing repaints the same region (the fill is
        effectively idempotent), so a mid-stroke canvas loss is safe to retry."""
        return self._with_retry(lambda: self._executor.execute(intent))

    def _with_retry(self, action):
        # Only LookupError (the easel failing to LOCATE the canvas) is treated as a
        # transient, recoverable condition. Any other exception is a genuine bug (e.g. a
        # frame/target size mismatch from observe(), a planner/executor error) and is
        # deliberately left to propagate loud rather than be masked as canvas_lost.
        last: Optional[LookupError] = None
        for attempt in range(self._capture_retries + 1):
            try:
                return action()
            except LookupError as ex:
                last = ex
                if attempt < self._capture_retries:
                    time.sleep(self._capture_retry_delay)
        raise _CanvasLost() from last

    # --- termination + emission --------------------------------------------------
    def _finish(self, iteration: int, reason: str, converged: bool) -> RunDone:
        ge = float("nan") if self._last_global is None else float(self._last_global)
        self._emit(StateUpdate(
            iteration=iteration, global_error=ge, status=STATUS_DONE, converged=converged
        ))
        done = RunDone(
            iteration=iteration, global_error=ge, reason=reason, converged=converged
        )
        self._emit(done)
        return done

    def _emit(self, event) -> None:
        """Best-effort emission: a sink/transport failure must never kill a paint run
        (Principle 5 — the orchestrator neither knows nor depends on who is listening)."""
        try:
            self._sink.emit(event)
        except Exception:
            pass
