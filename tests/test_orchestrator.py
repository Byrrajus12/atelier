"""Tests for core/orchestrator.py — the full loop against a FAKE easel and a recording
sink. No browser, no websocket, no real capture: the orchestrator must be entirely
testable in-process (that is the point of the EventSink boundary and the Easel contract).

FakeEasel holds an in-memory canvas and, on apply_stroke, fills the grid cell band implied
by the stroke (approximating the executor's filled intent region), so
perceived error genuinely drops as regions are painted. ``dead_boxes`` mark pixel
rectangles the easel refuses to change, making a region permanently unpaintable — used to
exercise rejection, blacklisting, and the stalled termination.
"""

import math

import numpy as np
import pytest

from core.adapter import Capabilities, Easel, Frame
from core.events import (
    REASON_BUDGET,
    REASON_CANVAS_LOST,
    REASON_CONVERGED,
    REASON_STALLED,
    REASON_STALLED_NO_PROGRESS,
    STATUS_DONE,
    ExecuteDone,
    PlanDone,
    RecordingSink,
    RunDone,
    RunStart,
)
from core.executor import Executor
from core.orchestrator import Orchestrator
from core.perception import cell_box
from core.planner import GreedyPlanner
from core.target import Target
from core.verifier import Verifier

RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
CANVAS = (48, 48)
N = 4  # 4x4 grid -> 12px cells over a 48x48 canvas

# Effectively disables the global-plateau stall detector. The blacklist tests below build
# a global plateau on purpose (a dead region is hammered while global error stays flat) to
# isolate the per-region blacklist -> `stalled` path; the plateau detector would otherwise
# fire first and end them as `stalled_no_progress`. Its own behavior is covered by
# test_no_progress_plateau_terminates_before_budget.
NO_STALL_DETECTOR = 10_000


def make_target(cells, canvas_size=CANVAS, n=N) -> Target:
    """A white target with the given grid-aligned solid-color cells: ``cells`` maps
    ``(i, j) -> color``."""
    w, h = canvas_size
    img = np.full((h, w, 3), WHITE, dtype=np.uint8)
    for (i, j), color in cells.items():
        x0, y0, x1, y1 = cell_box(i, j, n, canvas_size)
        img[y0:y1, x0:x1] = color
    return Target(np.ascontiguousarray(img))


class FakeEasel(Easel):
    def __init__(self, canvas_size=CANVAS, background=WHITE, dead_boxes=()):
        self._w, self._h = canvas_size
        self._canvas = np.full((self._h, self._w, 3), background, dtype=np.uint8)
        self._writable = np.ones((self._h, self._w), dtype=bool)
        for (x0, y0, x1, y1) in dead_boxes:
            self._writable[y0:y1, x0:x1] = False

    def capabilities(self):
        return Capabilities(reversible=False, has_undo=False)

    def canvas_size(self):
        return (self._w, self._h)

    def capture(self):
        return Frame(self._canvas.copy(), timestamp=0.0)

    def apply_stroke(self, stroke):
        xs = [p.x for p in stroke.path]
        ys = [p.y for p in stroke.path]
        x0 = max(0, int(math.floor(min(xs))))
        x1 = min(self._w, int(math.ceil(max(xs))))
        cell_h = self._h // N
        row = min(N - 1, max(0, int(float(np.mean(ys)) // cell_h)))
        y0 = row * cell_h
        y1 = self._h if row == N - 1 else (row + 1) * cell_h
        color = np.array(stroke.brush.color, dtype=np.uint8)
        region = self._canvas[y0:y1, x0:x1]
        mask = self._writable[y0:y1, x0:x1]
        region[mask] = color

    def cell_at(self, i, j):
        """The current canvas pixels of grid cell (i, j) — for assertions."""
        x0, y0, x1, y1 = cell_box(i, j, N, (self._w, self._h))
        return self._canvas[y0:y1, x0:x1]


class RaisingEasel(Easel):
    """Never locates the canvas: every capture raises LookupError. Exercises the
    bounded-retry-then-canvas_lost policy."""

    def __init__(self):
        self.calls = 0

    def capabilities(self):
        return Capabilities(reversible=False, has_undo=False)

    def canvas_size(self):
        return CANVAS

    def capture(self):
        self.calls += 1
        raise LookupError("fiducials not found")

    def apply_stroke(self, stroke):
        raise AssertionError("should not paint when capture never succeeds")


def build(easel, target, sink, **kw):
    return Orchestrator(
        easel, target, GreedyPlanner(), Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0, **kw,
    )


def types(sink):
    return [e.type for e in sink.events]


# --- convergence -----------------------------------------------------------------
def test_converges_on_solvable_target():
    target = make_target({(0, 0): RED, (0, 1): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    result = build(easel, target, sink).run()

    assert isinstance(result, RunDone)
    assert result.reason == REASON_CONVERGED and result.converged is True
    assert result.global_error < 0.02
    # the fake paints exact colors on grid-aligned cells, so the canvas reaches target
    assert np.array_equal(easel._canvas, target.image)


def test_event_sequence_is_complete_and_ordered():
    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    build(easel, target, sink).run()

    strokes = sum(isinstance(e, ExecuteDone) for e in sink.events)
    assert strokes >= 2
    expected = (
        ["run.start", "observe.done", "frame.captured", "state.update"]
        + strokes * ["plan.done", "execute.done", "observe.done", "frame.captured", "verify.done", "state.update"]
        + ["state.update", "run.done"]
    )
    assert types(sink) == expected
    # first/last and the terminal state.update carries status=done
    assert isinstance(sink.events[0], RunStart)
    assert isinstance(sink.events[-1], RunDone)
    assert sink.events[-2].status == STATUS_DONE and sink.events[-2].converged is True


# --- termination: budget ---------------------------------------------------------
def test_budget_termination():
    target = make_target({(0, 0): RED, (0, 1): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    result = build(easel, target, sink, max_iterations=1).run()

    assert result.reason == REASON_BUDGET and result.converged is False
    assert result.iteration == 1
    assert sum(isinstance(e, ExecuteDone) for e in sink.events) == 1  # exactly one intent


def test_convergence_on_the_last_allowed_stroke_reports_converged_not_budget():
    # A one-stroke target run with max_iterations=1: the run both finishes AND exhausts
    # its budget on the same stroke. "Nothing left to do" must win over the cap, or a
    # consumer branching on RunDone.converged gets a false negative.
    target = make_target({(0, 0): RED})
    easel = FakeEasel()
    sink = RecordingSink()
    result = build(easel, target, sink, max_iterations=1).run()

    assert result.reason == REASON_CONVERGED and result.converged is True
    assert result.iteration == 1
    assert sum(isinstance(e, ExecuteDone) for e in sink.events) == 1
    assert np.array_equal(easel._canvas, target.image)


# --- termination: stalled + blacklist on a repeatedly-failing region -------------
def test_stalled_when_only_region_is_unpaintable():
    dead = cell_box(1, 1, N, CANVAS)
    target = make_target({(1, 1): RED})
    easel = FakeEasel(dead_boxes=[dead])
    sink = RecordingSink()
    result = build(easel, target, sink,
                   max_region_failures=3, max_stall_iterations=NO_STALL_DETECTOR).run()

    assert result.reason == REASON_STALLED and result.converged is False
    # the unpaintable region is tried exactly max_region_failures times, then blacklisted
    picks = [e.intent.cell for e in sink.events if isinstance(e, PlanDone)]
    assert picks == [(1, 1), (1, 1), (1, 1)]
    assert result.iteration == 3


def test_blacklist_masking_uses_row_col_not_transposed():
    # Asymmetric dead cell (1, 3): a transposed mask (zeroing (3, 1) instead) would let
    # the loop keep re-picking (1, 3) forever or misreport the outcome. The paintable
    # cell (3, 0) must still get painted; (1, 3) is tried exactly the cap, then blacklisted.
    dead = cell_box(1, 3, N, CANVAS)
    target = make_target({(1, 3): RED, (3, 0): BLUE})
    easel = FakeEasel(dead_boxes=[dead])
    sink = RecordingSink()
    result = build(easel, target, sink, max_region_failures=3, max_iterations=50,
                   max_stall_iterations=NO_STALL_DETECTOR).run()

    assert result.reason == REASON_STALLED
    assert np.array_equal(easel.cell_at(3, 0), np.full((12, 12, 3), BLUE, np.uint8))
    dead_picks = [e for e in sink.events
                  if isinstance(e, PlanDone) and e.intent.cell == (1, 3)]
    assert len(dead_picks) == 3


def test_masking_does_not_mutate_the_source_observation():
    from core.perception import observe
    from core.planner import GreedyPlanner

    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel()
    orch = Orchestrator(easel, target, GreedyPlanner(), Executor(easel), Verifier(),
                        RecordingSink(), grid_n=N)
    orch._blacklist = {(0, 0)}  # pretend (0, 0) is blacklisted
    before = observe(easel.capture(), target, n=N)
    original = before.region_error.copy()
    orch._masked(before)  # must return a masked COPY, leaving `before` untouched
    assert np.array_equal(before.region_error, original)


def test_rejected_region_is_skipped_and_loop_moves_on():
    # (0,0) is unpaintable (dead), (3,3) is paintable. The loop must paint (3,3) and
    # terminate (stalled once (0,0) is blacklisted), never looping forever.
    dead = cell_box(0, 0, N, CANVAS)
    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel(dead_boxes=[dead])
    sink = RecordingSink()
    result = build(easel, target, sink, max_region_failures=3, max_iterations=50,
                   max_stall_iterations=NO_STALL_DETECTOR).run()

    assert result.reason == REASON_STALLED  # only the dead region remains, blacklisted
    # the paintable region did get painted (moved on despite the other failing)
    assert np.array_equal(easel.cell_at(3, 3), np.full((12, 12, 3), BLUE, np.uint8))
    # the dead region was attempted exactly the failure cap, then never again
    dead_picks = [e for e in sink.events
                  if isinstance(e, PlanDone) and e.intent.cell == (0, 0)]
    assert len(dead_picks) == 3


# --- termination: no-progress plateau (global error flat) ------------------------
def test_no_progress_plateau_terminates_before_budget():
    # A dead region the planner keeps re-picking: global error never moves, so the loop
    # plateaus. With the per-region reject cap and the iteration budget both set well
    # above the stall window, the ONLY thing that can end this run is the global-plateau
    # detector -> it must stop after max_stall_iterations flat strokes with the distinct
    # `stalled_no_progress` reason, not grind on to budget or wait for blacklisting.
    dead = cell_box(1, 1, N, CANVAS)
    target = make_target({(1, 1): RED})
    easel = FakeEasel(dead_boxes=[dead])
    sink = RecordingSink()
    result = build(easel, target, sink,
                   max_stall_iterations=3, max_region_failures=99, max_iterations=99).run()

    assert result.reason == REASON_STALLED_NO_PROGRESS and result.converged is False
    assert result.iteration == 3           # stopped the moment the plateau hit the window
    # it was the plateau, not the reject cap: the region was never blacklisted (99 > 3)
    picks = [e.intent.cell for e in sink.events if isinstance(e, PlanDone)]
    assert picks == [(1, 1), (1, 1), (1, 1)]
    assert isinstance(sink.events[-1], RunDone)


def test_steady_progress_does_not_trip_the_stall_detector():
    # Five paintable cells, more than the 3-stroke stall window: every stroke drops global
    # error by well over progress_epsilon, so the no-progress counter resets each time and
    # the run reaches the target normally (`converged`), never false-firing the detector.
    target = make_target({(0, 0): RED, (0, 1): RED, (0, 2): BLUE, (3, 0): BLUE, (3, 3): RED})
    easel = FakeEasel()
    sink = RecordingSink()
    result = build(easel, target, sink, max_stall_iterations=3).run()

    assert result.reason == REASON_CONVERGED and result.converged is True
    assert sum(isinstance(e, ExecuteDone) for e in sink.events) == 5  # all five painted
    assert np.array_equal(easel._canvas, target.image)


# --- termination: canvas lost ----------------------------------------------------
def test_canvas_lost_aborts_cleanly_after_bounded_retry():
    easel = RaisingEasel()
    sink = RecordingSink()
    result = build(easel, make_target({(0, 0): RED}), sink, capture_retries=2).run()

    assert result.reason == REASON_CANVAS_LOST and result.converged is False
    assert math.isnan(result.global_error)  # never got a baseline
    assert easel.calls == 3  # initial + 2 retries
    assert isinstance(sink.events[0], RunStart)   # run.start still emitted
    assert isinstance(sink.events[-1], RunDone)


# --- best-effort emit ------------------------------------------------------------
def test_bad_sink_never_kills_the_run():
    class ExplodingSink(RecordingSink):
        def emit(self, event):
            raise RuntimeError("transport down")

    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel()
    # Every emit raises, yet the paint run must still converge.
    result = build(easel, target, ExplodingSink()).run()
    assert result.reason == REASON_CONVERGED and result.converged is True
    assert np.array_equal(easel._canvas, target.image)


# --- constructor validation ------------------------------------------------------
def test_constructor_rejects_bad_params():
    easel = FakeEasel()
    target = make_target({(0, 0): RED})
    args = (easel, target, GreedyPlanner(), Executor(easel), Verifier(), RecordingSink())
    for bad in (dict(grid_n=0), dict(max_iterations=0), dict(max_region_failures=0),
                dict(capture_retries=-1), dict(capture_retry_delay=-1.0),
                dict(progress_epsilon=-0.1), dict(max_stall_iterations=0)):
        with pytest.raises(ValueError):
            Orchestrator(*args, **bad)
