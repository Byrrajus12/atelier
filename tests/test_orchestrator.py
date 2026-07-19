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
    REASON_PLANNER_SKIPPED,
    REASON_STALLED,
    REASON_STALLED_NO_PROGRESS,
    STATUS_DONE,
    ExecuteDone,
    PlanDone,
    PlannerSkipped,
    RecordingSink,
    RunDone,
    RunStart,
)
from core.executor import Executor
from core.orchestrator import Orchestrator
from core.perception import cell_box
from core.planner import GreedyPlanner, PaintIntent, PlannerSkip
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
                dict(progress_epsilon=-0.1), dict(max_stall_iterations=0),
                dict(max_consecutive_skips=0), dict(observer_palette=())):
        with pytest.raises(ValueError):
            Orchestrator(*args, **bad)


# --- non-blocking observer (Phase 1) ---------------------------------------------
# The observer RECORDS whether a planned move would damage a no-undo canvas; it never
# blocks one. That is what makes an unconstrained model planner measurable.
PALETTE = (RED, BLUE, (17, 17, 17))


class FixedPlanner(GreedyPlanner):
    """Emits a scripted sequence of intents regardless of what it perceives, so an
    unconstrained (possibly self-damaging) choice can be forced deterministically."""

    def __init__(self, intents):
        super().__init__()
        self._intents = list(intents)

    def plan(self, observation):
        return self._intents.pop(0) if self._intents else None


def observer_flags(sink):
    return [e for e in sink.events if e.type == "observer.flag"]


def test_observer_is_off_unless_a_palette_is_given():
    target = make_target({(0, 0): RED, (3, 3): BLUE})
    sink = RecordingSink()
    build(FakeEasel(), target, sink).run()
    assert observer_flags(sink) == []


def test_observer_emits_a_flag_for_each_planned_intent():
    target = make_target({(0, 0): RED, (3, 3): BLUE})
    sink = RecordingSink()
    build(FakeEasel(), target, sink, observer_palette=PALETTE).run()
    flags = observer_flags(sink)
    plans = [e for e in sink.events if isinstance(e, PlanDone)]
    assert len(flags) == len(plans) and len(flags) > 0
    assert [f.cell for f in flags] == [tuple(p.intent.cell) for p in plans]


def test_observer_flags_a_self_damaging_move_without_blocking_it():
    """Painting a white background cell red cannot be undone and provably does not help.
    The observer must flag it AND the move must still be executed — the planner's choice
    stands (a guard that silently corrected it would erase the measurement)."""
    target = make_target({(0, 0): RED})  # cell (2,2) is white background in the target
    easel = FakeEasel()
    box = cell_box(2, 2, N, CANVAS)
    bad = PaintIntent(cell=(2, 2), box=box, color=RED, error=0.9)
    sink = RecordingSink()
    Orchestrator(
        easel, target, FixedPlanner([bad]), Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0, observer_palette=PALETTE,
    ).run()

    flags = observer_flags(sink)
    assert len(flags) == 1
    assert flags[0].cell == (2, 2)
    assert flags[0].self_damaging is True
    # ...and the paint actually happened anyway: the cell is red, not white.
    assert np.array_equal(easel.cell_at(2, 2), np.full_like(easel.cell_at(2, 2), RED))


def test_observer_does_not_flag_a_genuinely_helpful_move():
    target = make_target({(0, 0): RED})
    easel = FakeEasel()
    good = PaintIntent(cell=(0, 0), box=cell_box(0, 0, N, CANVAS), color=RED, error=0.9)
    sink = RecordingSink()
    Orchestrator(
        easel, target, FixedPlanner([good]), Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0, observer_palette=PALETTE,
    ).run()
    flags = observer_flags(sink)
    assert len(flags) == 1
    assert flags[0].self_damaging is False


def test_observer_reports_the_swatch_that_would_actually_be_painted():
    """The intent asks for a color; the easel snaps it to a palette. The flag must show
    the color that would truly land, or a consumer misreads what happened."""
    target = make_target({(0, 0): RED})
    easel = FakeEasel()
    # Requests an off-red the palette does not contain; nearest swatch is RED.
    intent = PaintIntent(cell=(0, 0), box=cell_box(0, 0, N, CANVAS), color=(250, 10, 10), error=0.9)
    sink = RecordingSink()
    Orchestrator(
        easel, target, FixedPlanner([intent]), Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0, observer_palette=PALETTE,
    ).run()
    flag = observer_flags(sink)[0]
    assert flag.requested_color == (250, 10, 10)
    assert flag.predicted_color == RED


def test_observer_failure_never_kills_the_run(caplog, monkeypatch):
    """An observer is pure instrumentation; a fault in it must not cost a paint run — but
    it must also not vanish silently, or a zero-flag run would be indistinguishable from a
    perfectly-behaved one and the measurement would be quietly worthless.

    The fault is injected at the exact function the observer calls, so the failure path is
    explicit rather than relying on some value incidentally raising."""
    def boom(*args, **kwargs):
        raise RuntimeError("observer is broken")

    monkeypatch.setattr("core.orchestrator.swatch_would_improve", boom)

    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    orch = build(easel, target, sink, observer_palette=PALETTE)
    with caplog.at_level("WARNING"):
        result = orch.run()

    # The run still converges and paints correctly...
    assert result.reason == REASON_CONVERGED and result.converged is True
    assert np.array_equal(easel._canvas, target.image)
    # ...no flags were produced, and the failure was reported rather than swallowed.
    assert observer_flags(sink) == []
    assert any("observer failed" in r.message for r in caplog.records)


# --- PlannerSkip: "I could not decide" is NOT "the canvas is done" ----------------
# The bug this section guards: a VLM that failed one iteration returned None, the
# orchestrator read None as convergence, and the run ended reported converged=True at
# ~13% global error with most cells unpainted. None and PlannerSkip must stay distinct.


class SkippingPlanner(GreedyPlanner):
    """Raises ``PlannerSkip`` on the calls listed in ``skip_on`` (1-indexed), and plans
    greedily on every other call. ``skip_on=None`` means skip forever — a planner that
    can never decide, e.g. one whose backing API is down."""

    def __init__(self, skip_on=None):
        super().__init__()
        self.skip_on = skip_on
        self.calls = 0

    def plan(self, observation):
        self.calls += 1
        if self.skip_on is None or self.calls in self.skip_on:
            raise PlannerSkip(f"scripted skip on call {self.calls}")
        return super().plan(observation)


def skip_events(sink):
    return [e for e in sink.events if isinstance(e, PlannerSkipped)]


def test_a_skip_does_not_terminate_the_run_and_the_next_iteration_paints():
    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    result = Orchestrator(
        easel, target, SkippingPlanner(skip_on={1}), Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0,
    ).run()

    # The skipped first iteration cost the run nothing: it still paints and converges.
    assert result.reason == REASON_CONVERGED and result.converged is True
    assert np.array_equal(easel._canvas, target.image)
    assert sum(isinstance(e, ExecuteDone) for e in sink.events) >= 2
    # The skip is visible in the stream, and it did not advance the paint counter.
    assert [(e.iteration, e.consecutive) for e in skip_events(sink)] == [(0, 1)]


def test_a_planner_that_never_decides_terminates_unconverged_at_the_true_error():
    # The regression test for the shipped bug, stated as the requirement: a run that
    # stopped because the planner could not decide must NOT report converged=True.
    target = make_target({(0, 0): RED, (0, 1): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    result = Orchestrator(
        easel, target, SkippingPlanner(), Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0, max_consecutive_skips=5,
    ).run()

    assert result.reason == REASON_PLANNER_SKIPPED
    assert result.converged is False
    assert result.iteration == 0                    # nothing was ever painted
    assert result.global_error > 0.02               # and the canvas says so
    assert sum(isinstance(e, ExecuteDone) for e in sink.events) == 0
    assert not np.array_equal(easel._canvas, target.image)
    # It stopped AT the cap, not after spinning forever.
    assert [e.consecutive for e in skip_events(sink)] == [1, 2, 3, 4, 5]
    assert sink.events[-2].status == STATUS_DONE and sink.events[-2].converged is False


def test_the_skip_counter_counts_consecutive_skips_and_a_decision_resets_it():
    # Eight skips total, but never five in a row: a transient planner is tolerated
    # indefinitely as long as it keeps making progress in between.
    target = make_target({(0, 0): RED, (3, 3): BLUE})
    easel = FakeEasel()
    sink = RecordingSink()
    planner = SkippingPlanner(skip_on={1, 2, 3, 4, 6, 7, 8, 9})
    result = Orchestrator(
        easel, target, planner, Executor(easel), Verifier(), sink,
        grid_n=N, capture_retry_delay=0.0, max_consecutive_skips=5,
    ).run()

    assert result.reason == REASON_CONVERGED and result.converged is True
    assert np.array_equal(easel._canvas, target.image)
    assert len(skip_events(sink)) == 8
    assert max(e.consecutive for e in skip_events(sink)) == 4  # reset by call 5's decision


def test_greedy_returning_none_still_terminates_converged():
    # The other half of the distinction, guarded explicitly: PlannerSkip must not have
    # bled into the meaning of None. A planner that returns None IS asserting the canvas
    # is done, and that claim must still be honored as convergence.
    from core.perception import observe

    target = make_target({(0, 0): RED})
    easel = FakeEasel()
    sink = RecordingSink()
    result = build(easel, target, sink).run()

    # On the finished canvas the greedy planner returns None — it does not raise.
    assert GreedyPlanner().plan(observe(easel.capture(), target, n=N)) is None
    assert result.reason == REASON_CONVERGED and result.converged is True
    assert result.global_error < 0.02
    assert skip_events(sink) == []
