"""Tests for core/events.py — the typed events and the sink boundary. Pure, no
orchestrator, no transport: construct events, check they are frozen and carry their
fields, and that RecordingSink records in emission order."""

import dataclasses

import numpy as np
import pytest

from core.events import (
    REASON_BUDGET,
    REASON_CANVAS_LOST,
    REASON_CONVERGED,
    REASON_STALLED,
    STATUS_RUNNING,
    Event,
    EventSink,
    ExecuteDone,
    FrameCaptured,
    ObserveDone,
    PlanDone,
    RecordingSink,
    RunDone,
    RunStart,
    StateUpdate,
    VerifyDone,
)
from core.planner import PaintIntent
from core.verifier import Verdict


def a_intent(cell=(1, 2)):
    return PaintIntent(cell=cell, box=(0, 0, 10, 10), color=(1, 2, 3), error=0.7)


def a_verdict(cell=(1, 2)):
    return Verdict(
        accepted=True, cell=cell, region_before=0.8, region_after=0.2,
        region_delta=0.6, global_before=0.5, global_after=0.4, global_delta=0.1,
    )


# --- type tags are stable and distinct -------------------------------------------
def test_event_type_tags():
    assert RunStart.type == "run.start"
    assert ObserveDone.type == "observe.done"
    assert PlanDone.type == "plan.done"
    assert ExecuteDone.type == "execute.done"
    assert VerifyDone.type == "verify.done"
    assert StateUpdate.type == "state.update"
    assert RunDone.type == "run.done"
    tags = {c.type for c in
            (RunStart, ObserveDone, PlanDone, ExecuteDone, VerifyDone, StateUpdate, RunDone)}
    assert len(tags) == 7  # all distinct


def test_all_events_are_events():
    ev = StateUpdate(iteration=3, global_error=0.2)
    assert isinstance(ev, Event)
    # instances carry the class tag
    assert ev.type == "state.update"


# --- events carry their fields and are frozen ------------------------------------
def test_run_start_carries_config():
    ev = RunStart(
        canvas_size=(600, 600), grid_n=16, max_iterations=200, max_region_failures=3,
        error_threshold=0.02, improvement_threshold=0.005,
        reversible=False, has_undo=False, stroke_cost=1.0,
    )
    assert ev.canvas_size == (600, 600)
    assert ev.grid_n == 16 and ev.max_region_failures == 3
    assert ev.reversible is False and ev.has_undo is False


def test_observe_done_carries_arrays():
    grid = np.zeros((4, 4), dtype=np.float64)
    heat = np.zeros((8, 8, 3), dtype=np.uint8)
    ev = ObserveDone(iteration=2, global_error=0.3, region_error=grid, heatmap=heat)
    assert ev.iteration == 2 and ev.global_error == 0.3
    assert ev.region_error is grid and ev.heatmap is heat


def test_frame_captured_carries_raw_array():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    ev = FrameCaptured(iteration=3, frame=frame)
    assert ev.type == "frame.captured"
    assert ev.iteration == 3 and ev.frame is frame


def test_plan_and_verify_carry_domain_objects():
    it = a_intent()
    vd = a_verdict()
    assert PlanDone(iteration=1, intent=it).intent is it
    assert VerifyDone(iteration=1, verdict=vd).verdict is vd


def test_state_update_defaults():
    ev = StateUpdate(iteration=0, global_error=0.9)
    assert ev.status == STATUS_RUNNING and ev.converged is False


def test_run_done_reason():
    ev = RunDone(iteration=42, global_error=0.01, reason=REASON_CONVERGED, converged=True)
    assert ev.reason == REASON_CONVERGED and ev.converged is True


def test_events_are_frozen():
    ev = StateUpdate(iteration=1, global_error=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ev.iteration = 2  # type: ignore[misc]


def test_reason_constants_distinct():
    assert len({REASON_CONVERGED, REASON_BUDGET, REASON_STALLED, REASON_CANVAS_LOST}) == 4


# --- RecordingSink ----------------------------------------------------------------
def test_recording_sink_records_in_order():
    sink = RecordingSink()
    assert isinstance(sink, EventSink)
    a = StateUpdate(iteration=0, global_error=0.9)
    b = StateUpdate(iteration=1, global_error=0.5)
    c = RunDone(iteration=1, global_error=0.5, reason=REASON_BUDGET, converged=False)
    for ev in (a, b, c):
        sink.emit(ev)
    assert sink.events == [a, b, c]
    assert [e.type for e in sink.events] == ["state.update", "state.update", "run.done"]
