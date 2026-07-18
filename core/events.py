"""The typed event stream — how the orchestrator reports what it is doing without
knowing who (if anyone) is listening (CLAUDE.md Principle 5: the orchestrator emits a
typed event stream and knows nothing about any interface; consumers are pure listeners).

Two things live here, both pure and dependency-light (Principle 2):

  * the **event types** — a frozen dataclass per event, each with a stable ``type`` tag
    so a sink can serialize it without reflection or inheritance-ordering games. Events
    carry the domain objects they report (a ``PaintIntent``, a ``Verdict``) directly, so
    there is a single source of truth and no field duplication.
  * the **sink boundary** — ``EventSink``, the abstract ``emit(event)`` interface that is
    the orchestrator's ONLY view of "who's listening", plus ``RecordingSink``, an
    in-memory list used by tests.

Concrete transports (a websocket publisher, a file logger) are ``EventSink``
implementations that live OUTSIDE the core (e.g. ``dashboard/publisher.py``): the
websockets/asyncio machinery is I/O transport, not domain logic, and must not weigh down
the core. The core depends only on this ABC.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, List, Tuple

import numpy as np

from core.planner import PaintIntent
from core.verifier import Verdict

# Why a run ended (RunDone.reason). Named so the orchestrator and its tests agree on the
# exact strings rather than sprinkling literals.
REASON_CONVERGED = "converged"      # planner found nothing worth painting (primary "done")
REASON_BUDGET = "budget"            # hit the max-iterations safety cap
REASON_STALLED = "stalled"          # only over-threshold regions left are blacklisted
REASON_STALLED_NO_PROGRESS = "stalled_no_progress"  # global error flat for N iterations
REASON_CANVAS_LOST = "canvas_lost"  # capture/stroke kept failing to locate the canvas

STATUS_RUNNING = "running"
STATUS_DONE = "done"


class Event:
    """Base type for every orchestrator event. Subclasses are frozen dataclasses that
    set a stable ``type`` tag (``ClassVar``, so it is not a dataclass field and does not
    affect field ordering). The base intentionally declares no fields, so subclasses are
    free to order their own without default-ordering conflicts."""

    type: ClassVar[str]


@dataclass(frozen=True)
class RunStart(Event):
    """A run is beginning against a target. Carries the run's fixed configuration so a
    late-joining consumer can render context, including the easel's declared
    capabilities (Principle 7 observability)."""

    type: ClassVar[str] = "run.start"
    canvas_size: Tuple[int, int]
    grid_n: int
    max_iterations: int
    max_region_failures: int
    error_threshold: float
    improvement_threshold: float
    reversible: bool
    has_undo: bool
    stroke_cost: float


@dataclass(frozen=True)
class ObserveDone(Event):
    """A fresh perception result. ``iteration`` 0 is the baseline observe before any
    stroke; ``iteration`` k (>=1) is the observe taken after the k-th stroke.

    ``region_error`` (n x n) and ``heatmap`` (HxWx3 uint8) are carried as arrays for
    in-process consumers. Over a wire transport the heatmap is referenced by
    ``iteration`` (a "heatmap ref"), not shipped as pixels — that is the publisher's
    serialization choice, not this event's concern."""

    type: ClassVar[str] = "observe.done"
    iteration: int
    global_error: float
    region_error: np.ndarray
    heatmap: np.ndarray


@dataclass(frozen=True)
class FrameCaptured(Event):
    """The raw canvas capture for this iteration, offered as its own event so frame
    cadence/publishing policy (e.g. periodic sampling for a wire transport) can vary
    independently of error-state consumers (Principle 5: a sink decides sampling, the
    core doesn't). ``frame`` is the captured RGB image (HxWx3 uint8); ``iteration`` uses
    the same numbering as ``ObserveDone`` (0 is the pre-stroke baseline)."""

    type: ClassVar[str] = "frame.captured"
    iteration: int
    frame: np.ndarray


@dataclass(frozen=True)
class PlanDone(Event):
    """The planner's decision for this iteration: the region-level ``PaintIntent`` it
    chose (which cell, what color, the error that drove it)."""

    type: ClassVar[str] = "plan.done"
    iteration: int
    intent: PaintIntent


@dataclass(frozen=True)
class ExecuteDone(Event):
    """The executor finished painting the intent: which ``cell`` and how many strokes it
    laid through the easel."""

    type: ClassVar[str] = "execute.done"
    iteration: int
    cell: Tuple[int, int]
    stroke_count: int


@dataclass(frozen=True)
class VerifyDone(Event):
    """The verifier's judgment of the stroke just applied — the full ``Verdict``
    (accepted, region/global before/after/delta)."""

    type: ClassVar[str] = "verify.done"
    iteration: int
    verdict: Verdict


@dataclass(frozen=True)
class StateUpdate(Event):
    """A running summary after an iteration: how many paint intents so far, the current global
    error, and whether the loop is still running or finished."""

    type: ClassVar[str] = "state.update"
    iteration: int
    global_error: float
    status: str = STATUS_RUNNING
    converged: bool = False


@dataclass(frozen=True)
class RunDone(Event):
    """The run has ended. ``reason`` is one of the ``REASON_*`` constants; ``converged``
    says whether it ended by reaching the target (vs a budget/stall/failure stop)."""

    type: ClassVar[str] = "run.done"
    iteration: int
    global_error: float
    reason: str
    converged: bool


class EventSink(ABC):
    """The orchestrator's sole view of the outside world (Principle 5). It emits events;
    it never learns whether they are recorded, printed, sent over a socket, or dropped.

    ``emit`` should be treated as best-effort by callers (the orchestrator swallows sink
    exceptions so a flaky transport can never kill a paint run), so implementations
    should still avoid raising where they reasonably can."""

    @abstractmethod
    def emit(self, event: Event) -> None:
        """Publish one event. Must not block the caller for long."""


class RecordingSink(EventSink):
    """An in-memory sink that appends every event to ``events`` in emission order. The
    default sink for tests: assert on the recorded sequence, no transport needed."""

    def __init__(self) -> None:
        self.events: List[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)
