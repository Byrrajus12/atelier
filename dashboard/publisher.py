"""WebsocketPublisher — a concrete ``EventSink`` that broadcasts the orchestrator's event
stream to connected websocket clients (e.g. the M8 dashboard).

This is I/O transport, not domain logic, so it lives OUTSIDE ``core/`` (CLAUDE.md
Principles 2 & 5): the orchestrator knows only the abstract ``EventSink``; this module
pulls in ``websockets``/``asyncio``/``threading`` and the core never sees them. The
orchestrator runs a synchronous, blocking paint loop; this publisher runs its own
asyncio loop in a daemon thread and ``emit`` (called from the loop's thread) schedules a
non-blocking broadcast onto it. ``emit`` never blocks the paint loop and never raises out
to it — dropping a message is always preferable to stalling or killing a run.

Serialization is explicit per event type so the wire stays small: ``region_error`` is
sent as a nested list, but the heatmap is sent only as a *reference* (its iteration
index), never as pixels. NaN floats (e.g. a global error before the first capture) are
sent as ``null`` so the payload is valid JSON.

``FrameCaptured`` is the one exception to "no pixels on the wire": it carries the raw
canvas capture so the dashboard can show the live painting. Sampling and encoding are
both publisher policy, not core's — ``frame_cadence`` drops all but every Nth capture
before it is ever encoded, and ``frame_max_width``/``frame_jpeg_quality`` control the
downsample/JPEG quality trade-off. The run's *final* frame is force-published on
``run.done`` regardless of cadence, so a run whose terminal iteration falls off-cadence
never leaves the dashboard showing a stale, unfinished-looking canvas.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import threading
from typing import Any, Dict, List, Optional, Set

import cv2
import websockets

from core.events import Event, FrameCaptured, RunDone


def _num(x: Any) -> Any:
    """JSON-safe number: NaN/inf -> None (null), so the payload is valid JSON."""
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


def event_to_message(event: Event) -> Dict[str, Any]:
    """Turn an event into a JSON-safe dict. Explicit per type so no ndarray or oversized
    payload can leak onto the wire; the heatmap is referenced by iteration, not shipped."""
    t = event.type
    if t == "run.start":
        return {
            "type": t,
            "canvas_size": list(event.canvas_size),
            "grid_n": event.grid_n,
            "max_iterations": event.max_iterations,
            "max_region_failures": event.max_region_failures,
            "error_threshold": _num(event.error_threshold),
            "improvement_threshold": _num(event.improvement_threshold),
            "reversible": event.reversible,
            "has_undo": event.has_undo,
            "stroke_cost": _num(event.stroke_cost),
        }
    if t == "observe.done":
        return {
            "type": t,
            "iteration": event.iteration,
            "global_error": _num(event.global_error),
            "region_error": [[_num(v) for v in row] for row in event.region_error.tolist()],
            "heatmap_ref": event.iteration,  # a reference, not pixels
        }
    if t == "plan.done":
        it = event.intent
        return {
            "type": t,
            "iteration": event.iteration,
            "intent": {
                "cell": list(it.cell),
                "box": list(it.box),
                "color": list(it.color),
                "error": _num(it.error),
                "size": _num(it.size),
            },
        }
    if t == "execute.done":
        return {
            "type": t,
            "iteration": event.iteration,
            "cell": list(event.cell),
            "stroke_count": event.stroke_count,
        }
    if t == "verify.done":
        v = event.verdict
        return {
            "type": t,
            "iteration": event.iteration,
            "verdict": {
                "accepted": v.accepted,
                "cell": list(v.cell),
                "region_before": _num(v.region_before),
                "region_after": _num(v.region_after),
                "region_delta": _num(v.region_delta),
                "global_before": _num(v.global_before),
                "global_after": _num(v.global_after),
                "global_delta": _num(v.global_delta),
            },
        }
    if t == "state.update":
        return {
            "type": t,
            "iteration": event.iteration,
            "global_error": _num(event.global_error),
            "status": event.status,
            "converged": event.converged,
        }
    if t == "run.done":
        return {
            "type": t,
            "iteration": event.iteration,
            "global_error": _num(event.global_error),
            "reason": event.reason,
            "converged": event.converged,
        }
    # Unknown event: still forward its type so a consumer isn't left guessing.
    return {"type": t}


class WebsocketPublisher:
    """A best-effort broadcast sink. Call ``start()`` before running the orchestrator and
    ``close()`` after (or use it as a context manager). Implements the ``EventSink``
    ``emit`` protocol structurally — it is passed wherever an ``EventSink`` is expected."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        frame_cadence: int = 3,
        frame_max_width: int = 300,
        frame_jpeg_quality: int = 60,
    ):
        self._host = host
        self._port = port
        # Frame publishing policy — config knobs, not core concerns (Principle 2): core
        # emits a FrameCaptured every iteration; the publisher decides how much of that
        # actually goes out over the wire.
        self._frame_cadence = frame_cadence
        self._frame_max_width = frame_max_width
        self._frame_jpeg_quality = frame_jpeg_quality
        self._clients: Set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        # Bootstrap snapshot for late-joining clients: the run's start, latest state, and
        # the latest published frame.
        self._last_run_start: Optional[str] = None
        self._last_state: Optional[str] = None
        self._last_frame: Optional[str] = None
        # Most recent FrameCaptured seen, kept raw so run.done can force-publish it even
        # if its iteration was off-cadence (the finished canvas must never be dropped).
        self._last_frame_event: Optional[FrameCaptured] = None

    @property
    def port(self) -> int:
        """The bound port (resolves ``port=0`` to the actual chosen port after start)."""
        return self._port

    # --- lifecycle ---------------------------------------------------------------
    def start(self, timeout: float = 5.0) -> "WebsocketPublisher":
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("websocket server did not start in time")
        return self

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start_server())
        self._ready.set()
        self._loop.run_forever()

    async def _start_server(self) -> None:
        self._server = await websockets.serve(self._handler, self._host, self._port)
        # Resolve port=0 to the actually bound port.
        self._port = self._server.sockets[0].getsockname()[1]

    async def _handler(self, websocket) -> None:
        # Register first, then send the bootstrap snapshot: once a client has received
        # bootstrap it is already in the broadcast set, so no live event can slip past it.
        self._clients.add(websocket)
        try:
            for snap in (self._last_run_start, self._last_state, self._last_frame):
                if snap is not None:
                    await websocket.send(snap)
            await websocket.wait_closed()
        finally:
            self._clients.discard(websocket)

    def close(self) -> None:
        loop = self._loop
        if loop is None:
            return

        async def _shutdown() -> None:
            if self._server is not None:
                self._server.close()
                try:
                    await self._server.wait_closed()
                except Exception:
                    pass
            loop.stop()

        # Schedule the async shutdown on the loop's own thread: close the server and
        # await it fully before stopping the loop, so no server task is left pending.
        loop.call_soon_threadsafe(lambda: loop.create_task(_shutdown()))
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._loop = None

    def __enter__(self) -> "WebsocketPublisher":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- EventSink protocol ------------------------------------------------------
    def emit(self, event: Event) -> None:
        """Serialize and schedule a non-blocking broadcast. Never blocks the paint loop;
        never raises out to it (a transport error just drops the message)."""
        if isinstance(event, FrameCaptured):
            self._last_frame_event = event
            if event.iteration % self._frame_cadence == 0:
                self._publish_frame(event)
            return
        if isinstance(event, RunDone):
            # The run just ended: force out the last capture even if its iteration was
            # off-cadence and got skipped, so the dashboard never ends on a stale,
            # unfinished-looking canvas.
            last = self._last_frame_event
            if last is not None and last.iteration % self._frame_cadence != 0:
                self._publish_frame(last)
        try:
            data = json.dumps(event_to_message(event))
        except Exception:
            return
        # These two are written here (caller thread) and read in _handler (loop thread)
        # for the bootstrap snapshot. No lock: a plain attribute rebind is atomic under
        # the GIL, and a late joiner missing the very latest snapshot by a hair is
        # harmless — the live stream corrects it on the next event.
        if event.type == "run.start":
            self._last_run_start = data
        elif event.type == "state.update":
            self._last_state = data
        loop = self._loop
        if loop is None:
            return  # not started (or already closed): drop, best-effort
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(data), loop)
        except Exception:
            pass

    def _encode_frame(self, event: FrameCaptured) -> Dict[str, Any]:
        """Downsample + JPEG + base64 a captured frame. Encoding lives here, not in
        ``event_to_message``, because it needs this instance's config knobs
        (``_frame_max_width``/``_frame_jpeg_quality``) — core hands over a bare ndarray
        and never sees the encoding (Principle 2)."""
        frame = event.frame
        h, w = frame.shape[:2]
        if w > self._frame_max_width:
            scale = self._frame_max_width / w
            frame = cv2.resize(
                frame, (self._frame_max_width, max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buf = cv2.imencode(
            ".jpg", frame[:, :, ::-1],  # RGB -> BGR for OpenCV
            [cv2.IMWRITE_JPEG_QUALITY, self._frame_jpeg_quality],
        )
        if not ok:
            raise RuntimeError("jpeg encode failed")
        h2, w2 = frame.shape[:2]
        return {
            "type": "frame.captured",
            "iteration": event.iteration,
            "width": w2,
            "height": h2,
            "image": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"),
        }

    def _publish_frame(self, event: FrameCaptured) -> None:
        """Encode and broadcast one frame, best-effort — an encode failure or closed
        transport just drops it, same contract as ``emit``."""
        try:
            data = json.dumps(self._encode_frame(event))
        except Exception:
            return
        self._last_frame = data
        loop = self._loop
        if loop is None:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._broadcast(data), loop)
        except Exception:
            pass

    async def _broadcast(self, data: str) -> None:
        for ws in list(self._clients):
            try:
                await ws.send(data)
            except Exception:
                self._clients.discard(ws)
