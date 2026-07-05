"""Smoke test for dashboard/publisher.py — the ONE async / real-socket test in the
suite. It proves the transport end to end: start the server, connect a client, and check
that both a bootstrap snapshot (on connect) and a live broadcast arrive, JSON-decoded and
correctly serialized. The orchestrator tests never touch this — the loop is fully
testable without any websocket.

Uses port 0 (an OS-chosen free port) so the test can't collide with a real run."""

import asyncio
import base64
import json

import numpy as np

from core.events import FrameCaptured, ObserveDone, REASON_CONVERGED, RunDone, RunStart, StateUpdate
from dashboard.publisher import WebsocketPublisher, event_to_message


def test_event_serialization_is_json_safe_and_omits_pixels():
    # NaN -> null, heatmap referenced by iteration (not pixels), region grid -> lists.
    ev = ObserveDone(
        iteration=2, global_error=float("nan"),
        region_error=np.array([[0.1, 0.2], [0.3, 0.4]]),
        heatmap=np.zeros((8, 8, 3), dtype=np.uint8),
    )
    msg = event_to_message(ev)
    assert msg["global_error"] is None            # NaN -> null
    assert msg["heatmap_ref"] == 2                 # a ref, not pixels
    assert "heatmap" not in msg                    # pixels never serialized
    assert msg["region_error"] == [[0.1, 0.2], [0.3, 0.4]]
    json.dumps(msg)                                # must be valid JSON


def test_publisher_delivers_bootstrap_and_live_events():
    pub = WebsocketPublisher(port=0).start()
    try:
        # Emitted before any client connects: a late joiner should still get it (bootstrap).
        pub.emit(RunStart(
            canvas_size=(48, 48), grid_n=4, max_iterations=10, max_region_failures=3,
            error_threshold=0.02, improvement_threshold=0.005,
            reversible=False, has_undo=False, stroke_cost=1.0,
        ))

        async def scenario():
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{pub.port}") as client:
                # bootstrap snapshot arrives on connect
                boot = json.loads(await asyncio.wait_for(client.recv(), timeout=5))
                assert boot["type"] == "run.start"
                assert boot["canvas_size"] == [48, 48]

                # a live event emitted now reaches the (already-registered) client
                pub.emit(StateUpdate(iteration=1, global_error=0.5))
                live = json.loads(await asyncio.wait_for(client.recv(), timeout=5))
                assert live["type"] == "state.update"
                assert live["iteration"] == 1 and live["global_error"] == 0.5

        asyncio.run(scenario())
    finally:
        pub.close()
    # close() must actually tear the server thread down, not just time out.
    assert pub._thread is not None and not pub._thread.is_alive()


def _frame(iteration: int, w: int = 90, h: int = 60, value: int = 10) -> FrameCaptured:
    return FrameCaptured(iteration=iteration, frame=np.full((h, w, 3), value, dtype=np.uint8))


def test_frame_cadence_skips_off_cadence_iterations():
    pub = WebsocketPublisher(port=0, frame_cadence=3)
    pub.emit(_frame(1))
    pub.emit(_frame(2))
    assert pub._last_frame is None  # neither iteration is a multiple of 3
    pub.emit(_frame(3))
    assert pub._last_frame is not None


def test_frame_on_cadence_is_downsampled_and_jpeg_encoded():
    pub = WebsocketPublisher(port=0, frame_cadence=1, frame_max_width=40)
    pub.emit(_frame(1, w=90, h=60))

    msg = json.loads(pub._last_frame)
    assert msg["type"] == "frame.captured"
    assert msg["iteration"] == 1
    assert msg["width"] == 40
    assert msg["height"] == max(1, round(60 * (40 / 90)))
    assert msg["image"].startswith("data:image/jpeg;base64,")
    raw = base64.b64decode(msg["image"].split(",", 1)[1])
    assert raw[:2] == b"\xff\xd8"  # JPEG magic bytes


def test_final_frame_is_force_published_even_if_off_cadence():
    # A run converging at iteration 2 with cadence 3 would, without the force-publish
    # rule, end the dashboard on whatever frame preceded it (or none) -- the finished
    # canvas is the single most important frame and must never be dropped.
    pub = WebsocketPublisher(port=0, frame_cadence=3)
    pub.emit(_frame(1))
    pub.emit(_frame(2))
    assert pub._last_frame is None  # iteration 2 is off-cadence, not sent yet

    pub.emit(RunDone(iteration=2, global_error=0.01, reason=REASON_CONVERGED, converged=True))

    assert pub._last_frame is not None
    msg = json.loads(pub._last_frame)
    assert msg["type"] == "frame.captured" and msg["iteration"] == 2


def test_bootstrap_replays_last_frame_to_late_joiner():
    pub = WebsocketPublisher(port=0, frame_cadence=1).start()
    try:
        pub.emit(_frame(1))

        async def scenario():
            import websockets
            async with websockets.connect(f"ws://127.0.0.1:{pub.port}") as client:
                boot = json.loads(await asyncio.wait_for(client.recv(), timeout=5))
                assert boot["type"] == "frame.captured"
                assert boot["iteration"] == 1

        asyncio.run(scenario())
    finally:
        pub.close()
