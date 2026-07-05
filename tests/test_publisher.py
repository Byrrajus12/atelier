"""Smoke test for dashboard/publisher.py — the ONE async / real-socket test in the
suite. It proves the transport end to end: start the server, connect a client, and check
that both a bootstrap snapshot (on connect) and a live broadcast arrive, JSON-decoded and
correctly serialized. The orchestrator tests never touch this — the loop is fully
testable without any websocket.

Uses port 0 (an OS-chosen free port) so the test can't collide with a real run."""

import asyncio
import json

import numpy as np

from core.events import ObserveDone, RunStart, StateUpdate
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
