"""M7.4 — the live convergence run (the payoff).

Drives the full perceive -> plan -> paint -> verify loop against the REAL browser canvas
easel until it converges (or hits the iteration cap), watching a blank white canvas
become a simple multi-region target. It:

  * prints the global error on every iteration, live, so you can watch the number
    descend as strokes land (that descending number is the whole point);
  * saves periodic captured frames + error heatmaps to an output folder so you can scrub
    through the run afterward;
  * broadcasts the same event stream over a websocket (dashboard/publisher.py) so the M8
    dashboard has something to connect to — no dashboard is required to run this.

This is a demo harness, not core: target generation and file output live here, out of the
domain-agnostic core (CLAUDE.md Scope). Run it from the repo root:  python scripts/live_run.py
"""

from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import time
from ctypes import wintypes

import numpy as np

# Make the repo root importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from core.events import (  # noqa: E402
    EventSink,
    ObserveDone,
    RunDone,
    RunStart,
    StateUpdate,
    VerifyDone,
)
from core.executor import Executor  # noqa: E402
from core.orchestrator import Orchestrator  # noqa: E402
from core.perception import cell_box  # noqa: E402
from core.planner import GreedyPlanner  # noqa: E402
from core.target import Target  # noqa: E402
from core.verifier import Verifier  # noqa: E402
from dashboard.publisher import WebsocketPublisher  # noqa: E402
from easels.browser_canvas import BrowserCanvasEasel  # noqa: E402

# A deliberately COOPERATIVE demo target (see the M7.4 rerun notes):
#
#   * White is the canvas's start color, but it is UNPAINTABLE — the palette is
#     red/blue/black and nearest_palette_color(white) == red. So a white cell that gets
#     bled into is picked and "painted toward white" -> red, and the colored block then
#     grows across any white gap until it collides with another block (the border war).
#   * The cure used here is a COARSE grid: at grid_n=6 a cell is 100px, so the executor's
#     ~6px bleed is only ~6% of a rim cell — below the planner's 0.02 error threshold — so
#     bled white rim cells stay unpicked and blocks do NOT grow. Blocks are grid-aligned
#     (whole cells), well separated by white, and no two differently-colored blocks are
#     adjacent, so the only interactions are same-color (cooperative) fills.
GRID_N = 6
MAX_ITERATIONS = 60
SAVE_EVERY = 3  # capture a canvas frame + heatmap every N iterations

RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (17, 17, 17)


def build_target(canvas_size) -> Target:
    """White canvas with three 2x2-cell solid blocks (red / blue / black), grid-aligned
    and separated by a full white cell each way, so no two differently-colored blocks
    touch. At grid_n=6 the blocks are 200x200 px with 100px white gaps between them."""
    w, h = canvas_size
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    def fill(i0, i1, j0, j1, color):
        x0, y0, _, _ = cell_box(i0, j0, GRID_N, canvas_size)
        _, _, x1, y1 = cell_box(i1, j1, GRID_N, canvas_size)
        img[y0:y1, x0:x1] = color

    fill(1, 2, 1, 2, RED)     # top-left     (cols/rows 1-2)
    fill(1, 2, 4, 5, BLUE)    # top-right    (gap column 3)
    fill(4, 5, 1, 2, BLACK)   # bottom-left  (gap row 3)
    return Target(np.ascontiguousarray(img))


def _save_rgb(path: str, rgb: np.ndarray) -> None:
    cv2.imwrite(path, rgb[:, :, ::-1])  # RGB -> BGR for OpenCV


def focus_browser(title_substr: str = "atelier reference canvas", timeout: float = 6.0) -> bool:
    """Bring the reference-canvas browser window to the foreground and pin it topmost, so
    nothing (an editor, a terminal) can occlude the canvas mid-run — occlusion made the
    first live run paint onto whatever was in front. Best-effort and Windows-only; returns
    whether the window was found. The window is matched by the page's <title>."""
    user32 = ctypes.windll.user32
    proc_ty = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    found = []

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            if title_substr.lower() in buff.value.lower():
                found.append(hwnd)
        return True

    deadline = time.time() + timeout
    while time.time() < deadline:
        found.clear()
        user32.EnumWindows(proc_ty(_cb), 0)
        if found:
            break
        time.sleep(0.3)
    if not found:
        return False

    hwnd = found[0]
    SW_RESTORE, HWND_TOPMOST = 9, -1
    SWP_NOMOVE, SWP_NOSIZE = 0x0002, 0x0001
    user32.ShowWindow(hwnd, SW_RESTORE)
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    # The ALT nudge satisfies Windows' foreground-lock so SetForegroundWindow takes.
    user32.keybd_event(0x12, 0, 0, 0)
    user32.SetForegroundWindow(hwnd)
    user32.keybd_event(0x12, 0, 2, 0)
    return True


class ConsoleSink(EventSink):
    """Prints a live, one-line-per-stroke trend and saves periodic frames/heatmaps. Holds
    the easel only to re-capture the canvas for periodic PNGs; it never plans or paints."""

    def __init__(self, easel, out_dir: str, save_every: int = SAVE_EVERY):
        self._easel = easel
        self._out_dir = out_dir
        self._save_every = save_every

    def emit(self, event) -> None:
        if isinstance(event, RunStart):
            print(f"run.start  canvas={event.canvas_size}  grid={event.grid_n}  "
                  f"cap={event.max_iterations}  reversible={event.reversible}")
        elif isinstance(event, StateUpdate) and event.iteration == 0:
            print(f"iter {event.iteration:3d}  baseline           global={event.global_error:.4f}")
        elif isinstance(event, VerifyDone):
            v = event.verdict
            mark = "accept" if v.accepted else "REJECT"
            print(f"iter {event.iteration:3d}  cell{tuple(v.cell)!s:<8} {mark}  "
                  f"regionΔ={v.region_delta:+.4f}  global={v.global_after:.4f}")
        elif isinstance(event, ObserveDone) and event.iteration and (
            event.iteration % self._save_every == 0
        ):
            self._save_frame(event.iteration, event.heatmap)
        elif isinstance(event, RunDone):
            print(f"\nrun.done   reason={event.reason}  converged={event.converged}  "
                  f"final_global={event.global_error:.4f}  iterations={event.iteration}")

    def _save_frame(self, iteration: int, heatmap: np.ndarray) -> None:
        try:
            canvas = self._easel.capture().image
            _save_rgb(os.path.join(self._out_dir, f"frame_{iteration:03d}.png"), canvas)
            _save_rgb(os.path.join(self._out_dir, f"heatmap_{iteration:03d}.png"), heatmap)
        except Exception as ex:  # a save must never disturb the run
            print(f"  (frame save at iter {iteration} skipped: {ex})")


class FanoutSink(EventSink):
    """Forwards each event to several sinks, best-effort (one failing sink never blocks
    the others or the run)."""

    def __init__(self, sinks):
        self._sinks = list(sinks)

    def emit(self, event) -> None:
        for s in self._sinks:
            try:
                s.emit(event)
            except Exception:
                pass


def main() -> int:
    out_dir = os.path.join(tempfile.gettempdir(), f"atelier_live_run_{int(time.time())}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"output folder: {out_dir}")

    easel = BrowserCanvasEasel(launch=True)
    publisher = WebsocketPublisher(port=8765).start()
    try:
        target = build_target(easel.canvas_size())
        _save_rgb(os.path.join(out_dir, "target.png"), target.image)

        # Force the browser foreground/topmost and confirm the canvas is locatable BEFORE
        # the first stroke, so nothing occludes it mid-run.
        if focus_browser():
            print("browser window focused + pinned topmost")
        else:
            print("WARNING: browser window not found to focus; keep it foreground manually")
        time.sleep(0.5)
        easel.capture()  # raises LookupError here if the canvas still isn't visible
        _save_rgb(os.path.join(out_dir, "frame_start.png"), easel.capture().image)

        sink = FanoutSink([ConsoleSink(easel, out_dir), publisher])
        orch = Orchestrator(
            easel, target, GreedyPlanner(), Executor(easel), Verifier(), sink,
            grid_n=GRID_N, max_iterations=MAX_ITERATIONS,
        )
        result = orch.run()

        _save_rgb(os.path.join(out_dir, "frame_final.png"), easel.capture().image)
        print(f"frames saved to {out_dir}")
        return 0 if result.converged else 1
    finally:
        publisher.close()
        easel.close()


if __name__ == "__main__":
    raise SystemExit(main())
