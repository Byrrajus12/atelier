"""BrowserCanvasEasel — the reference environment: a local browser HTML5 canvas,
driven vision-only through ``mss`` screen capture and ``pydirectinput`` synthetic
cursor input.

This is the concrete implementation of the ``Easel`` contract (core/adapter.py),
factored up from the Milestone 1 spike. Everything environment-specific lives here:
process DPI awareness, fiducial colors, the palette, window launch, and the timing the
browser needs to register a drag (all recorded in ``spike/FINDINGS.md``). The core sees
none of it — only canvas-space frames and canvas-space strokes.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from core.adapter import (
    Capabilities,
    Color,
    Easel,
    Frame,
    Point,
    Stroke,
)


def _set_dpi_aware() -> str:
    """Make this process per-monitor DPI aware so ``mss`` capture pixels and
    ``pydirectinput`` cursor pixels share ONE physical-pixel space (the load-bearing
    coordinate-transform finding from M1). Must run before mss/pydirectinput touch
    Win32 state, hence at import time below."""
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
        return "system"
    except Exception:
        return "none"


DPI_LEVEL = _set_dpi_aware()

import mss  # noqa: E402  (import after DPI awareness is established)
import pydirectinput  # noqa: E402

pydirectinput.FAILSAFE = False  # don't abort if the cursor nears a screen corner
pydirectinput.PAUSE = 0.0       # we insert our own explicit sleeps (see FINDINGS)

# --- Environment constants (must match easels/canvas_page/index.html) -------------
FIDUCIAL_COLORS = {
    "tl": (255, 0, 255),   # magenta  top-left
    "tr": (0, 255, 255),   # cyan     top-right
    "bl": (255, 255, 0),   # yellow   bottom-left
    "br": (0, 255, 0),     # green    bottom-right
}
PALETTE: Tuple[Color, ...] = ((255, 0, 0), (0, 0, 255), (17, 17, 17))  # red/blue/black
# Fiducials live in the frame padding OUTSIDE the paintable canvas (see index.html),
# so a stroke can never overwrite them. Each centroid sits 15 canvas-px diagonally
# *outside* its corner -> a negative inset (the dst quad expands beyond the canvas).
FIDUCIAL_INSET = -15.0
CANVAS_SIZE = (600, 600)    # canvas-space frame, matches the page's CSS canvas
_SWATCH_MASK_PAD = 12       # extra px around the canvas footprint when hiding it

# Timing that reliably registers input in the browser (from FINDINGS).
_MOVE_DT = 0.03
_CLICK_DT = 0.05
_LOAD_WAIT = 3.0

# _geometry is imported lazily inside methods so headless tests can import this module
# (and its pure helpers) without paying for cv2/scipy up front.


def nearest_palette_color(requested: Color, palette: Tuple[Color, ...] = PALETTE) -> Color:
    """Pick the palette swatch nearest ``requested`` in RGB (Euclidean). The Easel can
    only paint colors the UI offers; the core requests an ideal color and verifies the
    realized one by re-capture, so an approximation here is fine."""
    r = np.array(requested, dtype=float)
    return min(palette, key=lambda c: float(np.sum((np.array(c, dtype=float) - r) ** 2)))


class BrowserCanvasEasel(Easel):
    def __init__(
        self,
        url: str = "http://localhost:8000/index.html",
        canvas_size: Tuple[int, int] = CANVAS_SIZE,
        launch: bool = False,
        serve_dir: Optional[str] = None,
    ):
        self._url = url
        self._canvas_size = canvas_size
        self._sct = None
        self._httpd: Optional[subprocess.Popen] = None
        self._browser: Optional[subprocess.Popen] = None
        self._profile_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_easel_profile"
        )
        if launch:
            self._launch(serve_dir)

    # --- capability declaration --------------------------------------------------
    def capabilities(self) -> Capabilities:
        # The reference page has no undo affordance, so strokes commit irreversibly.
        # The core must treat mistakes as permanent (Principle 7).
        return Capabilities(reversible=False, has_undo=False, stroke_cost=1.0)

    def canvas_size(self) -> Tuple[int, int]:
        return self._canvas_size

    # --- perception --------------------------------------------------------------
    def _locate_canvas(self, retries: int = 8, delay: float = 0.4):
        """Grab the screen and locate the four fiducials, retrying (re-grabbing) to
        ride out a still-loading page or a transient occlusion. Returns
        ``(screen, corners)`` or raises the last ``LookupError``."""
        from easels import _geometry as G

        last: Optional[LookupError] = None
        for attempt in range(max(1, retries)):
            screen = self._grab_screen()
            try:
                corners = G.find_fiducials(screen, FIDUCIAL_COLORS)
                return screen, corners
            except LookupError as ex:
                last = ex
                if attempt < retries - 1:
                    time.sleep(delay)
        raise last  # type: ignore[misc]

    def capture(self) -> Frame:
        from easels import _geometry as G

        screen, corners = self._locate_canvas()
        h_s2c, _ = G.canvas_homographies(corners, self._canvas_size, FIDUCIAL_INSET)
        canvas = G.warp_to_canvas(screen, h_s2c, self._canvas_size)
        return Frame(image=canvas, timestamp=time.monotonic())

    # --- action ------------------------------------------------------------------
    def apply_stroke(self, stroke: Stroke) -> None:
        from easels import _geometry as G

        screen, corners = self._locate_canvas()
        _, h_c2s = G.canvas_homographies(corners, self._canvas_size, FIDUCIAL_INSET)

        # 1. Realize the brush color by clicking the nearest palette swatch. Search
        #    with the canvas region masked out, so already-painted pixels of that
        #    color can't be mistaken for the swatch. A swatch that cannot be located
        #    is a hard failure: painting on with an unknown active color would be a
        #    silent, unverifiable color error, so we raise rather than guess.
        swatch = nearest_palette_color(stroke.brush.color)
        loc = self._locate_swatch(screen, h_c2s, swatch)
        if loc is None:
            raise LookupError(f"palette swatch for color {swatch} not found on screen")
        self._click(int(loc[0]), int(loc[1]))

        # 2. Map the canvas-space path to screen pixels and drag it.
        #    NOTE: BrushSpec.size is intentionally NOT realized here. The reference
        #    page hardcodes lineWidth=12; wiring variable stroke width is deferred to
        #    M5 (motion), where varied widths first get generated. No core path may
        #    assume brush size takes effect until then.
        screen_path = [G.apply_homography((p.x, p.y), h_c2s) for p in stroke.path]
        self._drag([(int(x), int(y)) for (x, y) in screen_path])

    # --- environment plumbing ----------------------------------------------------
    def _grab_screen(self) -> np.ndarray:
        if self._sct is None:
            self._sct = mss.mss()
        mon = self._sct.monitors[1]
        raw = np.array(self._sct.grab(mon))     # BGRA
        return raw[:, :, [2, 1, 0]].copy()      # -> RGB

    def _locate_swatch(self, screen, h_c2s, swatch: Color):
        from easels import _geometry as G

        # Hide the *true* canvas footprint (its four corners mapped to screen via the
        # homography), not the fiducial-centroid box — the fiducials sit outside the
        # canvas, and even a positive inset would leave a ring of paintable canvas
        # unmasked. Pad outward so no painted pixel near the edge can masquerade as a
        # swatch.
        w, h = self._canvas_size
        corners_screen = [
            G.apply_homography((cx, cy), h_c2s)
            for cx, cy in ((0, 0), (w, 0), (0, h), (w, h))
        ]
        xs = [p[0] for p in corners_screen]
        ys = [p[1] for p in corners_screen]
        pad = _SWATCH_MASK_PAD
        x0 = max(0, int(min(xs)) - pad)
        y0 = max(0, int(min(ys)) - pad)
        x1 = int(max(xs)) + pad
        y1 = int(max(ys)) + pad
        masked = screen.copy()
        masked[y0:y1, x0:x1] = 0
        hit = G.find_color_centroid(masked, swatch)
        return None if hit is None else (hit[0], hit[1])

    def _click(self, x: int, y: int) -> None:
        pydirectinput.moveTo(x, y)
        time.sleep(_CLICK_DT)
        pydirectinput.mouseDown()
        time.sleep(_CLICK_DT)
        pydirectinput.mouseUp()
        time.sleep(_CLICK_DT)

    def _drag(self, points: List[Tuple[int, int]]) -> None:
        if not points:
            return
        x0, y0 = points[0]
        pydirectinput.moveTo(x0, y0)
        time.sleep(_MOVE_DT)
        pydirectinput.mouseDown()
        time.sleep(_MOVE_DT)
        for (x, y) in points[1:]:
            pydirectinput.moveTo(x, y)
            time.sleep(_MOVE_DT)
        pydirectinput.mouseUp()
        time.sleep(_MOVE_DT)

    def _launch(self, serve_dir: Optional[str]) -> None:
        if serve_dir is None:
            serve_dir = os.path.join(os.path.dirname(__file__), "canvas_page")
        self._httpd = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8000"],
            cwd=serve_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        candidates = [
            r"C:\Program Files (x86)\Microsoft Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ]
        for exe in candidates:
            if os.path.exists(exe):
                self._browser = subprocess.Popen([
                    exe, f"--app={self._url}",
                    "--window-position=0,0", "--window-size=900,760",
                    "--new-window", "--no-first-run", "--disable-features=Translate",
                    "--user-data-dir=" + self._profile_dir,
                ])
                break
        else:
            import webbrowser
            webbrowser.open(self._url)
        time.sleep(_LOAD_WAIT)
        # Block until the page has painted and the canvas is locatable.
        self._locate_canvas(retries=20, delay=0.5)

    def close(self) -> None:
        for proc in (self._browser, self._httpd):
            if proc is not None:
                try:
                    proc.terminate()
                except Exception:
                    pass
        self._browser = self._httpd = None
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
