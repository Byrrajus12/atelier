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
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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


@dataclass(frozen=True)
class WidthPreset:
    name: str
    width: float
    locator_color: Color


WIDTH_PRESETS: Tuple[WidthPreset, ...] = (
    WidthPreset("thin", 4.0, (255, 140, 0)),
    WidthPreset("medium", 12.0, (140, 0, 210)),
    WidthPreset("thick", 24.0, (0, 150, 150)),
)

# Fiducials live in the frame padding OUTSIDE the paintable canvas (see index.html),
# so a stroke can never overwrite them. Each centroid sits 15 canvas-px diagonally
# *outside* its corner -> a negative inset (the dst quad expands beyond the canvas).
FIDUCIAL_INSET = -15.0
CANVAS_SIZE = (600, 600)    # canvas-space frame, matches the page's CSS canvas

# The palette swatch column lives just to the RIGHT of the canvas at a fixed layout
# offset (canvas_page/index.html: a 24px wrap gap past the 33px frame padding, then a
# 44px-wide column of three 44px swatches, gap 10px). We search for a swatch ONLY within
# this strip — mapped to the screen through the canvas homography — instead of over the
# whole screen. That removes an entire class of "something off-canvas looks like a
# swatch" bugs: dark window chrome, a dark-themed editor, or the taskbar are all large
# near-black regions that otherwise beat the tiny (44px) near-black swatch and steal the
# click. Coordinates are CANVAS pixels (CSS-px offsets from the canvas top-left, which
# the canvas homography maps to the screen); the box wraps the three swatches with
# ~14px of margin.
CONTROL_COLUMN_WIDTH_CANVAS = 44.0
CONTROL_COLUMN_GAP_CANVAS = 24.0
PALETTE_REGION_CANVAS = (643.0, -47.0, 715.0, 133.0)  # (x0, y0, x1, y1) in canvas px
WIDTH_REGION_CANVAS = tuple(
    x + CONTROL_COLUMN_WIDTH_CANVAS + CONTROL_COLUMN_GAP_CANVAS if i in (0, 2) else x
    for i, x in enumerate(PALETTE_REGION_CANVAS)
)

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


def nearest_width_preset(
    requested: float,
    presets: Tuple[WidthPreset, ...] = WIDTH_PRESETS,
) -> WidthPreset:
    """Pick the discrete width preset nearest the requested canvas-pixel width."""
    return min(presets, key=lambda p: abs(p.width - requested))


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

    def realizable_width(self, requested: float) -> float:
        return nearest_width_preset(requested).width

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
        self.apply_strokes((stroke,))

    def apply_strokes(self, strokes: Sequence[Stroke]) -> None:
        if not strokes:
            return
        brush = strokes[0].brush
        if any(stroke.brush != brush for stroke in strokes):
            raise ValueError("BrowserCanvasEasel.apply_strokes requires one shared brush")

        from easels import _geometry as G

        screen, corners = self._locate_canvas()
        _, h_c2s = G.canvas_homographies(corners, self._canvas_size, FIDUCIAL_INSET)

        # 1. Realize the brush width and color by clicking the nearest controls. Both
        #    controls are located by vision-searching restricted screen regions
        #    projected through the canvas homography. A missing control is a hard
        #    failure: painting on with an unknown brush would be a silent,
        #    unverifiable error, so we raise rather than guess.
        width = nearest_width_preset(brush.size)
        width_loc = self._locate_width_button(screen, h_c2s, width.locator_color)
        if width_loc is None:
            raise LookupError(f"width button for preset {width.name} not found on screen")
        self._click(int(width_loc[0]), int(width_loc[1]))

        swatch = nearest_palette_color(brush.color)
        loc = self._locate_swatch(screen, h_c2s, swatch)
        if loc is None:
            raise LookupError(f"palette swatch for color {swatch} not found on screen")
        self._click(int(loc[0]), int(loc[1]))

        # 2. Map the canvas-space paths to screen pixels and drag them.
        for stroke in strokes:
            screen_path = [G.apply_homography((p.x, p.y), h_c2s) for p in stroke.path]
            self._drag([(int(x), int(y)) for (x, y) in screen_path])

    # --- environment plumbing ----------------------------------------------------
    def _grab_screen(self) -> np.ndarray:
        if self._sct is None:
            self._sct = mss.mss()
        mon = self._sct.monitors[1]
        raw = np.array(self._sct.grab(mon))     # BGRA
        return raw[:, :, [2, 1, 0]].copy()      # -> RGB

    def _project_canvas_region(self, region_canvas, h_c2s, screen_shape):
        """Screen-pixel bounding box of a canvas-space control strip,
        mapped through the canvas->screen homography and clipped to the capture. Returns
        ``(x0, y0, x1, y1)`` or ``None`` if it falls entirely off-screen."""
        from easels import _geometry as G

        x0c, y0c, x1c, y1c = region_canvas
        pts = [
            G.apply_homography(c, h_c2s)
            for c in ((x0c, y0c), (x1c, y0c), (x0c, y1c), (x1c, y1c))
        ]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        h, w = screen_shape[:2]
        x0 = max(0, int(min(xs)))
        y0 = max(0, int(min(ys)))
        x1 = min(w, int(max(xs)) + 1)
        y1 = min(h, int(max(ys)) + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    def _palette_region(self, h_c2s, screen_shape):
        return self._project_canvas_region(PALETTE_REGION_CANVAS, h_c2s, screen_shape)

    def _width_region(self, h_c2s, screen_shape):
        return self._project_canvas_region(WIDTH_REGION_CANVAS, h_c2s, screen_shape)

    def _locate_swatch(self, screen, h_c2s, swatch: Color):
        from easels import _geometry as G

        # Search ONLY the palette strip (right of the canvas), not the whole screen, so
        # no off-canvas near-color region can be mistaken for a swatch. This is robust for
        # ALL swatches, and especially the near-black one, whose color is common off-canvas
        # (dark chrome/editor/taskbar) and previously lost the largest-blob contest.
        region = self._palette_region(h_c2s, screen.shape)
        if region is None:
            return None
        rx0, ry0, rx1, ry1 = region
        hit = G.find_color_centroid(screen[ry0:ry1, rx0:rx1], swatch)
        if hit is None:
            return None
        return (hit[0] + rx0, hit[1] + ry0)  # crop-local -> screen coords

    def _locate_width_button(self, screen, h_c2s, color: Color):
        from easels import _geometry as G

        region = self._width_region(h_c2s, screen.shape)
        if region is None:
            return None
        rx0, ry0, rx1, ry1 = region
        hit = G.find_color_centroid(screen[ry0:ry1, rx0:rx1], color)
        if hit is None:
            return None
        return (hit[0] + rx0, hit[1] + ry0)  # crop-local -> screen coords

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
