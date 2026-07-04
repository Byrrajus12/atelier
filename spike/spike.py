"""THROWAWAY vision-only spike (Milestone 1).

Not part of core/. Deliberately procedural and hardcoded. Its only job is to prove
the vision-only path on a real browser canvas and surface the messy truths that will
drive the Easel interface (M2). Once spike/FINDINGS.md records those truths, this
file is disposable.

What it does, end to end:
  1. Make the process DPI-aware so screen-capture pixels and cursor coordinates share
     one coordinate space (the coordinate-transform truth we most need to nail down).
  2. Launch a chromeless browser window on the local canvas page.
  3. Screen-capture (mss) -> before.png.
  4. Locate the canvas by its corner fiducials, purely from captured pixels.
  5. Select a paint color by moving the REAL cursor to a palette swatch and clicking.
  6. Lay one stroke: move -> mouseDown -> drag through points -> mouseUp.
  7. Screen-capture -> after.png, diff, and assert pixels changed in the canvas.

Run:  .venv/Scripts/python.exe spike/spike.py
"""

import ctypes
import os
import subprocess
import sys
import time

# --- 1. DPI awareness FIRST, before mss / pydirectinput touch any Win32 state. -----
def set_dpi_aware():
    """Make this process per-monitor DPI aware so mss (physical pixels) and
    pydirectinput (SetCursorPos / SendInput, normalized against GetSystemMetrics)
    reason in the SAME physical pixel space. Returns the level actually applied."""
    try:
        # PER_MONITOR_AWARE_V2 = -4
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return "per-monitor-v2"
    except Exception:
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return "per-monitor"
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()  # system aware
        return "system"
    except Exception:
        return "none"


DPI_LEVEL = set_dpi_aware()

import numpy as np  # noqa: E402
import mss  # noqa: E402
import pydirectinput  # noqa: E402
from scipy import ndimage  # noqa: E402

pydirectinput.FAILSAFE = False   # do not abort if the cursor nears a screen corner
pydirectinput.PAUSE = 0.0        # we insert our own explicit sleeps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
URL = "http://localhost:8000/index.html"
OUT = HERE  # write before/after/diff PNGs next to this script

# Pure locator colors baked into the page (RGB).
FID_MAGENTA = (255, 0, 255)   # top-left
FID_CYAN = (0, 255, 255)      # top-right
FID_YELLOW = (255, 255, 0)    # bottom-left
FID_GREEN = (0, 255, 0)       # bottom-right
SWATCH_RED = (255, 0, 0)      # the color we will paint with


def start_http_server():
    return subprocess.Popen(
        [sys.executable, "-m", "http.server", "8000"],
        cwd=os.path.join(ROOT, "easels", "canvas_page"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def launch_browser():
    """Open a chromeless --app window at the top-left. Fiducial-based location means
    the exact position does not matter, but a chromeless window keeps stray UI colors
    out of the capture."""
    candidates = [
        r"C:\Program Files (x86)\Microsoft Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for exe in candidates:
        if os.path.exists(exe):
            return subprocess.Popen([
                exe,
                f"--app={URL}",
                "--window-position=0,0",
                "--window-size=900,760",
                "--new-window",
                "--no-first-run",
                "--disable-features=Translate",
                r"--user-data-dir=" + os.path.join(HERE, "_spike_profile"),
            ]), exe
    # Fallback: default browser (will have chrome/tabs, but fiducials still locate it).
    import webbrowser
    webbrowser.open(URL)
    return None, "default-browser"


def grab_screen(sct):
    mon = sct.monitors[1]  # primary monitor, physical pixels
    raw = np.array(sct.grab(mon))          # H x W x 4, BGRA
    rgb = raw[:, :, [2, 1, 0]].copy()      # -> RGB
    return rgb, mon


def find_blob_centroid(rgb, target, tol=60):
    """Return (cx, cy, area) of the largest connected region matching `target`,
    in captured-image pixel coords, or None."""
    diff = np.abs(rgb.astype(np.int16) - np.array(target, dtype=np.int16)).sum(axis=2)
    mask = diff < tol
    if not mask.any():
        return None
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    sizes = ndimage.sum(mask, labels, index=range(1, n + 1))
    biggest = int(np.argmax(sizes)) + 1
    ys, xs = np.where(labels == biggest)
    return int(xs.mean()), int(ys.mean()), int(sizes[biggest - 1])


def quad_point(corners, u, v):
    """Bilinear interpolation over the canvas quad. corners = (TL, TR, BL, BR),
    each an (x, y) tuple; u,v in [0,1] -> screen pixel (x, y)."""
    tl, tr, bl, br = (np.array(c, dtype=float) for c in corners)
    top = tl * (1 - u) + tr * u
    bot = bl * (1 - u) + br * u
    p = top * (1 - v) + bot * v
    return int(round(p[0])), int(round(p[1]))


def click(x, y):
    pydirectinput.moveTo(x, y)
    time.sleep(0.05)
    pydirectinput.mouseDown()
    time.sleep(0.05)
    pydirectinput.mouseUp()
    time.sleep(0.05)


def drag_stroke(points, hold=0.03):
    """points: list of (x, y) in screen coords. Press at the first, drag through the
    rest with the button held, release at the last."""
    x0, y0 = points[0]
    pydirectinput.moveTo(x0, y0)
    time.sleep(hold)
    pydirectinput.mouseDown()
    time.sleep(hold)
    for (x, y) in points[1:]:
        pydirectinput.moveTo(x, y)
        time.sleep(hold)
    pydirectinput.mouseUp()
    time.sleep(hold)


def save_png(rgb, path):
    from PIL import Image
    Image.fromarray(rgb.astype(np.uint8), "RGB").save(path)


def main():
    print(f"[dpi] process DPI awareness applied: {DPI_LEVEL}")
    log = ctypes.windll.user32.GetSystemMetrics
    print(f"[dpi] GetSystemMetrics logical primary size: "
          f"{log(0)} x {log(1)}")

    httpd = start_http_server()
    proc, exe = launch_browser()
    print(f"[browser] launched via: {exe}")
    print("[browser] waiting for page load / paint...")
    time.sleep(3.0)

    with mss.mss() as sct:
        before, mon = grab_screen(sct)
        print(f"[capture] mss physical monitor: {mon['width']} x {mon['height']} "
              f"@ ({mon['left']},{mon['top']})")
        print(f"[capture] captured array: {before.shape[1]} x {before.shape[0]}")

        # Locate the four fiducials in captured pixels.
        tl = find_blob_centroid(before, FID_MAGENTA)
        tr = find_blob_centroid(before, FID_CYAN)
        bl = find_blob_centroid(before, FID_YELLOW)
        br = find_blob_centroid(before, FID_GREEN)
        for name, f in [("TL/magenta", tl), ("TR/cyan", tr),
                        ("BL/yellow", bl), ("BR/green", br)]:
            print(f"[fiducial] {name:12s} -> {f}")
        if not all([tl, tr, bl, br]):
            print("[FATAL] could not locate all fiducials. Is the window visible / "
                  "on top? Aborting.")
            save_png(before, os.path.join(OUT, "before.png"))
            return 2

        corners = ((tl[0], tl[1]), (tr[0], tr[1]), (bl[0], bl[1]), (br[0], br[1]))
        canvas_w = tr[0] - tl[0]
        canvas_h = bl[1] - tl[1]
        print(f"[geometry] canvas fiducial span in captured px: "
              f"{canvas_w} x {canvas_h}")
        print(f"[geometry] implied device-pixel scale (span / 600 CSS px): "
              f"{canvas_w / 600:.3f} x {canvas_h / 600:.3f}")

        # Select red: locate the red swatch (only red thing before we paint) & click.
        red = find_blob_centroid(before, SWATCH_RED)
        print(f"[color] red swatch centroid: {red}")
        if red:
            click(red[0], red[1])
            time.sleep(0.2)

        # Lay one zigzag stroke well inside the canvas quad.
        uv = [(0.25, 0.55), (0.45, 0.30), (0.60, 0.60), (0.78, 0.35)]
        pts = [quad_point(corners, u, v) for (u, v) in uv]
        print(f"[stroke] screen points: {pts}")
        drag_stroke(pts)
        time.sleep(0.4)

        after, _ = grab_screen(sct)

    # Diff within the canvas bounding box.
    x0 = min(tl[0], bl[0]); x1 = max(tr[0], br[0])
    y0 = min(tl[1], tr[1]); y1 = max(bl[1], br[1])
    b = before[y0:y1, x0:x1].astype(np.int16)
    a = after[y0:y1, x0:x1].astype(np.int16)
    delta = np.abs(a - b).sum(axis=2)
    changed = int((delta > 40).sum())
    total = delta.size
    print(f"[diff] changed pixels in canvas region: {changed} / {total} "
          f"({100 * changed / total:.2f}%)")
    if changed:
        ys, xs = np.where(delta > 40)
        print(f"[diff] change bounding box (within canvas region): "
              f"x[{xs.min()}..{xs.max()}] y[{ys.min()}..{ys.max()}]")

    save_png(before, os.path.join(OUT, "before.png"))
    save_png(after, os.path.join(OUT, "after.png"))
    heat = np.zeros_like(before)
    heat[y0:y1, x0:x1, 0] = np.clip(delta, 0, 255).astype(np.uint8)
    save_png(heat, os.path.join(OUT, "diff.png"))
    print(f"[out] wrote before.png / after.png / diff.png to {OUT}")

    verdict = "PASS" if changed > 200 else "FAIL"
    print(f"\n[VERDICT] vision-only stroke observable by re-capture: {verdict}")

    try:
        httpd.terminate()
    except Exception:
        pass
    return 0 if changed > 200 else 1


if __name__ == "__main__":
    sys.exit(main())
