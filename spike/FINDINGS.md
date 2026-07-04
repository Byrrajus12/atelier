# Milestone 1 — Spike findings

The spike (`spike/spike.py`, throwaway) drove the reference browser canvas
(`easels/canvas_page/index.html`) end to end using **only** `mss` screen capture and
`pydirectinput` real-cursor input, and confirmed a stroke by re-capture. This file
records the real, measured truths that must shape the Easel interface (M2). The spike
code itself is now disposable; **these findings are the deliverable.**

Verdict: **PASS** — a red zigzag stroke appeared on the canvas, observable only by
re-capturing the screen. Nothing was read from the page's JS, DOM, or canvas API.

## Environment as measured (this machine, this run)
- Primary monitor: **2560 × 1440 physical pixels**, origin (0,0) (`mss.monitors[1]`).
- Windows display scaling: **150%** → browser `devicePixelRatio ≈ 1.5`.
- The 600 CSS-px canvas rendered to **~900 device px** (the 18px corner fiducials sit
  inset, so the measured centroid-to-centroid span was 874 px = 582 CSS · 1.5).

## The coordinate-transform truth (the thing we most needed to nail)
This is the load-bearing finding. There are three coordinate spaces in play:

1. **CSS pixels** — the page's logical units (canvas is 600×600 CSS).
2. **Device / captured pixels** — what `mss` returns (physical, 2560×1440), scaled
   from CSS by `devicePixelRatio`.
3. **Cursor pixels** — what `pydirectinput` expects for absolute `moveTo`.

Resolution that works cleanly: **make the process per-monitor-DPI-aware V2 at startup,
before importing `mss`/`pydirectinput`** (`SetProcessDpiAwarenessContext(-4)`, with
fallbacks). With that:
- `mss` capture space and `pydirectinput` cursor space become the **same physical
  pixel space** — `GetSystemMetrics(0/1)` returned `2560×1440` (physical, not the
  scaled 1707×960), and a feature located at captured pixel `(Px,Py)` is reached by
  `moveTo(Px, Py)` directly. **Scale factor between capture and cursor = 1.**
- `devicePixelRatio` never enters our screen-space math: we locate the canvas by
  fiducials **in captured pixels** and act in captured pixels. dPR only governs the
  CSS→canvas mapping *inside* the page, which the page handles itself.

Implication: DPI awareness must be established process-wide, once, before any Win32
capture/input state is touched. This belongs in the concrete browser Easel's setup,
NOT in `core/`.

## Canvas localization (vision-only, no window-position assumptions)
- Four corner fiducials in pure locator colors (magenta TL, cyan TR, yellow BL,
  green BR) are found by **largest connected color-blob centroid**
  (`scipy.ndimage.label`), giving the canvas quad in captured pixels regardless of
  where the window is.
- **Caveat for M2:** fiducial *centroids* are inset from the true canvas corners by
  half the fiducial size (9 CSS px here). The naive `span/600` gave 1.457 instead of
  1.5 for exactly this reason. The Easel's canvas→screen mapping must account for the
  inset (or the target must be mapped to the centroid quad consistently, end to end).
- Mapping canvas (u,v)∈[0,1] → screen used **bilinear interpolation** over the four
  corners. A homography would be more correct for a skewed/tilted quad; bilinear was
  sufficient for an axis-aligned browser window and is a fine starting point.

## Observability
- Stroke results are observable **only by re-capture.** There is no readback path —
  we diffed `before`/`after` captures (`abs RGB delta > 40`), got 17,162 changed px
  (2.25% of the canvas region), with a change bounding box matching the drawn zigzag.
- This validates design Principle 3 (closed-loop): every action must be verified by a
  fresh capture; there is no cheaper signal.

## Acting on the canvas
- **Color/brush selection is itself a vision-only click**: locate the palette swatch
  by its color, `moveTo` + `mouseDown`/`mouseUp` on it. There is no color "API" — the
  Easel's `set_color`/`select_brush` will be composed of the same move+click that
  strokes use. (Because painting red then makes red appear in the canvas, swatch
  location must happen on a *pre-paint* capture, or swatches need locator colors
  distinct from paint colors.)
- **Strokes** = `moveTo(start)` → `mouseDown` → `moveTo` through path points (button
  held) → `mouseUp`.

## Timing that worked
- ~**3.0 s** wait after browser launch for the page to load and paint.
- **~30 ms** sleep between successive drag `moveTo`s, plus ~50 ms around
  `mouseDown`/`mouseUp`. Below this the browser can miss intermediate pointer moves.
- `pydirectinput.FAILSAFE = False` (so nearing a screen corner doesn't abort) and
  `PAUSE = 0.0` (we insert explicit sleeps) were both necessary.
- Coordinates passed to `pydirectinput` must be **ints**.

## What the spike actually needed → seeds for the Easel interface (M2)
The concrete operations the spike used, which the `Easel` ABC should factor out:
- `capture() -> Frame` — full RGB screen grab (physical pixels).
- canvas localization → a **canvas→screen coordinate mapping** (`canvas_to_screen(u,v)`
  or `(px,py)`), derived from fiducials, hiding the DPI/inset details.
- `move_to(pt)`, and a `stroke(path)` built from move + down + timed moves + up.
- `set_color(...)` / `select_brush(...)` — themselves vision-only clicks on the UI.
- `capabilities() -> {reversible, has_undo, stroke_cost}` — the browser canvas is
  effectively **not reversible without an explicit undo affordance** (this page has
  none), so the core must treat strokes as committing. This directly informs
  Principle 7 handling in the verifier.

## Explicitly out of scope for the spike (left undone on purpose)
- No sub-pixel/homography calibration, no multi-monitor handling, no window focus
  management, no brush-size control — all deferred to the real Easel where needed.
- The spike leaves a `_spike_profile/` browser profile and PNGs behind; disposable.
