# atelier

Atelier (artist's workshop) is a vision-only computer-use agent for painting. It reads the screen, works
out what to paint, and drives a real paint tool with synthetic mouse and keyboard
input. It uses the same inputs a person does.

## How it works

The agent runs a loop over a canvas. It captures the canvas and compares it to a
target image, which gives a per-region error map based on color distance and edge
difference. It picks the region that's furthest off, figures out roughly what color
that region should be, and fills it with brush strokes through the paint tool. Then it
captures again and checks whether the error in that region went down. It
keeps going until the canvas is close to the target.

Everything talks to the environment through one interface, the easel (the surface the
agent paints on), so the core agent has no idea what specific paint program it's driving. The reference environment
is a local browser canvas, driven by screen capture and synthetic input.

## Architecture

The core (`core/`) is environment-agnostic. It only deals with a canvas, a target, and
an abstract set of actions.

- `adapter.py` — the Easel interface. The core talks to environments only through this.
- `perception.py` — compares canvas to target and produces an Observation (error grid,
  global error, heatmap).
- `planner.py` — finds the highest-error region and the color it should be, and returns
  a region-level paint intent.
- `executor.py` — turns an intent into brush strokes and paints them through the easel.
- `verifier.py` — checks whether a stroke reduced its region's error.
- `motion.py` — stroke pacing, so paths render as connected strokes you can actually
  watch.

Environments live in `easels/` and implement the Easel interface. The reference one is
`easels/browser_canvas.py`: DPI-aware screen capture with `mss`, fiducial-based canvas
localization, and synthetic input via `pydirectinput`.

## Status

The perceive, decide, paint, and check pieces are built and tested. The full
planner → executor → easel chain paints a region toward a target end to end. Still to
come: the orchestrator that runs the repeating loop, and the dashboard that visualizes
it.

## Design principles

- Vision-only. The agent only sees screen pixels and only acts through synthetic input.
- Closed-loop. It observes and checks every action before choosing the next one.
- Environment-agnostic core. Anything environment-specific stays behind the Easel
  interface.
- Pluggable planner. The decision-maker is an interface, so the greedy baseline can be
  swapped for something smarter later without touching any of the drawing code.

## Development

```
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt
python -m pytest -q
```