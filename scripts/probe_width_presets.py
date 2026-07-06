"""Manual M9 width-preset probe.

Runs two one-stroke PaintIntents through the real browser canvas easel: one thick,
then one thin. This is intentionally not the closed-loop run; it is a small live
verification tool so a human can watch the agent vision-locate the width buttons and
see visibly different stroke widths land.

Run from the repo root:
    .venv/Scripts/python.exe scripts/probe_width_presets.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.executor import Executor  # noqa: E402
from core.planner import PaintIntent  # noqa: E402
from easels.browser_canvas import BrowserCanvasEasel  # noqa: E402


def main() -> int:
    easel = BrowserCanvasEasel(launch=True)
    try:
        ex = Executor(easel)

        print("thick probe: expect click on THICK width button, then a thick red line")
        ex.execute(
            PaintIntent(
                cell=(0, 0),
                box=(80, 170, 520, 171),
                color=(255, 0, 0),
                error=1.0,
                size=24.0,
            )
        )

        time.sleep(0.8)

        print("thin probe: expect click on THIN width button, then a thin blue line")
        ex.execute(
            PaintIntent(
                cell=(0, 0),
                box=(80, 320, 520, 321),
                color=(0, 0, 255),
                error=1.0,
                size=4.0,
            )
        )

        print("done: compare the red thick line with the blue thin line")
        input("press Enter to close the probe browser...")
        return 0
    finally:
        easel.close()


if __name__ == "__main__":
    raise SystemExit(main())
