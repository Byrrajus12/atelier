"""Tests for core/executor.py — scanline fill + the executor driving a FAKE easel
(no browser). Verifies coverage (no gap wider than the brush), that the fill stays on
the intent's box, and that the executor produces the right stroke for an intent and
routes it through the Easel interface only."""

import numpy as np
import pytest

from core.adapter import BrushSpec, Capabilities, Easel, Frame, Stroke
from core.executor import Executor, scanline_fill
from core.planner import PaintIntent


# --- a recording Easel: records strokes, no screen/cursor ------------------------
class RecordingEasel(Easel):
    def __init__(self, size=(600, 600)):
        self._size = size
        self.strokes = []

    def capabilities(self):
        return Capabilities(reversible=False, has_undo=False)

    def canvas_size(self):
        return self._size

    def capture(self):
        w, h = self._size
        return Frame(np.zeros((h, w, 3), dtype=np.uint8))

    def apply_stroke(self, stroke: Stroke) -> None:
        self.strokes.append(stroke)


def intent(box, color=(200, 40, 60)):
    return PaintIntent(cell=(0, 0), box=box, color=color, error=0.9)


# --- scanline_fill: coverage -----------------------------------------------------
def test_scanline_covers_every_row_within_half_brush():
    box = (100, 50, 140, 90)  # 40x40
    brush = 12.0
    path = scanline_fill(box, spacing=10.0, brush_width=brush)
    scan_ys = sorted({p.y for p in path})
    # every pixel row in [y0, y1) is within brush/2 of some scanline
    for y in range(box[1], box[3]):
        assert min(abs(y - sy) for sy in scan_ys) <= brush / 2 + 1e-9


def test_scanline_spacing_never_exceeds_brush_width():
    path = scanline_fill((0, 0, 37, 37), spacing=10.0, brush_width=12.0)
    scan_ys = sorted({p.y for p in path})
    gaps = [b - a for a, b in zip(scan_ys, scan_ys[1:])]
    assert gaps  # multiple scanlines
    assert max(gaps) <= 12.0 + 1e-9


def test_scanline_spans_the_box_width():
    box = (100, 50, 140, 90)
    path = scanline_fill(box, spacing=10.0, brush_width=12.0)
    xs = [p.x for p in path]
    assert min(xs) == box[0]              # reaches the left edge
    assert max(xs) == box[2] - 1          # ...and the last pixel column


def test_scanline_is_serpentine_connected():
    box = (0, 0, 30, 40)
    path = scanline_fill(box, spacing=10.0, brush_width=12.0)
    # consecutive points share either a row (horizontal pass) or a column (vertical
    # connector) — i.e. it is one continuous polyline, never a disjoint jump.
    for a, b in zip(path, path[1:]):
        assert (a.y == b.y) or (a.x == b.x)


def test_scanline_degenerate_thin_box():
    path = scanline_fill((10, 10, 11, 11), spacing=10.0, brush_width=12.0)
    assert len(path) >= 1  # a valid (dab-like) path, no crash


def test_scanline_zero_area_box_does_not_crash():
    for box in [(10, 10, 10, 10), (10, 10, 10, 20), (10, 10, 20, 10)]:
        assert len(scanline_fill(box, spacing=10.0, brush_width=12.0)) >= 1


def test_scanline_fill_enforces_spacing_le_brush_width():
    # The pure function now owns its coverage precondition (not just the Executor).
    with pytest.raises(ValueError):
        scanline_fill((0, 0, 40, 40), spacing=20.0, brush_width=12.0)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            scanline_fill((0, 0, 40, 40), spacing=bad, brush_width=12.0)
        with pytest.raises(ValueError):
            scanline_fill((0, 0, 40, 40), spacing=10.0, brush_width=bad)


# --- Executor over the fake easel ------------------------------------------------
def test_executor_paints_one_stroke_with_intent_color():
    easel = RecordingEasel()
    ex = Executor(easel)
    strokes = ex.execute(intent((100, 50, 140, 90), color=(10, 220, 30)))

    assert len(easel.strokes) == 1
    assert strokes == easel.strokes            # returns what it applied
    laid = easel.strokes[0]
    assert isinstance(laid, Stroke)
    assert laid.brush.color == (10, 220, 30)   # realizes the intent's color


def test_executor_fill_stays_on_the_intent_box():
    easel = RecordingEasel()
    box = (100, 50, 140, 90)
    Executor(easel).execute(intent(box))
    pts = easel.strokes[0].path
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    # the fill's bounding box matches the intent box's pixel extent (position-verified)
    assert min(xs) == box[0] and max(xs) == box[2] - 1
    assert min(ys) == box[1] and max(ys) == box[3] - 1


def test_executor_densifies_for_watchability():
    easel = RecordingEasel()
    # a coarse fill vs the default: smaller max_step_px => strictly more points
    Executor(easel, max_step_px=3.0).execute(intent((0, 0, 120, 120)))
    fine = len(easel.strokes[0].path)
    easel2 = RecordingEasel()
    Executor(easel2, max_step_px=12.0).execute(intent((0, 0, 120, 120)))
    coarse = len(easel2.strokes[0].path)
    assert fine > coarse


def test_executor_size_hint_passed_through():
    easel = RecordingEasel()
    it = PaintIntent(cell=(0, 0), box=(0, 0, 40, 40), color=(1, 2, 3), error=0.5, size=9.0)
    Executor(easel).execute(it)
    assert easel.strokes[0].brush.size == 9.0  # carried, even though easel ignores it


def test_executor_rejects_bad_params():
    easel = RecordingEasel()
    with pytest.raises(ValueError):
        Executor(easel, brush_width=0)
    with pytest.raises(ValueError):
        Executor(easel, spacing=0)
    with pytest.raises(ValueError):
        Executor(easel, spacing=20, brush_width=12)  # passes wouldn't overlap
    with pytest.raises(ValueError):
        Executor(easel, max_step_px=0)
