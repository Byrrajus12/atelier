"""Tests for core/executor.py - scanline fill + the executor driving a FAKE easel
(no browser). These tests are the executable executor contract: scanline fills only,
coverage at the supported reference widths, bounded edge bleed, and routing through the
Easel interface only."""

import math

import numpy as np
import pytest

from core.adapter import Capabilities, Easel, Frame, Point, Stroke
from core.executor import DEFAULT_BRUSH_WIDTH, DEFAULT_SPACING, Executor, scanline_fill
from core.motion import densify
from core.planner import PaintIntent
from scripts.check_edges import (
    _bleed_report,
    _interior_uncovered,
    _rasterize_alpha,
    _realized_paths,
    _visible_mask,
)


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


class FixedWidthEasel(RecordingEasel):
    def __init__(self, realized_width, size=(600, 600)):
        super().__init__(size=size)
        self.realized_width = realized_width

    def realizable_width(self, requested: float) -> float:
        return self.realized_width


class BatchRecordingEasel(RecordingEasel):
    def __init__(self, size=(600, 600)):
        super().__init__(size=size)
        self.batches = []

    def apply_strokes(self, strokes):
        self.batches.append(tuple(strokes))
        self.strokes.extend(strokes)


def intent(box, color=(200, 40, 60), size=12.0):
    return PaintIntent(cell=(0, 0), box=box, color=color, error=0.9, size=size)


def _scanline_ys(paths):
    return sorted({p.y for path in paths for p in path})


def _acceptance(width, box=(200, 200, 400, 400)):
    spacing = width * DEFAULT_SPACING / DEFAULT_BRUSH_WIDTH
    paths = _realized_paths(tuple(densify(path) for path in scanline_fill(box, spacing, width)))
    pad = int(width)
    visible = _visible_mask(_rasterize_alpha(paths, width, "butt", box, pad))
    return _interior_uncovered(visible, box, pad), _bleed_report(visible, box, pad)


def test_executor_contract_fill_coverage_and_edge_bleed_at_reference_widths():
    """Executor fill contract, tested against the reference page's butt-cap raster model.

    The executor emits independent horizontal scanline strokes. At each reference
    browser width it fully covers the half-open box interior, adds no lateral edge
    connector bleed, and leaves only the inherent top/bottom brush footprint bounded by
    half the realized brush width.
    """
    for width in (4.0, 12.0, 24.0):
        uncovered, bleed = _acceptance(width)
        assert uncovered == 0
        assert bleed["left"]["pixels"] == 0
        assert bleed["right"]["pixels"] == 0
        assert bleed["top"]["max_depth"] <= math.ceil(width / 2)
        assert bleed["bottom"]["max_depth"] <= math.ceil(width / 2)


def test_scanline_spacing_never_exceeds_brush_width():
    paths = scanline_fill((0, 0, 37, 37), spacing=10.0, brush_width=12.0)
    gaps = [b - a for a, b in zip(_scanline_ys(paths), _scanline_ys(paths)[1:])]
    assert gaps
    assert max(gaps) <= 12.0 + 1e-9


def test_scanline_paths_are_independent_horizontal_strokes():
    box = (100, 50, 140, 90)
    paths = scanline_fill(box, spacing=10.0, brush_width=12.0)
    assert len(paths) > 1
    for path in paths:
        assert len(path) == 2
        assert path[0].y == path[1].y
        assert {path[0].x, path[1].x} == {float(box[0]), float(box[2])}


def test_scanline_degenerate_thin_box():
    paths = scanline_fill((10, 10, 11, 11), spacing=10.0, brush_width=12.0)
    assert paths == ((Point(10.0, 10.0), Point(11.0, 10.0)),)


def test_scanline_zero_area_box_does_not_crash():
    for box in [(10, 10, 10, 10), (10, 10, 10, 20), (10, 10, 20, 10)]:
        assert len(scanline_fill(box, spacing=10.0, brush_width=12.0)) >= 1


def test_scanline_fill_enforces_spacing_le_brush_width():
    with pytest.raises(ValueError):
        scanline_fill((0, 0, 40, 40), spacing=20.0, brush_width=12.0)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            scanline_fill((0, 0, 40, 40), spacing=bad, brush_width=12.0)
        with pytest.raises(ValueError):
            scanline_fill((0, 0, 40, 40), spacing=10.0, brush_width=bad)


def test_executor_paints_scanline_strokes_with_intent_color():
    easel = RecordingEasel()
    strokes = Executor(easel).execute(intent((100, 50, 140, 90), color=(10, 220, 30)))

    assert strokes == easel.strokes
    assert len(strokes) > 1
    assert all(isinstance(stroke, Stroke) for stroke in strokes)
    assert {stroke.brush.color for stroke in strokes} == {(10, 220, 30)}


def test_executor_applies_one_batch_per_intent():
    easel = BatchRecordingEasel()
    strokes = Executor(easel).execute(intent((100, 50, 140, 90)))

    assert len(easel.batches) == 1
    assert easel.batches[0] == tuple(strokes)


def test_executor_default_geometry_matches_scanline_fill():
    easel = RecordingEasel()
    box = (100, 50, 140, 90)
    Executor(easel).execute(intent(box))
    assert [stroke.path for stroke in easel.strokes] == list(
        scanline_fill(box, spacing=DEFAULT_SPACING, brush_width=DEFAULT_BRUSH_WIDTH)
    )


def test_executor_spacing_derives_from_realized_width():
    easel = FixedWidthEasel(realized_width=24.0)
    box = (0, 0, 37, 37)
    Executor(easel).execute(
        PaintIntent(cell=(0, 0), box=box, color=(1, 2, 3), error=0.5, size=4.0)
    )
    assert [stroke.path for stroke in easel.strokes] == list(
        scanline_fill(box, spacing=18.648, brush_width=24.0)
    )


def test_executor_fill_geometry_stays_on_horizontal_box_boundaries():
    easel = RecordingEasel()
    box = (100, 50, 140, 90)
    Executor(easel).execute(intent(box))
    xs = [p.x for stroke in easel.strokes for p in stroke.path]
    ys = [p.y for stroke in easel.strokes for p in stroke.path]
    assert min(xs) == box[0] and max(xs) == box[2]
    assert min(ys) == box[1] and max(ys) == box[3] - 1


def test_executor_emits_each_fill_pass_as_a_single_two_point_segment():
    """Fill passes are raw 2-point strokes (start -> end), never densified: joining
    butt-capped sub-segments left 1px gaps between them ("corduroy"), so the executor
    must hand each scanline to the Easel as one segment, not many."""
    easel = RecordingEasel()
    Executor(easel).execute(intent((0, 0, 120, 120)))
    assert easel.strokes
    assert all(len(stroke.path) == 2 for stroke in easel.strokes)


def test_executor_size_hint_passed_through():
    easel = RecordingEasel()
    Executor(easel).execute(intent((0, 0, 40, 40), color=(1, 2, 3), size=9.0))
    assert {stroke.brush.size for stroke in easel.strokes} == {9.0}


def test_executor_rejects_bad_params():
    easel = RecordingEasel()
    with pytest.raises(ValueError):
        Executor(easel, spacing_ratio=0)
    with pytest.raises(ValueError):
        Executor(easel, spacing_ratio=1.1)
