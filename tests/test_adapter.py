"""Tests for the Easel interface (core/adapter.py) — the sole core/environment
contract. These cover the value types' invariants and prove the ABC is implementable
and behaves (abstractness, default undo, context-manager close)."""

import numpy as np
import pytest

from core.adapter import (
    BrushSpec,
    Capabilities,
    Color,
    Easel,
    Frame,
    Point,
    Stroke,
    UnsupportedOperation,
)


# --- a minimal concrete Easel, used to exercise the ABC --------------------------
class FakeEasel(Easel):
    """In-memory Easel: records strokes, hands back a blank canvas. No screen, no
    cursor — lets us test the contract itself headlessly."""

    def __init__(self, size=(20, 10), caps=None):
        self._size = size
        self._caps = caps or Capabilities(reversible=False, has_undo=False)
        self.strokes = []
        self.closed = False

    def capabilities(self) -> Capabilities:
        return self._caps

    def canvas_size(self):
        return self._size

    def capture(self) -> Frame:
        w, h = self._size
        return Frame(np.zeros((h, w, 3), dtype=np.uint8), timestamp=1.0)

    def apply_stroke(self, stroke: Stroke) -> None:
        self.strokes.append(stroke)

    def close(self) -> None:
        self.closed = True


# --- Point -----------------------------------------------------------------------
def test_point_is_tuple_like():
    p = Point(3.0, 4.5)
    assert (p.x, p.y) == (3.0, 4.5)
    assert tuple(p) == (3.0, 4.5)  # unpacks like a plain coordinate


# --- Capabilities ----------------------------------------------------------------
def test_capabilities_defaults_neutral_cost():
    assert Capabilities(reversible=True, has_undo=False).stroke_cost == 1.0


def test_capabilities_rejects_negative_cost():
    with pytest.raises(ValueError):
        Capabilities(reversible=False, has_undo=False, stroke_cost=-0.1)


def test_capabilities_undo_requires_reversible():
    with pytest.raises(ValueError):
        Capabilities(reversible=False, has_undo=True)


def test_capabilities_is_frozen():
    caps = Capabilities(reversible=True, has_undo=True)
    with pytest.raises(Exception):
        caps.reversible = False  # type: ignore[misc]


# --- BrushSpec -------------------------------------------------------------------
def test_brush_valid():
    b = BrushSpec(color=(255, 0, 0), size=8)
    assert b.color == (255, 0, 0) and b.size == 8


def test_brush_rejects_nonpositive_size():
    with pytest.raises(ValueError):
        BrushSpec(color=(0, 0, 0), size=0)


@pytest.mark.parametrize("bad", [(256, 0, 0), (-1, 0, 0), (0, 0), (0, 0, 0, 0)])
def test_brush_rejects_bad_color(bad):
    with pytest.raises(ValueError):
        BrushSpec(color=bad)  # type: ignore[arg-type]


# --- Stroke ----------------------------------------------------------------------
def test_stroke_requires_a_point():
    with pytest.raises(ValueError):
        Stroke(path=(), brush=BrushSpec(color=(0, 0, 0)))


def test_stroke_holds_path_and_brush():
    s = Stroke(path=(Point(0, 0), Point(5, 5)), brush=BrushSpec(color=(1, 2, 3)))
    assert len(s.path) == 2 and s.brush.color == (1, 2, 3)


# --- Frame -----------------------------------------------------------------------
def test_frame_size_is_width_height():
    f = Frame(np.zeros((10, 20, 3), dtype=np.uint8))
    assert f.size == (20, 10)  # (width, height)


def test_frame_rejects_non_rgb():
    with pytest.raises(ValueError):
        Frame(np.zeros((10, 20), dtype=np.uint8))  # missing channel axis
    with pytest.raises(ValueError):
        Frame(np.zeros((10, 20, 4), dtype=np.uint8))  # RGBA, not RGB


# --- Easel ABC -------------------------------------------------------------------
def test_easel_is_abstract():
    with pytest.raises(TypeError):
        Easel()  # type: ignore[abstract]


def test_fake_easel_records_and_captures():
    e = FakeEasel(size=(20, 10))
    assert e.canvas_size() == (20, 10)
    frame = e.capture()
    assert frame.size == (20, 10)
    stroke = Stroke(path=(Point(1, 1), Point(2, 2)), brush=BrushSpec(color=(9, 9, 9)))
    e.apply_stroke(stroke)
    assert e.strokes == [stroke]


def test_undo_unsupported_by_default():
    e = FakeEasel()
    with pytest.raises(UnsupportedOperation):
        e.undo()


def test_context_manager_closes():
    with FakeEasel() as e:
        assert not e.closed
    assert e.closed


def test_color_alias_is_rgb_triple():
    c: Color = (10, 20, 30)
    assert BrushSpec(color=c).color == c
