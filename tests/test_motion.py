"""Tests for core/motion.py — deterministic stroke-pacing densify. No browser, no
human-motion behavior: just that the path gets denser while its geometry is preserved
exactly (endpoints and vertices don't drift)."""

import math

import pytest

from core.adapter import Point
from core.motion import DEFAULT_MAX_STEP_PX, densify


def _consecutive_dists(pts):
    return [math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(pts, pts[1:])]


def test_densify_preserves_endpoints_exactly():
    path = (Point(10.0, 20.0), Point(210.0, 20.0))
    out = densify(path, max_step_px=6.0)
    assert out[0] == Point(10.0, 20.0)   # exact, not approximate
    assert out[-1] == Point(210.0, 20.0)


def test_densify_consecutive_points_within_max_step():
    path = (Point(0.0, 0.0), Point(100.0, 40.0))  # length ~107.7
    step = 6.0
    out = densify(path, max_step_px=step)
    dists = _consecutive_dists(out)
    assert dists  # more than one point
    assert max(dists) <= step + 1e-9


def test_densify_points_lie_on_the_original_segment_no_drift():
    # Straight diagonal: every produced point must be collinear with the endpoints
    # and lie between them (parameter t in [0,1]).
    a, b = Point(0.0, 0.0), Point(90.0, 30.0)
    out = densify((a, b), max_step_px=5.0)
    for p in out:
        # cross-product of (b-a) x (p-a) == 0  => collinear
        cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
        assert abs(cross) < 1e-6
        t = (p.x - a.x) / (b.x - a.x)
        assert -1e-9 <= t <= 1 + 1e-9


def test_densify_preserves_interior_vertices_exactly():
    # An L-shaped path: the corner vertex must survive verbatim in the output.
    corner = Point(50.0, 0.0)
    path = (Point(0.0, 0.0), corner, Point(50.0, 40.0))
    out = densify(path, max_step_px=7.0)
    assert out[0] == path[0]
    assert out[-1] == path[-1]
    assert corner in out  # exact vertex retained


def test_densify_single_point_unchanged():
    assert densify((Point(3.0, 4.0),)) == (Point(3.0, 4.0),)
    assert densify(()) == ()


def test_densify_smaller_step_yields_more_points():
    path = (Point(0.0, 0.0), Point(120.0, 0.0))
    coarse = densify(path, max_step_px=12.0)
    fine = densify(path, max_step_px=3.0)
    assert len(fine) > len(coarse)


def test_densify_default_step_is_used():
    path = (Point(0.0, 0.0), Point(60.0, 0.0))
    out = densify(path)  # default DEFAULT_MAX_STEP_PX
    assert max(_consecutive_dists(out)) <= DEFAULT_MAX_STEP_PX + 1e-9


def test_densify_rejects_nonpositive_step():
    path = (Point(0.0, 0.0), Point(10.0, 0.0))
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            densify(path, max_step_px=bad)


def test_densify_handles_duplicate_consecutive_points():
    path = (Point(0.0, 0.0), Point(0.0, 0.0), Point(30.0, 0.0))
    out = densify(path, max_step_px=6.0)
    assert out[0] == Point(0.0, 0.0)
    assert out[-1] == Point(30.0, 0.0)
    assert max(_consecutive_dists(out)) <= 6.0 + 1e-9
