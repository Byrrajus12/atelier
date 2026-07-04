"""Tests for core/planner.py — the greedy region-level planner. Exercised against
synthetic Observations (no browser, no display): both genuine ones from
perception.observe() and hand-fabricated ones (Observation is mutable) to control the
error grid precisely and decouple the policy from the metric."""

import numpy as np
import pytest

from core.adapter import Frame
from core.perception import Observation, cell_box, observe
from core.planner import (
    GreedyPlanner,
    PaintIntent,
    Planner,
    region_mean_color,
)
from core.target import Target


def solid(h, w, rgb):
    return np.full((h, w, 3), rgb, dtype=np.uint8)


def make_observation(canvas, target, region_error, n=None):
    """A fabricated Observation with a chosen region_error (frame/target real for
    color reads). heatmap/global_error are irrelevant to the planner, so kept trivial."""
    return Observation(
        frame=Frame(canvas),
        target=Target(target),
        global_error=float(region_error.mean()),
        region_error=region_error,
        heatmap=np.zeros_like(canvas),
    )


# --- PaintIntent / region_mean_color ---------------------------------------------
def test_paint_intent_defaults_size_to_12():
    intent = PaintIntent(cell=(1, 2), box=(0, 0, 10, 10), color=(1, 2, 3), error=0.5)
    assert intent.size == 12.0


def test_region_mean_color_reads_the_box():
    img = solid(40, 40, (10, 20, 30))
    img[0:20, 20:40] = (200, 100, 50)  # top-right quadrant a different color
    # box = (x0, y0, x1, y1): the top-right quadrant
    assert region_mean_color(img, (20, 0, 40, 20)) == (200, 100, 50)
    assert region_mean_color(img, (0, 20, 20, 40)) == (10, 20, 30)


def test_region_mean_color_rounds_and_rejects_empty():
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    img[0, 0] = (0, 0, 0)
    img[0, 1] = (0, 0, 3)
    assert region_mean_color(img, (0, 0, 2, 1)) == (0, 0, 2)  # mean of B 0 and 3 -> 1.5 -> 2
    with pytest.raises(ValueError):
        region_mean_color(img, (2, 2, 2, 2))  # empty box


# --- Planner interface (Principle 6) ---------------------------------------------
def test_planner_is_abstract():
    with pytest.raises(TypeError):
        Planner()  # type: ignore[abstract]


def test_greedy_planner_is_a_planner():
    assert isinstance(GreedyPlanner(), Planner)


def test_greedy_rejects_negative_threshold():
    with pytest.raises(ValueError):
        GreedyPlanner(error_threshold=-0.01)


# --- policy: highest-error region ------------------------------------------------
def test_greedy_picks_highest_error_cell():
    grid = np.zeros((4, 4))
    grid[2, 3] = 0.9  # the single hottest region
    obs = make_observation(solid(40, 80, (0, 0, 0)), solid(40, 80, (0, 0, 0)), grid)
    intent = GreedyPlanner().plan(obs)
    assert intent is not None
    assert intent.cell == (2, 3)
    assert intent.error == pytest.approx(0.9)


def test_greedy_tie_break_is_first_in_row_major_order():
    """On an exact error tie, the FIRST cell in row-major (i, then j) order wins.
    The orchestrator relies on this determinism, so pin it against a silent change."""
    grid = np.zeros((3, 3))
    grid[0, 2] = 0.7  # row-major flat index 2
    grid[2, 1] = 0.7  # row-major flat index 7 — same error, but later
    obs = make_observation(solid(30, 30, (0, 0, 0)), solid(30, 30, (0, 0, 0)), grid)
    intent = GreedyPlanner().plan(obs)
    assert intent.cell == (0, 2)  # the earlier cell, not (2, 1)


def test_greedy_box_matches_cell_box_for_the_picked_cell():
    grid = np.zeros((4, 4))
    grid[1, 2] = 0.5
    canvas, target = solid(40, 80, (0, 0, 0)), solid(40, 80, (0, 0, 0))
    obs = make_observation(canvas, target, grid)
    intent = GreedyPlanner().plan(obs)
    assert intent.box == cell_box(1, 2, 4, (80, 40))  # (width, height)


def test_greedy_reads_target_color_from_the_picked_region():
    """Desired color must be the TARGET's mean color over the picked cell's box —
    position-verified so a transposed/misboxed read fails."""
    n = 4
    target = solid(40, 80, (10, 10, 10))
    target[0:10, 60:80] = (200, 40, 60)  # top-right cell (row 0, col 3): distinct color
    canvas = solid(40, 80, (10, 10, 10))
    grid = np.zeros((n, n))
    grid[0, 3] = 0.8  # force the planner to pick that exact cell
    obs = make_observation(canvas, target, grid)

    intent = GreedyPlanner().plan(obs)
    assert intent.cell == (0, 3)
    assert intent.color == (200, 40, 60)  # read from the target at the right box


def test_greedy_end_to_end_on_real_observation():
    """On a genuine Observation from observe(): a blue patch on a red canvas/target
    picks that region and the target color there (red)."""
    target = solid(80, 80, (200, 0, 0))
    canvas = target.copy()
    canvas[60:80, 0:20] = (0, 0, 200)  # bottom-left region diverges (row 3, col 0)
    obs = observe(Frame(canvas), Target(target), n=4)
    intent = GreedyPlanner().plan(obs)
    assert intent is not None
    assert intent.cell == (3, 0)
    # target there is red; realized color request should be ~red
    assert intent.color[0] > 150 and intent.color[1] < 40 and intent.color[2] < 40


# --- convergence -----------------------------------------------------------------
def test_greedy_returns_none_when_converged():
    grid = np.full((4, 4), 0.01)  # everything below the default 0.02 threshold
    obs = make_observation(solid(40, 40, (0, 0, 0)), solid(40, 40, (0, 0, 0)), grid)
    assert GreedyPlanner().plan(obs) is None


def test_threshold_is_a_boundary_the_planner_respects():
    grid = np.zeros((2, 2))
    grid[0, 0] = 0.05
    canvas = target = solid(20, 20, (0, 0, 0))
    obs = make_observation(canvas, target, grid)
    assert GreedyPlanner(error_threshold=0.04).plan(obs) is not None  # above -> act
    assert GreedyPlanner(error_threshold=0.05).plan(obs) is None      # at -> converged
    assert GreedyPlanner(error_threshold=0.06).plan(obs) is None      # below -> converged


def test_identical_canvas_and_target_converges_immediately():
    img = solid(64, 64, (30, 60, 90))
    obs = observe(Frame(img.copy()), Target(img.copy()), n=8)
    assert GreedyPlanner().plan(obs) is None
