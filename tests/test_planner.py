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
    nearest_swatch,
    region_mean_color,
    swatch_would_improve,
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
    grid = np.full((4, 4), 0.01)  # everything below the default 0.08 threshold
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


# --- no-undo palette guard (M7.6) ------------------------------------------------
# A palette without white, matching the reference easel's red/blue/near-black.
PALETTE = ((255, 0, 0), (0, 0, 255), (17, 17, 17))


def test_guard_skips_unpaintable_cell_and_picks_a_lower_improvable_one():
    """The hottest region's target is the white background — no swatch improves it, so a
    guarded planner must skip it and paint the lower-error region a swatch CAN improve,
    rather than blindly self-damaging the white gap (the iter-13 live-run failure)."""
    n = 2
    canvas = solid(40, 40, (255, 255, 255))          # blank white canvas
    target = solid(40, 40, (255, 255, 255))
    target[0:20, 20:40] = (0, 0, 255)                # cell (0,1): blue, paintable
    grid = np.zeros((n, n))
    grid[0, 0] = 0.9                                 # white gap: hottest but unpaintable
    grid[0, 1] = 0.5                                 # blue: cooler but improvable
    obs = make_observation(canvas, target, grid)

    intent = GreedyPlanner(palette=PALETTE).plan(obs)
    assert intent is not None
    assert intent.cell == (0, 1)                     # skipped the white gap, took the blue


def test_guard_off_by_default_still_picks_the_white_gap():
    """palette=None keeps the plain argmax baseline: the same white-gap region is picked
    (the pre-M7.6 behavior), proving the guard is strictly opt-in."""
    n = 2
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))
    target[0:20, 20:40] = (0, 0, 255)
    grid = np.zeros((n, n))
    grid[0, 0] = 0.9
    grid[0, 1] = 0.5
    obs = make_observation(canvas, target, grid)

    intent = GreedyPlanner().plan(obs)               # no palette -> blind
    assert intent.cell == (0, 0)                     # picks the unpaintable hottest cell


def test_guard_returns_none_when_no_above_threshold_cell_is_improvable():
    """If every region above threshold is color-unpaintable, the guarded planner reports
    convergence (None) rather than making the canvas worse irreversibly."""
    n = 2
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))          # all-white target, all-white canvas
    grid = np.zeros((n, n))
    grid[0, 0] = 0.9                                 # a hot region, but nothing helps it
    obs = make_observation(canvas, target, grid)

    assert GreedyPlanner(palette=PALETTE).plan(obs) is None


def test_guard_does_not_over_skip_an_improvable_cell():
    """A region a swatch clearly improves is still selected — the guard only skips the
    genuinely-unpaintable, it does not suppress normal work."""
    n = 2
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))
    target[0:20, 0:20] = (255, 0, 0)                 # cell (0,0): red, paintable
    grid = np.zeros((n, n))
    grid[0, 0] = 0.7
    obs = make_observation(canvas, target, grid)

    intent = GreedyPlanner(palette=PALETTE).plan(obs)
    assert intent is not None
    assert intent.cell == (0, 0)
    assert intent.color == (255, 0, 0)               # requests the ideal target color


def test_guard_rejects_empty_palette():
    with pytest.raises(ValueError):
        GreedyPlanner(palette=())


def test_nearest_swatch_uses_euclidean_rgb_matching_the_easel():
    """The guard must predict the swatch the easel would truly paint (Euclidean-RGB
    nearest), or it could green-light a move that lands a different, worse color."""
    p = GreedyPlanner(palette=PALETTE)
    assert p._nearest_swatch((250, 10, 10)) == (255, 0, 0)     # -> red
    assert p._nearest_swatch((10, 10, 250)) == (0, 0, 255)     # -> blue
    assert p._nearest_swatch((30, 25, 20)) == (17, 17, 17)     # -> near-black


# --- the shared self-damage test (Phase 1) ---------------------------------------
# Promoted out of GreedyPlanner so the orchestrator's non-blocking observer can apply
# EXACTLY the same rule the greedy guard uses — one definition, two callers with
# different powers (skip vs merely record).
def test_nearest_swatch_function_matches_the_planner_method():
    for requested in ((250, 10, 10), (10, 10, 250), (30, 25, 20), (128, 128, 128)):
        assert (
            nearest_swatch(requested, PALETTE)
            == GreedyPlanner(palette=PALETTE)._nearest_swatch(requested)
        )


def test_nearest_swatch_rejects_an_empty_palette():
    with pytest.raises(ValueError):
        nearest_swatch((0, 0, 0), ())


def test_swatch_would_improve_is_false_for_an_unpaintable_white_cell():
    """The canonical no-undo trap: a white-background cell that no swatch can improve."""
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))
    obs = make_observation(canvas, target, np.zeros((2, 2)))
    assert swatch_would_improve(obs, (0, 0, 20, 20), PALETTE) is False


def test_swatch_would_improve_is_true_for_a_paintable_cell():
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))
    target[0:20, 0:20] = (255, 0, 0)
    obs = make_observation(canvas, target, np.zeros((2, 2)))
    assert swatch_would_improve(obs, (0, 0, 20, 20), PALETTE) is True


def test_swatch_would_improve_honors_an_explicitly_requested_color():
    """The observer passes the planner's REQUESTED color (a model may ask for anything),
    rather than assuming the target's mean — so a bad requested color must read as
    self-damaging even where the cell is otherwise paintable."""
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))
    target[0:20, 0:20] = (255, 0, 0)  # cell target is red
    obs = make_observation(canvas, target, np.zeros((2, 2)))
    box = (0, 0, 20, 20)
    assert swatch_would_improve(obs, box, PALETTE, (255, 0, 0)) is True   # right color
    assert swatch_would_improve(obs, box, PALETTE, (0, 0, 255)) is False  # blue on a red target


def test_greedy_guard_delegates_to_the_shared_function():
    """The planner method and the shared function must agree, or the observer would be
    measuring a different rule than the one the greedy guard enforces."""
    canvas = solid(40, 40, (255, 255, 255))
    target = solid(40, 40, (255, 255, 255))
    target[0:20, 20:40] = (0, 0, 255)
    obs = make_observation(canvas, target, np.zeros((2, 2)))
    planner = GreedyPlanner(palette=PALETTE)
    for box in ((0, 0, 20, 20), (20, 0, 40, 20)):
        assert planner._swatch_improves(obs, box) == swatch_would_improve(obs, box, PALETTE)
