import numpy as np

from core.adapter import Frame
from core.perception import Observation, cell_box
from core.planner import DEFAULT_ERROR_THRESHOLD
from core.target import Target
from core.validator import validate_intent

PALETTE = ((255, 0, 0), (0, 0, 255), (17, 17, 17))


def solid(h, w, rgb):
    return np.full((h, w, 3), rgb, dtype=np.uint8)


def make_observation(*, n=4, cell=(1, 2), target_color=(255, 0, 0), canvas_color=(255, 255, 255), error=0.8):
    h = w = 40
    canvas = solid(h, w, (255, 255, 255))
    target = solid(h, w, (255, 255, 255))
    x0, y0, x1, y1 = cell_box(cell[0], cell[1], n, (w, h))
    canvas[y0:y1, x0:x1] = canvas_color
    target[y0:y1, x0:x1] = target_color
    grid = np.zeros((n, n), dtype=float)
    grid[cell] = error
    return Observation(
        frame=Frame(canvas),
        target=Target(target),
        global_error=float(grid.mean()),
        region_error=grid,
        heatmap=np.zeros_like(canvas),
    )


def test_converged_cell_is_blocked_first():
    obs = make_observation(error=DEFAULT_ERROR_THRESHOLD)

    ok, reason = validate_intent(obs, obs.target, (1, 2), (255, 0, 0), 4, PALETTE)

    assert ok is False
    assert "already converged" in reason
    assert "error=" in reason


def test_swatch_that_would_not_improve_is_blocked():
    obs = make_observation(target_color=(255, 255, 255), canvas_color=(255, 255, 255), error=0.8)

    ok, reason = validate_intent(obs, obs.target, (1, 2), (255, 0, 0), 4, PALETTE)

    assert ok is False
    assert "would not reduce error" in reason
    assert "available palette" in reason


def test_wrong_color_is_blocked_after_swatch_would_improve():
    obs = make_observation(target_color=(17, 17, 17), canvas_color=(255, 255, 255), error=0.8)

    ok, reason = validate_intent(obs, obs.target, (1, 2), (255, 0, 0), 4, PALETTE)

    assert ok is False
    assert "doesn't match target color" in reason
    assert "(17, 17, 17)" in reason


def test_valid_move_passes():
    obs = make_observation(target_color=(255, 0, 0), canvas_color=(255, 255, 255), error=0.8)

    ok, reason = validate_intent(obs, obs.target, (1, 2), (255, 0, 0), 4, PALETTE)

    assert ok is True
    assert reason == ""