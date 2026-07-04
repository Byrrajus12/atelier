"""Tests for core/perception.py, exercised on hand-built canvas/target pairs with
known error. Small images keep CIEDE2000 fast; the invariants are what matter:
identical -> ~0, a known mismatch -> error localized where we put it."""

import numpy as np
import pytest

from core import perception as P


def solid(h, w, rgb):
    return np.full((h, w, 3), rgb, dtype=np.uint8)


# --- primitives: identical pairs are ~0 ------------------------------------------
def test_identical_images_have_zero_error():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert np.allclose(P.color_error(img, img), 0.0, atol=1e-6)
    assert np.allclose(P.edge_error(img, img), 0.0, atol=1e-6)
    assert np.allclose(P.pixel_error(img, img), 0.0, atol=1e-6)


# --- color term: a recolored patch lights up locally, not elsewhere --------------
def test_color_error_is_localized_to_the_recolored_patch():
    target = solid(80, 80, (200, 0, 0))      # all red
    canvas = target.copy()
    canvas[30:50, 30:50] = (0, 0, 200)       # a blue patch in the middle

    err = P.color_error(canvas, target)
    patch = err[30:50, 30:50]
    outside = err.copy()
    outside[30:50, 30:50] = 0.0
    assert patch.mean() > 20.0               # big perceptual gap in the patch
    assert outside.max() < 1e-6              # untouched region stays ~0


# --- edge/structure term ---------------------------------------------------------
def test_edge_error_responds_to_structure_not_uniform_recolor():
    gray = solid(80, 80, (128, 128, 128))

    # (a) A uniform recolor: different color everywhere, but NO new edges.
    uniform = solid(80, 80, (90, 90, 90))
    assert P.edge_error(gray, uniform).max() < 1e-6

    # (b) Same base, but with a darker square added -> real edges at its border.
    withbox = gray.copy()
    withbox[30:50, 30:50] = (60, 60, 60)
    edge = P.edge_error(gray, withbox)
    assert edge.max() > 0.05                 # border of the square lights up
    # the response hugs the square's boundary, not its flat interior/exterior
    assert edge[0:20, 0:20].max() < 1e-6     # far corner: no edge


# --- blend weighting means what it says ------------------------------------------
def test_color_weight_shifts_blend_between_color_and_edge():
    # A pure uniform recolor: strong COLOR error, ~zero EDGE error everywhere.
    target = solid(64, 64, (200, 0, 0))
    canvas = solid(64, 64, (0, 0, 200))

    pure_color = P.pixel_error(canvas, target, color_weight=1.0).mean()
    pure_edge = P.pixel_error(canvas, target, color_weight=0.0).mean()
    default = P.pixel_error(canvas, target).mean()  # 0.75

    assert pure_edge < 1e-6                  # no structural difference to see
    assert pure_color > 0.3                  # large normalized color error
    assert pure_edge < default < pure_color  # weight interpolates as documented


def test_pixel_error_is_bounded_and_validates_weight():
    target = solid(32, 32, (10, 200, 30))
    canvas = solid(32, 32, (240, 10, 220))
    err = P.pixel_error(canvas, target)
    assert err.min() >= 0.0 and err.max() <= 1.0
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError):
            P.pixel_error(canvas, target, color_weight=bad)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        P.color_error(solid(10, 10, (0, 0, 0)), solid(10, 12, (0, 0, 0)))


# --- aggregation: region grid ----------------------------------------------------
def test_region_grid_partitions_full_canvas_without_dropping_pixels():
    """600 is not divisible by 16. The grid must tile the FULL canvas: reconstructing
    the total error from cell means x cell areas must equal the raw pixel sum exactly
    (proves every pixel is counted once — no dropped edge pixels, no double counting)."""
    rng = np.random.default_rng(1)
    perr = rng.random((600, 600))
    n = 16
    grid = P.region_grid(perr, n)
    assert grid.shape == (n, n)
    rows = np.linspace(0, 600, n + 1).astype(int)
    cols = np.linspace(0, 600, n + 1).astype(int)
    total = 0.0
    for i in range(n):
        for j in range(n):
            area = (rows[i + 1] - rows[i]) * (cols[j + 1] - cols[j])
            total += grid[i, j] * area
    assert total == pytest.approx(perr.sum(), rel=1e-9)


def test_region_grid_indexing_is_row_i_col_j():
    """Pin the convention: region_error[i][j] = row i (canvas y), col j (canvas x).
    A hot patch in the TOP-RIGHT (small y, large x) must light EXACTLY cell (0, 3)."""
    perr = np.zeros((80, 80))
    perr[5:15, 60:70] = 1.0  # y in [5,15) -> row 0 ; x in [60,70) -> col 3  (n=4, step 20)
    grid = P.region_grid(perr, 4)
    hot = np.argwhere(grid > 1e-9).tolist()
    assert hot == [[0, 3]]  # specific cell, position-verified — not merely "non-zero"


def test_region_grid_rejects_bad_n():
    perr = np.zeros((16, 16))
    with pytest.raises(ValueError):
        P.region_grid(perr, 0)
    with pytest.raises(ValueError):
        P.region_grid(perr, 17)  # larger than the map


def test_region_grid_indexing_holds_on_non_square_canvas():
    """Anti-transposition guard: on a NON-square map (H != W) a swapped-h/w bug would
    light the wrong cell — invisible on square fixtures. A hot patch at (canvas y in
    the top band, canvas x in the right band) must light exactly its row/col cell.

    perr is H=40 rows x W=80 cols, n=4 -> row step 10 (y), col step 20 (x).
    Patch at y in [2,8) -> row 0 ; x in [62,70) -> col 3."""
    perr = np.zeros((40, 80))
    perr[2:8, 62:70] = 1.0
    grid = P.region_grid(perr, 4)
    assert grid.shape == (4, 4)
    assert np.argwhere(grid > 1e-9).tolist() == [[0, 3]]


# --- observe(): the bundle -------------------------------------------------------
def _frame_target(canvas, target):
    from core.adapter import Frame
    from core.target import Target

    return Frame(canvas), Target(target)


def test_observe_identical_is_zero_and_well_shaped():
    frame, target = _frame_target(solid(64, 64, (30, 60, 90)), solid(64, 64, (30, 60, 90)))
    obs = P.observe(frame, target, n=8)
    assert obs.global_error == pytest.approx(0.0, abs=1e-6)
    assert obs.region_error.shape == (8, 8)
    assert np.allclose(obs.region_error, 0.0, atol=1e-6)
    assert obs.heatmap.shape == (64, 64, 3) and obs.heatmap.dtype == np.uint8
    assert obs.frame is frame and obs.target is target


def test_observe_lights_the_correct_region_cell():
    """A known mismatch patch in the BOTTOM-LEFT must make cell (3, 0) the hottest —
    position-verified, so M4 can't act on a transposed grid."""
    target = solid(80, 80, (200, 0, 0))
    canvas = target.copy()
    canvas[60:80, 0:20] = (0, 0, 200)  # y in [60,80) -> row 3 ; x in [0,20) -> col 0
    frame, tgt = _frame_target(canvas, target)

    obs = P.observe(frame, tgt, n=4)
    grid = obs.region_error
    assert np.unravel_index(np.argmax(grid), grid.shape) == (3, 0)
    assert grid[3, 0] > 0.2
    assert obs.global_error > 0.0


def test_observe_size_mismatch_raises():
    frame, target = _frame_target(solid(64, 64, (0, 0, 0)), solid(64, 80, (0, 0, 0)))
    with pytest.raises(ValueError):
        P.observe(frame, target, n=8)


def test_observe_lights_correct_cell_on_non_square_canvas():
    """End-to-end axis guarantee through observe() on a NON-square canvas (H=40, W=80).
    A recolored patch in the TOP-RIGHT (small y, large x) must make cell (0, 3) the
    hottest — a swapped-h/w bug would light a different cell here but not on a square."""
    target = solid(40, 80, (200, 0, 0))       # H=40, W=80
    canvas = target.copy()
    canvas[2:8, 62:70] = (0, 0, 200)          # y -> row 0 ; x -> col 3  (n=4)
    frame, tgt = _frame_target(canvas, target)

    obs = P.observe(frame, tgt, n=4, color_weight=1.0)  # pure color: no edge bleed
    grid = obs.region_error
    assert np.unravel_index(np.argmax(grid), grid.shape) == (0, 3)
    others = grid.copy()
    others[0, 3] = 0.0
    assert others.max() < 1e-6                # only the patch's cell lit


def test_public_error_fns_reject_non_uint8():
    """M4/the verifier may call these directly; a non-uint8 array would be silently
    mis-scaled by the /255 in _as_float_rgb, so it must be rejected."""
    u = solid(16, 16, (10, 20, 30))
    f = (u.astype(np.float32) / 255.0)        # already-normalized float — the trap
    for fn in (P.color_error, P.edge_error, P.pixel_error):
        with pytest.raises(ValueError):
            fn(f, u)
        with pytest.raises(ValueError):
            fn(u, f)
