"""Tests for core/verifier.py — the pure per-stroke judge. No browser, no capture, no
perception: Observations are hand-built from crafted error grids, since the verifier
reads only ``region_error`` (grid), ``global_error`` (scalar), and ``intent.cell``. The
image fields of an Observation are never touched, so they are left as ``None`` here."""

import numpy as np
import pytest

from core.perception import Observation
from core.planner import PaintIntent
from core.verifier import DEFAULT_IMPROVEMENT_THRESHOLD, Verdict, Verifier


def make_obs(region_grid, global_err) -> Observation:
    """Build an Observation from a hand-crafted region-error grid and a global scalar.
    frame/target/heatmap are irrelevant to the verifier and left as None."""
    return Observation(
        frame=None,
        target=None,
        global_error=float(global_err),
        region_error=np.asarray(region_grid, dtype=np.float64),
        heatmap=None,
    )


def intent_at(cell, error=0.9) -> PaintIntent:
    """A PaintIntent aimed at ``cell``. box/color/size don't affect the verdict; only
    ``cell`` is read by the verifier."""
    return PaintIntent(cell=cell, box=(0, 0, 10, 10), color=(1, 2, 3), error=error)


# --- accept / reject on the targeted region --------------------------------------
def test_stroke_that_reduces_region_error_is_accepted():
    before = make_obs([[0.80, 0.10], [0.10, 0.10]], global_err=0.30)
    after = make_obs([[0.20, 0.10], [0.10, 0.10]], global_err=0.15)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert v.accepted is True
    assert v.region_before == 0.80
    assert v.region_after == 0.20
    assert v.region_delta == pytest.approx(0.60)


def test_no_change_is_rejected():
    grid = [[0.50, 0.10], [0.10, 0.10]]
    before = make_obs(grid, global_err=0.20)
    after = make_obs(grid, global_err=0.20)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert v.accepted is False
    assert v.region_delta == 0.0


def test_region_error_going_up_is_rejected():
    before = make_obs([[0.30, 0.10], [0.10, 0.10]], global_err=0.15)
    after = make_obs([[0.70, 0.10], [0.10, 0.10]], global_err=0.25)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert v.accepted is False
    assert v.region_delta == pytest.approx(-0.40)


# --- threshold boundary flips the verdict ----------------------------------------
def test_threshold_boundary_flips_verdict():
    # Values chosen binary-exact so "exactly at threshold" is actually representable:
    # 0.75 - 0.50 == 0.25 exactly, and 0.25 is the threshold.
    thr = 0.25
    verifier = Verifier(improvement_threshold=thr)

    # exactly at the threshold: accepted (accept rule is delta >= threshold)
    before = make_obs([[0.75]], global_err=0.5)
    at = make_obs([[0.50]], global_err=0.5)
    assert verifier.verify(before, at, intent_at((0, 0))).accepted is True

    # just under the threshold: rejected
    just_under = make_obs([[0.50 + 1e-6]], global_err=0.5)
    assert verifier.verify(before, just_under, intent_at((0, 0))).accepted is False


def test_zero_threshold_accepts_any_improvement():
    # threshold == 0.0 is valid (>= 0 guard): any non-negative delta accepts, including
    # exactly no change.
    verifier = Verifier(improvement_threshold=0.0)
    before = make_obs([[0.50]], global_err=0.5)
    tiny_drop = make_obs([[0.50 - 1e-9]], global_err=0.5)
    assert verifier.verify(before, tiny_drop, intent_at((0, 0))).accepted is True
    no_change = make_obs([[0.50]], global_err=0.5)
    assert verifier.verify(before, no_change, intent_at((0, 0))).accepted is True


def test_default_threshold_is_the_module_default():
    v = Verifier()
    assert v.improvement_threshold == DEFAULT_IMPROVEMENT_THRESHOLD
    # a drop smaller than the default reads as noise -> rejected
    before = make_obs([[0.50]], global_err=0.5)
    after = make_obs([[0.50 - DEFAULT_IMPROVEMENT_THRESHOLD / 2]], global_err=0.5)
    assert Verifier().verify(before, after, intent_at((0, 0))).accepted is False


# --- targeted-region judgment is position-correct --------------------------------
def test_judges_targeted_cell_not_a_neighbor():
    # The targeted cell (0,0) barely moves (0.002 drop, below the 0.005 default); a
    # neighbor (0,1) improves a lot. Judging the neighbor would wrongly accept — the
    # verdict must follow intent.cell, so this is rejected.
    before = make_obs([[0.400, 0.90], [0.10, 0.10]], global_err=0.4)
    after = make_obs([[0.398, 0.10], [0.10, 0.10]], global_err=0.2)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert v.cell == (0, 0)
    assert v.region_before == 0.400 and v.region_after == pytest.approx(0.398)
    assert v.accepted is False  # 0.002 drop on the targeted cell is below threshold


def test_row_col_not_transposed():
    # Asymmetric grid: cell (0,1) improves, cell (1,0) does not. A transposed read
    # (j,i) would judge (1,0) and reject; the correct (i,j) read judges (0,1).
    before = make_obs([[0.10, 0.80], [0.80, 0.10]], global_err=0.45)
    after = make_obs([[0.10, 0.20], [0.80, 0.10]], global_err=0.30)
    v = Verifier().verify(before, after, intent_at((0, 1)))
    assert v.cell == (0, 1)
    assert v.region_before == 0.80 and v.region_after == 0.20
    assert v.accepted is True


# --- global error carried through (but does not gate) ----------------------------
def test_global_error_carried_through():
    before = make_obs([[0.80]], global_err=0.62)
    after = make_obs([[0.20]], global_err=0.41)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert v.global_before == 0.62
    assert v.global_after == 0.41
    assert v.global_delta == pytest.approx(0.21)


def test_global_regression_does_not_reject_a_good_region_stroke():
    # Targeted region improves past threshold, but global error rose (expected bleed
    # into neighbors). Region-only rule: still accepted.
    before = make_obs([[0.80, 0.10], [0.10, 0.10]], global_err=0.20)
    after = make_obs([[0.20, 0.30], [0.30, 0.10]], global_err=0.35)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert v.accepted is True
    assert v.global_delta < 0  # global got worse, but the verdict is region-only


# --- validation ------------------------------------------------------------------
def test_mismatched_grid_shapes_raise():
    before = make_obs([[0.5, 0.1], [0.1, 0.1]], global_err=0.2)
    after = make_obs([[0.5, 0.1, 0.1]], global_err=0.2)
    with pytest.raises(ValueError):
        Verifier().verify(before, after, intent_at((0, 0)))


def test_cell_out_of_range_raises():
    before = make_obs([[0.5, 0.1], [0.1, 0.1]], global_err=0.2)
    after = make_obs([[0.2, 0.1], [0.1, 0.1]], global_err=0.15)
    with pytest.raises(IndexError):
        Verifier().verify(before, after, intent_at((2, 0)))
    with pytest.raises(IndexError):
        Verifier().verify(before, after, intent_at((0, 5)))
    # negative cell must NOT silently wrap (Python negative indexing) into a valid cell
    with pytest.raises(IndexError):
        Verifier().verify(before, after, intent_at((-1, 0)))


def test_negative_threshold_rejected():
    with pytest.raises(ValueError):
        Verifier(improvement_threshold=-0.01)


def test_verdict_is_plain_data():
    before = make_obs([[0.8]], global_err=0.5)
    after = make_obs([[0.2]], global_err=0.3)
    v = Verifier().verify(before, after, intent_at((0, 0)))
    assert isinstance(v, Verdict)
    assert v.cell == (0, 0)
