"""Tests for the Target contract (core/target.py): the value-type invariants and the
trivial file loader (decode + BGR->RGB + resize-to-canvas)."""

import numpy as np
import pytest

from core.target import Target, load_target


# --- Target invariants -----------------------------------------------------------
def test_target_valid_and_size_is_width_height():
    t = Target(np.zeros((30, 40, 3), dtype=np.uint8))  # H=30, W=40
    assert t.size == (40, 30)  # (width, height)


def test_target_rejects_non_rgb():
    with pytest.raises(ValueError):
        Target(np.zeros((30, 40), dtype=np.uint8))       # no channel axis
    with pytest.raises(ValueError):
        Target(np.zeros((30, 40, 4), dtype=np.uint8))    # RGBA


def test_target_rejects_non_uint8():
    with pytest.raises(ValueError):
        Target(np.zeros((30, 40, 3), dtype=np.float32))


# --- load_target -----------------------------------------------------------------
def test_load_target_resizes_to_canvas_and_is_uint8_rgb(tmp_path):
    import cv2

    src = tmp_path / "src.png"
    # cv2 writes arrays as BGR: pure-red-in-RGB is BGR (0, 0, 255).
    cv2.imwrite(str(src), np.full((10, 10, 3), (0, 0, 255), dtype=np.uint8))

    t = load_target(str(src), size=(600, 600))
    assert t.image.shape == (600, 600, 3)
    assert t.image.dtype == np.uint8
    assert t.size == (600, 600)
    # BGR->RGB conversion happened: the image should read as red in RGB.
    assert t.image[..., 0].min() > 250  # R high
    assert t.image[..., 1].max() < 5    # G low
    assert t.image[..., 2].max() < 5    # B low


def test_load_target_size_is_width_height(tmp_path):
    import cv2

    src = tmp_path / "s.png"
    cv2.imwrite(str(src), np.zeros((10, 10, 3), dtype=np.uint8))
    t = load_target(str(src), size=(40, 30))  # (width, height)
    assert t.image.shape == (30, 40, 3)  # (H, W, 3)
    assert t.size == (40, 30)


def test_load_target_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_target(str(tmp_path / "does-not-exist.png"))
