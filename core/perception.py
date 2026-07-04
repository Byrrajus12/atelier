"""Perception — the "perceive -> diff" half of the closed loop (CLAUDE.md Principle 3).

Given the current canvas (a captured ``Frame``) and the ``Target``, perception measures
*how wrong* the canvas is and *where*, and bundles that into an ``Observation`` for the
planner (M4) to act on. Everything here is pure ``ndarray`` math: no screen capture, no
synthetic input, no environment specifics (Principle 2), and no rendering or UI — the
error heatmap is produced as plain image data the dashboard later consumes (Principle 5).

Error metric: a per-pixel blend of
  * perceptual **color** distance — CIEDE2000 in CIELAB (``skimage``), and
  * a **structural** term — the difference of Sobel edge magnitudes,
with the color/edge weighting a parameter (default color-dominant). Each term is
normalized to roughly ``[0, 1]`` before blending so the weight means what it says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.adapter import Frame
from core.target import Target

# Normalization / default knobs (all overridable as parameters; nothing downstream
# should hardcode these).
DELTA_E_REF = 100.0          # CIEDE2000 scale used to normalize color error to ~[0,1]
DEFAULT_COLOR_WEIGHT = 0.75  # color-dominant blend by default
DEFAULT_GRID_N = 16          # default per-region grid resolution (N x N)


def _as_float_rgb(image: np.ndarray) -> np.ndarray:
    """uint8 RGB -> float64 RGB in [0, 1], as skimage color/edge ops expect."""
    return image.astype(np.float64) / 255.0


def _check_pair(canvas: np.ndarray, target: np.ndarray) -> None:
    if canvas.shape != target.shape:
        raise ValueError(
            f"canvas and target must match: {canvas.shape} != {target.shape}"
        )
    if canvas.ndim != 3 or canvas.shape[2] != 3:
        raise ValueError("canvas and target must be HxWx3 RGB")
    # dtype guard: these functions are part of the public API (M4/the verifier may
    # call them directly, not only via observe() with Frame/Target-sourced arrays).
    # _as_float_rgb divides by 255 unconditionally, so a non-uint8 (e.g. already
    # [0,1] float) array would be silently mis-scaled — reject it rather than compute
    # a wrong error.
    if canvas.dtype != np.uint8 or target.dtype != np.uint8:
        raise ValueError(
            f"canvas and target must be uint8 (0..255), got "
            f"{canvas.dtype} and {target.dtype}"
        )


def color_error(canvas: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pixel perceptual color distance (CIEDE2000 in CIELAB). Returns an ``HxW``
    float array, ~``0..100``: 0 where the colors match, larger the more they differ
    to a human eye."""
    from skimage.color import deltaE_ciede2000, rgb2lab

    _check_pair(canvas, target)
    lab_c = rgb2lab(_as_float_rgb(canvas))
    lab_t = rgb2lab(_as_float_rgb(target))
    return np.asarray(deltaE_ciede2000(lab_c, lab_t), dtype=np.float64)


def edge_error(canvas: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pixel structural difference: ``|sobel(gray(canvas)) - sobel(gray(target))|``.
    Returns an ``HxW`` float array, ~``0..1``: responds where one image has an edge the
    other lacks (or a differently-placed one), even if the average colors are close."""
    from skimage.color import rgb2gray
    from skimage.filters import sobel

    _check_pair(canvas, target)
    edge_c = sobel(rgb2gray(_as_float_rgb(canvas)))
    edge_t = sobel(rgb2gray(_as_float_rgb(target)))
    return np.abs(edge_c - edge_t)


def pixel_error(
    canvas: np.ndarray,
    target: np.ndarray,
    color_weight: float = DEFAULT_COLOR_WEIGHT,
) -> np.ndarray:
    """Blended per-pixel error in ``[0, 1]``: ``color_weight`` on the normalized color
    term, ``1 - color_weight`` on the normalized edge term. ``color_weight=1`` is pure
    color, ``0`` is pure structure. Default is color-dominant."""
    if not 0.0 <= color_weight <= 1.0:
        raise ValueError("color_weight must be in [0, 1]")
    color = np.clip(color_error(canvas, target) / DELTA_E_REF, 0.0, 1.0)
    edge = np.clip(edge_error(canvas, target), 0.0, 1.0)
    return color_weight * color + (1.0 - color_weight) * edge


# --- aggregation -----------------------------------------------------------------
def global_error(pixel_err: np.ndarray) -> float:
    """One scalar: the mean per-pixel error over the whole canvas, in ``[0, 1]``."""
    return float(pixel_err.mean())


def region_grid(pixel_err: np.ndarray, n: int = DEFAULT_GRID_N) -> np.ndarray:
    """Mean-pool an ``HxW`` per-pixel error map into an ``n x n`` grid of region means.

    Indexing convention (pinned so the planner can't paint in a transposed place):
    ``region_error[i][j]`` is the mean error of the canvas region at **row i, column
    j** — i.e. ``i`` indexes the vertical axis (canvas **y**, top -> bottom) and ``j``
    indexes the horizontal axis (canvas **x**, left -> right). So ``[0][0]`` is the
    top-left region and ``[n-1][n-1]`` is the bottom-right.

    The grid tiles the **entire** canvas: cell edges are ``np.linspace(0, H, n+1)`` /
    ``linspace(0, W, n+1)`` boundaries, so when H/W is not divisible by ``n`` the
    remainder is absorbed by making some cells one pixel larger — no edge pixels are
    dropped and none are counted twice.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    h, w = pixel_err.shape[:2]
    if n > min(h, w):
        raise ValueError(f"n={n} too large for a {h}x{w} error map")
    rows = np.linspace(0, h, n + 1).astype(int)
    cols = np.linspace(0, w, n + 1).astype(int)
    grid = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            block = pixel_err[rows[i] : rows[i + 1], cols[j] : cols[j + 1]]
            grid[i, j] = float(block.mean())
    return grid


def heatmap(pixel_err: np.ndarray) -> np.ndarray:
    """Render the per-pixel error map as an ``HxWx3`` uint8 RGB image for the dashboard.
    Absolute scale: error ``0..1`` maps directly to the colormap (no per-frame contrast
    stretch), so a near-perfect canvas reads as uniformly dark rather than amplifying
    noise. This is plain image *data* — perception does no display (Principle 5)."""
    import cv2

    gray = (np.clip(pixel_err, 0.0, 1.0) * 255).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return np.ascontiguousarray(bgr[:, :, ::-1])  # BGR -> RGB


# --- the bundle handed to the planner --------------------------------------------
@dataclass
class Observation:
    """What perception hands the planner (M4): the canvas ``frame`` and the ``target``
    it was compared against, plus the computed error at three granularities —
    ``global_error`` (scalar), ``region_error`` (``n x n``; see ``region_grid`` for the
    indexing convention), and ``heatmap`` (an ``HxWx3`` uint8 image for the dashboard)."""

    frame: Frame
    target: Target
    global_error: float
    region_error: np.ndarray
    heatmap: np.ndarray


def observe(
    frame: Frame,
    target: Target,
    n: int = DEFAULT_GRID_N,
    color_weight: float = DEFAULT_COLOR_WEIGHT,
) -> Observation:
    """Diff ``frame`` against ``target`` and bundle the result as an ``Observation``.

    This is the perception boundary where a captured frame meets a target, so it is
    where their sizes must agree — a ``Frame`` alone does not know the canvas size, so
    the check lives here rather than in the ``Frame`` contract."""
    if frame.size != target.size:
        raise ValueError(
            f"frame size {frame.size} != target size {target.size}"
        )
    perr = pixel_error(frame.image, target.image, color_weight=color_weight)
    return Observation(
        frame=frame,
        target=target,
        global_error=global_error(perr),
        region_error=region_grid(perr, n),
        heatmap=heatmap(perr),
    )
