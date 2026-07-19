"""Shared VLM request construction — the exact image encoding and request shape the
STEP 0 probe validated against the live Fireworks API.

Both ``scripts/probe_vlm.py`` and ``planners/vlm_planner.py`` build their requests from
here, so the planner sends what the probe proved works rather than a re-derived variant
that could silently drift (different resolution, missing grid labels, a tool_choice form
the model ignores).

What the probe established, and why each piece is here:

  * **Axis-labeled images.** Grid lines plus spreadsheet-style row/column headers let
    the model ground a cell index visually without drawing annotation pixels inside the
    cells whose colors it must judge. In-cell labels caused a systematic row association
    error, and colored label halos looked like paint defects to the model.
  * **Required tool call.** ``tool_choice`` requires a tool call while offering either
    ``propose_paint_cell`` or ``report_canvas_complete``. Left to its own devices the
    model narrated prose and never called the tool; forcing the specific paint function
    worked for Phase 1 but gave the model no honest way to say the canvas was done.
  * **Reasoning budget.** The model spends its reasoning *before* emitting the tool
    call, so the completion budget must clear that or the call is truncated away
    (``finish_reason=length``). How much it spends depends on the canvas: a blank one is
    an easy call, a half-painted one invites a long cell-by-cell comparison. Two levers
    keep the call inside the budget — a budget sized for the mid-run case
    (``DEFAULT_MAX_TOKENS``) and a prompt that asks for a prompt decision rather than an
    exhaustive audit (``build_prompt``).
  * **Reasoning lives in ``reasoning_content``.** This model returns chain-of-thought in a
    separate message field and leaves the schema's ``reasoning`` argument empty. The
    schema field is kept (other models do populate it) but ``reasoning_content`` is read
    as the primary source — see ``extract_reasoning``.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

DEFAULT_PROBE_SIZE = 128  # downsampled square edge length; validated legible by the probe
# Clears the model's pre-call reasoning spend with headroom. Tuned for the MID-RUN case,
# not the blank-canvas one: on a partially-painted canvas the model compares target vs
# current cell by cell and writes far longer reasoning before it emits the call. 500 was
# enough for the easy probe and truncated in-loop (finish_reason=length, no tool call).
DEFAULT_MAX_TOKENS = 1500

TOOL_NAME = "propose_paint_cell"
DONE_TOOL_NAME = "report_canvas_complete"

# Require some tool call. The model may choose between proposing paint and reporting
# completion, but it still cannot satisfy the request with prose alone.
TOOL_CHOICE_REQUIRED = "required"

# Pinned to the specific function. Kept for probes/tests that intentionally exercise the
# original single-function shape; the planner now defaults to TOOL_CHOICE_REQUIRED so the
# model can also report completion.
TOOL_CHOICE_SPECIFIC: Dict[str, Any] = {
    "type": "function",
    "function": {"name": TOOL_NAME},
}

PAINT_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Choose ONE grid cell to paint next on the canvas, and the RGB color to "
            "paint it, to make the canvas look more like the target."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cell": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "[i, j] grid indices: i = row (top->bottom), j = column "
                        "(left->right), 0-indexed."
                    ),
                },
                "color": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 3,
                    "maxItems": 3,
                    "description": "[r, g, b], each 0-255.",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this cell and color were chosen.",
                },
            },
            "required": ["cell", "color", "reasoning"],
        },
    },
}

DONE_TOOL_SCHEMA: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": DONE_TOOL_NAME,
        "description": (
            "Report that the current canvas already matches the target well enough and "
            "no paint action should be taken."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why the canvas appears complete.",
                },
            },
        },
    },
}


@dataclass(frozen=True)
class PaintDecision:
    cell: Tuple[int, int]
    color: Tuple[int, int, int]
    reasoning: str


@dataclass(frozen=True)
class DoneDecision:
    reasoning: str


ToolDecision = Union[PaintDecision, DoneDecision]


def _axis_margin(size: int) -> int:
    """Header-strip thickness for row/column labels in a final model image."""
    return min(max(12, int(round(size * 0.14))), max(1, size // 4))


def _put_centered_text(
    img: np.ndarray,
    text: str,
    center: Tuple[int, int],
    font: int,
    font_scale: float,
    color: Tuple[int, int, int],
) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, 1)
    cx, cy = center
    origin = (int(cx - text_w / 2), int(cy + text_h / 2))
    cv2.putText(img, text, origin, font, font_scale, color, 1, cv2.LINE_AA)


def draw_grid(img: np.ndarray, n: int) -> np.ndarray:
    """Render ``img`` into a grid with spreadsheet-style axis labels.

    Row numbers live in a left margin and column numbers in a top margin. Nothing is drawn
    inside any cell except faint grid boundaries, so the model sees clean target/canvas
    color when judging whether a cell is correct.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    h, w = img.shape[:2]
    top = _axis_margin(h)
    left = _axis_margin(w)
    grid_h = h - top
    grid_w = w - left
    if grid_h < 1 or grid_w < 1:
        raise ValueError("image is too small for axis-labeled grid")

    out = np.full_like(img, 255)
    content = cv2.resize(img, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
    out[top:h, left:w] = content

    row_edges = top + np.linspace(0, grid_h, n + 1).astype(int)
    col_edges = left + np.linspace(0, grid_w, n + 1).astype(int)
    for y in row_edges:
        cv2.line(out, (left, int(y)), (w, int(y)), (128, 128, 128), 1)
    for x in col_edges:
        cv2.line(out, (int(x), top), (int(x), h), (128, 128, 128), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.35, min(0.7, min(h, w) / 280.0))
    label_color = (0, 0, 0)
    for i in range(n):
        cy = int((row_edges[i] + row_edges[i + 1]) // 2)
        _put_centered_text(out, str(i), (left // 2, cy), font, font_scale, label_color)
    for j in range(n):
        cx = int((col_edges[j] + col_edges[j + 1]) // 2)
        _put_centered_text(out, str(j), (cx, top // 2), font, font_scale, label_color)
    return out


def prepare_image(img: np.ndarray, n: int, size: int = DEFAULT_PROBE_SIZE) -> np.ndarray:
    """Downsample ``img`` to ``size x size`` and label it with the ``n x n`` grid — the
    exact preprocessing the probe validates. Labels are drawn after resizing, in top/left
    margins outside the grid, so cell interiors remain uncontaminated."""
    if size < 1:
        raise ValueError("size must be >= 1")
    small = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return draw_grid(small, n)


def to_data_uri(img: np.ndarray) -> str:
    """RGB uint8 ndarray -> base64 PNG data URI for a vision chat message."""
    ok, buf = cv2.imencode(".png", img[:, :, ::-1])  # RGB -> BGR for OpenCV
    if not ok:
        raise RuntimeError("failed to PNG-encode image")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def build_prompt(n: int) -> str:
    return (
        f"You are looking at two images of the same {n}x{n} grid overlaid on a square "
        f"canvas. Image 1 is the TARGET (what the canvas should look like). Image 2 is "
        f"the CURRENT CANVAS (what it looks like now). Rows are numbered along the left "
        f"margin from top to bottom, 0-indexed. Columns are numbered along the top "
        f"margin from left to right, 0-indexed. Identify a cell by reading its row "
        f"number from the left margin and its column number from the top margin. Cell "
        f"interiors contain only paint; there are no labels inside cells. Both images "
        f"use the same grid.\n\n"
        f"If the current canvas already matches the target, call {DONE_TOOL_NAME}. Do "
        f"not keep searching a matching canvas for tiny or imagined differences.\n\n"
        f"If there is a clear mismatch, pick exactly ONE cell where the current canvas "
        f"differs most from the target, and the RGB color that would make that cell "
        f"match the target. Call {TOOL_NAME} with your choice.\n\n"
        f"Decide quickly. First decide whether the canvas is already complete. If it "
        f"is, report complete promptly. Otherwise scan for the single most obviously "
        f"wrong cell and call the paint function as soon as you have it. Do NOT audit "
        f"the grid cell by cell, do not list or compare every cell's state, and do not "
        f"rank candidates — one clear mismatch is enough. Keep your reasoning to one "
        f"or two sentences. The output you owe is a tool call, not a survey of the "
        f"canvas."
    )


def build_messages(target_uri: str, canvas_uri: str, n: int) -> List[Dict[str, Any]]:
    """The two-image user message: prompt, then the target, then the current canvas,
    each image preceded by a text label so their roles are unambiguous."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(n)},
                {"type": "text", "text": "Image 1 (TARGET):"},
                {"type": "image_url", "image_url": {"url": target_uri}},
                {"type": "text", "text": "Image 2 (CURRENT CANVAS):"},
                {"type": "image_url", "image_url": {"url": canvas_uri}},
            ],
        }
    ]


def build_request(
    model: str,
    target_img: np.ndarray,
    canvas_img: np.ndarray,
    n: int,
    *,
    size: int = DEFAULT_PROBE_SIZE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    tool_choice: Any = None,
) -> Dict[str, Any]:
    """Build the complete chat-completions request body for one paint decision.

    ``target_img`` / ``canvas_img`` are full-resolution canvas-space RGB arrays; they are
    downsampled and grid-labeled here. ``tool_choice`` defaults to the specific-function
    form the probe validated.
    """
    target_uri = to_data_uri(prepare_image(target_img, n, size))
    canvas_uri = to_data_uri(prepare_image(canvas_img, n, size))
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": build_messages(target_uri, canvas_uri, n),
        "tools": [PAINT_TOOL_SCHEMA, DONE_TOOL_SCHEMA],
        "tool_choice": TOOL_CHOICE_REQUIRED if tool_choice is None else tool_choice,
    }


def extract_reasoning(message: Dict[str, Any], tool_args: Dict[str, Any]) -> str:
    """Pull the model's stated reasoning out of a response message.

    Priority is ``reasoning_content`` (where the probed model actually puts its
    chain-of-thought, leaving the schema's ``reasoning`` argument empty), then the
    schema's ``reasoning`` field (which other models do populate), then ``content``.
    Returns "" when a model offers none — reasoning is captured for the later
    reasoning-stream feature, never required for a decision to be valid.
    """
    for candidate in (
        message.get("reasoning_content"),
        tool_args.get("reasoning"),
        message.get("content"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return ""


def parse_tool_call(response: Dict[str, Any], n: int) -> ToolDecision:
    """Extract a typed tool decision from a chat-completions response.

    Raises ``ValueError`` for anything unusable — no tool call, unknown function,
    malformed JSON arguments, wrong arity, a cell outside the ``n x n`` grid, or a
    channel outside 0..255. The caller (the planner) turns that into a retry and
    ultimately a skipped iteration; a bad response must never crash a paint run.
    """
    import json

    try:
        choice = response["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as ex:
        raise ValueError(f"response has no choices[0].message: {ex}") from ex

    tool_calls = message.get("tool_calls") or []
    if not tool_calls:
        finish = choice.get("finish_reason")
        raise ValueError(
            f"no tool_calls in response (finish_reason={finish!r}); the model did not "
            f"emit structured output"
        )
    if len(tool_calls) != 1:
        raise ValueError(
            f"expected exactly one tool_call, got {len(tool_calls)}; paint and done "
            f"decisions must not be combined"
        )

    try:
        function = tool_calls[0]["function"]
        name = function["name"]
        raw_args = function.get("arguments", "{}")
    except (KeyError, IndexError, TypeError) as ex:
        raise ValueError(f"malformed tool_call structure: {ex}") from ex

    # Fireworks returns arguments as a JSON string; tolerate a pre-parsed dict too.
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError as ex:
            raise ValueError(f"tool_call arguments are not valid JSON: {ex}") from ex
    elif isinstance(raw_args, dict):
        args = raw_args
    else:
        raise ValueError(f"tool_call arguments have unexpected type {type(raw_args)}")

    if not isinstance(args, dict):
        raise ValueError(f"tool_call arguments must be an object, got {type(args)}")

    if name == DONE_TOOL_NAME:
        return DoneDecision(reasoning=extract_reasoning(message, args))

    if name != TOOL_NAME:
        raise ValueError(f"unexpected tool call {name!r}")

    cell = _as_int_tuple(args.get("cell"), 2, "cell")
    color = _as_int_tuple(args.get("color"), 3, "color")

    i, j = cell
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"cell {cell} out of range for a {n}x{n} grid")
    if not all(0 <= c <= 255 for c in color):
        raise ValueError(f"color {color} has a channel outside 0..255")

    return PaintDecision(
        cell=(cell[0], cell[1]),
        color=(color[0], color[1], color[2]),
        reasoning=extract_reasoning(message, args),
    )


def _as_int_tuple(value: Any, arity: int, name: str) -> Tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list, got {type(value)}")
    if len(value) != arity:
        raise ValueError(f"{name} must have {arity} elements, got {len(value)}")
    out = []
    for v in value:
        # bool is an int subclass but is never a meaningful cell index or channel value.
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} contains a non-numeric element {v!r}")
        if isinstance(v, float) and not float(v).is_integer():
            raise ValueError(f"{name} contains a non-integer element {v!r}")
        out.append(int(v))
    return tuple(out)
