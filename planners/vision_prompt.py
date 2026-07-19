"""Shared VLM request construction — the exact image encoding and request shape the
STEP 0 probe validated against the live Fireworks API.

Both ``scripts/probe_vlm.py`` and ``planners/vlm_planner.py`` build their requests from
here, so the planner sends what the probe proved works rather than a re-derived variant
that could silently drift (different resolution, missing grid labels, a tool_choice form
the model ignores).

What the probe established, and why each piece is here:

  * **Grid-labeled images.** Grid lines plus an ``i,j`` label at each cell center let the
    model ground a cell index visually instead of inferring pixel math from a text
    description. Validated legible at 128x128.
  * **Forced tool call.** ``tool_choice`` pinned to the specific ``propose_paint_cell``
    function. Left to its own devices the model narrated prose and never called the tool.
  * **Reasoning budget.** The model spends its reasoning *before* emitting the call, so
    the completion budget must clear that or the call is truncated away
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
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

DEFAULT_PROBE_SIZE = 128  # downsampled square edge length; validated legible by the probe
# Clears the model's pre-call reasoning spend with headroom. Tuned for the MID-RUN case,
# not the blank-canvas one: on a partially-painted canvas the model compares target vs
# current cell by cell and writes far longer reasoning before it emits the call. 500 was
# enough for the easy probe and truncated in-loop (finish_reason=length, no tool call).
DEFAULT_MAX_TOKENS = 1500

TOOL_NAME = "propose_paint_cell"

# Pinned to the specific function. "auto" (and an unforced tool list) let the model
# answer in prose instead; the specific form is what the probe confirmed fires.
TOOL_CHOICE_SPECIFIC: Dict[str, Any] = {
    "type": "function",
    "function": {"name": TOOL_NAME},
}

TOOL_SCHEMA: Dict[str, Any] = {
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


def draw_grid(img: np.ndarray, n: int) -> np.ndarray:
    """Overlay grid lines + ``i,j`` labels at each cell center on a COPY of ``img``.

    Labels are drawn with a black outline under white fill so they stay readable over any
    cell color (including the near-black swatch)."""
    if n < 1:
        raise ValueError("n must be >= 1")
    out = img.copy()
    h, w = out.shape[:2]
    row_edges = np.linspace(0, h, n + 1).astype(int)
    col_edges = np.linspace(0, w, n + 1).astype(int)
    for y in row_edges:
        cv2.line(out, (0, int(y)), (w, int(y)), (128, 128, 128), 1)
    for x in col_edges:
        cv2.line(out, (int(x), 0), (int(x), h), (128, 128, 128), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(n):
        for j in range(n):
            cy = int((row_edges[i] + row_edges[i + 1]) // 2)
            cx = int((col_edges[j] + col_edges[j + 1]) // 2)
            label = f"{i},{j}"
            cv2.putText(out, label, (cx - 12, cy + 4), font, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, label, (cx - 12, cy + 4), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def prepare_image(img: np.ndarray, n: int, size: int = DEFAULT_PROBE_SIZE) -> np.ndarray:
    """Downsample ``img`` to ``size x size`` and label it with the ``n x n`` grid — the
    exact preprocessing the probe validated. Downsample first, then label, so the labels
    are drawn at final resolution and stay crisp instead of being resampled to mush."""
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
        f"the CURRENT CANVAS (what it looks like now). Grid cells are labeled 'i,j' at "
        f"their centers: i is the row (0 at top), j is the column (0 at left). Both "
        f"images use the same grid.\n\n"
        f"Pick exactly ONE cell where the current canvas differs most from the target, "
        f"and the RGB color that would make that cell match the target. Call "
        f"{TOOL_NAME} with your choice.\n\n"
        f"Decide quickly. Scan for the single most obviously wrong cell and call the "
        f"function as soon as you have it. Do NOT audit the grid cell by cell, do not "
        f"list or compare every cell's state, and do not rank candidates — one clear "
        f"mismatch is enough. Keep your reasoning to one or two sentences. The output "
        f"you owe is a decision, not a survey of the canvas."
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
        "messages": build_messages(target_uri, canvas_uri, n),
        "tools": [TOOL_SCHEMA],
        "tool_choice": TOOL_CHOICE_SPECIFIC if tool_choice is None else tool_choice,
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


def parse_tool_call(
    response: Dict[str, Any], n: int
) -> Tuple[Tuple[int, int], Tuple[int, int, int], str]:
    """Extract ``(cell, color, reasoning)`` from a chat-completions response.

    Raises ``ValueError`` for anything unusable — no tool call, malformed JSON arguments,
    wrong arity, a cell outside the ``n x n`` grid, or a channel outside 0..255. The
    caller (the planner) turns that into a retry and ultimately a skipped iteration; a
    bad response must never crash a paint run.
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

    try:
        raw_args = tool_calls[0]["function"]["arguments"]
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

    cell = _as_int_tuple(args.get("cell"), 2, "cell")
    color = _as_int_tuple(args.get("color"), 3, "color")

    i, j = cell
    if not (0 <= i < n and 0 <= j < n):
        raise ValueError(f"cell {cell} out of range for a {n}x{n} grid")
    if not all(0 <= c <= 255 for c in color):
        raise ValueError(f"color {color} has a channel outside 0..255")

    return cell, (color[0], color[1], color[2]), extract_reasoning(message, args)


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
