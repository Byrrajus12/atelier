"""VLM planner probe (Phase 1, STEP 0) — ONE real Fireworks API call to check whether a
serverless vision model can look at a target + a blank canvas and pick a sensible grid
cell + color via structured/function-call output, before anything is built around it.

Throwaway and manual: run it yourself with FIREWORKS_API_KEY set. Never imported by
pytest, never called by the orchestrator or any core/ module.

    FIREWORKS_API_KEY=... python scripts/probe_vlm.py --model accounts/fireworks/models/...

If the model id 404s, this automatically lists what Fireworks actually serves so the
right id is one glance away instead of a guessing game.

The probe target is the same 3-block layout as scripts/live_run.py's build_target
(red/blue/black blocks on white), so its correct answer is known ahead of time — the
printed "ground truth" lets you judge whether the model chose the RIGHT cell and a
SENSIBLE color, not merely whether it returned well-formed JSON. A model that returns
clean JSON but the WRONG cell is telling you the image resolution (--size) is too small
to read the grid labels, which is exactly the failure mode this probe exists to catch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.perception import cell_box  # noqa: E402
from planners.vision_prompt import (  # noqa: E402
    DEFAULT_PROBE_SIZE,
    TOOL_CHOICE_SPECIFIC,
    build_request,
)

FIREWORKS_CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"
FIREWORKS_MODELS_URL = "https://api.fireworks.ai/inference/v1/models"

GRID_N = 6
CANVAS_SIZE = (300, 300)  # (w, h) fed into cell_box before downsampling for the API

RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (17, 17, 17)

# Ground truth for this probe's fixed target (same layout as live_run.build_target),
# keyed by the same (i, j) cell numbering the model is asked to use, so its answer can
# be checked against a known-correct choice.
EXPECTED_CELLS = {
    (1, 1): "red",
    (1, 4): "blue",
    (4, 1): "black",
}


def build_target(size: tuple[int, int]) -> np.ndarray:
    """Same 3-block layout as scripts/live_run.py's build_target: white canvas, three
    2x2-cell solid blocks, grid-aligned and separated by a full white cell each way."""
    w, h = size
    img = np.full((h, w, 3), 255, dtype=np.uint8)

    def fill(i0, i1, j0, j1, color):
        x0, y0, _, _ = cell_box(i0, j0, GRID_N, size)
        _, _, x1, y1 = cell_box(i1, j1, GRID_N, size)
        img[y0:y1, x0:x1] = color

    fill(1, 2, 1, 2, RED)
    fill(1, 2, 4, 5, BLUE)
    fill(4, 5, 1, 2, BLACK)
    return img


# "required" is the looser "must call SOME tool" fallback, kept as a --tool-choice option
# in case a model/deployment rejects the specific-function form (TOOL_CHOICE_SPECIFIC,
# imported from planners.vision_prompt — the form this probe validated).
TOOL_CHOICE_REQUIRED = "required"


def call_fireworks(
    api_key: str,
    model: str,
    target_img: np.ndarray,
    canvas_img: np.ndarray,
    n: int,
    size: int,
    max_tokens: int,
    tool_choice,
) -> requests.Response:
    """Build the request through the SHARED helper the planner uses, so what this probe
    validates is exactly what the planner will send."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = build_request(
        model, target_img, canvas_img, n,
        size=size, max_tokens=max_tokens, tool_choice=tool_choice,
    )
    return requests.post(FIREWORKS_CHAT_URL, headers=headers, json=body, timeout=60)


def list_vision_models(api_key: str) -> list[str]:
    """Fallback diagnostic when the requested model id 404s: list what Fireworks
    actually serves, so the right id is one glance away instead of a guessing game."""
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(FIREWORKS_MODELS_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ids = [m.get("id", "") for m in data.get("data", [])]
    # Best-effort filter for vision-ish ids (readability only); return everything if the
    # filter finds nothing so the fallback is never empty-handed.
    vision_like = [i for i in ids if any(k in i.lower() for k in ("vl", "vision"))]
    return vision_like or ids


def looks_like_model_not_found(resp: requests.Response) -> bool:
    if resp.status_code == 404:
        return True
    if resp.status_code >= 400:
        try:
            msg = json.dumps(resp.json()).lower()
        except ValueError:
            msg = resp.text.lower()
        return "model" in msg and (
            "not found" in msg or "does not exist" in msg or "unknown" in msg
        )
    return False


def looks_like_tool_choice_rejected(resp: requests.Response) -> bool:
    """True if the API itself rejected the tool_choice value (not a model-not-found
    error) -- distinct from the model merely ignoring/under-using the tool, which shows
    up as a 200 with no tool_calls, not an error status."""
    if resp.status_code < 400:
        return False
    try:
        msg = json.dumps(resp.json()).lower()
    except ValueError:
        msg = resp.text.lower()
    return "tool_choice" in msg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="accounts/fireworks/models/qwen3p7-plus",
        help="Fireworks model id to probe (override if this 404s).",
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_PROBE_SIZE,
        help="Downsampled square edge length (px) sent to the API.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=500,
        help="Completion token budget -- needs headroom for reasoning before the tool call.",
    )
    parser.add_argument(
        "--tool-choice", choices=("specific", "required"), default="specific",
        help=(
            "'specific' forces exactly propose_paint_cell (preferred). 'required' is the "
            "looser 'must call some tool' form -- use it if 'specific' gets rejected by "
            "the API for this model/deployment."
        ),
    )
    parser.add_argument(
        "--price-per-million-in", type=float, default=None,
        help="Optional $/1M input tokens (current Fireworks pricing) to compute a cost estimate.",
    )
    parser.add_argument(
        "--price-per-million-out", type=float, default=None,
        help="Optional $/1M output tokens, to compute a cost estimate.",
    )
    args = parser.parse_args()
    tool_choice = TOOL_CHOICE_SPECIFIC if args.tool_choice == "specific" else TOOL_CHOICE_REQUIRED

    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        print("FIREWORKS_API_KEY is not set.", file=sys.stderr)
        return 1

    target = build_target(CANVAS_SIZE)
    canvas = np.full_like(target, 255)  # blank white canvas, same size as target

    print(f"probe image size: {args.size}x{args.size}  grid: {GRID_N}x{GRID_N}")
    print("ground truth for this fixed target:")
    for cell, color in EXPECTED_CELLS.items():
        print(f"  cell {cell}: {color}")
    print("  (every other cell is white/unpainted in the target)")

    print(f"\ncalling model: {args.model}  tool_choice={args.tool_choice}  max_tokens={args.max_tokens}")
    t0 = time.time()
    try:
        resp = call_fireworks(
            api_key, args.model, target, canvas, GRID_N,
            args.size, args.max_tokens, tool_choice,
        )
    except requests.RequestException as ex:
        print(f"request failed: {ex}")
        return 1
    latency = time.time() - t0

    if looks_like_model_not_found(resp):
        print(f"\nmodel '{args.model}' was not found/rejected (status {resp.status_code}).")
        print("fetching available models from Fireworks to help pick the right id...")
        try:
            candidates = list_vision_models(api_key)
            print("candidate vision-ish model ids:")
            for c in candidates:
                print(f"  {c}")
        except Exception as ex:
            print(f"  (could not list models: {ex})")
        print(f"\nraw error body: {resp.text}")
        return 1

    if looks_like_tool_choice_rejected(resp):
        print(
            f"\nrequest failed (status {resp.status_code}) and the error mentions "
            f"tool_choice -- this model/deployment likely rejects the '{args.tool_choice}' "
            f"form. Re-run with "
            f"--tool-choice {'required' if args.tool_choice == 'specific' else 'specific'} "
            f"or try a different serverless model."
        )
        print(f"\nraw error body: {resp.text}")
        return 1

    if resp.status_code >= 400:
        print(f"\nrequest failed: status {resp.status_code}")
        print(resp.text)
        return 1

    data = resp.json()
    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")

    print(f"\nlatency: {latency:.2f}s")
    print(f"tokens: prompt={prompt_tokens}  completion={completion_tokens}")

    if (
        args.price_per_million_in is not None
        and args.price_per_million_out is not None
        and prompt_tokens is not None
        and completion_tokens is not None
    ):
        cost = (
            prompt_tokens / 1_000_000 * args.price_per_million_in
            + completion_tokens / 1_000_000 * args.price_per_million_out
        )
        print(f"estimated cost: ${cost:.6f}  (using supplied $/1M prices)")
    else:
        print(
            "estimated cost: not computed (pass --price-per-million-in/out with current "
            "Fireworks pricing to compute one)"
        )

    print("\nraw response JSON (truncated to 4000 chars):")
    print(json.dumps(data, indent=2)[:4000])

    # --- parse + validate the structured tool call -------------------------------
    choice0 = data["choices"][0]
    finish_reason = choice0.get("finish_reason")
    print(f"\nfinish_reason: {finish_reason}")
    try:
        message = choice0["message"]
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            print("\nPARSE: no tool_calls in response -- structured output did not fire.")
            prose = message.get("content") or ""
            if prose:
                print("model emitted prose instead (first 1000 chars):")
                print(prose[:1000])
            if finish_reason == "length":
                print(
                    f"finish_reason=length: it ran out of the {args.max_tokens}-token "
                    f"budget before reaching the tool call. Try a larger --max-tokens."
                )
            return 1
        raw_args = tool_calls[0]["function"]["arguments"]
        parsed = json.loads(raw_args)
        cell = tuple(int(v) for v in parsed["cell"])
        color = tuple(int(v) for v in parsed["color"])
        reasoning = parsed.get("reasoning", "")
    except Exception as ex:
        print(f"\nPARSE FAILED: {ex}")
        return 1

    cell_valid = len(cell) == 2 and all(0 <= c < GRID_N for c in cell)
    color_valid = len(color) == 3 and all(0 <= c <= 255 for c in color)

    print("\n--- parsed model choice -------------------------------------------")
    print(f"cell:      {cell}   (in-range: {cell_valid})")
    print(f"color:     {color}  (valid RGB: {color_valid})")
    print(f"reasoning: {reasoning}")

    print("\n--- how to judge this ----------------------------------------------")
    if cell in EXPECTED_CELLS:
        print(
            f"cell {cell} IS one of the painted target cells "
            f"(expected color family: {EXPECTED_CELLS[cell]})."
        )
    else:
        print(
            f"cell {cell} is NOT one of the painted target cells "
            f"{list(EXPECTED_CELLS.keys())} -- if this looks wrong, the "
            f"{args.size}x{args.size} downsample or the grid labels may be too small "
            f"to read reliably; try a larger --size."
        )
    print(
        "Judge for yourself: does the chosen cell/color look like a sensible next "
        "move, and does the reasoning describe something actually visible in the "
        "images?"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
