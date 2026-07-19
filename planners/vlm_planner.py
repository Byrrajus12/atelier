"""``VLMPlanner`` — a vision model in the planner seat (Phase 1).

Where ``GreedyPlanner`` computes the worst cell arithmetically and fills it with its mean
color, this planner *looks* at the target and the current canvas and decides which cell to
paint next and what color, through a forced function call. It implements the same
``core.planner.Planner`` ABC and emits the same ``PaintIntent``, so the executor, easel,
verifier, and orchestrator are unchanged (CLAUDE.md Principle 6 — the planner is an
interface, not a fixed implementation).

Two design points worth stating plainly:

  * **The model proposes; the metric gates.** A paint decision must pass the pure core
    validator before it becomes a ``PaintIntent``. Rejected decisions are fed back to the
    model for another try, which keeps no-undo damage off the canvas without changing the
    planner/orchestrator contract.
  * **Reasoning is captured at the moment of decision.** The ``reasoning`` on the returned
    intent's model output comes from the *same* call that produced the cell and color, not
    from a second "explain yourself" round trip (Principle 6: stated reasoning must be the
    reasoning that actually drove the strokes). It is recorded now and surfaced later.

Failure policy: any unusable response — malformed JSON, no tool call, an out-of-range cell,
a network error, or a validation rejection — is retried a few times, and if no usable
decision remains ``plan`` raises ``PlannerSkip``.
The orchestrator treats that as a no-op: it re-observes and asks again next iteration, and
ends the run only if the skips keep coming. A flaky API degrades the run's pace, never its
correctness, and never crashes it.

The planner may return ``None`` only when the model explicitly reports completion and
the observation's own region-error grid confirms every cell is within the same
threshold used by ``GreedyPlanner``. A model that reports completion too early is
treated like any other unusable decision: the planner feeds the rejection back and retries
before eventually raising ``PlannerSkip``. The model proposes "done"; the metric decides
whether that claim is honest.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional, Tuple

from core.adapter import Color
from core.perception import Observation, cell_box
from core.planner import DEFAULT_ERROR_THRESHOLD, PaintIntent, Planner, PlannerSkip
from core.validator import validate_intent
from planners.fireworks_client import FireworksClient, FireworksError
from planners.vision_prompt import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_PROBE_SIZE,
    DoneDecision,
    PaintDecision,
    build_request,
    parse_tool_call,
)

DEFAULT_MODEL = "accounts/fireworks/models/qwen3p7-plus"
DEFAULT_PALETTE: Tuple[Color, ...] = ((255, 0, 0), (0, 0, 255), (17, 17, 17))
MAX_ATTEMPTS = 4

log = logging.getLogger(__name__)


class VLMPlanner(Planner):
    """Plans one paint action or verified completion by asking a vision model to choose
    between painting one grid cell and reporting the canvas complete.

    The ``client`` is injected so tests run against a fake and never touch the network.
    ``image_size`` defaults to the value the STEP 0 probe validated against the live API
    (128 px images stay legible). ``max_tokens`` is sized for the mid-run case rather than
    the probe's blank canvas: the model reasons before it emits the call, and on a
    partially-painted canvas that reasoning runs much longer, so a probe-tuned budget
    truncates the call away in-loop.
    """

    def __init__(
        self,
        client: FireworksClient,
        *,
        model: str = DEFAULT_MODEL,
        image_size: int = DEFAULT_PROBE_SIZE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        palette: Tuple[Color, ...] = DEFAULT_PALETTE,
    ) -> None:
        if image_size < 1:
            raise ValueError("image_size must be >= 1")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not palette:
            raise ValueError("palette must be non-empty")
        self._client = client
        self.model = model
        self.image_size = image_size
        self.max_tokens = max_tokens
        self.palette = tuple(palette)
        # The reasoning the model gave for its most recent decision, captured for the
        # later reasoning-stream feature. Not part of the Planner contract and not read
        # by the loop; None when the last call was skipped.
        self.last_reasoning: Optional[str] = None

    def plan(self, observation: Observation) -> Optional[PaintIntent]:
        """Ask the model for the next decision.

        Returns a ``PaintIntent`` for a paint decision, ``None`` for metric-confirmed
        completion, and raises ``PlannerSkip`` if every bounded attempt failed to produce
        a usable, validated answer."""
        n = int(observation.region_error.shape[0])
        body = build_request(
            self.model,
            observation.target.image,
            observation.frame.image,
            n,
            size=self.image_size,
            max_tokens=self.max_tokens,
        )

        request_body = body
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                response = self._client.create_chat_completion(request_body)
            except FireworksError as ex:
                log.warning("VLM call failed (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, ex)
                continue

            log.debug("VLM raw response (attempt %d/%d): %s", attempt, MAX_ATTEMPTS, response)

            try:
                decision = parse_tool_call(response, n)
            except ValueError as ex:
                log.warning(
                    "VLM response unusable (attempt %d/%d): %s -- raw response: %s",
                    attempt, MAX_ATTEMPTS, ex, response,
                )
                continue

            self.last_reasoning = decision.reasoning
            if isinstance(decision, DoneDecision):
                max_error = float(observation.region_error.max())
                if max_error <= DEFAULT_ERROR_THRESHOLD:
                    log.info(
                        "VLM reported canvas complete; metric confirmed max region "
                        "error %.4f <= %.4f: %s",
                        max_error, DEFAULT_ERROR_THRESHOLD,
                        decision.reasoning[:200] or "(none given)",
                    )
                    return None
                reason = (
                    f"canvas is not complete: max region error {max_error:.4f} > "
                    f"{DEFAULT_ERROR_THRESHOLD:.4f}; choose a cell that still differs"
                )
                log.warning("VLM reported canvas complete but metric rejected it: %s", reason)
                request_body = self._request_with_feedback(body, response, reason)
                continue

            if not isinstance(decision, PaintDecision):
                log.warning("VLM response produced unknown decision type %r", decision)
                continue

            ok, reason = validate_intent(
                observation,
                observation.target,
                decision.cell,
                decision.color,
                n,
                self.palette,
            )
            if not ok:
                log.warning("VLM paint decision rejected before execution: %s", reason)
                request_body = self._request_with_feedback(body, response, reason)
                continue

            i, j = decision.cell
            intent = PaintIntent(
                cell=(i, j),
                box=cell_box(i, j, n, observation.frame.size),
                color=decision.color,
                error=float(observation.region_error[i, j]),
            )
            log.info(
                "VLM chose cell %s color %s (region error %.4f): %s",
                intent.cell, intent.color, intent.error,
                decision.reasoning[:200] or "(none given)",
            )
            return intent

        self.last_reasoning = None
        log.warning("VLM produced no usable decision after %d attempts; skipping iteration", MAX_ATTEMPTS)
        raise PlannerSkip(f"no usable VLM decision after {MAX_ATTEMPTS} attempts")

    def _request_with_feedback(self, base_body, response, reason: str):
        """Return a fresh request body that includes one rejected decision and feedback."""
        body = copy.deepcopy(base_body)
        messages = list(body["messages"])
        try:
            rejected = response["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            rejected = {"role": "assistant", "content": "I made a move that was rejected."}
        messages.append(copy.deepcopy(rejected))
        messages.append({
            "role": "user",
            "content": (
                f"That move was rejected: {reason}. Please choose a different cell or "
                f"color. Do not repeat rejected moves."
            ),
        })
        body["messages"] = messages
        return body
