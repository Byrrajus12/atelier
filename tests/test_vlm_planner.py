"""Tests for planners/ — the VLM planner, its request construction, and its response
parsing. Every test runs against a FAKE injected client: no API key, no network, no cost.
The real Fireworks call happens only in the live run and the manual probe, never here."""

import json

import numpy as np
import pytest

from core.adapter import Frame
from core.perception import Observation, cell_box
from core.planner import PaintIntent, Planner, PlannerSkip
from core.target import Target
from planners.fireworks_client import (
    FireworksClient,
    FireworksError,
    HTTPFireworksClient,
)
from planners.vision_prompt import (
    DONE_TOOL_NAME,
    DoneDecision,
    PaintDecision,
    TOOL_CHOICE_REQUIRED,
    TOOL_NAME,
    build_request,
    draw_grid,
    extract_reasoning,
    parse_tool_call,
    prepare_image,
)
from planners.vlm_planner import VLMPlanner


def solid(h, w, rgb):
    return np.full((h, w, 3), rgb, dtype=np.uint8)


def make_observation(n=4, h=40, w=40, region_error=None):
    """A fabricated Observation, mirroring tests/test_planner.py's helper. The canvas is
    white and the target has a red patch in cell (1, 2), so color reads are checkable."""
    canvas = solid(h, w, (255, 255, 255))
    target = solid(h, w, (255, 255, 255))
    x0, y0, x1, y1 = cell_box(1, 2, n, (w, h))
    target[y0:y1, x0:x1] = (255, 0, 0)
    grid = region_error if region_error is not None else np.zeros((n, n))
    if region_error is None:
        grid[1, 2] = 0.8
    return Observation(
        frame=Frame(canvas),
        target=Target(target),
        global_error=float(grid.mean()),
        region_error=grid,
        heatmap=np.zeros_like(canvas),
    )


def tool_response(cell, color, reasoning_content=None, tool_reasoning=""):
    """A well-formed chat-completions response carrying one propose_paint_cell call."""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": TOOL_NAME,
                    "arguments": json.dumps(
                        {"cell": list(cell), "color": list(color), "reasoning": tool_reasoning}
                    ),
                },
            }
        ],
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {"choices": [{"message": message, "finish_reason": "tool_calls"}]}


def done_response(reasoning_content=None, tool_reasoning=""):
    """A well-formed chat-completions response carrying report_canvas_complete."""
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": DONE_TOOL_NAME,
                    "arguments": json.dumps({"reasoning": tool_reasoning}),
                },
            }
        ],
    }
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {"choices": [{"message": message, "finish_reason": "tool_calls"}]}


class FakeFireworksClient(FireworksClient):
    """Returns canned responses in order; a response may be an Exception to raise. Records
    every request body so prompt construction can be asserted on."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create_chat_completion(self, body):
        self.requests.append(body)
        if not self.responses:
            raise AssertionError("FakeFireworksClient called more times than scripted")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    @property
    def call_count(self):
        return len(self.requests)


# --- the Planner contract (Principle 6) -------------------------------------------
def test_vlm_planner_is_a_planner():
    assert isinstance(VLMPlanner(FakeFireworksClient([])), Planner)


def test_vlm_planner_rejects_bad_params():
    with pytest.raises(ValueError):
        VLMPlanner(FakeFireworksClient([]), image_size=0)
    with pytest.raises(ValueError):
        VLMPlanner(FakeFireworksClient([]), max_tokens=0)


# --- happy path: grounding a chosen cell into a PaintIntent ------------------------
def test_plan_builds_intent_from_the_models_choice():
    obs = make_observation(n=4, h=40, w=40)
    client = FakeFireworksClient([tool_response((1, 2), (255, 0, 0))])
    intent = VLMPlanner(client).plan(obs)

    assert isinstance(intent, PaintIntent)
    assert intent.cell == (1, 2)
    assert intent.color == (255, 0, 0)
    assert client.call_count == 1


def test_plan_grounds_the_cell_to_the_matching_box():
    """The (i,j) -> box mapping must go through cell_box, or the executor paints in the
    wrong place (or transposed)."""
    n, h, w = 4, 40, 40
    obs = make_observation(n=n, h=h, w=w)
    client = FakeFireworksClient([tool_response((0, 3), (10, 20, 30))])
    intent = VLMPlanner(client).plan(obs)
    assert intent.box == cell_box(0, 3, n, (w, h))


def test_plan_carries_the_region_error_of_the_chosen_cell():
    """error is read from the observation at the model's chosen cell — not the max error,
    and not whatever the model might claim."""
    grid = np.zeros((4, 4))
    grid[1, 2] = 0.8
    grid[3, 3] = 0.9  # hotter, but the model picks (1,2) anyway; its choice stands
    obs = make_observation(n=4, region_error=grid)
    client = FakeFireworksClient([tool_response((1, 2), (255, 0, 0))])
    intent = VLMPlanner(client).plan(obs)
    assert intent.cell == (1, 2)
    assert intent.error == pytest.approx(0.8)


def test_plan_uses_grid_n_from_the_observation():
    """Grid size is derived from the observation's region_error, so the planner and the
    perception grid can never disagree about what (i,j) means."""
    n = 6
    grid = np.zeros((n, n))
    grid[5, 5] = 0.5
    obs = make_observation(n=n, h=60, w=60, region_error=grid)
    client = FakeFireworksClient([tool_response((5, 5), (0, 0, 255))])
    intent = VLMPlanner(client).plan(obs)
    assert intent.cell == (5, 5)  # would be out of range for the n=4 default
    assert intent.box == cell_box(5, 5, n, (60, 60))


def test_plan_returns_none_when_done_is_metric_confirmed():
    grid = np.full((4, 4), 0.01)
    obs = make_observation(n=4, region_error=grid)
    client = FakeFireworksClient([done_response(reasoning_content="everything matches")])
    planner = VLMPlanner(client)

    assert planner.plan(obs) is None
    assert planner.last_reasoning == "everything matches"
    assert client.call_count == 1


def test_plan_rejects_false_done_as_planner_skip():
    grid = np.zeros((4, 4))
    grid[1, 2] = 0.8
    obs = make_observation(n=4, region_error=grid)
    client = FakeFireworksClient([done_response(), done_response()])

    with pytest.raises(PlannerSkip):
        VLMPlanner(client).plan(obs)
    assert client.call_count == 2


def test_malformed_done_response_retries_and_can_recover():
    grid = np.full((4, 4), 0.01)
    obs = make_observation(n=4, region_error=grid)
    bad = done_response()
    bad["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    client = FakeFireworksClient([bad, done_response(reasoning_content="retry complete")])
    planner = VLMPlanner(client)

    assert planner.plan(obs) is None
    assert planner.last_reasoning == "retry complete"
    assert client.call_count == 2


# --- reasoning capture -------------------------------------------------------------
def test_reasoning_content_is_captured_as_the_primary_source():
    """The probed model puts chain-of-thought in reasoning_content and leaves the schema's
    reasoning argument empty — so reasoning_content must win."""
    obs = make_observation()
    client = FakeFireworksClient([
        tool_response((1, 2), (255, 0, 0), reasoning_content="the target is red here")
    ])
    planner = VLMPlanner(client)
    planner.plan(obs)
    assert planner.last_reasoning == "the target is red here"


def test_schema_reasoning_field_is_used_when_reasoning_content_is_absent():
    """Other models populate the tool schema's reasoning field instead; both are read."""
    obs = make_observation()
    client = FakeFireworksClient([
        tool_response((1, 2), (255, 0, 0), tool_reasoning="picked the red block")
    ])
    planner = VLMPlanner(client)
    planner.plan(obs)
    assert planner.last_reasoning == "picked the red block"


def test_extract_reasoning_prefers_reasoning_content_over_both_others():
    msg = {"reasoning_content": "primary", "content": "prose"}
    assert extract_reasoning(msg, {"reasoning": "schema"}) == "primary"
    assert extract_reasoning({"content": "prose"}, {"reasoning": "schema"}) == "schema"
    assert extract_reasoning({"content": "prose"}, {}) == "prose"
    assert extract_reasoning({}, {}) == ""


def test_missing_reasoning_does_not_block_a_valid_decision():
    """Reasoning is captured for a later feature; its absence must never void a decision."""
    obs = make_observation()
    client = FakeFireworksClient([tool_response((1, 2), (255, 0, 0))])
    planner = VLMPlanner(client)
    intent = planner.plan(obs)
    assert intent is not None
    assert planner.last_reasoning == ""


# --- request construction ----------------------------------------------------------
def test_request_requires_a_tool_call_and_offers_paint_or_done():
    """The model must call a tool, but may choose paint or verified completion."""
    body = build_request("m", solid(40, 40, (255, 255, 255)), solid(40, 40, (0, 0, 0)), 4)
    assert body["tool_choice"] == TOOL_CHOICE_REQUIRED
    assert {tool["function"]["name"] for tool in body["tools"]} == {
        TOOL_NAME, DONE_TOOL_NAME,
    }


def test_request_carries_both_images_and_the_grid_size():
    n = 6
    body = build_request("m", solid(60, 60, (255, 255, 255)), solid(60, 60, (0, 0, 0)), n)
    content = body["messages"][0]["content"]
    images = [c for c in content if c["type"] == "image_url"]
    assert len(images) == 2  # target + current canvas
    assert all(i["image_url"]["url"].startswith("data:image/png;base64,") for i in images)
    prompt = content[0]["text"]
    assert f"{n}x{n}" in prompt


def test_request_defaults_match_the_probe_validated_values():
    body = build_request("m", solid(40, 40, (0, 0, 0)), solid(40, 40, (0, 0, 0)), 4)
    # Sized for the mid-run case, not the blank-canvas probe: on a partially-painted
    # canvas the model's pre-call reasoning runs long enough that 500 truncated the call
    # away in a live run (finish_reason=length, no tool_calls).
    assert body["max_tokens"] >= 1500
    assert body["model"] == "m"


def test_prepare_image_downsamples_and_labels():
    img = solid(300, 300, (255, 255, 255))
    out = prepare_image(img, 6, size=128)
    assert out.shape == (128, 128, 3)
    # Grid lines + labels must have marked the image; a blank white frame means the
    # model would receive no visual grounding for cell indices.
    assert not np.all(out == 255)


def test_draw_grid_does_not_mutate_its_input():
    img = solid(60, 60, (255, 255, 255))
    before = img.copy()
    draw_grid(img, 6)
    assert np.array_equal(img, before)


def test_planner_passes_its_image_size_through_to_the_request():
    obs = make_observation()
    client = FakeFireworksClient([tool_response((1, 2), (255, 0, 0))])
    VLMPlanner(client, image_size=64, max_tokens=700, model="my-model").plan(obs)
    body = client.requests[0]
    assert body["model"] == "my-model"
    assert body["max_tokens"] == 700


# --- parsing: every rejection path -------------------------------------------------
def test_parse_accepts_a_well_formed_paint_call():
    decision = parse_tool_call(tool_response((1, 2), (255, 0, 0)), 4)
    assert isinstance(decision, PaintDecision)
    assert decision.cell == (1, 2)
    assert decision.color == (255, 0, 0)
    assert decision.reasoning == ""


def test_parse_accepts_a_well_formed_done_call():
    decision = parse_tool_call(done_response(reasoning_content="looks finished"), 4)
    assert isinstance(decision, DoneDecision)
    assert decision.reasoning == "looks finished"


def test_parse_rejects_ambiguous_multiple_tool_calls():
    resp = done_response(reasoning_content="finished")
    resp["choices"][0]["message"]["tool_calls"].append(
        tool_response((1, 2), (255, 0, 0))["choices"][0]["message"]["tool_calls"][0]
    )
    with pytest.raises(ValueError, match="exactly one tool_call"):
        parse_tool_call(resp, 4)


def test_parse_accepts_dict_arguments_as_well_as_a_json_string():
    resp = tool_response((1, 2), (255, 0, 0))
    resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {
        "cell": [1, 2], "color": [255, 0, 0]
    }
    decision = parse_tool_call(resp, 4)
    assert isinstance(decision, PaintDecision)
    assert decision.cell == (1, 2)


@pytest.mark.parametrize("bad_response", [
    {},                                                        # no choices
    {"choices": []},                                           # empty choices
    {"choices": [{"message": {}, "finish_reason": "length"}]},  # no tool call (truncated)
    {"choices": [{"message": {"tool_calls": []}}]},            # empty tool_calls
])
def test_parse_rejects_structurally_broken_responses(bad_response):
    with pytest.raises(ValueError):
        parse_tool_call(bad_response, 4)


def test_parse_rejects_malformed_json_arguments():
    resp = tool_response((1, 2), (255, 0, 0))
    resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = "{not json"
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_tool_call(resp, 4)


@pytest.mark.parametrize("cell", [(4, 0), (0, 4), (-1, 0), (0, -1), (99, 99)])
def test_parse_rejects_out_of_range_cells(cell):
    with pytest.raises(ValueError, match="out of range"):
        parse_tool_call(tool_response(cell, (255, 0, 0)), 4)


@pytest.mark.parametrize("color", [(256, 0, 0), (-1, 0, 0), (0, 0, 300)])
def test_parse_rejects_out_of_range_colors(color):
    with pytest.raises(ValueError, match="outside 0..255"):
        parse_tool_call(tool_response((1, 2), color), 4)


@pytest.mark.parametrize("args", [
    {"cell": [1], "color": [255, 0, 0]},              # wrong arity
    {"cell": [1, 2, 3], "color": [255, 0, 0]},        # wrong arity
    {"cell": [1, 2], "color": [255, 0]},              # wrong arity
    {"cell": "1,2", "color": [255, 0, 0]},            # not a list
    {"cell": [1, "x"], "color": [255, 0, 0]},         # non-numeric
    {"cell": [1.5, 2], "color": [255, 0, 0]},         # non-integer
    {"color": [255, 0, 0]},                           # missing cell
    {"cell": [1, 2]},                                 # missing color
])
def test_parse_rejects_bad_argument_shapes(args):
    resp = tool_response((1, 2), (255, 0, 0))
    resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    with pytest.raises(ValueError):
        parse_tool_call(resp, 4)


# --- error handling: retry once, then skip -----------------------------------------
# A failure to decide raises PlannerSkip, NEVER returns None. None means "the canvas has
# converged", which a model that did not answer is in no position to claim — the live bug
# this guards ended a run reported converged=True at ~13% error with the canvas mostly
# blank. See core.planner.PlannerSkip.
def test_network_error_retries_once_then_skips():
    obs = make_observation()
    client = FakeFireworksClient([
        FireworksError("connection reset"),
        FireworksError("connection reset again"),
    ])
    with pytest.raises(PlannerSkip):
        VLMPlanner(client).plan(obs)
    assert client.call_count == 2  # exactly one retry, no infinite hammering


def test_malformed_response_retries_once_then_skips():
    obs = make_observation()
    bad = {"choices": [{"message": {"content": "I think cell 1,2 looks wrong."},
                        "finish_reason": "length"}]}
    client = FakeFireworksClient([bad, bad])
    with pytest.raises(PlannerSkip):
        VLMPlanner(client).plan(obs)
    assert client.call_count == 2


def test_out_of_range_cell_retries_once_then_skips():
    obs = make_observation(n=4)
    client = FakeFireworksClient([
        tool_response((9, 9), (255, 0, 0)),
        tool_response((9, 9), (255, 0, 0)),
    ])
    with pytest.raises(PlannerSkip):
        VLMPlanner(client).plan(obs)
    assert client.call_count == 2


def test_a_failure_to_decide_is_never_reported_as_convergence():
    """The core of the fix, asserted directly: whatever goes wrong, ``plan`` raises
    PlannerSkip rather than returning None. The orchestrator reads None as "converged",
    so returning it here would mislabel an unfinished run a success."""
    obs = make_observation()
    for failure in (FireworksError("down"),
                    {"choices": [{"message": {"content": "prose"},
                                  "finish_reason": "length"}]},
                    {"unexpected": "shape"}):
        planner = VLMPlanner(FakeFireworksClient([failure, failure]))
        with pytest.raises(PlannerSkip):
            planner.plan(obs)


def test_a_retry_that_succeeds_yields_a_valid_intent():
    """The retry exists to survive a one-off bad sample, so a good second answer must be
    used rather than discarded."""
    obs = make_observation()
    client = FakeFireworksClient([
        FireworksError("timeout"),
        tool_response((1, 2), (255, 0, 0), reasoning_content="second try"),
    ])
    planner = VLMPlanner(client)
    intent = planner.plan(obs)
    assert intent is not None
    assert intent.cell == (1, 2)
    assert planner.last_reasoning == "second try"
    assert client.call_count == 2


def test_skip_clears_stale_reasoning():
    """A skipped iteration must not leave the previous decision's reasoning behind, or a
    later consumer would attribute it to the wrong (non-)decision."""
    obs = make_observation()
    client = FakeFireworksClient([
        tool_response((1, 2), (255, 0, 0), reasoning_content="first"),
        FireworksError("down"),
        FireworksError("still down"),
    ])
    planner = VLMPlanner(client)
    planner.plan(obs)
    assert planner.last_reasoning == "first"
    with pytest.raises(PlannerSkip):
        planner.plan(obs)
    assert planner.last_reasoning is None


def test_a_hostile_response_produces_a_skip_and_nothing_worse():
    """No API response should be able to crash a paint run with an unexpected exception —
    the loop must always get either an intent or a PlannerSkip it knows how to handle."""
    obs = make_observation()
    for junk in ({"choices": [{"message": {"tool_calls": [{"function": {}}]}}]},
                 {"choices": [{"message": {"tool_calls": [{}]}}]},
                 {"unexpected": "shape"}):
        client = FakeFireworksClient([junk, junk])
        with pytest.raises(PlannerSkip):
            VLMPlanner(client).plan(obs)


# --- the real client's construction (no network) -----------------------------------
def test_http_client_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="no Fireworks API key"):
        HTTPFireworksClient()


def test_http_client_reads_the_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    client = HTTPFireworksClient()
    assert client._url.endswith("/chat/completions")
