import { describe, expect, it } from "vitest";
import { initialState, reduce, type State } from "./reduce";
import type {
  FrameCapturedMsg,
  PlanDoneMsg,
  RunDoneMsg,
  RunStartMsg,
  StateUpdateMsg,
  VerifyDoneMsg,
} from "./types";

function runStart(overrides: Partial<RunStartMsg> = {}): RunStartMsg {
  return {
    type: "run.start",
    canvas_size: [600, 600],
    grid_n: 6,
    max_iterations: 60,
    max_region_failures: 3,
    error_threshold: 0.02,
    improvement_threshold: 0.001,
    reversible: true,
    has_undo: false,
    stroke_cost: 1,
    ...overrides,
  };
}

function stateUpdate(overrides: Partial<StateUpdateMsg> = {}): StateUpdateMsg {
  return {
    type: "state.update",
    iteration: 0,
    global_error: 0.17,
    status: "running",
    converged: false,
    ...overrides,
  };
}

function planDone(overrides: Partial<PlanDoneMsg> = {}): PlanDoneMsg {
  return {
    type: "plan.done",
    iteration: 1,
    intent: {
      cell: [1, 1],
      box: [100, 100, 200, 200],
      color: [255, 0, 0],
      error: 0.3,
      size: 100,
    },
    ...overrides,
  };
}

function verifyDone(overrides: Partial<VerifyDoneMsg> = {}): VerifyDoneMsg {
  return {
    type: "verify.done",
    iteration: 1,
    verdict: {
      accepted: true,
      cell: [1, 1],
      region_before: 0.3,
      region_after: 0.05,
      region_delta: -0.25,
      global_before: 0.17,
      global_after: 0.15,
      global_delta: -0.02,
    },
    ...overrides,
  };
}

function frameCaptured(overrides: Partial<FrameCapturedMsg> = {}): FrameCapturedMsg {
  return {
    type: "frame.captured",
    iteration: 3,
    width: 300,
    height: 300,
    image: "data:image/jpeg;base64,AAAA",
    ...overrides,
  };
}

function runDone(overrides: Partial<RunDoneMsg> = {}): RunDoneMsg {
  return {
    type: "run.done",
    iteration: 12,
    global_error: 0.0186,
    reason: "converged",
    converged: true,
    ...overrides,
  };
}

describe("reduce", () => {
  it("starts with an empty, disconnected baseline", () => {
    expect(initialState.connection).toBe("connecting");
    expect(initialState.config).toBeNull();
    expect(initialState.errorSeries).toEqual([]);
    expect(initialState.strokeLog).toEqual([]);
    expect(initialState.terminal).toBeNull();
    expect(initialState.frame).toBeNull();
  });

  it("captures run.start config", () => {
    const s = reduce(initialState, runStart({ grid_n: 8 }));
    expect(s.config?.grid_n).toBe(8);
  });

  it("folds the baseline state.update (iteration 0) into current + errorSeries", () => {
    const s = reduce(initialState, stateUpdate({ iteration: 0, global_error: 0.1717 }));
    expect(s.current).toEqual({
      iteration: 0,
      global_error: 0.1717,
      status: "running",
      converged: false,
    });
    expect(s.errorSeries).toEqual([{ iteration: 0, global_error: 0.1717 }]);
  });

  it("appends a new iteration's state.update to the series in order", () => {
    let s: State = initialState;
    s = reduce(s, stateUpdate({ iteration: 0, global_error: 0.17 }));
    s = reduce(s, stateUpdate({ iteration: 1, global_error: 0.15 }));
    s = reduce(s, stateUpdate({ iteration: 2, global_error: 0.12 }));
    expect(s.errorSeries.map((p) => p.iteration)).toEqual([0, 1, 2]);
    expect(s.errorSeries.map((p) => p.global_error)).toEqual([0.17, 0.15, 0.12]);
  });

  it("dedupes the series by iteration, keeping the latest value for that iteration", () => {
    // Mirrors orchestrator._finish: the final iteration gets a running update,
    // then a second status="done" update at the SAME iteration.
    let s: State = initialState;
    s = reduce(s, stateUpdate({ iteration: 12, global_error: 0.02, status: "running" }));
    s = reduce(s, stateUpdate({ iteration: 12, global_error: 0.0186, status: "done", converged: true }));
    expect(s.errorSeries).toEqual([{ iteration: 12, global_error: 0.0186 }]);
    expect(s.current?.status).toBe("done");
  });

  it("handles out-of-order arrival by re-sorting on iteration", () => {
    let s: State = initialState;
    s = reduce(s, stateUpdate({ iteration: 2, global_error: 0.1 }));
    s = reduce(s, stateUpdate({ iteration: 1, global_error: 0.15 }));
    expect(s.errorSeries.map((p) => p.iteration)).toEqual([1, 2]);
  });

  it("joins plan.done and verify.done for the same iteration into one stroke row", () => {
    let s: State = initialState;
    s = reduce(s, planDone({ iteration: 5 }));
    expect(s.strokeLog).toHaveLength(1);
    expect(s.strokeLog[0].verdict).toBeNull();

    s = reduce(s, verifyDone({ iteration: 5 }));
    expect(s.strokeLog).toHaveLength(1);
    expect(s.strokeLog[0].intent.cell).toEqual([1, 1]);
    expect(s.strokeLog[0].verdict?.accepted).toBe(true);
    expect(s.strokeLog[0].verdict?.region_delta).toBeCloseTo(-0.25);
  });

  it("keeps separate stroke rows per iteration", () => {
    let s: State = initialState;
    s = reduce(s, planDone({ iteration: 1 }));
    s = reduce(s, verifyDone({ iteration: 1, verdict: { ...verifyDone().verdict, accepted: true } }));
    s = reduce(s, planDone({ iteration: 2, intent: { ...planDone().intent, cell: [2, 2] } }));
    s = reduce(s, verifyDone({ iteration: 2, verdict: { ...verifyDone().verdict, accepted: false, cell: [2, 2] } }));

    expect(s.strokeLog).toHaveLength(2);
    expect(s.strokeLog[0].verdict?.accepted).toBe(true);
    expect(s.strokeLog[1].verdict?.accepted).toBe(false);
    expect(s.strokeLog[1].cell).toEqual([2, 2]);
  });

  it("stores the latest frame.captured, replacing the previous one", () => {
    let s: State = initialState;
    s = reduce(s, frameCaptured({ iteration: 3, image: "data:image/jpeg;base64,AAA" }));
    expect(s.frame).toEqual({ iteration: 3, width: 300, height: 300, image: "data:image/jpeg;base64,AAA" });

    s = reduce(s, frameCaptured({ iteration: 6, image: "data:image/jpeg;base64,BBB" }));
    expect(s.frame).toEqual({ iteration: 6, width: 300, height: 300, image: "data:image/jpeg;base64,BBB" });
  });

  it("records run.done as the terminal state", () => {
    const s = reduce(initialState, runDone({ reason: "budget", converged: false }));
    expect(s.terminal).toEqual(runDone({ reason: "budget", converged: false }));
  });

  it("treats a null/NaN-turned-null global_error as a valid, renderable point", () => {
    const s = reduce(initialState, stateUpdate({ iteration: 0, global_error: null }));
    expect(s.errorSeries).toEqual([{ iteration: 0, global_error: null }]);
    expect(s.current?.global_error).toBeNull();
  });

  it("resets series/log/terminal on a new run.start, without dropping connection status", () => {
    let s: State = initialState;
    s = reduce(s, { type: "__connection", status: "open" });
    s = reduce(s, runStart());
    s = reduce(s, stateUpdate({ iteration: 0, global_error: 0.17 }));
    s = reduce(s, planDone({ iteration: 1 }));
    s = reduce(s, verifyDone({ iteration: 1 }));
    s = reduce(s, frameCaptured({ iteration: 1 }));
    s = reduce(s, runDone({ reason: "converged" }));
    expect(s.errorSeries).toHaveLength(1);
    expect(s.strokeLog).toHaveLength(1);
    expect(s.terminal).not.toBeNull();
    expect(s.frame).not.toBeNull();

    // A second run begins on the same connection (e.g. the dashboard stayed
    // open across a restart of scripts/live_run.py).
    s = reduce(s, runStart({ grid_n: 8 }));
    expect(s.connection).toBe("open");
    expect(s.config?.grid_n).toBe(8);
    expect(s.errorSeries).toEqual([]);
    expect(s.strokeLog).toEqual([]);
    expect(s.terminal).toBeNull();
    expect(s.current).toBeNull();
    expect(s.frame).toBeNull();
  });

  it("updates connection status via the local connection action without touching data", () => {
    let s: State = initialState;
    s = reduce(s, runStart());
    s = reduce(s, { type: "__connection", status: "open" });
    expect(s.connection).toBe("open");
    expect(s.config).not.toBeNull();
  });
});
