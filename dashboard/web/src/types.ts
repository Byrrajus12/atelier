// Wire message shapes — kept field-for-field identical to
// dashboard/publisher.py's `event_to_message`, which itself mirrors
// core/events.py. Do NOT rename fields to camelCase: a paraphrase here is
// exactly the kind of drift that renders blank against a real run while
// still passing tests against a hand-written mock.

export interface RunStartMsg {
  type: "run.start";
  canvas_size: [number, number];
  grid_n: number;
  max_iterations: number;
  max_region_failures: number;
  error_threshold: number | null;
  improvement_threshold: number | null;
  reversible: boolean;
  has_undo: boolean;
  stroke_cost: number | null;
}

export interface ObserveDoneMsg {
  type: "observe.done";
  iteration: number;
  global_error: number | null;
  region_error: (number | null)[][];
  heatmap_ref: number;
}

export interface PlanIntent {
  cell: [number, number];
  box: [number, number, number, number];
  color: [number, number, number];
  error: number | null;
  size: number | null;
}

export interface PlanDoneMsg {
  type: "plan.done";
  iteration: number;
  intent: PlanIntent;
}

export interface ExecuteDoneMsg {
  type: "execute.done";
  iteration: number;
  cell: [number, number];
  stroke_count: number;
}

export interface Verdict {
  accepted: boolean;
  cell: [number, number];
  region_before: number | null;
  region_after: number | null;
  region_delta: number | null;
  global_before: number | null;
  global_after: number | null;
  global_delta: number | null;
}

export interface VerifyDoneMsg {
  type: "verify.done";
  iteration: number;
  verdict: Verdict;
}

// Sent only periodically (publisher-side cadence), plus always the run's final frame —
// see dashboard/publisher.py's frame_cadence knob and force-publish-on-run.done rule.
export interface FrameCapturedMsg {
  type: "frame.captured";
  iteration: number;
  width: number;
  height: number;
  image: string; // data:image/jpeg;base64,...
}

export const STATUS_RUNNING = "running";
export const STATUS_DONE = "done";

export interface StateUpdateMsg {
  type: "state.update";
  iteration: number;
  global_error: number | null;
  status: typeof STATUS_RUNNING | typeof STATUS_DONE;
  converged: boolean;
}

export const REASON_CONVERGED = "converged";
export const REASON_BUDGET = "budget";
export const REASON_STALLED = "stalled";
export const REASON_STALLED_NO_PROGRESS = "stalled_no_progress";
export const REASON_CANVAS_LOST = "canvas_lost";

export interface RunDoneMsg {
  type: "run.done";
  iteration: number;
  global_error: number | null;
  reason: string;
  converged: boolean;
}

export type WireMessage =
  | RunStartMsg
  | ObserveDoneMsg
  | PlanDoneMsg
  | ExecuteDoneMsg
  | VerifyDoneMsg
  | FrameCapturedMsg
  | StateUpdateMsg
  | RunDoneMsg;

export type ConnectionStatus = "connecting" | "open" | "reconnecting" | "closed";

// A local, non-wire action the hook dispatches through the same reducer so
// `reduce` stays the single fold point for everything that changes state.
export interface ConnectionAction {
  type: "__connection";
  status: ConnectionStatus;
}

export type ReducerAction = WireMessage | ConnectionAction;
