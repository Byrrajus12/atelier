import type {
  ConnectionStatus,
  PlanIntent,
  ReducerAction,
  RunDoneMsg,
  RunStartMsg,
  Verdict,
} from "./types";

// One stroke-log row: the plan.done that proposed it, joined with the
// verify.done that judged it (verify arrives after plan for the same
// iteration; `verdict` is null for the brief window between the two).
export interface StrokeRow {
  iteration: number;
  cell: [number, number];
  intent: PlanIntent;
  verdict: Verdict | null;
}

export interface ErrorPoint {
  iteration: number;
  global_error: number | null;
}

export interface CurrentState {
  iteration: number;
  global_error: number | null;
  status: string;
  converged: boolean;
}

export interface State {
  connection: ConnectionStatus;
  config: RunStartMsg | null;
  errorSeries: ErrorPoint[];
  strokeLog: StrokeRow[];
  current: CurrentState | null;
  terminal: RunDoneMsg | null;
}

export const initialState: State = {
  connection: "connecting",
  config: null,
  errorSeries: [],
  strokeLog: [],
  current: null,
  terminal: null,
};

function upsertErrorPoint(series: ErrorPoint[], point: ErrorPoint): ErrorPoint[] {
  const idx = series.findIndex((p) => p.iteration === point.iteration);
  if (idx === -1) return [...series, point].sort((a, b) => a.iteration - b.iteration);
  const next = series.slice();
  next[idx] = point;
  return next;
}

function upsertStrokeRow(
  log: StrokeRow[],
  iteration: number,
  patch: Partial<Pick<StrokeRow, "cell" | "intent" | "verdict">>,
): StrokeRow[] {
  const idx = log.findIndex((r) => r.iteration === iteration);
  if (idx === -1) {
    return [
      ...log,
      {
        iteration,
        cell: patch.cell ?? [0, 0],
        intent: patch.intent ?? { cell: [0, 0], box: [0, 0, 0, 0], color: [0, 0, 0], error: null, size: null },
        verdict: patch.verdict ?? null,
      },
    ].sort((a, b) => a.iteration - b.iteration);
  }
  const next = log.slice();
  next[idx] = { ...next[idx], ...patch };
  return next;
}

/** The pure heart of the dashboard: one reducer, folding every wire message
 * (plus the hook's local connection-status action) into the render state. */
export function reduce(state: State, action: ReducerAction): State {
  switch (action.type) {
    case "__connection":
      return { ...state, connection: action.status };

    case "run.start":
      // A fresh run: drop the previous run's series/log/terminal state so two
      // runs never concatenate on one chart if the dashboard stays open
      // across a restart. Connection status isn't run-scoped, so it survives.
      return { ...initialState, connection: state.connection, config: action };

    case "observe.done":
      return state; // heatmap/region_error are per-frame, not part of this dashboard's state

    case "plan.done":
      return {
        ...state,
        strokeLog: upsertStrokeRow(state.strokeLog, action.iteration, {
          cell: action.intent.cell,
          intent: action.intent,
        }),
      };

    case "execute.done":
      return state; // stroke_count isn't surfaced in this dashboard's components

    case "verify.done":
      return {
        ...state,
        strokeLog: upsertStrokeRow(state.strokeLog, action.iteration, {
          cell: action.verdict.cell,
          verdict: action.verdict,
        }),
      };

    case "state.update":
      return {
        ...state,
        current: {
          iteration: action.iteration,
          global_error: action.global_error,
          status: action.status,
          converged: action.converged,
        },
        errorSeries: upsertErrorPoint(state.errorSeries, {
          iteration: action.iteration,
          global_error: action.global_error,
        }),
      };

    case "run.done":
      return { ...state, terminal: action };

    default:
      return state;
  }
}
