import type { CurrentState } from "../reduce";
import type { ConnectionStatus, RunDoneMsg, RunStartMsg } from "../types";
import {
  REASON_BUDGET,
  REASON_CANVAS_LOST,
  REASON_CONVERGED,
  REASON_STALLED,
  REASON_STALLED_NO_PROGRESS,
} from "../types";
import "./RunHeader.css";

interface RunHeaderProps {
  connection: ConnectionStatus;
  config: RunStartMsg | null;
  current: CurrentState | null;
  terminal: RunDoneMsg | null;
}

const CONNECTION_LABEL: Record<ConnectionStatus, string> = {
  connecting: "connecting",
  open: "connected",
  reconnecting: "reconnecting",
  closed: "disconnected",
};

function connectionClass(status: ConnectionStatus): string {
  if (status === "open") return "signal";
  if (status === "closed") return "danger";
  return "muted";
}

function terminalClass(reason: string): string {
  if (reason === REASON_CONVERGED) return "signal";
  if (reason === REASON_BUDGET || reason === REASON_STALLED || reason === REASON_STALLED_NO_PROGRESS) {
    return "warning";
  }
  if (reason === REASON_CANVAS_LOST) return "danger";
  return "muted";
}

function formatError(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(4);
}

export function RunHeader({ connection, config, current, terminal }: RunHeaderProps) {
  const iteration = current?.iteration ?? 0;
  const maxIterations = config?.max_iterations ?? null;

  return (
    <header className="run-header">
      <div className="run-header__row">
        <div className={`run-header__connection run-header__connection--${connectionClass(connection)}`}>
          <span className="run-header__dot" aria-hidden="true" />
          <span>{CONNECTION_LABEL[connection]}</span>
        </div>

        {terminal && (
          <div className={`run-header__badge run-header__badge--${terminalClass(terminal.reason)}`}>
            {terminal.reason}
          </div>
        )}
      </div>

      <div className="run-header__row run-header__row--main">
        <div className="run-header__metric">
          <div className="run-header__metric-label">global error</div>
          <div className="run-header__metric-value">{formatError(current?.global_error)}</div>
        </div>

        <div className="run-header__iteration">
          <span className="run-header__iteration-current">{iteration}</span>
          {maxIterations !== null && (
            <>
              <span className="run-header__iteration-sep">/</span>
              <span className="run-header__iteration-max">{maxIterations}</span>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
