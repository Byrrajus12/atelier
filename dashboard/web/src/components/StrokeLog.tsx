import type { StrokeRow } from "../reduce";
import "./StrokeLog.css";

interface StrokeLogProps {
  rows: StrokeRow[];
}

function formatSigned(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(4)}`;
}

function formatValue(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toFixed(4);
}

export function StrokeLog({ rows }: StrokeLogProps) {
  const ordered = [...rows].reverse(); // newest first: a live log is scanned top-down

  return (
    <div className="stroke-log">
      <div className="stroke-log__label">stroke log</div>
      <div className="stroke-log__header">
        <span>iter</span>
        <span>cell</span>
        <span>result</span>
        <span>region Δ</span>
        <span>global</span>
      </div>
      <div className="stroke-log__rows">
        {ordered.length === 0 && <div className="stroke-log__empty">no strokes yet</div>}
        {ordered.map((row) => {
          const pending = row.verdict === null;
          const accepted = row.verdict?.accepted ?? false;
          return (
            <div
              key={row.iteration}
              className={`stroke-log__row ${pending ? "stroke-log__row--pending" : ""}`}
            >
              <span className="stroke-log__iteration">{row.iteration}</span>
              <span className="stroke-log__cell">
                ({row.cell[0]}, {row.cell[1]})
              </span>
              <span
                className={`stroke-log__result stroke-log__result--${
                  pending ? "pending" : accepted ? "accept" : "reject"
                }`}
              >
                <span className="stroke-log__dot" aria-hidden="true" />
                {pending ? "pending" : accepted ? "accept" : "reject"}
              </span>
              <span className="stroke-log__delta">{formatSigned(row.verdict?.region_delta)}</span>
              <span className="stroke-log__global">{formatValue(row.verdict?.global_after)}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
