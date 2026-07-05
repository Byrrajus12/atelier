import { useMemo } from "react";
import type { ErrorPoint } from "../reduce";
import "./ErrorCurve.css";

interface ErrorCurveProps {
  series: ErrorPoint[];
  errorThreshold: number | null;
}

const WIDTH = 960;
const HEIGHT = 320;
const PAD_X = 32;
const PAD_TOP = 20;
const PAD_BOTTOM = 32;

const GRID_LINES = 4;

export function ErrorCurve({ series, errorThreshold }: ErrorCurveProps) {
  const { linePath, areaPath, last, xOf, yOf, maxError, gridValues } = useMemo(() => {
    const plottable = series.filter(
      (p): p is { iteration: number; global_error: number } =>
        p.global_error !== null && Number.isFinite(p.global_error),
    );

    const maxIteration = Math.max(1, ...plottable.map((p) => p.iteration));
    const maxErrorRaw = Math.max(0.001, errorThreshold ?? 0, ...plottable.map((p) => p.global_error));
    const maxError = maxErrorRaw * 1.1; // headroom so the peak isn't clipped at the top edge

    const innerW = WIDTH - PAD_X * 2;
    const innerH = HEIGHT - PAD_TOP - PAD_BOTTOM;

    const xOf = (iteration: number) => PAD_X + (iteration / maxIteration) * innerW;
    // Lower error at the bottom, higher at the top: the trace visibly descends.
    const yOf = (error: number) => PAD_TOP + (1 - error / maxError) * innerH;

    let linePath = "";
    let started = false;
    for (const p of plottable) {
      const cmd = started ? "L" : "M";
      linePath += `${cmd}${xOf(p.iteration).toFixed(2)},${yOf(p.global_error).toFixed(2)} `;
      started = true;
    }

    let areaPath = "";
    if (plottable.length > 0) {
      const baseline = PAD_TOP + innerH;
      areaPath = `M${xOf(plottable[0].iteration).toFixed(2)},${baseline} `;
      for (const p of plottable) {
        areaPath += `L${xOf(p.iteration).toFixed(2)},${yOf(p.global_error).toFixed(2)} `;
      }
      areaPath += `L${xOf(plottable[plottable.length - 1].iteration).toFixed(2)},${baseline} Z`;
    }

    const gridValues = Array.from({ length: GRID_LINES + 1 }, (_, i) => (maxError / GRID_LINES) * i);

    return {
      linePath,
      areaPath,
      last: plottable[plottable.length - 1] ?? null,
      xOf,
      yOf,
      maxError,
      gridValues,
    };
  }, [series, errorThreshold]);

  const empty = series.length === 0;

  return (
    <div className="error-curve">
      <div className="error-curve__label">global error / iteration</div>
      {empty ? (
        <div className="error-curve__placeholder">waiting for data…</div>
      ) : (
        <svg
          className="error-curve__svg"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          preserveAspectRatio="none"
          role="img"
          aria-label={`Global error curve, currently ${last ? last.global_error.toFixed(4) : "unknown"}`}
        >
          {gridValues.map((v) => (
            <line
              key={v}
              className="error-curve__grid"
              x1={PAD_X}
              x2={WIDTH - PAD_X}
              y1={yOf(v)}
              y2={yOf(v)}
            />
          ))}

          {errorThreshold !== null && errorThreshold <= maxError && (
            <line
              className="error-curve__threshold"
              x1={PAD_X}
              x2={WIDTH - PAD_X}
              y1={yOf(errorThreshold)}
              y2={yOf(errorThreshold)}
            />
          )}

          {areaPath && <path className="error-curve__area" d={areaPath} />}
          {linePath && <path className="error-curve__line" d={linePath} />}

          {last && (
            <circle
              key={last.iteration}
              className="error-curve__dot"
              cx={xOf(last.iteration)}
              cy={yOf(last.global_error)}
              r={4.5}
            />
          )}
        </svg>
      )}
    </div>
  );
}
