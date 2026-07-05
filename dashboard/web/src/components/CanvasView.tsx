import type { FrameState } from "../reduce";
import "./CanvasView.css";

interface CanvasViewProps {
  frame: FrameState | null;
  canvasSize: [number, number] | null;
}

// Frames arrive periodically (publisher cadence), not every iteration, so this panel
// deliberately labels which iteration the picture actually reflects rather than
// implying it is current-as-of-now.
export function CanvasView({ frame, canvasSize }: CanvasViewProps) {
  // Fixed aspect ratio even before any frame lands (from run.start's canvas_size, else
  // a square placeholder), so the panel doesn't pop/resize when the first frame arrives.
  const aspectRatio = frame
    ? `${frame.width} / ${frame.height}`
    : canvasSize
      ? `${canvasSize[0]} / ${canvasSize[1]}`
      : "1 / 1";

  return (
    <div className="canvas-view">
      <div className="canvas-view__label">
        canvas
        {frame && (
          <>
            {" — frame @ iter "}
            <span className="canvas-view__label-num">{frame.iteration}</span>
          </>
        )}
      </div>
      <div className="canvas-view__frame" style={{ aspectRatio }}>
        {frame ? (
          <img
            className="canvas-view__image"
            src={frame.image}
            width={frame.width}
            height={frame.height}
            alt={`Captured canvas at iteration ${frame.iteration}`}
          />
        ) : (
          <div className="canvas-view__placeholder">waiting for frame…</div>
        )}
      </div>
    </div>
  );
}
