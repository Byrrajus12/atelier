import { CanvasView } from "./components/CanvasView";
import { ErrorCurve } from "./components/ErrorCurve";
import { RunHeader } from "./components/RunHeader";
import { StrokeLog } from "./components/StrokeLog";
import { useEventStream } from "./useEventStream";
import "./App.css";

const WS_URL = import.meta.env.VITE_ATELIER_WS_URL ?? "ws://127.0.0.1:8765";

function App() {
  const state = useEventStream(WS_URL);

  return (
    <div className="app">
      <RunHeader
        connection={state.connection}
        config={state.config}
        current={state.current}
        terminal={state.terminal}
      />
      <div className="app__instruments">
        <ErrorCurve series={state.errorSeries} errorThreshold={state.config?.error_threshold ?? null} />
        <CanvasView frame={state.frame} canvasSize={state.config?.canvas_size ?? null} />
      </div>
      <StrokeLog rows={state.strokeLog} />
    </div>
  );
}

export default App;
