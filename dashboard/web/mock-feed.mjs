// Dev-only test harness: a Node websocket server on :8765 that replays a
// canned converging run in the EXACT wire shape dashboard/publisher.py
// produces from core/events.py (see event_to_message there). Lets the
// dashboard be built/verified without the real orchestrator or a browser
// canvas running. Not shipped, not imported by the app — start with
// `npm run mock-feed`.
import { WebSocketServer } from "ws";

const PORT = 8765;
const STEP_MS = Number(process.env.MOCK_FEED_STEP_MS ?? 250);
const LOOP_PAUSE_MS = Number(process.env.MOCK_FEED_LOOP_PAUSE_MS ?? 3000);

const GRID_N = 6;
const MAX_ITERATIONS = 60;
const ERROR_THRESHOLD = 0.02;

// A hand-picked descending curve landing near the real M7.4 live-run numbers
// (0.1717 -> 0.0186), one plan/execute/verify per point, one deliberate
// reject along the way so the log shows both outcomes.
const ITERATIONS = [
  { cell: [1, 1], global: 0.1717, accepted: true },
  { cell: [4, 1], global: 0.152, accepted: true },
  { cell: [1, 4], global: 0.129, accepted: true },
  { cell: [1, 1], global: 0.111, accepted: true },
  { cell: [4, 1], global: 0.098, accepted: true },
  { cell: [2, 2], global: 0.093, accepted: false },
  { cell: [1, 4], global: 0.071, accepted: true },
  { cell: [1, 1], global: 0.056, accepted: true },
  { cell: [4, 1], global: 0.041, accepted: true },
  { cell: [3, 3], global: 0.038, accepted: false },
  { cell: [1, 4], global: 0.027, accepted: true },
  { cell: [1, 1], global: 0.0186, accepted: true },
];

const COLORS = { "1,1": [255, 0, 0], "4,1": [0, 0, 255], "1,4": [17, 17, 17] };

function colorFor(cell) {
  return COLORS[`${cell[0]},${cell[1]}`] ?? [128, 128, 128];
}

// Two precomputed 120x120 JPEGs (blank + fully-painted, matching the ITERATIONS
// cells/colors above) so the mock feed can exercise CanvasView without a real
// orchestrator/canvas. Mirrors what dashboard/publisher.py sends: a data URL, sampled
// periodically (here every FRAME_CADENCE iterations), with the final frame always sent
// regardless of cadence.
const FRAME_CADENCE = 3;
const BLANK_FRAME =
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAB4AHgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD2aiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q==";
const PAINTED_FRAME =
  "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAoHBwgHBgoICAgLCgoLDhgQDg0NDh0VFhEYIx8lJCIfIiEmKzcvJik0KSEiMEExNDk7Pj4+JS5ESUM8SDc9Pjv/2wBDAQoLCw4NDhwQEBw7KCIoOzs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozs7Ozv/wAARCAB4AHgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD2aiiigAooooAKKKKACiiigAooooAKKKKACiiigAooooAKKKKAOb8XeLv+EV+yf6D9q+07/wDlrs27dv8AsnP3v0rm/wDhbv8A1A//ACb/APsKPi7/AMwj/tt/7JXm9eZiMRUhUcYvQ+5ynKcFiMFCrVheTv1fdroz0j/hbv8A1A//ACb/APsK9Ir5vr6QrbCVZ1L8zPMz/AYfCez9hG1731b2t3YUUUV2nzIUUUUAeSf8L1/6lv8A8nv/ALXR/wAL1/6lv/ye/wDtdeS0V+mf2Dl3/Pv8Zf5nF7Wfc+h/Anjv/hNft3/Et+xfY/L/AOW/mb927/ZGMbf1rrq8k+BX/Md/7d//AGpXrdfDZtQp4fGzpUlaKt+SfU6qbbimwooorzCwooooA83+Lv8AzCP+23/sleb19IUVxVcJ7Sblc+mwGf8A1TDxoezva+t7btvsfN9fSFFFaUKHsr63ucWa5r/aHJ7nLy3633t5LsFFFFdJ4wUUUUAfJVFfWtFfZ/61f9Of/Jv/ALU5/YeZ5J8Cv+Y7/wBu/wD7Ur1uiivmcfivreIlXta9tN9kkbRjyqwUUUVxFBRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAeP8Ax7/5gP8A28f+0q8gr1/49/8AMB/7eP8A2lXkFABX1/XyBX1/QAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQBx/j7wD/wAJx9g/4mf2H7H5n/LDzN+/b/tDGNv61x//AAoT/qZv/JH/AO2V7BRQB4//AMKE/wCpm/8AJH/7ZXsFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQAUUUUAFFFFABRRRQB//Z";

function frameCapturedMsg(iteration, painted) {
  return {
    type: "frame.captured",
    iteration,
    width: 120,
    height: 120,
    image: `data:image/jpeg;base64,${painted ? PAINTED_FRAME : BLANK_FRAME}`,
  };
}

function cellBox(cell) {
  const size = 600 / GRID_N;
  const [i, j] = cell;
  return [Math.round(i * size), Math.round(j * size), Math.round((i + 1) * size), Math.round((j + 1) * size)];
}

function runStartMsg() {
  return {
    type: "run.start",
    canvas_size: [600, 600],
    grid_n: GRID_N,
    max_iterations: MAX_ITERATIONS,
    max_region_failures: 3,
    error_threshold: ERROR_THRESHOLD,
    improvement_threshold: 0.001,
    reversible: true,
    has_undo: false,
    stroke_cost: 1.0,
  };
}

function stateUpdateMsg(iteration, global_error, status, converged) {
  return { type: "state.update", iteration, global_error, status, converged };
}

function planDoneMsg(iteration, cell, error) {
  return {
    type: "plan.done",
    iteration,
    intent: { cell, box: cellBox(cell), color: colorFor(cell), error, size: 100 },
  };
}

function executeDoneMsg(iteration, cell) {
  return { type: "execute.done", iteration, cell, stroke_count: 3 };
}

function verifyDoneMsg(iteration, cell, before, after, accepted) {
  const regionBefore = accepted ? 0.35 : 0.09;
  const regionAfter = accepted ? 0.04 : 0.085;
  return {
    type: "verify.done",
    iteration,
    verdict: {
      accepted,
      cell,
      region_before: regionBefore,
      region_after: regionAfter,
      region_delta: regionAfter - regionBefore,
      global_before: before,
      global_after: after,
      global_delta: after - before,
    },
  };
}

function runDoneMsg(iteration, global_error) {
  return { type: "run.done", iteration, global_error, reason: "converged", converged: true };
}

const wss = new WebSocketServer({ port: PORT });
const clients = new Set();
let lastRunStart = null;
let lastState = null;
let lastFrame = null;

wss.on("connection", (ws) => {
  clients.add(ws);
  console.log(`[mock-feed] client connected (${clients.size} total)`);
  // Mirror publisher.py's bootstrap: replay the last run.start + state.update + frame
  // to a late joiner so it renders mid-run without waiting for the next tick.
  if (lastRunStart) ws.send(lastRunStart);
  if (lastState) ws.send(lastState);
  if (lastFrame) ws.send(lastFrame);
  ws.on("close", () => {
    clients.delete(ws);
    console.log(`[mock-feed] client disconnected (${clients.size} total)`);
  });
});

function broadcast(msg) {
  const data = JSON.stringify(msg);
  if (msg.type === "run.start") lastRunStart = data;
  if (msg.type === "state.update") lastState = data;
  if (msg.type === "frame.captured") lastFrame = data;
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) ws.send(data);
  }
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function runOnce() {
  console.log("[mock-feed] run.start");
  broadcast(runStartMsg());
  await sleep(STEP_MS);

  broadcast(stateUpdateMsg(0, 0.1717, "running", false));
  broadcast(frameCapturedMsg(0, false));
  await sleep(STEP_MS);

  let before = 0.1717;
  for (let idx = 0; idx < ITERATIONS.length; idx++) {
    const iteration = idx + 1;
    const { cell, global: after, accepted } = ITERATIONS[idx];

    broadcast(planDoneMsg(iteration, cell, accepted ? 0.3 : 0.09));
    await sleep(STEP_MS);

    broadcast(executeDoneMsg(iteration, cell));
    await sleep(STEP_MS);

    broadcast(verifyDoneMsg(iteration, cell, before, after, accepted));
    await sleep(STEP_MS);

    broadcast(stateUpdateMsg(iteration, after, "running", false));
    // Mirrors the publisher's cadence sampling: only every FRAME_CADENCEth
    // iteration gets a frame, and the (canned) canvas is "painted" from
    // iteration 3 on, once the three distinct cells have all landed once.
    if (iteration % FRAME_CADENCE === 0) {
      broadcast(frameCapturedMsg(iteration, iteration >= 3));
    }
    await sleep(STEP_MS);

    before = after;
  }

  const finalIteration = ITERATIONS.length;
  // Mirrors the publisher's force-publish-on-run.done rule: the finished canvas is
  // always sent, even if the terminal iteration falls off-cadence.
  if (finalIteration % FRAME_CADENCE !== 0) {
    broadcast(frameCapturedMsg(finalIteration, true));
  }
  const finalError = ITERATIONS[ITERATIONS.length - 1].global;
  broadcast(stateUpdateMsg(finalIteration, finalError, "done", true));
  broadcast(runDoneMsg(finalIteration, finalError));
  console.log(`[mock-feed] run.done reason=converged global=${finalError}`);
}

async function loop() {
  for (;;) {
    await runOnce();
    await sleep(LOOP_PAUSE_MS);
  }
}

console.log(`[mock-feed] listening on ws://127.0.0.1:${PORT}`);
loop();
