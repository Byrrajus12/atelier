import { useEffect, useReducer, useRef } from "react";
import { initialState, reduce, type State } from "./reduce";
import type { WireMessage } from "./types";

const INITIAL_BACKOFF_MS = 500;
const MAX_BACKOFF_MS = 8000;

/** Connects to the publisher's websocket, dispatches every message through
 * `reduce`, and reconnects with exponential backoff so the dashboard can be
 * opened before a run starts (or survive one restarting). Backoff resets on
 * any successful open. */
export function useEventStream(url: string): State {
  const [state, dispatch] = useReducer(reduce, initialState);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      dispatch({ type: "__connection", status: "connecting" });
      socket = new WebSocket(url);

      socket.onopen = () => {
        backoffRef.current = INITIAL_BACKOFF_MS;
        dispatch({ type: "__connection", status: "open" });
      };

      socket.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data) as WireMessage;
          dispatch(msg);
        } catch {
          // malformed frame: drop it, the stream is best-effort
        }
      };

      socket.onclose = () => {
        if (cancelled) return;
        dispatch({ type: "__connection", status: "reconnecting" });
        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, MAX_BACKOFF_MS);
        retryTimer = setTimeout(connect, delay);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onclose = null;
        socket.onerror = null;
        socket.close();
      }
      dispatch({ type: "__connection", status: "closed" });
    };
  }, [url]);

  return state;
}
