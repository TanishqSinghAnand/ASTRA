"use client";

import { useEffect, useRef, useState } from "react";
import { WS_URL } from "./api";
import type { LiveMessage } from "./types";

const RECONNECT_DELAY_MS = 2000;

/**
 * Owns the /ws/live connection and reconnects automatically (the backend
 * may not be up yet when the dashboard first loads, or the demo laptop's
 * camera/backend process could be restarted mid-session) — every incoming
 * message is handed to onMessage; this hook holds no derived state of its
 * own so there's exactly one place (Dashboard) that owns "what does the UI
 * currently show".
 */
export function useLiveSocket(onMessage: (msg: LiveMessage) => void) {
  const [connected, setConnected] = useState(false);
  const onMessageRef = useRef(onMessage);

  useEffect(() => {
    onMessageRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      socket = new WebSocket(WS_URL);

      socket.onopen = () => setConnected(true);

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data) as LiveMessage;
          onMessageRef.current(msg);
        } catch {
          // Malformed frame — ignore rather than crash the live view.
        }
      };

      socket.onclose = () => {
        setConnected(false);
        if (!cancelled) {
          reconnectTimer = setTimeout(connect, RECONNECT_DELAY_MS);
        }
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, []);

  return { connected };
}
