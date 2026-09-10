import type { DetectorMode, ExperimentDefinition, ModeResponse, SequenceEvent, StatusDict } from "./types";

// Configurable per the project's "every tunable lives in config, not
// scattered through code" rule (config/config.yaml on the backend side) —
// set NEXT_PUBLIC_API_BASE / NEXT_PUBLIC_WS_URL for anything other than a
// backend running locally on the default port.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/live";
// Camera-frame *upload* socket — only used when the backend's
// camera.source is "browser" (a cloud deployment with no local capture
// device of its own; see backend/websocket/ingest.py). Derived from
// WS_URL rather than a separate env var so the two sockets can't
// accidentally point at different hosts.
export const WS_INGEST_URL = WS_URL.replace(/\/ws\/live$/, "/ws/ingest");

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

async function postJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { method: "POST" });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getStatus: () => getJson<StatusDict>("/api/status"),
  getExperiment: () => getJson<ExperimentDefinition>("/api/experiment"),
  getMode: () => getJson<{ mode: DetectorMode; available: DetectorMode[] }>("/api/mode"),
  getLog: () => getJson<{ events: SequenceEvent[] }>("/api/experiment/log"),
  start: () => postJson<StatusDict>("/api/experiment/start"),
  stop: () => postJson<StatusDict>("/api/experiment/stop"),
  reset: () => postJson<StatusDict>("/api/experiment/reset"),
  // Tears down the running InferenceService and rebuilds one from the
  // other detector's config.yaml (backend/api/routes.py) — response
  // carries the *new* experiment (different object ids/steps, not a
  // status refresh) and a fresh IDLE status.
  setMode: (mode: DetectorMode) => postJson<ModeResponse>(`/api/mode/${mode}`),
};
