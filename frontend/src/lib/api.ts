import type { ExperimentDefinition, SequenceEvent, StatusDict } from "./types";

// Configurable per the project's "every tunable lives in config, not
// scattered through code" rule (config/config.yaml on the backend side) —
// set NEXT_PUBLIC_API_BASE / NEXT_PUBLIC_WS_URL for anything other than a
// backend running locally on the default port.
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";
export const WS_URL =
  process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/live";

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
  getLog: () => getJson<{ events: SequenceEvent[] }>("/api/experiment/log"),
  start: () => postJson<StatusDict>("/api/experiment/start"),
  stop: () => postJson<StatusDict>("/api/experiment/stop"),
  reset: () => postJson<StatusDict>("/api/experiment/reset"),
};
