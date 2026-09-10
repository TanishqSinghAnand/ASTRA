"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import { ERROR_STATUSES } from "@/lib/types";
import type { DetectorMode, DwellProgress, ExperimentDefinition, LiveMessage, SequenceEvent, StatusDict } from "@/lib/types";
import { useLiveSocket } from "@/lib/useLiveSocket";
import { useBrowserCameraUpload } from "@/lib/useBrowserCameraUpload";
import { Header } from "./Header";
import { ModeTabs } from "./ModeTabs";
import { CameraView } from "./CameraView";
import { StepChecklist } from "./StepChecklist";
import { StatusPanel, type ErrorBannerState } from "./StatusPanel";
import { Controls } from "./Controls";
import { EventLog } from "./EventLog";

const STATUS_POLL_MS = 2000;

export function Dashboard() {
  const [status, setStatus] = useState<StatusDict | null>(null);
  const [experiment, setExperiment] = useState<ExperimentDefinition | null>(null);
  const [events, setEvents] = useState<SequenceEvent[]>([]);
  const [frameImage, setFrameImage] = useState<string | null>(null);
  const [lastAction, setLastAction] = useState<string | null>(null);
  const [lastConfidence, setLastConfidence] = useState(0);
  const [dwell, setDwell] = useState<DwellProgress | null>(null);
  const [bannerState, setBannerState] = useState<ErrorBannerState>("none");
  const [errorEvent, setErrorEvent] = useState<SequenceEvent | null>(null);
  const [busy, setBusy] = useState(false);
  const [mode, setMode] = useState<DetectorMode>("color");
  const [availableModes, setAvailableModes] = useState<DetectorMode[]>(["color"]);

  const handleMessage = useCallback((msg: LiveMessage) => {
    switch (msg.type) {
      case "status": {
        setStatus(msg);
        break;
      }
      case "perception": {
        setLastAction(msg.action);
        setLastConfidence(msg.confidence);
        setDwell(msg.dwell);
        setStatus((prev) =>
          prev ? { ...prev, current_step: msg.step, fps: msg.fps } : prev,
        );
        break;
      }
      case "sequence_event": {
        setEvents((prev) => [...prev, msg]);
        if (ERROR_STATUSES.includes(msg.status)) {
          setBannerState("error");
          setErrorEvent(msg);
        } else if (msg.status === "RECOVERED") {
          setBannerState("recovered");
          setErrorEvent(null);
        } else {
          setBannerState("none");
          setErrorEvent(null);
        }
        break;
      }
      case "frame": {
        setFrameImage(msg.image);
        break;
      }
    }
  }, []);

  const { connected } = useLiveSocket(handleMessage);

  // Only relevant when the backend has no local camera of its own
  // (camera.source == "browser", set for a cloud deployment) — otherwise
  // this is a no-op and getUserMedia is never even requested.
  const { status: browserCameraStatus } = useBrowserCameraUpload(
    status?.camera_source === "browser" && status?.status === "RUNNING",
  );

  // Initial REST seed — the experiment definition + past event log (per
  // spec: "seeded from GET /api/experiment/log and appended live from
  // sequence_event WS messages") — plus a status baseline in case the WS
  // greet message races with this fetch.
  useEffect(() => {
    api.getExperiment().then(setExperiment).catch(() => {});
    api.getLog().then((r) => setEvents(r.events)).catch(() => {});
    api.getStatus().then(setStatus).catch(() => {});
    api.getMode().then((r) => {
      setMode(r.mode);
      setAvailableModes(r.available);
    }).catch(() => {});
  }, []);

  // Periodic fallback poll: the WS only pushes a full StatusDict once, on
  // connect. Fields like camera_running/finished/error never otherwise
  // update over the socket, so this keeps them honest even across long
  // idle stretches or a missed WS message.
  useEffect(() => {
    const id = setInterval(() => {
      api
        .getStatus()
        .then((fresh) => setStatus((prev) => (prev ? { ...prev, ...fresh } : fresh)))
        .catch(() => {});
    }, STATUS_POLL_MS);
    return () => clearInterval(id);
  }, []);

  const runControl = useCallback(
    async (action: () => Promise<StatusDict>, clearEvents: boolean) => {
      setBusy(true);
      try {
        const fresh = await action();
        setStatus(fresh);
        setBannerState("none");
        setErrorEvent(null);
        if (clearEvents) {
          setEvents([]);
          setFrameImage(null);
          setLastAction(null);
          setLastConfidence(0);
          setDwell(null);
        }
      } catch {
        // Surfaced via status.error on the next poll rather than a toast —
        // this is a single-operator local demo tool, not a multi-user app.
      } finally {
        setBusy(false);
      }
    },
    [],
  );

  const handleModeSelect = useCallback(
    async (next: DetectorMode) => {
      if (next === mode || busy) return;
      setBusy(true);
      try {
        // Backend tears down and rebuilds the whole InferenceService for
        // the other detector (api.ts's setMode comment) — a different
        // experiment entirely, not just a display filter, so every piece
        // of live/run state resets along with it.
        const { mode: confirmed, status: freshStatus, experiment: freshExperiment } = await api.setMode(next);
        setMode(confirmed);
        setStatus(freshStatus);
        setExperiment(freshExperiment);
        setEvents([]);
        setFrameImage(null);
        setLastAction(null);
        setLastConfidence(0);
        setDwell(null);
        setBannerState("none");
        setErrorEvent(null);
      } catch {
        // Same "surfaced on the next status poll" policy as runControl.
      } finally {
        setBusy(false);
      }
    },
    [mode, busy],
  );

  return (
    <div className="flex flex-1 flex-col">
      <Header connected={connected} device={status?.device ?? "CPU (MediaPipe)"} />
      <ModeTabs mode={mode} available={availableModes} busy={busy} onSelect={handleModeSelect} />

      <div className="flex flex-1 gap-4 overflow-hidden p-4">
        <div className="min-w-0 flex-[3]">
          <CameraView
            frameImage={frameImage}
            cameraRunning={status?.camera_running ?? false}
            cameraSource={status?.camera_source ?? "—"}
            fps={status?.fps ?? 0}
            lastAction={lastAction}
            lastConfidence={lastConfidence}
            dwell={dwell}
            browserCameraStatus={browserCameraStatus}
          />
        </div>

        <div className="flex min-w-0 flex-[2] flex-col gap-4 overflow-hidden">
          <div className="flex flex-col gap-2 overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-panel)] p-4">
            <h2 className="font-mono text-xs tracking-widest text-[var(--color-text-faint)]">
              PROCEDURE
            </h2>
            <div className="astra-scroll overflow-y-auto">
              <StepChecklist
                steps={experiment?.steps ?? []}
                currentStep={status?.current_step ?? 1}
                finished={status?.finished ?? false}
              />
            </div>
          </div>

          <StatusPanel
            currentStep={status?.current_step ?? 1}
            totalSteps={status?.total_steps ?? experiment?.steps.length ?? 0}
            currentInstruction={status?.current_instruction ?? null}
            finished={status?.finished ?? false}
            bannerState={bannerState}
            errorEvent={errorEvent}
          />

          <Controls
            status={status?.status ?? "IDLE"}
            busy={busy}
            onStart={() => runControl(api.start, true)}
            onStop={() => runControl(api.stop, false)}
            onReset={() => runControl(api.reset, true)}
          />
        </div>
      </div>

      <div className="h-56 shrink-0 border-t border-[var(--color-border)] bg-[var(--color-panel)]">
        <EventLog events={events} />
      </div>
    </div>
  );
}
