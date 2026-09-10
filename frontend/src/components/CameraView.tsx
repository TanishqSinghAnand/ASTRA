import type { BrowserCameraStatus } from "@/lib/useBrowserCameraUpload";
import type { DwellProgress } from "@/lib/types";

interface CameraViewProps {
  frameImage: string | null;
  cameraRunning: boolean;
  cameraSource: string;
  fps: number;
  lastAction: string | null;
  lastConfidence: number;
  dwell: DwellProgress | null;
  browserCameraStatus: BrowserCameraStatus;
}

const BROWSER_CAMERA_MESSAGES: Record<Exclude<BrowserCameraStatus, "idle" | "streaming">, string> = {
  requesting: "Requesting camera access — check for a browser permission prompt.",
  denied: "Camera access was denied. Allow camera permission for this site and press Start again.",
  unavailable: "No camera available in this browser (needs HTTPS and a device with a camera).",
};

export function CameraView({
  frameImage,
  cameraRunning,
  cameraSource,
  fps,
  lastAction,
  lastConfidence,
  dwell,
  browserCameraStatus,
}: CameraViewProps) {
  const isBrowserSource = cameraSource === "browser";
  const browserIssue =
    isBrowserSource && browserCameraStatus !== "idle" && browserCameraStatus !== "streaming"
      ? BROWSER_CAMERA_MESSAGES[browserCameraStatus]
      : null;

  return (
    <div className="flex flex-col overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-panel)]">
      <div className="relative aspect-video w-full bg-black">
        {frameImage ? (
          // Live annotated JPEG pushed over the WS `frame` message — plain
          // <img> is correct here (a base64 data: URI, not an optimizable
          // static asset next/image could cache).
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={frameImage}
            alt="Live camera feed"
            className="h-full w-full object-contain"
          />
        ) : (
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 px-6 text-center text-[var(--color-text-faint)]">
            <span className="font-mono text-sm">
              {browserIssue ?? (cameraRunning ? "WAITING FOR FIRST FRAME..." : "CAMERA IDLE")}
            </span>
            {isBrowserSource && !browserIssue && cameraRunning && (
              <span className="font-mono text-xs">
                Using this browser&apos;s own camera — allow access if prompted.
              </span>
            )}
          </div>
        )}

        <div className="absolute left-3 top-3 flex items-center gap-2">
          <span
            className={`h-2 w-2 rounded-full ${cameraRunning ? "bg-[var(--color-red)] animate-pulse-dot" : "bg-[var(--color-text-faint)]"}`}
          />
          <span className="font-mono text-xs uppercase tracking-wider text-[var(--color-text-muted)]">
            {cameraSource}
          </span>
        </div>

        <div className="absolute right-3 top-3 font-mono text-xs text-[var(--color-text-muted)]">
          {fps.toFixed(1)} fps
        </div>
      </div>

      <div className="flex items-center justify-between border-t border-[var(--color-border)] px-4 py-2">
        <span className="font-mono text-xs text-[var(--color-text-muted)]">
          DETECTED ACTION
        </span>
        <span className="font-mono text-xs">
          {lastAction ? (
            <>
              <span className="text-[var(--color-cyan)]">{lastAction}</span>
              <span className="ml-2 text-[var(--color-text-faint)]">
                {(lastConfidence * 100).toFixed(0)}%
              </span>
            </>
          ) : (
            <span className="text-[var(--color-text-faint)]">observing...</span>
          )}
        </span>
      </div>

      {/* Live dwell-timer progress (backend/perception/interaction.py:
          PICK needs sustained hand contact, PLACE needs sustained zone
          occupancy — both timed, not instant) — shown whenever any
          tracked object currently has one running, so "why hasn't this
          registered yet" has a visible answer instead of just silence. */}
      {dwell && (
        <div className="flex items-center gap-3 border-t border-[var(--color-border)] px-4 py-2">
          <span className="font-mono text-xs uppercase tracking-wider text-[var(--color-text-muted)]">
            {dwell.object}
          </span>
          <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-border)]">
            <div
              className="h-full rounded-full bg-[var(--color-cyan)] transition-[width] duration-150 ease-linear"
              style={{ width: `${Math.min(100, (dwell.elapsed / dwell.required) * 100)}%` }}
            />
          </div>
          <span className="font-mono text-xs text-[var(--color-text-faint)]">
            {dwell.elapsed.toFixed(1)}s / {dwell.required.toFixed(1)}s
          </span>
        </div>
      )}
    </div>
  );
}
