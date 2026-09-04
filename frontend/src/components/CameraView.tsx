interface CameraViewProps {
  frameImage: string | null;
  cameraRunning: boolean;
  cameraSource: string;
  fps: number;
  lastAction: string | null;
  lastConfidence: number;
}

export function CameraView({
  frameImage,
  cameraRunning,
  cameraSource,
  fps,
  lastAction,
  lastConfidence,
}: CameraViewProps) {
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
          <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-[var(--color-text-faint)]">
            <span className="font-mono text-sm">
              {cameraRunning ? "WAITING FOR FIRST FRAME..." : "CAMERA IDLE"}
            </span>
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
    </div>
  );
}
