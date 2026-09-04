import type { SequenceEvent } from "@/lib/types";

export type ErrorBannerState = "none" | "error" | "recovered";

interface StatusPanelProps {
  currentStep: number;
  totalSteps: number;
  currentInstruction: string | null;
  finished: boolean;
  bannerState: ErrorBannerState;
  errorEvent: SequenceEvent | null;
}

export function StatusPanel({
  currentStep,
  totalSteps,
  currentInstruction,
  finished,
  bannerState,
  errorEvent,
}: StatusPanelProps) {
  if (bannerState === "error" && errorEvent) {
    return (
      <div className="rounded-lg border border-[var(--color-red)]/40 bg-[var(--color-red)]/10 p-4">
        <div className="mb-2 flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-[var(--color-red)]" />
          <span className="font-mono text-xs font-semibold tracking-widest text-[var(--color-red)]">
            {errorEvent.status.replaceAll("_", " ")}
          </span>
        </div>
        <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 font-mono text-xs">
          <dt className="text-[var(--color-text-faint)]">EXPECTED</dt>
          <dd className="text-[var(--color-text)]">{errorEvent.expected ?? "—"}</dd>
          <dt className="text-[var(--color-text-faint)]">DETECTED</dt>
          <dd className="text-[var(--color-text)]">{errorEvent.detected ?? "—"}</dd>
          <dt className="text-[var(--color-text-faint)]">CONFIDENCE</dt>
          <dd className="text-[var(--color-text)]">
            {(errorEvent.confidence * 100).toFixed(0)}%
          </dd>
        </dl>
        {errorEvent.explanation.length > 0 && (
          <ul className="mt-3 space-y-1 border-t border-[var(--color-red)]/20 pt-2 text-xs text-[var(--color-text-muted)]">
            {errorEvent.explanation.map((line, i) => (
              <li key={i}>{line}</li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  if (bannerState === "recovered") {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-[var(--color-green)]/40 bg-[var(--color-green)]/10 p-4">
        <span className="text-[var(--color-green)]">✓</span>
        <span className="font-mono text-xs font-semibold tracking-widest text-[var(--color-green)]">
          RECOVERY VERIFIED
        </span>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-[var(--color-border)] bg-[var(--color-panel-raised)] p-4">
      <div className="mb-1 font-mono text-xs text-[var(--color-text-faint)]">
        {finished ? "EXPERIMENT COMPLETE" : `STEP ${currentStep} / ${totalSteps}`}
      </div>
      <p className="text-sm leading-6 text-[var(--color-text)]">
        {finished
          ? "All steps verified — experiment closed out successfully."
          : (currentInstruction ?? "Waiting for experiment to start…")}
      </p>
    </div>
  );
}
