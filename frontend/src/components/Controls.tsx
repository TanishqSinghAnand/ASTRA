import type { RunStatus } from "@/lib/types";

interface ControlsProps {
  status: RunStatus;
  busy: boolean;
  onStart: () => void;
  onStop: () => void;
  onReset: () => void;
}

export function Controls({ status, busy, onStart, onStop, onReset }: ControlsProps) {
  const running = status === "RUNNING";

  return (
    <div className="flex gap-2">
      <button
        onClick={onStart}
        disabled={busy || running}
        className="flex-1 rounded-md border border-[var(--color-green)]/40 bg-[var(--color-green)]/10 px-3 py-2 font-mono text-xs font-semibold tracking-wide text-[var(--color-green)] transition-colors hover:bg-[var(--color-green)]/20 disabled:cursor-not-allowed disabled:opacity-40"
      >
        START
      </button>
      <button
        onClick={onStop}
        disabled={busy || !running}
        className="flex-1 rounded-md border border-[var(--color-red)]/40 bg-[var(--color-red)]/10 px-3 py-2 font-mono text-xs font-semibold tracking-wide text-[var(--color-red)] transition-colors hover:bg-[var(--color-red)]/20 disabled:cursor-not-allowed disabled:opacity-40"
      >
        STOP
      </button>
      <button
        onClick={onReset}
        disabled={busy}
        className="flex-1 rounded-md border border-[var(--color-border)] bg-[var(--color-panel-raised)] px-3 py-2 font-mono text-xs font-semibold tracking-wide text-[var(--color-text-muted)] transition-colors hover:bg-[var(--color-border)]/40 disabled:cursor-not-allowed disabled:opacity-40"
      >
        RESET
      </button>
    </div>
  );
}
