interface HeaderProps {
  connected: boolean;
  device: string;
}

export function Header({ connected, device }: HeaderProps) {
  return (
    <header className="flex items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-panel)] px-6 py-4">
      <div className="flex items-baseline gap-3">
        <span className="font-mono text-xl font-bold tracking-[0.2em] text-[var(--color-cyan)]">
          ASTRA
        </span>
        <span className="hidden text-xs text-[var(--color-text-muted)] sm:inline">
          AI Human Activity Recognition &middot; on-board BAS experiments
        </span>
      </div>

      <div className="flex items-center gap-3">
        <span className="flex items-center gap-1.5 rounded-full border border-[var(--color-green)]/30 bg-[var(--color-green)]/10 px-3 py-1 text-xs font-medium tracking-wide text-[var(--color-green)]">
          <span
            className={`h-1.5 w-1.5 rounded-full bg-[var(--color-green)] ${connected ? "animate-pulse-dot" : ""}`}
          />
          OFFLINE AI
        </span>
        <span
          className={`flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium tracking-wide ${
            connected
              ? "border-[var(--color-cyan)]/30 bg-[var(--color-cyan)]/10 text-[var(--color-cyan)]"
              : "border-[var(--color-red)]/30 bg-[var(--color-red)]/10 text-[var(--color-red)]"
          }`}
        >
          <span
            className={`h-1.5 w-1.5 rounded-full ${connected ? "bg-[var(--color-cyan)]" : "bg-[var(--color-red)]"}`}
          />
          {connected ? "LIVE" : "DISCONNECTED"}
        </span>
        <span className="hidden font-mono text-xs text-[var(--color-text-faint)] md:inline">
          {device}
        </span>
      </div>
    </header>
  );
}
