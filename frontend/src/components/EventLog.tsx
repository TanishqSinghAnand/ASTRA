import type { SequenceEvent } from "@/lib/types";
import { ERROR_STATUSES } from "@/lib/types";

interface EventLogProps {
  events: SequenceEvent[];
}

function statusColor(status: SequenceEvent["status"]): string {
  if (status === "CORRECT" || status === "COMPLETE") return "var(--color-green)";
  if (status === "RECOVERED") return "var(--color-cyan)";
  if (status === "LOW_CONFIDENCE") return "var(--color-amber)";
  if (ERROR_STATUSES.includes(status)) return "var(--color-red)";
  return "var(--color-text-muted)";
}

export function EventLog({ events }: EventLogProps) {
  const ordered = [...events].reverse(); // newest first

  return (
    <div className="astra-scroll flex h-full flex-col gap-1 overflow-y-auto p-3">
      {ordered.length === 0 && (
        <p className="p-2 font-mono text-xs text-[var(--color-text-faint)]">
          No events yet — start the experiment to begin the log.
        </p>
      )}
      {ordered.map((event, i) => (
        <div
          key={`${event.timestamp}-${i}`}
          className="flex items-center gap-3 rounded border border-[var(--color-border)]/60 px-3 py-1.5 font-mono text-xs"
        >
          <span
            className="w-1.5 shrink-0 self-stretch rounded-full"
            style={{ background: statusColor(event.status) }}
          />
          <span className="w-16 shrink-0 text-[var(--color-text-faint)]">
            {new Date(event.timestamp * 1000).toLocaleTimeString([], {
              hour12: false,
            })}
          </span>
          <span
            className="w-32 shrink-0 font-semibold"
            style={{ color: statusColor(event.status) }}
          >
            {event.status.replaceAll("_", " ")}
          </span>
          <span className="truncate text-[var(--color-text-muted)]">
            step {event.step} · {event.detected ?? "—"}
          </span>
        </div>
      ))}
    </div>
  );
}
