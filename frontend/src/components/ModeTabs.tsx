import type { DetectorMode } from "@/lib/types";

interface ModeTabsProps {
  mode: DetectorMode;
  busy: boolean;
  onSelect: (mode: DetectorMode) => void;
}

const TABS: { mode: DetectorMode; label: string; hint: string }[] = [
  { mode: "color", label: "COLOR DETECTION", hint: "blue / yellow boxes" },
  { mode: "yolo", label: "YOLO DETECTION", hint: "remote / bottle" },
];

// Switching tabs tears down and rebuilds the whole InferenceService on
// the backend (backend/api/routes.py's /api/mode — a different detector
// backend means different tracked classes and a different experiment
// entirely, not just a display filter), so this always implies stopping
// whatever's currently running. No confirmation dialog: this is a local,
// single-operator demo tool, not a multi-user app where that could
// surprise someone else.
export function ModeTabs({ mode, busy, onSelect }: ModeTabsProps) {
  return (
    <div className="flex gap-1 border-b border-[var(--color-border)] px-4 pt-3">
      {TABS.map((tab) => {
        const active = tab.mode === mode;
        return (
          <button
            key={tab.mode}
            type="button"
            disabled={busy}
            onClick={() => onSelect(tab.mode)}
            className={`flex flex-col items-start gap-0.5 rounded-t-md border border-b-0 px-4 py-2 font-mono text-xs tracking-wide transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
              active
                ? "border-[var(--color-border)] bg-[var(--color-panel)] text-[var(--color-cyan)]"
                : "border-transparent text-[var(--color-text-muted)] hover:text-[var(--color-text)]"
            }`}
          >
            <span className="font-semibold">{tab.label}</span>
            <span className="text-[10px] text-[var(--color-text-faint)]">{tab.hint}</span>
          </button>
        );
      })}
    </div>
  );
}
