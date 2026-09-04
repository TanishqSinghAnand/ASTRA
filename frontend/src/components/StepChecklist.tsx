import type { ExperimentStep } from "@/lib/types";

interface StepChecklistProps {
  steps: ExperimentStep[];
  currentStep: number;
  finished: boolean;
}

export function StepChecklist({ steps, currentStep, finished }: StepChecklistProps) {
  return (
    <ol className="flex flex-col gap-1.5">
      {steps.map((step) => {
        const done = finished || step.id < currentStep;
        const active = !finished && step.id === currentStep;
        return (
          <li
            key={step.id}
            className={`flex items-start gap-3 rounded-md border px-3 py-2 transition-colors ${
              active
                ? "border-[var(--color-cyan)]/40 bg-[var(--color-cyan)]/10"
                : "border-transparent"
            }`}
          >
            <span
              className={`mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full font-mono text-[10px] ${
                done
                  ? "bg-[var(--color-green)]/20 text-[var(--color-green)]"
                  : active
                    ? "border border-[var(--color-cyan)] text-[var(--color-cyan)]"
                    : "border border-[var(--color-border)] text-[var(--color-text-faint)]"
              }`}
            >
              {done ? "✓" : step.id}
            </span>
            <span
              className={`text-sm leading-5 ${
                active
                  ? "text-[var(--color-text)]"
                  : done
                    ? "text-[var(--color-text-muted)] line-through decoration-[var(--color-border)]"
                    : "text-[var(--color-text-faint)]"
              }`}
            >
              {step.instruction}
            </span>
          </li>
        );
      })}
    </ol>
  );
}
