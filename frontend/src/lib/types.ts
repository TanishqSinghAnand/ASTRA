// Mirrors backend/perception/base.py and backend/services/inference_service.py's
// status_dict()/WS message shapes exactly — this is the only place the
// frontend's understanding of the backend's schema lives.

export type SequenceStatus =
  | "CORRECT"
  | "WRONG_OBJECT"
  | "SKIPPED_STEP"
  | "OUT_OF_SEQUENCE"
  | "REPEATED_STEP"
  | "LOW_CONFIDENCE"
  | "RECOVERED"
  | "COMPLETE";

export const ERROR_STATUSES: readonly SequenceStatus[] = [
  "WRONG_OBJECT",
  "SKIPPED_STEP",
  "OUT_OF_SEQUENCE",
  "REPEATED_STEP",
];

export interface SequenceEvent {
  timestamp: number;
  step: number;
  expected: string | null;
  detected: string | null;
  confidence: number;
  status: SequenceStatus;
  error_type: string | null;
  recovered: boolean | null;
  explanation: string[];
}

export interface ExperimentObject {
  label: string;
  color_hint: string | null;
}

export interface ExperimentStep {
  id: number;
  action: string;
  object: string;
  target?: string | null;
  instruction: string;
  voice_next: string;
}

export interface ExperimentDefinition {
  experiment_id: string;
  name: string;
  description: string;
  objects: Record<string, ExperimentObject>;
  steps: ExperimentStep[];
}

export type RunStatus = "IDLE" | "RUNNING" | "FINISHED" | "STOPPED" | "ERROR";

// The dashboard's two tabs — matches backend/api/routes.py's
// _MODE_CONFIGS keys exactly.
export type DetectorMode = "color" | "yolo";

export interface ModeResponse {
  mode: DetectorMode;
  status: StatusDict;
  experiment: ExperimentDefinition;
}

export interface StatusDict {
  status: RunStatus;
  error: string | null;
  camera_source: string;
  camera_running: boolean;
  fps: number;
  current_step: number;
  total_steps: number;
  current_instruction: string | null;
  finished: boolean;
  device: string;
  offline: boolean;
}

export interface DwellProgress {
  object: string;
  elapsed: number;
  required: number;
}

export type LiveMessage =
  | ({ type: "status" } & StatusDict)
  | {
      type: "perception";
      action: string | null;
      confidence: number;
      step: number;
      fps: number;
      dwell: DwellProgress | null;
    }
  | ({ type: "sequence_event" } & SequenceEvent)
  | { type: "frame"; image: string; frame_index: number };
