"""Persistent event logging + run reports (Phase 8).

event_log on InferenceService is in-memory only and lost on restart —
this is the durable counterpart. Two artifacts per run, both under
paths.reports_dir (data/reports/):

  - A timestamped, append-as-it-happens JSONL run log
    (run_<run_id>.jsonl, one SequenceEvent per line) — the full historical
    record, written incrementally so a crash mid-run doesn't lose earlier
    events.
  - A single experiment_report.json (fixed name, overwritten each run —
    "the latest run's report") summarizing: start/end timestamps, total
    duration, total deviations, and per-step first-try-correct vs
    needed-correction.

"First-try-correct" vs "needed correction" is read directly off each
step's own final outcome status in the run (CORRECT/COMPLETE = first try;
RECOVERED = needed at least one correction before it was confirmed) —
no separate counting logic needed since the sequence engine already
encodes this distinction in the status it emits.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from backend.experiment.experiment_loader import ExperimentDefinition
from backend.perception.base import ERROR_STATUSES, SequenceEvent, SequenceStatus

logger = logging.getLogger("astra.logging")

_STEP_OUTCOME_STATUSES = (SequenceStatus.CORRECT, SequenceStatus.RECOVERED, SequenceStatus.COMPLETE)


class LoggingService:
    def __init__(self, experiment: ExperimentDefinition, reports_dir: Path):
        self.experiment = experiment
        self.reports_dir = reports_dir
        self._run_id: Optional[str] = None
        self._log_file = None
        self._started_at: Optional[float] = None
        self._step_outcomes: dict[int, SequenceStatus] = {}
        self._deviation_count = 0

    # -- lifecycle -----------------------------------------------------------

    def start_run(self) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        # A uuid suffix (not just the timestamp) avoids two runs started
        # within the same second — e.g. a quick start/stop/start in a demo,
        # or a tight test loop — silently appending to the same JSONL file.
        self._run_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}_{self.experiment.experiment_id}"
        self._step_outcomes = {}
        self._deviation_count = 0
        self._started_at = time.time()
        try:
            self._log_file = open(self.reports_dir / f"run_{self._run_id}.jsonl", "a", encoding="utf-8")
        except OSError:
            logger.exception("Could not open run log file — continuing without persistent per-event logging.")
            self._log_file = None

    def log_event(self, event: SequenceEvent) -> None:
        if self._started_at is None:
            return  # start_run() was never called, or finalize_run() already closed this run
        if self._log_file is not None:
            try:
                self._log_file.write(json.dumps(event.model_dump(mode="json")) + "\n")
                self._log_file.flush()
            except OSError:
                logger.exception("Failed to write event to run log.")
        if event.status in ERROR_STATUSES:
            self._deviation_count += 1
        if event.status in _STEP_OUTCOME_STATUSES:
            self._step_outcomes[event.step] = event.status

    def finalize_run(self, finished: bool) -> Optional[Path]:
        if self._started_at is None:
            return None  # already finalized (or never started) — idempotent no-op

        ended_at = time.time()
        report = {
            "experiment_id": self.experiment.experiment_id,
            "run_id": self._run_id,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "duration_seconds": round(ended_at - self._started_at, 2),
            "finished": finished,
            "total_steps": self.experiment.total_steps(),
            "total_deviations": self._deviation_count,
            "steps": [self._step_summary(step_id) for step_id in range(1, self.experiment.total_steps() + 1)],
        }

        report_path = self.reports_dir / "experiment_report.json"
        try:
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except OSError:
            logger.exception("Failed to write experiment_report.json.")
            report_path = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None
        self._started_at = None  # marks this run as closed; further log_event() calls are ignored
        return report_path

    # -- internals -----------------------------------------------------------

    def _step_summary(self, step_id: int) -> dict:
        step = self.experiment.step_by_id(step_id)
        outcome = self._step_outcomes.get(step_id)
        return {
            "step_id": step_id,
            "action_key": step.action_key if step else None,
            "completed": outcome is not None,
            "first_try_correct": outcome in (SequenceStatus.CORRECT, SequenceStatus.COMPLETE),
            "needed_correction": outcome == SequenceStatus.RECOVERED,
        }
