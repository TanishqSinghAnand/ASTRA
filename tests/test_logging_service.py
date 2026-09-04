"""Tests for backend/services/logging_service.py. Uses pytest's tmp_path
fixture for real file I/O — this module's entire job is persistence, so
faking the filesystem would test nothing meaningful."""
from __future__ import annotations

import json

from backend.experiment.experiment_loader import ExperimentDefinition
from backend.perception.base import SequenceEvent, SequenceStatus
from backend.services.logging_service import LoggingService

EXPERIMENT = ExperimentDefinition(
    experiment_id="TEST_LOG",
    name="Test experiment",
    objects={
        "RED_BOX": {"label": "red container"},
        "BLUE_BOX": {"label": "blue container"},
        "EXPERIMENT_AREA": {"label": "experiment area"},
    },
    steps=[
        {"id": 1, "action": "PICK", "object": "RED_BOX", "instruction": "Pick red.", "voice_next": "Place red."},
        {
            "id": 2,
            "action": "PLACE",
            "object": "RED_BOX",
            "target": "EXPERIMENT_AREA",
            "instruction": "Place red.",
            "voice_next": "Pick blue.",
        },
        {"id": 3, "action": "COMPLETE", "object": "EXPERIMENT", "instruction": "Done.", "voice_next": "Done."},
    ],
)


def _event(step, status, expected="", detected="") -> SequenceEvent:
    return SequenceEvent(step=step, expected=expected, detected=detected, confidence=0.9, status=status)


def test_start_run_creates_reports_dir_and_jsonl_file(tmp_path):
    service = LoggingService(EXPERIMENT, tmp_path / "reports")
    service.start_run()
    jsonl_files = list((tmp_path / "reports").glob("run_*.jsonl"))
    assert len(jsonl_files) == 1


def test_log_event_appends_to_jsonl(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    service.log_event(_event(1, SequenceStatus.CORRECT, "PICK_RED_BOX", "PICK_RED_BOX"))
    service.log_event(_event(2, SequenceStatus.WRONG_OBJECT, "PLACE_RED_BOX", "PICK_BLUE_BOX"))

    jsonl_path = next(reports_dir.glob("run_*.jsonl"))
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["status"] == "CORRECT"
    second = json.loads(lines[1])
    assert second["status"] == "WRONG_OBJECT"


def test_finalize_run_writes_report_with_correct_totals(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    service.log_event(_event(1, SequenceStatus.CORRECT))
    service.log_event(_event(2, SequenceStatus.WRONG_OBJECT))  # deviation, step 2 not yet resolved
    service.log_event(_event(2, SequenceStatus.RECOVERED))  # step 2 needed a correction
    service.log_event(_event(3, SequenceStatus.COMPLETE))

    report_path = service.finalize_run(finished=True)
    assert report_path is not None
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["experiment_id"] == "TEST_LOG"
    assert report["finished"] is True
    assert report["total_deviations"] == 1
    assert report["total_steps"] == 3
    assert report["duration_seconds"] >= 0

    steps_by_id = {s["step_id"]: s for s in report["steps"]}
    assert steps_by_id[1]["first_try_correct"] is True
    assert steps_by_id[1]["needed_correction"] is False
    assert steps_by_id[2]["first_try_correct"] is False
    assert steps_by_id[2]["needed_correction"] is True
    assert steps_by_id[3]["first_try_correct"] is True  # COMPLETE counts as first-try


def test_finalize_run_uses_fixed_report_filename(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    service.finalize_run(finished=False)
    assert (reports_dir / "experiment_report.json").exists()


def test_finalize_run_is_idempotent(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    first = service.finalize_run(finished=True)
    assert first is not None
    second = service.finalize_run(finished=True)
    assert second is None  # already closed — no-op, doesn't crash or overwrite silently


def test_log_event_after_finalize_is_ignored(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    service.finalize_run(finished=True)
    # Must not raise (e.g. writing to a closed file handle).
    service.log_event(_event(1, SequenceStatus.CORRECT))


def test_second_run_gets_its_own_jsonl_file(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    service.log_event(_event(1, SequenceStatus.CORRECT))
    service.finalize_run(finished=False)

    service.start_run()
    service.log_event(_event(1, SequenceStatus.CORRECT))
    service.finalize_run(finished=False)

    jsonl_files = list(reports_dir.glob("run_*.jsonl"))
    assert len(jsonl_files) == 2


def test_incomplete_step_reports_not_completed(tmp_path):
    reports_dir = tmp_path / "reports"
    service = LoggingService(EXPERIMENT, reports_dir)
    service.start_run()
    service.log_event(_event(1, SequenceStatus.CORRECT))
    report_path = service.finalize_run(finished=False)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    steps_by_id = {s["step_id"]: s for s in report["steps"]}
    assert steps_by_id[1]["completed"] is True
    assert steps_by_id[2]["completed"] is False
    assert steps_by_id[2]["first_try_correct"] is False
    assert steps_by_id[2]["needed_correction"] is False
