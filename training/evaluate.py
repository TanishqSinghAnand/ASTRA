#!/usr/bin/env python3
"""Evaluation report reader (Phase 12 stub/placeholder).

Per the project's no-fabricated-metrics rule (see color_detector.py's
honesty note on HSV "confidence", and every other honesty note across
this codebase): reports "N/A — insufficient evaluation data" rather than
inventing a number, until train_baseline.py has actually run against a
real recorded dataset.

Reads (rather than regenerates) data/reports/evaluation_report.json —
train_baseline.py is the only place that trains a model and writes that
file, so this script doesn't duplicate its data-loading/feature-extraction
logic. If no report exists yet at all (train_baseline.py has never been
run), this writes a placeholder N/A one so a frontend Evaluation view (or
this script run again) always has something honest to read, rather than a
missing file.

Usage:
    python training/evaluate.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import load_settings  # noqa: E402

_NEVER_RUN_REASON = (
    "insufficient evaluation data: training/train_baseline.py has never "
    "been run. Record a labeled dataset with tools/record_dataset.py, "
    "then run training/train_baseline.py."
)


def main() -> None:
    settings = load_settings()
    repo_root = Path(__file__).resolve().parents[1]
    report_path = repo_root / settings.paths.reports_dir / "evaluation_report.json"

    if not report_path.exists():
        report_path.parent.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "status": "N/A",
            "reason": _NEVER_RUN_REASON,
            "accuracy": None,
            "labeled_frame_count": 0,
            "class_count": 0,
            "generated_at": time.time(),
        }
        report_path.write_text(json.dumps(placeholder, indent=2), encoding="utf-8")
        print(_NEVER_RUN_REASON)
        return

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "EVALUATED":
        print(report.get("reason", "N/A — insufficient evaluation data."))
        return

    print(f"Status: EVALUATED")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Labeled frames: {report['labeled_frame_count']}  Classes: {report['class_count']}")
    print(f"Caveat: {report['reason']}")


if __name__ == "__main__":
    main()
