#!/usr/bin/env python3
"""Baseline classifier training (Phase 12 stub/placeholder).

Explicitly a placeholder, not this project's real action-recognition
approach — backend/temporal/action_recognizer.py's RuleBasedActionRecognizer
is what the MVP demo actually runs, and needs zero labeled data, for the
same "no dataset exists yet" reason color_detector.py uses HSV thresholding
instead of a trained detector (see its own module docstring). This script
exists only so the shape of a future training pipeline
(features.py -> train_baseline.py -> evaluate.py) is in place if/when a
real dataset is built up via tools/record_dataset.py.

Honesty note: tools/record_dataset.py currently saves one isolated frame
per labeled keypress, not a temporal window — so training/features.py's
displacement/velocity features are MISSING (see its MISSING sentinel) for
every example recorded this way; only the per-frame hand-distance and
detector-confidence features carry real signal until record_dataset.py (or
a successor) captures actual multi-frame sequences per label. Any accuracy
number this script prints reflects that limitation — it is not evidence of
real temporal action recognition.

Writes data/reports/evaluation_report.json either way (insufficient-data
or a real result), so evaluate.py has something honest to display without
re-running training itself.

Usage:
    python training/train_baseline.py --data-dir data/raw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402

from backend.config.settings import load_settings  # noqa: E402
from backend.perception.real_engine import RealPerceptionEngine  # noqa: E402
from training.features import extract_window_features  # noqa: E402

MIN_LABELED_FRAMES = 20  # below this, any accuracy number is noise, not a real evaluation basis
MIN_CLASSES = 2


def _load_manifest_records(data_dir: Path) -> list[dict]:
    records: list[dict] = []
    if not data_dir.exists():
        return records
    for manifest_path in sorted(data_dir.glob("*/manifest.jsonl")):
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def _write_report(reports_dir: Path, report: dict) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / "evaluation_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _insufficient_data_report(record_count: int, class_count: int, reason_extra: str = "") -> dict:
    reason = (
        f"insufficient evaluation data: {record_count} labeled frame(s) "
        f"across {class_count} class(es) recorded (need >= {MIN_LABELED_FRAMES} "
        f"frames across >= {MIN_CLASSES} classes).{reason_extra} "
        "Run tools/record_dataset.py to collect more."
    )
    return {
        "status": "N/A",
        "reason": reason,
        "accuracy": None,
        "labeled_frame_count": record_count,
        "class_count": class_count,
        "generated_at": time.time(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default="data/raw")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / args.data_dir
    settings = load_settings()
    reports_dir = repo_root / settings.paths.reports_dir

    records = _load_manifest_records(data_dir)
    labels_seen = {r["label"] for r in records}

    if len(records) < MIN_LABELED_FRAMES or len(labels_seen) < MIN_CLASSES:
        report = _insufficient_data_report(len(records), len(labels_seen))
        _write_report(reports_dir, report)
        print(report["reason"])
        return

    tracked_classes = list(settings.perception.colors.keys())
    engine = RealPerceptionEngine(settings.perception)

    X: list[dict] = []
    y: list[str] = []
    try:
        for record in records:
            image_path = repo_root / record["file"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            pframe = engine.process(frame, record.get("frame_index", 0))
            features = extract_window_features([pframe], tracked_classes)
            if not features:
                continue
            X.append(features)
            y.append(record["label"])
    finally:
        engine.close()

    if len(X) < MIN_LABELED_FRAMES or len(set(y)) < MIN_CLASSES:
        report = _insufficient_data_report(
            len(X), len(set(y)), reason_extra=" (some recorded frames failed to load or process.)"
        )
        _write_report(reports_dir, report)
        print(report["reason"])
        return

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    vectorizer = DictVectorizer()
    X_vec = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.25, random_state=0, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    caveat = (
        "Each training example is a single-frame window (see module "
        "docstring) — displacement/velocity features carried no signal; "
        "this number reflects hand-distance + detector-confidence "
        "features only, not real temporal action recognition."
    )
    report = {
        "status": "EVALUATED",
        "reason": caveat,
        "accuracy": round(accuracy, 4),
        "labeled_frame_count": len(X),
        "class_count": len(set(y)),
        "test_set_size": len(y_test),
        "generated_at": time.time(),
    }
    report_path = _write_report(reports_dir, report)

    print(f"Trained on {len(X)} labeled frames ({len(set(y))} classes).")
    print(f"Held-out accuracy: {accuracy:.2%} (test set: {len(y_test)} frames)")
    print(f"Caveat: {caveat}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
