#!/usr/bin/env python3
"""Webcam dataset recording tool (Phase 11) — groundwork for Phase 12's
training pipeline. Not used by the MVP demo loop.

Shows a live preview; press a labeled number key to save the *current*
frame under data/raw/<session_id>/<label>/ and append a manifest row
(JSON Lines — one record per line, flushed immediately, so a crash mid-
session doesn't lose earlier captures, same pattern as logging_service.py).

Labels are read directly from the loaded experiment's steps (their
action_key), not hardcoded — press 1..N for however many steps the
experiment defines, so a different experiment's label set doesn't require
touching this code.

This is fundamentally an interactive, human-in-the-loop tool (a person
decides the exact moment to press a key), so unlike test_camera.py /
live_preview.py / calibrate_hsv.py there is no meaningful headless mode —
cv2.waitKey only receives keystrokes through an actual GUI window. If no
display is available, this tool prints an explanation and exits rather
than pretending to work.

Usage:
    python tools/record_dataset.py --source webcam --index 0
    python tools/record_dataset.py --source webcam --session my_session_01

Press the printed number keys to label the current frame, 'q' to quit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import load_settings  # noqa: E402
from backend.experiment.experiment_loader import load_experiment  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402
from backend.services.display_utils import gui_available  # noqa: E402

CONFIRMATION_DISPLAY_S = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["webcam", "video_file"], default="webcam")
    parser.add_argument("--index", type=int, default=None, help="Camera index (webcam source)")
    parser.add_argument("--session", type=str, default=None, help="Session id (default: timestamp)")
    args = parser.parse_args()

    if not gui_available():
        print(
            "No display available — this tool needs a GUI window to receive "
            "labeling keystrokes and can't run headlessly. Run it on a "
            "machine with a display attached.",
            file=sys.stderr,
        )
        sys.exit(1)

    settings = load_settings()
    experiment = settings.resolve_path(settings.paths.experiment_config)
    experiment_def = load_experiment(experiment)
    labels = [step.action_key for step in experiment_def.steps]
    if len(labels) > 9:
        print("More than 9 steps — only the first 9 are key-bindable (keys 1-9).", file=sys.stderr)
        labels = labels[:9]
    key_to_label = {ord(str(i + 1)): label for i, label in enumerate(labels)}

    overrides: dict = {"source": args.source}
    if args.index is not None:
        overrides["camera_index"] = args.index
    cam_settings = settings.camera.model_copy(update=overrides)

    repo_root = Path(__file__).resolve().parents[1]
    session_id = args.session or time.strftime("%Y%m%dT%H%M%S")
    session_dir = repo_root / settings.paths.raw_data_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = session_dir / "manifest.jsonl"

    camera = CameraService(cam_settings, repo_root)
    try:
        camera.start()
    except CameraError as e:
        print(f"FAILED to start camera: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Session: {session_id}  ->  {session_dir}")
    print("Key bindings:")
    for i, label in enumerate(labels):
        print(f"  [{i + 1}] {label}")
    print("  [q] quit")

    counts = {label: 0 for label in labels}
    frame_index = 0
    last_saved_label: str | None = None
    last_saved_at = 0.0

    manifest_file = open(manifest_path, "a", encoding="utf-8")
    try:
        while True:
            captured = camera.read_latest()
            if captured is None:
                time.sleep(0.02)
                continue

            display = captured.frame.copy()
            h = display.shape[0]
            binding_text = "  ".join(f"[{i + 1}] {label}" for i, label in enumerate(labels))
            cv2.putText(display, binding_text, (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1, cv2.LINE_AA)
            counts_text = "  ".join(f"{label}:{counts[label]}" for label in labels)
            cv2.putText(display, counts_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
            if last_saved_label and (time.time() - last_saved_at) < CONFIRMATION_DISPLAY_S:
                cv2.putText(display, f"SAVED: {last_saved_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imshow("ASTRA dataset recorder (Phase 11) — press a label key, q to quit", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in key_to_label:
                label = key_to_label[key]
                label_dir = session_dir / label
                label_dir.mkdir(parents=True, exist_ok=True)
                frame_index += 1
                out_path = label_dir / f"frame_{frame_index:05d}.jpg"
                cv2.imwrite(str(out_path), captured.frame)

                record = {
                    "file": str(out_path.relative_to(repo_root)).replace("\\", "/"),
                    "timestamp": captured.timestamp,
                    "label": label,
                    "frame_index": captured.frame_index,
                }
                manifest_file.write(json.dumps(record) + "\n")
                manifest_file.flush()

                counts[label] += 1
                last_saved_label = label
                last_saved_at = time.time()
                print(f"Saved {out_path.name} -> {label}  (total {label}: {counts[label]})")
    finally:
        manifest_file.close()
        camera.stop()
        cv2.destroyAllWindows()

    print("\nSession summary:")
    for label, count in counts.items():
        print(f"  {label}: {count}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
