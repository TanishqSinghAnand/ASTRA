#!/usr/bin/env python3
"""Full-pipeline local demo — perception + action recognition + sequence
validation, running entirely in-process. No backend/frontend needed.

Unlike tools/live_preview.py (perception only — objects/pose/hands, no
notion of "action" or "correct"), this runs the exact same pipeline the
web dashboard's backend does — RealPerceptionEngine ->
RuleBasedActionRecognizer -> RuleBasedSequenceEngine — directly against a
camera source, single-threaded, no asyncio/WebSocket/FastAPI involved.

Shows, both in the terminal and as an on-screen overlay:
  - which step you're on and what to do next
  - the detected action + confidence as it's recognized
  - an explicit confirmation banner the instant a PICK/PLACE is judged
    CORRECT/RECOVERED/COMPLETE, or flagged as a deviation (wrong object,
    wrong location, skipped step, etc.) — the same statuses
    backend/experiment/validator.py produces for the real dashboard, just
    printed/drawn locally instead of broadcast over a WebSocket.

Usage:
    python tools/live_demo.py --source webcam --index 0

Press 'q' to quit, 'r' to reset the sequence and start over.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import CameraSettings, load_settings  # noqa: E402
from backend.experiment.experiment_loader import load_experiment  # noqa: E402
from backend.experiment.validator import RuleBasedSequenceEngine  # noqa: E402
from backend.perception.base import ERROR_STATUSES, SequenceEvent, SequenceStatus  # noqa: E402
from backend.perception.real_engine import RealPerceptionEngine  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402
from backend.services.display_utils import DEFAULT_DISPLAY_MAX_WIDTH, fit_for_display, gui_available  # noqa: E402
from backend.services.overlay import draw_label, draw_perception_overlay  # noqa: E402
from backend.temporal.action_recognizer import RuleBasedActionRecognizer  # noqa: E402

WINDOW_NAME = "ASTRA local demo — press q to quit, r to reset"
BANNER_DISPLAY_S = 2.5

_SUCCESS_COLOR = (0, 200, 0)
_ERROR_COLOR = (0, 0, 230)


def _status_color(status: SequenceStatus) -> tuple[int, int, int]:
    return _ERROR_COLOR if status in ERROR_STATUSES else _SUCCESS_COLOR


def _print_event(event: SequenceEvent) -> None:
    marker = "!!" if event.status in ERROR_STATUSES else "OK"
    print(
        f"{marker} [step {event.step}] {event.status.value}: "
        f"expected={event.expected} detected={event.detected} confidence={event.confidence:.0%}"
    )
    for line in event.explanation:
        print(f"      {line}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["webcam", "video_file", "synthetic"], default=None)
    parser.add_argument("--index", type=int, default=None, help="Camera index (webcam source)")
    parser.add_argument("--path", type=str, default=None, help="Video file path (video_file source)")
    parser.add_argument("--display-width", type=int, default=DEFAULT_DISPLAY_MAX_WIDTH)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a config.yaml variant (default: config/config.yaml). "
             "E.g. config/config.yolo.yaml for the YOLO cup/bottle setup.",
    )
    args = parser.parse_args()

    settings = load_settings(args.config)
    overrides: dict = {}
    if args.source:
        overrides["source"] = args.source
    if args.index is not None:
        overrides["camera_index"] = args.index
    if args.path:
        overrides["video_file_path"] = args.path
    cam_settings = CameraSettings(**{**settings.camera.model_dump(), **overrides})

    repo_root = Path(__file__).resolve().parents[1]
    experiment = load_experiment(repo_root / settings.paths.experiment_config)

    camera = CameraService(cam_settings, repo_root)
    perception = RealPerceptionEngine(settings.perception)
    action_recognizer = RuleBasedActionRecognizer(settings.perception, settings.temporal)
    sequence_engine = RuleBasedSequenceEngine(experiment, settings.perception.confidence_threshold)

    print(f"Starting local demo: source={cam_settings.source!r} index={cam_settings.camera_index}")
    print(f"Experiment: {experiment.name} ({experiment.total_steps()} steps)")
    try:
        camera.start()
    except CameraError as e:
        print(f"FAILED to start camera: {e}", file=sys.stderr)
        sys.exit(1)

    display_available = gui_available()
    if display_available:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    def print_current_step() -> None:
        if sequence_engine.is_finished():
            print("\n=== EXPERIMENT COMPLETE ===")
        else:
            step = experiment.step_by_id(sequence_engine.current_step())
            print(f"\n--- Step {step.id}/{experiment.total_steps()}: {step.instruction} ---")

    print_current_step()

    last_event: SequenceEvent | None = None
    last_event_at = 0.0
    last_debug_print = 0.0

    try:
        while True:
            captured = camera.read_latest()
            if captured is None:
                time.sleep(0.02)
                continue

            pframe = perception.process(captured.frame, captured.frame_index)
            prediction = action_recognizer.update(pframe)

            # -- debug: per-object motion state, to see exactly why a pick/
            # place isn't registering instead of guessing blind. Reflects
            # action_recognizer.last_events, set by the update() call above
            # — not a second, separate detection pass.
            debug_lines: list[str] = [
                f"{cls}: {event.state.value}" + (f" (zone={event.zone})" if event.zone else "")
                for cls, event in action_recognizer.last_events.items()
            ]
            now = time.time()
            if debug_lines and now - last_debug_print > 1.0:
                last_debug_print = now
                for line in debug_lines:
                    print(f"[debug] {line}  frame={pframe.frame_width}x{pframe.frame_height}")

            # Same history-diff pattern as inference_service.py:
            # submit_action returns only its single "headline" event, but
            # the auto-COMPLETE advance can append a second one.
            history_len_before = len(sequence_engine.history)
            returned_event = sequence_engine.submit_action(prediction)
            if returned_event is not None:
                for event in sequence_engine.history[history_len_before:]:
                    _print_event(event)
                    last_event = event
                    last_event_at = time.time()
                print_current_step()

            annotated = draw_perception_overlay(captured.frame, pframe)

            if sequence_engine.is_finished():
                step_text = "EXPERIMENT COMPLETE"
                instruction = "All steps verified."
            else:
                step = experiment.step_by_id(sequence_engine.current_step())
                step_text = f"STEP {step.id}/{experiment.total_steps()}"
                instruction = step.instruction
            draw_label(annotated, step_text, (14, 40), (255, 255, 255), font_scale=0.9, thickness=2)
            draw_label(annotated, instruction, (14, 75), (255, 255, 255), font_scale=0.7, thickness=2)
            for i, line in enumerate(debug_lines):
                draw_label(annotated, line, (14, 110 + i * 28), (0, 210, 255), font_scale=0.55, thickness=2)

            action_text = (
                f"detected: {prediction.action}  ({prediction.confidence:.0%})"
                if prediction.action
                else "detected: observing..."
            )
            draw_label(annotated, action_text, (14, annotated.shape[0] - 16), (0, 255, 60), font_scale=0.6)

            if last_event is not None and (time.time() - last_event_at) < BANNER_DISPLAY_S:
                banner = last_event.status.value.replace("_", " ")
                color = _status_color(last_event.status)
                (tw, th), baseline = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                bx = max(10, (annotated.shape[1] - tw) // 2)
                by = 100
                pad = 18
                cv2.rectangle(annotated, (bx - pad, by - th - pad), (bx + tw + pad, by + baseline + pad), (0, 0, 0), -1)
                cv2.rectangle(annotated, (bx - pad, by - th - pad), (bx + tw + pad, by + baseline + pad), color, 3)
                draw_label(annotated, banner, (bx, by), color, font_scale=1.2, thickness=3)

            if display_available:
                try:
                    cv2.imshow(WINDOW_NAME, fit_for_display(annotated, args.display_width))
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        sequence_engine.reset()
                        action_recognizer.reset()
                        last_event = None
                        print("\n=== RESET ===")
                        print_current_step()
                except cv2.error:
                    display_available = False
                    print("No display backend available — this tool needs a GUI window (it's interactive).")
    except KeyboardInterrupt:
        pass
    finally:
        camera.stop()
        perception.close()
        if display_available:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


if __name__ == "__main__":
    main()
