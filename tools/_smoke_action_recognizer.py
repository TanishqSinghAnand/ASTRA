"""Throwaway manual smoke test for Phase 4 (backend/temporal/action_recognizer.py)
against a real webcam: runs RealPerceptionEngine + RuleBasedActionRecognizer
together and prints any emitted ActionPrediction, so a real pick/place run
with an actual red/blue box can be eyeballed end-to-end.

    python tools/_smoke_action_recognizer.py --seconds 20
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import load_settings  # noqa: E402
from backend.perception.real_engine import RealPerceptionEngine  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402
from backend.temporal.action_recognizer import RuleBasedActionRecognizer  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="webcam")
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--seconds", type=float, default=20.0)
args = parser.parse_args()

settings = load_settings()
cam_settings = settings.camera.model_copy(update={"source": args.source, "camera_index": args.index})
repo_root = Path(__file__).resolve().parents[1]

camera = CameraService(cam_settings, repo_root)
engine = RealPerceptionEngine(settings.perception)
recognizer = RuleBasedActionRecognizer(settings.perception, settings.temporal)

try:
    camera.start()
except CameraError as e:
    print(f"FAILED to start camera: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Running for {args.seconds}s — pick up a red/blue box and place it in the experiment area.")
deadline = time.time() + args.seconds
frames = 0
crashes = 0
try:
    while time.time() < deadline:
        captured = camera.read_latest()
        if captured is None:
            time.sleep(0.02)
            continue
        try:
            perception = engine.process(captured.frame, captured.frame_index)
            prediction = recognizer.update(perception)
        except Exception as e:  # deliberately broad: this is a crash smoke test
            crashes += 1
            print(f"CRASH on frame {captured.frame_index}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        frames += 1
        if prediction.action is not None:
            print(f"frame {captured.frame_index:5d}  ACTION -> {prediction.action}  confidence={prediction.confidence:.2f}")
finally:
    camera.stop()
    engine.close()

print(f"\nDone. frames processed={frames}  crashes={crashes}")
