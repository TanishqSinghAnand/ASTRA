"""Throwaway manual smoke test for Phase 3 (backend/perception/interaction.py)
against a real webcam: runs RealPerceptionEngine + InteractionReasoner
together and prints any interaction state that isn't NONE, so a run with a
real red/blue box + hand in frame can be eyeballed. Not part of the
deliverable (see tools/_*.py convention for other one-off checks).

    python tools/_smoke_interaction.py --seconds 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.config.settings import load_settings  # noqa: E402
from backend.perception.interaction import InteractionReasoner  # noqa: E402
from backend.perception.real_engine import RealPerceptionEngine  # noqa: E402
from backend.services.camera_service import CameraError, CameraService  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="webcam")
parser.add_argument("--index", type=int, default=0)
parser.add_argument("--seconds", type=float, default=15.0)
args = parser.parse_args()

settings = load_settings()
overrides = {"source": args.source, "camera_index": args.index}
cam_settings = settings.camera.model_copy(update=overrides)
repo_root = Path(__file__).resolve().parents[1]

camera = CameraService(cam_settings, repo_root)
engine = RealPerceptionEngine(settings.perception)
reasoner = InteractionReasoner(settings.perception)

try:
    camera.start()
except CameraError as e:
    print(f"FAILED to start camera: {e}", file=sys.stderr)
    sys.exit(1)

print(f"Running for {args.seconds}s — hold a red/blue box, approach/touch/lift/move/place it near the experiment area.")
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
            events = reasoner.update(perception)
        except Exception as e:  # deliberately broad: this is a crash smoke test
            crashes += 1
            print(f"CRASH on frame {captured.frame_index}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        frames += 1
        for cls, ev in events.items():
            if ev.changed:
                print(f"frame {captured.frame_index:5d}  {cls:10s} -> {ev.state.value}")
        time.sleep(0.03)
finally:
    camera.stop()
    engine.close()

print(f"\nDone. frames processed={frames}  crashes={crashes}")
