"""Mock perception + action recognizer (spec section 35): lets the full
CAMERA -> PERCEPTION -> ACTION -> SEQUENCE -> OUTPUT loop run end-to-end —
including over a real webcam feed, if one is attached — before the real
MediaPipe/HSV perception (Phase 2) and rule-based temporal recognizer
(Phase 4) exist. This is also how the pipeline gets verified in this
dev sandbox, which has no camera.

MockActionRecognizer plays back the exact scripted scenario from spec
section 40 / 53 — the canonical "first demo target": two correct steps, a
deliberate deviation, a recovery, then completion — so the very first
runnable backend already demonstrates the full closed loop the judges care
about.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from backend.perception.base import (
    ActionPrediction,
    HandFrame,
    Landmark,
    PerceptionFrame,
    PoseFrame,
)
from backend.perception.interfaces import ActionRecognizer, PerceptionEngine


class MockPerceptionEngine(PerceptionEngine):
    """Returns a plausible-looking (but fabricated) PerceptionFrame so
    downstream services and the UI have something to render. Replaced by
    perception/detector.py + pose.py + hands.py in Phase 2."""

    def process(self, frame: np.ndarray, frame_index: int) -> PerceptionFrame:
        h, w = frame.shape[:2]
        return PerceptionFrame(
            frame_index=frame_index,
            frame_width=w,
            frame_height=h,
            objects=[],
            pose=PoseFrame(
                detected=True,
                landmarks=[
                    Landmark(name="right_wrist", x=0.55, y=0.5, visibility=0.9),
                    Landmark(name="left_wrist", x=0.45, y=0.5, visibility=0.9),
                ],
            ),
            hands=[HandFrame(detected=True, handedness="Right")],
        )


@dataclass
class _ScriptedEvent:
    action: str
    confidence: float
    fire_at_seconds: float  # elapsed seconds since recognizer start/reset


# The canonical section-40/53 demo scenario: correct, correct, DEVIATION,
# recovery, correct, correct, complete.
DEFAULT_SCRIPT: list[_ScriptedEvent] = [
    _ScriptedEvent("PICK_RED_BOX", 0.96, 3.0),
    _ScriptedEvent("PICK_BLUE_BOX", 0.94, 7.0),   # deliberate deviation (expected PLACE_RED_BOX)
    _ScriptedEvent("PLACE_RED_BOX", 0.95, 11.0),  # correction
    _ScriptedEvent("PICK_BLUE_BOX", 0.93, 15.0),
    _ScriptedEvent("PLACE_BLUE_BOX", 0.97, 19.0),
    _ScriptedEvent("COMPLETE_EXPERIMENT", 0.99, 23.0),
]


class MockActionRecognizer(ActionRecognizer):
    """Emits ActionPrediction(action=None) ("observing...") most of the
    time, and fires the next scripted action once its timer elapses. Each
    scripted action fires exactly once per run, mirroring how a real
    temporally-smoothed recognizer only emits a *change* in recognized
    action, not the same action every frame (spec section 30)."""

    def __init__(self, script: list[_ScriptedEvent] | None = None):
        self._script = script or DEFAULT_SCRIPT
        self._next_index = 0
        self._start_time = time.time()

    def update(self, perception_frame: PerceptionFrame) -> ActionPrediction:
        elapsed = time.time() - self._start_time
        if self._next_index < len(self._script):
            ev = self._script[self._next_index]
            if elapsed >= ev.fire_at_seconds:
                self._next_index += 1
                return ActionPrediction(action=ev.action, confidence=ev.confidence, source="mock")
        return ActionPrediction(action=None, confidence=0.0, source="mock")

    def reset(self) -> None:
        self._next_index = 0
        self._start_time = time.time()
