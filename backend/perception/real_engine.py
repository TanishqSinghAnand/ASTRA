"""Composes the color detector + hands wrappers into one PerceptionEngine,
per the PerceptionEngine interface. This is the drop-in replacement for
backend/mocks/mock_engines.py::MockPerceptionEngine — inference_service.py
can swap one line to point here once action recognition (Phase 3/4) is
ready to consume real detections. Until then, it's exercised directly by
tools/live_preview.py.

No Pose/body-skeleton tracking: this demo runs on a tabletop with the
camera fixed on the work surface, not a person's whole body, and nothing
in the interaction/action logic ever read pose data (see interaction.py's
motion-based pick/place — hand landmarks are for the on-screen overlay
only now, not state transitions either). Running MediaPipe Pose every
frame for a skeleton nobody's logic needs was pure wasted CPU.
"""
from __future__ import annotations

import numpy as np

from backend.config.settings import PerceptionSettings
from backend.perception.base import PerceptionFrame
from backend.perception.color_detector import ColorBoxDetector
from backend.perception.hands import HandEstimator
from backend.perception.interfaces import PerceptionEngine


class RealPerceptionEngine(PerceptionEngine):
    def __init__(self, settings: PerceptionSettings):
        self.detector = ColorBoxDetector(settings)
        self.hands = HandEstimator()

    def process(self, frame: np.ndarray, frame_index: int) -> PerceptionFrame:
        h, w = frame.shape[:2]
        objects = self.detector.detect(frame)
        hand_frames = self.hands.process(frame)
        return PerceptionFrame(
            frame_index=frame_index,
            frame_width=w,
            frame_height=h,
            objects=objects,
            hands=hand_frames,
        )

    def close(self) -> None:
        self.hands.close()
