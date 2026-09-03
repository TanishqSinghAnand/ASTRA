"""Composes the color detector + pose + hands wrappers into one
PerceptionEngine, per the PerceptionEngine interface. This is the drop-in
replacement for backend/mocks/mock_engines.py::MockPerceptionEngine —
inference_service.py can swap one line to point here once action
recognition (Phase 3/4) is ready to consume real detections. Until then,
it's exercised directly by tools/live_preview.py.
"""
from __future__ import annotations

import numpy as np

from backend.config.settings import PerceptionSettings
from backend.perception.base import PerceptionFrame
from backend.perception.color_detector import ColorBoxDetector
from backend.perception.hands import HandEstimator
from backend.perception.interfaces import PerceptionEngine
from backend.perception.pose import PoseEstimator


class RealPerceptionEngine(PerceptionEngine):
    def __init__(self, settings: PerceptionSettings):
        self.detector = ColorBoxDetector(settings)
        self.pose = PoseEstimator()
        self.hands = HandEstimator()

    def process(self, frame: np.ndarray, frame_index: int) -> PerceptionFrame:
        objects = self.detector.detect(frame)
        pose_frame = self.pose.process(frame)
        hand_frames = self.hands.process(frame)
        return PerceptionFrame(
            frame_index=frame_index,
            objects=objects,
            pose=pose_frame,
            hands=hand_frames,
        )

    def close(self) -> None:
        self.pose.close()
        self.hands.close()
