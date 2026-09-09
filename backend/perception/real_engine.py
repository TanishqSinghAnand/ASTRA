"""Composes an object detector (HSV or, from v2.0, YOLO) + pose + hands
wrappers into one PerceptionEngine, per the PerceptionEngine interface.
This is the drop-in replacement for
backend/mocks/mock_engines.py::MockPerceptionEngine.
"""
from __future__ import annotations

import numpy as np

from backend.config.settings import PerceptionSettings
from backend.perception.base import PerceptionFrame
from backend.perception.color_detector import ColorBoxDetector
from backend.perception.hands import HandEstimator
from backend.perception.interfaces import PerceptionEngine
from backend.perception.pose import PoseEstimator
from backend.perception.zones import zone_objects


class RealPerceptionEngine(PerceptionEngine):
    def __init__(self, settings: PerceptionSettings):
        self.settings = settings
        if settings.detector_backend == "yolo":
            # Deliberately lazy: object_detector.py imports ultralytics,
            # which pulls in torch — a real weight/memory jump nothing
            # else here needs. A deployment running detector_backend=hsv
            # (e.g. the cloud demo, see render.yaml) shouldn't pay that
            # cost just because this module got imported.
            from backend.perception.object_detector import ObjectDetector

            self.detector = ObjectDetector(settings)
        else:
            self.detector = ColorBoxDetector(settings)
        self.pose = PoseEstimator()
        self.hands = HandEstimator()

    def process(self, frame: np.ndarray, frame_index: int) -> PerceptionFrame:
        h, w = frame.shape[:2]
        objects = self.detector.detect(frame) + zone_objects(self.settings, w, h)
        pose_frame = self.pose.process(frame)
        hand_frames = self.hands.process(frame)
        return PerceptionFrame(
            frame_index=frame_index,
            frame_width=w,
            frame_height=h,
            objects=objects,
            pose=pose_frame,
            hands=hand_frames,
        )

    def close(self) -> None:
        self.pose.close()
        self.hands.close()
