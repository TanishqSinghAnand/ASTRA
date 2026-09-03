"""MediaPipe Pose wrapper — body/arm skeleton tracking.

Pretrained, offline (the model ships inside the mediapipe pip package — no
separate download), CPU-only. Per spec section 10, MVP only needs
hand/arm motion, so we track shoulders/elbows/wrists (+ hips as a stable
torso reference) rather than the full 33-point body mesh.
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from backend.perception.base import Landmark, PoseFrame

_mp_pose = mp.solutions.pose

# Name -> mediapipe PoseLandmark index. Coordinates come back normalized
# (0-1 of frame width/height), which is what PerceptionFrame expects.
_RELEVANT_LANDMARKS: dict[str, int] = {
    "left_shoulder": _mp_pose.PoseLandmark.LEFT_SHOULDER,
    "right_shoulder": _mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "left_elbow": _mp_pose.PoseLandmark.LEFT_ELBOW,
    "right_elbow": _mp_pose.PoseLandmark.RIGHT_ELBOW,
    "left_wrist": _mp_pose.PoseLandmark.LEFT_WRIST,
    "right_wrist": _mp_pose.PoseLandmark.RIGHT_WRIST,
    "left_hip": _mp_pose.PoseLandmark.LEFT_HIP,
    "right_hip": _mp_pose.PoseLandmark.RIGHT_HIP,
    "nose": _mp_pose.PoseLandmark.NOSE,
}


class PoseEstimator:
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            # model_complexity=1 ("full") ships inside the mediapipe pip
            # package itself. complexity=0 ("lite") is NOT bundled and
            # triggers a one-time download from Google Cloud Storage on
            # first use — which breaks the "runs offline" promise (and
            # fails outright on a restricted network, as it did in dev).
            # 1 is barely heavier and keeps this genuinely offline-capable.
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> PoseFrame:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        if not results.pose_landmarks:
            return PoseFrame(detected=False)

        landmarks = []
        for name, idx in _RELEVANT_LANDMARKS.items():
            lm = results.pose_landmarks.landmark[idx]
            landmarks.append(Landmark(name=name, x=lm.x, y=lm.y, z=lm.z, visibility=lm.visibility))
        return PoseFrame(detected=True, landmarks=landmarks)

    def close(self) -> None:
        self._pose.close()
