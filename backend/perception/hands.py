"""MediaPipe Hands wrapper — dedicated hand/fingertip tracking.

Pose's wrist landmarks are enough for coarse arm motion, but hand-object
interaction reasoning (Phase 3: approaching/touching/holding/releasing)
needs a tighter fix on where the hand actually is, so this tracks the wrist
+ palm-center landmarks per detected hand, up to 2 hands, offline/CPU like
pose.py.
"""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from backend.perception.base import HandFrame, Landmark

_mp_hands = mp.solutions.hands

# A hand's "position" for interaction reasoning is taken as wrist + the
# base knuckles of index/pinky (a stable approximation of palm center),
# rather than fingertips, which move a lot relative to the object being
# held.
_RELEVANT_LANDMARKS: dict[str, int] = {
    "wrist": _mp_hands.HandLandmark.WRIST,
    "index_mcp": _mp_hands.HandLandmark.INDEX_FINGER_MCP,
    "pinky_mcp": _mp_hands.HandLandmark.PINKY_MCP,
    "thumb_tip": _mp_hands.HandLandmark.THUMB_TIP,
    "index_tip": _mp_hands.HandLandmark.INDEX_FINGER_TIP,
}


class HandEstimator:
    def __init__(self, max_num_hands: int = 2, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._hands = _mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, frame_bgr: np.ndarray) -> list[HandFrame]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            return []

        hand_frames: list[HandFrame] = []
        handedness_list = results.multi_handedness or []
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            landmarks = []
            for name, idx in _RELEVANT_LANDMARKS.items():
                lm = hand_landmarks.landmark[idx]
                landmarks.append(Landmark(name=name, x=lm.x, y=lm.y, z=lm.z, visibility=1.0))
            handedness = None
            if i < len(handedness_list) and handedness_list[i].classification:
                handedness = handedness_list[i].classification[0].label
            hand_frames.append(HandFrame(detected=True, landmarks=landmarks, handedness=handedness))
        return hand_frames

    def close(self) -> None:
        self._hands.close()
