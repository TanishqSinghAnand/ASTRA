"""Draws perception results onto a frame: object boxes, pose skeleton, hand
markers. Shared by tools/live_preview.py now and by the WebSocket frame
stream / recording_service.py once real perception is wired into the
backend (Phase 9) — annotation logic should only exist once.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.perception.base import PerceptionFrame

_OBJECT_COLORS = {
    "red_box": (60, 60, 230),
    "blue_box": (220, 120, 40),
    "experiment_area": (150, 150, 150),
}

_POSE_BONES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
]

_HAND_BONES = [("wrist", "index_mcp"), ("wrist", "pinky_mcp"), ("index_mcp", "pinky_mcp")]

POSE_COLOR = (58, 217, 203)     # accent cyan, BGR
HAND_COLOR = (0, 220, 255)      # amber-yellow, BGR


def draw_perception_overlay(frame: np.ndarray, perception: PerceptionFrame) -> np.ndarray:
    """Returns a new annotated frame; does not mutate the input."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for obj in perception.objects:
        color = _OBJECT_COLORS.get(obj.cls, (200, 200, 200))
        p1 = (int(obj.bbox.x1), int(obj.bbox.y1))
        p2 = (int(obj.bbox.x2), int(obj.bbox.y2))
        thickness = 1 if obj.cls == "experiment_area" else 2
        style_dashed = obj.cls == "experiment_area"
        if style_dashed:
            _dashed_rect(annotated, p1, p2, color, thickness)
        else:
            cv2.rectangle(annotated, p1, p2, color, thickness)
        label = f"{obj.cls} {obj.confidence:.0%}"
        cv2.putText(annotated, label, (p1[0], max(15, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if perception.pose.detected:
        pts = {lm.name: (int(lm.x * w), int(lm.y * h)) for lm in perception.pose.landmarks}
        for a, b in _POSE_BONES:
            if a in pts and b in pts:
                cv2.line(annotated, pts[a], pts[b], POSE_COLOR, 2, cv2.LINE_AA)
        for name, pt in pts.items():
            cv2.circle(annotated, pt, 4, POSE_COLOR, -1, cv2.LINE_AA)

    for hand in perception.hands:
        pts = {lm.name: (int(lm.x * w), int(lm.y * h)) for lm in hand.landmarks}
        for a, b in _HAND_BONES:
            if a in pts and b in pts:
                cv2.line(annotated, pts[a], pts[b], HAND_COLOR, 2, cv2.LINE_AA)
        if "wrist" in pts:
            cv2.circle(annotated, pts["wrist"], 6, HAND_COLOR, -1, cv2.LINE_AA)
            label = hand.handedness or "hand"
            cv2.putText(annotated, label, (pts["wrist"][0] + 8, pts["wrist"][1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, HAND_COLOR, 1, cv2.LINE_AA)
        for tip_name in ("thumb_tip", "index_tip"):
            if tip_name in pts:
                cv2.circle(annotated, pts[tip_name], 3, HAND_COLOR, -1, cv2.LINE_AA)

    return annotated


def _dashed_rect(img, p1, p2, color, thickness, dash_len=8):
    x1, y1 = p1
    x2, y2 = p2
    for x in range(x1, x2, dash_len * 2):
        cv2.line(img, (x, y1), (min(x + dash_len, x2), y1), color, thickness)
        cv2.line(img, (x, y2), (min(x + dash_len, x2), y2), color, thickness)
    for y in range(y1, y2, dash_len * 2):
        cv2.line(img, (x1, y), (x1, min(y + dash_len, y2)), color, thickness)
        cv2.line(img, (x2, y), (x2, min(y + dash_len, y2)), color, thickness)
