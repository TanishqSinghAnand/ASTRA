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
}

# Bold, dark-saturated — legible against typical light/indoor backgrounds
# (the original pale grey nearly disappeared against light walls/curtains).
_EXPERIMENT_AREA_COLOR = (10, 10, 220)  # deep red, BGR

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


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color, font_scale: float = 0.65, thickness: int = 2) -> None:
    """Text with a black outline behind it — legible against any
    background brightness, not just ones the foreground color happens to
    contrast with. Used for every label this module draws."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def draw_perception_overlay(frame: np.ndarray, perception: PerceptionFrame) -> np.ndarray:
    """Returns a new annotated frame; does not mutate the input."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for obj in perception.objects:
        is_area = obj.cls == "experiment_area"
        color = _EXPERIMENT_AREA_COLOR if is_area else _OBJECT_COLORS.get(obj.cls, (200, 200, 200))
        p1 = (int(obj.bbox.x1), int(obj.bbox.y1))
        p2 = (int(obj.bbox.x2), int(obj.bbox.y2))
        thickness = 2 if is_area else 3
        if is_area:
            _dashed_rect(annotated, p1, p2, color, thickness)
        else:
            cv2.rectangle(annotated, p1, p2, color, thickness)
        label = obj.cls if is_area else f"{obj.cls} {obj.confidence:.0%}"
        draw_label(annotated, label, (p1[0], max(22, p1[1] - 10)), color, font_scale=0.7)

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
            draw_label(annotated, label, (pts["wrist"][0] + 8, pts["wrist"][1] - 8), HAND_COLOR, font_scale=0.6)
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
