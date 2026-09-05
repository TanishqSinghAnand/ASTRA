"""Draws perception results onto a frame: object boxes, pose skeleton, hand
markers. Shared by tools/live_preview.py now and by the WebSocket frame
stream / recording_service.py once real perception is wired into the
backend (Phase 9) — annotation logic should only exist once.

v2.0: object classes are no longer just "red_box"/"blue_box" (YOLO can
detect any of 80 COCO classes) and zones are no longer just one
"experiment_area" (settings.target_zones can name several). Rather than
hardcode every possible class/zone name, a calibrated zone is identified
generically (confidence == 1.0 — see zones.py's docstring: "always
certain", never a real detection score) and any object/zone class without
an explicit color gets one deterministically hashed from its name, so it's
at least *consistent* across frames without needing a color assigned
ahead of time.
"""
from __future__ import annotations

import colorsys
import zlib
from functools import lru_cache

import cv2
import numpy as np

from backend.perception.base import PerceptionFrame

_OBJECT_COLORS = {
    "red_box": (60, 60, 230),
    "blue_box": (220, 120, 40),
    "experiment_area": (150, 150, 150),
}

_ZONE_FALLBACK_COLOR = (170, 170, 170)


def _is_zone(obj_confidence: float) -> bool:
    return obj_confidence >= 0.999  # calibrated zones are always "certain" — see zones.py


@lru_cache(maxsize=128)
def _color_for_class(cls: str) -> tuple[int, int, int]:
    """Deterministic (crc32, not Python's randomized hash() — stable
    across process restarts, not just within one), reasonably saturated
    BGR color from a class name — used only when `cls` isn't in
    _OBJECT_COLORS above."""
    hue = (zlib.crc32(cls.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))

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
        zone = _is_zone(obj.confidence)
        color = _OBJECT_COLORS.get(obj.cls) or (_ZONE_FALLBACK_COLOR if zone else _color_for_class(obj.cls))
        p1 = (int(obj.bbox.x1), int(obj.bbox.y1))
        p2 = (int(obj.bbox.x2), int(obj.bbox.y2))
        thickness = 1 if zone else 2
        if zone:
            _dashed_rect(annotated, p1, p2, color, thickness)
        else:
            cv2.rectangle(annotated, p1, p2, color, thickness)
        label = obj.cls if zone else f"{obj.cls} {obj.confidence:.0%}"
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
