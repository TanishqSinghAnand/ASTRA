"""Draws perception results onto a frame: object boxes + hand markers.
Shared by tools/live_preview.py now and by the WebSocket frame stream /
recording_service.py once real perception is wired into the backend
(Phase 9) — annotation logic should only exist once.

No pose/body skeleton: this demo runs on a tabletop with the camera fixed
on the work surface, and nothing in the interaction/action logic ever
read pose data (see backend/perception/real_engine.py's docstring).

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

# Bold, dark-saturated — legible against typical light/indoor backgrounds
# (the original pale grey nearly disappeared against light walls/curtains).
_ZONE_FALLBACK_COLOR = (10, 10, 220)  # deep red, BGR


def _is_zone(obj_confidence: float) -> bool:
    return obj_confidence >= 0.999  # calibrated zones are always "certain" — see zones.py


@lru_cache(maxsize=128)
def _color_for_class(cls: str) -> tuple[int, int, int]:
    """Deterministic (crc32, not Python's randomized hash() — stable
    across process restarts, not just within one) BGR color from a class
    name — used only when `cls` isn't in _OBJECT_COLORS above. Value/
    saturation pinned high (not just hue-random) so every generated color
    stays bold and legible rather than occasionally landing on something
    pale."""
    hue = (zlib.crc32(cls.encode()) % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.85)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_label(img: np.ndarray, text: str, org: tuple[int, int], color, font_scale: float = 0.65, thickness: int = 2) -> None:
    """Text with a black outline behind it — legible against any
    background brightness, not just ones the foreground color happens to
    contrast with. Used for every label this module draws."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

_HAND_BONES = [("wrist", "index_mcp"), ("wrist", "pinky_mcp"), ("index_mcp", "pinky_mcp")]

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
        thickness = 2 if zone else 3
        if zone:
            _dashed_rect(annotated, p1, p2, color, thickness)
        else:
            cv2.rectangle(annotated, p1, p2, color, thickness)
        label = obj.cls if zone else f"{obj.cls} {obj.confidence:.0%}"
        draw_label(annotated, label, (p1[0], max(22, p1[1] - 10)), color, font_scale=0.7)

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
