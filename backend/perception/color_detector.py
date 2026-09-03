"""HSV color-based detector for the red/blue boxes + the (calibrated, not
detected) experiment area.

Why this instead of a trained object detector: see docs/README — briefly,
a pretrained COCO-style detector has no concept of "red box" vs "blue box",
and training a custom one needs labeled data before any demo works at all.
This needs none, runs in low-single-digit milliseconds on CPU, and returns
the same `{class, confidence, bbox}` shape a trained detector would, so it's
a drop-in replacement later (see training/ once that phase exists).

Honesty note on "confidence": there is no learned probability here. The
value returned is a deliberately simple, documented heuristic — how filled
and rectangular the detected color blob is — not a calibrated likelihood.
It's useful for thresholding ("is this blob solid enough to trust") but
should not be read as a model's confidence score.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject

MAX_HEURISTIC_CONFIDENCE = 0.97


class ColorBoxDetector:
    def __init__(self, settings: PerceptionSettings):
        self.settings = settings

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        objects: list[DetectedObject] = [self._experiment_area_object(w, h)]

        for class_name, color_spec in self.settings.colors.items():
            detection = self._detect_color(hsv, class_name, color_spec)
            if detection is not None:
                objects.append(detection)

        return objects

    # -- internals -----------------------------------------------------------

    def _experiment_area_object(self, frame_w: int, frame_h: int) -> DetectedObject:
        x1f, y1f, x2f, y2f = self.settings.experiment_area
        return DetectedObject(
            **{"class": "experiment_area"},
            confidence=1.0,  # calibrated region, not a detection — always "certain"
            bbox=BBox(x1=x1f * frame_w, y1=y1f * frame_h, x2=x2f * frame_w, y2=y2f * frame_h),
        )

    def _detect_color(self, hsv_frame: np.ndarray, class_name: str, color_spec) -> DetectedObject | None:
        mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
        for r in color_spec.ranges:
            lower = np.array(r.lower, dtype=np.uint8)
            upper = np.array(r.upper, dtype=np.uint8)
            mask |= cv2.inRange(hsv_frame, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self.settings.min_box_contour_area:
            return None

        x, y, w, h = cv2.boundingRect(best)
        bbox_area = w * h
        fill_ratio = area / bbox_area if bbox_area > 0 else 0.0
        # A solid, roughly rectangular blob (like a box face) fills most of
        # its own bounding box; noisy/fragmented color blobs don't.
        confidence = min(MAX_HEURISTIC_CONFIDENCE, 0.55 + 0.42 * fill_ratio)

        return DetectedObject(
            **{"class": class_name},
            confidence=round(confidence, 3),
            bbox=BBox(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h)),
        )
