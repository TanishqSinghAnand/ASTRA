"""General object detection via a pretrained YOLOv8 model (v2.0).

Replaces color_detector.py's HSV thresholding for object *identity*:
instead of "is this region red/blue" (needs a colored marker on every
object), this recognizes actual real-world objects — phone, cup, bottle,
book, etc. — using Ultralytics' YOLOv8, pretrained on COCO. Zero
training/dataset needed, for the exact reason color_detector.py's own
docstring gives for choosing HSV over a trained detector in v1: a demo
that needs labeled data before it works at all blocks the MVP entirely.
This is that same principle, taken further now that a decent pretrained
*general* detector is a practical, CPU-feasible default.

Honesty note: unlike color_detector.py's blob-fill heuristic, YOLO's
confidence is a real model score (objectness × class probability) — not
something this project invented, and not recalibrated or adjusted here.

Class naming: YOLO's COCO class names ("cell phone") are normalized to
this project's snake_case convention ("cell_phone") so
ExperimentStep.action_key's f"{action}_{object.upper()}" pattern produces
clean action keys (e.g. PICK_CELL_PHONE) with no embedded spaces.

Offline note: the pretrained weights (settings.yolo_model, default
"yolov8n.pt") download once from Ultralytics' GitHub releases on first use
and are cached locally afterward (see Ultralytics' own cache dir) — same
one-time-network-then-offline shape as MediaPipe's model_complexity=1
default, not a per-run network dependency.
"""
from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


class ObjectDetector:
    """Wraps a YOLO model, filtered to a configured set of tracked
    classes, returning the exact same DetectedObject shape
    color_detector.py does — an interchangeable PerceptionEngine
    component, not a different contract."""

    def __init__(self, settings: PerceptionSettings):
        self.settings = settings
        self._model = YOLO(settings.yolo_model)
        tracked = {normalize_class_name(c) for c in settings.yolo_classes}
        # Restrict inference itself to the classes we care about (when any
        # are configured) rather than filtering after the fact — faster,
        # and avoids irrelevant detections cluttering the overlay/log.
        self._class_ids = (
            [idx for idx, name in self._model.names.items() if normalize_class_name(name) in tracked]
            if tracked
            else None
        )

    def detect(self, frame: np.ndarray) -> list[DetectedObject]:
        results = self._model.predict(
            frame,
            classes=self._class_ids,
            conf=self.settings.yolo_confidence_threshold,
            imgsz=self.settings.yolo_imgsz,
            verbose=False,
        )
        objects: list[DetectedObject] = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = normalize_class_name(self._model.names[cls_id])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            objects.append(
                DetectedObject(
                    **{"class": name},
                    confidence=round(confidence, 3),
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )
        return objects

    def close(self) -> None:
        pass  # no explicit teardown needed for an Ultralytics YOLO model
