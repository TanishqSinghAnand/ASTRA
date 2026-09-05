"""Calibrated target-zone objects (v2.0).

v1 had exactly one placement target (`experiment_area`), hardcoded by name
throughout color_detector.py/interaction.py. v2.0 supports multiple named
zones — e.g. "zone_a" for a phone, "zone_b" for a cup — each experiment
step's `target` field (already part of the schema, see
experiment_loader.py) names which one an object must land in.

Zones are calibrated screen regions, not detected by any model — same
reasoning color_detector.py's own docstring already gives for v1's single
experiment_area: there's no physical marker to detect on a real tabletop.
This is shared by both detector backends (HSV or YOLO) rather than
duplicated in each — a detector's only job is finding real objects; zone
annotation is a separate, purely-configured overlay.
"""
from __future__ import annotations

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject


def zone_objects(settings: PerceptionSettings, frame_w: int, frame_h: int) -> list[DetectedObject]:
    """One DetectedObject per configured zone, confidence=1.0 (calibrated,
    not detected — always "certain"), in the same pixel space real
    detections use."""
    objects: list[DetectedObject] = []
    for name, (x1f, y1f, x2f, y2f) in settings.target_zones.items():
        objects.append(
            DetectedObject(
                **{"class": name},
                confidence=1.0,
                bbox=BBox(x1=x1f * frame_w, y1=y1f * frame_h, x2=x2f * frame_w, y2=y2f * frame_h),
            )
        )
    return objects
