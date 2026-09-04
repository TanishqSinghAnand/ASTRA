"""Simple geometric feature extraction from a window of PerceptionFrame
objects (Phase 12 — training pipeline stub/placeholder).

Placeholder, not this project's real action-recognition approach — see
train_baseline.py's module docstring for why. Reuses the same geometry
helpers as backend/perception/interaction.py (bbox_center, point_in_bbox,
hand_center_px) rather than re-deriving hand/object position math a third
time.

Feature naming: "<tracked_class>_<feature>", so the result is a flat
dict directly usable as one row for sklearn's DictVectorizer (see
train_baseline.py) — no per-project schema needed beyond whatever classes
config/config.yaml's perception.colors defines.

Missing-data convention: -1.0 marks "not computable from this window"
(e.g. velocity from a single-frame window, or a class never detected) —
never 0.0, which would misleadingly read as "hand right on the box" or
"zero velocity". Documented here once rather than repeated at each call
site.
"""
from __future__ import annotations

from backend.perception.base import PerceptionFrame
from backend.perception.interaction import bbox_center, hand_center_px

MISSING = -1.0


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def extract_window_features(frames: list[PerceptionFrame], tracked_classes: list[str]) -> dict[str, float]:
    """One flat feature vector for a window of frames (oldest first).

    A "window" may be a single frame (e.g. tools/record_dataset.py's
    current one-frame-per-keypress captures) — displacement/velocity
    features are simply MISSING in that case, everything else (hand
    distance, detector confidence) still carries real signal from that
    one frame. Returns {} only if the window has no usable frames at all
    (none with known frame_width/frame_height — see base.py).
    """
    usable = [f for f in frames if f.frame_width > 0 and f.frame_height > 0]
    if not usable:
        return {}

    duration_s = max(usable[-1].timestamp - usable[0].timestamp, 1e-6)

    features: dict[str, float] = {}
    for cls in tracked_classes:
        hand_distances: list[float] = []
        object_positions: list[tuple[float, float]] = []
        confidences: list[float] = []

        for frame in usable:
            obj = next((o for o in frame.objects if o.cls == cls), None)
            if obj is None:
                continue
            confidences.append(obj.confidence)
            center = bbox_center(obj.bbox)
            object_positions.append(center)

            hands = [
                c
                for h in frame.hands
                if (c := hand_center_px(h, frame.frame_width, frame.frame_height)) is not None
            ]
            if hands:
                hand_distances.append(min(_dist(h, center) for h in hands))

        features[f"{cls}_mean_hand_distance_px"] = (
            sum(hand_distances) / len(hand_distances) if hand_distances else MISSING
        )
        features[f"{cls}_min_hand_distance_px"] = min(hand_distances) if hand_distances else MISSING
        features[f"{cls}_mean_confidence"] = sum(confidences) / len(confidences) if confidences else MISSING

        if len(object_positions) >= 2:
            displacement = _dist(object_positions[0], object_positions[-1])
            features[f"{cls}_displacement_px"] = displacement
            features[f"{cls}_velocity_px_per_s"] = displacement / duration_s
        else:
            features[f"{cls}_displacement_px"] = MISSING
            features[f"{cls}_velocity_px_per_s"] = MISSING

    return features
