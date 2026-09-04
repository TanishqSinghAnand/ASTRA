"""Hand-object interaction reasoning (Phase 3).

Still perception, not experiment logic: this module only answers "is a hand
approaching/touching/holding/placing an object?" for each tracked box. It
knows nothing about PICK/PLACE experiment steps or the sequence — Phase 4
(backend/temporal/) is what turns an InteractionState transition into an
action_key like "PICK_RED_BOX".

Units: MediaPipe hand landmarks are normalized (0-1 of frame size);
DetectedObject bboxes from color_detector.py are already pixel-space. Both
are needed in the same units to compare, which is why this reads
PerceptionFrame.frame_width/frame_height (see base.py) rather than assuming
a fixed resolution.

Hand position is approximated as the average of wrist + index_mcp +
pinky_mcp (hands.py's own "palm-center" landmark set) — fingertips are too
noisy for distance-based reasoning.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from backend.config.settings import PerceptionSettings
from backend.perception.base import BBox, DetectedObject, HandFrame, InteractionState, PerceptionFrame

# Per-frame box displacement below this (pixels) is treated as detection/
# hand jitter, not the box actually moving with the hand.
_DISPLACEMENT_NOISE_FLOOR_PX = 6.0

# How many recent (hand, box) position samples to correlate when deciding
# whether a held object is actively moving.
_HISTORY_LEN = 8

_PALM_LANDMARKS = {"wrist", "index_mcp", "pinky_mcp"}


def bbox_center(bbox: BBox) -> tuple[float, float]:
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


def hand_center_px(hand: HandFrame, frame_w: int, frame_h: int) -> tuple[float, float] | None:
    pts = [lm for lm in hand.landmarks if lm.name in _PALM_LANDMARKS]
    if len(pts) < len(_PALM_LANDMARKS):
        return None
    x = sum(p.x for p in pts) / len(pts) * frame_w
    y = sum(p.y for p in pts) / len(pts) * frame_h
    return (x, y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2


@dataclass
class InteractionEvent:
    """Per-object result of one InteractionReasoner.update() call."""

    object_class: str
    state: InteractionState
    changed: bool  # True only on the frame the state actually transitioned


@dataclass
class _ObjectTrack:
    state: InteractionState = InteractionState.NONE
    hand_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))
    box_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=_HISTORY_LEN))


class InteractionReasoner:
    """Maintains per-object InteractionState across frames.

    Tracked object classes are read from settings.colors (i.e. whatever
    color_detector.py is configured to detect — "red_box"/"blue_box" for the
    sample experiment) rather than hardcoded, so a different experiment's
    object set doesn't require touching this code.
    """

    def __init__(self, settings: PerceptionSettings):
        self.settings = settings
        self._tracked_classes = list(settings.colors.keys())
        self._tracks: dict[str, _ObjectTrack] = {cls: _ObjectTrack() for cls in self._tracked_classes}

    def reset(self) -> None:
        self._tracks = {cls: _ObjectTrack() for cls in self._tracked_classes}

    def update(self, frame: PerceptionFrame) -> dict[str, InteractionEvent]:
        if frame.frame_width <= 0 or frame.frame_height <= 0:
            # No known pixel space this frame (e.g. a producer that doesn't
            # populate frame_width/height) — nothing safe to compute.
            return {
                cls: InteractionEvent(cls, track.state, changed=False)
                for cls, track in self._tracks.items()
            }

        objects_by_class = {obj.cls: obj for obj in frame.objects}
        experiment_area = objects_by_class.get("experiment_area")
        hand_centers = [
            c
            for h in frame.hands
            if (c := hand_center_px(h, frame.frame_width, frame.frame_height)) is not None
        ]

        events: dict[str, InteractionEvent] = {}
        for cls in self._tracked_classes:
            track = self._tracks[cls]
            new_state = self._update_track(track, objects_by_class.get(cls), hand_centers, experiment_area)
            changed = new_state != track.state
            track.state = new_state
            events[cls] = InteractionEvent(object_class=cls, state=new_state, changed=changed)
        return events

    # -- internals -----------------------------------------------------------

    def _update_track(
        self,
        track: _ObjectTrack,
        obj: DetectedObject | None,
        hand_centers: list[tuple[float, float]],
        experiment_area: DetectedObject | None,
    ) -> InteractionState:
        if obj is None or not hand_centers:
            # A held/moving object doesn't spontaneously release just
            # because one frame had no detection (a hand closing around a
            # box routinely occludes its own color blob) — hold the state
            # and wait for the next real observation.
            if track.state in (InteractionState.OBJECT_BEING_HELD, InteractionState.OBJECT_MOVING_WITH_HAND):
                return track.state
            return InteractionState.NONE

        box_center = bbox_center(obj.bbox)
        nearest_hand = min(hand_centers, key=lambda h: _dist(h, box_center))
        distance = _dist(nearest_hand, box_center)

        track.hand_history.append(nearest_hand)
        track.box_history.append(box_center)

        touch_threshold = self.settings.hand_object_touch_distance_px
        approach_threshold = self.settings.hand_object_distance_px

        currently_held = track.state in (
            InteractionState.OBJECT_BEING_HELD,
            InteractionState.OBJECT_MOVING_WITH_HAND,
        )
        currently_resting = track.state in (
            InteractionState.OBJECT_RELEASED,
            InteractionState.OBJECT_PLACED,
        )

        if currently_held:
            if distance <= touch_threshold:
                return InteractionState.OBJECT_MOVING_WITH_HAND if self._is_moving(track) else InteractionState.OBJECT_BEING_HELD
            # Hand pulled away while holding -> released, either into the
            # experiment area (placed) or elsewhere (just dropped/released).
            if experiment_area is not None and point_in_bbox(box_center, experiment_area.bbox):
                return InteractionState.OBJECT_PLACED
            return InteractionState.OBJECT_RELEASED

        if currently_resting:
            # A placed/released object doesn't spontaneously go back to
            # NONE just because the hand that let go of it moved away — it
            # physically stays where it was left. Only a hand actually
            # touching it again restarts the touch/hold cycle; this also
            # gives Phase 4's temporal smoother something that holds stable
            # across frames to debounce, instead of a one-frame edge event.
            if distance <= touch_threshold:
                return InteractionState.HAND_TOUCHING_OBJECT
            return track.state

        if distance <= touch_threshold:
            # Require a HAND_TOUCHING_OBJECT frame before counting the
            # object as held, so a hand merely brushing past doesn't
            # register as a pick.
            if track.state == InteractionState.HAND_TOUCHING_OBJECT:
                return InteractionState.OBJECT_BEING_HELD
            return InteractionState.HAND_TOUCHING_OBJECT
        if distance <= approach_threshold:
            return InteractionState.HAND_APPROACHING_OBJECT
        return InteractionState.NONE

    def _is_moving(self, track: _ObjectTrack) -> bool:
        if len(track.box_history) < 2:
            return False
        return _dist(track.box_history[0], track.box_history[-1]) > _DISPLACEMENT_NOISE_FLOOR_PX
